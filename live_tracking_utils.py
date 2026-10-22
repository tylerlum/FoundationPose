"""Shared helpers for FoundationPose live tracking with the ZED camera.

NOTE: when using ROS, the caller must `import rospy` BEFORE importing this
module. Loading rospy AFTER the FoundationPose stack (which this module
imports transitively) reliably segfaults — some native lib in the FP stack
clobbers rospy's C bindings on subsequent import. The combined
`live_tracking.py` handles this via a `--ros in sys.argv` gate.
"""

import os
import sys
import datetime as _dt
import importlib.util
import json
import multiprocessing as _mp
import threading
import time
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import cv2
import pyzed.sl as sl
import trimesh
import imageio
import nvdiffrast.torch as dr
from estimater import *  # noqa: F401,F403  (FoundationPose, ScorePredictor, etc.)
from generate_mask import generate_binary_mask_box


# ----------------------------- camera helpers --------------------------------

def construct_camera_intrinsics(camera_params, target_width, target_height,
                                camera_upsidedown=False):
    fx = camera_params.fx * target_width / camera_params.image_size.width
    fy = camera_params.fy * target_height / camera_params.image_size.height
    cx = camera_params.cx * target_width / camera_params.image_size.width
    cy = camera_params.cy * target_height / camera_params.image_size.height
    if camera_upsidedown:
        cx = target_width - cx
        cy = target_height - cy
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def init_zed(serial_number: str, exposure: int, gain: int,
             resolution: str = 'HD1080', camera_fps: int = 0,
             depth_mode=sl.DEPTH_MODE.NEURAL,
             coordinate_units=sl.UNIT.MILLIMETER):
    """Open the ZED. `camera_fps=0` asks the SDK for that resolution's default
    max (30 / 60 / 100 for HD1080 / HD720 / VGA on a ZED 2). For Fast-FS use
    `depth_mode=sl.DEPTH_MODE.NONE` and `coordinate_units=sl.UNIT.METER`."""
    zed = sl.Camera()
    init_params = sl.InitParameters(input_t=sl.InputType())
    init_params.svo_real_time_mode = True
    init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
    init_params.depth_mode = depth_mode
    init_params.coordinate_units = coordinate_units
    if camera_fps:
        init_params.camera_fps = int(camera_fps)
    init_params.set_from_serial_number(int(serial_number))

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)

    runtime_parameters = sl.RuntimeParameters()
    return zed, runtime_parameters, sl.Mat(), sl.Mat()


def _zed_view_to_rgb(mat, width, height, camera_upsidedown):
    img = mat.get_data()
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        img = img[..., :3]
    if camera_upsidedown:
        img = cv2.flip(img, -1)
    return cv2.resize(img, (width, height))


def read_zed_depth_frame(zed, runtime_parameters, image_mat, depth_mat,
                         width, height, camera_upsidedown=False):
    """Grab LEFT + ZED neural depth. Depth from the ZED is in `coordinate_units`
    (mm if `init_zed` used the default); we convert to meters here."""
    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
        return None, None

    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    img_rgb = _zed_view_to_rgb(image_mat, width, height, camera_upsidedown)

    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    depth_mm = depth_mat.get_data()
    if camera_upsidedown:
        depth_mm = cv2.flip(depth_mm, -1)
    depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[(depth_m < 0.001) | (~np.isfinite(depth_m))] = 0.0
    return img_rgb, depth_m


def read_zed_stereo_frame(zed, runtime_parameters, left_mat, right_mat,
                          width, height, camera_upsidedown=False):
    """Grab synchronized rectified LEFT + RIGHT at (width, height)."""
    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
        return None, None
    zed.retrieve_image(left_mat, sl.VIEW.LEFT)
    zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
    left = _zed_view_to_rgb(left_mat, width, height, camera_upsidedown)
    right = _zed_view_to_rgb(right_mat, width, height, camera_upsidedown)
    return left, right


# ----------------------------- async ZED producer ----------------------------

class ZedCaptureThread(threading.Thread):
    """Producer thread that owns the camera handle and continuously grabs
    frames into a single shared slot, decoupling the camera's tick from the
    consumer loop.

    Why threading (not multiprocessing): the pyzed binding releases the GIL
    on `grab` / `retrieve_image` / `retrieve_measure` (`with nogil:` blocks
    in src/pyzed/sl.pyx). While the producer waits for the next camera tick
    inside `grab()`, the main thread runs Python/CUDA work in parallel.
    Stereolabs' own zed-ros2-wrapper is structured the same way (one
    process, multiple threads with SCHED_FIFO priorities). They also
    explicitly recommend keeping `grab` + `retrieve_*` on a single thread.

    Why no multiprocessing: forces start_method='spawn' (CUDA can't survive
    fork), 5 s producer cold-start to re-init CUDA / TRT / ZED, and IPC of
    a 12 MB stereo pair through `multiprocessing.Queue` pickles to 30-80 ms
    (worse than the camera-tick problem). `shared_memory` is zero-copy but
    adds buffer-lifetime management.

    Layout per backend:
      - ``backend='zed_neural'`` -> slot is (ts_ns, rgb_uint8, depth_m_f32)
      - ``backend='fast_fs'``    -> slot is (ts_ns, left_rgb, right_rgb)
        with stereo-to-depth inference left to the consumer (CUDA context
        affinity).
    """

    def __init__(self, zed, runtime, mat_a, mat_b, backend, fp_size,
                 camera_upsidedown=False):
        super().__init__(daemon=True, name='ZedCapture')
        if backend not in ('zed_neural', 'fast_fs'):
            raise ValueError(f'unknown backend {backend!r}')
        self.zed = zed
        self.runtime = runtime
        self.mat_a = mat_a
        self.mat_b = mat_b
        self.backend = backend
        self.fp_size = tuple(fp_size)
        self.camera_upsidedown = camera_upsidedown
        self._shutdown = threading.Event()
        self._cv = threading.Condition()
        self._slot = None      # (ts_ns, frame_a, frame_b)
        self._error = None
        self._n_grabbed = 0
        self._n_delivered = 0

    def stop_capture(self):
        self._shutdown.set()
        with self._cv:
            self._cv.notify_all()

    def latest(self, since_ts_ns=0, timeout=2.0):
        """Return the most recent (ts_ns, frame_a, frame_b) with
        ts_ns > since_ts_ns. Blocks up to `timeout` seconds; returns None
        on timeout, producer error, or stop."""
        with self._cv:
            ok = self._cv.wait_for(
                lambda: (
                    self._shutdown.is_set() or self._error is not None
                    or (self._slot is not None and self._slot[0] > since_ts_ns)
                ),
                timeout=timeout)
            if not ok or self._shutdown.is_set() or self._error is not None:
                return None
            self._n_delivered += 1
            return self._slot

    @property
    def n_grabbed(self):
        return self._n_grabbed

    @property
    def n_delivered(self):
        return self._n_delivered

    @property
    def error(self):
        return self._error

    def run(self):
        W, H = self.fp_size
        try:
            while not self._shutdown.is_set():
                err = self.zed.grab(self.runtime)
                if err != sl.ERROR_CODE.SUCCESS:
                    continue
                ts_ns = self.zed.get_timestamp(
                    sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

                # MEM.CPU avoids CUDA-stream contention with the main
                # thread's torch / TRT work (we resize on CPU anyway).
                self.zed.retrieve_image(self.mat_a, sl.VIEW.LEFT, sl.MEM.CPU)
                rgb = _zed_view_to_rgb(self.mat_a, W, H,
                                       self.camera_upsidedown)

                if self.backend == 'zed_neural':
                    self.zed.retrieve_measure(
                        self.mat_b, sl.MEASURE.DEPTH, sl.MEM.CPU)
                    depth_mm = self.mat_b.get_data()
                    if self.camera_upsidedown:
                        depth_mm = cv2.flip(depth_mm, -1)
                    depth_mm = cv2.resize(depth_mm, (W, H),
                                          interpolation=cv2.INTER_NEAREST)
                    depth_m = depth_mm.astype(np.float32) / 1000.0
                    depth_m[(depth_m < 0.001) | (~np.isfinite(depth_m))] = 0.0
                    payload = (rgb, depth_m)
                else:
                    self.zed.retrieve_image(self.mat_b, sl.VIEW.RIGHT,
                                            sl.MEM.CPU)
                    right = _zed_view_to_rgb(self.mat_b, W, H,
                                             self.camera_upsidedown)
                    payload = (rgb, right)

                with self._cv:
                    self._slot = (ts_ns,) + payload
                    self._n_grabbed += 1
                    self._cv.notify_all()
        except Exception as e:
            with self._cv:
                self._error = e
                self._cv.notify_all()


# ----------------------------- SAM mask --------------------------------------

def select_mask_with_sam(rgb):
    """Interactive SAM box-based mask. SAM helper expects BGR."""
    mask = generate_binary_mask_box(rgb[..., ::-1], polygon_refinement=True)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


# ----------------------------- ROS publisher ---------------------------------

# T_RC_DEFAULT lives in live_tracking_replay (which has no heavy deps); re-
# exported here so existing callers can still `from live_tracking_utils import
# T_RC_DEFAULT`.
from live_tracking_replay import T_RC_DEFAULT  # noqa: F401  (re-export)


class RosPosePublisher:
    """ROS publisher wrapper; instantiated only when --ros is passed.

    The caller is responsible for `import rospy` BEFORE importing this module
    (see module docstring). The `import rospy` inside __init__ is then a no-op
    since rospy is already in sys.modules, avoiding the segfault that comes
    from loading rospy after the FoundationPose stack.
    """

    def __init__(self, frame_id, publish_robot_frame):
        import rospy
        from geometry_msgs.msg import PoseStamped
        self._rospy = rospy
        self._PoseStamped = PoseStamped
        rospy.init_node('foundationpose_live_tracking', anonymous=True)
        self.frame_id = frame_id
        self.cam_pub = rospy.Publisher(
            'camera_frame/current_object_pose', PoseStamped, queue_size=1)
        self.robot_pub = (
            rospy.Publisher('robot_frame/current_object_pose',
                            PoseStamped, queue_size=1)
            if publish_robot_frame else None
        )

    def _msg(self, T, frame_id):
        msg = self._PoseStamped()
        msg.header.stamp = self._rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(T[0, 3])
        msg.pose.position.y = float(T[1, 3])
        msg.pose.position.z = float(T[2, 3])
        quat_wxyz = trimesh.transformations.quaternion_from_matrix(T)
        msg.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.orientation.z = float(quat_wxyz[3])
        msg.pose.orientation.w = float(quat_wxyz[0])
        return msg

    def publish(self, pose_cam, T_RC=None):
        self.cam_pub.publish(self._msg(pose_cam, self.frame_id))
        if self.robot_pub is not None and T_RC is not None:
            self.robot_pub.publish(self._msg(T_RC @ pose_cam, 'robot_frame'))

    def shutdown_requested(self):
        return self._rospy.is_shutdown()


# ----------------------------- Fast-FS depth ---------------------------------

# Fast-FoundationStereo lives at <repo>/third_party/Fast-FoundationStereo/.
# This file is at <repo>/third_party/FoundationPose/live_tracking_utils.py,
# so the sibling vendored copy is one level up.
_FP_DIR = Path(__file__).resolve().parent
_FAST_FS_DIR = _FP_DIR.parent / 'Fast-FoundationStereo'

# Matches Fast-FoundationStereo/Utils.py:10. Inlined because both repos
# expose a top-level `Utils` module — see `_swap_in_fast_fs_utils` below.
import torch  # noqa: E402
_AMP_DTYPE = torch.float16


@contextmanager
def _swap_in_fast_fs_utils(fast_fs_dir: Path):
    """Temporarily swap sys.modules['Utils'] to Fast-FS's Utils.py.

    Both third_party/FoundationPose/Utils.py and
    third_party/Fast-FoundationStereo/Utils.py claim the module name `Utils`.
    `from estimater import *` at the top of this file populates
    sys.modules['Utils'] with FoundationPose's Utils — so when `torch.load`
    later triggers `import core.foundation_stereo` (which does
    `from Utils import AMP_DTYPE` deep in Fast-FS internals), the import fails
    because FP's Utils has no AMP_DTYPE. Swap in Fast-FS's Utils for the
    duration of the load, then put FP's Utils back. The loaded modules
    captured AMP_DTYPE at load time, so the swap is safe to undo after
    `torch.load` returns.
    """
    saved = sys.modules.get('Utils')
    spec = importlib.util.spec_from_file_location('Utils', fast_fs_dir / 'Utils.py')
    fs_utils = importlib.util.module_from_spec(spec)
    sys.modules['Utils'] = fs_utils
    spec.loader.exec_module(fs_utils)
    try:
        yield
    finally:
        if saved is not None:
            sys.modules['Utils'] = saved
        else:
            del sys.modules['Utils']


DEFAULT_FAST_FS_MODEL = _FAST_FS_DIR / 'weights' / '23-36-37' / 'model_best_bp2_serialize.pth'
# Preferred TRT engine dir — used when --depth_backend=fast_fs is selected
# and no explicit --fast_fs_trt_dir is given. 8 iters trades ~1 ms for
# better depth fidelity vs the 4-iter engine.
DEFAULT_FAST_FS_TRT_DIR = _FAST_FS_DIR / 'weights' / '23-36-37' / 'onnx_384x224_iters8'


class FastFsDepth:
    """Fast-FoundationStereo inference: stereo RGB -> depth (meters).

    Two backends:
      - **PyTorch** (default): loads ``model_best_bp2_serialize.pth`` via
        ``torch.load`` and calls ``model.forward(left, right, iters=...,
        test_mode=True)``. ~27 ms at 384x224 on RTX 6000 Ada.
      - **TensorRT** (when ``trt_dir`` is set): loads a pre-built engine pair
        (``feature_runner.engine`` + ``post_runner.engine``) and the matching
        ``onnx.yaml``. ~4 ms at 384x224 on RTX 6000 Ada — a ~7x speedup.

    Forward runs at ``fs_size``. With TRT, ``fs_size`` is forced to the
    engine's baked-in resolution (read from ``onnx.yaml``). Disparity is
    converted to depth using fx at that processing size, then resized up to
    ``fp_size`` (FoundationPose's input resolution).
    """

    def __init__(self, model_path, iters: int, max_disp: int,
                 baseline_m: float, fp_size, fs_size, fx_fp: float,
                 trt_dir=None):
        # sys.path is appended here (lazy) so importing live_tracking_utils
        # for non-fast_fs use doesn't require Fast-FS to be present.
        sys.path.insert(0, str(_FAST_FS_DIR))
        from core.utils.utils import InputPadder
        self._InputPadder = InputPadder

        import yaml
        from omegaconf import OmegaConf

        self.trt_dir = Path(trt_dir) if trt_dir else None
        self.iters = iters
        self.baseline_m = float(baseline_m)
        self.fp_size = tuple(fp_size)

        if self.trt_dir is not None:
            feat = self.trt_dir / 'feature_runner.engine'
            post = self.trt_dir / 'post_runner.engine'
            cfg_path = self.trt_dir / 'onnx.yaml'
            for p in (feat, post, cfg_path):
                if not p.is_file():
                    raise FileNotFoundError(
                        f"Fast-FS TRT artifact missing at {p}. Build engines "
                        f"per docs/venv_isaacsim_install.md §2c.")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            self.args = OmegaConf.create(cfg)
            img_H, img_W = cfg['image_size']
            self.fs_size = (int(img_W), int(img_H))
            with _swap_in_fast_fs_utils(_FAST_FS_DIR):
                from core.foundation_stereo import TrtRunner
                self.model = TrtRunner(self.args, str(feat), str(post))
            self.backend = 'trt'
        else:
            if not Path(model_path).is_file():
                raise FileNotFoundError(
                    f"Fast-FS weights not found at {model_path}. Download per "
                    f"docs/venv_isaacsim_install.md §2b.")
            cfg_path = Path(model_path).parent / 'cfg.yaml'
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            cfg['valid_iters'] = iters
            cfg['max_disp'] = max_disp
            self.args = OmegaConf.create(cfg)
            with _swap_in_fast_fs_utils(_FAST_FS_DIR):
                self.model = torch.load(model_path, map_location='cpu',
                                        weights_only=False)
            self.model.args.valid_iters = iters
            self.model.args.max_disp = max_disp
            self.model.cuda().eval()
            self.backend = 'pytorch'
            self.fs_size = tuple(fs_size)

        # fx scales linearly with horizontal resolution.
        self.fx_fs = float(fx_fp) * self.fs_size[0] / self.fp_size[0]

    @torch.no_grad()
    def __call__(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        """Inputs at ``fp_size``. Returns depth (H_fp, W_fp) float32 in meters."""
        left_small = cv2.resize(left_rgb, self.fs_size, interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_rgb, self.fs_size, interpolation=cv2.INTER_AREA)
        fs_W, fs_H = self.fs_size

        left = torch.as_tensor(left_small).cuda().float()[None].permute(0, 3, 1, 2)
        right = torch.as_tensor(right_small).cuda().float()[None].permute(0, 3, 1, 2)

        if self.backend == 'pytorch':
            padder = self._InputPadder(left.shape, divis_by=32, force_square=False)
            left, right = padder.pad(left, right)
            with torch.amp.autocast('cuda', enabled=True, dtype=_AMP_DTYPE):
                disp = self.model.forward(left, right, iters=self.iters,
                                          test_mode=True,
                                          optimize_build_volume='pytorch1')
            disp = padder.unpad(disp.float())
        else:
            disp = self.model.forward(left, right).float()

        disp = disp.cpu().numpy().reshape(fs_H, fs_W).clip(0, None)

        # Right-view correspondences that fall off the image are invalid.
        yy, xx = np.meshgrid(np.arange(fs_H), np.arange(fs_W), indexing='ij')
        invalid = (xx - disp) < 0
        depth_small = np.zeros((fs_H, fs_W), dtype=np.float32)
        valid = (disp > 1e-3) & (~invalid)
        depth_small[valid] = (self.fx_fs * self.baseline_m) / disp[valid]

        return cv2.resize(depth_small, self.fp_size, interpolation=cv2.INTER_NEAREST)


# ----------------------------- async ZED producer (process) -----------------

class ZedCaptureProcess:
    """Producer in a *separate* process, communicating via shared_memory.

    Use this when the producer must do GPU work (e.g. ZED's NEURAL depth)
    and you don't want that work to serialize on the same CUDA context as
    the consumer's torch / TRT work. Each process holds its own CUDA
    context; the GPU driver multiplexes them.

    Layout (sized at construction):
      - `rgb_shm`   :  H*W*3 uint8     (left RGB)
      - `d_shm`     :  H*W*4 float32   (depth, zed_neural)
                       OR  H*W*3 uint8 (right RGB, fast_fs)
      - `meta_shm`  :  4*int64 = [ts_ns, seq, error_flag, pad]

    Communication: parent reads `meta[1]` (sequence number) to detect new
    frames. Under `lock` it copies rgb + depth out; consumer never holds a
    raw view to shared memory (the producer overwrites them).

    Spawn (not fork) is mandatory: the main process has a live CUDA
    context (from torch / FP), and `fork()` would inherit a corrupted one
    in the child. Spawn costs ~1-3 s startup for the pyzed + camera open
    in the child.
    """

    def __init__(self, zed_kwargs, fp_size, backend,
                 camera_upsidedown=False):
        if backend not in ('zed_neural', 'fast_fs'):
            raise ValueError(f'unknown backend {backend!r}')
        self.backend = backend
        self.fp_size = tuple(fp_size)
        self.camera_upsidedown = camera_upsidedown

        W, H = self.fp_size
        rgb_size = H * W * 3
        d_size = H * W * (4 if backend == 'zed_neural' else 3)
        meta_size = 32  # 4 int64

        self.rgb_shm = SharedMemory(create=True, size=rgb_size)
        self.d_shm = SharedMemory(create=True, size=d_size)
        self.meta_shm = SharedMemory(create=True, size=meta_size)

        self.rgb_view = np.ndarray((H, W, 3), dtype=np.uint8,
                                   buffer=self.rgb_shm.buf)
        if backend == 'zed_neural':
            self.d_view = np.ndarray((H, W), dtype=np.float32,
                                     buffer=self.d_shm.buf)
        else:
            self.d_view = np.ndarray((H, W, 3), dtype=np.uint8,
                                     buffer=self.d_shm.buf)
        self.meta = np.ndarray((4,), dtype=np.int64, buffer=self.meta_shm.buf)
        self.meta[:] = 0

        ctx = _mp.get_context('spawn')
        self._stop_evt = ctx.Event()
        self._ready_evt = ctx.Event()
        self._lock = ctx.Lock()
        self._info_queue = ctx.Queue(maxsize=1)
        self.K = None              # filled in by start() from child
        self.baseline_m = None     # filled in by start() from child

        # Lazily import here so the parent doesn't depend on the child
        # module's full import graph at top-level.
        import live_tracking_capture_proc as cap_mod
        self._proc = ctx.Process(
            target=cap_mod.producer_main,
            args=(
                {
                    'rgb': self.rgb_shm.name,
                    'd': self.d_shm.name,
                    'meta': self.meta_shm.name,
                },
                self._stop_evt, self._lock, self._ready_evt,
                self._info_queue, zed_kwargs, self.fp_size,
                backend, camera_upsidedown,
            ),
            daemon=True,
        )

    def start(self, ready_timeout=20.0):
        """Spawn the child and block until it has produced one frame
        (or set the error flag). Raises on camera-open failure. Populates
        `self.K` and `self.baseline_m` from the child's camera info."""
        self._proc.start()
        # Pick up intrinsics first (sent before the first frame).
        try:
            info = self._info_queue.get(timeout=ready_timeout)
        except Exception as e:
            self._stop_evt.set()
            raise RuntimeError(
                f'capture child did not publish camera info within '
                f'{ready_timeout:.1f}s: {e}')
        if 'error' in info:
            self._stop_evt.set()
            self._proc.join(timeout=2.0)
            raise RuntimeError(
                f'capture child failed to open ZED: {info["error"]}')
        self.K = np.asarray(info['K'], dtype=np.float64)
        self.baseline_m = float(info['baseline_m'])
        if not self._ready_evt.wait(timeout=ready_timeout):
            self._stop_evt.set()
            raise RuntimeError(
                f'capture child opened the camera but produced no frame '
                f'within {ready_timeout:.1f}s')
        if int(self.meta[2]) != 0:
            self._stop_evt.set()
            self._proc.join(timeout=2.0)
            raise RuntimeError('capture child failed to grab a frame')

    def latest(self, since_seq=0, timeout=2.0, poll_s=0.001):
        """Return (seq, ts_ns, rgb_copy, depth_copy) once a frame with
        seq > since_seq is available. Frames are *copied* out of shared
        memory so the consumer can hold them safely while the producer
        overwrites the next one. Returns None on timeout / error / stop."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._proc.is_alive() or int(self.meta[2]) != 0:
                return None
            cur = int(self.meta[1])
            if cur > since_seq:
                with self._lock:
                    cur = int(self.meta[1])
                    if cur <= since_seq:
                        continue
                    ts_ns = int(self.meta[0])
                    rgb = self.rgb_view.copy()
                    depth = self.d_view.copy()
                return cur, ts_ns, rgb, depth
            time.sleep(poll_s)
        return None

    def stop_capture(self):
        self._stop_evt.set()
        # Producer publishes its per-stage stats on the info_queue before
        # exiting; pick that up so the parent can log it.
        self.producer_stats = None
        try:
            msg = self._info_queue.get(timeout=3.0)
            if isinstance(msg, dict):
                self.producer_stats = msg.get('producer_stats')
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=3.0)
        for shm in (self.rgb_shm, self.d_shm, self.meta_shm):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


# ----------------------------- viser replay ----------------------------------
# unproject_strided, _free_port, launch_viser_replay live in
# live_tracking_replay (no heavy deps). Re-exported below so callers that
# already import them from here keep working.


def save_viz_data(run_dir, viz_log, pose_log, K, stride, mesh_path, compress=True,
                  timestamps=None):
    """Stash the strided RGB/depth + poses + intrinsics into
    `viz_data.npz` inside `run_dir`, so the run can be replayed in viser
    later via `live_tracking_viz_replay.py <run_dir>`.

    Saving the *strided* arrays (not the unprojected points) keeps the
    file small and lets the replay pick its own stride/point_size.
    """
    if not viz_log:
        return None
    rgb_s = np.stack([r for r, _ in viz_log], axis=0)
    depth_s = np.stack([d for _, d in viz_log], axis=0)
    n = len(pose_log)
    poses = np.full((n, 4, 4), np.nan, dtype=np.float64)
    for i, p in enumerate(pose_log):
        if p is not None:
            poses[i] = p
    ts = (np.asarray(timestamps, dtype=np.int64)
          if timestamps is not None else np.zeros(n, dtype=np.int64))
    path = os.path.join(run_dir, 'viz_data.npz')
    # savez_compressed costs 3.83 s where savez costs 0.09 s (42x) for
    # only 12% more bytes -- not a good trade inside a live loop.
    _savez = np.savez_compressed if compress else np.savez
    _savez(
        path,
        rgb=rgb_s, depth=depth_s, poses=poses,
        K=np.asarray(K, dtype=np.float64),
        stride=np.int64(stride),
        mesh_path=str(mesh_path or ''),
        timestamps=ts,
    )
    print(f'[viz] saved viz data to {path}  '
          f'({rgb_s.shape[0]} frames, '
          f'{rgb_s.nbytes/1e6 + depth_s.nbytes/1e6:.1f} MB raw)')
    return path


from live_tracking_replay import (  # noqa: E402,F401  (re-exports)
    unproject_strided, _free_port, launch_viser_replay,
)


# ----------------------------- run output ------------------------------------

# Axis endpoints in object frame for the overlay (10 cm).
_AXIS_X = np.array([0.1, 0.0, 0.0, 1.0])
_AXIS_Y = np.array([0.0, 0.1, 0.0, 1.0])
_AXIS_Z = np.array([0.0, 0.0, 0.1, 1.0])
_ORIGIN_H = np.array([0.0, 0.0, 0.0, 1.0])


def _stage_stats_ms(times):
    t = np.asarray(times)
    return {
        'mean': float(t.mean() * 1e3),
        'p50': float(np.median(t) * 1e3),
        'p95': float(np.percentile(t, 95) * 1e3),
        'min': float(t.min() * 1e3),
        'max': float(t.max() * 1e3),
    }


def save_run(save_dir, backend_name, resolution,
             raw_frames, pose_log, K, loop_start_time, frame_times,
             target_fps, stage_times=None,
             encode_preset=None, encode_crf=None, write_overlay=True):
    """Write raw.mp4 + overlay.mp4 into a fresh datetimed subdir of
    save_dir and print loop-timing stats. Returns the run directory.

    encode_preset
        x264 preset. None keeps ffmpeg's default (medium), which is what this
        function always used. Measured on 400 frames of 960x540: default
        4.15 s, veryfast 0.80 s (5.2x faster), ultrafast 0.29 s (14x). Two
        videos are written per take, so this dominates save latency.
    encode_crf
        Use -crf N rate control instead of imageio's quality=8. x264 does not
        honour the qscale that quality= maps to, which is why the files come
        out enormous (200 MB for a short take). CRF fixes size and speed at
        once.
    write_overlay
        Skip overlay.mp4. It is reconstructible from raw.mp4 plus the poses.
    """
    if not raw_frames or loop_start_time is None:
        return None

    elapsed = max(time.time() - loop_start_time, 1e-6)
    measured_fps = max(1, int(round(len(raw_frames) / elapsed)))
    stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(
        save_dir,
        f'{stamp}_{backend_name}_{resolution}_fps{measured_fps}')
    os.makedirs(run_dir, exist_ok=True)
    raw_path = os.path.join(run_dir, 'raw.mp4')
    overlay_path = os.path.join(run_dir, 'overlay.mp4')

    _enc = {'macro_block_size': 1}
    _params = []
    if encode_preset:
        _params += ['-preset', str(encode_preset)]
    if encode_crf is not None:
        _params += ['-crf', str(encode_crf)]
    else:
        _enc['quality'] = 8
    if _params:
        _enc['output_params'] = _params

    raw_writer = imageio.get_writer(raw_path, fps=measured_fps,
                                    codec='libx264', **_enc)
    for f in raw_frames:
        raw_writer.append_data(f)
    raw_writer.close()

    overlay_writer = None
    if write_overlay:
        overlay_writer = imageio.get_writer(overlay_path, fps=measured_fps,
                                            codec='libx264', **_enc)
    else:
        overlay_path = '(skipped)'
    for rgb_frame, pose in (zip(raw_frames, pose_log) if overlay_writer else ()):
        if pose is None or K is None:
            overlay_writer.append_data(rgb_frame)
            continue
        overlay = rgb_frame.copy()
        o = tuple(project_3d_to_2d(_ORIGIN_H, K, pose))
        px = tuple(project_3d_to_2d(_AXIS_X, K, pose))
        py = tuple(project_3d_to_2d(_AXIS_Y, K, pose))
        pz = tuple(project_3d_to_2d(_AXIS_Z, K, pose))
        cv2.arrowedLine(overlay, o, px, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.arrowedLine(overlay, o, py, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.arrowedLine(overlay, o, pz, (0, 0, 255), 3, cv2.LINE_AA)
        overlay_writer.append_data(overlay)
    if overlay_writer is not None:
        overlay_writer.close()

    print(f'Saved: {raw_path} and {overlay_path} '
          f'({len(raw_frames)} frames @ {measured_fps} FPS over {elapsed:.2f}s)')

    if frame_times:
        ft = np.asarray(frame_times)
        fps_arr = 1.0 / np.clip(ft, 1e-6, None)
        hit_pct = float((fps_arr >= target_fps).mean() * 100.0)
        stages = {
            name: _stage_stats_ms(ts)
            for name, ts in (stage_times or {}).items() if ts
        }
        metrics = {
            'backend': backend_name,
            'resolution': resolution,
            'timestamp': stamp,
            'n_frames': int(len(raw_frames)),
            'elapsed_s': float(elapsed),
            'target_fps': float(target_fps),
            'achieved_fps': float(measured_fps),
            'per_frame_ms': _stage_stats_ms(frame_times),
            'stages_ms': stages,
            'hit_target_pct': hit_pct,
            'budget_ms': float(1e3 / target_fps),
        }
        metrics_path = os.path.join(run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        lines = [
            '--- Loop timing (excludes sleep-to-target) ---',
            f'  target FPS:           {target_fps:.1f}',
            f'  achieved (wall) FPS:  {measured_fps:.1f}',
            f'  per-frame work (ms):  '
            f'mean={ft.mean()*1e3:.1f}  p50={np.median(ft)*1e3:.1f}  '
            f'p95={np.percentile(ft, 95)*1e3:.1f}  '
            f'min={ft.min()*1e3:.1f}  max={ft.max()*1e3:.1f}',
            f'  frames with work <= 1/target ({1e3/target_fps:.1f}ms): '
            f'{hit_pct:.1f}%',
        ]
        if stages:
            lines.append('  per-stage (ms): '
                         '         mean    p50    p95    min    max')
            for name, s in stages.items():
                lines.append(
                    f"    {name:<10s}{s['mean']:7.1f}{s['p50']:7.1f}"
                    f"{s['p95']:7.1f}{s['min']:7.1f}{s['max']:7.1f}")
        lines.append(f'  metrics: {metrics_path}')
        print('\n'.join(lines))
    return run_dir
