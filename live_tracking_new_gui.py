"""Live 6-DoF object pose tracking with the ZED, dispatched on depth backend.

  --depth_backend zed_neural   ZED SDK's onboard NEURAL depth.
  --depth_backend fast_fs      Fast-FoundationStereo over ZED's LEFT+RIGHT.

Pass --ros to publish PoseStamped on:
  - camera_frame/current_object_pose
  - robot_frame/current_object_pose  (uses T_RC_DEFAULT unless --calibration
    overrides it; pass --no_robot_frame to skip the robot-frame publish)

    python third_party/FoundationPose/live_tracking.py --depth_backend zed_neural --camera_resolution HD1080 --camera_fps 30 --fps 30 --mesh_path assets/urdf/fabrica/beam_3x/0/0_colored.obj --save_dir May21_tests_policy/ --ros

--------------------------------------------------------------------------
VARIANT: browser GUI + swap object/prompt mid-run (on top of auto-record)
--------------------------------------------------------------------------

Everything live_tracking_new_autorecord.py does, plus:

1. A viser control panel in the browser instead of the OpenCV keyboard
   window. Re-register, save, discard, toggle auto-record, and read the live
   loop rate. No camera frames are rendered in it -- that is deliberate, the
   loop's throughput must not pay for the UI. The old window cost real
   FPS: it forced cv2.imshow + waitKey(1) EVERY iteration, and dropping it
   measured 36 -> 40 FPS at a 40 FPS cap.

2. Swap the tracked object WITHOUT restarting: type a new text prompt, or
   point at a different mesh, and hit the button. FoundationPose's
   reset_object() rebuilds the pose grid for the new mesh in place, then the
   script re-registers on a fresh frame. Restarting used to mean paying the
   ~25 s camera + model startup again.

3. Mark the object by hand (13 Oct 2026), for when a text prompt finds
   nothing: --mask_mode points opens a popup (point_prompt_mask.py) where you
   drag a box around the object and/or click points on it (right click: not
   the object); SAM redraws the mask after every change and Enter accepts.
   --text_fallback_click opens the same popup when a text prompt misses, and
   the panel's 'Re-register by clicking' button opens it mid-run. Needs an X
   display.

Kept separate from live_tracking_new_autorecord.py on purpose so the working
script stays untouched.

--------------------------------------------------------------------------
Inherited: auto-record gated on the RL policy
--------------------------------------------------------------------------

Derived from live_tracking_new.py. Two changes:

1. AUTO-RECORD (the point of this file). Frames are buffered only while
   rl_policy_node.py is actually publishing joint position targets, and the
   take is written to disk the moment it stops. So a save contains just the
   rollout, not the minutes of idling around it.

   rl_policy_node.py publishes sensor_msgs/JointState on /iiwa/joint_cmd and
   /sharpa/joint_cmd from one function at its 60 Hz control rate
   (`self.control_dt = 1.0 / 60`), so either topic is a faithful "policy is
   running" signal. We watch one; --policy_topic picks which.

   Pose tracking itself never pauses -- FoundationPose stays locked on the
   object the whole time. Only *buffering* is gated, so starting a new
   rollout does not cost a re-register.

   Requires --ros. Disable with --no_auto_record to get plain
   live_tracking_new.py behaviour.

2. LAZY HEAVY IMPORTS (fixes docs/live_tracking_troubleshooting.md section 2).
   torch / nvdiffrast / trimesh / the FoundationPose stack used to be
   imported at module top level. ZedCaptureProcess starts its child with the
   'spawn' method, and spawn re-executes the parent module's top level via
   runpy.run_path -- so every capture child paid a full torch + nvdiffrast +
   Warp import. That is the duplicated "Warp initialized" / "[buffer]
   retaining..." output, and it widened the window in which a stray Ctrl-C
   produced the bogus sre_compile UnboundLocalError of section 1. They now
   load inside main(), which the child never runs.

Saves also get a metadata.json and the object name in the directory name.
"""

import os
import re
import sys
import time
import argparse
import math
import json as _json
import threading as _threading
import multiprocessing as _mp
from multiprocessing.shared_memory import SharedMemory
import datetime as _dt
from collections import deque as _deque
from pathlib import Path

_FP_DIR = Path(__file__).resolve().parent

# rospy MUST be imported BEFORE live_tracking_utils (which loads the FP stack)
# — see the segfault note in live_tracking_utils.py. Gated on --ros so non-ROS
# runs don't pay the import cost.
if '--ros' in sys.argv:
    import rospy  # noqa: F401
    from geometry_msgs.msg import PoseStamped  # noqa: F401
    from sensor_msgs.msg import JointState  # noqa: F401


def _import_heavy():
    """Import the heavy stack into module globals.

    Called once at the top of main(). Deliberately NOT at module top level:
    ZedCaptureProcess spawns its child, and spawn re-runs this module's top
    level through runpy.run_path. Anything imported up there is imported
    again in every capture child -- a full torch/nvdiffrast/Warp load per
    child. See docs/live_tracking_troubleshooting.md section 2.

    pyzed is not imported here at all: this module never references `sl`
    directly (live_tracking_utils owns the camera).
    """
    global np, cv2, torch, trimesh, dr
    global select_mask_with_sam, RosPosePublisher, T_RC_DEFAULT
    global FastFsDepth, DEFAULT_FAST_FS_MODEL, DEFAULT_FAST_FS_TRT_DIR, save_run
    global ZedCaptureProcess, unproject_strided, launch_viser_replay, save_viz_data
    global ScorePredictor, PoseRefinePredictor, FoundationPose

    import numpy as np
    import cv2
    import torch
    import trimesh
    import nvdiffrast.torch as dr

    from live_tracking_utils import (
        select_mask_with_sam, RosPosePublisher, T_RC_DEFAULT,
        FastFsDepth, DEFAULT_FAST_FS_MODEL, DEFAULT_FAST_FS_TRT_DIR, save_run,
        ZedCaptureProcess, unproject_strided, launch_viser_replay, save_viz_data,
    )
    from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose


def _policy_watch_proc(topic, shm_name):
    """Child process: watch `topic` and stamp shared memory. Nothing else.

    This lives in its own process ON PURPOSE. When a ROS publisher exits,
    rospy's receive_loop is stranded on a half-closed socket: sock.recv()
    returns b'' immediately instead of blocking, forever, logging on every
    iteration. Profiled on this box that thread burns 100% of a core (45%
    recv_buff, 41% roslogging.findCaller). Being a Python loop it holds the
    GIL, and in-process it dragged the tracking loop from 30 FPS to 1.6 FPS
    and made the viser panel unresponsive.

    unregister() does NOT stop that thread -- measured, it keeps spinning.
    So the only reliable containment is a separate interpreter: let it spin
    over here where it cannot touch the tracking loop's GIL.
    """
    import time as _t
    import numpy as _np
    from multiprocessing.shared_memory import SharedMemory as _Shm
    import rospy as _rospy
    from sensor_msgs.msg import JointState as _JointState

    shm = _Shm(name=shm_name)
    arr = _np.ndarray((2,), dtype=_np.float64, buffer=shm.buf)

    _rospy.init_node('fp_policy_watch', anonymous=True, disable_signals=True)

    def _cb(_msg):
        arr[0] = _t.time()
        arr[1] += 1.0

    _rospy.Subscriber(topic, _JointState, _cb, queue_size=10)
    _rospy.spin()


class PolicyActivityWatcher:
    """Is rl_policy_node.py currently publishing joint position targets?

    The actual subscription runs in a child process (see
    _policy_watch_proc); this side only reads a timestamp out of shared
    memory, which costs nothing and cannot be starved by rospy.

    active() is True while a message arrived within the last `idle_timeout`
    seconds, once `start_msgs` have been seen at all. At the policy's 60 Hz
    the default 0.5 s means ~30 missed messages before a take closes.
    """

    def __init__(self, topic, idle_timeout=0.5, start_msgs=2,
                 recycle_after=2.0):
        import numpy as np  # _import_heavy() has already run
        self.topic = topic
        self.idle_timeout = float(idle_timeout)
        self.start_msgs = int(start_msgs)

        self._shm = SharedMemory(create=True, size=16)
        self._arr = np.ndarray((2,), dtype=np.float64, buffer=self._shm.buf)
        self._arr[:] = 0.0

        # spawn, not fork: this process holds a live CUDA context.
        ctx = _mp.get_context('spawn')
        self._proc = ctx.Process(target=_policy_watch_proc,
                                 args=(topic, self._shm.name),
                                 name='policy_watch', daemon=True)
        self._proc.start()

    @property
    def total_msgs(self):
        return int(self._arr[1])

    def active(self):
        last, count = float(self._arr[0]), float(self._arr[1])
        if count < self.start_msgs:
            return False
        return (time.time() - last) < self.idle_timeout

    def close(self):
        try:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=3.0)
        except Exception:
            pass
        try:
            self._shm.close(); self._shm.unlink()
        except Exception:
            pass


def _select_mask_safe(rgb):
    """select_mask_with_sam (the 4-corner box picker), but neutralizes the
    CUDA default tensor type that FoundationPose's Utils.py installs during
    register/track_one (as in live_tracking_new.py). It was referenced below
    but never defined here, so --mask_mode sam raised a NameError."""
    was_cuda = torch.tensor(0.).is_cuda
    if was_cuda:
        torch.set_default_tensor_type(torch.FloatTensor)
    try:
        return select_mask_with_sam(rgb)
    finally:
        if was_cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)


def _click_mask(rgb, title):
    """The popup: drag a box and/or click points; SAM shows the mask."""
    from point_prompt_mask import select_mask_interactive
    return select_mask_interactive(rgb, title=title)


def _center_box_mask(rgb, h_frac, w_frac):
    """Fixed box at the image centre. No SAM, no clicking, no display."""
    h, w = rgb.shape[:2]
    dh, dw = int(h * h_frac), int(w * w_frac)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2 - dh:h // 2 + dh, w // 2 - dw:w // 2 + dw] = 1
    return mask


def _build_parser():
    parser = argparse.ArgumentParser(description="Live 6D object pose tracking.")
    parser.add_argument('--depth_backend', choices=['zed_neural', 'fast_fs'],
                        required=True,
                        help='Which depth source to feed FoundationPose.')
    parser.add_argument('--mesh_path', type=str, required=True,
                        help='Path to object mesh file (.obj, in meters).')
    parser.add_argument('--ros', action='store_true',
                        help='Publish PoseStamped on camera_frame/current_object_pose '
                             'and robot_frame/current_object_pose '
                             '(robot frame uses T_RC_DEFAULT from live_tracking_utils '
                             'unless --calibration overrides it).')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Optional 4x4 camera-to-robot transform '
                             '(.npy or .txt) overriding T_RC_DEFAULT. '
                             'Only used when --ros is set.')
    parser.add_argument('--no_robot_frame', action='store_true',
                        help='With --ros, skip the robot_frame/current_object_pose '
                             'publish (only camera_frame is published).')
    parser.add_argument('--serial_number', type=str, default='15107')
    parser.add_argument('--camera_upsidedown', action='store_true')
    parser.add_argument('--camera_resolution', choices=['HD1080', 'HD720', 'VGA'],
                        default='HD1080',
                        help='ZED capture resolution. HD1080=30fps max, HD720=60fps, VGA=100fps.')
    parser.add_argument('--camera_fps', type=int, default=0,
                        help='ZED capture FPS. 0 = use the resolution default '
                             '(30/60/100 for HD1080/HD720/VGA on ZED 2).')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--exposure', type=int, default=25)
    parser.add_argument('--gain', type=int, default=40)
    parser.add_argument('--fps', type=float, default=40.0,
                        help='Target tracking-loop FPS (distinct from --camera_fps).')
    parser.add_argument('--est_refine_iter', type=int, default=5,
                        help='Refinement iterations for initial registration.')
    parser.add_argument('--track_refine_iter', type=int, default=2,
                        help='Refinement iterations per tracking step.')
    parser.add_argument('--save_dir', type=str, default='live_tracking_results')
    parser.add_argument('--max_buffer_seconds', type=float, default=60.0,
                        help='Keep only this many seconds of frames/poses in RAM '
                             'before dropping the oldest entries. 0 disables the cap.')
    parser.add_argument('--max_buffer_frames', type=int, default=0,
                        help='Exact frame buffer cap. 0 derives the cap from '
                             '--max_buffer_seconds and --fps.')
    parser.add_argument('--frame_id', type=str, default='camera_frame')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='Run the tracking loop for this many seconds (after initial '
                             'register) then exit. 0 = run until ROS shutdown.')
    # Fast-FS-specific knobs (ignored when --depth_backend=zed_neural).
    parser.add_argument('--fast_fs_model', type=str, default=str(DEFAULT_FAST_FS_MODEL),
                        help='Path to Fast-FS model_best_bp2_serialize.pth (PyTorch path).')
    parser.add_argument('--fast_fs_iters', type=int, default=4,
                        help='Fast-FS refinement iterations (TRT engines bake this in).')
    parser.add_argument('--fast_fs_max_disp', type=int, default=192)
    parser.add_argument('--fast_fs_width', type=int, default=384,
                        help='Fast-FS processing width (forced to engine size in TRT mode).')
    parser.add_argument('--fast_fs_height', type=int, default=224,
                        help='Fast-FS processing height (forced to engine size in TRT mode).')
    parser.add_argument('--fast_fs_trt_dir', type=str,
                        default=str(DEFAULT_FAST_FS_TRT_DIR),
                        help='Directory with feature_runner.engine, post_runner.engine, '
                             'and onnx.yaml from docs/venv_isaacsim_install.md §2c. '
                             'Defaults to the 8-iter engine; pass an empty string to '
                             'fall back to the PyTorch path.')
    parser.add_argument('--visualize', action='store_true',
                        help='After the episode, launch a viser server replaying '
                             'the recorded run with per-frame point clouds, the '
                             'tracked pose, and the object mesh overlay.')
    parser.add_argument('--viz_stride', type=int, default=4,
                        help='Stride for the per-frame point cloud sampling '
                             '(higher = sparser; only used with --visualize).')
    parser.add_argument('--fp_refiner_trt', type=str,
                        default=str(_FP_DIR / 'weights' / '2023-10-28-18-33-37'
                                    / 'refine_net_fp16.plan'),
                        help='Path to a TensorRT engine (.plan) for the FP '
                             'refiner. Defaults to the bundled fp16 engine. '
                             'Pass an empty string to fall back to PyTorch.')
    # ---- auto-record gating (this variant) --------------------------------
    parser.add_argument('--no_auto_record', action='store_true',
                        help='Disable policy-gated recording and behave like '
                             'live_tracking_new.py (buffer everything, save '
                             'on quit / r key).')
    parser.add_argument('--policy_topic', type=str, default='/iiwa/joint_cmd',
                        help='JointState topic whose publishing gates '
                             'recording. rl_policy_node.py publishes '
                             '/iiwa/joint_cmd and /sharpa/joint_cmd together '
                             'at 60 Hz, so either works.')
    parser.add_argument('--policy_idle_timeout', type=float, default=0.5,
                        help='Stop the take after this many seconds with no '
                             'message on --policy_topic. Default 0.5 s = 30 '
                             'missed messages at the 60 Hz control rate.')
    parser.add_argument('--policy_start_msgs', type=int, default=2,
                        help='Messages needed inside one --policy_idle_timeout '
                             'window before a take opens.')
    parser.add_argument('--selftest_swap', action='store_true',
                        help='Drive the panel from a timer instead of a '
                             'browser: change prompt, swap mesh, re-register. '
                             'Exercises exactly the code path the GUI '
                             'callbacks use, for headless verification.')
    parser.add_argument('--selftest_saves', action='store_true',
                        help='Selftest variant that fires repeated saves, to '
                             'measure what a save does to the loop rate.')
    parser.add_argument('--keyboard_window', action='store_true',
                        help='Also open the old OpenCV keyboard window '
                             'alongside the viser panel. Off by default: it '
                             'forces a cv2.imshow + waitKey on every loop '
                             'iteration.')
    parser.add_argument('--no_gui', action='store_true',
                        help='Disable the viser control panel.')
    parser.add_argument('--gui_port', type=int, default=8090,
                        help='Port for the viser panel: the first free one '
                             'from this up (the same URL on every rerun; '
                             'never a port another process is using). 0 '
                             'picks any free port.')
    parser.add_argument('--mesh_choices', type=str, nargs='*', default=None,
                        help='Extra mesh paths to offer in the GUI dropdown '
                             'for mid-run swapping.')
    parser.add_argument('--headless', action='store_true',
                        help='Never open the OpenCV keyboard-control window. '
                             'Implied when $DISPLAY is unset. Use with '
                             '--mask_mode center_box for a fully '
                             'non-interactive run.')
    # ---- encoder ----------------------------------------------------------
    parser.add_argument('--encode_preset', type=str, default='veryfast',
                        help="x264 preset for the saved videos. The original "
                             "code passed none, so ffmpeg used 'medium': "
                             "measured 4.15s per 400 frames vs 0.80s for "
                             "'veryfast'. Two videos per take, so this is the "
                             "bulk of save time. Pass '' for the old default.")
    parser.add_argument('--encode_crf', type=int, default=20,
                        help='x264 -crf. Replaces imageio quality=8, which '
                             'maps to a qscale x264 ignores -- the reason '
                             'takes came out at hundreds of MB. Lower = '
                             'better quality and bigger. -1 restores the old '
                             'quality=8 behaviour.')
    parser.add_argument('--no_overlay_video', action='store_true',
                        help='Skip overlay.mp4 (halves encode time). The pose '
                             'overlay is reconstructible from raw.mp4 plus '
                             'the saved poses.')
    parser.add_argument('--viz_data', action='store_true',
                        help='Also write viz_data.npz (strided RGB + depth '
                             'point-cloud data for the viser replay). OFF by '
                             'default: it is 40-90 MB per take and was the '
                             'single slowest part of a save. Only needed for '
                             'live_tracking_viz_replay.py / --visualize.')
    parser.add_argument('--viz_compress', action='store_true',
                        help='Compress viz_data.npz. Off by default: 3.83s vs '
                             '0.09s to write, for 12%% fewer bytes.')
    parser.add_argument('--text_prompt', type=str, default=None,
                        help="What to track, in words, e.g. 'black spoon'. "
                             "Implies --mask_mode text. Removes the 4-corner "
                             "clicking entirely, including on every "
                             "re-register.")
    parser.add_argument('--text_backend', choices=['sam3', 'gdino'],
                        default='sam3',
                        help="Text-masking backend. 'sam3' (facebook/sam3) "
                             "goes straight from text to instance masks and "
                             "is a GATED HF repo. 'gdino' uses Grounding DINO "
                             "for a box then the existing SAM path, and is "
                             "ungated.")
    parser.add_argument('--text_min_score', type=float, default=0.3,
                        help='Reject a text match below this confidence.')
    parser.add_argument('--text_fallback_click', action='store_true',
                        help='If the text prompt matches nothing, open the '
                             'popup (drag a box / click points) instead of '
                             'giving up.')
    parser.add_argument('--mask_mode', choices=['sam', 'points', 'center_box', 'text'],
                        default='sam',
                        help="How to get the initial object mask. 'points' "
                             "opens a popup: drag a box around the object "
                             "and/or click points (right click: not the "
                             "object); SAM shows the mask live; Enter "
                             "accepts. 'sam' is the original interactive "
                             "4-corner picker. 'center_box' takes a fixed box "
                             "at the image centre with no clicking -- for "
                             "headless runs, smoke tests, and rigs where the "
                             "object is reliably centred.")
    parser.add_argument('--center_box_frac', type=float, nargs=2,
                        default=(0.25, 0.33), metavar=('H_FRAC', 'W_FRAC'),
                        help='Half-height and half-width of the center_box '
                             'mask, as a fraction of the frame.')
    parser.add_argument('--object_name', type=str, default=None,
                        help='Name recorded in metadata.json and the save '
                             'directory name. Defaults to the mesh filename.')
    return parser


def _buffer_frame_cap(args):
    if args.max_buffer_frames > 0:
        return int(args.max_buffer_frames)
    if args.max_buffer_seconds <= 0:
        return None
    fps_for_cap = args.fps if args.fps > 0 else args.camera_fps
    if fps_for_cap <= 0:
        fps_for_cap = {'HD1080': 30, 'HD720': 60, 'VGA': 100}.get(
            args.camera_resolution, 30)
    return max(1, int(math.ceil(args.max_buffer_seconds * fps_for_cap)))


def main():
    _import_heavy()
    args = _build_parser().parse_args()

    auto_record = not args.no_auto_record
    if auto_record and not args.ros:
        sys.exit('--ros is required for auto-record (it watches '
                 f'{_build_parser().get_default("policy_topic")}). '
                 'Pass --no_auto_record to run without ROS.')
    if args.text_prompt and args.mask_mode == 'sam':
        args.mask_mode = 'text'          # --text_prompt implies text masking
    if args.mask_mode == 'text' and not args.text_prompt:
        sys.exit('--mask_mode text requires --text_prompt')
    if args.mask_mode == 'points' and not os.environ.get('DISPLAY'):
        sys.exit('--mask_mode points opens a popup: run it on the workstation '
                 'desktop or over ssh -X (no DISPLAY here)')
    # viz_log is only worth building if it will be written or replayed.
    _want_viz = bool(args.viz_data or args.visualize)
    object_name = (args.object_name
                   or (args.text_prompt.replace(' ', '_') if args.text_prompt
                       else None)
                   or Path(args.mesh_path).stem)
    buffer_frame_cap = _buffer_frame_cap(args)

    T_RC = None
    if args.ros and not args.no_robot_frame:
        if args.calibration is not None:
            arr = np.load(args.calibration) if args.calibration.endswith('.npy') \
                  else np.loadtxt(args.calibration)
            T_RC = arr.reshape(4, 4)
        else:
            T_RC = T_RC_DEFAULT
    elif args.calibration is not None and not args.ros:
        print('[warn] --calibration ignored (only used with --ros)')

    ros_pub = RosPosePublisher(args.frame_id, publish_robot_frame=(T_RC is not None)) \
              if args.ros else None

    os.makedirs(args.save_dir, exist_ok=True)
    raw_frames = _deque(maxlen=buffer_frame_cap)
    pose_log = _deque(maxlen=buffer_frame_cap)
    frame_times = _deque(maxlen=buffer_frame_cap)
    stage_times = {
        name: _deque(maxlen=buffer_frame_cap)
        for name in ('grab', 'depth', 'track', 'publish', 'total')
    }
    # Strided slices (rgb[::s, ::s].copy(), depth[::s, ::s].copy()) per
    # frame. Always recorded so the run can be replayed in viser later
    # via `live_tracking_viz_replay.py <run_dir>` — the --visualize flag
    # only controls whether we *auto-launch* viser at the end.
    viz_log = _deque(maxlen=buffer_frame_cap)
    if buffer_frame_cap is None:
        print('[buffer] frame buffer cap disabled')
    else:
        print(f'[buffer] retaining newest {buffer_frame_cap} frames in RAM')
    K = None
    loop_start_time = None
    backend_tag = args.depth_backend
    capture = None
    interrupted = False

    def _save_loop_start_time():
        if buffer_frame_cap is not None and frame_times:
            return time.time() - float(sum(frame_times))
        return loop_start_time

    # Defined before the try so the finally block can always reach them.
    gui_state = None
    gui_stop = None

    def _clear_buffers():
        raw_frames.clear()
        pose_log.clear()
        frame_times.clear()
        for v in stage_times.values():
            v.clear()
        viz_log.clear()

    def _write_take(reason, raw, poses, fts, sts, vizs, started):
        """Encode and write one take. A pure function of its snapshot args, so
        it is safe to run on a worker thread while the loop refills the live
        deques."""
        _t_enc = time.time()
        n = len(raw)
        # Sum of per-frame wall times = the span actually recorded. Computed
        # from the snapshot, never from time.time() after the encode.
        recorded_s = float(sum(fts))
        take_dir = save_run(
            save_dir=args.save_dir,
            backend_name=f'{backend_tag}_{object_name}',
            resolution=args.camera_resolution,
            raw_frames=raw, pose_log=poses, K=K,
            loop_start_time=started,
            frame_times=fts, stage_times=sts,
            target_fps=args.fps,
            encode_preset=(args.encode_preset or None),
            encode_crf=(None if args.encode_crf < 0 else args.encode_crf),
            write_overlay=not args.no_overlay_video,
        )
        if take_dir:
            meta = {
                'object': object_name,
                'text_prompt': args.text_prompt,
                'mask_mode': args.mask_mode,
                'mesh_path': os.path.abspath(args.mesh_path),
                'depth_backend': args.depth_backend,
                'backend_tag': backend_tag,
                'camera_resolution': args.camera_resolution,
                'camera_fps': args.camera_fps,
                'serial_number': args.serial_number,
                'target_fps': args.fps,
                'track_refine_iter': args.track_refine_iter,
                'est_refine_iter': args.est_refine_iter,
                'n_frames': n,
                'duration_s': recorded_s,
                'stop_reason': reason,
                'auto_record': auto_record,
                'policy_topic': args.policy_topic if auto_record else None,
                'policy_idle_timeout': args.policy_idle_timeout if auto_record else None,
                'ros': bool(args.ros),
                'ros_master_uri': os.environ.get('ROS_MASTER_URI'),
                'saved_at': _dt.datetime.now().isoformat(timespec='seconds'),
            }
            with open(os.path.join(take_dir, 'metadata.json'), 'w') as fh:
                _json.dump(meta, fh, indent=2)
            if vizs and K is not None and args.viz_data:
                save_viz_data(take_dir, vizs, poses, K,
                              args.viz_stride, args.mesh_path,
                              compress=args.viz_compress)
            print(f'[autorecord] saved {n} frames -> {take_dir} ({reason}) '
                  f'[encode {time.time()-_t_enc:.1f}s]', flush=True)
        return take_dir

    def _snapshot():
        return (list(raw_frames), list(pose_log), list(frame_times),
                {k: list(v) for k, v in stage_times.items()}, list(viz_log),
                _save_loop_start_time())

    def _save_take(reason, clear=True):
        """Save the take, blocking until it is on disk.

        Deliberately synchronous. An earlier version encoded on a worker
        thread so the loop could keep running, but overlapping a save with
        live tracking made behaviour hard to reason about for no real gain
        once the encode itself got ~6x faster. The loop pauses for the save;
        that is the intended trade.
        """
        if not raw_frames:
            print(f'[autorecord] {reason}: nothing buffered, skipping save')
            return None
        snap = _snapshot()
        _t_save = time.time()
        take_dir = _write_take(reason, *snap)
        print(f'[autorecord] save blocked the loop for '
              f'{time.time()-_t_save:.1f}s', flush=True)
        if clear:
            _clear_buffers()
        return take_dir


    try:
        # Producer process owns the camera. Spawn cost ~10 s; thereafter
        # `latest()` returns the freshest frame instantly. Critically, the
        # child has its OWN CUDA context, so ZED's NEURAL depth doesn't
        # serialize behind FoundationPose's track_one in the main process.
        t_spawn = time.time()
        capture = ZedCaptureProcess(
            zed_kwargs={
                'resolution': args.camera_resolution,
                'camera_fps': args.camera_fps,
                'serial_number': args.serial_number,
                'exposure': args.exposure,
                'gain': args.gain,
            },
            fp_size=(args.width, args.height),
            backend=args.depth_backend,
            camera_upsidedown=args.camera_upsidedown,
        )
        capture.start(ready_timeout=30.0)
        K = capture.K
        baseline_m = capture.baseline_m
        print(f'[capture] producer ready in {time.time()-t_spawn:.2f}s; '
              f'K[0,0]={K[0,0]:.1f}, baseline={baseline_m*1000:.2f}mm')
        last_seq = 0

        fast_fs = None
        if args.depth_backend == 'fast_fs':
            fast_fs = FastFsDepth(
                model_path=args.fast_fs_model,
                iters=args.fast_fs_iters,
                max_disp=args.fast_fs_max_disp,
                baseline_m=baseline_m,
                fp_size=(args.width, args.height),
                fs_size=(args.fast_fs_width, args.fast_fs_height),
                fx_fp=float(K[0, 0]),
                trt_dir=args.fast_fs_trt_dir,
            )
            if fast_fs.backend == 'trt':
                m = re.search(r'iters(\d+)', args.fast_fs_trt_dir or '')
                iters_tag = f'iters{m.group(1)}' if m else 'itersN'
            else:
                iters_tag = f'iters{args.fast_fs_iters}'
            backend_tag = f'fast_fs_{fast_fs.backend}_{iters_tag}'
            print(f'Fast-FS backend: {fast_fs.backend} at {fast_fs.fs_size}; '
                  f'depth upsampled to {fast_fs.fp_size}')

        def fetch():
            """Block until a fresh frame is ready; return (rgb, depth_m,
            ts_ns, grab_s, depth_s). For fast_fs, depth_s isolates the
            stereo-to-depth TRT inference cost (run on the consumer side)."""
            nonlocal last_seq
            t0 = time.time()
            out = capture.latest(since_seq=last_seq, timeout=2.0)
            if out is None:
                raise RuntimeError('capture process delivered no frame')
            seq, ts_ns, a, b = out
            last_seq = seq
            grab_s = time.time() - t0
            if args.depth_backend == 'zed_neural':
                return a, b, ts_ns, grab_s, 0.0
            t1 = time.time()
            depth = fast_fs(a, b)
            return a, depth, ts_ns, grab_s, time.time() - t1

        mesh = trimesh.load(args.mesh_path, process=False)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor(
            trt_engine_path=(args.fp_refiner_trt or None))
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=scorer, refiner=refiner,
            debug=0, glctx=glctx, debug_dir=args.save_dir,
        )

        def _get_mask(frame, force_click=False):
            if force_click or args.mask_mode == 'points':
                return _click_mask(frame, f'{object_name}: drag a box or click it')
            if args.mask_mode == 'center_box':
                m = _center_box_mask(frame, *args.center_box_frac)
                print(f'[mask] center_box: {int(m.sum())} px', flush=True)
                return m
            if args.mask_mode == 'text':
                from text_mask import mask_from_text
                _t = time.time()
                m, score = mask_from_text(frame, args.text_prompt,
                                          backend=args.text_backend,
                                          min_score=args.text_min_score)
                dt = time.time() - _t
                if m is None or m.sum() == 0:
                    print(f'[mask] text {args.text_prompt!r} matched nothing '
                          f'(best score {score:.2f} < {args.text_min_score}) '
                          f'in {dt:.2f}s', flush=True)
                    if args.text_fallback_click:
                        print('[mask] falling back to the popup (drag a box '
                              'or click the object)', flush=True)
                        return _click_mask(frame, f'{object_name}: text found '
                                           'nothing; drag a box or click it')
                    return None
                print(f'[mask] text {args.text_prompt!r} -> {int(m.sum())} px '
                      f'(score {score:.2f}) in {dt:.2f}s', flush=True)
                return m
            return _select_mask_safe(frame)

        rgb, depth, _, _, _ = fetch()
        mask = _get_mask(rgb)
        if mask is None or mask.sum() == 0:
            print('Empty ROI selected. Exiting.')
            return

        print('Registering initial pose...')
        t0 = time.time()
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask.astype(bool),
                            iteration=args.est_refine_iter)
        print(f'Initial registration done in {time.time()-t0:.3f}s')
        if ros_pub is not None:
            ros_pub.publish(pose, T_RC)

        # Tiny control window — focus it and press a key to act on the
        # tracking loop. RAM-conscious: 'd' discards the in-memory buffer
        # so a long session only keeps the takes worth saving. The SAM
        # picker calls destroyAllWindows on exit, so we recreate this
        # window after every reset.
        _lines = ['d: discard buffer',
                  'r: re-register (drops buffer)',
                  'q / ESC: quit']
        if auto_record:
            _lines.insert(0, 'AUTO: rec gated on policy')
        control_img = np.zeros((30 * len(_lines) + 40, 380, 3), dtype=np.uint8)
        for i, line in enumerate(_lines):
            cv2.putText(control_img, line, (10, 35 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # cv2's Qt plugin calls abort() (SIGABRT, core dumped) when it cannot
        # reach an X display -- it does not raise something catchable at the
        # Python level reliably. So decide up front and never touch HighGUI
        # when headless. With --auto_record the policy drives everything and
        # the keyboard window is not needed at all.
        # The viser panel replaces this window -- running both means paying
        # cv2.imshow + cv2.waitKey(1) EVERY iteration for a window nothing
        # uses. Only open it when the panel is off (or explicitly asked for).
        headless = (args.headless
                    or not os.environ.get('DISPLAY')
                    or (not args.no_gui and not args.keyboard_window))
        if headless:
            _why = ('no DISPLAY' if not os.environ.get('DISPLAY')
                    else 'viser panel is handling control')
            print(f'[control] OpenCV keyboard window disabled ({_why}); '
                  'Ctrl-C once to stop and save', flush=True)

        def _show_control_window():
            if headless:
                return
            cv2.namedWindow('Keyboard Control', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Keyboard Control', 380, 30 * len(_lines) + 40)
            cv2.imshow('Keyboard Control', control_img)
            cv2.waitKey(1)

        _show_control_window()

        # ---- viser control panel ------------------------------------------
        if not args.no_gui:
            from live_tracking_gui_panel import GuiState, start_gui
            from live_tracking_replay import _free_port, first_free_port
            port = first_free_port(args.gui_port) if args.gui_port else _free_port()
            gui_state = GuiState(text_prompt=args.text_prompt,
                                 mesh_path=args.mesh_path,
                                 auto_record=auto_record,
                                 mask_mode=args.mask_mode)
            choices = [args.mesh_path] + list(args.mesh_choices or [])
            _gui_server, gui_stop = start_gui(gui_state, port, choices)
            import socket as _socket
            print(f'[gui] control panel: http://{_socket.getfqdn()}:{port}  '
                  f'(or http://localhost:{port} with an ssh -L tunnel)',
                  flush=True)

            if args.selftest_swap:
                def _selftest():
                    alt = (args.mesh_choices or [None])[0]
                    if args.selftest_saves:
                        steps = [(12.0, 'save', True), (24.0, 'save', True),
                                 (36.0, 'save', True), (48.0, 'save', True)]
                    else:
                        steps = [(6.0, 'prompt', 'robot arm'),
                                 (12.0, 'prompt', 'black spoon')]
                        if alt:
                            steps.append((18.0, 'mesh', alt))
                        steps.append((24.0, 'reregister', True))
                    t0 = time.time()
                    for when, key, val in steps:
                        time.sleep(max(0.0, when - (time.time() - t0)))
                        print(f'[selftest] t+{time.time()-t0:4.1f}s '
                              f'request {key}={val!r}', flush=True)
                        gui_state.request(key, val)
                _threading.Thread(target=_selftest, name='selftest',
                                  daemon=True).start()

        def _do_reregister(reason, new_mesh_path=None, new_prompt=None,
                           force_click=False):
            """Drop the buffer, optionally swap object/prompt, re-register.

            The swap is the point of this variant: reset_object() rebuilds
            FoundationPose's pose grid for a new mesh in place, so changing
            what you track costs one registration (~1 s) instead of a full
            restart (~25 s of camera + model startup).
            """
            nonlocal pose, mesh, object_name, recording, loop_start_time
            n_dropped = len(raw_frames)
            _clear_buffers()
            recording = not auto_record

            if new_prompt and new_prompt != args.text_prompt:
                args.text_prompt = new_prompt
                args.mask_mode = 'text'
                if not args.object_name:
                    object_name = new_prompt.replace(' ', '_')
                print(f'[swap] text prompt -> {new_prompt!r}', flush=True)

            if new_mesh_path and new_mesh_path != args.mesh_path:
                _t0 = time.time()
                try:
                    new_mesh = trimesh.load(new_mesh_path, process=False)
                except Exception as exc:
                    print(f'[swap] could not load {new_mesh_path}: {exc!r}',
                          flush=True)
                    return True
                est.reset_object(model_pts=new_mesh.vertices,
                                 model_normals=new_mesh.vertex_normals,
                                 mesh=new_mesh)
                mesh = new_mesh
                args.mesh_path = new_mesh_path
                if not args.object_name and not args.text_prompt:
                    object_name = Path(new_mesh_path).stem
                print(f'[swap] mesh -> {new_mesh_path} '
                      f'({len(new_mesh.vertices)} verts) in '
                      f'{time.time()-_t0:.2f}s', flush=True)

            print(f'[live_tracking] re-registering ({reason}); dropped '
                  f'{n_dropped} buffered frames', flush=True)
            _t_rr = time.time()
            rgb_r, depth_r, _, _, _ = fetch()
            _t_mask = time.time()
            mask_r = _get_mask(rgb_r, force_click=force_click)
            print(f'[live_tracking]   mask in {time.time()-_t_mask:.2f}s',
                  flush=True)
            if mask_r is None or mask_r.sum() == 0:
                # Unlike the keyboard variant this does NOT exit the loop: a
                # text prompt that misses should not end the session.
                print('[live_tracking]   no mask; keeping the previous pose',
                      flush=True)
                _show_control_window()
                loop_start_time = time.time()
                return True
            _t_reg = time.time()
            pose = est.register(K=K, rgb=rgb_r, depth=depth_r,
                                ob_mask=mask_r.astype(bool),
                                iteration=args.est_refine_iter)
            print(f'[live_tracking]   re-register in '
                  f'{time.time()-_t_reg:.2f}s (total '
                  f'{time.time()-_t_rr:.2f}s)', flush=True)
            _clear_buffers()
            if ros_pub is not None:
                ros_pub.publish(pose, T_RC)
            _show_control_window()
            loop_start_time = time.time()
            return True

        # ---- policy-gated recording ---------------------------------------
        watcher = None
        recording = not auto_record   # manual mode buffers from the start
        if auto_record:
            watcher = PolicyActivityWatcher(
                args.policy_topic, args.policy_idle_timeout,
                args.policy_start_msgs)
            print(f'[autorecord] watching {args.policy_topic} '
                  f'(idle_timeout={args.policy_idle_timeout}s, '
                  f'start_msgs={args.policy_start_msgs}); '
                  f'tracking now, buffering once the policy publishes')

        _takes = 0
        target_frame_time = 1.0 / args.fps if args.fps > 0 else 0.0
        loop_start_time = time.time()

        # FP's Utils.set_logging_format() installs root logger at INFO
        # at import time, so estimater.track_one prints "Welcome" / depth
        # processing lines every iter. Bump to WARNING for the loop so the
        # live [fps] line below stays readable; setup logs above are kept.
        import logging as _logging
        _logging.getLogger().setLevel(_logging.WARNING)

        # Rolling window of work_elapsed (sleep/waitKey excluded) so the
        # printed avg reflects achievable throughput — can exceed args.fps
        # when there's headroom in the per-iter budget. Window length tracks
        # the target so avg = (recent ~1s of work).
        work_history = _deque(
            maxlen=max(30, int(args.fps) if args.fps > 0 else 30))

        def keep_running():
            # rospy.init_node() installs its own SIGINT handler that flips
            # is_shutdown() instead of raising KeyboardInterrupt, so the
            # `except KeyboardInterrupt` path below never fires in --ros mode.
            # Treat a ROS-shutdown exit as an interrupt so the save prompt
            # still triggers.
            nonlocal interrupted
            if ros_pub is not None and ros_pub.shutdown_requested():
                interrupted = True
                return False
            if args.duration > 0 and (time.time() - loop_start_time) >= args.duration:
                return False
            return True

        try:
            while keep_running():
                loop_start = time.time()

                rgb, depth, _ts, t_grab, t_depth = fetch()

                t0 = time.time()
                pose = est.track_one(rgb=rgb, depth=depth, K=K,
                                     iteration=args.track_refine_iter)
                t_track = time.time() - t0

                t_publish = 0.0
                if ros_pub is not None:
                    t0 = time.time()
                    ros_pub.publish(pose, T_RC)
                    t_publish = time.time() - t0

                # Buffer raw RGB + pose; overlay rendering deferred to save_run().
                # Gated on `recording` so an auto-record session holds only the
                # policy rollouts, not the idle minutes between them. Tracking
                # itself is never gated -- FoundationPose stays locked on.
                if recording:
                    raw_frames.append(rgb)
                    pose_log.append(pose)
                    if _want_viz:
                        # ~230 kB/frame at stride 4. Skipped entirely unless
                        # something will actually consume it.
                        s = args.viz_stride
                        viz_log.append((rgb[::s, ::s].copy(),
                                        depth[::s, ::s].copy()))

                work_elapsed = time.time() - loop_start
                sleep_time = target_frame_time - work_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Refresh the control image every iter — without a periodic
                # imshow, waitKey can stall for tens-to-hundreds of ms per
                # call on some HighGUI backends, tanking the loop's fps.
                if headless:
                    key = 255  # no HighGUI, so no keypress
                else:
                    cv2.imshow('Keyboard Control', control_img)
                    key = cv2.waitKey(1) & 0xFF

                # Record iter wall time AFTER sleep + waitKey so stage_times
                # reflects the real loop cadence (otherwise achieved_fps in
                # the dir name disagrees with hit_target_pct in metrics).
                iter_elapsed = time.time() - loop_start
                # Gated with the frame buffer above: metrics.json and the
                # measured_fps in the save directory name are both derived
                # from these, so they must cover exactly the frames that were
                # actually kept -- otherwise idle time between policy
                # rollouts would dilute the take's reported rate.
                if recording:
                    frame_times.append(iter_elapsed)
                    stage_times['grab'].append(t_grab)
                    stage_times['depth'].append(t_depth)
                    stage_times['track'].append(t_track)
                    stage_times['publish'].append(t_publish)
                    stage_times['total'].append(iter_elapsed)

                work_history.append(work_elapsed)
                fps_inst = 1.0 / max(work_elapsed, 1e-6)
                fps_avg = len(work_history) / sum(work_history)
                if buffer_frame_cap is None:
                    buffer_count = f'{len(frame_times):4d}'
                else:
                    buffer_count = f'{len(frame_times):4d}/{buffer_frame_cap}'
                print(f'\r[fps] achieved inst={fps_inst:5.1f}  '
                      f'avg{len(work_history)}={fps_avg:5.1f}  '
                      f'(cap={args.fps:.0f})  n={buffer_count}  ',
                      end='', flush=True)

                # ---- policy-gated start/stop --------------------------
                if watcher is not None:
                    now_active = watcher.active()
                    if now_active and not recording:
                        _clear_buffers()
                        loop_start_time = time.time()
                        recording = True
                        print(f'\n[autorecord] {args.policy_topic} publishing '
                              f'-> recording', flush=True)
                    elif recording and not now_active:
                        recording = False
                        print(f'\n[autorecord] {args.policy_topic} idle for '
                              f'{args.policy_idle_timeout}s -> saving take',
                              flush=True)
                        _save_take('policy_stopped')
                        _takes += 1
                        if gui_state is not None:
                            gui_state.publish(takes_saved=_takes)
                        loop_start_time = time.time()

                # ---- GUI requests -------------------------------------
                # One dict copy under a lock per iteration. Nothing here
                # renders, so the loop's rate is unaffected.
                if gui_state is not None:
                    _rq = gui_state.take_requests()
                    if _rq['auto_record'] is not None:
                        auto_record = bool(_rq['auto_record'])
                        if auto_record and watcher is None and args.ros:
                            watcher = PolicyActivityWatcher(
                                args.policy_topic, args.policy_idle_timeout,
                                args.policy_start_msgs)
                        if not auto_record:
                            recording = True
                        gui_state.publish(
                            auto_record=auto_record,
                            last_event=('auto-record on' if auto_record
                                        else 'auto-record off'))
                    if _rq['discard']:
                        _n = len(raw_frames)
                        _clear_buffers()
                        loop_start_time = time.time()
                        gui_state.publish(last_event=f'discarded {_n} frames')
                    if _rq['save']:
                        _save_take('gui_save')
                        _takes += 1
                        gui_state.publish(takes_saved=_takes,
                                          last_event='saving take')
                    if _rq.get('reregister_click'):
                        gui_state.publish(last_event='re-registering by clicking '
                                          '(popup on the workstation screen)...')
                        _do_reregister('gui click', force_click=True)
                        gui_state.publish(last_event='re-registered by clicking',
                                          object_name=object_name)
                    if _rq['prompt'] or _rq['mesh'] or _rq['reregister']:
                        gui_state.publish(last_event='re-registering...')
                        _do_reregister('gui', new_mesh_path=_rq['mesh'],
                                       new_prompt=_rq['prompt'])
                        gui_state.publish(last_event='re-registered',
                                          mesh_path=args.mesh_path,
                                          text_prompt=args.text_prompt or '',
                                          object_name=object_name,
                                          mask_mode=args.mask_mode)
                    if _rq['quit']:
                        print('[gui] stop requested', flush=True)
                        break
                    gui_state.publish(fps=fps_avg, recording=recording,
                                      n_buffered=len(raw_frames),
                                      object_name=object_name)

                if key == ord('d'):
                    n = len(raw_frames)
                    _clear_buffers()
                    # Reset the take's start time so the next save's
                    # measured_fps (= n_frames / elapsed) isn't diluted by
                    # the discarded interval. Mirrors the 'r' branch below.
                    loop_start_time = time.time()
                    print(f'[live_tracking] discarded {n} frames')
                elif key == ord('r'):
                    if not _do_reregister('key r'):
                        break
                elif key in (ord('q'), 27):
                    print('[live_tracking] quit key pressed; exiting loop.')
                    break
        except KeyboardInterrupt:
            interrupted = True
            print('\n[live_tracking] Ctrl+C received, stopping capture.')

    finally:
        print()  # terminate the in-place [fps] line from the loop
        producer_stats = None
        if capture is not None:
            capture.stop_capture()
            producer_stats = capture.producer_stats
        if producer_stats:
            lines = ['--- Producer per-stage timings (ms) ---',
                     f"  frames: {producer_stats['n_frames']}",
                     '              mean    p50    p95    p99    min    max']
            for stage in ('grab', 'left',
                          'depth' if args.depth_backend == 'zed_neural'
                          else 'right',
                          'publish_lock'):
                s = producer_stats.get(stage)
                if not s:
                    continue
                lines.append(
                    f"  {stage:<12s}"
                    f"{s['mean_ms']:7.1f}{s['p50_ms']:7.1f}"
                    f"{s['p95_ms']:7.1f}{s['p99_ms']:7.1f}"
                    f"{s['min_ms']:7.1f}{s['max_ms']:7.1f}")
            print('\n'.join(lines))
        if gui_stop is not None:
            gui_stop()
        if os.environ.get('DISPLAY'):
            cv2.destroyAllWindows()

        # On Ctrl+C (in either non-ROS or --ros mode), prompt before the
        # (slow) encode+save. Natural exits (duration timer) save without
        # prompting. Non-tty stdin also saves silently so non-interactive
        # runs stay unchanged.
        # _save_take is defined above the try, so it always exists here.
        do_save = bool(raw_frames)
        if interrupted and raw_frames and sys.stdin.isatty():
            # rospy.init_node installs a SIGINT handler that flips
            # is_shutdown() instead of raising KeyboardInterrupt. Restore the
            # default handler so a SECOND Ctrl+C at the prompt actually
            # aborts the input() instead of being silently eaten.
            import signal as _signal
            try:
                _signal.signal(_signal.SIGINT, _signal.default_int_handler)
            except Exception:
                pass
            try:
                ans = input(f'Save {len(raw_frames)} frames to '
                            f'{args.save_dir}? [Y/n] ').strip().lower()
                do_save = ans in ('', 'y', 'yes')
            except (KeyboardInterrupt, EOFError):
                # Second Ctrl+C (or piped EOF) => skip save, exit fast.
                print('\n[live_tracking] aborting without save.')
                do_save = False

        run_dir = None
        if do_save:
            # clear=False: the viser replay below still needs viz_log.
            # _save_take already writes viz_data.npz and metadata.json.
            run_dir = _save_take('interrupted' if interrupted else 'exit',
                                 clear=False)

            if viz_log and K is not None and run_dir:
                if args.visualize:
                    print(f'[viser] preparing replay ({len(viz_log)} frames at '
                          f'stride={args.viz_stride}) ...')
                    pc_log = [
                        unproject_strided(rgb_s, depth_s, K, stride=args.viz_stride)
                        for rgb_s, depth_s in viz_log
                    ]
                    launch_viser_replay(pc_log, list(pose_log), K,
                                        mesh_path=args.mesh_path,
                                        T_RC=T_RC_DEFAULT)


if __name__ == '__main__':
    main()
