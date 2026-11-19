"""Live 6-DoF pose tracking of TWO objects with the ZED: the browser-GUI variant, with SAM2 checks on object 1.

ROBUST COPY (29 Oct 2026) of live_tracking_new_gui_2_objects.py, which stays unchanged. Without --sam2_tool it runs
exactly as the original. With --sam2_tool (robust_tool.py):
  - SAM2 (the real-time fork, tiny) tracks object 1's mask every --sam2_every frames, started from its registration
    mask (and again after every re-registration of object 1);
  - each frame FoundationPose's silhouette (mesh at its pose) is compared with that mask: iou, centre distance;
    a frame is bad when iou < --lost_iou AND the centres are > --lost_px apart; LOST after --lost_frames bad frames;
  - when lost (and --lost_action reregister, the default): restart FoundationPose from the policy node's chosen pose
    (--reseed_topic, e.g. /robot_frame/tool_pose_node from --tool_hand_hold_s, if fresh), else re-register object 1
    from SAM2's mask; at most once per --lost_cooldown_s;
  - publishes std_msgs/Float32MultiArray on /robot_frame/<pose_topic>_quality: [badness 0..1, lost 0/1, tag_seen
    (always 0 here), iou, centre_px] for the node's --tool_quality_topic;
  - every take also gets robust_log.json (per frame: sam_ms, iou, dpx, badness, lost, action).
Offline evidence (the 29 Oct 03:22 clip): SAM2 kept the driver through grasp, end-on hold, fall and table; iou 0.82
tracked on the table, ~0 once FoundationPose was lost, 0.0-0.5 end-on in the hand while roughly right.


The two-object version of live_tracking_new_gui.py. It keeps everything that
script does -- the viser control panel, text-prompted masks (SAM3), policy-gated
auto-record, mid-run swaps, fast saves -- and tracks two objects at once:

  object 1: --mesh_path   / --text_prompt    -> <frame>/current_object_pose
  object 2: --mesh_path_2 / --text_prompt_2  -> <frame>/current_object_pose_2

(--pose_topic / --pose_topic_2 rename the topics, e.g. --pose_topic_2
workpiece_pose for deployment/tool_program_node.py.) Both objects register on
the same frame, and every loop iteration tracks both (two FoundationPose
estimators that share one scorer, refiner and rasterizer), so a take holds both
pose streams frame for frame. The panel re-registers either object or both, and
swaps either object's prompt or mesh mid-run.

A take writes raw.mp4, overlay.mp4 (object 1: red/green/blue axes, object 2:
cyan/magenta/yellow), poses_object_1.npy and poses_object_2.npy (camera frame,
NaN where missing), K.txt, metadata.json and metrics.json. --viz_data adds
viz_data.npz with both pose streams; live_tracking_viz_replay_2_objects.py
replays it (and --visualize opens that replay at the end).

Four additions (13 Oct 2026, for the printable tool kit):

1. Upright registration. A wrench (symmetric about its handle axis) or a plain
   box-like head looks the same turned over, so register() can return the
   turned-over copy: in a synthetic check the bolt head came back upside down
   (frame origin 68 mm too high). For the meshes named in --upright_meshes
   (default: the kit's open_wrench_18, screw_p1_head, bolt_thick_head), the
   script keeps, at register time only, the copy whose +z points up (robot +Z
   via T_RC), by turning FoundationPose's own state over so track_one() keeps
   that copy. Never per frame: in the hand a wrench's +z can point down.
   --no_upright turns it off.
2. Joined text masks. SAM3 returns the screw head's two halves (one each side
   of the slot) as separate instances, and mask_from_text keeps the best one.
   --text_join_px N / --text_join_px_2 N join every instance whose box centre
   lies within N px of the best one's (60 for the screw head at 960 x 540).
3. Offline replay, to test without the camera. --replay_dir DIR reads
   rgb/*.png, depth/*.png (uint16, mm) or depth/*.npy (float, m) and cam_K.txt
   (FoundationPose's demo layout) instead of the ZED. With --mask_mode replay,
   masks_1/<frame>.png and masks_2/<frame>.png give each object's mask.
4. Marking objects by hand. --mask_mode points opens a popup per object
   (point_prompt_mask.py): drag a box around the object and/or click points
   on it (right click: not the object); SAM redraws the mask after every
   change and Enter accepts. --text_fallback_click opens the same popup when
   a text prompt misses, and the panel's 'Re-register object k by clicking'
   opens it mid-run. Needs an X display (the workstation desktop or ssh -X).

Example (bohg-ws-16):

    drl
    M=/juno/u/tylerlum/kit_tracking_meshes_20261013
    python third_party/FoundationPose/live_tracking_new_gui_2_objects.py \\
      --depth_backend zed_neural --camera_resolution HD1080 --camera_fps 30 --fps 30 \\
      --mesh_path $M/open_wrench_18.obj --text_prompt "gray wrench" \\
      --mesh_path_2 $M/bolt_thick_head.obj --text_prompt_2 "small gray cube" \\
      --mesh_choices $M/round_screwdriver_p1.obj $M/screw_p1_head.obj \\
      --save_dir /juno/u/tylerlum/2026-10-13_kit_tracking_2obj --ros

Two objects cost two track_one() calls per frame: expect roughly half the
single-object loop rate.
--track_hz_2 1 tracks object 2 once per second (a bowl or a screw that stays
still) and publishes its last pose again between updates: ~11 ms less per frame.

Capture stamps and the payload topics (17 Nov 2026, for the brush + cube sweep):

- Every PoseStamped (camera_frame/ and robot_frame/) is stamped with the ZED
  image capture time (sl.TIME_REFERENCE.IMAGE, host clock = ROS time) of the
  frame that produced that object's pose, not the publish time. An object not
  tracked on a frame (--track_hz_2) keeps the stamp of the frame it was.
- --payload_object K (1 or 2; 0 = off) also publishes object K's robot-frame
  pose on --payload_pose_topic (default /perception/payload/pose_robot): the
  same message as robot_frame/<topic>, so T_RC is applied once. With it,
  --payload_status_topic (default /perception/payload/status) carries a
  std_msgs/String JSON {"valid": bool, "stamp": <header.stamp.to_sec() of that
  pose>, "reason": ..., ...}. valid comes from payload_check.PayloadCheck on
  that same frame (colour, silhouette IoU, depth surface, jump), whose
  defaults are the sweep deployment's cube contract (--payload_config reads
  another JSON's "cube" section). It is false for a lost, hidden, or rejected
  observation and after a failed registration.
- A take also writes frame_log.jsonl: per frame, each object's capture stamp
  and the payload verdict.
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


# The kit's parts that look the same turned over (see the module docstring).
_UPRIGHT_MESHES_DEFAULT = ('open_wrench_18', 'screw_p1_head', 'bolt_thick_head')
# Overlay axis colours (RGB): object 1 keeps red/green/blue, object 2 gets
# cyan/magenta/yellow, as in live_tracking_utils_2_objects.py.
_OBJ_AXIS_COLORS = (((255, 0, 0), (0, 255, 0), (0, 0, 255)),
                    ((0, 255, 255), (255, 0, 255), (255, 255, 0)))


def _import_heavy():
    """Import the heavy stack into module globals.

    Called once at the top of main(). Deliberately NOT at module top level:
    ZedCaptureProcess spawns its child, and spawn re-runs this module's top
    level through runpy.run_path. Anything imported up there is imported
    again in every capture child. See docs/live_tracking_troubleshooting.md
    section 2.
    """
    global np, cv2, torch, trimesh, dr, imageio
    global select_mask_with_sam, T_RC_DEFAULT
    global FastFsDepth, DEFAULT_FAST_FS_MODEL, DEFAULT_FAST_FS_TRT_DIR
    global ZedCaptureProcess, unproject_strided
    global ScorePredictor, PoseRefinePredictor, FoundationPose

    import numpy as np
    import cv2
    import torch
    import trimesh
    import imageio
    import nvdiffrast.torch as dr

    from live_tracking_utils import (
        select_mask_with_sam, T_RC_DEFAULT,
        FastFsDepth, DEFAULT_FAST_FS_MODEL, DEFAULT_FAST_FS_TRT_DIR,
        ZedCaptureProcess, unproject_strided,
    )
    from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose


def _policy_watch_proc(topic, shm_name):
    """Child process: watch `topic` and stamp shared memory. Nothing else.

    Its own process ON PURPOSE (see live_tracking_new_gui.py): a rospy
    subscriber whose publisher exits spins at 100 % of a core holding the GIL,
    which dragged the tracking loop from 30 to 1.6 FPS when in-process.
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

    The subscription runs in a child process (see _policy_watch_proc); this
    side only reads a timestamp out of shared memory. active() is True while a
    message arrived within the last `idle_timeout` seconds, once `start_msgs`
    have been seen at all.
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


class RosPosePublisher2:
    """PoseStamped publishers for two objects, with configurable topic names.

    object k publishes on camera_frame/<topics[k]> (header frame_id) and, when
    publish_robot_frame, on robot_frame/<topics[k]> (pose T_RC @ pose_cam).
    Headers carry the capture stamp passed in (ts_ns, ROS time in ns).
    payload_k (0-based, or None) also publishes that object's robot-frame
    message on payload_pose_topic and its validity on payload_status_topic.
    The caller imports rospy BEFORE the FoundationPose stack (see above).
    """

    def __init__(self, frame_id, publish_robot_frame, topics, payload_k=None,
                 payload_pose_topic=None, payload_status_topic=None):
        import rospy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
        self._rospy = rospy
        self._PoseStamped = PoseStamped
        self._String = String
        rospy.init_node('foundationpose_live_tracking_2_objects', anonymous=True)
        self.frame_id = frame_id
        self.cam_pubs = [rospy.Publisher(f'camera_frame/{t}', PoseStamped,
                                         queue_size=1) for t in topics]
        self.robot_pubs = ([rospy.Publisher(f'robot_frame/{t}', PoseStamped,
                                            queue_size=1) for t in topics]
                           if publish_robot_frame else [None, None])
        self.payload_k = payload_k
        self.payload_pose_pub = self.payload_status_pub = None
        if payload_k is not None:
            self.payload_pose_pub = rospy.Publisher(payload_pose_topic, PoseStamped,
                                                    queue_size=1)
            self.payload_status_pub = rospy.Publisher(payload_status_topic, String,
                                                      queue_size=1)

    def _stamp(self, ts_ns):
        if ts_ns is None:
            return self._rospy.Time.now()
        ts_ns = int(ts_ns)
        return self._rospy.Time(secs=ts_ns // 1_000_000_000,
                                nsecs=ts_ns % 1_000_000_000)

    def _msg(self, T, frame_id, stamp):
        msg = self._PoseStamped()
        msg.header.stamp = stamp
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

    def publish(self, k, pose_cam, T_RC=None, ts_ns=None, status=None):
        """status: the payload verdict for this pose (dict with 'valid',
        'reason', ...), used only for object payload_k."""
        stamp = self._stamp(ts_ns)
        robot_msg = None
        if pose_cam is not None and np.isfinite(pose_cam).all():
            self.cam_pubs[k].publish(self._msg(pose_cam, self.frame_id, stamp))
            if self.robot_pubs[k] is not None and T_RC is not None:
                robot_msg = self._msg(T_RC @ pose_cam, 'robot_frame', stamp)
                self.robot_pubs[k].publish(robot_msg)
        if k != self.payload_k:
            return
        if robot_msg is not None:
            self.payload_pose_pub.publish(robot_msg)
        status = dict(status or {})
        body = {'valid': bool(status.get('valid', False)) and robot_msg is not None,
                'stamp': stamp.to_sec(),
                'reason': status.get('reason', '') if robot_msg is not None else 'no_pose',
                'source': 'foundationpose_live_tracking_2_objects'}
        for key in ('color_px', 'iou', 'depth_px', 'surface_frac', 'jump_m'):
            if key in status:
                v = status[key]
                body[key] = v if (not isinstance(v, float) or math.isfinite(v)) else None
        self.payload_status_pub.publish(self._String(data=_json.dumps(body)))

    def publish_objs(self, objs, T_RC=None):
        for o in objs:
            self.publish(o.k, o.pose, T_RC, ts_ns=o.ts_ns, status=o.status)

    def shutdown_requested(self):
        return self._rospy.is_shutdown()


class ReplayCapture:
    """Stands in for ZedCaptureProcess: serves recorded RGB-D frames.

    replay_dir holds rgb/*.png, depth/*.png (uint16, mm) or depth/*.npy
    (float, m) with the same stems, and cam_K.txt (FoundationPose's demo
    layout). Frames are resized to the tracker's size (fp_size = (W, H)) and K
    is scaled to match. One frame per latest() call; `exhausted` after the
    last. `frame_name` is the stem of the frame last served. An optional
    ts_ns.txt ("<stem> <capture ns>" per line) replays the original capture
    stamps; without it a frame is stamped when it is read.
    """

    def __init__(self, replay_dir, fp_size, realtime_scale=0.0, fps=30.0):
        # realtime_scale > 0 (19 Nov): skip frames like a live camera would. A virtual clock advances by the wall
        # time between latest() calls x realtime_scale (< 1 when this machine runs the replay slower than live, e.g.
        # 0.5 with a live tracker sharing the GPU); the served frame is the one the camera would have delivered.
        self.realtime_scale, self.fps, self._vclock, self._tlast = float(realtime_scale), float(fps), 0.0, None
        self.n_skipped = 0
        self.dir = Path(replay_dir)
        self.rgb_files = sorted((self.dir / 'rgb').glob('*.png'))
        if not self.rgb_files:
            raise FileNotFoundError(f'no rgb/*.png under {self.dir}')
        self.depth_files = {p.stem: p for p in (self.dir / 'depth').glob('*')
                            if p.suffix in ('.png', '.npy')}
        self.fp_size = tuple(fp_size)
        self.K = None
        self.baseline_m = 0.0
        self.producer_stats = None
        self.exhausted = False
        self.frame_name = None
        self._i = 0
        self.ts_ns = {}
        if (self.dir / 'ts_ns.txt').is_file():
            for line in (self.dir / 'ts_ns.txt').read_text().split('\n'):
                if line.strip():
                    stem, ns = line.split()
                    self.ts_ns[stem] = int(ns)

    def start(self, ready_timeout=None):
        K = np.loadtxt(self.dir / 'cam_K.txt').reshape(3, 3)
        h0, w0 = cv2.imread(str(self.rgb_files[0])).shape[:2]
        W, H = self.fp_size
        K = K.copy()
        K[0] *= W / w0
        K[1] *= H / h0
        self.K = K

    def latest(self, since_seq=0, timeout=2.0):
        if self._i >= len(self.rgb_files):
            self.exhausted = True
            return None
        if self.realtime_scale > 0:
            now = time.time()
            if self._tlast is not None:
                self._vclock += (now - self._tlast) * self.realtime_scale
            self._tlast = now
            want = int(self._vclock * self.fps)
            if want > self._i:
                self.n_skipped += min(want, len(self.rgb_files)) - self._i
                self._i = want
            if self._i >= len(self.rgb_files):
                self.exhausted = True
                return None
        p = self.rgb_files[self._i]
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        dp = self.depth_files[p.stem]
        if dp.suffix == '.npy':
            depth = np.load(dp).astype(np.float32)
        else:
            depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        W, H = self.fp_size
        if rgb.shape[1] != W or rgb.shape[0] != H:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        self.frame_name = p.stem
        self._i += 1
        return self._i, self.ts_ns.get(p.stem, time.time_ns()), rgb, depth

    def stop_capture(self):
        pass


class _ReplayDone(Exception):
    """The replay folder has no more frames."""


def _center_box_mask(rgb, h_frac, w_frac):
    """Fixed box at the image centre. No SAM, no clicking, no display."""
    h, w = rgb.shape[:2]
    dh, dw = int(h * h_frac), int(w * w_frac)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2 - dh:h // 2 + dh, w // 2 - dw:w // 2 + dw] = 1
    return mask


def _select_mask_safe(rgb):
    """select_mask_with_sam, but neutralizes the CUDA default tensor type that
    FoundationPose's Utils.py installs during register/track_one (as in
    live_tracking_new.py). Without this, transformers' from_pretrained inside
    the SAM loader raises ValueError('... requires accelerate')."""
    was_cuda = torch.tensor(0.).is_cuda
    if was_cuda:
        torch.set_default_tensor_type(torch.FloatTensor)
    try:
        return select_mask_with_sam(rgb)
    finally:
        if was_cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)


def _mask_from_text_joined(rgb, prompt, near_px, backend, min_score):
    """text_mask.mask_from_text, but with near_px > 0 (sam3 only) it joins every
    instance whose box centre lies within near_px of the best instance's: the
    screw head is two halves either side of its slot."""
    from text_mask import mask_from_text
    if near_px <= 0 or backend != 'sam3':
        return mask_from_text(rgb, prompt, backend=backend, min_score=min_score)
    from PIL import Image
    from text_mask import _cpu_default_tensor, _device, _load_sam3
    H, W = rgb.shape[:2]
    proc, model = _load_sam3('facebook/sam3')
    with _cpu_default_tensor():
        inputs = proc(images=Image.fromarray(rgb), text=prompt,
                      return_tensors='pt').to(_device())
    with torch.inference_mode():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(
        out, threshold=0.3, mask_threshold=0.5, target_sizes=[(H, W)])[0]
    scores = torch.as_tensor(res['scores']).float().cpu().numpy()
    if len(scores) == 0:
        return None, 0.0
    best = int(scores.argmax())
    if scores[best] < min_score:
        return None, float(scores[best])
    masks = [m.detach().cpu().numpy() > 0 for m in res['masks']]

    def centre(m):
        ys, xs = np.nonzero(m)
        if not len(xs):
            return np.array([np.inf, np.inf])
        return np.array([0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())])

    c0 = centre(masks[best])
    keep = [i for i, m in enumerate(masks) if np.all(np.abs(centre(m) - c0) < near_px)]
    return np.any([masks[i] for i in keep], axis=0).astype(np.uint8), float(scores[best])


def _project(K, pose, p_obj):
    q = pose @ p_obj
    if q[2] <= 1e-6:
        raise ValueError('behind the camera')
    uv = K @ (q[:3] / q[2])
    return int(round(uv[0])), int(round(uv[1]))


def _draw_axes(img, K, pose, k, length=0.1):
    """Object k's axes (10 cm) and label onto img (RGB, in place)."""
    if pose is None or K is None or not np.all(np.isfinite(pose)):
        return
    cols = _OBJ_AXIS_COLORS[k]
    try:
        o = _project(K, pose, np.array([0.0, 0.0, 0.0, 1.0]))
        ends = [_project(K, pose, np.array([*(length * e), 1.0]))
                for e in np.eye(3)]
    except ValueError:
        return
    for end, c in zip(ends, cols):
        cv2.arrowedLine(img, o, end, c, 3, cv2.LINE_AA)
    cv2.putText(img, f'obj{k + 1}', (o[0] + 6, o[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cols[0], 2, cv2.LINE_AA)


def _pose_array(pose_log):
    poses = np.full((len(pose_log), 4, 4), np.nan, dtype=np.float64)
    for i, p in enumerate(pose_log):
        if p is not None:
            poses[i] = p
    return poses


def _stage_stats_ms(times):
    t = np.asarray(times, dtype=np.float64)
    return {'mean': float(t.mean() * 1e3), 'p50': float(np.median(t) * 1e3),
            'p95': float(np.percentile(t, 95) * 1e3),
            'min': float(t.min() * 1e3), 'max': float(t.max() * 1e3)}


def _save_run_2(save_dir, backend_name, resolution, raw_frames, pose_logs, K,
                loop_start_time, frame_times, target_fps, stage_times=None,
                encode_preset=None, encode_crf=None, write_overlay=True):
    """live_tracking_utils.save_run for two objects: raw.mp4, overlay.mp4 with
    both objects' axes, poses_object_{1,2}.npy, K.txt and metrics.json in a
    fresh datetimed take directory. Same encoder settings as save_run."""
    if not raw_frames or loop_start_time is None:
        return None
    elapsed = max(time.time() - loop_start_time, 1e-6)
    measured_fps = max(1, int(round(len(raw_frames) / elapsed)))
    stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(
        save_dir, f'{stamp}_{backend_name}_2objects_{resolution}_fps{measured_fps}')
    os.makedirs(run_dir, exist_ok=True)

    enc = {'macro_block_size': 1}
    params = []
    if encode_preset:
        params += ['-preset', str(encode_preset)]
    if encode_crf is not None:
        params += ['-crf', str(encode_crf)]
    else:
        enc['quality'] = 8
    if params:
        enc['output_params'] = params

    raw_path = os.path.join(run_dir, 'raw.mp4')
    writer = imageio.get_writer(raw_path, fps=measured_fps, codec='libx264', **enc)
    for f in raw_frames:
        writer.append_data(f)
    writer.close()

    overlay_path = '(skipped)'
    if write_overlay:
        overlay_path = os.path.join(run_dir, 'overlay.mp4')
        writer = imageio.get_writer(overlay_path, fps=measured_fps,
                                    codec='libx264', **enc)
        for i, f in enumerate(raw_frames):
            ov = f.copy()
            for k in (0, 1):
                _draw_axes(ov, K, pose_logs[k][i] if i < len(pose_logs[k]) else None, k)
            writer.append_data(ov)
        writer.close()

    for k in (0, 1):
        np.save(os.path.join(run_dir, f'poses_object_{k + 1}.npy'),
                _pose_array(pose_logs[k]))
    np.savetxt(os.path.join(run_dir, 'K.txt'), np.asarray(K, dtype=np.float64))
    print(f'Saved: {raw_path}, {overlay_path}, poses_object_1/2.npy '
          f'({len(raw_frames)} frames @ {measured_fps} FPS over {elapsed:.2f}s)')

    if frame_times:
        ft = np.asarray(frame_times, dtype=np.float64)
        fps_arr = 1.0 / np.clip(ft, 1e-6, None)
        hit_pct = float((fps_arr >= target_fps).mean() * 100.0) if target_fps > 0 else None
        stages = {name: _stage_stats_ms(ts)
                  for name, ts in (stage_times or {}).items() if ts}
        metrics = {
            'backend': backend_name, 'resolution': resolution,
            'timestamp': stamp, 'n_frames': int(len(raw_frames)),
            'elapsed_s': float(elapsed), 'target_fps': float(target_fps),
            'achieved_fps': float(measured_fps),
            'per_frame_ms': _stage_stats_ms(frame_times), 'stages_ms': stages,
            'hit_target_pct': hit_pct,
            'budget_ms': float(1e3 / target_fps) if target_fps > 0 else None,
        }
        metrics_path = os.path.join(run_dir, 'metrics.json')
        with open(metrics_path, 'w') as fh:
            _json.dump(metrics, fh, indent=2)
        lines = ['--- Loop timing (both objects) ---',
                 f'  target FPS:           {target_fps:.1f}',
                 f'  achieved (wall) FPS:  {measured_fps:.1f}',
                 f'  per-frame total (ms): mean={ft.mean()*1e3:.1f}  '
                 f'p50={np.median(ft)*1e3:.1f}  p95={np.percentile(ft, 95)*1e3:.1f}  '
                 f'min={ft.min()*1e3:.1f}  max={ft.max()*1e3:.1f}']
        if stages:
            lines.append('  per-stage (ms):          mean    p50    p95    min    max')
            for name, s in stages.items():
                lines.append(f"    {name:<13s}{s['mean']:7.1f}{s['p50']:7.1f}"
                             f"{s['p95']:7.1f}{s['min']:7.1f}{s['max']:7.1f}")
        lines.append(f'  metrics: {metrics_path}')
        print('\n'.join(lines))
    return run_dir


def _save_viz_data_2(run_dir, viz_log, pose_logs, K, stride, mesh_paths,
                     compress=False):
    """viz_data.npz with both pose streams. Keys: rgb, depth (strided), poses
    (= object 1, so live_tracking_viz_replay.py still opens it), poses_1,
    poses_2, K, stride, mesh_path (= object 1), mesh_path_1, mesh_path_2."""
    if not viz_log:
        return None
    rgb_s = np.stack([r for r, _ in viz_log], axis=0)
    depth_s = np.stack([d for _, d in viz_log], axis=0)
    n = len(viz_log)
    p1, p2 = (_pose_array(list(pose_logs[k])[:n]) for k in (0, 1))
    path = os.path.join(run_dir, 'viz_data.npz')
    (np.savez_compressed if compress else np.savez)(
        path, rgb=rgb_s, depth=depth_s, poses=p1, poses_1=p1, poses_2=p2,
        K=np.asarray(K, dtype=np.float64), stride=np.int64(stride),
        mesh_path=str(mesh_paths[0]), mesh_path_1=str(mesh_paths[0]),
        mesh_path_2=str(mesh_paths[1]), timestamps=np.zeros(n, dtype=np.int64))
    print(f'[viz] saved two-object viz data to {path} ({n} frames)')
    return path


def _build_parser():
    parser = argparse.ArgumentParser(
        description='Live 6D pose tracking of two objects (viser GUI).')
    parser.add_argument('--depth_backend', choices=['zed_neural', 'fast_fs'],
                        default=None,
                        help='Which depth source to feed FoundationPose. '
                             'Required unless --replay_dir.')
    parser.add_argument('--mesh_path', type=str, required=True,
                        help='Object 1 mesh (.obj, in meters).')
    parser.add_argument('--mesh_path_2', type=str, required=True,
                        help='Object 2 mesh (.obj, in meters).')
    parser.add_argument('--ros', action='store_true',
                        help='Publish PoseStamped for both objects on '
                             'camera_frame/<topic> and robot_frame/<topic> '
                             '(robot frame uses T_RC_DEFAULT unless '
                             '--calibration overrides it).')
    parser.add_argument('--pose_topic', type=str, default='current_object_pose',
                        help='Object 1 topic name (under camera_frame/ and '
                             'robot_frame/).')
    parser.add_argument('--pose_topic_2', type=str,
                        default='current_object_pose_2',
                        help='Object 2 topic name, e.g. workpiece_pose for '
                             'deployment/tool_program_node.py.')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Optional 4x4 camera-to-robot transform '
                             '(.npy or .txt) overriding T_RC_DEFAULT.')
    parser.add_argument('--no_robot_frame', action='store_true',
                        help='With --ros, skip the robot_frame publishes.')
    parser.add_argument('--payload_object', type=int, choices=[0, 1, 2], default=0,
                        help='With --ros: also publish this object (1 or 2) as the '
                             'payload: its robot-frame pose on --payload_pose_topic '
                             'and a JSON validity status on --payload_status_topic. '
                             '0 = off.')
    parser.add_argument('--payload_pose_topic', type=str,
                        default='/perception/payload/pose_robot')
    parser.add_argument('--payload_status_topic', type=str,
                        default='/perception/payload/status')
    parser.add_argument('--payload_config', type=str, default=None,
                        help='JSON whose "cube" section (or top level) overrides '
                             'payload_check.SWEEP_CUBE_29MM, e.g. the sweep '
                             "deployment's sweep_payload_29mm.json.")
    parser.add_argument('--serial_number', type=str, default='15107')
    parser.add_argument('--camera_upsidedown', action='store_true')
    parser.add_argument('--camera_resolution', choices=['HD1080', 'HD720', 'VGA'],
                        default='HD1080',
                        help='ZED capture resolution. HD1080=30fps max, HD720=60fps, VGA=100fps.')
    parser.add_argument('--camera_fps', type=int, default=0,
                        help='ZED capture FPS. 0 = use the resolution default.')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--exposure', type=int, default=25)
    parser.add_argument('--gain', type=int, default=40)
    parser.add_argument('--fps', type=float, default=40.0,
                        help='Target tracking-loop FPS (both objects per '
                             'iteration).')
    parser.add_argument('--est_refine_iter', type=int, default=5,
                        help='Refinement iterations for initial registration.')
    parser.add_argument('--track_refine_iter', type=int, default=2,
                        help='Refinement iterations per tracking step.')
    parser.add_argument('--save_dir', type=str,
                        default='live_tracking_2_objects_results')
    parser.add_argument('--max_buffer_seconds', type=float, default=60.0,
                        help='Keep only this many seconds of frames/poses in RAM '
                             'before dropping the oldest. 0 disables the cap.')
    parser.add_argument('--max_buffer_frames', type=int, default=0,
                        help='Exact frame buffer cap. 0 derives it from '
                             '--max_buffer_seconds and --fps.')
    parser.add_argument('--frame_id', type=str, default='camera_frame')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='Run the tracking loop for this many seconds '
                             '(after the initial register) then exit. 0 = run '
                             'until ROS shutdown.')
    # Fast-FS-specific knobs (ignored when --depth_backend=zed_neural).
    parser.add_argument('--fast_fs_model', type=str, default=str(DEFAULT_FAST_FS_MODEL))
    parser.add_argument('--fast_fs_iters', type=int, default=4)
    parser.add_argument('--fast_fs_max_disp', type=int, default=192)
    parser.add_argument('--fast_fs_width', type=int, default=384)
    parser.add_argument('--fast_fs_height', type=int, default=224)
    parser.add_argument('--fast_fs_trt_dir', type=str,
                        default=str(DEFAULT_FAST_FS_TRT_DIR))
    parser.add_argument('--visualize', action='store_true',
                        help='After the run, open the two-object viser replay '
                             '(implies --viz_data).')
    parser.add_argument('--viz_stride', type=int, default=4,
                        help='Stride for the per-frame point cloud sampling.')
    parser.add_argument('--fp_refiner_trt', type=str,
                        default=str(_FP_DIR / 'weights' / '2023-10-28-18-33-37'
                                    / 'refine_net_fp16.plan'),
                        help='TensorRT engine (.plan) for the FP refiner. Pass '
                             'an empty string to fall back to PyTorch.')
    # ---- auto-record gating -------------------------------------------------
    parser.add_argument('--no_auto_record', action='store_true',
                        help='Disable policy-gated recording (buffer '
                             'everything, save on quit / the panel).')
    parser.add_argument('--policy_topic', type=str, default='/iiwa/joint_cmd',
                        help='JointState topic whose publishing gates recording.')
    parser.add_argument('--policy_idle_timeout', type=float, default=0.5,
                        help='Stop the take after this many seconds with no '
                             'message on --policy_topic.')
    parser.add_argument('--policy_start_msgs', type=int, default=2,
                        help='Messages needed before a take opens.')
    parser.add_argument('--selftest_swap', action='store_true',
                        help='Drive the panel from a timer instead of a '
                             'browser: re-register each object, swap object '
                             "2's mesh (first --mesh_choices) and prompt, "
                             're-register both (and object 2 by clicking, '
                             'with --mask_mode points).')
    parser.add_argument('--selftest_saves', action='store_true',
                        help='Selftest variant that fires repeated saves.')
    parser.add_argument('--keyboard_window', action='store_true',
                        help='Also open the OpenCV keyboard window alongside '
                             'the viser panel.')
    parser.add_argument('--no_gui', action='store_true',
                        help='Disable the viser control panel.')
    parser.add_argument('--live_view', choices=['3d', '2d', 'off'], nargs='*',
                        default=None,
                        help="Which of the panel's live views start on: 3d (point "
                             'cloud + both meshes + the image in the camera '
                             'frustum, robot frame) and/or 2d (the image with '
                             'both poses), or off. Default (and bare '
                             '--live_view): both. Each can be switched on and '
                             'off from the panel either way; off costs nothing.')
    parser.add_argument('--live_view_hz', type=float, default=5.0,
                        help='Live 3D view update rate (also a panel slider).')
    parser.add_argument('--live_view_stride', type=int, default=4,
                        help='Live 3D view point cloud pixel stride.')
    parser.add_argument('--gui_port', type=int, default=8100,
                        help='Port for the viser panel: the first free one '
                             'from this up (the same URL on every rerun). '
                             '0 picks any free port.')
    parser.add_argument('--mesh_choices', type=str, nargs='*', default=None,
                        help='Extra mesh paths offered in both objects\' '
                             'dropdowns for mid-run swapping.')
    parser.add_argument('--headless', action='store_true',
                        help='Never open the OpenCV keyboard-control window.')
    # ---- encoder -------------------------------------------------------------
    parser.add_argument('--encode_preset', type=str, default='veryfast',
                        help="x264 preset for the saved videos ('' = ffmpeg's "
                             'default).')
    parser.add_argument('--encode_crf', type=int, default=20,
                        help='x264 -crf. -1 restores the old quality=8.')
    parser.add_argument('--no_overlay_video', action='store_true',
                        help='Skip overlay.mp4 (halves encode time).')
    parser.add_argument('--viz_data', action='store_true',
                        help='Also write viz_data.npz (strided RGB + depth and '
                             'both pose streams) for '
                             'live_tracking_viz_replay_2_objects.py.')
    parser.add_argument('--viz_compress', action='store_true',
                        help='Compress viz_data.npz (slow to write).')
    # ---- masks -------------------------------------------------------------
    parser.add_argument('--text_prompt', type=str, default=None,
                        help="Object 1 in words, e.g. 'gray wrench'. Object 1 "
                             'then uses text masking.')
    parser.add_argument('--text_prompt_2', type=str, default=None,
                        help="Object 2 in words, e.g. 'small gray cube'.")
    parser.add_argument('--text_backend', choices=['sam3', 'gdino'],
                        default='sam3',
                        help="Text-masking backend (see live_tracking_new_gui.py).")
    parser.add_argument('--text_min_score', type=float, default=0.3,
                        help='Reject a text match below this confidence.')
    parser.add_argument('--text_join_px', type=int, default=0,
                        help='Object 1: join every SAM3 instance whose box '
                             'centre is within this many px of the best one '
                             "(0 = the best instance only).")
    parser.add_argument('--text_join_px_2', type=int, default=0,
                        help='Object 2: the same. 60 for the screw head.')
    parser.add_argument('--text_fallback_click', action='store_true',
                        help='If a text prompt matches nothing, open the popup '
                             '(drag a box / click points) instead of giving up.')
    parser.add_argument('--mask_mode', choices=['sam', 'points', 'center_box', 'text', 'replay'],
                        default='sam',
                        help="How to mask an object that has no text prompt. "
                             "'points': a popup per object -- drag a box and/or "
                             "click points (right click: not the object), SAM "
                             "shows the mask, Enter accepts. 'sam': the "
                             "interactive 4-corner box picker. 'center_box': a "
                             "fixed centre box (the same for both objects; "
                             "smoke tests only). 'replay': "
                             "masks_<k>/<frame>.png under --replay_dir.")
    parser.add_argument('--center_box_frac', type=float, nargs=2,
                        default=(0.25, 0.33), metavar=('H_FRAC', 'W_FRAC'))
    parser.add_argument('--object_name', type=str, default=None,
                        help='Object 1 name for metadata.json and the take '
                             'directory. Defaults to the prompt or mesh stem.')
    parser.add_argument('--object_name_2', type=str, default=None,
                        help='Object 2 name.')
    # ---- symmetric parts -----------------------------------------------------
    parser.add_argument('--upright_meshes', type=str, nargs='*',
                        default=list(_UPRIGHT_MESHES_DEFAULT),
                        help='Mesh file stems registered upright: at register '
                             'time keep the copy whose +z points up (robot +Z).')
    parser.add_argument('--no_upright', action='store_true',
                        help='Never turn a registered pose over.')
    parser.add_argument('--force_upright', action='store_true',
                        help='For --upright_meshes: register from upright '
                             'guesses only (+z up, 24 headings) and keep +z '
                             'exactly up every tracked frame (rotating about '
                             'the mesh centre), so only the heading and the '
                             'position are tracked.')
    parser.add_argument('--lock_position', action='store_true',
                        help='For --upright_meshes: also keep the mesh centre '
                             'where register put it (re-registering moves it), '
                             'so only the heading about +z is tracked. Implies '
                             '--force_upright.')
    parser.add_argument('--track_hz_2', type=float, default=0.0,
                        help='Track object 2 only this many times per second '
                             '(e.g. 1 for a bowl or a screw that stays still); '
                             'between updates its last pose is published '
                             'again. 0 (default): track it every frame.')
    # ---- offline -------------------------------------------------------------
    parser.add_argument('--replay_realtime_scale', type=float, default=0.0,
                        help='With --replay_dir: > 0 skips frames like a live 30 fps camera, with processing time '
                             'scaled by this (0.5 = this replay runs 2x slower than live). 0 = every frame.')
    parser.add_argument('--replay_dir', type=str, default=None,
                        help='Read RGB-D frames from this folder (rgb/, '
                             'depth/, cam_K.txt) instead of the ZED. The run '
                             'ends after the last frame.')
    # ---- SAM2 checks on object 1 (robust copy, 29 Oct 2026) --------------------
    parser.add_argument('--sam2_tool', action='store_true',
                        help='Track object 1 with SAM2 and check FoundationPose against it (robust_tool.py).')
    parser.add_argument('--sam2_ckpt', type=str, default=None,
                        help='SAM2 checkpoint (default: the real-time fork tiny under /home/tylerlum/.deps_sam2rt_20261029)')
    parser.add_argument('--sam2_cfg', type=str, default=None, help='SAM2 config name (default sam2_hiera_t.yaml)')
    parser.add_argument('--sam2_image_size', type=int, default=1024,
                        help='SAM2 input size (multiple of 32). 640: ~13 ms/frame vs ~21 ms at 1024 on the 4090, '
                             'mask IoU 0.958 vs 0.964 (18 Nov synthetic test).')
    parser.add_argument('--sam2_async', action='store_true',
                        help='Run SAM2 on its own thread / CUDA stream (overlaps FoundationPose). Each frame then uses '
                             'the previous frame\'s mask, compared with the previous frame\'s pose.')
    parser.add_argument('--sam2_cpu_prep', action='store_true',
                        help='Use SAM2\'s stock CPU frame preparation (default: same steps on the GPU, ~12 ms faster).')
    parser.add_argument('--check_cpu_silhouette', action='store_true',
                        help='Rasterise the check silhouette on the CPU (default: nvdiffrast, ~3 ms vs ~12 ms).')
    parser.add_argument('--sam2_every', type=int, default=1,
                        help='Run SAM2 every N frames (~18 ms each, tiny, 960 x 540); reuse its last mask between.')
    parser.add_argument('--lost_iou', type=float, default=0.1)
    parser.add_argument('--lost_px', type=float, default=30.0)
    parser.add_argument('--lost_frames', type=int, default=4)
    parser.add_argument('--lost_metric', choices=['iou', 'cover'], default='iou',
                        help='cover (18 Nov, for held tools): bad when < --lost_cover of SAM2\'s mask lies inside '
                             'FoundationPose\'s silhouette (iou is ~0.1 under heavy occlusion even when right).')
    parser.add_argument('--lost_cover', type=float, default=0.3)
    parser.add_argument('--lost_try_anchor', action='store_true',
                        help='When lost, first move the last pose onto SAM2\'s mask (silhouette delta) and refine with '
                             '--lost_anchor_iter track iterations on mask-blanked depth; keep it if coverage >= '
                             '--lost_anchor_accept, else re-register as before (a re-register from a sliver can flip).')
    parser.add_argument('--lost_anchor_iter', type=int, default=4)
    parser.add_argument('--lost_anchor_accept', type=float, default=0.6)
    parser.add_argument('--lost_action', choices=['report', 'reregister'], default='reregister')
    parser.add_argument('--lost_cooldown_s', type=float, default=1.0)
    parser.add_argument('--lost_max_cooldown_s', type=float, default=8.0,
                        help='Fixes that keep failing double the cooldown up to this (re-registration blocks ~0.6 s).')
    parser.add_argument('--lost_mask_max_ratio', type=float, default=3.0,
                        help='Re-register from SAM2 only if its mask is <= this x the registration mask (not the hand).')
    parser.add_argument('--arbitrate', nargs='?', const='depth', default='off', choices=['off', 'depth', 'color'],
                        help='(19 Nov) Before trusting a SAM2-vs-FoundationPose disagreement, check FoundationPose '
                             'against the observed depth. If SAM2\'s mask grew past --bleed_ratio x the whole '
                             'silhouette while the pose explains the depth (fit >= --fit_ok), SAM2 bled onto the '
                             'hand/arm: not lost, re-seed SAM2 from the pose\'s visible pixels. After a '
                             're-registration keep the new pose only if cover x fit beats the old one.')
    parser.add_argument('--sam2_simple_prompt', action='store_true',
                        help='Prompt SAM2 with box + 1 point (before 19 Nov). Default: box + 4 positive points + 6 '
                             'negative points around the mask, which stops SAM2 taking the hand and arm.')
    parser.add_argument('--log_checks', action='store_true',
                        help='(19 Nov) Log, every --color_every frames, object 1\'s mesh-colour error (ce), depth fit '
                             '(fit), visible pixels (n_vis) and learned-appearance distance (app: HSV histogram vs the '
                             'first frames after registration) into robust_log.json. Logging only (~3-6 ms on those '
                             'frames); use with --save_depth to tune thresholds offline.')
    parser.add_argument('--bleed_ratio', type=float, default=1.5)
    parser.add_argument('--color_good', type=float, default=0.15,
                        help='--arbitrate color: the pose is trusted when its rendered colours differ from the image by '
                             'less than this (mean |rgb|, 0-1) on >= --color_min_px visible pixels.')
    parser.add_argument('--color_bad', type=float, default=0.22)
    parser.add_argument('--color_bad_checks', type=int, default=3,
                        help='--arbitrate color: lost after this many checks in a row with colour error > --color_bad, '
                             'even if SAM2 agrees (catches poses on the hand and upside-down tools).')
    parser.add_argument('--color_every', type=int, default=2)
    parser.add_argument('--color_mid_checks', type=int, default=10,
                        help='--arbitrate color: after this many checks in a row between --color_good and '
                             '--color_bad, try a (verified) re-registration.')
    parser.add_argument('--color_min_px', type=int, default=400)
    parser.add_argument('--flip_check_every', type=int, default=0,
                        help='(19 Nov) Every N object-1 frames, refine the pose and its 180-degree twins about '
                             '--flip_axes (mesh frame) and keep the one FoundationPose\'s scorer ranks best (margin '
                             '--flip_margin). Fixes the tool tracked upside down after occlusion, which no 2D mask '
                             'check sees. 0 = off.')
    parser.add_argument('--flip_axes', type=str, default='x', help='Subset of xyz, e.g. x or xz.')
    parser.add_argument('--flip_refine_iter', type=int, default=1)
    parser.add_argument('--flip_margin', type=float, default=0.0, help='scorer judge: score margin to switch.')
    parser.add_argument('--flip_judge', choices=['color', 'scorer'], default='color',
                        help='color: render the textured mesh at each candidate and compare with the image on its '
                             'visible, unoccluded pixels (the scorer\'s margins between twins were bf16 noise in '
                             'sim); switch if the twin\'s colour error < --flip_ratio x the current one.')
    parser.add_argument('--flip_ratio', type=float, default=0.8)
    parser.add_argument('--flip_min_px', type=int, default=400)
    parser.add_argument('--fit_tol', type=float, default=0.015, help='Depth match tolerance (m) for --arbitrate.')
    parser.add_argument('--fit_ok', type=float, default=0.6)
    parser.add_argument('--verify_margin', type=float, default=0.05)
    parser.add_argument('--sam_reseed_min_s', type=float, default=0.5)
    # FoundationPose++-style aids (fpp_tracking.py, 18 Nov 2026); all need --sam2_tool, all default off.
    parser.add_argument('--save_depth', action='store_true',
                        help='Also write <take>/replay/ (rgb/*.png, depth/*.png uint16 mm, ts_ns.txt, cam_K.txt, '
                             'masks_1/2 from the first frame\'s poses) so the take can be re-run with --replay_dir. '
                             '~1 MB RAM per buffered frame; saving takes a few seconds longer.')
    parser.add_argument('--fpp_anchor_xy', action='store_true',
                        help='Before track_one, move object 1\'s starting pose so its centre projects onto the centre '
                             'of SAM2\'s mask (FoundationPose++). SAM2 then runs every frame.')
    parser.add_argument('--fpp_anchor_mode', choices=['silhouette', 'centre'], default='silhouette',
                        help='silhouette: shift by (mask centre - centre of the silhouette part inside the mask), only '
                             'when coverage < --fpp_anchor_gate_cover (leaves good tracks alone). centre: '
                             'FoundationPose++\'s rule, put the pose centre on the mask centre every frame.')
    parser.add_argument('--fpp_anchor_gate_cover', type=float, default=0.7,
                        help='Silhouette anchor only when less than this fraction of SAM2\'s mask lies inside '
                             'FoundationPose\'s silhouette (i.e. it has started to slide off).')
    parser.add_argument('--fpp_max_shift_px', type=float, default=60.0,
                        help='Skip the xy anchor on frames where it would move the start more than this (px).')
    parser.add_argument('--fpp_mask_depth_px', type=int, default=-1,
                        help='>= 0: track object 1 on depth blanked outside SAM2\'s mask dilated by this many px, so '
                             'FoundationPose cannot slide onto the hand / arm / table. -1 = off.')
    parser.add_argument('--fpp_filter_alpha', type=float, default=0.5,
                        help='Velocity low-pass for --fpp_filter (19 Nov sim: 0.5 at 30 Hz, 0.8 when frames skip).')
    parser.add_argument('--fpp_filter_max_v', type=float, default=0.05, help='m per frame cap.')
    parser.add_argument('--fpp_filter_max_w', type=float, default=20.0, help='deg per frame cap.')
    parser.add_argument('--fpp_filter', action='store_true',
                        help='Start object 1\'s track_one from a constant-velocity prediction of its last poses.')
    parser.add_argument('--fpp_min_mask_px', type=int, default=60,
                        help='Use the aids only on frames whose SAM2 mask has at least this many pixels '
                             '(and at most --lost_mask_max_ratio x the registration mask).')
    parser.add_argument('--reseed_topic', type=str, default='',
                        help='PoseStamped (robot frame) to restart FoundationPose from when lost, e.g. '
                             '/robot_frame/tool_pose_node (the node with --tool_hand_hold_s). Needs --calibration.')
    parser.add_argument('--reseed_max_age_s', type=float, default=0.2)
    parser.add_argument('--reseed_on_message', action='store_true',
                        help='(29 Oct, the two-camera hand-off) restart FoundationPose from EVERY new message on '
                             '--reseed_topic (deployment/fp_kit/tool_camera_selector.py sends one when this '
                             'camera disagrees with the held tool), not only when SAM2 says lost')
    parser.add_argument('--reseed_message_action', choices=['seed', 'register'], default='seed',
                        help='(29 Oct) with --reseed_on_message: seed = restart from the message pose; register = '
                             're-register object 1 on THIS camera from SAM2\'s mask (its own roll; the other '
                             'camera\'s roll is what goes wrong at the lift), at most once per --lost_cooldown_s, '
                             'the message pose only if SAM2\'s mask is unusable')
    parser.add_argument('--end_tag_id', type=int, default=-1,
                        help='ArUco id of the tag on object 1\'s flat end (the driver: 15 mm 4x4_50 id 1, TOP edge '
                             'toward the red fin). -1: off. Needs --width 1920 --height 1080 (at 960 x 540 the tag '
                             'is ~10 px).')
    parser.add_argument('--end_tag_size_m', type=float, default=0.015, help='the black square (m)')
    parser.add_argument('--end_tag_dict', type=str, default='DICT_4X4_50')
    parser.add_argument('--end_tag_end_x', type=float, default=-0.05, help='the end face along object 1\'s x (m)')
    parser.add_argument('--end_tag_roll_deg', type=float, default=0.0,
                        help='the sticker\'s roll about the shaft; learned from the first --end_tag_calib_frames '
                             'trusted frames unless --end_tag_no_calib')
    parser.add_argument('--end_tag_no_calib', action='store_true')
    parser.add_argument('--end_tag_calib_frames', type=int, default=15)
    parser.add_argument('--end_tag_roi_px', type=int, default=200,
                        help='search this many px around FoundationPose\'s end face (0: the whole frame)')
    parser.add_argument('--end_tag_fix_deg', type=float, default=15.0,
                        help='restart FoundationPose from the tag (its shaft + the tag\'s roll and end position) '
                             'when the roll differs by more than this')
    parser.add_argument('--end_tag_fix_m', type=float, default=0.015, help='... or the end face by more than this')
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


class _Obj:
    """One tracked object's state."""

    def __init__(self, k, mesh_path, text_prompt, object_name, join_px, topic):
        self.k = k
        self.mesh_path = mesh_path
        self.text_prompt = text_prompt
        self.object_name_arg = object_name
        self.join_px = join_px
        self.topic = topic
        self.mesh = None
        self.est = None
        self.pose = None
        self.upright = False
        self.n_turned = 0          # registrations turned upright
        self.ts_ns = None          # capture stamp of the frame behind self.pose
        self.status = None         # payload verdict for self.pose (payload object only)
        self.check = None          # PayloadCheck (payload object only)

    @property
    def name(self):
        return (self.object_name_arg
                or (self.text_prompt.replace(' ', '_') if self.text_prompt else None)
                or Path(self.mesh_path).stem)

    def mask_mode(self, args):
        return 'text' if self.text_prompt else args.mask_mode


def main():
    _import_heavy()
    args = _build_parser().parse_args()
    replay = args.replay_dir is not None
    if args.lock_position:
        args.force_upright = True
    if not replay and args.depth_backend is None:
        sys.exit('--depth_backend is required (unless --replay_dir)')

    auto_record = not args.no_auto_record
    if auto_record and not args.ros:
        sys.exit('--ros is required for auto-record (it watches '
                 f'{args.policy_topic}). Pass --no_auto_record to run without ROS.')
    if args.mask_mode == 'text':
        if not (args.text_prompt and args.text_prompt_2):
            sys.exit('--mask_mode text requires --text_prompt and --text_prompt_2')
    if args.mask_mode == 'replay' and not replay:
        sys.exit('--mask_mode replay requires --replay_dir')
    if args.mask_mode == 'points' and not os.environ.get('DISPLAY'):
        sys.exit('--mask_mode points opens a popup: run it on the workstation '
                 'desktop or over ssh -X (no DISPLAY here)')
    if args.visualize:
        args.viz_data = True
    _want_viz = bool(args.viz_data)

    objs = [_Obj(0, args.mesh_path, args.text_prompt, args.object_name,
                 args.text_join_px, args.pose_topic),
            _Obj(1, args.mesh_path_2, args.text_prompt_2, args.object_name_2,
                 args.text_join_px_2, args.pose_topic_2)]
    buffer_frame_cap = _buffer_frame_cap(args)

    T_RC = None
    if args.ros and not args.no_robot_frame:
        if args.calibration is not None:
            arr = np.load(args.calibration) if args.calibration.endswith('.npy') \
                  else np.loadtxt(args.calibration)
            T_RC = arr.reshape(4, 4)
        else:
            T_RC = T_RC_DEFAULT
    # "Up" for upright registration: the robot's +Z in camera coordinates,
    # from the calibration (or T_RC_DEFAULT), whether or not --ros publishes.
    if T_RC is not None:
        T_up = T_RC
    elif args.calibration is not None:
        arr = np.load(args.calibration) if args.calibration.endswith('.npy') \
              else np.loadtxt(args.calibration)
        T_up = arr.reshape(4, 4)
    else:
        T_up = T_RC_DEFAULT
    up_cam = np.asarray(T_up, dtype=np.float64)[:3, :3].T @ np.array([0.0, 0.0, 1.0])

    payload_k = args.payload_object - 1 if args.payload_object else None
    payload_cfg = None
    if payload_k is not None:
        if T_RC is None:
            sys.exit('--payload_object needs --ros and the robot frame '
                     '(not --no_robot_frame)')
        from payload_check import SWEEP_CUBE_29MM
        payload_cfg = dict(SWEEP_CUBE_29MM)
        if args.payload_config:
            with open(args.payload_config) as fh:
                _pc = _json.load(fh)
            _pc = _pc.get('cube', _pc)
            payload_cfg.update({k: _pc[k] for k in SWEEP_CUBE_29MM if k in _pc})
        print(f'[payload] object {args.payload_object} -> {args.payload_pose_topic} + '
              f'{args.payload_status_topic}; check {payload_cfg}', flush=True)

    ros_pub = (RosPosePublisher2(args.frame_id, publish_robot_frame=(T_RC is not None),
                                 topics=[o.topic for o in objs], payload_k=payload_k,
                                 payload_pose_topic=args.payload_pose_topic,
                                 payload_status_topic=args.payload_status_topic)
               if args.ros else None)

    os.makedirs(args.save_dir, exist_ok=True)
    raw_frames = _deque(maxlen=buffer_frame_cap)
    pose_logs = [_deque(maxlen=buffer_frame_cap), _deque(maxlen=buffer_frame_cap)]
    frame_times = _deque(maxlen=buffer_frame_cap)
    stage_times = {name: _deque(maxlen=buffer_frame_cap)
                   for name in ('grab', 'depth', 'track_1', 'track_2',
                                'publish', 'total')}
    viz_log = _deque(maxlen=buffer_frame_cap)
    robust_log = _deque(maxlen=buffer_frame_cap)     # robust copy: object 1's SAM2 check per recorded frame
    frame_log = _deque(maxlen=buffer_frame_cap)   # per frame: capture stamps + payload verdict
    depth_frames = _deque(maxlen=buffer_frame_cap)   # --save_depth: uint16 mm per recorded frame
    if buffer_frame_cap is None:
        print('[buffer] frame buffer cap disabled')
    else:
        print(f'[buffer] retaining newest {buffer_frame_cap} frames in RAM')
    K = None
    loop_start_time = None
    backend_tag = 'replay' if replay else args.depth_backend
    capture = None
    interrupted = False

    def _save_loop_start_time():
        if buffer_frame_cap is not None and frame_times:
            return time.time() - float(sum(frame_times))
        return loop_start_time

    # Defined before the try so the finally block can always reach them.
    gui_state = None
    gui_stop = None
    live_view = None
    watcher = None

    def _clear_buffers():
        raw_frames.clear()
        for log in pose_logs:
            log.clear()
        frame_times.clear()
        for v in stage_times.values():
            v.clear()
        viz_log.clear()
        robust_log.clear()
        frame_log.clear()
        depth_frames.clear()

    def _write_take(reason, raw, poses, fts, sts, vizs, started, flog, dlog=None):
        """Encode and write one take from a snapshot of the buffers."""
        _t_enc = time.time()
        n = len(raw)
        recorded_s = float(sum(fts))
        take_dir = _save_run_2(
            save_dir=args.save_dir,
            backend_name=f'{backend_tag}_{objs[0].name}+{objs[1].name}',
            resolution='replay' if replay else args.camera_resolution,
            raw_frames=raw, pose_logs=poses, K=K, loop_start_time=started,
            frame_times=fts, stage_times=sts, target_fps=args.fps,
            encode_preset=(args.encode_preset or None),
            encode_crf=(None if args.encode_crf < 0 else args.encode_crf),
            write_overlay=not args.no_overlay_video,
        )
        if take_dir:
            meta = {
                'objects': [{
                    'name': o.name, 'text_prompt': o.text_prompt,
                    'mask_mode': o.mask_mode(args),
                    'mesh_path': os.path.abspath(o.mesh_path),
                    'topic': o.topic, 'text_join_px': o.join_px,
                    'upright_at_register': o.upright,
                    'registrations_turned_upright': o.n_turned,
                    'poses_file': f'poses_object_{o.k + 1}.npy',
                } for o in objs],
                'poses_frame': 'camera (OpenCV); robot = T_RC @ camera',
                'T_RC': (np.asarray(T_RC).tolist() if T_RC is not None else None),
                'depth_backend': args.depth_backend,
                'backend_tag': backend_tag,
                'replay_dir': (os.path.abspath(args.replay_dir) if replay else None),
                'camera_resolution': args.camera_resolution,
                'camera_fps': args.camera_fps,
                'serial_number': args.serial_number,
                'target_fps': args.fps,
                'track_refine_iter': args.track_refine_iter,
                'est_refine_iter': args.est_refine_iter,
                'track_hz_2': args.track_hz_2,
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
            if args.sam2_tool or args.end_tag_id >= 0:
                meta['robust'] = dict(sam2_tool=bool(args.sam2_tool), end_tag_id=args.end_tag_id,
                                      end_tag_roll_deg=(tag.roll_deg if tag is not None else None), sam2_ckpt=args.sam2_ckpt, sam2_every=args.sam2_every,
                                      lost_iou=args.lost_iou, lost_px=args.lost_px, lost_frames=args.lost_frames,
                                      lost_action=args.lost_action, reseed_topic=args.reseed_topic,
                                      n_logged=len(robust_log))
                rows = [{k: (v if not isinstance(v, float) or np.isfinite(v) else None) for k, v in r.items()}
                        for r in list(robust_log)]
                with open(os.path.join(take_dir, 'robust_log.json'), 'w') as fh:
                    _json.dump(rows, fh)
            meta['stamps'] = 'ZED image capture time (sl.TIME_REFERENCE.IMAGE), ROS time'
            meta['payload'] = (None if payload_k is None else {
                'object': args.payload_object, 'pose_topic': args.payload_pose_topic,
                'status_topic': args.payload_status_topic,
                'config_file': (os.path.abspath(args.payload_config)
                                if args.payload_config else None),
                'check': payload_cfg,
                'n_valid': int(sum(1 for r in flog if r.get('payload_valid'))),
                'n_frames': len(flog)})
            with open(os.path.join(take_dir, 'metadata.json'), 'w') as fh:
                _json.dump(meta, fh, indent=2)
            with open(os.path.join(take_dir, 'frame_log.jsonl'), 'w') as fh:
                for r in flog:
                    fh.write(_json.dumps(r) + '\n')
            if vizs and K is not None and args.viz_data:
                _save_viz_data_2(take_dir, vizs, poses, K, args.viz_stride,
                                 [o.mesh_path for o in objs],
                                 compress=args.viz_compress)
            print(f'[autorecord] saved {n} frames -> {take_dir} ({reason}) '
                  f'[encode {time.time()-_t_enc:.1f}s]', flush=True)
        if take_dir and dlog and len(dlog) == n:
            _t_d = time.time()
            rd = os.path.join(take_dir, 'replay')
            for sub in ('rgb', 'depth', 'masks_1', 'masks_2'):
                os.makedirs(os.path.join(rd, sub), exist_ok=True)
            for i in range(n):
                cv2.imwrite(os.path.join(rd, 'rgb', f'{i:05d}.png'), cv2.cvtColor(raw[i], cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(rd, 'depth', f'{i:05d}.png'), dlog[i])
            np.savetxt(os.path.join(rd, 'cam_K.txt'), K)
            with open(os.path.join(rd, 'ts_ns.txt'), 'w') as fh:
                fh.write(''.join(f'{i:05d} {int(flog[i]["capture_ns"])}\n' for i in range(min(n, len(flog)))))
            import robust_tool as _rt
            for o in objs:
                T0 = poses[o.k][0] if poses[o.k] else None
                m = _rt.ToolCheck(o.mesh, K, args.width, args.height).silhouette(T0)
                cv2.imwrite(os.path.join(rd, f'masks_{o.k + 1}', '00000.png'), (m * 255).astype(np.uint8))
            print(f'[save_depth] wrote {rd} ({n} frames, {time.time() - _t_d:.1f}s)', flush=True)
        return take_dir

    def _snapshot():
        return (list(raw_frames), [list(pose_logs[0]), list(pose_logs[1])],
                list(frame_times), {k: list(v) for k, v in stage_times.items()},
                list(viz_log), _save_loop_start_time(), list(frame_log), list(depth_frames))

    def _save_take(reason, clear=True):
        """Save the take, blocking until it is on disk (as the one-object
        GUI script does)."""
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
        t_spawn = time.time()
        if replay:
            capture = ReplayCapture(args.replay_dir, (args.width, args.height),
                                    realtime_scale=args.replay_realtime_scale)
            capture.start()
            K = capture.K
            print(f'[capture] replaying {len(capture.rgb_files)} frames from '
                  f'{args.replay_dir}; K[0,0]={K[0, 0]:.1f}')
        else:
            # Producer process owns the camera; its own CUDA context keeps the
            # ZED's NEURAL depth off FoundationPose's stream.
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
            print(f'[capture] producer ready in {time.time()-t_spawn:.2f}s; '
                  f'K[0,0]={K[0,0]:.1f}, baseline={capture.baseline_m*1000:.2f}mm')
        baseline_m = capture.baseline_m
        last_seq = 0

        fast_fs = None
        if not replay and args.depth_backend == 'fast_fs':
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
            """Block until a fresh frame is ready; return (rgb, depth_m, ts_ns,
            grab_s, depth_s)."""
            nonlocal last_seq
            t0 = time.time()
            out = capture.latest(since_seq=last_seq, timeout=2.0)
            if out is None:
                if replay and capture.exhausted:
                    raise _ReplayDone()
                raise RuntimeError('capture process delivered no frame')
            seq, ts_ns, a, b = out
            last_seq = seq
            grab_s = time.time() - t0
            if replay or args.depth_backend == 'zed_neural':
                return a, b, ts_ns, grab_s, 0.0
            t1 = time.time()
            depth = fast_fs(a, b)
            return a, depth, ts_ns, grab_s, time.time() - t1

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor(
            trt_engine_path=(args.fp_refiner_trt or None))
        glctx = dr.RasterizeCudaContext()

        def _set_mesh(o, mesh_path):
            mesh = trimesh.load(mesh_path, process=False)
            if o.est is None:
                o.est = FoundationPose(
                    model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                    mesh=mesh, scorer=scorer, refiner=refiner, debug=0,
                    glctx=glctx,
                    debug_dir=os.path.join(args.save_dir, f'object_{o.k + 1}_debug'))
            else:
                o.est.reset_object(model_pts=mesh.vertices,
                                   model_normals=mesh.vertex_normals, mesh=mesh)
            o.mesh, o.mesh_path = mesh, mesh_path
            o.upright = (not args.no_upright
                         and Path(mesh_path).stem in set(args.upright_meshes))
            if o.k == payload_k:
                from payload_check import PayloadCheck
                from Utils import make_mesh_tensors
                o.check = PayloadCheck(make_mesh_tensors(mesh), K, args.width,
                                       args.height, glctx, **payload_cfg)
                o.status = dict(valid=False, reason='not_registered')

        def _observe(o, rgb, depth, ts_ns, registered=False):
            """o.pose now comes from the frame captured at ts_ns: record the
            stamp and, for the payload, judge that frame."""
            o.ts_ns = ts_ns
            if o.check is None:
                return
            if registered:
                o.check.reset()
            o.status = o.check.check(rgb, depth, o.pose, ts_ns * 1e-9)

        def _reject(o, reason):
            """The pose was not refreshed from a frame (failed registration)."""
            if o.check is not None:
                o.status = dict(valid=False, reason=reason)

        for o in objs:
            _set_mesh(o, o.mesh_path)
            print(f'[object {o.k + 1}] {o.name}: {o.mesh_path} '
                  f'({len(o.mesh.vertices)} verts), mask {o.mask_mode(args)}, '
                  f'topic {o.topic}, upright at register: {o.upright}', flush=True)

        def _click_mask(o, frame, why=''):
            from point_prompt_mask import select_mask_interactive
            return select_mask_interactive(
                frame, title=f'object {o.k + 1}: {o.name}{why}',
                color=((0, 200, 0), (220, 0, 220))[o.k])

        def _get_mask(o, frame, force_click=False):
            mode = 'points' if force_click else o.mask_mode(args)
            if mode == 'points':
                return _click_mask(o, frame)
            if mode == 'center_box':
                m = _center_box_mask(frame, *args.center_box_frac)
                print(f'[mask {o.k + 1}] center_box: {int(m.sum())} px', flush=True)
                return m
            if mode == 'replay':
                p = Path(args.replay_dir) / f'masks_{o.k + 1}' / f'{capture.frame_name}.png'
                if not p.is_file():
                    print(f'[mask {o.k + 1}] no {p}', flush=True)
                    return None
                m = (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
                if m.shape != frame.shape[:2]:
                    m = cv2.resize(m, (frame.shape[1], frame.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                print(f'[mask {o.k + 1}] {p.name}: {int(m.sum())} px', flush=True)
                return m
            if mode == 'text':
                _t = time.time()
                m, score = _mask_from_text_joined(
                    frame, o.text_prompt, o.join_px, args.text_backend,
                    args.text_min_score)
                dt = time.time() - _t
                if m is None or m.sum() == 0:
                    print(f'[mask {o.k + 1}] text {o.text_prompt!r} matched '
                          f'nothing (best score {score:.2f} < '
                          f'{args.text_min_score}) in {dt:.2f}s', flush=True)
                    if args.text_fallback_click:
                        print(f'[mask {o.k + 1}] falling back to the popup '
                              '(drag a box or click the object)', flush=True)
                        return _click_mask(o, frame, ' (text found nothing)')
                    return None
                print(f'[mask {o.k + 1}] text {o.text_prompt!r} -> '
                      f'{int(m.sum())} px (score {score:.2f}) in {dt:.2f}s',
                      flush=True)
                return m
            print(f'Select ROI for object {o.k + 1} ({o.name}).', flush=True)
            return _select_mask_safe(frame)

        def _upright_rot_grid(n=24):
            """Rotations (camera frame) with the mesh's +z along up_cam, at n
            headings about it: FoundationPose's register() hypotheses for
            --force_upright."""
            z = up_cam / np.linalg.norm(up_cam)
            a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = np.cross(a, z); x /= np.linalg.norm(x)
            R0 = np.stack([x, np.cross(z, x), z], 1)
            out = np.tile(np.eye(4), (n, 1, 1))
            for i, th in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
                c, s_ = np.cos(th), np.sin(th)
                out[i, :3, :3] = R0 @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            return out

        def _snap_upright(o, lock_here=False):
            """Turn est.pose_last (the centred mesh) so the mesh's +z is
            exactly up_cam, keeping its heading and its centre; return the
            pose in the mesh's own frame, as register() / track_one() do.
            With --lock_position the centre stays at o.locked_t, which
            lock_here (register) sets to the current one."""
            shape = o.est.pose_last.shape       # (4, 4) after register, (1, 4, 4) after track_one
            T = o.est.pose_last.detach().cpu().numpy().astype(np.float64).reshape(4, 4)
            R = T[:3, :3]
            z = up_cam / np.linalg.norm(up_cam)
            x = R[:, 0] - (R[:, 0] @ z) * z
            if np.linalg.norm(x) < 0.3:          # x nearly vertical: use y
                y = R[:, 1] - (R[:, 1] @ z) * z
                x = np.cross(y / np.linalg.norm(y), z)
            x /= np.linalg.norm(x)
            T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
            if args.lock_position:
                if lock_here or getattr(o, 'locked_t', None) is None:
                    o.locked_t = T[:3, 3].copy()
                    print(f'[lock {o.k + 1}] position locked at '
                          f'{np.round(o.locked_t, 4).tolist()} m (camera frame)',
                          flush=True)
                T[:3, 3] = o.locked_t
            o.est.pose_last = torch.as_tensor(T, device=o.est.pose_last.device,
                                              dtype=o.est.pose_last.dtype).reshape(shape)
            return (o.est.pose_last @ o.est.get_tf_to_centered_mesh()
                    ).data.cpu().numpy().reshape(4, 4)

        def _forced(o):
            return args.force_upright and o.upright

        def _register(o, rgb, depth, mask):
            """est.register(); for an upright part keep the copy whose +z
            points up, turning FoundationPose's own state over so track_one()
            keeps it. With --force_upright, register from upright guesses
            only and snap +z exactly up."""
            grid = None
            if _forced(o):
                grid = o.est.rot_grid
                o.est.rot_grid = torch.as_tensor(_upright_rot_grid(), device=grid.device,
                                                 dtype=grid.dtype)
            prev = o.est.pose_last
            o.est.pose_last = None      # register() leaves it unset when it gives up
            try:
                pose = o.est.register(K=K, rgb=rgb, depth=depth,
                                      ob_mask=mask.astype(bool),
                                      iteration=args.est_refine_iter)
            finally:
                if grid is not None:
                    o.est.rot_grid = grid
            if o.est.pose_last is None:
                o.est.pose_last = prev
                # register() gave up (no valid depth under the mask): it
                # returned a placeholder and has no state to track from.
                raise RuntimeError(
                    f'object {o.k + 1} ({o.name}): no valid depth under the '
                    f'mask ({int(mask.sum())} px); FoundationPose could not '
                    'register it')
            if _forced(o):
                return _snap_upright(o, lock_here=True)
            if o.upright and pose[:3, 2] @ up_cam < 0:
                turn = torch.as_tensor(np.diag([1.0, -1.0, -1.0, 1.0]),
                                       device=o.est.pose_last.device,
                                       dtype=o.est.pose_last.dtype)
                o.est.pose_last = o.est.pose_last @ turn
                pose = (o.est.pose_last @ o.est.get_tf_to_centered_mesh()
                        ).data.cpu().numpy().reshape(4, 4)
                o.n_turned += 1
                print(f'[upright {o.k + 1}] registered upside down; turned '
                      'it over (+z now up)', flush=True)
            return pose

        rgb, depth, ts0, _, _ = fetch()
        for o in objs:
            mask = _get_mask(o, rgb)
            if mask is None or mask.sum() == 0:
                print(f'Empty ROI for object {o.k + 1} ({o.name}). Exiting.')
                return
            print(f'Registering object {o.k + 1} ({o.name})...')
            t0 = time.time()
            o.pose = _register(o, rgb, depth, mask)
            _observe(o, rgb, depth, ts0, registered=True)
            print(f'Object {o.k + 1} registered in {time.time()-t0:.3f}s')
            if o.k == 0:
                tool_mask0 = mask
        if ros_pub is not None:
            ros_pub.publish_objs(objs, T_RC)

        # ---- SAM2 checks / end-face tag on object 1 (robust copy) -------------
        sam = check = tag = None
        robust_state = dict(frame=0, last_fix=0.0, n_lost=0, n_fix=0, mask0_px=0, last_iou=float('nan'),
                            tag_calibrated=bool(args.end_tag_no_calib), n_tag=0, n_tag_fix=0)
        quality_pub = None
        reseed_msg = dict(T=None, t=0.0)
        if args.sam2_tool or args.end_tag_id >= 0:
            import robust_tool
            robust_tool.RICH_PROMPT = not args.sam2_simple_prompt
        if args.sam2_tool:
            _t = time.time()
            sam = robust_tool.SamTool(ckpt=args.sam2_ckpt or robust_tool.SAM2_CKPT,
                                      cfg=args.sam2_cfg or robust_tool.SAM2_CFG,
                                      image_size=args.sam2_image_size, gpu_prep=not args.sam2_cpu_prep)
            box, pt = sam.reset(rgb, tool_mask0)
            if args.sam2_async:
                sam = robust_tool.AsyncSam(sam)
            check = robust_tool.ToolCheck(objs[0].mesh, K, args.width, args.height, lost_iou=args.lost_iou,
                                          lost_px=args.lost_px, lost_frames=args.lost_frames,
                                          glctx=None if args.check_cpu_silhouette else glctx,
                                          metric=args.lost_metric, lost_cover=args.lost_cover)
            robust_state['mask0_px'] = int(np.asarray(tool_mask0).astype(bool).sum())
            print(f'[sam2] object 1 tracked by SAM2 ({time.time() - _t:.1f}s to load; box '
                  f'{np.round(box).astype(int).tolist()}, point {np.round(pt[0]).astype(int).tolist()}); lost = iou < '
                  f'{args.lost_iou} and centres > {args.lost_px:g} px for {args.lost_frames} frames; action '
                  f'{args.lost_action}', flush=True)
        if args.end_tag_id >= 0:
            tag = robust_tool.EndFaceTag(K, tag_id=args.end_tag_id, dict_name=args.end_tag_dict,
                                         size_m=args.end_tag_size_m, end_x=args.end_tag_end_x,
                                         roll_deg=args.end_tag_roll_deg, roi_px=args.end_tag_roi_px)
            print(f'[tag] end-face tag {args.end_tag_dict} id {args.end_tag_id}, {args.end_tag_size_m * 1e3:g} mm, '
                  f'roll {args.end_tag_roll_deg:g} deg ({"fixed" if args.end_tag_no_calib else "learned first"}); '
                  f'fix at > {args.end_tag_fix_deg:g} deg roll or > {args.end_tag_fix_m * 100:g} cm', flush=True)
            if args.width < 1600:
                print(f'[tag] WARNING: at {args.width} x {args.height} the tag is ~{K[0, 0] * args.end_tag_size_m / 0.8:.0f} '
                      'px at 0.8 m (too small); run with --width 1920 --height 1080', flush=True)
        if (sam is not None or tag is not None) and ros_pub is not None:
            import rospy as _rospy_q
            from std_msgs.msg import Float32MultiArray as _F32A
            quality_pub = _rospy_q.Publisher(f'robot_frame/{objs[0].topic}_quality', _F32A, queue_size=1)
            print(f'[robust] quality on /robot_frame/{objs[0].topic}_quality', flush=True)
            if args.reseed_topic:
                if T_RC is None:
                    sys.exit('--reseed_topic needs --calibration (robot -> camera)')
                from geometry_msgs.msg import PoseStamped as _PS
                from scipy.spatial.transform import Rotation as _Rot
                T_CR = np.linalg.inv(np.asarray(T_RC, np.float64))

                def _reseed_cb(m):
                    p, q = m.pose.position, m.pose.orientation
                    T = np.eye(4)
                    T[:3, :3] = _Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    T[:3, 3] = [p.x, p.y, p.z]
                    reseed_msg['T'], reseed_msg['t'] = T_CR @ T, time.time()
                _rospy_q.Subscriber(args.reseed_topic, _PS, _reseed_cb, queue_size=1)
                print(f'[robust] restart FoundationPose from {args.reseed_topic} when lost', flush=True)

        fpp_on = args.fpp_anchor_xy or args.fpp_mask_depth_px >= 0 or args.fpp_filter
        if fpp_on and sam is None:
            raise SystemExit('--fpp_anchor_xy / --fpp_mask_depth_px / --fpp_filter need --sam2_tool (SAM2 gives the mask)')
        if fpp_on:
            import fpp_tracking
            print(f'[fpp] object 1: anchor_xy={args.fpp_anchor_xy} ({args.fpp_anchor_mode}, max '
                  f'{args.fpp_max_shift_px:g} px), mask_depth_px={args.fpp_mask_depth_px}, filter={args.fpp_filter}; '
                  f'on SAM2 frames (every {max(args.sam2_every, 1)})', flush=True)
        pose_filter = (fpp_tracking.PoseFilter(args.fpp_filter_alpha, args.fpp_filter_max_v,
                                               np.radians(args.fpp_filter_max_w)) if args.fpp_filter else None)
        fpp_mask = dict(m=None, ms=0.0, pose=None)  # SAM2 mask made before track_one this frame; _robust_step reuses it
        robust_state['prev_pose'] = None             # --sam2_async: object 1's pose on the frame the mask comes from

        def _sam_mask(rgb):
            """(mask, ms, pose the mask should be compared with). Sync: this frame's mask, pose None (= current).
            Async: submit this frame, return the newest finished mask (normally the previous frame's) and that frame's
            pose; (None, 0, None) until the first result."""
            if not args.sam2_async:
                _t0 = time.time()
                return sam.track(rgb), (time.time() - _t0) * 1e3, None
            res = sam.latest()
            sam.submit(rgb, robust_state['frame'])
            if res is None:
                return None, 0.0, None
            return res[0], res[2], robust_state['prev_pose']

        def _fpp_pre(o, rgb, depth):
            """SAM2 on this frame, then (if its mask is usable) the FoundationPose++ aids on object 1's starting
            pose and depth. Returns (depth for track_one, log fields). Frames between SAM2 frames run plain."""
            if (robust_state['frame'] + 1) % max(args.sam2_every, 1) != 0:
                return depth, dict(fpp_px=0, fpp_used=False, fpp_shift_px=float('nan'))
            m, ms, pose_m = _sam_mask(rgb)
            if m is None:
                return depth, dict(fpp_px=0, fpp_used=False, fpp_shift_px=float('nan'))
            fpp_mask['m'], fpp_mask['ms'], fpp_mask['pose'] = m, ms, pose_m
            n = int(m.sum())
            row = dict(fpp_px=n, fpp_used=False, fpp_shift_px=float('nan'))
            if not (args.fpp_min_mask_px <= n <= args.lost_mask_max_ratio * max(robust_state['mask0_px'], 1)):
                return depth, row
            row['fpp_used'] = True
            if args.fpp_anchor_xy and args.fpp_anchor_mode == 'centre':
                o.est.pose_last, row['fpp_shift_px'] = fpp_tracking.anchor_xy(
                    o.est.pose_last, K, fpp_tracking.mask_centre(m), args.fpp_max_shift_px)
            elif args.fpp_anchor_xy:
                sil = check.silhouette(o.pose if pose_m is None else pose_m)
                dl = fpp_tracking.silhouette_delta(m, sil)
                if dl is not None:
                    cover = (m & sil).sum() / max(int(m.sum()), 1)
                    row['fpp_iou'], row['fpp_cover'] = dl[2], cover
                    if cover < args.fpp_anchor_gate_cover:
                        o.est.pose_last, row['fpp_shift_px'] = fpp_tracking.shift_pose_last_px(
                            o.est.pose_last, K, dl[0], dl[1], args.fpp_max_shift_px)
            if args.fpp_mask_depth_px >= 0:
                depth = fpp_tracking.masked_depth(depth, m, args.fpp_mask_depth_px)
            return depth, row

        def _try_anchor(o, rgb, depth, mask):
            """Lost recovery without re-registration: shift the starting pose onto SAM2's mask, refine on
            mask-blanked depth, accept if enough of the mask then lies inside the silhouette."""
            import fpp_tracking as _fpp
            keep_pl, keep_pose = o.est.pose_last.clone(), o.pose
            dl = _fpp.silhouette_delta(mask, check.silhouette(o.pose))
            if dl is None:
                return False
            o.est.pose_last, _ = _fpp.shift_pose_last_px(o.est.pose_last, K, dl[0], dl[1])
            T = o.est.track_one(rgb=rgb, depth=_fpp.masked_depth(depth, mask, 8), K=K,
                                iteration=args.lost_anchor_iter)
            sil = check.silhouette(T)
            cover = (mask & sil).sum() / max(int(mask.sum()), 1)
            if cover >= args.lost_anchor_accept:
                o.pose = T
                return True
            o.est.pose_last, o.pose = keep_pl, keep_pose
            return False

        def _set_state(o, T):
            """Restart FoundationPose's track_one from pose T (mesh frame, camera)."""
            if pose_filter is not None and o.k == 0:
                pose_filter.reset()
            shape = o.est.pose_last.shape
            pl = robust_tool.pose_last_from_pose(T, o.est.model_center)
            o.est.pose_last = torch.as_tensor(pl, device=o.est.pose_last.device,
                                              dtype=o.est.pose_last.dtype).reshape(shape)
            o.pose = T

        _flip_mats = None

        def _flip_check(o, rgb, depth):
            """Refine pose_last and its 180-degree twins (one track iteration each), score all with FoundationPose's
            scorer, switch to a twin if it wins by --flip_margin. Returns (switched axis or '', scores)."""
            nonlocal _flip_mats
            from Utils import erode_depth, bilateral_filter_depth, depth2xyzmap_batch
            from scipy.spatial.transform import Rotation as _Rf
            est = o.est
            P = est.pose_last.reshape(-1, 4, 4)[0]
            if _flip_mats is None:
                _flip_mats = []
                for ax in args.flip_axes:
                    F = torch.eye(4, device=P.device, dtype=P.dtype)
                    F[:3, :3] = torch.as_tensor(_Rf.from_euler(ax, 180, degrees=True).as_matrix(), device=P.device,
                                                dtype=P.dtype)
                    _flip_mats.append(F)
            # Twins about the axes through the mesh centre (pose_last is the centred mesh's pose).
            cand = torch.stack([P] + [P @ F for F in _flip_mats])
            d = torch.as_tensor(depth, device='cuda', dtype=torch.float)
            d = bilateral_filter_depth(erode_depth(d, radius=2, device='cuda'), radius=2, device='cuda')
            Kt = torch.as_tensor(K, dtype=torch.float, device='cuda')
            xyz = depth2xyzmap_batch(d[None], Kt[None], zfar=np.inf)[0]
            ref, _ = est.refiner.predict(mesh=est.mesh, mesh_tensors=est.mesh_tensors, rgb=rgb, depth=d, K=K,
                                         ob_in_cams=cand.data.cpu().numpy(), normal_map=None, xyz_map=xyz,
                                         mesh_diameter=est.diameter, glctx=est.glctx,
                                         iteration=args.flip_refine_iter, get_vis=False)
            if args.flip_judge == 'scorer':
                sc, _ = est.scorer.predict(mesh=est.mesh, rgb=rgb, depth=d, K=K, ob_in_cams=ref.data.cpu().numpy(),
                                           normal_map=None, mesh_tensors=est.mesh_tensors, glctx=est.glctx,
                                           mesh_diameter=est.diameter, get_vis=False)
                sc = torch.as_tensor(sc).float().reshape(-1).cpu().numpy()
                b = int(np.argmax(sc))
                if b == 0 or sc[b] - sc[0] <= args.flip_margin:
                    return '', sc
            else:
                from Utils import nvdiffrast_render
                col, dep, _ = nvdiffrast_render(K=K, H=d.shape[0], W=d.shape[1], ob_in_cams=ref.float(),
                                                glctx=est.glctx, mesh_tensors=est.mesh_tensors)
                obs = torch.as_tensor(rgb, device='cuda', dtype=torch.float) / 255.0
                errs, npx = [], []
                for i in range(len(ref)):
                    vis_i = (dep[i] > 0) & (d > 0.05) & (d >= dep[i] - args.fit_tol)   # drawn, not covered by the hand
                    n = int(vis_i.sum())
                    npx.append(n)
                    errs.append(float((col[i][vis_i].clamp(0, 1) - obs[vis_i]).abs().mean()) if n else float('inf'))
                sc = np.array(errs)
                b = int(np.argmin(sc))
                if b == 0 or npx[b] < args.flip_min_px or npx[0] < args.flip_min_px or sc[b] >= args.flip_ratio * sc[0]:
                    return '', sc
            est.pose_last = ref[b].reshape(est.pose_last.shape).to(est.pose_last.dtype)
            o.pose = (ref[b] @ est.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)
            return args.flip_axes[b - 1], sc

        def _fp_visible_mask(match):
            """Largest blob of FoundationPose's depth-matching pixels (closed), to re-seed SAM2."""
            m = cv2.morphologyEx(match.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            if n <= 1:
                return None
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            return lab == k if stats[k, cv2.CC_STAT_AREA] >= 200 else None

        def _arb_score(mask, T, depth, rgb=None):
            """Re-registration check. depth mode: cover (SAM2 mask inside the silhouette) x depth fit. colour mode:
            1 - colour error (0 when hidden)."""
            if T is None or not np.isfinite(T).all():
                return 0.0
            if args.arbitrate == 'color':
                ce, n, _ = check.color_err(rgb, depth, T, args.fit_tol)
                if np.isfinite(ce) and n >= args.color_min_px:
                    return 1.0 - ce
                # Too little of it visible to judge colour: fall back to the depth-mode score, scaled into
                # [0, 0.5) so any pose with a colour verdict ranks above it.
            sil = check.silhouette(T)
            cover = (mask & sil).sum() / max(int(mask.sum()), 1)
            fit, _, nrest = check.depth_fit(depth, T, args.fit_tol)
            sc = float(cover) * (float(fit) if np.isfinite(fit) else 0.0)
            return 0.5 * sc if args.arbitrate == 'color' else sc

        def _robust_step(o, rgb, depth):
            """The end-face tag (roll / end position fix), SAM2's mask, the check, and the fix when lost (object 1
            only). Returns this frame's log row."""
            robust_state['frame'] += 1
            row = dict(t=time.time(), sam_ms=0.0, action='')
            now = time.time()
            if (args.reseed_on_message and reseed_msg['T'] is not None
                    and reseed_msg['t'] > robust_state.get('reseed_t', 0.0)
                    and now - reseed_msg['t'] < args.reseed_max_age_s):
                robust_state['reseed_t'] = reseed_msg['t']
                how = 'seed'
                if args.reseed_message_action == 'register' and sam is not None and sam.last is not None:
                    how = 'wait'                 # registered less than --lost_cooldown_s ago: let it track
                    if now - robust_state.get('last_msg_reg', 0.0) >= args.lost_cooldown_s:
                        robust_state['last_msg_reg'] = now
                        m = np.asarray(sam.last).astype(bool)
                        how = 'seed'
                        if 0 < m.sum() <= args.lost_mask_max_ratio * max(robust_state['mask0_px'], 1):
                            try:
                                o.pose = _register(o, rgb, depth, m)
                                how = 'register'
                            except RuntimeError as exc:
                                print(f'\n[robust] register from SAM2 failed ({exc}); seeding', flush=True)
                if how == 'seed':
                    _set_state(o, reseed_msg['T'])
                if how != 'wait':
                    robust_state['n_fix'] += 1
                    row['action'] = 'reseed_message' if how == 'seed' else 'register_message'
                    if check is not None:
                        check.n_bad = 0
                    print(f'\n[robust] object 1 {"restarted from the selector" if how == "seed" else "re-registered on SAM2 (selector)"} '
                          f'({time.time() - now:.2f}s)', flush=True)
            tag_seen = False
            if tag is not None:
                det = tag.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), depth, o.pose)
                tag_seen = det is not None
                row['tag'] = tag_seen
                if tag_seen:
                    robust_state['n_tag'] += 1
                    Tf, rd, dp = tag.fuse(det['T'], o.pose)
                    row.update(tag_side_px=det['side_px'], tag_roll_diff=rd, tag_dpos=dp)
                    if Tf is not None and not robust_state['tag_calibrated']:
                        trusted = sam is None or (np.isfinite(robust_state['last_iou'])
                                                  and robust_state['last_iou'] >= 0.5)
                        if trusted and dp < 0.02:
                            new = tag.calibrate(det['T'], o.pose, n=args.end_tag_calib_frames)
                            if new is not None:
                                tag.set_roll(new)
                                robust_state['tag_calibrated'] = True
                                print(f'\n[tag] sticker roll learned: {new:+.1f} deg', flush=True)
                    elif Tf is not None and (abs(rd) > args.end_tag_fix_deg or dp > args.end_tag_fix_m):
                        _set_state(o, Tf)
                        row['action'] = f'tag_fix (roll {rd:+.0f} deg, end {dp * 100:.1f} cm)'
                        robust_state['n_tag_fix'] += 1
            if args.log_checks and o.pose is not None and robust_state['frame'] % max(args.color_every, 1) == 0:
                ck = check.checks(rgb, depth, o.pose, args.fit_tol)
                row.update(log_ce=ck['ce'], log_fit=ck['fit'], log_nvis=ck['n_vis'])
                if ck['vis'] is not None and ck['n_vis'] >= 400:
                    h = robust_tool.appearance_hist(rgb, ck['vis'])
                    if robust_state.get('app_ref') is None or robust_state.get('app_ref_n', 0) < 5:
                        # reference = mean of the first 5 checked frames after (re)start (pose assumed good there)
                        n0 = robust_state.get('app_ref_n', 0)
                        r0 = robust_state.get('app_ref')
                        robust_state['app_ref'] = h if r0 is None else (r0 * n0 + h) / (n0 + 1)
                        robust_state['app_ref_n'] = n0 + 1
                    row['log_app'] = robust_tool.appearance_dist(h, robust_state['app_ref'])
            c = dict(badness=0.0, lost=False, iou=float('nan'), dpx=float('nan'))
            if sam is not None:
                pose_m = None
                if fpp_mask['m'] is not None:          # already run before track_one (--fpp_*)
                    mask, row['sam_ms'], pose_m = fpp_mask['m'], fpp_mask['ms'], fpp_mask['pose']
                    fpp_mask['m'] = None
                elif robust_state['frame'] % max(args.sam2_every, 1) == 0:
                    mask, row['sam_ms'], pose_m = _sam_mask(rgb)
                else:
                    mask = sam.last
                c = check.check(mask, o.pose if pose_m is None else pose_m)
                robust_state['prev_pose'] = None if o.pose is None else np.array(o.pose)
                robust_state['last_iou'] = c['iou']
                row.update({k: c[k] for k in ('iou', 'dpx', 'badness', 'bad', 'lost', 'sam_px', 'fp_px', 'usable')})
                row['cover'] = c.get('cover', float('nan'))
                if args.arbitrate == 'color' and o.pose is not None and (
                        robust_state['frame'] % max(args.color_every, 1) == 0 or c['bad']):
                    ce, ce_n, ce_good = check.color_err(rgb, depth, o.pose, args.fit_tol)
                    row['ce'], row['ce_px'] = ce, ce_n
                    seen = np.isfinite(ce) and ce_n >= args.color_min_px
                    fp_good = seen and ce < args.color_good
                    robust_state['n_cbad'] = robust_state.get('n_cbad', 0) + 1 if (seen and ce > args.color_bad) else 0
                    robust_state['n_cmid'] = (robust_state.get('n_cmid', 0) + 1
                                              if (seen and args.color_good <= ce <= args.color_bad) else 0)
                    disagree = c.get('usable') and (c['bad'] or c['sam_px'] > args.bleed_ratio * max(c['fp_px'], 1))
                    if fp_good and disagree:
                        # SAM2 disagrees, but the pose's rendered colours match the image: trust the pose and
                        # re-seed SAM2 from the pose's matching pixels.
                        check.n_bad = 0
                        c['bad'] = c['lost'] = False
                        row['bad'] = row['lost'] = False
                        if now - robust_state.get('last_sam_reseed', 0.0) > args.sam_reseed_min_s:
                            vm = _fp_visible_mask(ce_good)
                            if vm is not None:
                                sam.reset(rgb, vm)
                                robust_state['last_sam_reseed'] = now
                                row['action'] = 'sam2_reseed_from_fp (colour ok)'
                    elif robust_state['n_cbad'] >= args.color_bad_checks and c.get('usable'):
                        c['lost'] = True
                        row['lost'] = True
                        row['why'] = 'colour'
                    elif robust_state['n_cmid'] >= args.color_mid_checks and c.get('usable'):
                        # Neither clearly right nor clearly wrong for a while: try a re-registration; the colour
                        # check below keeps it only if it is better.
                        robust_state['n_cmid'] = 0
                        c['lost'] = True
                        row['lost'] = True
                        row['why'] = 'colour uncertain'
                if args.arbitrate == 'depth' and c.get('usable') and (c['bad'] or c['sam_px'] > args.bleed_ratio * max(c['fp_px'], 1)):
                    fit, match, _ = check.depth_fit(depth, o.pose, args.fit_tol)
                    row['fit'] = fit
                    if (np.isfinite(fit) and fit >= args.fit_ok
                            and c['sam_px'] > args.bleed_ratio * max(c['fp_px'], 1)):
                        # SAM2's mask is larger than the whole tool at FoundationPose's pose, and that pose explains
                        # the depth: SAM2 bled onto the hand / arm. Not lost; re-seed SAM2 from the pose.
                        check.n_bad = 0
                        c['bad'] = c['lost'] = False
                        row['bad'] = row['lost'] = False
                        if now - robust_state.get('last_sam_reseed', 0.0) > args.sam_reseed_min_s:
                            vm = _fp_visible_mask(match)
                            if vm is not None:
                                sam.reset(rgb, vm)
                                robust_state['last_sam_reseed'] = now
                                robust_state['n_sam_reseed'] = robust_state.get('n_sam_reseed', 0) + 1
                                row['action'] = 'sam2_reseed_from_fp (bleed)'
                if c['lost']:
                    robust_state['n_lost'] += 1
                if (c['lost'] and args.lost_action == 'reregister'
                        and now - robust_state['last_fix'] > robust_state.get('cooldown', args.lost_cooldown_s)):
                    T = reseed_msg['T'] if now - reseed_msg['t'] < args.reseed_max_age_s else None
                    sam_ok = 0 < c['sam_px'] <= args.lost_mask_max_ratio * max(robust_state['mask0_px'], 1)
                    if T is not None and not (args.reseed_message_action == 'register' and sam_ok):
                        _set_state(o, T)
                        row['action'] = 'reseed_node'
                    elif sam_ok and args.lost_try_anchor and _try_anchor(o, rgb, depth, mask):
                        row['action'] = 'anchor_sam2'
                    elif sam_ok:
                        try:
                            old_pose, old_pl = o.pose, o.est.pose_last.clone()
                            s_old = _arb_score(mask, old_pose, depth, rgb) if args.arbitrate != 'off' else 0.0
                            newT = _register(o, rgb, depth, mask)
                            s_new = _arb_score(mask, newT, depth, rgb) if args.arbitrate != 'off' else 1.0
                            # colour mode: an old pose that is clearly wrong (colour error > --color_bad, i.e.
                            # score < 1 - color_bad) or unjudgeable needs no margin, only "no worse".
                            old_bad = args.arbitrate == 'color' and s_old < 1.0 - args.color_bad
                            if s_new >= s_old + (0.0 if old_bad else args.verify_margin) or (
                                    args.arbitrate == 'color' and s_new >= 1.0 - args.color_good):
                                o.pose = newT
                                row['action'] = 'reregister_sam2'
                                if pose_filter is not None:
                                    pose_filter.reset()
                            else:
                                # The re-registration explains SAM2's mask + the depth no better: keep tracking.
                                o.est.pose_last, o.pose = old_pl, old_pose
                                row['action'] = f'reregister_rejected ({s_new:.2f} vs {s_old:.2f})'
                                if args.arbitrate == 'color':
                                    ce, ce_n, ce_good = check.color_err(rgb, depth, old_pose, args.fit_tol)
                                    ok_old = np.isfinite(ce) and ce_n >= args.color_min_px and ce < args.color_good
                                    vm = _fp_visible_mask(ce_good) if ok_old else None
                                else:
                                    fit, match, _ = check.depth_fit(depth, old_pose, args.fit_tol)
                                    vm = _fp_visible_mask(match) if np.isfinite(fit) and fit >= args.fit_ok else None
                                if vm is not None:
                                    sam.reset(rgb, vm)
                                    robust_state['last_sam_reseed'] = now
                                    row['action'] += '; sam2_reseed_from_fp'
                        except RuntimeError as exc:
                            row['action'] = f'reregister_failed: {exc}'
                            if T is not None:
                                _set_state(o, T)
                                row['action'] += '; reseed_node'
                    else:
                        row['action'] = 'skip: SAM2 mask too large'
                    # Back off when fixes keep failing (lost again soon after the last fix): each re-registration
                    # blocks the loop ~0.6 s, so a SAM2 mask on the wrong thing could otherwise stall tracking.
                    if now - robust_state['last_fix'] < 3.0 * robust_state.get('cooldown', args.lost_cooldown_s):
                        robust_state['cooldown'] = min(2.0 * robust_state.get('cooldown', args.lost_cooldown_s),
                                                       args.lost_max_cooldown_s)
                    else:
                        robust_state['cooldown'] = args.lost_cooldown_s
                    robust_state['last_fix'] = now
                    robust_state['n_fix'] += 1
                    check.n_bad = 0
                    robust_state['n_cbad'] = 0
                    print(f'\n[sam2] object 1 LOST (iou {c["iou"]:.2f}, centres {c["dpx"]:.0f} px): {row["action"]} '
                          f'({time.time() - now:.2f}s)', flush=True)
            if quality_pub is not None:
                _iou = c['iou'] if np.isfinite(c['iou']) else -1.0
                _dpx = c['dpx'] if np.isfinite(c['dpx']) else -1.0
                quality_pub.publish(_F32A(data=[float(c['badness']), float(c['lost']), float(tag_seen),
                                                float(_iou), float(_dpx)]))
            return row

        _lines = ['d: discard buffer',
                  'r: re-register both (drops buffer)',
                  '1 / 2: re-register object 1 / 2',
                  'q / ESC: quit']
        if auto_record:
            _lines.insert(0, 'AUTO: rec gated on policy')
        control_img = np.zeros((30 * len(_lines) + 40, 420, 3), dtype=np.uint8)
        for i, line in enumerate(_lines):
            cv2.putText(control_img, line, (10, 35 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # cv2's Qt plugin aborts without an X display, and the viser panel
        # replaces this window (which costs a cv2.imshow + waitKey every
        # iteration). Only open it when the panel is off or asked for.
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
            cv2.resizeWindow('Keyboard Control', 420, 30 * len(_lines) + 40)
            cv2.imshow('Keyboard Control', control_img)
            cv2.waitKey(1)

        _show_control_window()

        def _gui_objects():
            return [{'name': o.name, 'text_prompt': o.text_prompt or '',
                     'mesh_path': o.mesh_path, 'mask_mode': o.mask_mode(args),
                     'topic': o.topic, 'upright': o.upright} for o in objs]

        # ---- viser control panel ------------------------------------------
        if not args.no_gui:
            from live_tracking_gui_panel_2_objects import GuiState2, start_gui_2
            from live_tracking_replay import _free_port, first_free_port
            port = first_free_port(args.gui_port) if args.gui_port else _free_port()
            gui_state = GuiState2(_gui_objects(), auto_record=auto_record)
            choices = list(dict.fromkeys([o.mesh_path for o in objs]
                                         + list(args.mesh_choices or [])))
            _gui_server, gui_stop = start_gui_2(gui_state, port, choices)
            gui_state.publish(last_event='tracking both objects')
            from live_tracking_live_view import LiveView3D
            _lv = [v for v in (args.live_view or ['3d', '2d']) if v != 'off']
            _lv_3d, _lv_2d = '3d' in _lv, '2d' in _lv
            live_view = LiveView3D(_gui_server, K, T_up, (args.width, args.height),
                                   view_3d=_lv_3d, view_2d=_lv_2d, hz=args.live_view_hz,
                                   stride=args.live_view_stride)
            import socket as _socket
            print(f'[gui] control panel: http://{_socket.getfqdn()}:{port}  '
                  f'(or http://localhost:{port} with an ssh -L tunnel)',
                  flush=True)

            if args.selftest_swap:
                def _selftest():
                    alt = (args.mesh_choices or [None])[0]
                    if args.selftest_saves:
                        steps = [(t, 'save', True, None) for t in (12.0, 24.0, 36.0, 48.0)]
                    else:
                        steps = [(4.0, 'reregister', True, 1),
                                 (8.0, 'reregister', True, 0)]
                        if objs[1].text_prompt:
                            steps += [(12.0, 'prompt', 'robot arm', 1),
                                      (16.0, 'prompt', objs[1].text_prompt, 1)]
                        if alt:
                            steps.append((20.0, 'mesh', alt, 1))
                        steps += [(24.0, 'reregister', True, 0),
                                  (24.0, 'reregister', True, 1)]
                        if args.mask_mode == 'points':      # opens the popup
                            steps.append((26.0, 'reregister_click', True, 1))
                        steps.append((28.0, 'save', True, None))
                    t0 = time.time()
                    for when, key, val, obj in steps:
                        time.sleep(max(0.0, when - (time.time() - t0)))
                        print(f'\n[selftest] t+{time.time()-t0:4.1f}s request '
                              f'{key}={val!r} object={obj}', flush=True)
                        if obj is None:
                            gui_state.request(key, val)
                        else:
                            gui_state.request_object(key, obj, val)
                _threading.Thread(target=_selftest, name='selftest',
                                  daemon=True).start()

        def _do_reregister(reason, which=(0, 1), new_mesh=(None, None),
                           new_prompt=(None, None), click=(False, False)):
            """Drop the buffer, optionally swap an object's prompt or mesh,
            and re-register the objects in `which` on one fresh frame. An
            object whose mask misses keeps its previous pose."""
            nonlocal recording, loop_start_time
            n_dropped = len(raw_frames)
            _clear_buffers()
            recording = not auto_record
            for k in which:
                o = objs[k]
                if new_prompt[k] and new_prompt[k] != o.text_prompt:
                    o.text_prompt = new_prompt[k]
                    print(f'[swap {k + 1}] text prompt -> {o.text_prompt!r}',
                          flush=True)
                if new_mesh[k] and new_mesh[k] != o.mesh_path:
                    _t0 = time.time()
                    try:
                        _set_mesh(o, new_mesh[k])
                    except Exception as exc:
                        print(f'[swap {k + 1}] could not load {new_mesh[k]}: '
                              f'{exc!r}', flush=True)
                        continue
                    print(f'[swap {k + 1}] mesh -> {o.mesh_path} '
                          f'({len(o.mesh.vertices)} verts) in '
                          f'{time.time()-_t0:.2f}s; upright at register: '
                          f'{o.upright}', flush=True)
            print(f'[live_tracking] re-registering object(s) '
                  f'{[k + 1 for k in which]} ({reason}); dropped {n_dropped} '
                  'buffered frames', flush=True)
            _t_rr = time.time()
            rgb_r, depth_r, ts_r, _, _ = fetch()
            for k in which:
                o = objs[k]
                _t_mask = time.time()
                mask_r = _get_mask(o, rgb_r, force_click=click[k])
                print(f'[live_tracking]   object {k + 1} mask in '
                      f'{time.time()-_t_mask:.2f}s', flush=True)
                if mask_r is None or mask_r.sum() == 0:
                    print(f'[live_tracking]   object {k + 1}: no mask; keeping '
                          'its previous pose', flush=True)
                    _reject(o, 'register_no_mask')
                    continue
                _t_reg = time.time()
                try:
                    o.pose = _register(o, rgb_r, depth_r, mask_r)
                except RuntimeError as exc:
                    print(f'[live_tracking]   {exc}; keeping its previous pose',
                          flush=True)
                    _reject(o, 'register_failed')
                    continue
                _observe(o, rgb_r, depth_r, ts_r, registered=True)
                print(f'[live_tracking]   object {k + 1} re-registered in '
                      f'{time.time()-_t_reg:.2f}s', flush=True)
                if k == 0 and pose_filter is not None:
                    pose_filter.reset()
                if k == 0 and sam is not None:
                    sam.reset(rgb_r, mask_r)
                    robust_state['mask0_px'] = int(np.asarray(mask_r).astype(bool).sum())
                    check.n_bad = 0
                    print('[sam2] restarted on the new object-1 mask', flush=True)
            print(f'[live_tracking]   total {time.time()-_t_rr:.2f}s', flush=True)
            _clear_buffers()
            if ros_pub is not None:
                ros_pub.publish_objs(objs, T_RC)
            _show_control_window()
            loop_start_time = time.time()
            return True

        # ---- policy-gated recording ---------------------------------------
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

        # FP's Utils.set_logging_format() installs the root logger at INFO;
        # bump to WARNING so the live [fps] line stays readable.
        import logging as _logging
        _logging.getLogger().setLevel(_logging.WARNING)

        work_history = _deque(
            maxlen=max(30, int(args.fps) if args.fps > 0 else 30))

        def keep_running():
            # rospy's SIGINT handler flips is_shutdown() instead of raising
            # KeyboardInterrupt; treat that as an interrupt so the save prompt
            # still triggers.
            nonlocal interrupted
            if ros_pub is not None and ros_pub.shutdown_requested():
                interrupted = True
                return False
            if args.duration > 0 and (time.time() - loop_start_time) >= args.duration:
                return False
            return True

        last_track_2 = 0.0
        if args.track_hz_2 > 0:
            print(f'[track] object 2 tracked at {args.track_hz_2:g} Hz; its '
                  f'last pose is published again between updates', flush=True)
        try:
            while keep_running():
                loop_start = time.time()

                try:
                    rgb, depth, ts_ns, t_grab, t_depth = fetch()
                except _ReplayDone:
                    print('\n[replay] last frame reached', flush=True)
                    break

                t_track = [0.0, 0.0]
                for o in objs:
                    t0 = time.time()
                    if o.k == 1 and args.track_hz_2 > 0:
                        if t0 - last_track_2 < 1.0 / args.track_hz_2:
                            continue        # keep (and publish again) its last pose
                        last_track_2 = t0
                    _depth_o, _fpp_row = depth, None
                    if pose_filter is not None and o.k == 0:
                        # constant-velocity start pose, every frame (19 Nov; was only on usable SAM2 frames)
                        _P = pose_filter.predict()
                        if _P is not None:
                            o.est.pose_last = torch.as_tensor(_P, device=o.est.pose_last.device,
                                                              dtype=o.est.pose_last.dtype).reshape(o.est.pose_last.shape)
                    if fpp_on and o.k == 0:
                        _depth_o, _fpp_row = _fpp_pre(o, rgb, depth)
                    o.pose = o.est.track_one(rgb=rgb, depth=_depth_o, K=K,
                                             iteration=args.track_refine_iter)
                    if pose_filter is not None and o.k == 0:
                        pose_filter.update(o.est.pose_last.reshape(-1, 4, 4)[0].detach().cpu().numpy())
                    if _forced(o):
                        o.pose = _snap_upright(o)
                    _flip_row = None
                    if o.k == 0 and args.flip_check_every > 0:
                        robust_state['flip_n'] = robust_state.get('flip_n', 0) + 1
                        if robust_state['flip_n'] % args.flip_check_every == 0:
                            _ax, _sc = _flip_check(o, rgb, depth)
                            _flip_row = dict(flip=_ax, flip_scores=[round(float(x), 3) for x in _sc])
                            if _ax:
                                if pose_filter is not None:
                                    pose_filter.reset()
                                robust_state['n_flip'] = robust_state.get('n_flip', 0) + 1
                                print(f'\n[flip] object 1 turned 180 deg about mesh {_ax} (scores {np.round(_sc, 2)})',
                                      flush=True)
                    _observe(o, rgb, depth, ts_ns)
                    if o.k == 0 and (sam is not None or tag is not None):
                        _rrow = _robust_step(o, rgb, depth)
                        if _flip_row:
                            _rrow.update(_flip_row)
                        if _fpp_row is not None:
                            _rrow.update(_fpp_row)
                        if recording:
                            robust_log.append(_rrow)
                    t_track[o.k] = time.time() - t0

                t_publish = 0.0
                if ros_pub is not None:
                    t0 = time.time()
                    ros_pub.publish_objs(objs, T_RC)
                    t_publish = time.time() - t0

                if live_view is not None:
                    live_view.offer(rgb, depth, [o.pose for o in objs],
                                    [o.mesh for o in objs])

                # Buffer raw RGB + both poses; the overlay is drawn at save.
                # Gated on `recording`, as in the one-object GUI script.
                if recording:
                    raw_frames.append(rgb)
                    if args.save_depth:
                        depth_frames.append(np.clip(np.nan_to_num(depth) * 1000, 0, 65535).astype(np.uint16))
                    for o in objs:
                        pose_logs[o.k].append(o.pose)
                    _row = {'capture_ns': int(ts_ns),
                            'object_capture_ns': [None if o.ts_ns is None else int(o.ts_ns)
                                                  for o in objs]}
                    if payload_k is not None:
                        _st = objs[payload_k].status or {}
                        _row['payload_valid'] = bool(_st.get('valid', False))
                        _row['payload_reason'] = _st.get('reason', '')
                        for _key in ('color_px', 'iou', 'depth_px', 'surface_frac', 'jump_m'):
                            if _key in _st and not (isinstance(_st[_key], float)
                                                    and not math.isfinite(_st[_key])):
                                _row[_key] = _st[_key]
                    frame_log.append(_row)
                    if _want_viz:
                        s = args.viz_stride
                        viz_log.append((rgb[::s, ::s].copy(),
                                        depth[::s, ::s].copy()))

                work_elapsed = time.time() - loop_start
                sleep_time = target_frame_time - work_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if headless:
                    key = 255  # no HighGUI, so no keypress
                else:
                    cv2.imshow('Keyboard Control', control_img)
                    key = cv2.waitKey(1) & 0xFF

                iter_elapsed = time.time() - loop_start
                if recording:
                    frame_times.append(iter_elapsed)
                    stage_times['grab'].append(t_grab)
                    stage_times['depth'].append(t_depth)
                    stage_times['track_1'].append(t_track[0])
                    stage_times['track_2'].append(t_track[1])
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
                      f'(cap={args.fps:.0f})  n={buffer_count}  '
                      f'track ms {t_track[0]*1e3:4.1f}+{t_track[1]*1e3:4.1f}  ',
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
                    elif recording and not now_active and auto_record:
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
                                          last_event='saved take')
                    which = tuple(k for k in (0, 1)
                                  if _rq['reregister'][k] or _rq['prompt'][k]
                                  or _rq['mesh'][k] or _rq['reregister_click'][k])
                    if which:
                        gui_state.publish(last_event='re-registering '
                                          f'{[k + 1 for k in which]}...')
                        _do_reregister('gui', which=which,
                                       new_mesh=_rq['mesh'],
                                       new_prompt=_rq['prompt'],
                                       click=_rq['reregister_click'])
                        for o in objs:
                            gui_state.publish_object(o.k, **_gui_objects()[o.k])
                        gui_state.publish(last_event='re-registered '
                                          f'{[k + 1 for k in which]}')
                    if _rq['quit']:
                        print('\n[gui] stop requested', flush=True)
                        break
                    gui_state.publish(fps=fps_avg, recording=recording,
                                      n_buffered=len(raw_frames))

                if key == ord('d'):
                    n = len(raw_frames)
                    _clear_buffers()
                    loop_start_time = time.time()
                    print(f'[live_tracking] discarded {n} frames')
                elif key == ord('r'):
                    _do_reregister('key r')
                elif key in (ord('1'), ord('2')):
                    _do_reregister(f'key {chr(key)}', which=(int(chr(key)) - 1,))
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
        if watcher is not None:
            watcher.close()
        if live_view is not None:
            live_view.stop()
        if gui_stop is not None:
            gui_stop()
        if os.environ.get('DISPLAY'):
            cv2.destroyAllWindows()

        # On Ctrl+C, ask before the (slow) encode + save; natural exits save
        # without asking, and so do non-tty runs.
        do_save = bool(raw_frames)
        if interrupted and raw_frames and sys.stdin.isatty():
            import signal as _signal
            try:
                _signal.signal(_signal.SIGINT, _signal.default_int_handler)
            except Exception:
                pass
            try:
                ans = input(f'Save {len(raw_frames)} two-object frames to '
                            f'{args.save_dir}? [Y/n] ').strip().lower()
                do_save = ans in ('', 'y', 'yes')
            except (KeyboardInterrupt, EOFError):
                print('\n[live_tracking] aborting without save.')
                do_save = False

        run_dir = None
        if do_save:
            # clear=False: the viser replay below still needs viz_log.
            run_dir = _save_take('interrupted' if interrupted else 'exit',
                                 clear=False)
            if args.visualize and viz_log and K is not None and run_dir:
                from live_tracking_viz_replay_2_objects import launch_viser_replay_2
                print(f'[viser] preparing two-object replay ({len(viz_log)} '
                      f'frames at stride={args.viz_stride}) ...')
                pc_log = [unproject_strided(r, d, K, stride=args.viz_stride)
                          for r, d in viz_log]
                launch_viser_replay_2(pc_log, [list(pose_logs[0]), list(pose_logs[1])],
                                      K, mesh_paths=[o.mesh_path for o in objs],
                                      T_RC=T_RC_DEFAULT)


if __name__ == '__main__':
    main()
