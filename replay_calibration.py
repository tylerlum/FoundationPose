"""Replay explicitly supplied calibration arm poses and capture new tag observations.

Default is offline dry-run (no ROS, camera or publishers). Motion additionally
requires --execute and typing REPLAY after checking the current start path.
The sampled URDF/table clearance check is approximate: it does not check
self-collisions, cables, the bowl, or objects/people in the real workspace.
Moving slowly does not make an unchecked trajectory safe.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

import calibrate_camera as cal


def load_poses(path, urdf):
    with np.load(path, allow_pickle=False) as data:
        q = np.asarray(data['q'], dtype=float).copy()
        tag_size = float(data['tag_size'])
        tag_id = int(data['tag_id'])
        ee_link = str(data['ee_link'])
        if 'joint_names' in data:
            names = list(data['joint_names'].astype(str))
            if len(names) != 7 or set(names) != set(cal.ARM_JOINTS):
                raise ValueError(f'Unsupported joint names: {names}')
            q = q[:, [names.index(name) for name in cal.ARM_JOINTS]]
    lower, upper = urdf.arm_limits()
    if q.ndim != 2 or q.shape[1] != 7 or len(q) < 3 or not np.isfinite(q).all():
        raise ValueError('NPZ q must contain at least three finite 7-joint poses in radians')
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError('Recorded poses exceed URDF arm joint limits')
    if not np.isfinite(tag_size) or tag_size <= 0 or ee_link not in urdf.fk({}):
        raise ValueError('Invalid saved tag size or end-effector link')
    return q, tag_size, tag_id, ee_link


def duration(a, b, requested, max_speed_deg):
    # Cosine interpolation has peak speed pi/2 times its average speed.
    return max(requested, math.pi / 2 * np.max(np.abs(np.degrees(b - a))) / max_speed_deg)


def check_paths(poses, checker, min_clear, start=None):
    sequence = poses if start is None else np.vstack([start, poses])
    lowest = float('inf')
    for i, q in enumerate(sequence):
        clearance = checker.clearance(q)
        if i:
            # Sample at least 32 points, and at most one degree between samples.
            steps = max(32, int(np.ceil(np.max(np.abs(np.degrees(q - sequence[i-1]))))) + 1)
            clearance = min(clearance, checker.path_clearance(sequence[i-1], q, steps=steps))
        lowest = min(lowest, clearance)
        if clearance < min_clear:
            raise ValueError(f'Path to pose {i} has only {100*clearance:.1f} cm table clearance; '
                             f'required {100*min_clear:.1f} cm. No commands sent.')
    return lowest


class JointMonitor:
    def __init__(self, rospy):
        from sensor_msgs.msg import JointState
        self.arm = self.hand = None
        self.arm_time = self.hand_time = 0.
        rospy.Subscriber('/iiwa/joint_states', JointState, self._arm, queue_size=1)
        rospy.Subscriber('/sharpa/joint_states', JointState, self._hand, queue_size=1)

    def _arm(self, msg):
        self.arm, self.arm_time = msg, time.monotonic()

    def _hand(self, msg):
        self.hand, self.hand_time = msg, time.monotonic()

    def current(self):
        if (self.arm is None or self.hand is None or
                time.monotonic() - min(self.arm_time, self.hand_time) > .25):
            raise RuntimeError('Arm/hand joint states missing or older than 250 ms; stopping publication')
        msg = self.arm
        names = [name.replace('iiwa14_', 'iiwa_') for name in msg.name]
        expected = [f'iiwa_joint_{i}' for i in range(1, 8)]
        if len(names) != 7 or set(names) != set(expected) or len(msg.position) != 7:
            raise RuntimeError(f'Unexpected arm joint-state layout: {list(msg.name)}')
        q = np.array([msg.position[names.index(name)] for name in expected])
        hand = np.asarray(self.hand.position)
        if len(hand) != 22 or len(self.hand.name) != 22 or len(set(self.hand.name)) != 22:
            raise RuntimeError('Expected 22 named hand joints')
        if not np.isfinite(q).all() or not np.isfinite(hand).all():
            raise RuntimeError('Nonfinite joint states')
        return q


def run(args):
    urdf = cal.Urdf(args.urdf)
    poses, args.tag_size, args.tag_id, args.ee_link = load_poses(args.poses, urdf)
    args.no_depth_refine = True
    checker = cal.TableCheck(urdf, args.table_z, args.table_box)
    rospy = monitor = None
    if args.execute:
        import rospy
        rospy.init_node('replay_camera_calibration', anonymous=True, disable_signals=True)
        monitor = JointMonitor(rospy)
        deadline = time.monotonic() + 10
        while True:
            try:
                start = monitor.current()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.1)
        checker = cal.TableCheck(urdf, args.table_z, args.table_box,
                                 cal.joint_map(urdf, None, monitor.hand))
    else:
        start = None if args.start_q is None else np.asarray(args.start_q)
    lower, upper = urdf.arm_limits()
    if start is not None and (not np.isfinite(start).all() or np.any(start < lower) or np.any(start > upper)):
        raise ValueError('Start pose invalid or outside joint limits')
    clearance = check_paths(poses, checker, args.min_clear, start)
    durations = [duration(a, b, args.move_s, args.max_joint_speed_deg) for a, b in zip(poses[:-1], poses[1:])]
    first = args.first_move_s if start is None else duration(start, poses[0], args.first_move_s, args.max_joint_speed_deg)
    print(f'Source: {args.poses.resolve()}\n{len(poses)} recorded poses; tag {args.tag_id}, '
          f'{args.tag_size*1000:.1f} mm; depth refinement OFF', flush=True)
    print(f'First move >= {first:.1f}s; subsequent moves {min(durations):.1f}–{max(durations):.1f}s; '
          f'settle/capture {args.settle_s + args.hold_s:.1f}s per pose.\n'
          f'Estimated total {first+sum(durations)+len(poses)*(args.settle_s+args.hold_s):.1f}s; '
          f'sampled table clearance >= {clearance*100:.1f}cm.', flush=True)
    if not args.execute:
        print('DRY RUN: no ROS connection, camera access or command publishers. '
              + ('Current-to-first path NOT checked; supply --start-q or use live execution preflight.'
                 if start is None else 'Start path checked against supplied --start-q.'))
        return
    # Open the camera before arming any publishers. Models, detection and output are ready first.
    cap = cal.open_camera(args)
    preview = None
    arm_pub = hand_pub = None
    Q, B, source_indices, corners_saved = [], [], [], []
    skipped_indices = []
    out_dir = Path(args.out_dir) / f'replay_{time.strftime("%Y%m%d_%H%M%S")}'
    try:
        detector = cal.make_detector()
        if args.view:
            preview = cal.RecordPreview(args.view_port)
        print('The robot must be in POSITION CONTROL, with a clear workspace and unchanged hand/tag attachment.\n'
              'Sampled table checks are not a collision guarantee. Ctrl-C ends publication; no automatic return. '
              'Keep the hardware stop available.', flush=True)
        if input(f'Type REPLAY to move through {len(poses)} saved poses: ') != 'REPLAY':
            print('Cancelled; no commands sent.')
            return
        current = monitor.current()
        if np.max(np.abs(current-start)) > np.radians(.5):
            raise RuntimeError('Arm moved since preflight; rerun to check the new start path')
        from sensor_msgs.msg import JointState
        arm_pub = rospy.Publisher('/iiwa/joint_cmd', JointState, queue_size=1)
        hand_pub = rospy.Publisher('/sharpa/joint_cmd', JointState, queue_size=1)
        hand_names, hand_q = list(monitor.hand.name), list(monitor.hand.position)
        deadline = time.monotonic() + 5
        while not (arm_pub.get_num_connections() and hand_pub.get_num_connections()):
            monitor.current()
            if time.monotonic() > deadline:
                raise RuntimeError('No arm/hand command subscribers; check position-control drivers')
            time.sleep(.05)

        def send(q, guard=True):
            actual = monitor.current()
            if rospy.is_shutdown():
                raise RuntimeError('ROS shutdown')
            if guard and np.max(np.abs(np.degrees(q-actual))) > args.max_tracking_error_deg:
                raise RuntimeError('Arm is not following the trajectory; stopping replay')
            stamp = rospy.Time.now()
            arm, hand = JointState(), JointState()
            arm.header.stamp = hand.header.stamp = stamp
            arm.name = [f'iiwa_joint_{i}' for i in range(1,8)]
            arm.position = np.asarray(q).tolist()
            hand.name, hand.position = hand_names, hand_q
            arm_pub.publish(arm)
            hand_pub.publish(hand)

        out_dir.mkdir(parents=True, exist_ok=False)
        np.save(out_dir / 'requested_q.npy', poses)
        seq = 0
        current = start.copy()
        for i, target in enumerate(poses):
            seconds = duration(current, target, args.first_move_s if i == 0 else args.move_s,
                               args.max_joint_speed_deg)
            print(f'[replay] moving to {i+1}/{len(poses)} over {seconds:.1f}s', flush=True)
            t0 = time.monotonic()
            while True:
                phase = min((time.monotonic()-t0)/seconds, 1.)
                send(current + .5*(1-math.cos(math.pi*phase))*(target-current))
                if phase >= 1:
                    break
                time.sleep(1/60)
            current = target.copy()
            # Keep 60 Hz hold commands independent of camera capture latency.
            import threading
            stop_hold = threading.Event()
            errors = []
            def hold():
                try:
                    while not stop_hold.wait(1/60):
                        send(current)
                except Exception as exc:
                    errors.append(exc)
            thread = threading.Thread(target=hold, daemon=True)
            thread.start()
            try:
                Tm, qm, n, spread, last, seq = cal.record_still(
                    cap, monitor, args, cap.K, detector, seq)
            finally:
                stop_hold.set()
                thread.join()
            if errors:
                raise errors[0]
            # Visibility affects calibration data, not whether a checked joint path can run.
            # Keep motion/freshness checks active even when no tag pose was measured.
            if np.max(np.abs(np.degrees(monitor.current()-target))) > args.capture_tolerance_deg:
                raise RuntimeError(f'Pose {i+1} not reached within capture tolerance')
            if Tm is None:
                skipped_indices.append(i)
                print(f'[replay] SKIP pose {i+1}: tag seen in only {n} frames; continuing to next pose', flush=True)
                if preview is not None:
                    preview.status.content = f'**Pose {i+1}: tag not detected reliably — sample skipped.** Samples: {len(Q)}'
                continue
            if np.max(np.abs(np.degrees(qm-target))) > args.capture_tolerance_deg:
                raise RuntimeError(f'Pose {i+1} not reached within capture tolerance')
            Q.append(qm); B.append(Tm); source_indices.append(i); corners_saved.append(last[1])
            img = cv2.cvtColor(last[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir/f'sample_{len(Q):02d}_raw.png'), img)
            cv2.polylines(img, [last[1].astype(np.int32)], True, (0,255,0), 2)
            cv2.imwrite(str(out_dir/f'sample_{len(Q):02d}.jpg'), img)
            print(f'[replay] captured {len(Q)}: {n} frames, spread {spread:.1f} mm', flush=True)
            if preview is not None:
                preview.update(last[0], last[1], True, True, True, True, len(Q), n)
    except KeyboardInterrupt:
        print('\nReplay interrupted; saving completed samples.', flush=True)
    finally:
        # Best-effort hold at measured position; never issue a return-home move.
        if arm_pub is not None and arm_pub.get_num_connections():
            try:
                send(monitor.current(), guard=False)
            except Exception as exc:
                print(f'Cannot publish final measured-position hold: {exc}', flush=True)
        cap.stop_capture()
        if preview is not None:
            preview.close()
        if out_dir.exists():
            np.savez(out_dir/'samples.npz', q=np.asarray(Q).reshape(-1,7),
                     T_cam_tag=np.asarray(B).reshape(-1,4,4), K=cap.K,
                     tag_size=args.tag_size, tag_id=args.tag_id, ee_link=args.ee_link, urdf=str(args.urdf),
                     source_npz=str(args.poses.resolve()), source_indices=source_indices,
                     skipped_source_indices=np.asarray(skipped_indices, dtype=int),
                     corners=np.asarray(corners_saved).reshape(-1,4,2),
                     **cal.calibration_metadata(args,Q))
            print(f'Saved {len(Q)} samples: {out_dir / "samples.npz"}', flush=True)
    if len(Q) >= 3:
        args.file = str(out_dir/'samples.npz')
        args.calibration = args.check = None
        cal.run_solve(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poses', required=True, type=Path, help='Explicit source calibration samples.npz; never overwritten')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Enable live preflight and confirmation before motion')
    mode.add_argument('--dry-run', action='store_true', help='Offline inspection only (default)')
    p.add_argument('--start-q', nargs=7, type=float, help='Offline start pose in radians; no default guess')
    p.add_argument('--first-move-s', type=float, default=10.)
    p.add_argument('--move-s', type=float, default=4.)
    p.add_argument('--max-joint-speed-deg', type=float, default=10.)
    p.add_argument('--max-tracking-error-deg', type=float, default=10.)
    p.add_argument('--capture-tolerance-deg', type=float, default=1.)
    p.add_argument('--settle-s', type=float, default=1.)
    p.add_argument('--hold-s', type=float, default=1.)
    p.add_argument('--table-z', type=float, default=.53)
    p.add_argument('--table-box', nargs=4, type=float, default=[-.6,.6,-1.4,-.55])
    p.add_argument('--min-clear', type=float, default=0.,
                   help='extra table clearance in metres (default 0; modeled penetration still rejected)')
    p.add_argument('--view', action='store_true')
    p.add_argument('--view-port', type=int, default=8111)
    p.add_argument('--serial-number', default='15107')
    p.add_argument('--camera-resolution', default='HD1080', choices=['HD1080','HD720','VGA'])
    p.add_argument('--camera-upsidedown', action='store_true')
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=540)
    p.add_argument('--exposure', type=int, default=25)
    p.add_argument('--gain', type=int, default=40)
    p.add_argument('--urdf', default=str(cal.URDF))
    p.add_argument('--out-dir', default='/juno/u/tylerlum/camera_calibration')
    args = p.parse_args()
    for key in ('first_move_s','move_s','max_joint_speed_deg','max_tracking_error_deg',
                'capture_tolerance_deg','settle_s','hold_s'):
        if not np.isfinite(getattr(args,key)) or getattr(args,key) <= 0:
            p.error(f'{key} must be positive and finite')
    if not np.isfinite(args.min_clear) or args.min_clear < 0:
        p.error('min_clear must be nonnegative and finite')
    if args.first_move_s < 10 or args.move_s < 4:
        p.error('First move must be at least 10 seconds; subsequent moves at least 4 seconds')
    run(args)


if __name__ == '__main__':
    main()
