"""Live 6-DoF object pose tracking with the ZED + AprilTag (FurnitureBench).

Spin-off of `live_tracking.py` that swaps FoundationPose for AprilTag
detection. Tracks ONE FurnitureBench part per run (default
`square_table_leg2`). Reuses the same ZED multiprocessing capture, ROS
publishing, save_run, and viser replay infrastructure.

  --depth_backend zed_neural   ZED SDK's onboard NEURAL depth (default).
                               AprilTag tracking doesn't use the depth,
                               but we keep it around so viser replay can
                               show per-frame point clouds.

On frames where no part tag is detected we hold and republish the last
good pose (see the --no-detection block below).

Example:
    python third_party/FoundationPose/live_tracking_apriltag.py \\
        --part square_table_leg2 --camera_resolution HD1080 --fps 30 \\
        --save_dir May21_tests_apriltag/ --ros
"""

import os
import sys
import time
import argparse
from pathlib import Path

_FP_DIR = Path(__file__).resolve().parent

# Same rospy-before-FP-stack ordering trick as live_tracking.py:26-28 —
# `live_tracking_utils` transitively loads FoundationPose, and importing
# rospy AFTER that reliably segfaults.
if '--ros' in sys.argv:
    import rospy  # noqa: F401
    from geometry_msgs.msg import PoseStamped  # noqa: F401

import numpy as np
import cv2

from live_tracking_utils import (
    RosPosePublisher, T_RC_DEFAULT,
    save_run, ZedCaptureProcess, unproject_strided, launch_viser_replay,
    save_viz_data,
)
from apriltag_tracker import AprilTagTracker
from furniture_bench_apriltag import config as fba_config


def _build_parser():
    parser = argparse.ArgumentParser(description="Live AprilTag pose tracking.")
    parser.add_argument('--part', type=str, default='square_table_leg2',
                        choices=sorted(fba_config.PARTS),
                        help='Vendored FurnitureBench part name (see '
                             'furniture_bench_apriltag/config.py).')
    parser.add_argument('--tag_family', type=str, default='tag36h11',
                        help='dt_apriltags family. FB hardcodes tag36h11.')
    parser.add_argument('--tag_size_m', type=float, default=None,
                        help='Override physical tag size (meters). Defaults '
                             'to the size in the vendored FB config '
                             '(0.0195 m for square_table parts).')
    parser.add_argument('--depth_backend', choices=['zed_neural', 'fast_fs'],
                        default='zed_neural',
                        help='ZED capture backend. AprilTag tracking only '
                             'needs RGB; zed_neural keeps depth around for '
                             'viser replay point clouds.')
    parser.add_argument('--mesh_path', type=str, default=None,
                        help='Optional mesh path for the viser replay '
                             'overlay (not used for tracking itself).')
    parser.add_argument('--ros', action='store_true',
                        help='Publish PoseStamped on camera_frame/current_object_pose '
                             'and robot_frame/current_object_pose '
                             '(robot frame uses T_RC_DEFAULT unless '
                             '--calibration overrides it).')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Optional 4x4 camera-to-robot transform '
                             '(.npy or .txt) overriding T_RC_DEFAULT. '
                             'Only used when --ros is set.')
    parser.add_argument('--no_robot_frame', action='store_true',
                        help='With --ros, skip robot_frame/current_object_pose '
                             '(only camera_frame is published).')
    parser.add_argument('--serial_number', type=str, default='15107')
    parser.add_argument('--camera_upsidedown', action='store_true')
    parser.add_argument('--camera_resolution', choices=['HD1080', 'HD720', 'VGA'],
                        default='HD1080')
    parser.add_argument('--camera_fps', type=int, default=0,
                        help='ZED capture FPS. 0 = use the resolution default.')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--exposure', type=int, default=25)
    parser.add_argument('--gain', type=int, default=40)
    parser.add_argument('--fps', type=float, default=40.0,
                        help='Target tracking-loop FPS.')
    parser.add_argument('--save_dir', type=str,
                        default='live_tracking_apriltag_results')
    parser.add_argument('--frame_id', type=str, default='camera_frame')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='Run the tracking loop for this many seconds, '
                             'then exit. 0 = run until ROS shutdown / Ctrl+C.')
    parser.add_argument('--visualize', action='store_true',
                        help='After the episode, auto-launch a viser replay.')
    parser.add_argument('--viz_stride', type=int, default=4)
    return parser


def main():
    args = _build_parser().parse_args()

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

    ros_pub = RosPosePublisher(args.frame_id,
                               publish_robot_frame=(T_RC is not None)) \
              if args.ros else None

    os.makedirs(args.save_dir, exist_ok=True)
    raw_frames = []
    pose_log = []
    frame_times = []
    stage_times = {'grab': [], 'depth': [], 'track': [], 'publish': [], 'total': []}
    viz_log = []
    K = None
    loop_start_time = None
    backend_tag = f'apriltag_{args.part}'
    capture = None
    interrupted = False
    last_pose = None
    n_detected_frames = 0
    n_total_frames = 0

    try:
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

        tracker = AprilTagTracker(
            args.part, K, family=args.tag_family,
            tag_size_override=args.tag_size_m,
        )
        print(f'[apriltag] part={args.part} tag_ids={tracker.part.tag_ids} '
              f'tag_size={tracker.detector.tag_size*1000:.2f}mm '
              f'family={args.tag_family}')

        def fetch():
            """(rgb, depth_or_right, ts_ns, grab_s, depth_s=0)."""
            nonlocal last_seq
            t0 = time.time()
            out = capture.latest(since_seq=last_seq, timeout=2.0)
            if out is None:
                raise RuntimeError('capture process delivered no frame')
            seq, ts_ns, a, b = out
            last_seq = seq
            return a, b, ts_ns, time.time() - t0, 0.0

        target_frame_time = 1.0 / args.fps if args.fps > 0 else 0.0
        loop_start_time = time.time()

        def keep_running():
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
                pose_new, n_tags = tracker.track(rgb)
                print(pose_new)
                t_track = time.time() - t0
                if pose_new is not None:
                    last_pose = pose_new
                    n_detected_frames += 1
                n_total_frames += 1

                t_publish = 0.0
                # Per design: hold AND republish the last good pose on
                # missed-detection frames. Skip only until the first
                # successful detection (last_pose is None).
                if ros_pub is not None and last_pose is not None:
                    t0 = time.time()
                    ros_pub.publish(last_pose, T_RC)
                    t_publish = time.time() - t0

                raw_frames.append(rgb)
                pose_log.append(last_pose if last_pose is not None
                                else np.eye(4))
                s = args.viz_stride
                viz_log.append((rgb[::s, ::s].copy(),
                                depth[::s, ::s].copy()))

                elapsed = time.time() - loop_start
                frame_times.append(elapsed)
                stage_times['grab'].append(t_grab)
                stage_times['depth'].append(t_depth)
                stage_times['track'].append(t_track)
                stage_times['publish'].append(t_publish)
                stage_times['total'].append(elapsed)
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            interrupted = True
            print('\n[live_tracking_apriltag] Ctrl+C received, stopping capture.')

    finally:
        producer_stats = None
        if capture is not None:
            capture.stop_capture()
            producer_stats = capture.producer_stats
        if producer_stats:
            lines = ['--- Producer per-stage timings (ms) ---',
                     f"  frames: {producer_stats['n_frames']}",
                     '              mean    p50    p95    p99    min    max']
            for stage in ('grab', 'left',
                          'depth' if args.depth_backend == 'zed_neural' else 'right',
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

        if n_total_frames:
            pct = 100.0 * n_detected_frames / n_total_frames
            print(f'[apriltag] tracked {n_detected_frames}/{n_total_frames} '
                  f'frames ({pct:.1f}% detection rate)')

        cv2.destroyAllWindows()

        do_save = True
        if interrupted and raw_frames and sys.stdin.isatty():
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
                print('\n[live_tracking_apriltag] aborting without save.')
                do_save = False

        run_dir = None
        if do_save and raw_frames:
            run_dir = save_run(
                save_dir=args.save_dir,
                backend_name=backend_tag,
                resolution=args.camera_resolution,
                raw_frames=raw_frames,
                pose_log=pose_log,
                K=K,
                loop_start_time=loop_start_time,
                frame_times=frame_times,
                stage_times=stage_times,
                target_fps=args.fps,
            )

            if viz_log and K is not None and run_dir:
                save_viz_data(run_dir, viz_log, pose_log, K,
                              args.viz_stride, args.mesh_path or '')
                if args.visualize:
                    print(f'[viser] preparing replay ({len(viz_log)} frames at '
                          f'stride={args.viz_stride}) ...')
                    pc_log = [
                        unproject_strided(rgb_s, depth_s, K, stride=args.viz_stride)
                        for rgb_s, depth_s in viz_log
                    ]
                    launch_viser_replay(pc_log, pose_log, K,
                                        mesh_path=args.mesh_path,
                                        T_RC=T_RC_DEFAULT)


if __name__ == '__main__':
    main()
