"""Live 6-DoF object pose tracking with the ZED, dispatched on depth backend.

  --depth_backend zed_neural   ZED SDK's onboard NEURAL depth.
  --depth_backend fast_fs      Fast-FoundationStereo over ZED's LEFT+RIGHT.

Pass --ros to publish PoseStamped on:
  - camera_frame/current_object_pose
  - robot_frame/current_object_pose  (uses T_RC_DEFAULT unless --calibration
    overrides it; pass --no_robot_frame to skip the robot-frame publish)

    python third_party/FoundationPose/live_tracking.py --depth_backend zed_neural --camera_resolution HD1080 --camera_fps 30 --fps 30 --mesh_path assets/urdf/fabrica/beam_3x/0/0_colored.obj --save_dir May21_tests_policy/ --ros
"""

import os
import re
import sys
import time
import argparse
import math
from collections import deque as _deque
from pathlib import Path

_FP_DIR = Path(__file__).resolve().parent

# rospy MUST be imported BEFORE live_tracking_utils (which loads the FP stack)
# — see the segfault note in live_tracking_utils.py. Gated on --ros so non-ROS
# runs don't pay the import cost.
if '--ros' in sys.argv:
    import rospy  # noqa: F401
    from geometry_msgs.msg import PoseStamped  # noqa: F401

import numpy as np
import cv2
import pyzed.sl as sl
import torch
import trimesh
import nvdiffrast.torch as dr

from live_tracking_utils import (
    select_mask_with_sam, RosPosePublisher, T_RC_DEFAULT,
    FastFsDepth, DEFAULT_FAST_FS_MODEL, DEFAULT_FAST_FS_TRT_DIR, save_run,
    ZedCaptureProcess, unproject_strided, launch_viser_replay, save_viz_data,
)
from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose


def _select_mask_safe(rgb):
    """select_mask_with_sam, but neutralizes the CUDA default tensor type
    that FoundationPose's Utils.py installs during register/track_one.
    Without this, transformers' from_pretrained inside the SAM loader
    raises ValueError('... requires accelerate') on every reset call.
    Mirrors the swap pattern already used inside generate_mask.py."""
    was_cuda = torch.tensor(0.).is_cuda
    if was_cuda:
        torch.set_default_tensor_type(torch.FloatTensor)
    try:
        return select_mask_with_sam(rgb)
    finally:
        if was_cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)


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
    args = _build_parser().parse_args()
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

        rgb, depth, _, _, _ = fetch()
        mask = _select_mask_safe(rgb)
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
        control_img = np.zeros((130, 360, 3), dtype=np.uint8)
        for i, line in enumerate(['d: discard buffer',
                                  'r: save take + re-register',
                                  'q / ESC: quit']):
            cv2.putText(control_img, line, (10, 35 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        def _show_control_window():
            cv2.namedWindow('Keyboard Control', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Keyboard Control', 360, 130)
            cv2.imshow('Keyboard Control', control_img)
            cv2.waitKey(1)

        _show_control_window()

        def _clear_buffers():
            raw_frames.clear()
            pose_log.clear()
            frame_times.clear()
            for v in stage_times.values():
                v.clear()
            viz_log.clear()

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
                raw_frames.append(rgb)
                pose_log.append(pose)
                s = args.viz_stride
                viz_log.append((rgb[::s, ::s].copy(), depth[::s, ::s].copy()))

                work_elapsed = time.time() - loop_start
                sleep_time = target_frame_time - work_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Refresh the control image every iter — without a periodic
                # imshow, waitKey can stall for tens-to-hundreds of ms per
                # call on some HighGUI backends, tanking the loop's fps.
                cv2.imshow('Keyboard Control', control_img)
                key = cv2.waitKey(1) & 0xFF

                # Record iter wall time AFTER sleep + waitKey so stage_times
                # reflects the real loop cadence (otherwise achieved_fps in
                # the dir name disagrees with hit_target_pct in metrics).
                iter_elapsed = time.time() - loop_start
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

                if key == ord('d'):
                    n = len(raw_frames)
                    _clear_buffers()
                    # Reset the take's start time so the next save's
                    # measured_fps (= n_frames / elapsed) isn't diluted by
                    # the discarded interval. Mirrors the 'r' branch below.
                    loop_start_time = time.time()
                    print(f'[live_tracking] discarded {n} frames')
                elif key == ord('r'):
                    if raw_frames:
                        take_dir = save_run(
                            save_dir=args.save_dir,
                            backend_name=backend_tag,
                            resolution=args.camera_resolution,
                            raw_frames=raw_frames,
                            pose_log=pose_log,
                            K=K,
                            loop_start_time=_save_loop_start_time(),
                            frame_times=frame_times,
                            stage_times=stage_times,
                            target_fps=args.fps,
                        )
                        if viz_log and K is not None and take_dir:
                            save_viz_data(take_dir, viz_log, pose_log, K,
                                          args.viz_stride, args.mesh_path)
                    _clear_buffers()
                    print('[live_tracking] re-registering on fresh frame...')
                    rgb, depth, _, _, _ = fetch()
                    mask = _select_mask_safe(rgb)
                    if mask is None or mask.sum() == 0:
                        print('Empty ROI on restart; exiting loop.')
                        break
                    pose = est.register(K=K, rgb=rgb, depth=depth,
                                        ob_mask=mask.astype(bool),
                                        iteration=args.est_refine_iter)
                    if ros_pub is not None:
                        ros_pub.publish(pose, T_RC)
                    _show_control_window()
                    loop_start_time = time.time()
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
        cv2.destroyAllWindows()

        # On Ctrl+C (in either non-ROS or --ros mode), prompt before the
        # (slow) encode+save. Natural exits (duration timer) save without
        # prompting. Non-tty stdin also saves silently so non-interactive
        # runs stay unchanged.
        do_save = True
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
            run_dir = save_run(
                save_dir=args.save_dir,
                backend_name=backend_tag,
                resolution=args.camera_resolution,
                raw_frames=raw_frames,
                pose_log=pose_log,
                K=K,
                loop_start_time=_save_loop_start_time(),
                frame_times=frame_times,
                stage_times=stage_times,
                target_fps=args.fps,
            )

            if viz_log and K is not None and run_dir:
                # Always stash to disk so the run can be replayed any time via
                # `python third_party/FoundationPose/live_tracking_viz_replay.py <run_dir>`.
                save_viz_data(run_dir, viz_log, pose_log, K,
                              args.viz_stride, args.mesh_path)
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
