"""Live 6-DoF tracking for two objects from the same ZED input stream.

Example:
  python /juno/u/tylerlum/FoundationPoseChanges/live_tracking_2_objects.py \
    --depth_backend zed_neural \
    --camera_resolution HD1080 \
    --camera_fps 30 \
    --fps 30 \
    --mesh_path /path/to/object_1.obj \
    --mesh_path_2 /path/to/object_2.obj \
    --save_dir live_tracking_2_objects_results \
    --ros

ROS topics:
  - object 1: camera_frame/current_object_pose
  - object 2: camera_frame/current_object_pose_2
  - robot-frame equivalents are also published unless --no_robot_frame is set.
"""

import argparse
import math
import os
import re
import sys
import time
from collections import deque as _deque
from pathlib import Path


_REPO_ROOT = Path(
    os.environ.get("DEPTHBASEDRL_ROOT", "/juno/u/kedia/depthbasedRL")
).resolve()
_FP_DIR = _REPO_ROOT / "third_party" / "FoundationPose"

for _path in (str(_FP_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# rospy MUST be imported before live_tracking_utils_2_objects, which imports
# the original live_tracking_utils.py and the FoundationPose stack.
if "--ros" in sys.argv:
    import rospy  # noqa: F401
    from geometry_msgs.msg import PoseStamped  # noqa: F401


import cv2  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # noqa: E402
from live_tracking_utils_2_objects import (  # noqa: E402
    DEFAULT_FAST_FS_MODEL,
    DEFAULT_FAST_FS_TRT_DIR,
    FastFsDepth,
    RosPosePublisher2Objects,
    T_RC_DEFAULT,
    ZedCaptureProcess,
    launch_viser_replay,
    save_run_2_objects,
    save_viz_data_2_objects,
    select_mask_with_sam,
    unproject_strided,
)


def _select_mask_safe(rgb):
    """Run SAM box masking even after FoundationPose changes torch defaults."""
    was_cuda = torch.tensor(0.0).is_cuda
    if was_cuda:
        torch.set_default_tensor_type(torch.FloatTensor)
    try:
        return select_mask_with_sam(rgb)
    finally:
        if was_cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Live 6D pose tracking for two objects."
    )
    parser.add_argument(
        "--depth_backend",
        choices=["zed_neural", "fast_fs"],
        required=True,
        help="Which depth source to feed FoundationPose.",
    )
    parser.add_argument(
        "--mesh_path",
        "--mesh_path_1",
        dest="mesh_path_1",
        type=str,
        default=None,
        help="Path to the first object mesh file (.obj, in meters).",
    )
    parser.add_argument(
        "--mesh_path_2",
        type=str,
        default=None,
        help="Path to the second object mesh file (.obj, in meters).",
    )
    parser.add_argument(
        "--mesh_paths",
        nargs=2,
        metavar=("MESH_1", "MESH_2"),
        default=None,
        help="Alternative: pass both object mesh paths at once.",
    )
    parser.add_argument(
        "--ros",
        action="store_true",
        help=(
            "Publish object 1 on current_object_pose and object 2 on "
            "current_object_pose_2."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help=(
            "Optional 4x4 camera-to-robot transform (.npy or .txt) overriding "
            "T_RC_DEFAULT. Only used when --ros is set."
        ),
    )
    parser.add_argument(
        "--no_robot_frame",
        action="store_true",
        help="With --ros, skip robot_frame pose publishing.",
    )
    parser.add_argument("--serial_number", type=str, default="15107")
    parser.add_argument("--camera_upsidedown", action="store_true")
    parser.add_argument(
        "--camera_resolution",
        choices=["HD1080", "HD720", "VGA"],
        default="HD1080",
        help="ZED capture resolution. HD1080=30fps max, HD720=60fps, VGA=100fps.",
    )
    parser.add_argument(
        "--camera_fps",
        type=int,
        default=0,
        help=(
            "ZED capture FPS. 0 = use the resolution default "
            "(30/60/100 for HD1080/HD720/VGA on ZED 2)."
        ),
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--exposure", type=int, default=25)
    parser.add_argument("--gain", type=int, default=40)
    parser.add_argument(
        "--fps",
        type=float,
        default=40.0,
        help="Target tracking-loop FPS, distinct from --camera_fps.",
    )
    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=5,
        help="Refinement iterations for initial registration.",
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=2,
        help="Refinement iterations per tracking step.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="live_tracking_2_objects_results"
    )
    parser.add_argument(
        "--max_buffer_seconds",
        type=float,
        default=60.0,
        help=(
            "Keep only this many seconds of frames/poses in RAM before "
            "dropping the oldest entries. 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--max_buffer_frames",
        type=int,
        default=0,
        help=(
            "Exact frame buffer cap. 0 derives the cap from "
            "--max_buffer_seconds and --fps."
        ),
    )
    parser.add_argument("--frame_id", type=str, default="camera_frame")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "Run the tracking loop for this many seconds after initial "
            "registration. 0 = run until ROS shutdown or quit key."
        ),
    )
    parser.add_argument(
        "--fast_fs_model",
        type=str,
        default=str(DEFAULT_FAST_FS_MODEL),
        help="Path to Fast-FS model_best_bp2_serialize.pth.",
    )
    parser.add_argument(
        "--fast_fs_iters",
        type=int,
        default=4,
        help="Fast-FS refinement iterations. TRT engines bake this in.",
    )
    parser.add_argument("--fast_fs_max_disp", type=int, default=192)
    parser.add_argument(
        "--fast_fs_width",
        type=int,
        default=384,
        help="Fast-FS processing width, forced to engine size in TRT mode.",
    )
    parser.add_argument(
        "--fast_fs_height",
        type=int,
        default=224,
        help="Fast-FS processing height, forced to engine size in TRT mode.",
    )
    parser.add_argument(
        "--fast_fs_trt_dir",
        type=str,
        default=str(DEFAULT_FAST_FS_TRT_DIR),
        help=(
            "Directory with feature_runner.engine, post_runner.engine, and "
            "onnx.yaml. Pass an empty string to use the PyTorch path."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "After saving, launch the existing viser replay for object 1. "
            "Both object pose streams are saved to viz_data.npz."
        ),
    )
    parser.add_argument(
        "--viz_stride",
        type=int,
        default=4,
        help="Stride for per-frame point cloud sampling.",
    )
    parser.add_argument(
        "--fp_refiner_trt",
        type=str,
        default=str(
            _FP_DIR / "weights" / "2023-10-28-18-33-37" / "refine_net_fp16.plan"
        ),
        help=(
            "Path to a TensorRT engine (.plan) for the FoundationPose refiner. "
            "Pass an empty string to fall back to PyTorch."
        ),
    )
    return parser


def _resolve_mesh_paths(args, parser):
    if args.mesh_paths is not None:
        mesh_paths = list(args.mesh_paths)
    else:
        mesh_paths = [args.mesh_path_1, args.mesh_path_2]
    if not mesh_paths[0] or not mesh_paths[1]:
        parser.error(
            "two mesh files are required: use --mesh_path OBJ1 --mesh_path_2 OBJ2 "
            "or --mesh_paths OBJ1 OBJ2"
        )
    resolved = []
    for mesh_path in mesh_paths:
        path = Path(mesh_path).expanduser()
        if not path.is_file():
            parser.error(f"mesh file does not exist: {mesh_path}")
        resolved.append(str(path.resolve()))
    args.mesh_paths = resolved
    args.mesh_path_1 = resolved[0]
    args.mesh_path_2 = resolved[1]
    return resolved


def _make_estimator(mesh_path, scorer, refiner, glctx, debug_dir):
    mesh = trimesh.load(mesh_path, process=False)
    return FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug=0,
        glctx=glctx,
        debug_dir=debug_dir,
    )


def _register_object(estimator, object_idx, K, rgb, depth, iteration):
    print(f"Select ROI for object {object_idx}.")
    mask = _select_mask_safe(rgb)
    if mask is None or mask.sum() == 0:
        raise RuntimeError(f"empty ROI selected for object {object_idx}")
    print(f"Registering initial pose for object {object_idx}...")
    t0 = time.time()
    pose = estimator.register(
        K=K, rgb=rgb, depth=depth, ob_mask=mask.astype(bool), iteration=iteration
    )
    print(
        f"Object {object_idx} initial registration done in "
        f"{time.time() - t0:.3f}s"
    )
    return pose


def _load_robot_transform(args):
    if args.ros and not args.no_robot_frame:
        if args.calibration is not None:
            arr = (
                np.load(args.calibration)
                if args.calibration.endswith(".npy")
                else np.loadtxt(args.calibration)
            )
            return arr.reshape(4, 4)
        return T_RC_DEFAULT
    if args.calibration is not None and not args.ros:
        print("[warn] --calibration ignored (only used with --ros)")
    return None


def _buffer_frame_cap(args):
    if args.max_buffer_frames > 0:
        return int(args.max_buffer_frames)
    if args.max_buffer_seconds <= 0:
        return None
    fps_for_cap = args.fps if args.fps > 0 else args.camera_fps
    if fps_for_cap <= 0:
        fps_for_cap = {"HD1080": 30, "HD720": 60, "VGA": 100}.get(
            args.camera_resolution, 30
        )
    return max(1, int(math.ceil(args.max_buffer_seconds * fps_for_cap)))


def main():
    parser = _build_parser()
    args = parser.parse_args()
    mesh_paths = _resolve_mesh_paths(args, parser)
    buffer_frame_cap = _buffer_frame_cap(args)

    T_RC = _load_robot_transform(args)
    ros_pub = (
        RosPosePublisher2Objects(
            args.frame_id, publish_robot_frame=(T_RC is not None)
        )
        if args.ros
        else None
    )

    os.makedirs(args.save_dir, exist_ok=True)
    raw_frames = _deque(maxlen=buffer_frame_cap)
    pose_log_1 = _deque(maxlen=buffer_frame_cap)
    pose_log_2 = _deque(maxlen=buffer_frame_cap)
    frame_times = _deque(maxlen=buffer_frame_cap)
    stage_times = {
        "grab": _deque(maxlen=buffer_frame_cap),
        "depth": _deque(maxlen=buffer_frame_cap),
        "track_1": _deque(maxlen=buffer_frame_cap),
        "track_2": _deque(maxlen=buffer_frame_cap),
        "track_total": _deque(maxlen=buffer_frame_cap),
        "publish": _deque(maxlen=buffer_frame_cap),
        "total": _deque(maxlen=buffer_frame_cap),
    }
    viz_log = _deque(maxlen=buffer_frame_cap)
    if buffer_frame_cap is None:
        print("[buffer] frame buffer cap disabled")
    else:
        print(f"[buffer] retaining newest {buffer_frame_cap} frames in RAM")
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
        t_spawn = time.time()
        capture = ZedCaptureProcess(
            zed_kwargs={
                "resolution": args.camera_resolution,
                "camera_fps": args.camera_fps,
                "serial_number": args.serial_number,
                "exposure": args.exposure,
                "gain": args.gain,
            },
            fp_size=(args.width, args.height),
            backend=args.depth_backend,
            camera_upsidedown=args.camera_upsidedown,
        )
        capture.start(ready_timeout=30.0)
        K = capture.K
        baseline_m = capture.baseline_m
        print(
            f"[capture] producer ready in {time.time() - t_spawn:.2f}s; "
            f"K[0,0]={K[0, 0]:.1f}, baseline={baseline_m * 1000:.2f}mm"
        )
        last_seq = 0

        fast_fs = None
        if args.depth_backend == "fast_fs":
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
            if fast_fs.backend == "trt":
                match = re.search(r"iters(\d+)", args.fast_fs_trt_dir or "")
                iters_tag = f"iters{match.group(1)}" if match else "itersN"
            else:
                iters_tag = f"iters{args.fast_fs_iters}"
            backend_tag = f"fast_fs_{fast_fs.backend}_{iters_tag}"
            print(
                f"Fast-FS backend: {fast_fs.backend} at {fast_fs.fs_size}; "
                f"depth upsampled to {fast_fs.fp_size}"
            )

        def fetch():
            nonlocal last_seq
            t0 = time.time()
            out = capture.latest(since_seq=last_seq, timeout=2.0)
            if out is None:
                raise RuntimeError("capture process delivered no frame")
            seq, ts_ns, frame_a, frame_b = out
            last_seq = seq
            grab_s = time.time() - t0
            if args.depth_backend == "zed_neural":
                return frame_a, frame_b, ts_ns, grab_s, 0.0
            t1 = time.time()
            depth = fast_fs(frame_a, frame_b)
            return frame_a, depth, ts_ns, grab_s, time.time() - t1

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor(trt_engine_path=(args.fp_refiner_trt or None))
        glctx = dr.RasterizeCudaContext()
        est_1 = _make_estimator(
            mesh_paths[0],
            scorer,
            refiner,
            glctx,
            os.path.join(args.save_dir, "object_1_debug"),
        )
        est_2 = _make_estimator(
            mesh_paths[1],
            scorer,
            refiner,
            glctx,
            os.path.join(args.save_dir, "object_2_debug"),
        )

        rgb, depth, _, _, _ = fetch()
        pose_1 = _register_object(
            est_1, 1, K, rgb, depth, args.est_refine_iter
        )
        pose_2 = _register_object(
            est_2, 2, K, rgb, depth, args.est_refine_iter
        )
        if ros_pub is not None:
            ros_pub.publish(pose_1, pose_2, T_RC)

        control_img = np.zeros((150, 430, 3), dtype=np.uint8)
        for i, line in enumerate(
            [
                "d: discard buffer",
                "r: save take + re-register both",
                "q / ESC: quit",
            ]
        ):
            cv2.putText(
                control_img,
                line,
                (10, 35 + 35 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

        def _show_control_window():
            cv2.namedWindow("Keyboard Control", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Keyboard Control", 430, 150)
            cv2.imshow("Keyboard Control", control_img)
            cv2.waitKey(1)

        _show_control_window()

        def _clear_buffers():
            raw_frames.clear()
            pose_log_1.clear()
            pose_log_2.clear()
            frame_times.clear()
            for values in stage_times.values():
                values.clear()
            viz_log.clear()

        target_frame_time = 1.0 / args.fps if args.fps > 0 else 0.0
        loop_start_time = time.time()

        import logging as _logging

        _logging.getLogger().setLevel(_logging.WARNING)

        work_history = _deque(
            maxlen=max(30, int(args.fps) if args.fps > 0 else 30)
        )

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
                pose_1 = est_1.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    iteration=args.track_refine_iter,
                )
                t_track_1 = time.time() - t0

                t0 = time.time()
                pose_2 = est_2.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    iteration=args.track_refine_iter,
                )
                t_track_2 = time.time() - t0
                t_track_total = t_track_1 + t_track_2

                t_publish = 0.0
                if ros_pub is not None:
                    t0 = time.time()
                    ros_pub.publish(pose_1, pose_2, T_RC)
                    t_publish = time.time() - t0

                raw_frames.append(rgb)
                pose_log_1.append(pose_1)
                pose_log_2.append(pose_2)
                s = args.viz_stride
                viz_log.append((rgb[::s, ::s].copy(), depth[::s, ::s].copy()))

                work_elapsed = time.time() - loop_start
                sleep_time = target_frame_time - work_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                cv2.imshow("Keyboard Control", control_img)
                key = cv2.waitKey(1) & 0xFF

                iter_elapsed = time.time() - loop_start
                frame_times.append(iter_elapsed)
                stage_times["grab"].append(t_grab)
                stage_times["depth"].append(t_depth)
                stage_times["track_1"].append(t_track_1)
                stage_times["track_2"].append(t_track_2)
                stage_times["track_total"].append(t_track_total)
                stage_times["publish"].append(t_publish)
                stage_times["total"].append(iter_elapsed)

                work_history.append(work_elapsed)
                fps_inst = 1.0 / max(work_elapsed, 1e-6)
                fps_avg = len(work_history) / sum(work_history)
                if buffer_frame_cap is None:
                    buffer_count = f"{len(frame_times):4d}"
                else:
                    buffer_count = f"{len(frame_times):4d}/{buffer_frame_cap}"
                print(
                    f"\r[fps] achieved inst={fps_inst:5.1f}  "
                    f"avg{len(work_history)}={fps_avg:5.1f}  "
                    f"(cap={args.fps:.0f})  n={buffer_count}  ",
                    end="",
                    flush=True,
                )

                if key == ord("d"):
                    n = len(raw_frames)
                    _clear_buffers()
                    loop_start_time = time.time()
                    print(f"[live_tracking_2_objects] discarded {n} frames")
                elif key == ord("r"):
                    if raw_frames:
                        take_dir = save_run_2_objects(
                            save_dir=args.save_dir,
                            backend_name=backend_tag,
                            resolution=args.camera_resolution,
                            raw_frames=raw_frames,
                            pose_log_1=pose_log_1,
                            pose_log_2=pose_log_2,
                            K=K,
                            loop_start_time=_save_loop_start_time(),
                            frame_times=frame_times,
                            stage_times=stage_times,
                            target_fps=args.fps,
                            mesh_paths=mesh_paths,
                        )
                        if viz_log and K is not None and take_dir:
                            save_viz_data_2_objects(
                                take_dir,
                                viz_log,
                                pose_log_1,
                                pose_log_2,
                                K,
                                args.viz_stride,
                                mesh_paths,
                            )
                    _clear_buffers()
                    print("[live_tracking_2_objects] re-registering on fresh frame...")
                    rgb, depth, _, _, _ = fetch()
                    pose_1 = _register_object(
                        est_1, 1, K, rgb, depth, args.est_refine_iter
                    )
                    pose_2 = _register_object(
                        est_2, 2, K, rgb, depth, args.est_refine_iter
                    )
                    if ros_pub is not None:
                        ros_pub.publish(pose_1, pose_2, T_RC)
                    _show_control_window()
                    loop_start_time = time.time()
                elif key in (ord("q"), 27):
                    print("[live_tracking_2_objects] quit key pressed; exiting loop.")
                    break
        except KeyboardInterrupt:
            interrupted = True
            print("\n[live_tracking_2_objects] Ctrl+C received, stopping capture.")

    finally:
        print()
        producer_stats = None
        if capture is not None:
            capture.stop_capture()
            producer_stats = capture.producer_stats
        if producer_stats:
            lines = [
                "--- Producer per-stage timings (ms) ---",
                f"  frames: {producer_stats['n_frames']}",
                "              mean    p50    p95    p99    min    max",
            ]
            for stage in (
                "grab",
                "left",
                "depth" if args.depth_backend == "zed_neural" else "right",
                "publish_lock",
            ):
                stats = producer_stats.get(stage)
                if not stats:
                    continue
                lines.append(
                    f"  {stage:<12s}"
                    f"{stats['mean_ms']:7.1f}{stats['p50_ms']:7.1f}"
                    f"{stats['p95_ms']:7.1f}{stats['p99_ms']:7.1f}"
                    f"{stats['min_ms']:7.1f}{stats['max_ms']:7.1f}"
                )
            print("\n".join(lines))
        cv2.destroyAllWindows()

        do_save = True
        if interrupted and raw_frames and sys.stdin.isatty():
            import signal as _signal

            try:
                _signal.signal(_signal.SIGINT, _signal.default_int_handler)
            except Exception:
                pass
            try:
                ans = (
                    input(
                        f"Save {len(raw_frames)} two-object frames to "
                        f"{args.save_dir}? [Y/n] "
                    )
                    .strip()
                    .lower()
                )
                do_save = ans in ("", "y", "yes")
            except (KeyboardInterrupt, EOFError):
                print("\n[live_tracking_2_objects] aborting without save.")
                do_save = False

        run_dir = None
        if do_save:
            run_dir = save_run_2_objects(
                save_dir=args.save_dir,
                backend_name=backend_tag,
                resolution=args.camera_resolution,
                raw_frames=raw_frames,
                pose_log_1=pose_log_1,
                pose_log_2=pose_log_2,
                K=K,
                loop_start_time=_save_loop_start_time(),
                frame_times=frame_times,
                stage_times=stage_times,
                target_fps=args.fps,
                mesh_paths=mesh_paths,
            )

            if viz_log and K is not None and run_dir:
                save_viz_data_2_objects(
                    run_dir,
                    viz_log,
                    pose_log_1,
                    pose_log_2,
                    K,
                    args.viz_stride,
                    mesh_paths,
                )
                if args.visualize:
                    print(
                        "[viser] launching existing replay for object 1 only; "
                        "object 2 poses are saved in viz_data.npz"
                    )
                    pc_log = [
                        unproject_strided(
                            rgb_s, depth_s, K, stride=args.viz_stride
                        )
                        for rgb_s, depth_s in viz_log
                    ]
                    launch_viser_replay(
                        pc_log,
                        list(pose_log_1),
                        K,
                        mesh_path=mesh_paths[0],
                        T_RC=T_RC_DEFAULT,
                    )


if __name__ == "__main__":
    main()
