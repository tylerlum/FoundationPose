"""Helpers for two-object FoundationPose live tracking.

This module intentionally lives outside the vendored FoundationPose tree. It
adds only the pieces needed by live_tracking_2_objects.py and reuses the
existing live_tracking_utils.py implementation for camera capture, depth, SAM
masking, and replay helpers.

When using ROS, import rospy before importing this module, matching the note in
third_party/FoundationPose/live_tracking_utils.py.
"""

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh


# This file lives at the root of the FoundationPose checkout, so the package
# directory is simply its own parent. (The upstream version of this script was
# written for a vendored copy at <repo>/third_party/FoundationPose and resolved
# the root via $DEPTHBASEDRL_ROOT; that indirection is not needed here.)
_FP_DIR = Path(__file__).resolve().parent

for _path in (str(_FP_DIR),):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import live_tracking_utils as _base  # noqa: E402
from live_tracking_utils import (  # noqa: E402,F401
    DEFAULT_FAST_FS_MODEL,
    DEFAULT_FAST_FS_TRT_DIR,
    FastFsDepth,
    T_RC_DEFAULT,
    ZedCaptureProcess,
    launch_viser_replay,
    save_viz_data,
    select_mask_with_sam,
    unproject_strided,
)


_AXIS_X = np.array([0.1, 0.0, 0.0, 1.0])
_AXIS_Y = np.array([0.0, 0.1, 0.0, 1.0])
_AXIS_Z = np.array([0.0, 0.0, 0.1, 1.0])
_ORIGIN_H = np.array([0.0, 0.0, 0.0, 1.0])

# RGB values written into RGB frames. Object 1 keeps the original red/green/blue
# axes. Object 2 uses cyan/magenta/yellow so the overlay is easy to separate.
_OBJ_AXIS_COLORS = (
    ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
    ((0, 255, 255), (255, 0, 255), (255, 255, 0)),
)


class RosPosePublisher2Objects:
    """Publish poses for two tracked objects.

    Object 1 publishes on the original topics:
      - camera_frame/current_object_pose
      - robot_frame/current_object_pose

    Object 2 publishes on:
      - camera_frame/current_object_pose_2
      - robot_frame/current_object_pose_2
    """

    def __init__(self, frame_id, publish_robot_frame):
        import rospy
        from geometry_msgs.msg import PoseStamped

        self._rospy = rospy
        self._PoseStamped = PoseStamped
        rospy.init_node("foundationpose_live_tracking_2_objects", anonymous=True)
        self.frame_id = frame_id
        self.cam_pub_1 = rospy.Publisher(
            "camera_frame/current_object_pose", PoseStamped, queue_size=1
        )
        self.cam_pub_2 = rospy.Publisher(
            "camera_frame/current_object_pose_2", PoseStamped, queue_size=1
        )
        self.robot_pub_1 = (
            rospy.Publisher(
                "robot_frame/current_object_pose", PoseStamped, queue_size=1
            )
            if publish_robot_frame
            else None
        )
        self.robot_pub_2 = (
            rospy.Publisher(
                "robot_frame/current_object_pose_2", PoseStamped, queue_size=1
            )
            if publish_robot_frame
            else None
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

    def publish(self, pose_cam_1, pose_cam_2, T_RC=None):
        self.cam_pub_1.publish(self._msg(pose_cam_1, self.frame_id))
        self.cam_pub_2.publish(self._msg(pose_cam_2, self.frame_id))
        if T_RC is not None:
            if self.robot_pub_1 is not None:
                self.robot_pub_1.publish(self._msg(T_RC @ pose_cam_1, "robot_frame"))
            if self.robot_pub_2 is not None:
                self.robot_pub_2.publish(self._msg(T_RC @ pose_cam_2, "robot_frame"))

    def shutdown_requested(self):
        return self._rospy.is_shutdown()


def _stage_stats_ms(times):
    t = np.asarray(times, dtype=np.float64)
    return {
        "mean": float(t.mean() * 1e3),
        "p50": float(np.median(t) * 1e3),
        "p95": float(np.percentile(t, 95) * 1e3),
        "min": float(t.min() * 1e3),
        "max": float(t.max() * 1e3),
    }


def _pose_array(pose_log):
    poses = np.full((len(pose_log), 4, 4), np.nan, dtype=np.float64)
    for i, pose in enumerate(pose_log):
        if pose is not None:
            poses[i] = pose
    return poses


def _draw_pose_axes(overlay, K, pose, obj_idx):
    if pose is None or K is None:
        return
    colors = _OBJ_AXIS_COLORS[obj_idx]
    try:
        origin = tuple(_base.project_3d_to_2d(_ORIGIN_H, K, pose))
        px = tuple(_base.project_3d_to_2d(_AXIS_X, K, pose))
        py = tuple(_base.project_3d_to_2d(_AXIS_Y, K, pose))
        pz = tuple(_base.project_3d_to_2d(_AXIS_Z, K, pose))
    except Exception:
        return
    cv2.arrowedLine(overlay, origin, px, colors[0], 3, cv2.LINE_AA)
    cv2.arrowedLine(overlay, origin, py, colors[1], 3, cv2.LINE_AA)
    cv2.arrowedLine(overlay, origin, pz, colors[2], 3, cv2.LINE_AA)
    cv2.putText(
        overlay,
        f"obj{obj_idx + 1}",
        (int(origin[0]) + 6, int(origin[1]) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        colors[0],
        2,
        cv2.LINE_AA,
    )


def save_run_2_objects(
    save_dir,
    backend_name,
    resolution,
    raw_frames,
    pose_log_1,
    pose_log_2,
    K,
    loop_start_time,
    frame_times,
    target_fps,
    stage_times=None,
    mesh_paths=None,
):
    """Write raw video, two-object overlay video, pose arrays, and metrics."""
    if not raw_frames or loop_start_time is None:
        return None

    elapsed = max(time.time() - loop_start_time, 1e-6)
    measured_fps = max(1, int(round(len(raw_frames) / elapsed)))
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        save_dir, f"{stamp}_{backend_name}_2objects_{resolution}_fps{measured_fps}"
    )
    os.makedirs(run_dir, exist_ok=True)

    raw_path = os.path.join(run_dir, "raw.mp4")
    overlay_path = os.path.join(run_dir, "overlay.mp4")

    raw_writer = imageio.get_writer(
        raw_path, fps=measured_fps, codec="libx264", quality=8, macro_block_size=1
    )
    for frame in raw_frames:
        raw_writer.append_data(frame)
    raw_writer.close()

    overlay_writer = imageio.get_writer(
        overlay_path,
        fps=measured_fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    for i, rgb_frame in enumerate(raw_frames):
        overlay = rgb_frame.copy()
        pose_1 = pose_log_1[i] if i < len(pose_log_1) else None
        pose_2 = pose_log_2[i] if i < len(pose_log_2) else None
        _draw_pose_axes(overlay, K, pose_1, 0)
        _draw_pose_axes(overlay, K, pose_2, 1)
        overlay_writer.append_data(overlay)
    overlay_writer.close()

    poses_1_path = os.path.join(run_dir, "poses_object_1.npy")
    poses_2_path = os.path.join(run_dir, "poses_object_2.npy")
    np.save(poses_1_path, _pose_array(pose_log_1))
    np.save(poses_2_path, _pose_array(pose_log_2))

    metadata = {
        "mesh_paths": list(mesh_paths or []),
        "topics": {
            "object_1_camera": "camera_frame/current_object_pose",
            "object_2_camera": "camera_frame/current_object_pose_2",
            "object_1_robot": "robot_frame/current_object_pose",
            "object_2_robot": "robot_frame/current_object_pose_2",
        },
    }
    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Saved: {raw_path}, {overlay_path}, {poses_1_path}, and {poses_2_path} "
        f"({len(raw_frames)} frames @ {measured_fps} FPS over {elapsed:.2f}s)"
    )

    if frame_times:
        ft = np.asarray(frame_times, dtype=np.float64)
        if target_fps > 0:
            fps_arr = 1.0 / np.clip(ft, 1e-6, None)
            hit_pct = float((fps_arr >= target_fps).mean() * 100.0)
            budget_ms = float(1e3 / target_fps)
        else:
            hit_pct = None
            budget_ms = None
        stages = {
            name: _stage_stats_ms(ts)
            for name, ts in (stage_times or {}).items()
            if ts
        }
        metrics = {
            "backend": backend_name,
            "resolution": resolution,
            "timestamp": stamp,
            "n_frames": int(len(raw_frames)),
            "elapsed_s": float(elapsed),
            "target_fps": float(target_fps),
            "achieved_fps": float(measured_fps),
            "per_frame_ms": _stage_stats_ms(frame_times),
            "stages_ms": stages,
            "hit_target_pct": hit_pct,
            "budget_ms": budget_ms,
        }
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        lines = [
            "--- Loop timing ---",
            f"  target FPS:           {target_fps:.1f}",
            f"  achieved (wall) FPS:  {measured_fps:.1f}",
            (
                "  per-frame total (ms): "
                f"mean={ft.mean() * 1e3:.1f}  "
                f"p50={np.median(ft) * 1e3:.1f}  "
                f"p95={np.percentile(ft, 95) * 1e3:.1f}  "
                f"min={ft.min() * 1e3:.1f}  max={ft.max() * 1e3:.1f}"
            ),
        ]
        if budget_ms is not None:
            lines.append(
                f"  frames with total <= 1/target ({budget_ms:.1f}ms): "
                f"{hit_pct:.1f}%"
            )
        if stages:
            lines.append("  per-stage (ms):          mean    p50    p95    min    max")
            for name, stats in stages.items():
                lines.append(
                    f"    {name:<13s}"
                    f"{stats['mean']:7.1f}{stats['p50']:7.1f}"
                    f"{stats['p95']:7.1f}{stats['min']:7.1f}"
                    f"{stats['max']:7.1f}"
                )
        lines.append(f"  metrics: {metrics_path}")
        print("\n".join(lines))

    return run_dir


def save_viz_data_2_objects(
    run_dir,
    viz_log,
    pose_log_1,
    pose_log_2,
    K,
    stride,
    mesh_paths,
    timestamps=None,
):
    """Save strided RGB/depth and both pose streams into viz_data.npz."""
    if not viz_log:
        return None
    rgb_s = np.stack([rgb for rgb, _ in viz_log], axis=0)
    depth_s = np.stack([depth for _, depth in viz_log], axis=0)
    n = len(viz_log)
    poses_1 = _pose_array(list(pose_log_1)[:n])
    poses_2 = _pose_array(list(pose_log_2)[:n])
    ts = (
        np.asarray(timestamps, dtype=np.int64)
        if timestamps is not None
        else np.zeros(n, dtype=np.int64)
    )
    mesh_path_1 = str(mesh_paths[0]) if mesh_paths else ""
    mesh_path_2 = str(mesh_paths[1]) if mesh_paths and len(mesh_paths) > 1 else ""
    path = os.path.join(run_dir, "viz_data.npz")
    np.savez_compressed(
        path,
        rgb=rgb_s,
        depth=depth_s,
        poses=poses_1,
        poses_1=poses_1,
        poses_2=poses_2,
        K=np.asarray(K, dtype=np.float64),
        stride=np.int64(stride),
        mesh_path=mesh_path_1,
        mesh_path_1=mesh_path_1,
        mesh_path_2=mesh_path_2,
        timestamps=ts,
    )
    print(
        f"[viz] saved two-object viz data to {path} "
        f"({rgb_s.shape[0]} frames, "
        f"{rgb_s.nbytes / 1e6 + depth_s.nbytes / 1e6:.1f} MB raw)"
    )
    return path
