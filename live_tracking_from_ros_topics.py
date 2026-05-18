#!/usr/bin/env python
"""FoundationPose tracker that consumes RGB-D frames from ROS topics.

This is the ROS-topic equivalent of ``live_tracking_with_ros.py``. It does not
open a ZED camera directly, so it can run in the existing FoundationPose env
even when ``pyzed.sl`` is unavailable.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import nvdiffrast.torch as dr
import numpy as np
import rospy
import trimesh
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, draw_xyz_axis
from generate_mask import generate_binary_mask_box


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    k: np.ndarray
    stamp: rospy.Time
    frame_id: str


class LatestRGBD:
    def __init__(self, *, fallback_k: np.ndarray | None = None) -> None:
        self._lock = threading.Lock()
        self._rgb: np.ndarray | None = None
        self._rgb_stamp: rospy.Time | None = None
        self._rgb_frame_id: str = "camera_frame"
        self._depth: np.ndarray | None = None
        self._depth_stamp: rospy.Time | None = None
        self._k: np.ndarray | None = fallback_k

    def rgb_callback(self, msg: Image) -> None:
        rgb = image_msg_to_rgb(msg)
        with self._lock:
            self._rgb = rgb
            self._rgb_stamp = msg.header.stamp
            self._rgb_frame_id = msg.header.frame_id or "camera_frame"

    def depth_callback(self, msg: Image) -> None:
        depth_m = image_msg_to_depth_m(msg)
        with self._lock:
            self._depth = depth_m
            self._depth_stamp = msg.header.stamp

    def camera_info_callback(self, msg: CameraInfo) -> None:
        k = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
        with self._lock:
            self._k = k

    def latest(self) -> RGBDFrame | None:
        with self._lock:
            if self._rgb is None or self._depth is None or self._k is None:
                return None
            rgb = self._rgb.copy()
            depth = self._depth.copy()
            k = self._k.copy()
            stamp = self._rgb_stamp if self._rgb_stamp is not None else rospy.Time.now()
            frame_id = self._rgb_frame_id

        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return RGBDFrame(rgb=rgb, depth_m=depth, k=k, stamp=stamp, frame_id=frame_id)


def image_msg_to_rgb(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    height, width = int(msg.height), int(msg.width)
    if encoding in {"rgb8", "bgr8"}:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, int(msg.step))
        arr = arr[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            arr = arr[..., ::-1]
        return np.ascontiguousarray(arr)
    if encoding in {"rgba8", "bgra8"}:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, int(msg.step))
        arr = arr[:, : width * 4].reshape(height, width, 4)[..., :3]
        if encoding == "bgra8":
            arr = arr[..., ::-1]
        return np.ascontiguousarray(arr)
    if encoding == "mono8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, int(msg.step))
        arr = arr[:, :width]
        return np.ascontiguousarray(np.repeat(arr[..., None], 3, axis=2))
    raise ValueError(f"Unsupported RGB image encoding: {msg.encoding!r}")


def image_msg_to_depth_m(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    height, width = int(msg.height), int(msg.width)
    if encoding in {"32fc1", "passthrough"}:
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(height, int(msg.step) // 4)
        depth = arr[:, :width].astype(np.float32, copy=True)
    elif encoding == "16uc1":
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, int(msg.step) // 2)
        depth = arr[:, :width].astype(np.float32) * 0.001
    else:
        raise ValueError(f"Unsupported depth image encoding: {msg.encoding!r}")
    depth[(depth < 0.001) | (~np.isfinite(depth))] = 0.0
    return depth


def select_mask_with_sam(rgb: np.ndarray) -> np.ndarray | None:
    bgr = rgb[..., ::-1]
    mask = generate_binary_mask_box(bgr, polygon_refinement=True)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def pose_matrix_to_posestamped(T: np.ndarray, frame_id: str, stamp: rospy.Time | None = None) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
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


def load_transform(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)
    if path.suffix == ".npy":
        transform = np.load(path)
    else:
        transform = np.loadtxt(path)
    return np.asarray(transform, dtype=np.float64).reshape(4, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh_path", required=True, help="Object mesh path in meters.")
    parser.add_argument("--calibration", type=Path, default=None, help="4x4 camera-to-robot transform.")
    parser.add_argument("--rgb_topic", default="/zed/zed_node/rgb/image_rect_color")
    parser.add_argument("--depth_topic", default="/zed/zed_node/depth/depth_registered")
    parser.add_argument("--camera_info_topic", default="/zed/zed_node/rgb/camera_info")
    parser.add_argument("--camera_pose_topic", default="/camera_frame/current_object_pose")
    parser.add_argument("--robot_pose_topic", default="/robot_frame/current_object_pose")
    parser.add_argument("--cam_K", type=Path, default=None, help="Fallback 3x3 K if CameraInfo is absent.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--save_dir", default="live_tracking_ros_topic_results")
    parser.add_argument("--wait_timeout_s", type=float, default=30.0)
    parser.add_argument("--max_depth_age_s", type=float, default=0.25)
    return parser.parse_args()


def wait_for_frame(cache: LatestRGBD, timeout_s: float) -> RGBDFrame:
    start = time.time()
    while not rospy.is_shutdown():
        frame = cache.latest()
        if frame is not None:
            return frame
        if timeout_s >= 0.0 and time.time() - start > timeout_s:
            raise TimeoutError("Timed out waiting for RGB, depth, and CameraInfo/cam_K.")
        time.sleep(0.01)
    raise RuntimeError("ROS shutdown while waiting for first RGB-D frame.")


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    fallback_k = np.loadtxt(args.cam_K).reshape(3, 3) if args.cam_K is not None else None
    t_rc = load_transform(args.calibration)

    rospy.init_node("foundationpose_ros_topic_tracking", anonymous=True)
    cache = LatestRGBD(fallback_k=fallback_k)
    rospy.Subscriber(args.rgb_topic, Image, cache.rgb_callback, queue_size=1)
    rospy.Subscriber(args.depth_topic, Image, cache.depth_callback, queue_size=1)
    rospy.Subscriber(args.camera_info_topic, CameraInfo, cache.camera_info_callback, queue_size=1)
    camera_pose_pub = rospy.Publisher(args.camera_pose_topic, PoseStamped, queue_size=1)
    robot_pose_pub = rospy.Publisher(args.robot_pose_topic, PoseStamped, queue_size=1)

    if args.debug >= 1:
        cv2.namedWindow("FoundationPose ROS Topics", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FoundationPose ROS Topics", 960, 540)

    mesh = trimesh.load(args.mesh_path, process=False)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug=args.debug,
        glctx=glctx,
        debug_dir=args.save_dir,
    )

    frame = wait_for_frame(cache, args.wait_timeout_s)
    mask = select_mask_with_sam(frame.rgb)
    if mask is None or mask.sum() == 0:
        raise RuntimeError("Empty or missing initial mask; cannot register FoundationPose.")

    print("[foundationpose_ros_topic] registering initial pose...", flush=True)
    pose = est.register(
        K=frame.k,
        rgb=frame.rgb,
        depth=frame.depth_m,
        ob_mask=mask.astype(bool),
        iteration=args.est_refine_iter,
    )
    camera_pose_pub.publish(pose_matrix_to_posestamped(pose, frame.frame_id, frame.stamp))
    robot_pose_pub.publish(pose_matrix_to_posestamped(t_rc @ pose, "robot_frame", frame.stamp))
    print("[foundationpose_ros_topic] initial registration complete", flush=True)

    rate = rospy.Rate(args.fps) if args.fps > 0 else None
    last_print = time.time()
    while not rospy.is_shutdown():
        loop_t0 = time.perf_counter()
        frame = cache.latest()
        if frame is None:
            time.sleep(0.001)
            continue
        age_s = max(0.0, (rospy.Time.now() - frame.stamp).to_sec()) if frame.stamp else 0.0
        pose = est.track_one(rgb=frame.rgb, depth=frame.depth_m, K=frame.k, iteration=args.track_refine_iter)
        camera_pose_pub.publish(pose_matrix_to_posestamped(pose, frame.frame_id, frame.stamp))
        robot_pose_pub.publish(pose_matrix_to_posestamped(t_rc @ pose, "robot_frame", frame.stamp))

        if args.debug >= 1:
            vis = draw_xyz_axis(
                frame.rgb.copy(),
                ob_in_cam=pose,
                scale=0.1,
                K=frame.k,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            cv2.imshow("FoundationPose ROS Topics", vis[..., ::-1])
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

        now = time.time()
        if now - last_print > 2.0:
            last_print = now
            print(
                "[foundationpose_ros_topic] "
                f"track_ms={1000.0 * (time.perf_counter() - loop_t0):.1f} "
                f"depth_age_ms={1000.0 * age_s:.1f} "
                f"rgb_shape={frame.rgb.shape} depth_shape={frame.depth_m.shape} "
                f"K_fx_fy_cx_cy={frame.k[0,0]:.2f}/{frame.k[1,1]:.2f}/{frame.k[0,2]:.2f}/{frame.k[1,2]:.2f}",
                flush=True,
            )
            if age_s > args.max_depth_age_s:
                print(
                    f"[foundationpose_ros_topic] WARNING stale RGB-D frame age={1000.0 * age_s:.1f}ms",
                    flush=True,
                )
        if rate is not None:
            rate.sleep()


if __name__ == "__main__":
    main()
