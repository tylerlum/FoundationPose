from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import trimesh
import tyro
import viser
from PIL import Image
from scipy.spatial.transform import Rotation as R


PoseKey = Literal["auto", "poses_cam", "poses_robot", "poses_world"]


# Manually edit this if you want to visualize camera-frame data in another world frame.
# Default is identity, so FoundationPose poses_cam and RGB-D points are shown in camera frame.
T_W_C = np.eye(4)


@dataclass
class Args:
    """Visualize RGB-D frames, a mesh, and a pose JSON with Viser."""

    video_dir: Path
    """Directory containing rgb/, depth/, and cam_K.txt."""

    mesh_path: Optional[Path] = None
    """Optional path to the object mesh."""

    pose_json_path: Optional[Path] = None
    """Optional pose JSON. Supports FoundationPose {poses_cam: [...]} or a raw pose list."""

    pose_key: PoseKey = "auto"
    """Pose key to read when pose_json_path is a dict."""

    port: int = 8081
    """Viser server port."""

    dt: float = 1.0 / 30.0
    """Playback timestep in seconds."""

    start_idx: int = 0
    """Initial frame index."""

    point_cloud_stride: int = 1
    """Pixel stride for RGB-D point cloud generation. 1 keeps all pixels."""

    point_cloud_subsample: int = 1
    """Subsample generated valid points by this factor. 1 keeps all valid points."""

    max_depth_m: float = 5.0
    """Maximum depth to visualize."""

    point_size: float = 0.002
    """Point cloud point size in meters."""

    show_point_cloud: bool = True
    """Whether to show per-frame RGB-D point clouds."""

    mesh_opacity: Optional[float] = None
    """Optional mesh opacity."""

    mesh_scale: float = 1.0
    """Uniform scale factor applied to mesh vertices after loading."""

    axes_length: float = 0.08
    """Length of visualized camera/object axes."""

    cache_size: int = 8
    """Number of processed RGB-D point clouds to cache."""


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = xyzw
    return np.array([w, x, y, z])


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (4, 4):
        return pose
    if pose.shape != (7,):
        raise ValueError(f"Expected pose shape (7,) or (4, 4), got {pose.shape}")

    transform = np.eye(4)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
    return transform


def transform_to_position_wxyz(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    position = transform[:3, 3]
    wxyz = xyzw_to_wxyz(R.from_matrix(transform[:3, :3]).as_quat())
    return position, wxyz


def load_poses(path: Path, pose_key: PoseKey) -> Tuple[np.ndarray, str]:
    with path.open("r") as f:
        data = json.load(f)

    selected_key = "raw_list"
    if isinstance(data, dict):
        if pose_key == "auto":
            for candidate in ("poses_cam", "poses_world", "poses_robot"):
                if candidate in data:
                    selected_key = candidate
                    data = data[candidate]
                    break
            else:
                raise ValueError(
                    f"{path} is a dict, but none of poses_cam/poses_world/poses_robot were found"
                )
        else:
            if pose_key not in data:
                raise ValueError(f"{path} does not contain key {pose_key}")
            selected_key = pose_key
            data = data[pose_key]
    elif pose_key != "auto":
        raise ValueError(f"pose_key={pose_key} was requested, but {path} is a raw list")

    transforms = np.stack([pose_to_transform(np.asarray(pose)) for pose in data], axis=0)
    return transforms, selected_key


def depth_to_points(
    depth_m: np.ndarray,
    K: np.ndarray,
    rgb: np.ndarray,
    stride: int,
    max_depth_m: float,
    subsample: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("point_cloud_stride must be >= 1")
    if subsample < 1:
        raise ValueError("point_cloud_subsample must be >= 1")

    h, w = depth_m.shape
    v_coords, u_coords = np.indices((h, w))
    v_coords = v_coords[::stride, ::stride]
    u_coords = u_coords[::stride, ::stride]
    depth = depth_m[::stride, ::stride]
    colors = rgb[::stride, ::stride, :]

    z = depth.reshape(-1)
    valid = (z > 0.0) & (z < max_depth_m)
    z = z[valid]
    u = u_coords.reshape(-1)[valid]
    v = v_coords.reshape(-1)[valid]
    cols = colors.reshape(-1, 3)[valid]

    if subsample > 1:
        z = z[::subsample]
        u = u[::subsample]
        v = v[::subsample]
        cols = cols[::subsample]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    points = np.stack([x, y, z], axis=1)
    return points.astype(np.float32), cols.astype(np.uint8)


def load_rgb_depth_points(
    rgb_path: Path,
    depth_path: Path,
    K: np.ndarray,
    args: Args,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb_image = Image.open(rgb_path).convert("RGB")
    depth = np.array(Image.open(depth_path)) / 1000.0
    if rgb_image.size != (depth.shape[1], depth.shape[0]):
        logging.warning(
            "RGB/depth size mismatch for %s: rgb=%s depth=%s; resizing RGB to depth size",
            rgb_path.name,
            (rgb_image.height, rgb_image.width),
            depth.shape[:2],
        )
        rgb_image = rgb_image.resize((depth.shape[1], depth.shape[0]), Image.Resampling.BILINEAR)
    rgb = np.array(rgb_image)
    return depth_to_points(
        depth_m=depth,
        K=K,
        rgb=rgb,
        stride=args.point_cloud_stride,
        max_depth_m=args.max_depth_m,
        subsample=args.point_cloud_subsample,
    )


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (points @ transform[:3, :3].T) + transform[:3, 3]


def estimate_center_pose_from_depth(depth_path: Path, K: np.ndarray, max_depth_m: float) -> np.ndarray:
    depth = np.array(Image.open(depth_path)) / 1000.0
    valid_depth = depth[(depth > 0.0) & (depth < max_depth_m)]
    if len(valid_depth) == 0:
        z = 1.0
        print("WARNING: No valid depth found for mesh-only placement; using z=1.0m")
    else:
        z = float(np.median(valid_depth))

    pose = np.eye(4)
    pose[:3, 3] = np.array([0.0, 0.0, z])
    return T_W_C @ pose


def main() -> None:
    args = tyro.cli(Args)

    rgb_dir = args.video_dir / "rgb"
    depth_dir = args.video_dir / "depth"
    cam_intrinsics_path = args.video_dir / "cam_K.txt"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
    if not cam_intrinsics_path.exists():
        raise FileNotFoundError(f"Missing camera intrinsics: {cam_intrinsics_path}")
    if args.mesh_path is not None and not args.mesh_path.exists():
        raise FileNotFoundError(f"Missing mesh: {args.mesh_path}")
    if args.pose_json_path is not None and not args.pose_json_path.exists():
        raise FileNotFoundError(f"Missing pose JSON: {args.pose_json_path}")

    rgb_paths = sorted(rgb_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    if len(rgb_paths) != len(depth_paths):
        raise ValueError(f"RGB/depth count mismatch: {len(rgb_paths)} vs {len(depth_paths)}")
    if not rgb_paths:
        raise ValueError(f"No PNG frames found in {rgb_dir}")

    K = np.loadtxt(cam_intrinsics_path).reshape(3, 3)
    poses = None
    selected_pose_key = "none"
    if args.pose_json_path is not None:
        poses, selected_pose_key = load_poses(args.pose_json_path, args.pose_key)
        if selected_pose_key == "poses_cam":
            poses = np.array([T_W_C @ T_C_O for T_C_O in poses])
    n_frames = min(len(rgb_paths), len(depth_paths), len(poses) if poses is not None else len(rgb_paths))
    if n_frames == 0:
        raise ValueError("No frames to visualize")
    start_idx = int(np.clip(args.start_idx, 0, n_frames - 1))

    mesh = None
    if args.mesh_path is not None:
        if args.mesh_scale <= 0.0:
            raise ValueError("mesh_scale must be > 0")
        mesh = trimesh.load(args.mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.apply_scale(args.mesh_scale)

    print(f"video_dir: {args.video_dir}")
    print(f"mesh_path: {args.mesh_path}")
    print(f"pose_json_path: {args.pose_json_path}")
    print(f"pose_key: {selected_pose_key}")
    print(f"frames: {n_frames}")
    print(f"point_cloud_stride: {args.point_cloud_stride}")
    print(f"point_cloud_subsample: {args.point_cloud_subsample}")

    server = viser.ViserServer(port=args.port)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = T_W_C[:3, 3]
        client.camera.wxyz = xyzw_to_wxyz(R.from_matrix(T_W_C[:3, :3]).as_quat())

    camera_position, camera_wxyz = transform_to_position_wxyz(T_W_C)
    server.scene.add_frame(
        "/camera",
        axes_length=args.axes_length,
        axes_radius=args.axes_length * 0.04,
        position=camera_position,
        wxyz=camera_wxyz,
    )
    server.scene.add_grid(
        "/camera/grid",
        width=1.0,
        height=1.0,
        cell_size=0.05,
        plane="xy",
        position=camera_position,
        wxyz=camera_wxyz,
    )

    object_frame = None
    mesh_handle = None
    dummy_handle = None
    if poses is not None:
        initial_object_pose = poses[start_idx]
    elif mesh is not None:
        print("WARNING: mesh_path was provided without pose_json_path; placing mesh at image-center xy and median first-frame depth.")
        initial_object_pose = estimate_center_pose_from_depth(depth_paths[start_idx], K, args.max_depth_m)
    else:
        initial_object_pose = None

    if initial_object_pose is not None:
        initial_position, initial_wxyz = transform_to_position_wxyz(initial_object_pose)
        object_frame = server.scene.add_frame(
            "/object",
            position=initial_position,
            wxyz=initial_wxyz,
            axes_length=args.axes_length,
            axes_radius=args.axes_length * 0.04,
        )

    if mesh is not None:
        if args.mesh_opacity is None:
            mesh_handle = server.scene.add_mesh_trimesh("/object/mesh", mesh=mesh)
        else:
            mesh_handle = server.scene.add_mesh_simple(
                "/object/mesh",
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.faces),
                color=(235, 180, 80),
                opacity=args.mesh_opacity,
            )
    elif poses is not None:
        print("WARNING: pose_json_path was provided without mesh_path; showing a dummy sphere at the object pose.")
        dummy_handle = server.scene.add_icosphere(
            "/object/dummy",
            radius=args.axes_length * 0.2,
            color=(255, 180, 0),
        )

    point_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    pcd_handle = None
    if args.show_point_cloud:
        points, colors = load_rgb_depth_points(rgb_paths[start_idx], depth_paths[start_idx], K, args)
        points = transform_points(T_W_C, points)
        point_cache[start_idx] = (points, colors)
        pcd_handle = server.scene.add_point_cloud(
            "/camera/rgbd_points",
            points=points,
            colors=colors,
            point_size=args.point_size,
            point_shape="rounded",
            precision="float32",
        )
        print(f"initial point cloud points: {len(points)}")

    fps = 1.0 / args.dt

    def frame_label(idx: int) -> str:
        current_time = idx * args.dt
        total_time = (n_frames - 1) * args.dt
        return f"{current_time:.3f}s/{total_time:.3f}s ({idx:04d}/{n_frames - 1:04d}) ({fps:.0f}fps)"

    frame_idx = start_idx
    paused = False

    def set_frame(idx: int) -> None:
        nonlocal frame_idx
        frame_idx = int(np.clip(idx, 0, n_frames - 1))
        if object_frame is not None:
            if poses is not None:
                object_pose = poses[frame_idx]
            else:
                object_pose = initial_object_pose
            position, wxyz = transform_to_position_wxyz(object_pose)
            object_frame.position = position
            object_frame.wxyz = wxyz
        frame_slider.label = frame_label(frame_idx)

        if pcd_handle is not None:
            if frame_idx not in point_cache:
                points, colors = load_rgb_depth_points(
                    rgb_paths[frame_idx], depth_paths[frame_idx], K, args
                )
                point_cache[frame_idx] = (transform_points(T_W_C, points), colors)
                while len(point_cache) > args.cache_size:
                    oldest_key = next(iter(point_cache))
                    if oldest_key == frame_idx and len(point_cache) > 1:
                        oldest_key = next(key for key in point_cache if key != frame_idx)
                    point_cache.pop(oldest_key)
            points, colors = point_cache[frame_idx]
            pcd_handle.points = points
            pcd_handle.colors = colors

    with server.gui.add_folder("Frame Controls"):
        frame_slider = server.gui.add_slider(
            label=frame_label(start_idx),
            min=0,
            max=n_frames - 1,
            step=1,
            initial_value=start_idx,
        )
        pause_button = server.gui.add_button("Pause")
        decrement_button = server.gui.add_button("Decrement")
        increment_button = server.gui.add_button("Increment")
        reset_button = server.gui.add_button("Reset")
        object_checkbox = server.gui.add_checkbox(
            "Show object",
            initial_value=object_frame is not None,
        )
        pcd_checkbox = server.gui.add_checkbox(
            "Show point cloud",
            initial_value=args.show_point_cloud,
        )

    @frame_slider.on_update
    def _(_) -> None:
        set_frame(int(frame_slider.value))

    @pause_button.on_click
    def _(_) -> None:
        nonlocal paused
        paused = not paused
        pause_button.label = "Play" if paused else "Pause"

    @decrement_button.on_click
    def _(_) -> None:
        nonlocal paused
        paused = True
        pause_button.label = "Play"
        frame_slider.value = int(np.clip(frame_idx - 1, 0, n_frames - 1))

    @increment_button.on_click
    def _(_) -> None:
        nonlocal paused
        paused = True
        pause_button.label = "Play"
        frame_slider.value = int(np.clip(frame_idx + 1, 0, n_frames - 1))

    @reset_button.on_click
    def _(_) -> None:
        frame_slider.value = 0

    @object_checkbox.on_update
    def _(_) -> None:
        visible = bool(object_checkbox.value)
        if object_frame is not None:
            object_frame.visible = visible
        if mesh_handle is not None:
            mesh_handle.visible = visible
        if dummy_handle is not None:
            dummy_handle.visible = visible

    @pcd_checkbox.on_update
    def _(_) -> None:
        if pcd_handle is not None:
            pcd_handle.visible = bool(pcd_checkbox.value)

    while True:
        start_time = time.time()
        if not paused:
            frame_slider.value = 0 if frame_idx >= n_frames - 1 else frame_idx + 1
        elapsed = time.time() - start_time
        sleep_time = args.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
