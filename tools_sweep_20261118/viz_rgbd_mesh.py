"""Viser view of one RGB-D frame (camera frame), its SAM mask, and meshes posed in that frame.

Based on tylerlum/FoundationPose visualize.py (2026-10-21_RobotTrackers_bohg-ws-16), single-frame version.
  python viz_rgbd_mesh.py --video_dir DIR [--mesh NAME=MESH.obj[@POSE.txt] ...] [--port 8081]
DIR holds rgb/*.png, depth/*.png (uint16 mm), cam_K.txt, optional masks/*.png. POSE.txt is a 4x4 T_cam_obj; without
it the mesh is shown as-is (already in camera frame, e.g. SAM3D's posed mesh).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh
import tyro
import viser
from PIL import Image
from scipy.spatial.transform import Rotation as R


@dataclass
class Args:
    video_dir: Path
    mesh: List[str] = field(default_factory=list)
    """NAME=MESH_PATH[@POSE_TXT], repeatable."""
    frame_idx: int = 0
    port: int = 8081
    max_depth_m: float = 3.0
    point_size: float = 0.002


def wxyz(Rm: np.ndarray) -> np.ndarray:
    x, y, z, w = R.from_matrix(Rm).as_quat()
    return np.array([w, x, y, z])


def backproject(depth_m, K, rgb, max_depth_m):
    h, w = depth_m.shape
    v, u = np.indices((h, w))
    z = depth_m.reshape(-1)
    ok = (z > 0) & (z < max_depth_m)
    u, v, z = u.reshape(-1)[ok], v.reshape(-1)[ok], z[ok]
    pts = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1).astype(np.float32)
    return pts, rgb.reshape(-1, 3)[ok], ok


def load_mesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, process=False)
    return m.dump(concatenate=True) if isinstance(m, trimesh.Scene) else m


def main(args: Args) -> None:
    d = args.video_dir
    rgb_path = sorted((d / "rgb").glob("*.png"))[args.frame_idx]
    depth_path = sorted((d / "depth").glob("*.png"))[args.frame_idx]
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
    K = np.loadtxt(d / "cam_K.txt").reshape(3, 3)
    mask_paths = sorted((d / "masks").glob("*.png")) if (d / "masks").exists() else []
    mask = (np.array(Image.open(mask_paths[args.frame_idx]).convert("L")) > 0) if mask_paths else None
    print(f"rgb {rgb_path}\ndepth {depth_path}\nmask {mask_paths[args.frame_idx] if mask_paths else None}")

    pts, cols, ok = backproject(depth, K, rgb, args.max_depth_m)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")  # camera frame: y down

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.0, -0.3, -0.6)
        client.camera.look_at = (0.0, 0.0, float(np.median(depth[depth > 0])) if (depth > 0).any() else 1.0)

    server.scene.add_frame("/camera", axes_length=0.08, axes_radius=0.003)
    h, w = depth.shape
    frustum = server.scene.add_camera_frustum("/camera/frustum", fov=2 * np.arctan2(h / 2, K[1, 1]), aspect=w / h, scale=0.15,
                                    image=rgb[::2, ::2])
    pcd = server.scene.add_point_cloud("/camera/rgbd", points=pts, colors=cols, point_size=args.point_size,
                                       point_shape="rounded", precision="float32")
    gui = {}
    with server.gui.add_folder("Scene"):
        gui["pcd"] = server.gui.add_checkbox("RGB-D points", True)
        gui["psize"] = server.gui.add_slider("Point size (mm)", 0.5, 8.0, 0.5, args.point_size * 1000)
        gui["frustum"] = server.gui.add_checkbox("Camera image", True)

    if mask is not None:
        m_ok = mask.reshape(-1)[ok]
        obj_pts = pts[m_ok]
        mask_pcd = server.scene.add_point_cloud("/camera/mask_points", points=obj_pts,
                                                colors=np.tile([[40, 220, 60]], (len(obj_pts), 1)).astype(np.uint8),
                                                point_size=args.point_size * 1.5, point_shape="rounded", visible=False)
        ext = obj_pts.max(0) - obj_pts.min(0) if len(obj_pts) else np.zeros(3)
        print(f"masked points: {len(obj_pts)}; extent (m) {np.round(ext, 3)}; median depth {np.median(obj_pts[:, 2]):.3f}")
        with server.gui.add_folder("Mask"):
            gui["mask"] = server.gui.add_checkbox("Mask points (green)", False)
            gui["only"] = server.gui.add_checkbox("Only masked points", False)

            @gui["mask"].on_update
            def _(_):
                mask_pcd.visible = gui["mask"].value

            @gui["only"].on_update
            def _(_):
                pcd.points, pcd.colors = (obj_pts, cols[m_ok]) if gui["only"].value else (pts, cols)

    palette = [(235, 180, 80), (80, 160, 235), (230, 90, 200), (120, 220, 120)]
    with server.gui.add_folder("Meshes"):
        for i, spec in enumerate(args.mesh):
            name, rest = spec.split("=", 1)
            mesh_path, _, pose_path = rest.partition("@")
            mesh = load_mesh(mesh_path)
            T = np.loadtxt(pose_path).reshape(4, 4) if pose_path else np.eye(4)
            print(f"mesh {name}: {mesh_path} extents (m) {np.round(mesh.extents, 3)} pose {pose_path or 'identity'}")
            fr = server.scene.add_frame(f"/camera/{name}", position=T[:3, 3], wxyz=wxyz(T[:3, :3]),
                                        axes_length=0.05, axes_radius=0.002)
            textured = server.scene.add_mesh_trimesh(f"/camera/{name}/textured", mesh=mesh)
            solid = server.scene.add_mesh_simple(f"/camera/{name}/solid", vertices=np.asarray(mesh.vertices),
                                                 faces=np.asarray(mesh.faces), color=palette[i % len(palette)],
                                                 opacity=0.6, visible=False)
            show = server.gui.add_checkbox(f"{name}", True)
            solid_cb = server.gui.add_checkbox(f"{name}: solid see-through", False)

            def upd(_, show=show, solid_cb=solid_cb, fr=fr, textured=textured, solid=solid):
                fr.visible = show.value
                textured.visible = show.value and not solid_cb.value
                solid.visible = show.value and solid_cb.value
            show.on_update(upd)
            solid_cb.on_update(upd)

    @gui["pcd"].on_update
    def _(_):
        pcd.visible = gui["pcd"].value

    @gui["psize"].on_update
    def _(_):
        pcd.point_size = gui["psize"].value / 1000

    @gui["frustum"].on_update
    def _(_):
        frustum.visible = gui["frustum"].value

    print(f"viser: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(Args))
