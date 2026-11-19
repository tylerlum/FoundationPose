"""Viser comparison of depth sources for one RGB frame: point clouds + per-source SAM3D mesh at its FoundationPose pose.

  python viz_compare.py --root DIR --variants fastfs_trt384 zed_neural foundationstereo [--port 8082]
Each DIR/<variant>/ holds rgb/00000.png, depth_m.npy, cam_K.txt, masks/00000.png, and optionally
sam3d_output/mesh/mesh.obj + fp/pose_cam.txt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import trimesh
import tyro
import viser
from PIL import Image
from scipy.spatial.transform import Rotation as R


@dataclass
class Args:
    root: Path
    variants: List[str] = field(default_factory=lambda: ["fastfs_trt384", "zed_neural", "foundationstereo"])
    port: int = 8082
    max_depth_m: float = 3.0
    point_size: float = 0.0025


def wxyz(Rm):
    x, y, z, w = R.from_matrix(Rm).as_quat()
    return np.array([w, x, y, z])


def main(args: Args) -> None:
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/camera", axes_length=0.08, axes_radius=0.003)
    palette = [(255, 80, 80), (80, 160, 255), (80, 220, 120), (230, 180, 60)]
    data = {}
    for i, v in enumerate(args.variants):
        d = args.root / v
        if not (d / "depth_m.npy").exists():
            print(f"skip {v}: no depth_m.npy")
            continue
        rgb = np.array(Image.open(d / "rgb/00000.png").convert("RGB"))
        depth = np.load(d / "depth_m.npy")
        K = np.loadtxt(d / "cam_K.txt")
        mask = np.array(Image.open(d / "masks/00000.png").convert("L")) > 0
        h, w = depth.shape
        vv, uu = np.indices((h, w))
        ok = (depth > 0) & (depth < args.max_depth_m)
        z = depth[ok]
        pts = np.stack([(uu[ok] - K[0, 2]) / K[0, 0] * z, (vv[ok] - K[1, 2]) / K[1, 1] * z, z], 1).astype(np.float32)
        cols = rgb[ok]
        m = mask[ok]
        pc = server.scene.add_point_cloud(f"/camera/{v}/points", points=pts, colors=cols, point_size=args.point_size,
                                          point_shape="rounded", visible=False)
        tint = server.scene.add_point_cloud(f"/camera/{v}/tinted", points=pts[m],
                                            colors=np.tile([palette[i % 4]], (int(m.sum()), 1)).astype(np.uint8),
                                            point_size=args.point_size, point_shape="rounded", visible=False)
        entry = dict(pc=pc, tint=tint, pts=pts, cols=cols, m=m, mesh=None)
        mp, pp = d / "sam3d_output/mesh/mesh.obj", d / "fp/pose_cam.txt"
        if mp.exists() and pp.exists():
            mesh = trimesh.load(mp, force="mesh", process=False)
            T = np.loadtxt(pp)
            fr = server.scene.add_frame(f"/camera/{v}/obj", position=T[:3, 3], wxyz=wxyz(T[:3, :3]),
                                        axes_length=0.05, axes_radius=0.002, visible=False)
            server.scene.add_mesh_simple(f"/camera/{v}/obj/mesh", vertices=np.asarray(mesh.vertices),
                                         faces=np.asarray(mesh.faces), color=palette[i % 4], opacity=0.55)
            entry["mesh"] = fr
            print(f"{v}: SAM3D mesh extents {np.round(mesh.extents, 3)} m, FP pose loaded")
        ext = pts[m].max(0) - pts[m].min(0)
        # Local flatness of the masked surface: RMS of residual to a best-fit plane over the pan region.
        print(f"{v}: {len(pts)} pts, dustpan {int(m.sum())} pts, extent {np.round(ext, 3)} m")
        data[v] = entry

    names = list(data)
    with server.gui.add_folder("Compare"):
        sel = server.gui.add_dropdown("Depth source", options=names, initial_value=names[0])
        overlay_all = server.gui.add_checkbox("Show all sources at once (tinted dustpan)", False)
        only = server.gui.add_checkbox("Dustpan points only", True)
        mesh_cb = server.gui.add_checkbox("SAM3D mesh at FP pose", False)
        psize = server.gui.add_slider("Point size (mm)", 0.5, 8.0, 0.5, args.point_size * 1000)

    def refresh(_=None):
        for v, e in data.items():
            active = v == sel.value
            if only.value:
                e["pc"].points, e["pc"].colors = e["pts"][e["m"]], e["cols"][e["m"]]
            else:
                e["pc"].points, e["pc"].colors = e["pts"], e["cols"]
            e["pc"].point_size = e["tint"].point_size = psize.value / 1000
            e["pc"].visible = active and not overlay_all.value
            e["tint"].visible = overlay_all.value
            if e["mesh"] is not None:
                e["mesh"].visible = mesh_cb.value and (active or overlay_all.value)

    for h in (sel, overlay_all, only, mesh_cb, psize):
        h.on_update(refresh)
    refresh()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        e = data[names[0]]
        c = e["pts"][e["m"]].mean(0) if e["m"].any() else np.array([0, 0, 1.0])
        client.camera.position = (float(c[0]), float(c[1]) - 0.35, float(c[2]) - 0.35)
        client.camera.look_at = tuple(float(x) for x in c)

    print(f"viser: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(Args))
