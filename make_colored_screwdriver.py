"""Split the fin screwdriver into 4 single-colour parts for a 4-filament print (Bambu AMS), and write the matching
coloured tracking mesh, from the same geometry so the print and the mesh agree.

    <python with trimesh + manifold3d> third_party/FoundationPose/make_colored_screwdriver.py [--out_dir DIR]

(manifold3d is in the simtoolreal_ros micromamba env, not in .venv.)

Mesh frame (round_screwdriver_p1_fin_red.obj, metres): x along the tool (handle x -50..50 mm, shaft to 150 mm), the
flag at +y (angle 0) and the fin at +z (angle 90 deg), both on the shaft at x 70..110 mm. Colours:
  - four quadrants of the handle and shaft, boundaries at 45 + k * 90 deg, so the flag sits mid-quadrant A and the fin
    mid-quadrant B: A (0 deg) red, B (90 deg) gray, C (180 deg) gold, D (270 deg) black;
  - flag black, fin gold (each differs from the quadrant it stands on and from the other plate).
The filaments are Bambu PLA Basic Red (#C12E1F), Gray (#8E9089), Gold (#E4BD68) and Black (#000000); the tracking
mesh uses those hex codes until a photo of the print is available.
Outputs in --out_dir:
  - print_<colour>.stl (mm), all in one print frame: axis horizontal, the gap between the plates facing up (+Z), so the
    plates print at 45 deg and the quadrant boundaries are horizontal / vertical (two colours per layer). Load all four
    in Bambu Studio together, answer Yes to 'load as a single object with multiple parts', set a filament per part;
  - round_screwdriver_p1_fin_quad.obj: the tracking mesh (metres, the original frame), outer surface only, vertex
    colours per face;
  - round_screwdriver_p1_fin_quad_preview.png.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh

SRC = '/juno/u/tylerlum/kit_tracking_meshes_20261013/round_screwdriver_p1_fin_red.obj'
COLORS = {'red': (193, 46, 31), 'gray': (142, 144, 137), 'gold': (228, 189, 104), 'black': (0, 0, 0)}   # Bambu hex
QUADRANTS = {0: 'red', 90: 'gray', 180: 'gold', 270: 'black'}   # quadrant centre (deg) -> colour
FLAG_COLOR, FIN_COLOR = 'black', 'gold'


def wedge(center_deg, half_deg=45.0, r=1.0, x0=-1.0, x1=1.0):
    """A solid 2 * half_deg sector about the x axis (convex for <= 180 deg)."""
    a = np.radians(np.linspace(center_deg - half_deg, center_deg + half_deg, 33))
    ring = np.c_[np.zeros_like(a), r * np.cos(a), r * np.sin(a)]
    pts = np.vstack([[[x, 0, 0], *(ring + [x, 0, 0])] for x in (x0, x1)])
    return trimesh.convex.convex_hull(pts)


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out_dir', default='/juno/u/tylerlum/kit_tracking_meshes_20261013/screwdriver_fin_quad_print')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src = trimesh.load(args.src, process=False)
    mesh = trimesh.Trimesh(src.vertices, src.faces, process=True)
    assert mesh.is_watertight, 'the source mesh must be watertight'
    I = lambda a, b: trimesh.boolean.intersection([a, b], engine='manifold')
    D = lambda a, b: trimesh.boolean.difference([a, b], engine='manifold')
    U = lambda ms: trimesh.boolean.union(ms, engine='manifold')

    # The plates: everything beyond the shaft (r = 5 mm) on their side, along x 60..120 mm.
    flag = I(mesh, box((0.060, 0.005, -0.008), (0.120, 0.030, 0.008)))
    rest = D(mesh, flag)
    fin = I(rest, box((0.060, -0.008, 0.005), (0.120, 0.008, 0.030)))
    body = D(rest, fin)
    quads = {c: I(body, wedge(deg)) for deg, c in QUADRANTS.items()}

    parts = {c: [q] for c, q in quads.items()}
    parts[FLAG_COLOR].append(flag)
    parts[FIN_COLOR].append(fin)
    parts = {c: (U(ms) if len(ms) > 1 else ms[0]) for c, ms in parts.items()}
    vol = sum(p.volume for p in parts.values())
    print(f'volume: parts {vol * 1e6:.3f} cm3, source {mesh.volume * 1e6:.3f} cm3; flag {flag.volume * 1e6:.3f}, '
          f'fin {fin.volume * 1e6:.3f}')
    for c, p in parts.items():
        print(f'  {c:6s} {p.volume * 1e6:6.2f} cm3  watertight {p.is_watertight}')

    # Print frame (mm): x (tool axis) -> +X; the plates' bisector (45 deg in y-z) -> +Z (up); on the bed.
    up = np.array([0.0, np.cos(np.radians(45)), np.sin(np.radians(45))])
    side = np.cross(up, [1.0, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([[1.0, 0.0, 0.0], side, up])   # rows: new X, Y, Z in mesh coordinates
    T[:3, :3] *= 1000.0
    zmin = min((p.vertices @ T[:3, :3].T)[:, 2].min() for p in parts.values())
    T[2, 3] = -zmin
    for c, p in parts.items():
        q = p.copy()
        q.apply_transform(T)
        q.export(out / f'print_{c}.stl')

    # Tracking mesh: the parts' faces on the outer surface (drop the internal cut walls), coloured per part.
    tris, cols = [], []
    # The cut walls lie on the quadrant planes (through the x axis at 45 + k * 90 deg) or on the plate cuts (y = 5 mm
    # under the flag, z = 5 mm under the fin); no outer face does. (mesh.contains() misfires on axis-aligned faces.)
    def on_cut(tri):
        v = tri.reshape(-1, 3)
        planes = [np.abs(v[:, 1] * np.cos(np.radians(a)) + v[:, 2] * np.sin(np.radians(a))) for a in (135, 45)]
        in_x = ((v[:, 0] > 0.060 - 1e-9) & (v[:, 0] < 0.120 + 1e-9)).reshape(-1, 3).all(1)   # the plate boxes only:
        on = [(d.reshape(-1, 3) < 1e-7).all(1) for d in planes]                                # the tip's flats sit
        on += [(np.abs(v[:, k] - 0.005).reshape(-1, 3) < 1e-7).all(1) & in_x for k in (1, 2)]  # at y = 5 mm too
        return np.any(on, axis=0)
    for c, p in parts.items():
        outside = ~on_cut(p.triangles)
        tris.append(p.triangles[outside])
        cols.append(np.tile([*COLORS[c], 255], (outside.sum(), 1)))
    tris, cols = np.vstack(tris), np.vstack(cols).astype(np.uint8)
    track = trimesh.Trimesh(tris.reshape(-1, 3), np.arange(3 * len(tris)).reshape(-1, 3), process=False)
    track.visual = trimesh.visual.ColorVisuals(track, vertex_colors=np.repeat(cols, 3, axis=0))
    print(f'tracking mesh: {len(track.faces)} faces, area {track.area * 1e4:.2f} cm2 (source {mesh.area * 1e4:.2f}), '
          f'bounds match {np.allclose(track.bounds, mesh.bounds, atol=1e-6)}')
    track.export(out / 'round_screwdriver_p1_fin_quad.obj')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fc = cols[:, :3] / 255.0
    light = np.array([0.3, 0.4, 0.85]) / np.linalg.norm([0.3, 0.4, 0.85])
    fig = plt.figure(figsize=(16, 4.5))
    for i, deg in enumerate((45, 135, 225, 315)):
        d = np.array([0.0, np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        R = np.stack([d, np.cross([-1.0, 0, 0], d), [-1.0, 0, 0]])
        n = track.face_normals @ R.T
        vis = n[:, 0] > -0.1
        sh = 0.55 + 0.45 * np.clip(n @ light, 0, 1)
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        ax.add_collection3d(Poly3DCollection((track.triangles @ R.T)[vis], facecolors=fc[vis] * sh[vis, None],
                                             edgecolor='none'))
        ax.set_xlim(-0.08, 0.08); ax.set_ylim(-0.08, 0.08); ax.set_zlim(-0.15, 0.05)
        ax.view_init(elev=8, azim=0); ax.set_axis_off(); ax.set_title(f'seen from {deg} deg')
    plt.tight_layout()
    plt.savefig(out / 'round_screwdriver_p1_fin_quad_preview.png', dpi=80)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
