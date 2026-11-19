"""Split vertex-coloured OBJs into one OBJ per colour (MuJoCo shows no vertex colours). Writes assets_kit/<name>/part_k.obj
and parts.json [(file, rgba)]; prints extents. python split_colours.py OBJ ..."""
import json, sys, os
import numpy as np, trimesh
for p in sys.argv[1:]:
    m = trimesh.load(p, process=False, force='mesh')
    name = os.path.splitext(os.path.basename(p))[0]
    out = os.path.join(os.path.dirname(p), name); os.makedirs(out, exist_ok=True)
    vc = np.asarray(m.visual.vertex_colors)[:, :3].astype(int) if m.visual.kind == 'vertex' else np.full((len(m.vertices), 3), 128)
    fc = vc[m.faces].reshape(len(m.faces), 3, 3)
    key = [tuple(x) for x in fc[:, 0]]          # first vertex's colour per face
    cols = sorted(set(key))
    parts = []
    for k, c in enumerate(cols):
        sel = np.array([kk == c for kk in key])
        sub = m.submesh([np.nonzero(sel)[0]], append=True)
        f = os.path.join(out, f'part_{k}.obj'); sub.export(f)
        parts.append((f, [c[0] / 255, c[1] / 255, c[2] / 255, 1.0]))
    json.dump(parts, open(os.path.join(out, 'parts.json'), 'w'))
    print(f'{name}: {len(m.vertices)} verts, extents {np.round(m.extents, 3)} m, bounds x {np.round(m.bounds[:, 0], 3)}, colours {cols}')
