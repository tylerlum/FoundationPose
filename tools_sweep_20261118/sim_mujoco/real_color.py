"""Colour-check value on REAL takes (no depth saved: no occlusion test, so occluded frames read high).
python real_color.py TAKE_DIR ...  -> colour error every 3rd frame with time; brush pose from the recorded track."""
import sys, json
sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
import numpy as np, cv2, torch, trimesh
import nvdiffrast.torch as dr
from Utils import nvdiffrast_render, make_mesh_tensors
mesh = trimesh.load('/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj', process=False)
mesh = mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh
mt = make_mesh_tensors(mesh); glctx = dr.RasterizeCudaContext()
for d in sys.argv[1:]:
    K = np.loadtxt(f'{d}/K.txt'); P = np.load(f'{d}/poses_object_1.npy'); m = json.load(open(f'{d}/metadata.json'))
    fps = len(P) / m['duration_s']
    cap = cv2.VideoCapture(f'{d}/raw.mp4'); out = []
    for i in range(len(P)):
        ok, f = cap.read()
        if not ok: break
        if i % 3 or not np.isfinite(P[i]).all(): continue
        rgb = cv2.cvtColor(cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        col, dep, _ = nvdiffrast_render(K=K, H=540, W=960, ob_in_cams=torch.as_tensor(P[i][None], dtype=torch.float32, device='cuda'), glctx=glctx, mesh_tensors=mt)
        vis = dep[0] > 0
        n = int(vis.sum())
        e = float((col[0].clamp(0, 1) - torch.as_tensor(rgb, device='cuda').float() / 255).abs().mean(-1)[vis].mean()) if n > 200 else np.nan
        out.append((i / fps, e, n))
    import os as _os
    _js = _os.environ.get('REAL_JSON')
    if _js:
        _all = json.load(open(_js)) if _os.path.exists(_js) else {}
        _all.setdefault('mesh_colour', {})[d.rstrip('/').split('/')[-1][:15]] = [[round(o[0], 2), round(float(o[1]), 3)] for o in out]
        json.dump(_all, open(_js, 'w'))
    print(d.rstrip('/').split('/')[-1][:15], 't(s):ce ', ' '.join(f'{t:.1f}:{e:.2f}' for t, e, n in out))
