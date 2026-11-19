"""Appearance check learned from the REAL first frame: HSV histogram of the pixels under the registered silhouette,
then per frame the Bhattacharyya distance (0 same .. 1 different) of the pixels under the tracked silhouette."""
import sys, json
sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
import numpy as np, cv2, torch, trimesh
import nvdiffrast.torch as dr
from Utils import nvdiffrast_render, make_mesh_tensors
mesh = trimesh.load('/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj', process=False)
mesh = mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh
mt = make_mesh_tensors(mesh); glctx = dr.RasterizeCudaContext()
def hist(rgb, m):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    dark = (hsv[..., 2] < 60) & m
    h = cv2.calcHist([hsv], [0, 1], (m & ~dark).astype(np.uint8), [18, 8], [0, 180, 0, 256]).ravel()
    h = np.append(h, dark.sum())          # one extra bin: dark (black bristles / handle)
    return h / max(h.sum(), 1)
def bhat(a, b):
    return float(np.sqrt(max(0.0, 1 - np.sum(np.sqrt(a * b)))))
for d in sys.argv[1:]:
    K = np.loadtxt(f'{d}/K.txt'); P = np.load(f'{d}/poses_object_1.npy'); fps = len(P) / json.load(open(f'{d}/metadata.json'))['duration_s']
    cap = cv2.VideoCapture(f'{d}/raw.mp4'); ref = None; out = []
    for i in range(len(P)):
        ok, f = cap.read()
        if not ok: break
        if i % 3 or not np.isfinite(P[i]).all(): continue
        rgb = cv2.cvtColor(cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        _, dep, _ = nvdiffrast_render(K=K, H=540, W=960, ob_in_cams=torch.as_tensor(P[i][None], dtype=torch.float32, device='cuda'), glctx=glctx, mesh_tensors=mt)
        m = (dep[0] > 0).cpu().numpy()
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # stay inside the outline
        if m.sum() < 200: continue
        h = hist(rgb, m)
        if ref is None: ref = h
        out.append((i / fps, bhat(h, ref)))
    print(d.rstrip('/').split('/')[-1][:15], ' '.join(f'{t:.1f}:{e:.2f}' for t, e in out))
