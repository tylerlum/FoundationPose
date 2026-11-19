"""Does rendered-colour agreement separate right from wrong brush poses? Offline, with ground truth.
python color_signal.py TAKE RUN_DIR [TAKE RUN_DIR ...]   (drl env)
For every 3rd frame: colour error of the tracked pose and of the true pose (visible, unoccluded pixels), and ADD."""
import json
import sys
import glob

sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
import cv2  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402
from Utils import nvdiffrast_render, make_mesh_tensors  # noqa: E402

M = '/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj'
mesh = trimesh.load(M, process=False)
if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)
mt = make_mesh_tensors(mesh)
pts = np.asarray(mesh.vertices)[::5]
glctx = dr.RasterizeCudaContext()


def metrics(T, rgb, depth, K, tol=0.015):
    col, dep, _ = nvdiffrast_render(K=K, H=540, W=960, ob_in_cams=torch.as_tensor(T[None], dtype=torch.float32,
                                    device='cuda'), glctx=glctx, mesh_tensors=mt)
    col, dep = col[0].clamp(0, 1).cpu().numpy(), dep[0].cpu().numpy()
    vis = (dep > 0) & (depth > 0.05) & (depth >= dep - tol)
    n = int(vis.sum())
    if n < 200:
        return np.nan, np.nan, n
    obs = rgb.astype(np.float32) / 255
    e_rgb = float(np.abs(col[vis] - obs[vis]).mean())
    # hue/brightness-robust: compare chromaticity + a dark/bright class
    def chrom(x):
        s = x.sum(-1, keepdims=True) + 1e-3
        return np.concatenate([x / s, (x.mean(-1, keepdims=True) > 0.25)], -1)
    e_chr = float(np.abs(chrom(col[vis]) - chrom(obs[vis])).mean())
    return e_rgb, e_chr, n


rows = []
args = sys.argv[1:]
for take, run in zip(args[::2], args[1::2]):
    K = np.loadtxt(f'{take}/replay/cam_K.txt')
    st = {int(l.split()[1]): l.split()[0] for l in open(f'{take}/replay/ts_ns.txt') if l.strip()}
    G = np.load(f'{take}/gt/T_cam_brush.npy')
    P = np.load(f'{run}/poses_object_1.npy')
    fl = [json.loads(l) for l in open(f'{run}/frame_log.jsonl')]
    for i in range(0, len(fl), 3):
        stem = st.get(fl[i]['object_capture_ns'][0])
        if stem is None:
            continue
        rgb = cv2.cvtColor(cv2.imread(f'{take}/replay/rgb/{stem}.png'), cv2.COLOR_BGR2RGB)
        depth = np.load(f'{take}/replay/depth/{stem}.npy')
        g = G[int(stem)]
        T = P[i]
        add = np.linalg.norm((pts @ T[:3, :3].T + T[:3, 3]) - (pts @ g[:3, :3].T + g[:3, 3]), axis=1).mean()
        rows.append((add, *metrics(T, rgb, depth, K), *metrics(g, rgb, depth, K)))
R = np.array(rows, float)
add = R[:, 0]
for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 9)):
    s = R[(add >= lo) & (add < hi)]
    if len(s):
        print(f'ADD {lo*100:4.0f}-{hi*100:4.0f} cm: n {len(s):4d}  rgb err med {np.nanmedian(s[:, 1]):.3f} '
              f'p90 {np.nanpercentile(s[:, 1], 90):.3f}  chrom med {np.nanmedian(s[:, 2]):.3f} '
              f'p90 {np.nanpercentile(s[:, 2], 90):.3f}  vis px med {np.nanmedian(s[:, 3]):.0f}')
print(f'true pose: rgb err med {np.nanmedian(R[:, 4]):.3f} p90 {np.nanpercentile(R[:, 4], 90):.3f}  chrom med '
      f'{np.nanmedian(R[:, 5]):.3f} p90 {np.nanpercentile(R[:, 5], 90):.3f}')
np.save('color_signal_rows.npy', R)
