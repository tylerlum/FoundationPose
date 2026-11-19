import time, numpy as np, torch, trimesh, cv2, sys
sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
import nvdiffrast.torch as dr
from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
import robust_tool
S=__import__('os').environ.get('FPSIM_ROOT', '.') + '/takes/fast_wave/replay'
mesh = trimesh.load('/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj', process=False)
mesh = mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug=0, glctx=glctx, debug_dir=__import__('os').environ.get('FPSIM_ROOT', '.') + '/fpdebug')
K = np.loadtxt(S+'/cam_K.txt'); rgb = cv2.cvtColor(cv2.imread(S+'/rgb/00000.png'), cv2.COLOR_BGR2RGB); depth = np.load(S+'/depth/00000.npy'); m = cv2.imread(S+'/masks_1/00000.png',0)>0
chk = robust_tool.ToolCheck(mesh, K, 960, 540, glctx=glctx)
def t(f, n=10):
    f(); torch.cuda.synchronize(); a=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-a)/n*1000
for it in (5, 2):
    print(f'T register (252 hypotheses, refine iter {it}): {t(lambda: est.register(K=K, rgb=rgb, depth=depth, ob_mask=m, iteration=it), 3):.0f} ms', flush=True)
est.register(K=K, rgb=rgb, depth=depth, ob_mask=m, iteration=5)
print(f'T track_one (iter 2): {t(lambda: est.track_one(rgb=rgb, depth=depth, K=K, iteration=2)):.1f} ms', flush=True)
T = est.track_one(rgb=rgb, depth=depth, K=K, iteration=2)
print(f'T colour check (render + compare): {t(lambda: chk.color_err(rgb, depth, T)):.1f} ms', flush=True)
print(f'T silhouette render: {t(lambda: chk.silhouette(T)):.1f} ms', flush=True)
print(f'T depth fit: {t(lambda: chk.depth_fit(depth, T)):.1f} ms', flush=True)
