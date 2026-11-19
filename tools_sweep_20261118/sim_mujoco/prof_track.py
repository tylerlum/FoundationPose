"""cProfile of FoundationPose track_one (host-side hot spots). python prof_track.py FP_DIR"""
import sys, time, cProfile, pstats, io
FPD = sys.argv[1]
sys.path.insert(0, FPD)
import os; os.chdir(FPD)
import numpy as np, torch, trimesh, cv2
import nvdiffrast.torch as dr
from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
S = __import__('os').environ.get('FPSIM_ROOT', '.') + '/takes/heldout_4/replay'
mesh = trimesh.load('/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj', process=False)
mesh = mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh
glctx = dr.RasterizeCudaContext()
trt = os.path.join(FPD, 'weights', '2023-10-28-18-33-37', 'refine_net_fp16.plan')
ref = PoseRefinePredictor(trt_engine_path=trt) if os.path.exists(trt) else PoseRefinePredictor()
print('TRT refiner:', os.path.exists(trt))
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=ScorePredictor(), refiner=ref, debug=0, glctx=glctx, debug_dir=__import__('os').environ.get('FPSIM_ROOT', '.') + '/fpdebug')
K = np.loadtxt(S + '/cam_K.txt')
fr = [(cv2.cvtColor(cv2.imread(f'{S}/rgb/{i:05d}.png'), cv2.COLOR_BGR2RGB), np.load(f'{S}/depth/{i:05d}.npy')) for i in range(60)]
m = cv2.imread(S + '/masks_1/00000.png', 0) > 0
est.register(K=K, rgb=fr[0][0], depth=fr[0][1], ob_mask=m, iteration=5)
for i in range(1, 10): est.track_one(rgb=fr[i][0], depth=fr[i][1], K=K, iteration=2)
torch.cuda.synchronize()
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
for i in range(10, 60): est.track_one(rgb=fr[i][0], depth=fr[i][1], K=K, iteration=2)
torch.cuda.synchronize(); pr.disable()
print(f'track_one mean {(time.time()-t0)/50*1000:.1f} ms')
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(22); print(s.getvalue()[:6000])
