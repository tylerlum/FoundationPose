"""Is a constant-velocity start pose (fpp_tracking.PoseFilter) closer to the truth than the last pose?
Ground-truth brush trajectories (sim, real-speed) + FoundationPose-like jitter, every `step` frames."""
import sys, glob, numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
from fpp_tracking import PoseFilter
rng = np.random.RandomState(0)
trajs = [np.load(f) for f in sorted(glob.glob('takes/*/gt/T_cam_brush.npy')) if '__' not in f]
def err(A, B):
    return np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000, np.degrees(R.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude())
def run(alpha, max_v, max_w, step, noise_mm=3, noise_deg=1.5):
    et, er = [], []
    for G in trajs:
        f = None if alpha is None else PoseFilter(alpha, max_v, np.radians(max_w))
        prev = None
        for i in range(0, len(G) - step, step):
            obs = G[i].copy()
            obs[:3, 3] += rng.normal(0, noise_mm / 1000 / np.sqrt(3), 3)
            obs[:3, :3] = R.from_rotvec(rng.normal(0, np.radians(noise_deg) / np.sqrt(3), 3)).as_matrix() @ obs[:3, :3]
            if f is None:
                pred = obs
            else:
                f.update(obs); pred = f.predict()
            a, b = err(pred, G[i + step]); et.append(a); er.append(b)
    et, er = np.array(et), np.array(er)
    return np.median(et), np.percentile(et, 95), np.median(er), np.percentile(er, 95)
for step in (1, 2, 3):
    print(f'--- every {step} frame(s) ({30/step:.0f} Hz tracking)')
    b = run(None, 0, 0, step)
    print(f'  last pose (no filter)        start err trans med {b[0]:5.1f} p95 {b[1]:5.1f} mm   rot med {b[2]:4.1f} p95 {b[3]:4.1f} deg')
    for alpha in (0.3, 0.5, 0.8, 1.0):
        for max_v, max_w in ((0.05, 20), (0.1, 40)):
            r = run(alpha, max_v, max_w, step)
            print(f'  CV alpha {alpha:.1f} vmax {max_v} wmax {max_w:2d}  start err trans med {r[0]:5.1f} p95 {r[1]:5.1f} mm   rot med {r[2]:4.1f} p95 {r[3]:4.1f} deg')
