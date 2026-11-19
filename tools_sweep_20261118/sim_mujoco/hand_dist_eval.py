"""How far is the brush (nearest mesh point) from the grasp centre, for the true pose and for tracked poses?
Does 'tracked brush far from the hand' flag the lost frames?"""
import glob, json, sys, numpy as np, trimesh
sys.path.insert(0, '.')
pts = np.asarray(trimesh.load('/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj', force='mesh', process=False).vertices)[::20]
TAKES = ['pick_lift', 'occlusion_tour', 'fast_wave', 'heldout_1', 'heldout_2', 'heldout_4', 'heldout_5', 'heldout_6']
def near(T, c):
    return np.min(np.linalg.norm(pts @ T[:3, :3].T + T[:3, 3] - c, axis=1))
D_gt, rows = [], []
for t in TAKES:
    G = np.load(f'takes/{t}/gt/T_world_brush.npy'); C = np.load(f'takes/{t}/replay/hand_center.npy'); TRC = np.loadtxt(f'takes/{t}/gt/T_RC.txt')
    att = np.load(f'takes/{t}/gt/attached.npy')
    D_gt += [near(G[i], C[i]) for i in range(len(G)) if att[i]]
    stems = {int(l.split()[1]): int(l.split()[0]) for l in open(f'takes/{t}/replay/ts_ns.txt') if l.strip()}
    for cfg in ('plain', 'v4np', 'v4npb', 'v4'):
        r = glob.glob(f'runs/{t}_rt/{cfg}/2026*/')
        if not r: continue
        P = np.load(r[0] + 'poses_object_1.npy'); fl = [json.loads(l) for l in open(r[0] + 'frame_log.jsonl')]
        for i, f in enumerate(fl):
            g = stems.get(f['object_capture_ns'][0])
            if g is None or i >= len(P) or not att[g]: continue
            T = TRC @ P[i]
            add = np.linalg.norm((pts @ T[:3, :3].T + T[:3, 3]) - (pts @ G[g][:3, :3].T + G[g][:3, 3]), axis=1).mean()
            rows.append((add, near(T, C[g])))
D_gt = np.array(D_gt); R = np.array(rows)
print(f'true pose, while held: nearest brush point to grasp centre  median {np.median(D_gt)*100:.1f} cm, p99 {np.percentile(D_gt,99)*100:.1f} cm, max {D_gt.max()*100:.1f} cm')
lost = R[:, 0] > 0.05; ok = R[:, 0] < 0.02
for thr in (0.06, 0.08, 0.10, 0.12, 0.15):
    far = R[:, 1] > thr
    print(f'flag "tracked brush > {thr*100:.0f} cm from hand": catches {far[lost].mean()*100:5.1f}% of lost frames, false alarms on {far[ok].mean()*100:4.1f}% of good frames  (lost n={lost.sum()}, good n={ok.sum()})')
