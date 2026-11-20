"""Two cameras, sweep takes: single-camera v4hc, cross-camera restart (v4hcx), and a real-safe per-frame selector."""
import sys, glob, json, numpy as np, trimesh
sys.path.insert(0, '.')
from score_sim import score
SW = ['pick_lift','occlusion_tour','fast_wave','heldout_1','heldout_2','heldout_4','heldout_5','heldout_6']
pts = np.asarray(trimesh.load('/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj', force='mesh', process=False).vertices)[::5]
def ok(take, cfg):
    r = glob.glob(f'runs/{take}_rt/{cfg}/2026*/')
    return score(f'takes/{take}', r[0])[1] if r else None
def held_series(take, cfg):
    """per true frame (zero-order hold): robot-frame pose, and a 'trust' signal: hand distance <= 10 cm and SAM2 cover."""
    r = glob.glob(f'runs/{take}_rt/{cfg}/2026*/')[0]
    TRC = np.loadtxt(f'takes/{take}/gt/T_RC.txt'); P = np.load(r + 'poses_object_1.npy')
    rows = [json.loads(l) for l in open(r + 'frame_log.jsonl')]; rl = json.load(open(r + 'robust_log.json'))
    st = {int(l.split()[1]): int(l.split()[0]) for l in open(f'takes/{take}/replay/ts_ns.txt') if l.strip()}
    n = len(np.load(f'takes/{take}/gt/T_world_brush.npy'))
    pose, trust = [None] * n, np.full(n, -1.0)
    served = sorted((st[x['object_capture_ns'][0]], i) for i, x in enumerate(rows) if i < len(P) and x['object_capture_ns'][0] in st)
    j = 0; last = -1.0
    for g in range(n):
        while j + 1 < len(served) and served[j + 1][0] <= g: j += 1
        if served and served[j][0] <= g:
            i = served[j][1]; pose[g] = TRC @ P[i]
            if i < len(rl):
                hd = rl[i].get('hand_d'); cv = rl[i].get('cover')
                if hd is not None:
                    last = (1.0 if hd <= 0.10 else 0.0) + (cv if isinstance(cv, (int, float)) else 0.5)
            trust[g] = last
    return pose, trust
def adds(T, G):
    if T is None or not np.isfinite(T).all(): return np.inf
    return np.linalg.norm((pts @ T[:3, :3].T + T[:3, 3]) - (pts @ G[:3, :3].T + G[:3, 3]), axis=1).mean()
res = {k: [] for k in ('ZED2 v4hc', 'ZED1 v4hc', 'ZED2 v4hcx (restart from ZED1)', 'ZED1 v4hcx (restart from ZED2)', 'fused v4hcx, real-safe selector', 'fused v4hcx, oracle')}
for t in SW:
    for k, (tk, c) in {'ZED2 v4hc': (t, 'v4hc'), 'ZED1 v4hc': ('zed1__' + t, 'v4hc'), 'ZED2 v4hcx (restart from ZED1)': (t, 'v4hcx'), 'ZED1 v4hcx (restart from ZED2)': ('zed1__' + t, 'v4hcx')}.items():
        o = ok(tk, c)
        if o: res[k].append(o['ok2cm'])
    try:
        pa, ta = held_series(t, 'v4hcx'); pb, tb = held_series('zed1__' + t, 'v4hcx')
    except IndexError:
        continue
    G = np.load(f'takes/{t}/gt/T_world_brush.npy')
    g0 = max(next(g for g in range(len(G)) if pa[g] is not None), next(g for g in range(len(G)) if pb[g] is not None))
    ea = np.array([adds(pa[g], G[g]) for g in range(g0, len(G))]); eb = np.array([adds(pb[g], G[g]) for g in range(g0, len(G))])
    sel = np.where(tb[g0:] > ta[g0:], eb, ea)          # ZED2 unless ZED1 is more trusted
    res['fused v4hcx, real-safe selector'].append(float((sel < 0.02).mean() * 100))
    res['fused v4hcx, oracle'].append(float((np.minimum(ea, eb) < 0.02).mean() * 100))
for k, v in res.items():
    print(f'{k:36s} n={len(v)}  frames within 2 cm {np.mean(v):5.1f}%   per take: {" ".join(f"{x:.0f}" for x in v)}')
