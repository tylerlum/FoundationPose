"""Collect all sim results into results.json for the report page (realtime replays, RT=0.5)."""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
from score_sim import score  # noqa: E402

TAKES = ['pick_lift', 'occlusion_tour', 'fast_wave', 'heldout_1', 'heldout_2', 'heldout_4', 'heldout_5', 'heldout_6']


def one(take_dir, run_glob):
    r = glob.glob(run_glob)
    if not r:
        return None
    try:
        o = score(take_dir, r[0])
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e)[:100])
    b, c = o[1], o[2]
    return dict(ok2cm=round(b['ok2cm'], 1), lost_pct=round(100 * b['lost'] / b['n'], 1), add_med=round(b['add_med_mm'], 1),
                end_mm=round(min(b['end_add_mm'], 9999), 1), recovered=bool(b['end_add_mm'] < 20), served=o['served'],
                n=b['n'], cube_med=round(c['add_med_mm'], 1), curve=[round(min(float(x), 2.0) * 1000, 1) for x in b['_curve']])


out = dict(main={}, cams={}, twocam={})
# main comparison on the current ZED2 view
cfg_dirs = {'plain': 'runs/{t}_rt/plain', 'v4 (old prompt)': 'runs/{t}_rt/v4', 'v4 + new prompt': 'runs/{t}_rt/v4np',
            'v4 + new prompt (repeat)': 'runs/{t}_rt/v4npb', 'v6': 'runs/{t}_rt/v6', 'v7': 'runs/{t}_rt/v7',
            'v8 (old prompt)': 'runs_oldprompt/{t}_rt/v8', 'v8 (old prompt, repeat)': 'runs_oldprompt/{t}_rt/v8b',
            'v8 + new prompt': 'runs/{t}_rt/v8', 'v8 + new prompt (repeat)': 'runs/{t}_rt/v8b',
            'v8 + filter': 'runs/{t}_rt/v8kf', 'v8 + filter (repeat)': 'runs/{t}_rt/v8kfb'}
for name, d in cfg_dirs.items():
    out['main'][name] = {t: one(f'takes/{t}', d.format(t=t) + '/2026*/') for t in TAKES}
for cam, pre in (('zed2_172707', ''), ('zed2_161129', 'zed2_161129__'), ('zed1', 'zed1__')):
    out['cams'][cam] = {}
    for name in ('plain', 'v8', 'v8b'):
        out['cams'][cam][name] = {t: one(f'takes/{pre}{t}', f'runs/{pre}{t}_rt/{name}/2026*/') for t in TAKES}
    vis = {t: float(np.mean(np.load(f'takes/{pre}{t}/gt/vis_brush.npy') < 0.6) * 100) for t in TAKES
           if os.path.exists(f'takes/{pre}{t}/gt/vis_brush.npy')}
    px = {t: float(np.median(np.load(f'takes/{pre}{t}/gt/px_brush.npy'))) for t in TAKES
          if os.path.exists(f'takes/{pre}{t}/gt/px_brush.npy')}
    out['cams'][cam]['_occluded_pct'] = vis
    out['cams'][cam]['_brush_px_med'] = px


def robot_poses(take_dir, run_dir):
    """Per true frame (zero-order hold) robot-frame brush pose of a run, plus its logged colour error (held)."""
    T_RC = np.loadtxt(f'{take_dir}/gt/T_RC.txt')
    stems = {int(l.split()[1]): int(l.split()[0]) for l in open(f'{take_dir}/replay/ts_ns.txt') if l.strip()}
    rows = [json.loads(l) for l in open(f'{run_dir}/frame_log.jsonl')]
    P = np.load(f'{run_dir}/poses_object_1.npy')
    rl = json.load(open(f'{run_dir}/robust_log.json')) if os.path.exists(f'{run_dir}/robust_log.json') else []
    n = len(np.load(f'{take_dir}/gt/T_world_brush.npy'))
    pose = [None] * n
    ce = [np.nan] * n
    last_ce = np.nan
    served = sorted((stems.get(r['object_capture_ns'][0]), i) for i, r in enumerate(rows)
                    if stems.get(r['object_capture_ns'][0]) is not None and i < len(P))
    j = 0
    for g in range(n):
        while j + 1 < len(served) and served[j + 1][0] <= g:
            j += 1
        if served and served[j][0] <= g:
            i = served[j][1]
            pose[g] = T_RC @ P[i]
            if i < len(rl) and rl[i].get('ce') is not None:
                last_ce = rl[i]['ce']
            ce[g] = last_ce
    return pose, ce


pts = None
try:
    import trimesh
    pts = np.asarray(trimesh.load('/juno/u/tylerlum/tracking_meshes_20261026/red_brush/red_brush.obj', force='mesh',
                                  process=False).vertices)[::5]
except Exception:  # noqa: BLE001
    pass


def add_err(T, G):
    if T is None or not np.isfinite(T).all():
        return np.inf
    return float(np.linalg.norm((pts @ T[:3, :3].T + T[:3, 3]) - (pts @ G[:3, :3].T + G[:3, 3]), axis=1).mean())


# two cameras: same motion seen by the current ZED2 and by ZED1 (or the diagonal ZED2), v8 runs
for pair in (('zed2_172707', 'zed1'), ('zed2_172707', 'zed2_161129')):
    pre = {'zed2_172707': '', 'zed2_161129': 'zed2_161129__', 'zed1': 'zed1__'}
    res = {}
    for t in TAKES:
        for rep in ('v8', 'v8b'):
            ra = glob.glob(f'runs/{pre[pair[0]]}{t}_rt/{rep}/2026*/')
            rb = glob.glob(f'runs/{pre[pair[1]]}{t}_rt/{rep}/2026*/')
            if not ra or not rb:
                continue
            G = np.load(f'takes/{t}/gt/T_world_brush.npy')
            pa, ca = robot_poses(f'takes/{pre[pair[0]]}{t}', ra[0])
            pb, cb = robot_poses(f'takes/{pre[pair[1]]}{t}', rb[0])
            # start where both trackers have published (single-camera scores also start at their first pose)
            g0 = max(next((g for g in range(len(G)) if pa[g] is not None), len(G)),
                     next((g for g in range(len(G)) if pb[g] is not None), len(G)))
            G, pa, pb, ca, cb = G[g0:], pa[g0:], pb[g0:], ca[g0:], cb[g0:]
            ea = np.array([add_err(pa[g], G[g]) for g in range(len(G))])
            eb = np.array([add_err(pb[g], G[g]) for g in range(len(G))])
            # selector: lower logged colour error (NaN = no verdict -> keep the previous choice; start with cam A)
            choice, cur = [], 0
            for g in range(len(G)):
                a, b = ca[g], cb[g]
                if np.isfinite(a) and np.isfinite(b):
                    cur = 0 if a <= b else 1
                elif np.isfinite(a) != np.isfinite(b):
                    cur = 0 if np.isfinite(a) else 1
                choice.append(cur)
            es = np.where(np.array(choice) == 0, ea, eb)
            eo = np.minimum(ea, eb)
            for name, e in ((pair[0], ea), (pair[1], eb), ('two cams, colour selector', es), ('two cams, oracle', eo)):
                res.setdefault(name, []).append(dict(take=t, rep=rep, ok2cm=float(np.mean(e < 0.02) * 100),
                                                     lost_pct=float(np.mean(e > 0.05) * 100)))
    out['twocam'][' + '.join(pair)] = {k: dict(ok2cm=round(float(np.mean([x['ok2cm'] for x in v])), 1),
                                               lost_pct=round(float(np.mean([x['lost_pct'] for x in v])), 1), n=len(v))
                                       for k, v in res.items()}
json.dump(out, open('results.json', 'w'))


def summ(d):
    v = [x for x in d.values() if isinstance(x, dict) and 'ok2cm' in x]
    if not v:
        return 'no runs'
    return (f'n={len(v)}  <2cm {np.mean([x["ok2cm"] for x in v]):5.1f}%  lost {np.mean([x["lost_pct"] for x in v]):5.1f}%  '
            f'recovered {sum(x["recovered"] for x in v)}/{len(v)}  cube {np.mean([x["cube_med"] for x in v]):.1f} mm')


print('== current ZED2 view (8 takes, realtime replay)')
for k, v in out['main'].items():
    print(f'  {k:28s} {summ(v)}')
for cam, d in out['cams'].items():
    print(f'== camera {cam}: occluded(<60% visible) frames {np.mean(list(d["_occluded_pct"].values())) if d["_occluded_pct"] else float("nan"):.1f}%, brush px med {np.mean(list(d["_brush_px_med"].values())) if d["_brush_px_med"] else float("nan"):.0f}')
    for k in ('plain', 'v8', 'v8b'):
        print(f'  {k:28s} {summ(d[k])}')
for k, v in out['twocam'].items():
    print('== two cameras', k)
    for n, s in v.items():
        print(f'  {n:28s} n={s["n"]}  <2cm {s["ok2cm"]:5.1f}%  lost {s["lost_pct"]:5.1f}%')
