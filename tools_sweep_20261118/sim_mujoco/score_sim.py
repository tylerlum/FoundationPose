"""Score tracker runs on synthetic takes against MuJoCo ground truth.

  python score_sim.py TAKE_DIR RUN_DIR [RUN_DIR ...]     (RUN_DIR = the tracker's take folder)
Per run: brush ADD (mean distance of mesh points, true vs tracked pose), origin error, rotation error;
% frames with ADD < 2 cm, frames 'lost' (ADD > 5 cm), longest lost streak, recovery after the worst-occluded part,
and the same for the cube. Frames are matched by capture stamp (ts_ns.txt <-> frame_log.jsonl).
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

M = '/juno/u/tylerlum/tracking_meshes_20261026'
KM = '/juno/u/tylerlum/kit_tracking_meshes_20261013'
MESHES = {'sweep': (f'{M}/red_brush/red_brush.obj', f'{M}/cube_29mm_gold.obj'),
          'screw': (f'{KM}/screwdriver_fin_quad_print/round_screwdriver_p1_fin_quad.obj', f'{KM}/screw_p1_head_bluegrey.obj'),
          'wrench': (f'{KM}/open_wrench_18_red.obj', f'{KM}/bolt_thick_head_bluegrey.obj')}
_PTS = {}


def pts_for(take, k):
    task = open(f'{take}/gt/task.txt').read().strip() if Path(f'{take}/gt/task.txt').exists() else 'sweep'
    if (task, k) not in _PTS:
        v = np.asarray(trimesh.load(MESHES[task][k - 1], force='mesh', process=False).vertices)
        step = (5 if k == 1 else 1) if task == 'sweep' else max(1, len(v) // 2400)   # sweep: same points as before
        _PTS[(task, k)] = v[::step]
    return _PTS[(task, k)]


def add(Ta, Tb, pts):
    a = pts @ Ta[:3, :3].T + Ta[:3, 3]
    b = pts @ Tb[:3, :3].T + Tb[:3, 3]
    return np.linalg.norm(a - b, axis=1).mean()


def streak(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def score(take, run):
    take, run = Path(take), Path(run)
    stems = {}
    for line in (take / 'replay' / 'ts_ns.txt').read_text().split('\n'):
        if line.strip():
            s, ns = line.split()
            stems[int(ns)] = int(s)
    rows = [json.loads(l) for l in open(run / 'frame_log.jsonl')]
    gt = {1: np.load(take / 'gt' / 'T_cam_brush.npy'), 2: np.load(take / 'gt' / 'T_cam_cube.npy')}
    vis = np.load(take / 'gt' / 'vis_brush.npy')
    out = {}
    for k in (1, 2):
        P = np.load(run / f'poses_object_{k}.npy')
        idx = [stems.get(r['object_capture_ns'][k - 1] if 'object_capture_ns' in r else r['capture_ns']) for r in rows]
        # Zero-order hold: every true frame from the first tracked one on is scored with the newest pose published
        # for a frame at or before it (what a policy sees while the tracker skips frames).
        served = sorted((gi, i) for i, gi in enumerate(idx) if gi is not None and i < len(P))
        hold = []
        j = 0
        for g in range(served[0][0], len(gt[k])):
            while j + 1 < len(served) and served[j + 1][0] <= g:
                j += 1
            hold.append((g, served[j][1]))
        e_add, e_t, e_r, v = [], [], [], []
        for gi, i in hold:
            T = P[i]
            if not np.isfinite(T).all():
                e_add.append(np.inf); e_t.append(np.inf); e_r.append(np.inf); v.append(vis[gi])
                continue
            G = gt[k][gi]
            e_add.append(add(T, G, pts_for(take, k)))
            e_t.append(np.linalg.norm(T[:3, 3] - G[:3, 3]))
            ang = np.degrees(R.from_matrix(G[:3, :3].T @ T[:3, :3]).magnitude())
            if k == 2:  # cube: 24-fold symmetry -> smallest angle to a cube rotation
                syms = R.create_group('O')
                ang = np.degrees(min((R.from_matrix(G[:3, :3].T @ T[:3, :3]) * s).magnitude() for s in syms))
                e_add[-1] = e_t[-1]
            e_r.append(ang)
            v.append(vis[gi])
        e_add, e_t, e_r, v = map(np.asarray, (e_add, e_t, e_r, v))
        lost = e_add > 0.05
        n = len(e_add)
        out[k] = dict(n=n, add_med_mm=float(np.median(e_add) * 1000), add_p90_mm=float(np.percentile(e_add, 90) * 1000),
                      t_med_mm=float(np.median(e_t) * 1000), r_med_deg=float(np.median(e_r)),
                      ok2cm=float((e_add < 0.02).mean() * 100), lost=int(lost.sum()), lost_streak=streak(lost),
                      end_add_mm=float(np.median(e_add[-15:]) * 1000),
                      occl_add_mm=float(np.median(e_add[v < 0.6]) * 1000) if k == 1 and (v < 0.6).any() else float('nan'))
        out[k]['_curve'] = e_add
    meta = json.load(open(run / 'metadata.json'))
    out['fps'] = len(rows) / max(meta.get('duration_s') or 1, 1e-6)
    out['served'] = len(rows)
    return out


if __name__ == '__main__':
    take = sys.argv[1]
    print(f'{"run":28s} {"obj":5s} {"n":>4s} {"ADDmed":>7s} {"ADDp90":>7s} {"tmed":>6s} {"rmed":>6s} {"<2cm%":>6s} '
          f'{"lost":>5s} {"streak":>6s} {"occl":>7s} {"end":>7s}')
    for run in sys.argv[2:]:
        o = score(take, run)
        name = f'{Path(run).parent.name} ({o["served"]} fr)'
        for k, nm in ((1, 'brush'), (2, 'cube')):
            s = o[k]
            print(f'{name:28s} {nm:5s} {s["n"]:4d} {s["add_med_mm"]:7.1f} {s["add_p90_mm"]:7.1f} {s["t_med_mm"]:6.1f} '
                  f'{s["r_med_deg"]:6.1f} {s["ok2cm"]:6.1f} {s["lost"]:5d} {s["lost_streak"]:6d} {s["occl_add_mm"]:7.1f} '
                  f'{s["end_add_mm"]:7.1f}')
