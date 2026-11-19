"""Score brush tracking on REAL takes (no ground truth): jumps, below-table frames, lost/recovery events, coverage.

python score_real_replay.py T_RC.txt TAKE_DIR [TAKE_DIR ...]
TAKE_DIR is a tracker take (poses_object_1.npy, metadata.json, optional robust_log.json). Typical use: record a rollout
with --save_depth, then re-run <take>/replay with different flags (see compare_replay_configs.sh) and score each output.
Flags a frame when the brush moves > 1 m/s or turns > 400 deg/s between frames, or its origin is > 3 cm below the table
(z < 0.51 m in robot frame; the table top measures 0.540 m).
"""
import sys
import json
import glob
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

TRC = np.loadtxt(sys.argv[1])
print(f'{"take":60s} {"n":>5s} {"fps":>5s} {">1m/s":>6s} {">400d/s":>8s} {"z<0.51":>7s} {"lost":>5s} {"fix":>4s} '
      f'{"cover med":>9s} {"cover<0.3":>9s}')
for d in sys.argv[2:]:
    P = np.load(os.path.join(d, 'poses_object_1.npy'))
    m = json.load(open(os.path.join(d, 'metadata.json')))
    dur = m.get('duration_s') or len(P) / 30.0
    fps = len(P) / max(dur, 1e-6)
    ok = np.isfinite(P[:, 0, 0])
    Rr = np.einsum('ij,njk->nik', TRC, np.where(ok[:, None, None], P, np.eye(4)))
    v = np.r_[0, np.linalg.norm(np.diff(Rr[:, :3, 3], axis=0), axis=1)] * fps
    w = np.r_[0, [np.degrees(R.from_matrix(Rr[i - 1, :3, :3].T @ Rr[i, :3, :3]).magnitude()) for i in range(1, len(Rr))]] * fps
    below = (Rr[:, 2, 3] < 0.51) & ok
    lost = fix = 0
    cov = []
    rl = os.path.join(d, 'robust_log.json')
    if os.path.exists(rl):
        rows = json.load(open(rl))
        lost = sum(1 for r in rows if r.get('lost'))
        fix = sum(1 for r in rows if r.get('action'))
        cov = [r['cover'] for r in rows if isinstance(r.get('cover'), (int, float))]
    cm = f'{np.median(cov):9.2f}' if cov else f'{"-":>9s}'
    cl = f'{int(np.sum(np.array(cov) < 0.3)):9d}' if cov else f'{"-":>9s}'
    print(f'{os.path.basename(d.rstrip("/"))[:60]:60s} {len(P):5d} {fps:5.1f} {int((v > 1).sum()):6d} {int((w > 400).sum()):8d} '
          f'{int(below.sum()):7d} {lost:5d} {fix:4d} {cm} {cl}')
