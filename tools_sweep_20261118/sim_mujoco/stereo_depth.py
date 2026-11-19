"""Fast-FS TRT depth (the live tracker's exact engine and call) from the rendered stereo pair of a synthetic take.

  python stereo_depth.py TAKE_DIR [TAKE_DIR ...]      (drl env)
Writes TAKE/replay/depth/NNNNN.npy (float m, 960x540) and TAKE/replay_gtdepth/ (symlinks + MuJoCo exact depth).
Prints depth error vs MuJoCo depth on the table, the brush and the cube.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402,F401
from live_tracking_utils import FastFsDepth, DEFAULT_FAST_FS_MODEL  # noqa: E402

TRT = '/home/tylerlum/github_repos/depthbasedRL/third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4'
fs = None
for take in map(Path, sys.argv[1:]):
    rd = take / 'replay'
    K = np.loadtxt(rd / 'cam_K.txt')
    if fs is None or abs(fs.fx_fs - float(K[0, 0]) * fs.fs_size[0] / 960) > 1e-3:   # rebuild per camera (fx)
        fs = FastFsDepth(model_path=str(DEFAULT_FAST_FS_MODEL), iters=4, max_disp=192, baseline_m=0.12,
                         fp_size=(960, 540), fs_size=(384, 224), fx_fp=float(K[0, 0]), trt_dir=TRT)
    (rd / 'depth').mkdir(exist_ok=True)
    g = take / 'replay_gtdepth'
    (g / 'depth').mkdir(parents=True, exist_ok=True)
    for sub in ('rgb', 'masks_1', 'masks_2', 'cam_K.txt', 'ts_ns.txt'):
        if os.path.lexists(g / sub) and not (g / sub).exists():
            os.remove(g / sub)                      # broken relative link from an older version
        if not os.path.lexists(g / sub):
            os.symlink((rd / sub).resolve(), g / sub)
    errs = []
    files = sorted((rd / 'rgb').glob('*.png'))
    for f in files:
        L = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        Rr = cv2.cvtColor(cv2.imread(str(rd / 'right' / f.name)), cv2.COLOR_BGR2RGB)
        d = fs(L, Rr)
        np.save(rd / 'depth' / (f.stem + '.npy'), d.astype(np.float32))
        gt = np.load(rd / 'depth_gt' / (f.stem + '.npy'))
        gt[gt > 5] = 0
        np.save(g / 'depth' / (f.stem + '.npy'), gt)
        ok = (d > 0) & (gt > 0)
        errs.append(np.median(np.abs(d[ok] - gt[ok])))
    print(f'{take.name}: {len(files)} frames, median |FastFS - true| depth {np.median(errs) * 1000:.1f} mm '
          f'(per-frame medians, whole image)', flush=True)
