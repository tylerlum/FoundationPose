"""ROI stereo for a small workpiece: the same Fast-FS TensorRT engine (384x224) run on a FULL-RESOLUTION 384x224 crop of
the rendered stereo pair around the workpiece (2.5x the pixel density there), pasted into the normal depth.
python stereo_roi.py TAKE ...  -> TAKE/replay_roi/ (symlinks + depth)"""
import os, sys
from pathlib import Path
sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
import numpy as np, cv2, torch  # noqa
from live_tracking_utils import FastFsDepth, DEFAULT_FAST_FS_MODEL
TRT = '/home/tylerlum/github_repos/depthbasedRL/third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4'
fs = None
for take in map(Path, sys.argv[1:]):
    rd = take / 'replay'; K = np.loadtxt(rd / 'cam_K.txt')
    if fs is None:   # crops are fed at their own size: fp_size == fs_size, so fx_fs == fx (full-res pixels)
        fs = FastFsDepth(model_path=str(DEFAULT_FAST_FS_MODEL), iters=4, max_disp=192, baseline_m=0.12, fp_size=(384, 224),
                         fs_size=(384, 224), fx_fp=float(K[0, 0]), trt_dir=TRT)
    m2 = cv2.imread(str(rd / 'masks_2' / '00000.png'), 0) > 0
    ys, xs = np.nonzero(m2); cx, cy = int(xs.mean()), int(ys.mean())
    x0 = int(np.clip(cx - 192 + 60, 0, 960 - 384)); y0 = int(np.clip(cy - 112, 0, 540 - 224))   # workpiece right of centre: its match sits ~100 px left
    out = take / 'replay_roi'; (out / 'depth').mkdir(parents=True, exist_ok=True)
    for sub in ('rgb', 'masks_1', 'masks_2', 'cam_K.txt', 'ts_ns.txt', 'hand_center.npy'):
        if not os.path.lexists(out / sub) and (rd / sub).exists(): os.symlink((rd / sub).resolve(), out / sub)
    gt0 = np.load(rd / 'depth_gt' / '00000.npy'); errs_lo, errs_hi = [], []
    for f in sorted((rd / 'rgb').glob('*.png')):
        L = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB); Rr = cv2.cvtColor(cv2.imread(str(rd / 'right' / f.name)), cv2.COLOR_BGR2RGB)
        d = np.load(rd / 'depth' / (f.stem + '.npy')).copy()
        dr = fs(L[y0:y0 + 224, x0:x0 + 384], Rr[y0:y0 + 224, x0:x0 + 384])
        inner = np.zeros_like(dr, bool); inner[8:-8, 120:-8] = True          # keep the part with valid matches
        sub = d[y0:y0 + 224, x0:x0 + 384]; sub[inner & (dr > 0)] = dr[inner & (dr > 0)]
        np.save(out / 'depth' / (f.stem + '.npy'), d.astype(np.float32))
        if f.stem == '00000':
            g = np.load(rd / 'depth_gt' / '00000.npy'); lo = np.load(rd / 'depth' / '00000.npy')
            mm = m2 & (g > 0)
            print(f'{take.name}: workpiece pixels |depth - true| median: normal {np.median(np.abs(lo[mm] - g[mm])) * 1000:.1f} mm, ROI {np.median(np.abs(d[mm] - g[mm])) * 1000:.1f} mm (ROI x {x0}..{x0 + 384}, y {y0}..{y0 + 224})', flush=True)
