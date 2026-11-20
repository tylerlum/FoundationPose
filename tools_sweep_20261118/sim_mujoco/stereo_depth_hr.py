"""Fast-FS (PyTorch) at a chosen resolution -> TAKE/replay_fs<W>/ (symlinked rgb/masks/K/ts/hand, new depth).
python stereo_depth_hr.py W H TAKE ..."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
import numpy as np, cv2, torch  # noqa
from live_tracking_utils import FastFsDepth, DEFAULT_FAST_FS_MODEL
Wf, Hf = int(sys.argv[1]), int(sys.argv[2])
TRT = '/home/tylerlum/github_repos/depthbasedRL/third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters8' if Wf == 8 else None
tag = 'it8' if Wf == 8 else str(Wf)
for take in map(Path, sys.argv[3:]):
    rd = take / 'replay'; K = np.loadtxt(rd / 'cam_K.txt')
    fs = FastFsDepth(model_path=str(DEFAULT_FAST_FS_MODEL), iters=4, max_disp=192 if TRT else 192 * Wf // 384, baseline_m=0.12,
                     fp_size=(960, 540), fs_size=(384, 224) if TRT else (Wf, Hf), fx_fp=float(K[0, 0]), trt_dir=TRT)
    out = take / f'replay_fs{tag}'; (out / 'depth').mkdir(parents=True, exist_ok=True)
    for sub in ('rgb', 'masks_1', 'masks_2', 'cam_K.txt', 'ts_ns.txt', 'hand_center.npy'):
        if os.path.lexists(out / sub) and not (out / sub).exists(): os.remove(out / sub)
        if not os.path.lexists(out / sub) and (rd / sub).exists(): os.symlink((rd / sub).resolve(), out / sub)
    ts, errs = [], []
    files = sorted((rd / 'rgb').glob('*.png'))
    for f in files:
        L = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB); Rr = cv2.cvtColor(cv2.imread(str(rd / 'right' / f.name)), cv2.COLOR_BGR2RGB)
        t = time.time(); d = fs(L, Rr); torch.cuda.synchronize(); ts.append(time.time() - t)
        np.save(out / 'depth' / (f.stem + '.npy'), d.astype(np.float32))
    print(f'{take.name}: Fast-FS pytorch {Wf}x{Hf}: {np.median(ts) * 1000:.0f} ms/frame', flush=True)
