"""Capture N RGB-D frames from a ZED (Fast-FS TRT depth) + cam_K.txt. rgb png, depth npy (m) and png (uint16 mm)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # third_party/FoundationPose
if __name__ == '__main__':
    import numpy as np, cv2
    from live_tracking_utils import ZedCaptureProcess, FastFsDepth, DEFAULT_FAST_FS_MODEL
    serial, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    for d in ('rgb', 'depth', 'depth_png'): os.makedirs(f'{out}/{d}', exist_ok=True)
    cap = ZedCaptureProcess(zed_kwargs=dict(resolution='HD1080', camera_fps=30, serial_number=serial, exposure=25, gain=40),
                            fp_size=(960, 540), backend='fast_fs', camera_upsidedown=False)
    cap.start(ready_timeout=30.0); K = cap.K
    fs = FastFsDepth(model_path=str(DEFAULT_FAST_FS_MODEL), iters=4, max_disp=192, baseline_m=cap.baseline_m, fp_size=(960, 540), fs_size=(384, 224),
                     fx_fp=float(K[0, 0]), trt_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Fast-FoundationStereo', 'weights', '23-36-37', 'onnx_384x224_iters4'))
    np.savetxt(f'{out}/cam_K.txt', K); seq = 0
    for i in range(n + 30):
        seq, ts, a, b = cap.latest(since_seq=seq, timeout=2.0)
        if i < 30: continue
        d = fs(a, b).astype(np.float32); stem = f'{i-30:05d}'
        cv2.imwrite(f'{out}/rgb/{stem}.png', cv2.cvtColor(a, cv2.COLOR_RGB2BGR)); np.save(f'{out}/depth/{stem}.npy', d)
        cv2.imwrite(f'{out}/depth_png/{stem}.png', np.clip(np.nan_to_num(d) * 1000, 0, 65535).astype(np.uint16))
        time.sleep(0.1)
    print('rgb', a.shape, 'depth', d.shape, 'valid', np.isfinite(d).mean(), 'median m', np.nanmedian(d)); print('K', K)
    cap.stop_capture()
