"""From a v3 stereo capture dir, write <dir>/<variant>/{rgb/00000.png, depth/00000.png (mm), depth_m.npy, cam_K.txt}."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # third_party/FoundationPose
if __name__ == '__main__':
    import numpy as np, cv2
    D = sys.argv[1]; which = sys.argv[2:]
    left = cv2.cvtColor(cv2.imread(f'{D}/left.png'), cv2.COLOR_BGR2RGB); right = cv2.cvtColor(cv2.imread(f'{D}/right.png'), cv2.COLOR_BGR2RGB)
    K = np.loadtxt(f'{D}/cam_K.txt'); B = float(open(f'{D}/baseline_m.txt').read())
    def write(name, depth):
        o = f'{D}/{name}'; os.makedirs(f'{o}/rgb', exist_ok=True); os.makedirs(f'{o}/depth', exist_ok=True)
        depth = np.nan_to_num(depth.astype(np.float32)); depth[(depth < 0.05) | (depth > 10)] = 0
        cv2.imwrite(f'{o}/rgb/00000.png', cv2.cvtColor(left, cv2.COLOR_RGB2BGR)); np.save(f'{o}/depth_m.npy', depth)
        cv2.imwrite(f'{o}/depth/00000.png', np.clip(depth * 1000, 0, 65535).astype(np.uint16)); np.savetxt(f'{o}/cam_K.txt', K)
        print(name, 'valid', round(float((depth > 0).mean()), 3), 'median m', round(float(np.median(depth[depth > 0])), 3))
    if 'zed' in which:
        write('zed_neural', np.load(f'{D}/depth_zed.npy'))
    if 'fastfs' in which:
        from live_tracking_utils import FastFsDepth, DEFAULT_FAST_FS_MODEL
        fs = FastFsDepth(model_path=str(DEFAULT_FAST_FS_MODEL), iters=4, max_disp=192, baseline_m=B, fp_size=(960, 540), fs_size=(384, 224),
                         fx_fp=float(K[0, 0]), trt_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Fast-FoundationStereo', 'weights', '23-36-37', 'onnx_384x224_iters4'))
        write('fastfs_trt384', fs(left, right))
    if 'fs' in which:   # full FoundationStereo disparity at 960x540 written by run_fs.py
        disp = np.load(f'{D}/fs_disp_960x540.npy'); depth = np.zeros_like(disp); ok = disp > 0.5
        depth[ok] = K[0, 0] * B / disp[ok]; write('foundationstereo', depth)
