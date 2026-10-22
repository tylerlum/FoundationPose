"""Live ZED point cloud in viser: the same capture (ZedCaptureProcess at the FoundationPose size) and depth backend
(--depth_backend zed_neural | fast_fs) the live trackers feed FoundationPose, so a hole here is a hole FoundationPose
sees.

    python third_party/FoundationPose/live_pointcloud_viser.py [--depth_backend fast_fs] [--calibration T_RC.txt]

The cloud is in the robot frame (T_RC from --calibration, else live_tracking_replay.T_RC_DEFAULT), with the table
grid at --table_z, or in the camera frame with --camera_frame. 'FoundationPose depth filter' applies the
erode_depth + bilateral_filter_depth that register() runs before it looks for valid depth. The image panel shows the
RGB with pixels that have no valid depth in red.
"""
import argparse
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_FP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_FP_DIR))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--serial_number', type=str, default='15107')
    ap.add_argument('--camera_resolution', choices=['HD1080', 'HD720', 'VGA'], default='HD1080')
    ap.add_argument('--camera_fps', type=int, default=0, help='ZED capture FPS. 0 = use the resolution default.')
    ap.add_argument('--camera_upsidedown', action='store_true')
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--exposure', type=int, default=25)
    ap.add_argument('--gain', type=int, default=40)
    ap.add_argument('--depth_backend', choices=['zed_neural', 'fast_fs'], default='fast_fs')
    ap.add_argument('--fast_fs_trt_dir', default=str(_FP_DIR.parents[0] / 'Fast-FoundationStereo' / 'weights'
                                                     / '23-36-37' / 'onnx_384x224_iters4'),
                    help='the Fast-FS TensorRT engine (the 4-iteration one, as in RUN.md)')
    ap.add_argument('--calibration', default=None, help='T_RC (.txt / .npy); default live_tracking_replay.T_RC_DEFAULT')
    ap.add_argument('--camera_frame', action='store_true', help='show the cloud in the camera frame, not the robot frame')
    ap.add_argument('--table_z', type=float, default=0.53, help='the table top in the robot frame (m)')
    ap.add_argument('--port', type=int, default=8125, help='viser: the first free port from this up')
    ap.add_argument('--stride', type=int, default=2, help='point cloud pixel stride')
    ap.add_argument('--min_depth', type=float, default=0.1)
    ap.add_argument('--max_depth', type=float, default=2.5)
    ap.add_argument('--point_size', type=float, default=0.003)
    ap.add_argument('--look_dist', type=float, default=0.5,
                    help='the initial view orbits about this point along the optical axis (m)')
    ap.add_argument('--hz', type=float, default=15.0)
    args = ap.parse_args()

    import viser
    from live_tracking_replay import T_RC_DEFAULT, first_free_port
    from live_tracking_utils import DEFAULT_FAST_FS_MODEL, FastFsDepth, ZedCaptureProcess

    if args.camera_frame:
        T_RC = np.eye(4)
    elif args.calibration:
        p = Path(args.calibration)
        T_RC = (np.load(p) if p.suffix == '.npy' else np.loadtxt(p)).reshape(4, 4)
    else:
        T_RC = np.asarray(T_RC_DEFAULT, float)
    print('[pointcloud] T_RC:\n' + np.array2string(T_RC, precision=4, suppress_small=True), flush=True)

    cap = ZedCaptureProcess(zed_kwargs={'resolution': args.camera_resolution, 'camera_fps': args.camera_fps,
                                        'serial_number': args.serial_number, 'exposure': args.exposure,
                                        'gain': args.gain},
                            fp_size=(args.width, args.height), backend=args.depth_backend,
                            camera_upsidedown=args.camera_upsidedown)
    cap.start(ready_timeout=30.0)
    K = cap.K
    print(f'[pointcloud] camera ready; K[0,0]={K[0, 0]:.1f}', flush=True)
    fast_fs = None
    if args.depth_backend == 'fast_fs':
        # the tracker's defaults (live_tracking_new_gui_2_objects.py)
        fast_fs = FastFsDepth(model_path=str(DEFAULT_FAST_FS_MODEL), iters=4, max_disp=192, baseline_m=cap.baseline_m,
                              fp_size=(args.width, args.height), fs_size=(384, 224), fx_fp=float(K[0, 0]),
                              trt_dir=args.fast_fs_trt_dir)
        print(f'[pointcloud] Fast-FS {fast_fs.backend}, baseline {cap.baseline_m * 1000:.1f} mm', flush=True)

    server = viser.ViserServer(port=first_free_port(args.port), label='ZED point cloud')
    server.scene.set_up_direction('-y' if args.camera_frame else '+z')
    server.scene.add_frame('/world', axes_length=0.2, axes_radius=0.005)
    table = server.scene.add_grid('/table', width=2.0, height=2.0, position=(0.0, 0.0, args.table_z),
                                  cell_size=0.1, visible=not args.camera_frame)
    cam = server.scene.add_frame('/camera', axes_length=0.1, axes_radius=0.004,
                                 position=tuple(float(v) for v in T_RC[:3, 3]),
                                 wxyz=tuple(float(v) for v in Rotation.from_matrix(T_RC[:3, :3]).as_quat()[[3, 0, 1, 2]]))
    server.scene.add_camera_frustum('/camera/frustum', fov=2 * math.atan(args.height / 2 / K[1, 1]),
                                    aspect=args.width / args.height, scale=0.1, color=(255, 255, 0))
    cloud = server.scene.add_point_cloud('/camera/cloud', points=np.zeros((1, 3), np.float32),
                                         colors=np.zeros((1, 3), np.uint8), point_size=args.point_size)
    del cam

    # Start the browser's view at x = y = 0 and the ZED's height, upright (+z up, no roll), looking at the point
    # --look_dist along the ZED's optical axis (on the table), which is also the orbit pivot. Viser's 'Reset View'
    # returns here.
    def view_from_camera():
        c, R = T_RC[:3, 3], T_RC[:3, :3]
        target = c + args.look_dist * R[:, 2]
        if args.camera_frame:
            return dict(position=(0.0, 0.0, 0.0), look_at=tuple(float(x) for x in target),
                        up_direction=(0.0, -1.0, 0.0))
        return dict(position=(0.0, 0.0, float(c[2])), look_at=tuple(float(x) for x in target),
                    up_direction=(0.0, 0.0, 1.0))

    v0 = view_from_camera()
    server.initial_camera.position = v0['position']
    server.initial_camera.look_at = v0['look_at']
    server.initial_camera.up = v0['up_direction']
    server.initial_camera.fov = 2 * math.atan(args.height / 2 / K[1, 1])

    gui = server.gui
    status = gui.add_markdown('')
    freeze = gui.add_checkbox('freeze', initial_value=False)
    fp_filter = gui.add_checkbox('FoundationPose depth filter', initial_value=False)
    psize = gui.add_slider('point size', min=0.001, max=0.02, step=0.0005, initial_value=args.point_size)
    stride = gui.add_slider('stride', min=1, max=8, step=1, initial_value=args.stride)
    dmin = gui.add_slider('min depth (m)', min=0.0, max=1.0, step=0.01, initial_value=args.min_depth)
    dmax = gui.add_slider('max depth (m)', min=0.2, max=5.0, step=0.05, initial_value=args.max_depth)
    show_table = gui.add_checkbox(f'table grid (z = {args.table_z} m)', initial_value=not args.camera_frame)
    btn_view = gui.add_button('View from the camera')
    show_img = gui.add_checkbox('image (red = no depth)', initial_value=True)
    img = gui.add_image(np.zeros((args.height // 2, args.width // 2, 3), np.uint8), label='rgb / invalid depth',
                        format='jpeg', jpeg_quality=80)
    psize.on_update(lambda _: setattr(cloud, 'point_size', psize.value))
    show_table.on_update(lambda _: setattr(table, 'visible', show_table.value))
    show_img.on_update(lambda _: setattr(img, 'visible', show_img.value))

    @btn_view.on_click
    def _(event):
        v = view_from_camera()
        cam_ = event.client.camera
        cam_.position, cam_.look_at, cam_.up_direction = v['position'], v['look_at'], v['up_direction']

    _fp = {}

    def fp_depth(depth):
        """What FoundationPose's register() sees (estimater.py)."""
        if not _fp:
            from Utils import bilateral_filter_depth, erode_depth
            _fp.update(erode=erode_depth, bilateral=bilateral_filter_depth)
        d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        d = _fp['erode'](d, radius=2, device='cuda')
        d = _fp['bilateral'](d, radius=2, device='cuda')
        return d.cpu().numpy() if hasattr(d, 'cpu') else np.asarray(d)

    print(f'[pointcloud] open http://{socket.getfqdn()}:{server.get_port()}', flush=True)
    last_seq, t_fps, n_fps, fps = 0, time.time(), 0, 0.0
    vv, uu = np.mgrid[0:args.height, 0:args.width]
    try:
        while True:
            t0 = time.time()
            if not freeze.value:
                out = cap.latest(since_seq=last_seq, timeout=1.0)
                if out is not None:
                    last_seq, _, rgb, depth = out
                    if fast_fs is not None:   # the capture hands over (left, right)
                        depth = fast_fs(rgb, depth)
                    if fp_filter.value:
                        depth = fp_depth(depth)
                    valid = np.isfinite(depth) & (depth >= 0.001)
                    s = int(stride.value)
                    d = depth[::s, ::s]
                    ok = valid[::s, ::s] & (d > dmin.value) & (d < dmax.value)
                    z = d[ok]
                    u, v = uu[::s, ::s][ok], vv[::s, ::s][ok]
                    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
                    cloud.points = pts.astype(np.float32)
                    cloud.colors = rgb[::s, ::s][ok]
                    if show_img.value:
                        vis = rgb.copy()
                        vis[~valid] = (0.5 * vis[~valid] + [127, 0, 0]).astype(np.uint8)
                        img.image = vis[::2, ::2]
                    n_fps += 1
                    if time.time() - t_fps > 1.0:
                        fps, n_fps, t_fps = n_fps / (time.time() - t_fps), 0, time.time()
                    status.content = (f'**{fps:.1f} Hz**, {len(z)} points\n\n'
                                      f'valid depth: {100 * valid.mean():.1f}% of pixels'
                                      + (' (after the FoundationPose filter)' if fp_filter.value else '')
                                      + f'\n\ndepth: {args.depth_backend}'
                                      + f'\n\nframe: {"camera" if args.camera_frame else "robot (T_RC)"}')
            time.sleep(max(0.0, 1.0 / args.hz - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop_capture()


if __name__ == '__main__':
    main()
