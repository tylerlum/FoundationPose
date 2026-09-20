"""Child-process entrypoint for the ZED capture producer.

Kept in a *separate* module so the multiprocessing spawn-based child does
not transitively import `live_tracking_utils` (which loads the
FoundationPose / warp / torch stack — adding ~5 s of cold-start). The
child only needs pyzed + numpy + cv2.
"""

import time
import numpy as np
import cv2
import pyzed.sl as sl
from multiprocessing.shared_memory import SharedMemory


def _zed_view_to_rgb(mat, width, height, camera_upsidedown):
    img = mat.get_data()
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        img = img[..., :3]
    if camera_upsidedown:
        img = cv2.flip(img, -1)
    return cv2.resize(img, (width, height))


def producer_main(shm_names, stop_event, lock, ready_event, info_queue,
                  zed_kwargs, fp_size, backend, camera_upsidedown):
    """Entrypoint. The child opens its OWN sl.Camera (and SDK CUDA context),
    grabs frames into shared memory, and signals the parent via `meta[1]`
    (frame sequence number). Camera intrinsics + stereo baseline are
    one-shot-published to the parent through `info_queue`."""
    W, H = fp_size
    rgb_shm = SharedMemory(name=shm_names['rgb'])
    d_shm = SharedMemory(name=shm_names['d'])
    meta_shm = SharedMemory(name=shm_names['meta'])

    rgb_view = np.ndarray((H, W, 3), dtype=np.uint8, buffer=rgb_shm.buf)
    if backend == 'zed_neural':
        d_view = np.ndarray((H, W), dtype=np.float32, buffer=d_shm.buf)
    else:
        d_view = np.ndarray((H, W, 3), dtype=np.uint8, buffer=d_shm.buf)
    # meta = [ts_ns, seq, error_flag, pad]
    meta = np.ndarray((4,), dtype=np.int64, buffer=meta_shm.buf)

    zed = sl.Camera()
    try:
        init_params = sl.InitParameters(input_t=sl.InputType())
        init_params.svo_real_time_mode = True
        init_params.camera_resolution = getattr(
            sl.RESOLUTION, zed_kwargs['resolution'])
        init_params.depth_mode = (sl.DEPTH_MODE.NEURAL
                                  if backend == 'zed_neural'
                                  else sl.DEPTH_MODE.NONE)
        init_params.coordinate_units = (sl.UNIT.MILLIMETER
                                        if backend == 'zed_neural'
                                        else sl.UNIT.METER)
        if zed_kwargs.get('camera_fps'):
            init_params.camera_fps = int(zed_kwargs['camera_fps'])
        init_params.set_from_serial_number(int(zed_kwargs['serial_number']))

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            with lock:
                meta[2] = 1
            try:
                info_queue.put({'error': str(err)}, timeout=1.0)
            except Exception:
                pass
            ready_event.set()
            return

        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,
                                int(zed_kwargs['exposure']))
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN,
                                int(zed_kwargs['gain']))

        # Publish intrinsics + baseline to the parent (one-shot).
        cam_info = zed.get_camera_information()
        lcp = cam_info.camera_configuration.calibration_parameters.left_cam
        fx = lcp.fx * W / lcp.image_size.width
        fy = lcp.fy * H / lcp.image_size.height
        cx = lcp.cx * W / lcp.image_size.width
        cy = lcp.cy * H / lcp.image_size.height
        if camera_upsidedown:
            cx = W - cx
            cy = H - cy
        K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        baseline_m = abs(
            cam_info.camera_configuration
            .calibration_parameters.get_camera_baseline())
        # SDK reports baseline in `coordinate_units` (METER for fast_fs,
        # MILLIMETER for zed_neural). Normalize to meters.
        if backend == 'zed_neural':
            baseline_m = baseline_m / 1000.0
        try:
            info_queue.put({'K': K, 'baseline_m': baseline_m},
                           timeout=1.0)
        except Exception:
            pass

        runtime = sl.RuntimeParameters()
        mat_a = sl.Mat()
        mat_b = sl.Mat()

        # Per-stage timing samples (seconds). Sent back to the parent at
        # shutdown via info_queue.
        t_grab_log = []
        t_left_log = []
        t_right_or_depth_log = []
        t_publish_log = []

        seq = 0
        first = True
        while not stop_event.is_set():
            t0 = time.time()
            err = zed.grab(runtime)
            t_grab_s = time.time() - t0
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            ts_ns = zed.get_timestamp(
                sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

            t0 = time.time()
            zed.retrieve_image(mat_a, sl.VIEW.LEFT, sl.MEM.CPU)
            rgb = _zed_view_to_rgb(mat_a, W, H, camera_upsidedown)
            t_left_s = time.time() - t0

            if backend == 'zed_neural':
                t0 = time.time()
                zed.retrieve_measure(mat_b, sl.MEASURE.DEPTH, sl.MEM.CPU)
                depth_mm = mat_b.get_data()
                if camera_upsidedown:
                    depth_mm = cv2.flip(depth_mm, -1)
                depth_mm = cv2.resize(depth_mm, (W, H),
                                      interpolation=cv2.INTER_NEAREST)
                depth_m = depth_mm.astype(np.float32) / 1000.0
                depth_m[(depth_m < 0.001) | (~np.isfinite(depth_m))] = 0.0
                t_b_s = time.time() - t0
                payload_b = depth_m
            else:
                t0 = time.time()
                zed.retrieve_image(mat_b, sl.VIEW.RIGHT, sl.MEM.CPU)
                right = _zed_view_to_rgb(mat_b, W, H, camera_upsidedown)
                t_b_s = time.time() - t0
                payload_b = right

            t0 = time.time()
            with lock:
                np.copyto(rgb_view, rgb)
                np.copyto(d_view, payload_b)
                seq += 1
                meta[0] = ts_ns
                meta[1] = seq
            t_pub_s = time.time() - t0

            t_grab_log.append(t_grab_s)
            t_left_log.append(t_left_s)
            t_right_or_depth_log.append(t_b_s)
            t_publish_log.append(t_pub_s)

            if first:
                ready_event.set()
                first = False
    finally:
        # Ship per-stage producer stats back to the parent before tearing
        # down — gives us "is the producer ever missing camera ticks?".
        try:
            if t_grab_log:
                ar = np.asarray
                stats = {
                    'producer_stats': {
                        'n_frames': int(len(t_grab_log)),
                        'grab': _stats(ar(t_grab_log)),
                        'left': _stats(ar(t_left_log)),
                        ('depth' if backend == 'zed_neural' else 'right'):
                            _stats(ar(t_right_or_depth_log)),
                        'publish_lock': _stats(ar(t_publish_log)),
                    },
                }
                info_queue.put(stats, timeout=2.0)
        except Exception:
            pass
        try:
            zed.close()
        except Exception:
            pass
        rgb_shm.close()
        d_shm.close()
        meta_shm.close()


def _stats(arr):
    return {
        'mean_ms': float(arr.mean() * 1e3),
        'p50_ms':  float(np.median(arr) * 1e3),
        'p95_ms':  float(np.percentile(arr, 95) * 1e3),
        'p99_ms':  float(np.percentile(arr, 99) * 1e3),
        'min_ms':  float(arr.min() * 1e3),
        'max_ms':  float(arr.max() * 1e3),
    }
