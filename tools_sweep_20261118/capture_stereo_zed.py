"""Save one rectified ZED stereo pair (HD1080 + 960x540), ZED NEURAL(_PLUS) depth, K for both sizes, and baseline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # third_party/FoundationPose
import numpy as np, cv2, pyzed.sl as sl
serial, out = sys.argv[1], sys.argv[2]
mode = sl.DEPTH_MODE.NEURAL
os.makedirs(out, exist_ok=True)
zed = sl.Camera(); ip = sl.InitParameters(); ip.camera_resolution = sl.RESOLUTION.HD1080; ip.camera_fps = 30
ip.depth_mode = mode; ip.coordinate_units = sl.UNIT.METER; ip.depth_minimum_distance = 0.2; ip.set_from_serial_number(int(serial))
assert zed.open(ip) == sl.ERROR_CODE.SUCCESS
zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 25); zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 40)
rt = sl.RuntimeParameters(); L, R, D = sl.Mat(), sl.Mat(), sl.Mat()
for i in range(60): zed.grab(rt)
assert zed.grab(rt) == sl.ERROR_CODE.SUCCESS
zed.retrieve_image(L, sl.VIEW.LEFT); zed.retrieve_image(R, sl.VIEW.RIGHT); zed.retrieve_measure(D, sl.MEASURE.DEPTH)
left = cv2.cvtColor(L.get_data(), cv2.COLOR_BGRA2BGR); right = cv2.cvtColor(R.get_data(), cv2.COLOR_BGRA2BGR)
depth = D.get_data().astype(np.float32); depth[~np.isfinite(depth)] = 0
cal = zed.get_camera_information().camera_configuration.calibration_parameters
lc = cal.left_cam; W, H = lc.image_size.width, lc.image_size.height
K_full = np.array([[lc.fx, 0, lc.cx], [0, lc.fy, lc.cy], [0, 0, 1]]); K_half = K_full.copy(); K_half[:2] *= 0.5
baseline = abs(cal.get_camera_baseline()) / (1000.0 if cal.get_camera_baseline() > 1 else 1.0)
cv2.imwrite(f'{out}/left_full.png', left); cv2.imwrite(f'{out}/right_full.png', right); np.save(f'{out}/depth_zed_full.npy', depth)
cv2.imwrite(f'{out}/left.png', cv2.resize(left, (960, 540), interpolation=cv2.INTER_AREA)); cv2.imwrite(f'{out}/right.png', cv2.resize(right, (960, 540), interpolation=cv2.INTER_AREA))
np.save(f'{out}/depth_zed.npy', cv2.resize(depth, (960, 540), interpolation=cv2.INTER_NEAREST))
np.savetxt(f'{out}/K_full.txt', K_full); np.savetxt(f'{out}/cam_K.txt', K_half); open(f'{out}/baseline_m.txt', 'w').write(f'{baseline}\n')
print('mode', mode, 'size', W, H, 'baseline m', baseline, 'zed depth valid frac', (depth > 0).mean()); print('K_half', K_half)
zed.close()
