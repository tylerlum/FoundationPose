"""Calibrate the ZED's pose in the robot frame (T_RC: camera -> robot base), two ways (13 Oct 2026).

Method 1, rough, by eye (a few minutes; use it after moving the camera):

    python third_party/FoundationPose/calibrate_camera.py viser [--calibration T_RC.txt]

  The live ZED point cloud, placed in the robot frame by T_RC, next to the robot model at its current joint angles
  (/iiwa/joint_states, /sharpa/joint_states). Rotate / translate the camera (small deltas about the robot's axes, as in
  Kushal's viser_compare_pointclouds.py; 'Bake' re-centres the sliders) until the arm's points sit on the model, then
  press 'Save T_RC'. Starts from --calibration, else from live_tracking_replay.T_RC_DEFAULT (the latest calibration;
  --start guess: a camera at --cam_xyz looking at --look_at). --no_ros: the model at simtoolreal's default pose.

Method 2, precise: an AprilTag (36h11) fixed anywhere on the hand or the arm's last link. Its pose on the robot is
unknown: the solve finds it, along with T_RC.

    python third_party/FoundationPose/calibrate_camera.py collect --tag_size 0.055 [--n 20] [--dry_run]
    python third_party/FoundationPose/calibrate_camera.py solve calib_<stamp>/samples.npz

  Safety: every pose, and the straight joint path to it from the base, keeps the robot --min_clear (5 cm) above the
  table (--table_z, over --table_box); the arm returns to the base between poses. --preview shows the poses in viser
  over the live point cloud first; --manual records wherever you put the arm (the script sends no commands).
  collect: around simtoolreal's default arm pose (--base current: the arm's pose now), moves to --n (30) random small
  offsets (--delta_deg on joints 1-4, --wrist_deg on joints 5-7; the wrist turning is what pins down rotation), holds
  each pose, averages the tag's pose over --hold_s of frames, and records the joint angles and the tag pose (and one
  image per pose). It asks before moving; --dry_run only prints the poses. Then it solves.
  solve: T_base_ee(q_i) @ T_ee_tag = T_RC @ T_cam_tag_i for T_RC and T_ee_tag: OpenCV's closed form
  (calibrateRobotWorldHandEye), then a robust least squares. Prints the fit per sample (mm / deg); writes
  T_RC.txt / T_RC.npy next to the samples.

    python third_party/FoundationPose/calibrate_camera.py record --tag_size 0.055
  record (teach pad): you move the arm yourself, joint by joint; it logs every frame (joints + tag) and a small video,
  takes a sample at each stop, then solves on Ctrl-C and scores the current T_RC_DEFAULT (or --check FILE) on the same
  samples -- a sanity check of a T_RC from the viser tool.

    python third_party/FoundationPose/calibrate_camera.py selftest        # the solver on synthetic data

Measure the printed tag's black square accurately: a tag-size error scales every tag distance.
Depth refinement is OFF by default. On 14 Oct 2026, ZED tag-region depth inflated translations
by up to 6.8x; disabling it improved the same 44-pose fit from 245 mm to 2.51 mm RMS.
Use --depth_refine only for experiments with independently validated depth.
Use the result with the trackers' --calibration T_RC.txt.
"""
import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_FP_DIR = Path(__file__).resolve().parent
_REPO = _FP_DIR.parents[1]
# The robot URDF. Default: depthbasedRL's copy (this folder is its third_party/FoundationPose). Outside depthbasedRL
# set ROBOT_URDF (e.g. the SimToolReal repo's assets/urdf/kuka_sharpa_description/...) or pass --urdf.
URDF = Path(os.environ.get(
    'ROBOT_URDF', _REPO / 'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf'))
ARM_JOINTS = [f'iiwa14_joint_{i}' for i in range(1, 8)]
# simtoolreal's default arm pose (isaacgymenvs/tasks/simtoolreal/env.py; Kushal's viser_compare_pointclouds.py)
DEFAULT_ARM_Q = [-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308]
# The arm with the fingers (and the tag on them) turned to the camera, as Tyler set it on 13 Oct 2026 (joint 7 +90 deg
# from the default): collect's base pose when there are no joint states (deg: -97.2 82.2 -11.4 57.2 0 84.8 164.6).
CALIB_BASE_Q = [-1.6967, 1.4353, -0.1989, 0.9991, -0.0004, 1.4801, 2.8734]
# The camera after the 13 Oct move, by eye: 45 cm on -x from the hand (the hand at the default pose is at
# about (0, -0.67, 0.73)), looking along +x at the table's centre (sim: (0, -0.80) at z 0.53) and a little up.
CAM_GUESS_XYZ = (-0.45, -0.60, 0.70)
CAM_GUESS_LOOK_AT = (0.0, -0.80, 0.58)
ROT_SCALE = 0.05     # m per rad in the residual: 1 deg ~ 0.9 mm

# rospy before anything that loads the FoundationPose stack (see live_tracking_utils.py)
if len(sys.argv) > 1 and sys.argv[1] in ('viser', 'collect', 'record') and '--no_ros' not in sys.argv:
    import rospy  # noqa: F401
    from sensor_msgs.msg import JointState  # noqa: F401


# --------------------------------------------------------------------------------------------- transforms
def make(rotvec_or_R, t):
    T = np.eye(4)
    R = np.asarray(rotvec_or_R, float)
    T[:3, :3] = R if R.shape == (3, 3) else Rotation.from_rotvec(R).as_matrix()
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def xyz_rpy(T):
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)


def from_xyz_rpy(xyz, rpy_deg):
    return make(Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix(), xyz)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """T_RC for a camera at `eye` looking at `target` (robot frame), image up toward +z: OpenCV axes (x right,
    y down, z forward) as the columns."""
    eye, z = np.asarray(eye, float), np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return make(np.stack([x, np.cross(z, x), z], 1), eye)


def load_T(path):
    return (np.load(path) if str(path).endswith('.npy') else np.loadtxt(path)).reshape(4, 4)


def save_T(T, stem):
    np.savetxt(f'{stem}.txt', T)
    np.save(f'{stem}.npy', T)
    print(f'wrote {stem}.txt and {stem}.npy:\n{np.array2string(T, precision=6, suppress_small=True)}', flush=True)


# --------------------------------------------------------------------------------------------- URDF kinematics
class Urdf:
    """Forward kinematics and visual meshes of a URDF tree (revolute / continuous / prismatic / fixed joints)."""

    def __init__(self, path=URDF):
        self.path = Path(path)
        root = ET.parse(self.path).getroot()
        self.joints, self.children = {}, {}
        for j in root.iter('joint'):
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            a = j.find('axis')
            lim = j.find('limit')
            jt = dict(name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'),
                      child=j.find('child').get('link'),
                      origin=make(Rotation.from_euler('xyz', rpy).as_matrix(), xyz),
                      axis=np.array([float(v) for v in (a.get('xyz') if a is not None else '1 0 0').split()]),
                      lower=float(lim.get('lower', '-inf')) if lim is not None else -np.inf,
                      upper=float(lim.get('upper', 'inf')) if lim is not None else np.inf)
            self.joints[jt['name']] = jt
            self.children.setdefault(jt['parent'], []).append(jt)
        child_links = {j['child'] for j in self.joints.values()}
        self.root = next(lk.get('name') for lk in root.iter('link') if lk.get('name') not in child_links)
        self.visuals = {}
        for lk in root.iter('link'):
            for v in lk.findall('visual'):
                m = v.find('geometry/mesh')
                if m is None:
                    continue
                o = v.find('origin')
                xyz = [float(x) for x in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
                rpy = [float(x) for x in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
                fn = m.get('filename').replace('package://', '')
                scale = [float(s) for s in m.get('scale', '1 1 1').split()]
                self.visuals.setdefault(lk.get('name'), []).append(
                    (self.path.parent / fn, make(Rotation.from_euler('xyz', rpy).as_matrix(), xyz), scale))
        self.movable = [n for n, j in self.joints.items() if j['type'] in ('revolute', 'continuous', 'prismatic')]

    def fk(self, q):
        """q: {joint name: value} (missing joints at 0). Returns {link: 4x4 in the root frame}."""
        out = {self.root: np.eye(4)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for j in self.children.get(parent, []):
                v = float(q.get(j['name'], 0.0))
                if j['type'] in ('revolute', 'continuous'):
                    M = make(Rotation.from_rotvec(j['axis'] * v).as_matrix(), np.zeros(3))
                elif j['type'] == 'prismatic':
                    M = make(np.eye(3), j['axis'] * v)
                else:
                    M = np.eye(4)
                out[j['child']] = out[parent] @ j['origin'] @ M
                stack.append(j['child'])
        return out

    def arm_limits(self):
        return (np.array([self.joints[n]['lower'] for n in ARM_JOINTS]),
                np.array([self.joints[n]['upper'] for n in ARM_JOINTS]))


def joint_map(urdf, arm_msg, hand_msg):
    """JointState messages -> {joint: value}. By name when the names match the URDF, else by order (the arm's 7
    joints; the hand's movable joints in URDF order)."""
    q = {}
    hand_joints = [n for n in urdf.movable if n not in ARM_JOINTS]
    for msg, default in ((arm_msg, ARM_JOINTS), (hand_msg, hand_joints)):
        if msg is None:
            continue
        names = list(msg.name)
        if names and all(n in urdf.joints for n in names):
            q.update(zip(names, msg.position))
        else:
            q.update(zip(default, msg.position))
    return q


# --------------------------------------------------------------------------------------------- the solve
def residuals(x, A, B):
    X, Y = make(x[0:3], x[3:6]), make(x[6:9], x[9:12])
    out = []
    for Ai, Bi in zip(A, B):
        D = inv(X @ Bi) @ (Ai @ Y)
        out.append(np.r_[ROT_SCALE * Rotation.from_matrix(D[:3, :3]).as_rotvec(), D[:3, 3]])
    return np.concatenate(out)


def solve(A, B, loss='soft_l1', f_scale=0.003):
    """A_i = T_base_ee(q_i), B_i = T_cam_tag_i. Returns T_RC (camera -> base), T_ee_tag and the fit."""
    import cv2
    A, B = list(A), list(B)
    assert len(A) == len(B) >= 3, 'need at least three poses'
    # OpenCV's robot-world/hand-eye form  cTw = cTg gTb bTw  with  w = the tag, g = our base, b = our ee:
    R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
        [b[:3, :3] for b in B], [b[:3, 3] for b in B], [a[:3, :3] for a in A], [a[:3, 3] for a in A])
    X0, Y0 = inv(make(R_g2c, t_g2c)), make(R_b2w, t_b2w)
    x0 = np.r_[Rotation.from_matrix(X0[:3, :3]).as_rotvec(), X0[:3, 3],
               Rotation.from_matrix(Y0[:3, :3]).as_rotvec(), Y0[:3, 3]]
    res = least_squares(residuals, x0, args=(A, B), loss=loss, f_scale=f_scale)
    r = residuals(res.x, A, B).reshape(-1, 6)
    deg = np.degrees(np.linalg.norm(r[:, :3], axis=1) / ROT_SCALE)
    mm = 1000 * np.linalg.norm(r[:, 3:], axis=1)
    Rs = Rotation.from_matrix(np.stack([a[:3, :3] for a in A]))
    v = (Rs.mean().inv() * Rs).as_rotvec()
    spread = np.degrees(np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(np.cov(v.T)))[::-1], 0, None)))
    return dict(T_RC=make(res.x[0:3], res.x[3:6]), T_ee_tag=make(res.x[6:9], res.x[9:12]), T_RC_closed_form=X0,
                mm=mm, deg=deg, rms_mm=float(np.sqrt(np.mean(mm ** 2))), rms_deg=float(np.sqrt(np.mean(deg ** 2))),
                wrist_spread_deg=spread)


def report(sol, T_ref=None):
    print(f"fit over {len(sol['mm'])} poses: rms {sol['rms_mm']:.2f} mm, {sol['rms_deg']:.2f} deg; "
          f"worst {sol['mm'].max():.2f} mm, {sol['deg'].max():.2f} deg")
    print('per pose (mm / deg):', ' '.join(f'{a:.1f}/{b:.2f}' for a, b in zip(sol['mm'], sol['deg'])))
    print('rotation spread of the hand (deg, three axes):', np.round(sol['wrist_spread_deg'], 1).tolist(),
          '(the smallest should be > ~5)')
    d = inv(sol['T_RC_closed_form']) @ sol['T_RC']
    print(f"least squares moved the closed form by {1000 * np.linalg.norm(d[:3, 3]):.1f} mm, "
          f"{np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())):.2f} deg")
    print('T_ee_tag (the tag on the hand): xyz (m)', np.round(sol['T_ee_tag'][:3, 3], 4).tolist())
    if T_ref is not None:
        d = inv(T_ref) @ sol['T_RC']
        print(f"T_RC vs the reference: {1000 * np.linalg.norm(d[:3, 3]):.1f} mm, "
              f"{np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())):.2f} deg")


# --------------------------------------------------------------------------------------------- the tag
def make_detector():
    import cv2
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)


def detect_tag(rgb, depth, K, tag_size, tag_id, detector, depth_refine=False):
    """T_cam_tag (OpenCV aruco frame: x right, y up the printed tag, z out of it) and the corners, or (None, None)."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None
    for c, i in zip(corners, ids.ravel()):
        if int(i) != tag_id:
            continue
        s = tag_size / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), K, None,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None, None
        T = make(rvec.reshape(3), tvec.reshape(3))
        if depth_refine and depth is not None:
            # rescale the tag's distance so its centre's depth matches the ZED's (median inside the tag)
            mask = np.zeros(depth.shape, np.uint8)
            cv2.fillConvexPoly(mask, c.reshape(4, 2).astype(np.int32), 1)
            d = depth[(mask > 0) & np.isfinite(depth) & (depth > 0.1)]
            if d.size > 20:
                T[:3, 3] *= float(np.median(d)) / T[2, 3]
        return T, c.reshape(4, 2)
    return None, None


def mean_pose(Ts):
    R = Rotation.from_matrix(np.stack([T[:3, :3] for T in Ts])).mean().as_matrix()
    return make(R, np.median(np.stack([T[:3, 3] for T in Ts]), axis=0))


# --------------------------------------------------------------------------------------------- camera
def open_camera(args):
    sys.path.insert(0, str(_FP_DIR))
    from live_tracking_utils import ZedCaptureProcess
    cap = ZedCaptureProcess(zed_kwargs={'resolution': args.camera_resolution, 'camera_fps': 0,
                                        'serial_number': args.serial_number, 'exposure': args.exposure,
                                        'gain': args.gain},
                            fp_size=(args.width, args.height), backend='zed_neural',
                            camera_upsidedown=args.camera_upsidedown)
    cap.start(ready_timeout=30.0)
    print(f'[camera] ready; K[0,0]={cap.K[0, 0]:.1f}', flush=True)
    return cap


class Joints:
    """The latest /iiwa/joint_states and /sharpa/joint_states (rospy subscribers)."""

    def __init__(self):
        import rospy
        from sensor_msgs.msg import JointState
        rospy.init_node('calibrate_camera', anonymous=True, disable_signals=True)
        self.arm = self.hand = None
        self.arm_hist = []
        rospy.Subscriber('/iiwa/joint_states', JointState, self._arm, queue_size=20)
        rospy.Subscriber('/sharpa/joint_states', JointState, self._hand, queue_size=5)

    def _arm(self, m):
        self.arm = m
        now = time.time()
        self.arm_hist.append((now, np.asarray(m.position[:7], float)))
        while self.arm_hist and now - self.arm_hist[0][0] > 3.0:     # the last 3 s, whatever the rate
            self.arm_hist.pop(0)

    def fresh(self, max_age=0.25):
        """Joint states arrived within max_age s (the FRI session can drop: then the last message is stale)."""
        return bool(self.arm_hist) and time.time() - self.arm_hist[-1][0] < max_age

    def still_for(self, seconds, tol_deg):
        """True when the joint states are fresh and every arm joint has stayed within tol_deg of its current angle
        for the last `seconds`."""
        if self.arm is None or not self.fresh():
            return False
        hist = list(self.arm_hist)
        now, q = time.time(), hist[-1][1]
        if now - hist[0][0] < seconds:
            return False
        recent = [qq for tt, qq in hist if now - tt <= seconds]
        if len(recent) < 2:
            return False
        return float(np.degrees(np.max(np.abs(np.stack(recent) - q)))) < tol_deg

    def _hand(self, m):
        self.hand = m

    def wait(self, need_hand=False, timeout=15.0):
        t0 = time.time()
        while (self.arm is None or (need_hand and self.hand is None)) and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.arm is not None and (self.hand is not None or not need_hand)


# --------------------------------------------------------------------------------------------- method 1
def run_viser(args):
    import trimesh
    import viser
    sys.path.insert(0, str(_FP_DIR))
    from live_tracking_replay import T_RC_DEFAULT
    if args.calibration:
        T0 = load_T(args.calibration)
    elif args.start == 'default':
        T0 = np.asarray(T_RC_DEFAULT, float)
    else:
        T0 = look_at(args.cam_xyz, args.look_at)
    print('[viser] starting T_RC:\n' + np.array2string(T0, precision=4, suppress_small=True), flush=True)
    urdf = Urdf(args.urdf)
    joints = None if args.no_ros else Joints()
    if joints is not None and not joints.wait():
        print('[viser] no /iiwa/joint_states yet: showing --q until it arrives', flush=True)
    cap = open_camera(args)

    from live_tracking_replay import first_free_port
    server = viser.ViserServer(port=first_free_port(args.port), label='camera calibration')
    server.scene.set_up_direction('+z')
    server.scene.add_frame('/robot_base', axes_length=0.2, axes_radius=0.005)
    # the table top (z = 0.53 in the robot frame, as in sim): its points should lie on this grid
    table = server.scene.add_grid('/table', width=2.0, height=2.0, position=(0.0, 0.0, args.table_z),
                                  cell_size=0.1)
    links = {}
    for link, vis in urdf.visuals.items():
        for k, (path, T_vis, scale) in enumerate(vis):
            try:
                m = trimesh.load(str(path), force='mesh')
            except Exception as e:
                print(f'[viser] {path}: {e}')
                continue
            m.apply_scale(scale)
            m.apply_transform(T_vis)
            links.setdefault(link, []).append(server.scene.add_mesh_simple(
                f'/robot/{link}_{k}', vertices=np.asarray(m.vertices, np.float32),
                faces=np.asarray(m.faces, np.uint32), color=(200, 120, 60), opacity=args.robot_opacity))
    cam = server.scene.add_frame('/camera', axes_length=0.1, axes_radius=0.004)
    K = cap.K
    server.scene.add_camera_frustum('/camera/frustum', fov=2 * math.atan(args.height / 2 / K[1, 1]),
                                    aspect=args.width / args.height, scale=0.1, color=(255, 255, 0))
    cloud = server.scene.add_point_cloud('/camera/cloud', points=np.zeros((1, 3), np.float32),
                                         colors=np.zeros((1, 3), np.uint8), point_size=args.point_size)
    gui = server.gui
    status = gui.add_markdown('')
    # Kushal's viser_compare_pointclouds.py style: small deltas on a base T_RC -- rotations about the ROBOT's x / y /
    # z axes at the camera, translations along the robot's axes. 'Bake' folds the deltas into the base and re-centres
    # the sliders, to go further than their range.
    base = dict(T=T0.copy())
    sl = {}
    with gui.add_folder('rotate about the robot axes (deg)'):
        for n in 'xyz':
            sl['r' + n] = gui.add_slider(f'rotate {n}', min=-10.0, max=10.0, step=0.05, initial_value=0.0)
    with gui.add_folder('translate along the robot axes (m)'):
        for n in 'xyz':
            sl['t' + n] = gui.add_slider(f'translate {n}', min=-0.1, max=0.1, step=0.0005, initial_value=0.0)
    freeze = gui.add_checkbox('freeze the cloud', initial_value=False)
    psize = gui.add_slider('point size', min=0.001, max=0.02, step=0.001, initial_value=args.point_size)
    show_robot = gui.add_checkbox('show the robot', initial_value=True)
    show_table = gui.add_checkbox(f'show the table grid (z = {args.table_z} m)', initial_value=True)
    show_table.on_update(lambda _: setattr(table, 'visible', show_table.value))
    btn_save = gui.add_button('Save T_RC')
    btn_bake = gui.add_button('Bake the deltas (re-centre the sliders)')
    btn_reset = gui.add_button('Reset to the start')
    out_stem = args.out
    os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)

    def current_T():
        T = base['T'].copy()
        R_adj = Rotation.from_euler('xyz', [sl['r' + n].value for n in 'xyz'], degrees=True).as_matrix()
        T[:3, :3] = R_adj @ T[:3, :3]
        T[:3, 3] += [sl['t' + n].value for n in 'xyz']
        return T

    def update_cam(_=None):
        T = current_T()
        cam.position = tuple(float(v) for v in T[:3, 3])
        cam.wxyz = tuple(float(v) for v in Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]])
        d = inv(T0) @ T
        status.content = ('**T_RC** (camera -> robot)\n\n```\n'
                          + np.array2string(T, precision=4, suppress_small=True)
                          + f'\n```\nfrom the start: {1000 * np.linalg.norm(d[:3, 3]):.1f} mm, '
                          f'{np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())):.2f} deg')

    def zero_sliders():
        for s in sl.values():
            s.value = 0.0

    for s in sl.values():
        s.on_update(update_cam)
    psize.on_update(lambda _: setattr(cloud, 'point_size', psize.value))

    @btn_save.on_click
    def _(_):
        save_T(current_T(), f'{out_stem}_{time.strftime("%Y%m%d_%H%M%S")}')

    @btn_bake.on_click
    def _(_):
        base['T'] = current_T()
        zero_sliders()
        update_cam()

    @btn_reset.on_click
    def _(_):
        base['T'] = T0.copy()
        zero_sliders()
        update_cam()

    @show_robot.on_update
    def _(_):
        for hs in links.values():
            for h in hs:
                h.visible = show_robot.value

    update_cam()
    print(f'[viser] open http://{__import__("socket").getfqdn()}:{server.get_port()}', flush=True)
    q_default = dict(zip(ARM_JOINTS, args.q))
    last_seq, t_robot = 0, 0.0
    v, u = np.mgrid[0:args.height:args.stride, 0:args.width:args.stride]
    try:
        while True:
            if not freeze.value:
                out = cap.latest(since_seq=last_seq, timeout=1.0)
                if out is not None:
                    last_seq, _, rgb, depth = out
                    d = depth[::args.stride, ::args.stride]
                    ok = np.isfinite(d) & (d > 0.1) & (d < args.max_depth)
                    z = d[ok]
                    pts = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)
                    cloud.points = pts.astype(np.float32)
                    cloud.colors = rgb[::args.stride, ::args.stride][ok]
            if time.time() - t_robot > 0.1:
                t_robot = time.time()
                q = dict(q_default)
                if joints is not None:
                    q.update(joint_map(urdf, joints.arm, joints.hand))
                poses = urdf.fk(q)
                for link, hs in links.items():
                    T = poses[link]
                    for h in hs:
                        h.position = tuple(float(x) for x in T[:3, 3])
                        h.wxyz = tuple(float(x) for x in Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]])
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        print('\n[viser] stopped; the last slider values:')
        save_T(current_T(), f'{out_stem}_last')
    finally:
        cap.stop_capture()


# --------------------------------------------------------------------------------------------- method 2
class TableCheck:
    """How far the robot stays above the table: the lowest point of any link's mesh that lies over the table's box
    (x0 x1 y0 y1 in the robot frame) minus the table top's height. Points outside the box (the base, the shoulder,
    the elbow beside the table) do not count."""

    def __init__(self, urdf, table_z, box, q_hand=None, per_link=400):
        import trimesh
        self.urdf, self.table_z, self.box = urdf, table_z, box
        self.q_hand = dict(q_hand or {})        # the fingers as they are now (the thumb hangs below the palm)
        rng = np.random.default_rng(0)
        self.pts = {}
        for link, vis in urdf.visuals.items():
            ps = []
            for path, T_vis, scale in vis:
                try:
                    m = trimesh.load(str(path), force='mesh')
                except Exception:
                    continue
                v = np.asarray(m.vertices) * np.asarray(scale)
                v = (T_vis[:3, :3] @ v.T).T + T_vis[:3, 3]
                ps.append(v[rng.choice(len(v), min(per_link, len(v)), replace=False)])
            if ps:
                self.pts[link] = np.concatenate(ps)

    def clearance(self, q_arm, q_hand=None):
        q = dict(self.q_hand)
        q.update(q_hand or {})
        q.update(zip(ARM_JOINTS, q_arm))
        P = self.urdf.fk(q)
        x0, x1, y0, y1 = self.box
        worst = np.inf
        for link, pts in self.pts.items():
            w = (P[link][:3, :3] @ pts.T).T + P[link][:3, 3]
            over = (w[:, 0] > x0) & (w[:, 0] < x1) & (w[:, 1] > y0) & (w[:, 1] < y1)
            if over.any():
                worst = min(worst, float(w[over, 2].min() - self.table_z))
        return worst

    def path_clearance(self, qa, qb, steps=8):
        return min(self.clearance(qa + s * (qb - qa)) for s in np.linspace(0, 1, steps))


def safe_poses(q0, n, delta_deg, wrist_deg, lower, upper, seed, check, min_clear, accept=None, margin_deg=5.0,
               tries=20000):
    """n random poses around q0 (joints 1-4 within +-delta_deg, 5-7 within +-wrist_deg, margin_deg inside the
    limits) whose straight joint-space path FROM q0 stays min_clear above the table, and that accept(q) passes (the
    tag in view). Returns (poses, clearances)."""
    rng = np.random.default_rng(seed)
    span = np.radians([delta_deg] * 4 + [wrist_deg] * 3)
    m = math.radians(margin_deg)
    poses, clears, near_table, out_of_view = [], [], 0, 0
    for _ in range(tries):
        if len(poses) == n:
            break
        q = np.clip(q0 + rng.uniform(-1, 1, 7) * span, lower + m, upper - m)
        if accept is not None and not accept(q):
            out_of_view += 1
            continue
        c = check.path_clearance(q0, q)
        if c >= min_clear:
            poses.append(q)
            clears.append(c)
        else:
            near_table += 1
    print(f'{len(poses)} poses kept; rejected: {near_table} within {100 * min_clear:.0f} cm of the table, '
          f'{out_of_view} with the tag out of view or turned away', flush=True)
    return poses, clears


class TagView:
    """Would the camera see the tag at arm pose q? From the tag's place on the hand, measured once at the current
    pose with the current T_RC (rough is fine): all four corners in the image with --tag_margin_px to spare, and the
    tag facing the camera within --max_tag_angle."""

    def __init__(self, urdf, ee_link, T_RC, K, wh, T_ee_tag, tag_size, q_hand, margin_px, max_angle_deg):
        self.urdf, self.ee, self.T_CR, self.K, self.wh = urdf, ee_link, inv(T_RC), K, wh
        self.T_ee_tag, self.q_hand = T_ee_tag, dict(q_hand or {})
        s = tag_size / 2
        self.corners = np.array([[-s, s, 0, 1], [s, s, 0, 1], [s, -s, 0, 1], [-s, -s, 0, 1]], float)
        self.margin, self.cos_max = margin_px, math.cos(math.radians(max_angle_deg))

    def angle_deg(self, q):
        """The tag's angle away from facing the camera (deg), or None when it is out of the image."""
        qq = dict(self.q_hand)
        qq.update(zip(ARM_JOINTS, q))
        T = self.T_CR @ self.urdf.fk(qq)[self.ee] @ self.T_ee_tag
        c = (T @ self.corners.T)[:3]
        if np.any(c[2] < 0.05):
            return None
        uv = (self.K @ (c / c[2]))[:2]
        W, H = self.wh
        if uv.min() < self.margin or uv[0].max() > W - self.margin or uv[1].max() > H - self.margin:
            return None
        cosang = float(T[:3, 2] @ (-T[:3, 3] / np.linalg.norm(T[:3, 3])))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))

    def __call__(self, q):
        a = self.angle_deg(q)
        return a is not None and math.cos(math.radians(a)) >= self.cos_max


def record_still(cap, joints, args, K, detector, seq, send=None, current=None):
    """Average the tag's pose over --hold_s of frames once --settle_s has passed (holding `current` with `send`).
    Returns (mean T_cam_tag or None, mean joint angles, n frames, frame-to-frame spread mm, last (rgb, corners), seq)."""
    t_hold, Ts, qs, last = time.time(), [], [], None
    while time.time() - t_hold < args.settle_s + args.hold_s:
        if send is not None:
            send(current)
        out = cap.latest(since_seq=seq, timeout=0.2)
        if out is None:
            continue
        seq = out[0]
        if time.time() - t_hold < args.settle_s:
            continue
        _, _, rgb, depth = out
        T, c = detect_tag(rgb, depth, K, args.tag_size, args.tag_id, detector, not args.no_depth_refine)
        if T is not None:
            Ts.append(T)
            qs.append(np.asarray(joints.arm.position[:7], float))
            last = (rgb, c)
    if len(Ts) < 3:
        return None, None, len(Ts), None, last, seq
    Tm = mean_pose(Ts)
    spread = 1000 * np.max(np.linalg.norm(np.stack([T[:3, 3] for T in Ts]) - Tm[:3, 3], axis=1))
    return Tm, np.mean(qs, axis=0), len(Ts), spread, last, seq


def preview_poses(args, urdf, poses, clears, check, T_RC, snap=None, K=None, q_hand=None, view=None):
    """viser: the robot (its fingers as they are now) at each planned pose (a slider, or the whole path played at
    real speed), the table box, and one snapshot of the point cloud. Moves nothing."""
    import trimesh
    import viser
    sys.path.insert(0, str(_FP_DIR))
    from live_tracking_replay import first_free_port
    out = (None, None, *snap) if snap is not None else None
    server = viser.ViserServer(port=first_free_port(8115), label='calibration poses')
    server.scene.set_up_direction('+z')
    x0, x1, y0, y1 = check.box
    server.scene.add_box('/table', dimensions=(x1 - x0, y1 - y0, 0.002), position=((x0 + x1) / 2, (y0 + y1) / 2,
                         check.table_z), color=(120, 200, 120), opacity=0.3)
    if out is not None:
        _, _, rgb, depth = out
        v, u = np.mgrid[0:depth.shape[0]:2, 0:depth.shape[1]:2]
        d = depth[::2, ::2]
        ok = np.isfinite(d) & (d > 0.1) & (d < 2.5)
        z = d[ok]
        pts = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)
        pts = (T_RC[:3, :3] @ pts.T).T + T_RC[:3, 3]
        server.scene.add_point_cloud('/cloud', points=pts.astype(np.float32), colors=rgb[::2, ::2][ok],
                                     point_size=0.004)
    links = {}
    for link, vis in urdf.visuals.items():
        for k, (path, T_vis, scale) in enumerate(vis):
            try:
                m = trimesh.load(str(path), force='mesh')
            except Exception:
                continue
            m.apply_scale(scale)
            m.apply_transform(T_vis)
            links.setdefault(link, []).append(server.scene.add_mesh_simple(
                f'/robot/{link}_{k}', vertices=np.asarray(m.vertices, np.float32),
                faces=np.asarray(m.faces, np.uint32), color=(200, 120, 60), opacity=0.7))
    status = server.gui.add_markdown('')
    slider = server.gui.add_slider('pose', min=0, max=len(poses) - 1, step=1, initial_value=0)
    play = server.gui.add_checkbox('play the whole path (real speed)', initial_value=False)

    def fk(q):
        qq = dict(q_hand or {})
        qq.update(zip(ARM_JOINTS, q))
        return urdf.fk(qq)

    def pose_to(q):
        P = fk(q)
        for link, hs in links.items():
            for h in hs:
                h.position = tuple(float(x) for x in P[link][:3, 3])
                h.wxyz = tuple(float(x) for x in Rotation.from_matrix(P[link][:3, :3]).as_quat()[[3, 0, 1, 2]])

    def path():
        """The run's joint path at 30 Hz: base -> pose -> base for each pose, at the run's speed."""
        q0 = poses[0]
        for k, q in enumerate(poses[1:], 1):
            for a, b in ((q0, q), (q, q0)):
                secs = max(args.move_s, math.pi / 2 * float(np.degrees(np.max(np.abs(b - a))))
                           / args.max_joint_speed_deg)
                for s in np.linspace(0, 1, max(2, int(secs * 30))):
                    yield k, a + (0.5 - 0.5 * math.cos(math.pi * s)) * (b - a)

    def show(i):
        P = fk(poses[i])
        for link, hs in links.items():
            for h in hs:
                h.position = tuple(float(x) for x in P[link][:3, 3])
                h.wxyz = tuple(float(x) for x in Rotation.from_matrix(P[link][:3, :3]).as_quat()[[3, 0, 1, 2]])
        a = view.angle_deg(poses[i]) if view is not None else None
        tag = ('' if view is None else (f'; the tag {a:.0f} deg from facing the camera' if a is not None
                                        else '; the tag OUT OF VIEW'))
        status.content = (f'pose {i} of {len(poses) - 1} (0 = the base): '
                          f'{np.round(np.degrees(poses[i]), 1).tolist()} deg; lowest point over the table '
                          f'{100 * clears[i]:.1f} cm above it (path from the base included){tag}')

    slider.on_update(lambda _: show(int(slider.value)))
    show(0)
    print(f'[preview] http://{__import__("socket").getfqdn()}:{server.get_port()}  -- scrub the poses; '
          'Ctrl-C to stop', flush=True)
    try:
        while True:
            if play.value:
                for k, q in path():
                    if not play.value:
                        break
                    pose_to(q)
                    status.content = f'playing: pose {k} of {len(poses) - 1} (base -> pose -> base)'
                    time.sleep(1 / 30)
                play.value = False
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


def calibration_metadata(args, joint_samples):
    """Persist FK poses explicitly so offline analysis does not need to rerun the robot."""
    urdf = Urdf(args.urdf)
    poses = [urdf.fk(dict(zip(ARM_JOINTS, q)))[args.ee_link] for q in joint_samples]
    return dict(T_base_ee=np.asarray(poses, dtype=float).reshape(-1, 4, 4),
                joint_names=np.asarray(ARM_JOINTS), depth_refine=not args.no_depth_refine,
                schema_version=2,
                transform_convention='T_parent_child maps child coordinates into parent coordinates')


def run_collect(args):
    import cv2
    import rospy
    from sensor_msgs.msg import JointState
    urdf = Urdf(args.urdf)
    if args.no_ros and not args.preview:
        sys.exit('--no_ros is for --preview only: moving or recording needs the joint states')
    joints = None if args.no_ros else Joints()
    if joints is not None and not joints.wait(need_hand=not args.preview):
        if not args.preview:
            sys.exit('no /iiwa/joint_states or /sharpa/joint_states: is the robot up?')
        print('[preview] no joint states: planning from --q (the 13 Oct tag pose)', flush=True)
    q_now = (np.asarray(joints.arm.position[:7], float) if joints is not None and joints.arm is not None
             else np.asarray(args.q, float))
    out_dir = Path(args.out_dir) / f'calib_{time.strftime("%Y%m%d_%H%M%S")}'
    detector = make_detector()
    Q, B = [], []

    def save_and_solve():
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / 'samples.npz', q=np.array(Q), T_cam_tag=np.array(B), K=K, tag_size=args.tag_size,
                 tag_id=args.tag_id, ee_link=args.ee_link, urdf=str(args.urdf),
                 **calibration_metadata(args, Q))
        print(f'{len(Q)} poses -> {out_dir / "samples.npz"}', flush=True)
        if len(Q) >= 3:
            args.file = str(out_dir / 'samples.npz')
            run_solve(args)

    def keep(i, Tm, qm, n, spread, last):
        if Tm is None:
            print(f'pose {i}: tag seen in {n} frames; skipped (is it in view, flat, lit?)', flush=True)
            return
        Q.append(qm)
        B.append(Tm)
        out_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.cvtColor(last[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f'pose_{len(Q):02d}_raw.png'), img)
        cv2.polylines(img, [last[1].astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / f'pose_{len(Q):02d}.jpg'), img)
        print(f'pose {i}: kept ({len(Q)} so far): tag in {n} frames, {1000 * np.linalg.norm(Tm[:3, 3]):.0f} mm '
              f'away, frame-to-frame spread {spread:.1f} mm', flush=True)

    if args.manual:
        # You move the arm (your own tools); the script only watches.
        cap = open_camera(args)
        K, seq, i = cap.K, 0, 0
        try:
            while True:
                if input(f'[{len(Q)} kept] put the arm still where the camera sees the tag, then Enter '
                         '(q + Enter to finish): ').strip().lower() == 'q':
                    break
                i += 1
                Tm, qm, n, spread, last, seq = record_still(cap, joints, args, K, detector, seq)
                keep(i, Tm, qm, n, spread, last)
        finally:
            cap.stop_capture()
        save_and_solve()
        return

    q0 = q_now.copy() if args.base == 'current' else np.asarray(args.q, float)
    lower, upper = urdf.arm_limits()
    q_hand = joint_map(urdf, None, joints.hand) if joints is not None and joints.hand is not None else {}
    check = TableCheck(urdf, args.table_z, args.table_box, q_hand)
    c0, c_now = check.clearance(q0), check.path_clearance(q_now, q0)
    print(f'now (deg): {np.round(np.degrees(q_now), 1).tolist()}' + ('' if q_hand else '  (no hand joints: fingers at 0)'))
    print(f'base ({args.base}, deg): {np.round(np.degrees(q0), 1).tolist()}: lowest point over the table '
          f'{100 * c0:.1f} cm above it; the move from now to it passes {100 * c_now:.1f} cm above it')
    if min(c0, c_now) < args.min_clear:
        sys.exit(f'the base pose (or the move to it) comes within {100 * args.min_clear:.0f} cm of the table: '
                 'move the arm up first, or change --q / --table_box / --min_clear')
    # The camera, once: a snapshot for the preview, and where the tag sits on the hand (seen now, placed with the
    # current T_RC) to keep only poses where the camera would still see it.
    sys.path.insert(0, str(_FP_DIR))
    from live_tracking_replay import T_RC_DEFAULT
    T_RC = load_T(args.calibration) if args.calibration else np.asarray(T_RC_DEFAULT, float)
    cap, snap, view, K = None, None, None, None
    try:
        cap = open_camera(args)
        K = cap.K
    except Exception as e:
        if not args.preview:
            raise
        print(f'[preview] no camera ({e}): the robot and the table only', flush=True)
    if cap is not None:
        Ts, seq, t0 = [], 0, time.time()
        while time.time() - t0 < 1.5:
            out = cap.latest(since_seq=seq, timeout=0.5)
            if out is None:
                continue
            seq, _, rgb, depth = out
            snap = (rgb, depth)
            T, _ = detect_tag(rgb, depth, K, args.tag_size, args.tag_id, detector, not args.no_depth_refine)
            if T is not None:
                Ts.append(T)
        if args.no_visibility:
            pass
        elif len(Ts) >= 3:
            qq = dict(q_hand)
            qq.update(zip(ARM_JOINTS, q_now))
            T_ee_tag = inv(urdf.fk(qq)[args.ee_link]) @ T_RC @ mean_pose(Ts)
            view = TagView(urdf, args.ee_link, T_RC, K, (args.width, args.height), T_ee_tag, args.tag_size, q_hand,
                           args.tag_margin_px, args.max_tag_angle)
            a = view.angle_deg(q_now)
            print(f'tag seen now in {len(Ts)} frames, ' + (f'{a:.0f} deg from facing the camera' if a is not None
                  else 'near the image edge') + f': keeping poses within {args.max_tag_angle:.0f} deg', flush=True)
        else:
            print('tag NOT seen at the current pose: planning WITHOUT the visibility test. Jog the arm to where the '
                  'camera sees the tag (and use --base current), or pass --no_visibility.', flush=True)
    poses, clears = safe_poses(q0, args.n, args.delta_deg, args.wrist_deg, lower, upper, args.seed, check,
                               args.min_clear, accept=view)
    for q, c in zip(poses, clears):
        a = view.angle_deg(q) if view is not None else None
        print(f'    {np.round(np.degrees(q), 1).tolist()}  clearance {100 * c:.1f} cm'
              + (f'  tag {a:.0f} deg' if a is not None else ''))
    if args.preview:
        if cap is not None:
            cap.stop_capture()
        preview_poses(args, urdf, [q0] + poses, [c0] + clears, check, T_RC, snap, K, q_hand, view)
        return
    if args.dry_run or not poses:
        cap.stop_capture()
        if not poses:
            sys.exit('no poses passed: smaller --delta_deg / --wrist_deg, a lower --min_clear, or another base')
        return
    durs = [max(args.move_s, math.pi / 2 * float(np.degrees(np.max(np.abs(q - q0)))) / args.max_joint_speed_deg)
            for q in poses]
    print(f'Each pose: base -> pose (at most {args.max_joint_speed_deg:.0f} deg/s on any joint: {min(durs):.1f}-'
          f'{max(durs):.1f} s), hold {args.settle_s + args.hold_s:.1f} s, back to the base. '
          f'About {sum(2 * d + args.settle_s + args.hold_s for d in durs) / 60:.1f} min.')
    if input(f'The arm will move to the base ({args.home_s} s), then through these {len(poses)} poses. '
             'Keep a hand on the e-stop. Type yes to go: ') != 'yes':
        cap.stop_capture()
        sys.exit('not moving')
    arm_pub = rospy.Publisher('/iiwa/joint_cmd', JointState, queue_size=1)
    hand_pub = rospy.Publisher('/sharpa/joint_cmd', JointState, queue_size=1)
    hold_hand = np.asarray(joints.hand.position, float)
    time.sleep(0.5)                      # let the publishers connect

    def send(q):
        m = JointState()
        m.header.stamp = rospy.Time.now()
        m.position = [float(v) for v in q]
        arm_pub.publish(m)
        h = JointState()
        h.header.stamp = m.header.stamp
        h.position = [float(v) for v in hold_hand]
        hand_pub.publish(h)

    def move(a, b, secs):
        # never faster than --max_joint_speed_deg on any joint (the cosine profile peaks at pi/2 x the mean speed)
        secs = max(secs, math.pi / 2 * float(np.degrees(np.max(np.abs(b - a)))) / args.max_joint_speed_deg)
        n = max(1, int(secs * 60))
        for k in range(1, n + 1):                                    # a smooth (cosine) move at 60 Hz
            send(a + (0.5 - 0.5 * math.cos(math.pi * k / n)) * (b - a))
            time.sleep(1 / 60)
        return b.copy()

    seq = 0
    try:
        current = q_now.copy() if np.allclose(q_now, q0) else move(q_now, q0, args.home_s)
        Tm, qm, n, spread, last, seq = record_still(cap, joints, args, K, detector, seq, send, current)
        keep(0, Tm, qm, n, spread, last)
        for i, target in enumerate(poses, 1):
            current = move(q0, target, args.move_s)                   # always from the base: the checked path
            Tm, qm, n, spread, last, seq = record_still(cap, joints, args, K, detector, seq, send, current)
            keep(i, Tm, qm, n, spread, last)
            current = move(target, q0, args.move_s)
    finally:
        cap.stop_capture()
        save_and_solve()


class RecordPreview:
    """Browser preview of the recorder's own frames; never opens a second camera."""

    def __init__(self, port):
        import socket
        import viser
        self.server = viser.ViserServer(host='0.0.0.0', port=port, label='AprilTag calibration')
        self.status = self.server.gui.add_markdown('Waiting for camera frames…')
        self.image = self.server.gui.add_image(np.zeros((540, 960, 3), np.uint8),
                                               label='Live AprilTag detection', format='jpeg', jpeg_quality=85)
        self.last_update = 0.0
        actual_port = self.server.get_port()
        print(f'[preview] http://{socket.gethostname()}:{actual_port} '
              f'(ssh -L {actual_port}:localhost:{actual_port} tylerlum@{socket.gethostname()})', flush=True)

    def update(self, rgb, corners, tag_seen, fresh, still, moved, count, window):
        import cv2
        now = time.monotonic()
        if now - self.last_update < 0.1:
            return
        self.last_update = now
        frame = np.ascontiguousarray(rgb.copy())
        if corners is not None:
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            for i, xy in enumerate(corners):
                cv2.putText(frame, str(i), tuple(xy.astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
        if not fresh:
            state = 'FRI STALE — samples paused'
        elif not still:
            state = 'Moving — pause to capture'
        elif not moved:
            state = 'Pose already captured — move to a new pose'
        elif not tag_seen:
            state = 'Waiting for tag'
        else:
            state = 'Averaging tag pose'
        visible = 'TAG SEEN' if tag_seen else 'TAG NOT SEEN'
        color = (0, 255, 0) if tag_seen else (255, 60, 60)
        cv2.putText(frame, f'{visible} | samples {count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if frame.shape[1] > 960:
            frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
        self.image.image = frame
        self.status.content = f'**{visible}** · **Samples: {count}** · {state} · averaging frames: {window}'

    def close(self):
        self.server.stop()


def run_record(args):
    """Teach-pad mode: you move the arm (FRI teach pad, joint by joint); the robot keeps publishing its joint states.
    This logs every frame (time, joint angles, the tag's pose or none) and a small video, and captures one sample
    each time the arm stops: still for --settle_s (every joint within --still_tol_deg), then the tag averaged over
    --hold_s while it stays still. The next sample needs the arm to move --min_move_deg first. Ctrl-C ends it, then
    it solves (and scores the current T_RC_DEFAULT, or --check, on the same samples)."""
    import cv2
    joints = Joints()
    if not joints.wait():
        sys.exit('no /iiwa/joint_states: is the robot up?')
    cap = open_camera(args)
    K, detector, seq = cap.K, make_detector(), 0
    out_dir = Path(args.out_dir) / f'record_{time.strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    video = None
    preview = None
    Q, B, stream = [], [], []
    win, last_q, t_status = [], None, 0.0
    print(f'[record] move the arm with the teach pad; stop at each pose for ~{args.settle_s + args.hold_s:.1f} s. '
          f'Ctrl-C when done (aim for 15-25 stops, the wrist turned 20-30 deg about different axes). -> {out_dir}',
          flush=True)
    try:
        if args.view:
            preview = RecordPreview(args.view_port)
        while True:
            out = cap.latest(since_seq=seq, timeout=1.0)
            if out is None or joints.arm is None:
                continue
            seq, _, rgb, depth = out
            now, q = time.time(), np.asarray(joints.arm.position[:7], float)
            T, c = detect_tag(rgb, depth, K, args.tag_size, args.tag_id, detector, not args.no_depth_refine)
            stream.append(np.r_[now, q, (T if T is not None else np.full((4, 4), np.nan)).reshape(-1)])
            still = joints.still_for(args.settle_s, args.still_tol_deg)
            moved = last_q is None or float(np.degrees(np.max(np.abs(q - last_q)))) > args.min_move_deg
            if not (still and moved):
                win = []
            elif T is not None:
                win.append((now, q, T, rgb, c))
                if win[-1][0] - win[0][0] >= args.hold_s and len(win) >= 3:
                    Tm = mean_pose([w[2] for w in win])
                    spread = 1000 * np.max(np.linalg.norm(np.stack([w[2][:3, 3] for w in win]) - Tm[:3, 3], axis=1))
                    Q.append(np.mean([w[1] for w in win], axis=0))
                    B.append(Tm)
                    img = cv2.cvtColor(win[-1][3], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_dir / f'sample_{len(Q):02d}_raw.png'), img)
                    cv2.polylines(img, [win[-1][4].astype(np.int32)], True, (0, 255, 0), 2)
                    cv2.imwrite(str(out_dir / f'sample_{len(Q):02d}.jpg'), img)
                    print(f'\n[record] sample {len(Q)}: tag in {len(win)} frames, '
                          f'{1000 * np.linalg.norm(Tm[:3, 3]):.0f} mm away, spread {spread:.1f} mm; '
                          f'joints (deg) {np.round(np.degrees(Q[-1]), 1).tolist()}', flush=True)
                    last_q, win = q.copy(), []
            if preview is not None:
                preview.update(rgb, c, T is not None, joints.fresh(), still, moved, len(Q), len(win))
            if args.video and len(stream) % 3 == 0:                   # ~10 fps, half size
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if c is not None:
                    cv2.polylines(frame, [c.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(frame, f'{"still" if still else "moving"}  tag {"seen" if T is not None else "-"}  '
                            f'samples {len(Q)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                if video is None:
                    video = cv2.VideoWriter(str(out_dir / 'record.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10,
                                            (frame.shape[1], frame.shape[0]))
                video.write(frame)
            if now - t_status > 1.0:
                t_status = now
                arm = ('JOINT STATES STALE (FRI dropped? restart it; samples paused)' if not joints.fresh()
                       else ('STILL ' if still else 'moving'))
                print(f'\r[record] {arm}  tag {"seen" if T is not None else "NOT seen"}  samples {len(Q)}   ',
                      end='', flush=True)
    except KeyboardInterrupt:
        print('\n[record] stopped', flush=True)
    finally:
        cap.stop_capture()
        if video is not None:
            video.release()
        np.savez(out_dir / 'samples.npz', q=np.array(Q), T_cam_tag=np.array(B), K=K, tag_size=args.tag_size,
                 tag_id=args.tag_id, ee_link=args.ee_link, urdf=str(args.urdf),
                 stream=np.array(stream), **calibration_metadata(args, Q))       # every frame: t, 7 joints, the tag's 4x4 (NaN when not seen)
        print(f'{len(Q)} samples, {len(stream)} frames -> {out_dir / "samples.npz"}', flush=True)
        if preview is not None:
            preview.close()
    if len(Q) >= 3:
        args.file = str(out_dir / 'samples.npz')
        run_solve(args)


def score_fixed(A, B, T_RC):
    """Hold T_RC fixed and fit only the tag's place on the hand: how well does this T_RC explain the samples?"""
    rel = [inv(a) @ T_RC @ b for a, b in zip(A, B)]
    y0 = np.r_[Rotation.from_matrix(np.stack([r[:3, :3] for r in rel])).mean().as_rotvec(),
               np.mean([r[:3, 3] for r in rel], axis=0)]

    def res(y):
        Y = make(y[:3], y[3:])
        out = []
        for a, b in zip(A, B):
            D = inv(T_RC @ b) @ (a @ Y)
            out.append(np.r_[ROT_SCALE * Rotation.from_matrix(D[:3, :3]).as_rotvec(), D[:3, 3]])
        return np.concatenate(out)

    r = least_squares(res, y0, loss='soft_l1', f_scale=0.003)
    rr = res(r.x).reshape(-1, 6)
    return 1000 * np.linalg.norm(rr[:, 3:], axis=1), np.degrees(np.linalg.norm(rr[:, :3], axis=1) / ROT_SCALE)


def run_solve(args):
    d = np.load(args.file)
    urdf = Urdf(str(d['urdf']) if 'urdf' in d and Path(str(d['urdf'])).is_file() else args.urdf)
    ee = str(d['ee_link']) if 'ee_link' in d else args.ee_link
    A = [urdf.fk(dict(zip(ARM_JOINTS, q)))[ee] for q in d['q']]
    B = list(d['T_cam_tag'])
    sol = solve(A, B)
    T_ref = load_T(args.calibration) if getattr(args, 'calibration', None) else None
    report(sol, T_ref)
    checks = []
    if getattr(args, 'check', None):
        checks.append((args.check, load_T(args.check)))
    else:
        sys.path.insert(0, str(_FP_DIR))
        from live_tracking_replay import T_RC_DEFAULT
        checks.append(('T_RC_DEFAULT (the current default)', np.asarray(T_RC_DEFAULT, float)))
    for name, T_chk in checks:
        mm, deg = score_fixed(A, B, T_chk)
        d = inv(T_chk) @ sol['T_RC']
        print(f'check {name}: held fixed, the samples fit to rms {np.sqrt(np.mean(mm ** 2)):.1f} mm, '
              f'{np.sqrt(np.mean(deg ** 2)):.2f} deg (the new fit: {sol["rms_mm"]:.1f} mm, {sol["rms_deg"]:.2f} deg); '
              f'it differs from the new fit by {1000 * np.linalg.norm(d[:3, 3]):.1f} mm, '
              f'{np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())):.2f} deg')
    stem = str(Path(args.file).with_name('T_RC'))
    save_T(sol['T_RC'], stem)
    np.savetxt(str(Path(args.file).with_name('T_ee_tag.txt')), sol['T_ee_tag'])


def run_selftest(args):
    rng = np.random.default_rng(args.seed)
    urdf = Urdf(args.urdf)
    lower, upper = urdf.arm_limits()
    q0 = np.asarray(DEFAULT_ARM_Q, float)
    X = from_xyz_rpy([0.4, -1.3, 1.0], [-115, 3, -17])          # roughly T_RC_DEFAULT's place
    Y = from_xyz_rpy([0.02, -0.05, 0.12], [10, -80, 30])         # the tag somewhere on the hand
    for noise_mm, noise_deg in ((0.0, 0.0), (1.0, 0.5), (2.0, 1.0)):
        A, B = [], []
        rq = np.random.default_rng(args.seed)
        span = np.radians([args.delta_deg] * 4 + [args.wrist_deg] * 3)
        for q in [q0] + [np.clip(q0 + rq.uniform(-1, 1, 7) * span, lower, upper) for _ in range(args.n)]:
            Ai = urdf.fk(dict(zip(ARM_JOINTS, q)))[args.ee_link]
            Bi = inv(X) @ Ai @ Y
            Bi = Bi @ make(Rotation.from_rotvec(rng.normal(0, math.radians(noise_deg), 3)).as_matrix(),
                           rng.normal(0, noise_mm / 1000, 3))
            A.append(Ai)
            B.append(Bi)
        print(f'--- noise {noise_mm} mm, {noise_deg} deg per tag pose, {len(A)} poses')
        report(solve(A, B), X)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, camera=True):
        p.add_argument('--urdf', default=str(URDF))
        p.add_argument('--ee_link', default='iiwa14_link_7', help='the link the tag is fixed to (or any link rigid '
                       'with it; the hand base is rigid with link 7)')
        if camera:
            p.add_argument('--serial_number', type=str, default='15107')
            p.add_argument('--camera_resolution', choices=['HD1080', 'HD720', 'VGA'], default='HD1080')
            p.add_argument('--camera_upsidedown', action='store_true')
            p.add_argument('--width', type=int, default=960)
            p.add_argument('--height', type=int, default=540)
            p.add_argument('--exposure', type=int, default=25)
            p.add_argument('--gain', type=int, default=40)

    v = sub.add_parser('viser', help='method 1: align the live point cloud to the robot model with sliders')
    common(v)
    v.add_argument('--calibration', default=None, help='start from this T_RC (.txt / .npy)')
    v.add_argument('--start', choices=['default', 'guess'], default='default',
                   help="without --calibration: live_tracking_replay.T_RC_DEFAULT (the latest saved calibration; "
                        "default), or 'guess': the camera at --cam_xyz looking at --look_at")
    v.add_argument('--cam_xyz', type=float, nargs=3, default=CAM_GUESS_XYZ, help='the camera (robot frame, m)')
    v.add_argument('--look_at', type=float, nargs=3, default=CAM_GUESS_LOOK_AT,
                   help='a point the camera looks at (robot frame, m)')
    v.add_argument('--no_ros', action='store_true', help='no joint states: the model at --q')
    v.add_argument('--q', type=float, nargs=7, default=DEFAULT_ARM_Q,
                   help="the arm (rad) without joint states; default simtoolreal's default pose")
    v.add_argument('--port', type=int, default=8110, help='viser: the first free port from this up (a stable URL)')
    v.add_argument('--stride', type=int, default=2, help='point cloud pixel stride')
    v.add_argument('--max_depth', type=float, default=2.5)
    v.add_argument('--point_size', type=float, default=0.004)
    v.add_argument('--robot_opacity', type=float, default=0.6)
    v.add_argument('--hz', type=float, default=10.0)
    v.add_argument('--out', default='/juno/u/tylerlum/camera_calibration/T_RC_viser',
                   help='file stem for Save (adds a timestamp)')
    v.add_argument('--table_z', type=float, default=0.53, help="the table top's height in the robot frame (m)")

    c = sub.add_parser('collect', help='method 2: move the arm, record the tag (then solves)')
    common(c)
    c.add_argument('--tag_size', type=float, required=True, help="the tag's black square side (m), as printed")
    c.add_argument('--tag_id', type=int, default=0)
    c.add_argument('--n', type=int, default=30, help='random poses around the current one (synthetic check, 2 mm / '
                   '1 deg tag noise: 20 poses 3-21 mm, 40 poses 3-9 mm; 40 at --delta_deg 15: 3-5 mm, 0.1-0.3 deg)')
    c.add_argument('--delta_deg', type=float, default=10.0, help='joints 1-4: uniform within +-this')
    c.add_argument('--wrist_deg', type=float, default=25.0, help='joints 5-7: uniform within +-this')
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--base', choices=['current', 'default'], default='current',
                   help="the poses' centre: the arm's current pose (jog it where the camera sees the tag first), "
                        "or simtoolreal's default arm pose (--q)")
    c.add_argument('--max_tag_angle', type=float, default=55.0,
                   help='keep poses where the tag faces the camera within this (deg); from the tag seen at start')
    c.add_argument('--tag_margin_px', type=float, default=40.0, help='...and stays this far inside the image')
    c.add_argument('--no_visibility', action='store_true', help='plan without the tag-in-view test')
    c.add_argument('--q', type=float, nargs=7, default=CALIB_BASE_Q,
                   help='the base pose (rad) for --base default and without joint states: the 13 Oct tag pose')
    c.add_argument('--home_s', type=float, default=6.0, help='the first move, from now to the base pose (s)')
    c.add_argument('--move_s', type=float, default=3.0, help='the shortest move (s)')
    c.add_argument('--max_joint_speed_deg', type=float, default=10.0,
                   help='no joint faster than this (deg/s); longer moves take longer')
    c.add_argument('--settle_s', type=float, default=1.0)
    c.add_argument('--hold_s', type=float, default=1.0)
    # Default to tag-corner PnP. Bad ZED depth on the printed tag can severely corrupt translation.
    depth_flags = c.add_mutually_exclusive_group()
    depth_flags.add_argument('--depth_refine', dest='no_depth_refine', action='store_false',
                             help='EXPERIMENTAL: rescale tag translation using ZED depth (default OFF)')
    depth_flags.add_argument('--no_depth_refine', dest='no_depth_refine', action='store_true',
                             help='use tag-corner PnP only (the default; measure tag size accurately)')
    c.set_defaults(no_depth_refine=True)
    c.add_argument('--dry_run', action='store_true', help='print the poses (and their table clearance) only')
    c.add_argument('--preview', action='store_true',
                   help='show the planned poses in viser over the live point cloud (port 8115+); moves nothing')
    c.add_argument('--no_ros', action='store_true', help='--preview without the robot: plan from the default pose')
    c.add_argument('--manual', action='store_true',
                   help='move the arm yourself; press Enter at each pose to record it (the script sends nothing)')
    c.add_argument('--table_z', type=float, default=0.53, help="the table top's height (robot frame, m)")
    c.add_argument('--table_box', type=float, nargs=4, default=[-0.6, 0.6, -1.4, -0.55],
                   metavar=('X0', 'X1', 'Y0', 'Y1'), help='where the table is (robot frame, m): only robot points '
                   'over this box count for the clearance. The near edge -0.55: at the default pose the elbow '
                   'hangs below table height out to y -0.53, so the real table must start beyond it')
    c.add_argument('--min_clear', type=float, default=0.05,
                   help='every pose, and the move to it from the base, keeps the robot this far above the table (m)')
    c.add_argument('--calibration', default=None, help='a T_RC to compare the result with')
    c.add_argument('--out_dir', default='/juno/u/tylerlum/camera_calibration')

    r = sub.add_parser('record', help='method 2 by teach pad: you move the arm, it records at every stop (then solves)')
    common(r)
    r.add_argument('--tag_size', type=float, required=True, help="the tag's black square side (m), as printed")
    r.add_argument('--tag_id', type=int, default=0)
    r.add_argument('--view', action='store_true', help='live browser preview with tag outline and sample status')
    r.add_argument('--view_port', type=int, default=8111, help='browser preview port (default 8111)')
    r.add_argument('--settle_s', type=float, default=0.7, help='still this long before recording a stop')
    r.add_argument('--hold_s', type=float, default=1.0, help='then average the tag over this long')
    r.add_argument('--still_tol_deg', type=float, default=0.05, help='"still": every joint within this (deg)')
    r.add_argument('--min_move_deg', type=float, default=3.0, help='a new stop needs this much motion first (deg)')
    # Default to tag-corner PnP. Bad ZED depth on the printed tag can severely corrupt translation.
    depth_flags = r.add_mutually_exclusive_group()
    depth_flags.add_argument('--depth_refine', dest='no_depth_refine', action='store_false',
                             help='EXPERIMENTAL: rescale tag translation using ZED depth (default OFF)')
    depth_flags.add_argument('--no_depth_refine', dest='no_depth_refine', action='store_true',
                             help='use tag-corner PnP only (the default; measure tag size accurately)')
    r.set_defaults(no_depth_refine=True)
    r.add_argument('--no_video', dest='video', action='store_false', help='skip record.mp4 (10 fps, half size)')
    r.add_argument('--check', default=None, help='score this T_RC on the samples (default: T_RC_DEFAULT)')
    r.add_argument('--calibration', default=None, help='a T_RC to compare the new fit with')
    r.add_argument('--out_dir', default='/juno/u/tylerlum/camera_calibration')

    s = sub.add_parser('solve', help='method 2: solve T_RC from collected samples')
    common(s, camera=False)
    s.add_argument('file')
    s.add_argument('--calibration', default=None, help='a T_RC to compare the result with')
    s.add_argument('--check', default=None, help='score this T_RC on the samples (default: T_RC_DEFAULT)')

    t = sub.add_parser('selftest', help='the solver on synthetic data')
    common(t, camera=False)
    t.add_argument('--n', type=int, default=30)
    t.add_argument('--delta_deg', type=float, default=10.0)
    t.add_argument('--wrist_deg', type=float, default=25.0)
    t.add_argument('--seed', type=int, default=0)

    args = ap.parse_args()
    {'viser': run_viser, 'collect': run_collect, 'record': run_record, 'solve': run_solve,
     'selftest': run_selftest}[args.cmd](args)


if __name__ == '__main__':
    main()
