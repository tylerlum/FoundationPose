"""MuJoCo sweep cell for tracker tests: iiwa14 + Sharpa hand + table + red brush (textured) + gold 29 mm cube.

World frame = ROBOT BASE frame (the frame of T_RC), so the tracker's T_RC applies directly.
Camera = ZED2 rectified left at T_RC (OpenCV optical) with the 960x540 K; a second camera 'zed_right' sits
BASELINE m along the left camera's +x for stereo depth.
The brush is a free body in hand-held mode: each frame its pose is set from the palm pose @ T_palm_brush
(kinematic grasp); the cube is a free body under physics on the table.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path('/home/tylerlum/github_repos/simtoolreal_dev')
sys.path.insert(0, str(REPO))
from realworld.mujoco_rollout.stir_mujoco_sim import IIWA14_XML, SHARPA_XML, TRAINING_VISUAL_URDF  # noqa: E402
from realworld.mujoco_rollout.robot_visuals import apply_training_robot_visuals  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get('FPSIM_MESH_DIR', '/juno/u/tylerlum/tracking_meshes_20261026/red_brush'))  # red_brush.obj + material_0.png
# Camera (env FPSIM_CAM): the calibrated poses compared on 19 Nov; K at 960x540.
CAMS = {
    'zed2_172707': ('/juno/u/tylerlum/camera_2_calibration_27432424/record_visual_20261118_172707/T_RC.txt',
                    (528.866, 528.866, 486.258, 276.284)),   # current, front-on
    'zed2_161129': ('/juno/u/tylerlum/camera_2_calibration_27432424/record_visual_20261118_161129/T_RC.txt',
                    (528.866, 528.866, 486.258, 276.284)),   # previous, diagonal from the robot's right
    'zed1': ('/juno/u/tylerlum/camera_calibration/replay_20261016_161152/T_RC.txt',
             (692.022, 692.022, 480.106, 282.767)),           # ZED1, left side
}
CAM = os.environ.get('FPSIM_CAM', 'zed2_172707')
TRC = CAMS[CAM][0]
_fx, _fy, _cx, _cy = CAMS[CAM][1]
K960 = np.array([[_fx, 0, _cx], [0, _fy, _cy], [0, 0, 1.0]])
W, H = 960, 540
BASELINE = 0.12
TABLE_TOP = 0.540
# Stir scene table (env frame root (0.1967, -0.1683, 0.38), size 1.1684 x 0.7366) moved into the robot frame
# (base at env (0, 0.8, 0)), top raised to the measured 0.540 m.
TABLE_CENTER_XY = (0.1967, -0.9683)
TABLE_SIZE = (1.1684, 0.7366, 0.30)
CUBE_HALF = 0.0145


def _cam_quat_mj(R_opt):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R_opt @ np.diag([1.0, -1.0, -1.0])).flatten())
    return q


def _set_intrinsics(cam, K, w, h):
    cam.sensor_size = [float(w), float(h)]
    cam.resolution = [w, h]
    cam.focal_pixel = [K[0, 0], K[1, 1]]
    cam.principal_pixel = [w / 2 - K[0, 2], h / 2 - K[1, 2]]


def build(T_RC=None, cube_xy=(0.10, -0.90), robot_visuals='training'):
    T_RC = np.loadtxt(TRC) if T_RC is None else T_RC
    T_RC = np.vstack([T_RC[:3], [0, 0, 0, 1]]) if T_RC.shape[0] == 3 else T_RC
    spec = mujoco.MjSpec.from_file(str(IIWA14_XML))
    spec.option.timestep = 1 / 600
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for key in list(spec.keys):
        spec.delete(key)
    sharpa = mujoco.MjSpec.from_file(str(SHARPA_XML))
    site = next(s for s in spec.sites if s.name == 'attachment_site')
    site.attach_body(sharpa.worldbody, 'palm', '')
    if robot_visuals != 'legacy':
        apply_training_robot_visuals(spec, TRAINING_VISUAL_URDF, mount_only=robot_visuals == 'mount')
    for b in spec.bodies:
        b.gravcomp = 1.0
    robot_bodies = {b.name for b in spec.bodies}

    wb = spec.worldbody
    # floor + back wall (grey, a bit of texture so stereo has something)
    tex = spec.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.42, 0.42, 0.44], rgb2=[0.36, 0.36, 0.38], width=512, height=512)
    mat = spec.add_material(name='floor_mat', texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
    g = wb.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], material='floor_mat')
    g = wb.add_geom(name='wall', type=mujoco.mjtGeom.mjGEOM_BOX, size=[3, 0.02, 1.5], pos=[0, 0.9, 1.5],
                    rgba=[0.80, 0.78, 0.74, 1], contype=0, conaffinity=0)
    # table with a cloth-like texture (stereo needs texture; the real table has a grey cloth)
    spec.add_texture(name='cloth', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     mark=mujoco.mjtMark.mjMARK_RANDOM, random=0.25, rgb1=[0.55, 0.55, 0.57], rgb2=[0.55, 0.55, 0.57],
                     markrgb=[0.35, 0.35, 0.37], width=1024, height=1024)
    cm = spec.add_material(name='cloth_mat', texrepeat=[3, 3], texuniform=True)
    cm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'cloth'
    tb = wb.add_body(name='table', pos=[*TABLE_CENTER_XY, TABLE_TOP - TABLE_SIZE[2] / 2])
    tb.add_geom(name='table_geom', type=mujoco.mjtGeom.mjGEOM_BOX, size=np.array(TABLE_SIZE) / 2, material='cloth_mat',
                friction=[0.8, 0.02, 0.001])

    # brush: textured visual mesh at the mesh's own frame (body frame == mesh frame == FoundationPose object frame)
    spec.add_texture(name='brush_tex', type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(ASSETS / 'material_0.png'))
    bm = spec.add_material(name='brush_mat')
    bm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'brush_tex'
    spec.add_mesh(name='brush_mesh', file=str(ASSETS / 'red_brush.obj'))
    bb = wb.add_body(name='brush', pos=[0.3, -0.8, TABLE_TOP + 0.02])
    bb.add_freejoint(name='brush_free')
    bb.add_geom(name='brush_geom', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='brush_mesh', material='brush_mat',
                contype=0, conaffinity=0, mass=0.08)
    # cube: physical box, gold
    cb = wb.add_body(name='cube', pos=[*cube_xy, TABLE_TOP + CUBE_HALF + 1e-3])
    cb.add_freejoint(name='cube_free')
    cb.add_geom(name='cube_geom', type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, rgba=[0.85, 0.66, 0.18, 1],
                mass=0.03, friction=[0.8, 0.02, 0.001])

    for name, off in (('zed_left', 0.0), ('zed_right', BASELINE)):
        cam = wb.add_camera(name=name)
        cam.pos = T_RC[:3, 3] + T_RC[:3, :3] @ np.array([off, 0, 0])
        cam.quat = _cam_quat_mj(T_RC[:3, :3])
        _set_intrinsics(cam, K960, W, H)
    for p, d in (([0.2, -1.6, 2.2], 0.8), ([-0.8, -0.6, 2.0], 0.5), ([0.9, -0.4, 1.8], 0.4)):
        li = wb.add_light(pos=p, dir=[0, 0.3, -1], castshadow=True)
        li.diffuse = [d, d, d]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    model = spec.compile()
    return model, mujoco.MjData(model), T_RC, robot_bodies


def arm_joint_ids(model):
    return [model.joint(f'joint{i}').id for i in range(1, 8)]


def hand_joint_ids(model):
    return [j for j in range(model.njnt) if model.joint(j).name.startswith('palmleft_')]


def body_T(model, data, name):
    b = model.body(name).id
    T = np.eye(4)
    T[:3, :3] = data.xmat[b].reshape(3, 3)
    T[:3, 3] = data.xpos[b]
    return T


def set_free(model, data, joint, T):
    adr = model.joint(joint).qposadr[0]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T[:3, :3].flatten())
    data.qpos[adr:adr + 3] = T[:3, 3]
    data.qpos[adr + 3:adr + 7] = q
    dof = model.joint(joint).dofadr[0]
    data.qvel[dof:dof + 6] = 0


PALM = 'palmleft_hand_C_MC'


def set_hand_curl(model, data, curl, thumb=None):
    """Power-grasp curl: flex MCP_FE/PIP/DIP of the four fingers by `curl` (fraction of range); thumb opposes."""
    thumb = curl if thumb is None else thumb
    for j in hand_joint_ids(model):
        n = model.joint(j).name
        lo, hi = model.jnt_range[j]
        a = model.jnt_qposadr[j]
        if n.endswith(('MCP_FE', 'PIP', 'DIP')) and 'thumb' not in n:
            data.qpos[a] = lo + curl * (hi - lo) * (0.75 if n.endswith('MCP_FE') else 0.6)
        elif 'thumb' in n and n.endswith(('CMC_FE', 'MCP_FE', 'IP')):
            data.qpos[a] = lo + thumb * (hi - lo) * 0.5
        else:
            data.qpos[a] = 0.0


def grasp_T(along, normal, roll_deg, side=0.0):
    """T_palm_brush: brush x (handle axis) across the palm (palm y), brush origin `along` m along the finger direction
    (-palm x), `normal` m out of the palm (palm z); roll about the handle axis."""
    from scipy.spatial.transform import Rotation as Rot
    T = np.eye(4)
    # brush axes in the palm frame: x_b = palm y, z_b = palm z (bristles out of the palm) before roll
    Rb = np.stack([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], 1).astype(float)  # columns: x_b, y_b, z_b
    Rb = Rb @ Rot.from_euler('x', roll_deg, degrees=True).as_matrix()
    T[:3, :3] = Rb
    T[:3, 3] = [-along, side, normal]
    return T


def grasp_T_auto(model, data, roll_deg, slide=0.0, inset=0.0):
    """T_palm_brush from the curled hand (call after set_hand_curl + mj_forward): brush origin at the centre of the
    four fingers' proximal+middle phalanges (shifted `inset` m toward the palm), handle axis index->pinky, `slide` m
    along it; roll about the handle axis."""
    from scipy.spatial.transform import Rotation as Rot
    Tp = body_T(model, data, PALM)
    names = [f'palmleft_{f}_{s}' for f in ('index', 'middle', 'ring', 'pinky') for s in ('PP', 'MP')]
    P = np.array([data.xpos[model.body(n).id] for n in names])
    c = P.mean(0)
    a = data.xpos[model.body('palmleft_index_PP').id] - data.xpos[model.body('palmleft_pinky_PP').id]
    a /= np.linalg.norm(a)
    to_palm = Tp[:3, 3] - c
    to_palm -= a * (a @ to_palm)
    n = to_palm / np.linalg.norm(to_palm)
    origin = c + inset * n + slide * a
    x = a
    z = np.cross(x, n)
    Rw = np.stack([x, np.cross(z, x), z], 1) @ Rot.from_euler('x', roll_deg, degrees=True).as_matrix()
    Tw = np.eye(4)
    Tw[:3, :3], Tw[:3, 3] = Rw, origin
    return np.linalg.inv(Tp) @ Tw
