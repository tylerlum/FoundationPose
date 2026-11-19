"""Generate a synthetic ZED2 take of the sweep cell in MuJoCo, in the tracker's replay layout plus ground truth.

  MUJOCO_GL=egl python gen_take.py --scenario pick_sweep --out DIR [--fps 30]
DIR/replay/{rgb,right,depth_gt}/NNNNN.(png|npy), cam_K.txt, ts_ns.txt, masks_1/00000.png (brush), masks_2/00000.png
(cube); DIR/gt/{T_cam_brush,T_cam_cube,T_world_brush,T_world_cube}.npy (N,4,4), vis_brush.npy (visible px / full
silhouette px), px_brush.npy, attached.npy; DIR/preview.mp4. World frame = robot base frame (T_RC applies).
Depth is added later (stereo -> Fast-FS, see stereo_depth.py); depth_gt is MuJoCo's exact depth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_scene as S  # noqa: E402
from realworld.common import stir_scene  # noqa: E402

TT = S.TABLE_TOP
CUBE_XY = (-0.077, -0.910)                       # real cube, takes 1734-1738
BRUSH_REST_XY = (0.100, -0.787)                  # real brush start, take 173827
R_REST = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float).T  # columns = brush x,y,z in robot frame (x->-x, y->up, z->+y)
R_REST = np.stack([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], 1).astype(float)
BRUSH_Y_HALF = 0.049                             # head half-height when lying on its side
R_BRISTLE_DOWN = R_REST @ Rot.from_euler('x', 90, degrees=True).as_matrix()   # brush z (bristles) -> -z robot


def T_of(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def scenario(name):
    """Keyframes: (t_s, T_world_brush, curl, attached). The hand follows the brush through T_palm_brush while
    attached; before the grasp it is placed so the brush would sit in it (pre-grasp poses are lifted)."""
    rest = T_of(R_REST, [*BRUSH_REST_XY, TT + BRUSH_Y_HALF])
    up = lambda T, dz: T_of(T[:3, :3], T[:3, 3] + [0, 0, dz])  # noqa: E731
    at = lambda R, x, y, z: T_of(R, [x, y, z])                 # noqa: E731
    R45 = R_REST @ Rot.from_euler('x', 45, degrees=True).as_matrix()
    if name == 'pick_lift':
        # Like the real takes: approach the brush on the table, close, lift to 0.62-0.75 m, carry it toward and over
        # the cube (the hand stays on the brush), then hold.
        return [(0.0, up(rest, 0.12), 0.0, False), (1.2, up(rest, 0.01), 0.0, False), (1.4, rest, 0.0, False),
                (2.0, rest, 0.8, True), (2.8, up(rest, 0.10), 0.8, True),
                (3.6, at(R45, 0.05, -0.80, TT + 0.16), 0.8, True), (4.4, at(R45, -0.04, -0.86, TT + 0.12), 0.8, True),
                (5.2, at(R_REST, -0.06, -0.86, TT + 0.16), 0.8, True), (6.0, at(R45, 0.02, -0.82, TT + 0.20), 0.8, True),
                (7.0, at(R45, 0.02, -0.82, TT + 0.20), 0.8, True)]
    if name.startswith('heldout_'):
        # Held-out paths (not used for tuning): 7 random poses from the 40% most occluded + 3 random others,
        # random order by nearest neighbour, seed from the name (heldout_<seed>).
        import json as _json
        seed = int(name.split('_')[1])
        rs = np.random.RandomState(seed)
        res = [r for r in _json.load(open(Path(__file__).resolve().parent / 'pose_search.json')) if r['slide'] == 0.02]
        res.sort(key=lambda r: r['vis'])
        q = lambda r: np.array(r['q'])  # noqa: E731
        n40 = int(0.4 * len(res))
        pick = [res[i] for i in rs.choice(n40, 7, replace=False)] + [res[i] for i in rs.choice(np.arange(n40, len(res)), 3, replace=False)]
        start = pick.pop(int(np.argmax([r['vis'] for r in pick])))
        path, cur, left = [start], start, list(pick)
        while left:
            nxt = min(left, key=lambda r: np.linalg.norm(q(r) - q(cur)))
            left.remove(nxt)
            path.append(nxt)
            cur = nxt
        keys = [(0.0, q(start), 0.8, True)]
        for k, r in enumerate(path[1:]):
            keys.append((0.6 + 0.7 * (k + 1), q(r), 0.8, True))
        keys.append((keys[-1][0] + 1.0, q(path[-1]), 0.8, True))
        return keys
    if name in ('occlusion_tour', 'fast_wave'):
        import json as _json
        res = [r for r in _json.load(open(Path(__file__).resolve().parent / 'pose_search.json')) if r['slide'] == 0.02]
        q = lambda r: np.array(r['q'])  # noqa: E731
        res.sort(key=lambda r: r['vis'])
        low = res[:8] if name == 'occlusion_tour' else res[len(res) // 3: len(res) // 3 + 8]
        mids = [r for r in res if r['vis'] > 0.85]
        start = min(mids, key=lambda r: np.linalg.norm(r['pos'] - np.array([0.05, -0.80, TT + 0.15])))
        path, cur, left = [start], start, list(low)
        while left:
            nxt = min(left, key=lambda r: np.linalg.norm(q(r) - q(cur)))
            left.remove(nxt)
            path.append(nxt)
            cur = nxt
        dt = 1.0 if name == 'occlusion_tour' else 0.45
        keys = [(0.0, q(start), 0.8, True)]
        for k, r in enumerate(path[1:] + [start]):
            keys.append((0.6 + dt * (k + 1), q(r), 0.8, True))
        keys.append((keys[-1][0] + 0.6, q(start), 0.8, True))
        return keys
    raise ValueError(name)


def interp(keys, t):
    """Min-jerk position + slerp rotation between keyframes; curl linear in the same phase; attached from the later key."""
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return keys[0][1], keys[0][2], keys[0][3]
    if t >= ts[-1]:
        return keys[-1][1], keys[-1][2], keys[-1][3]
    i = int(np.searchsorted(ts, t)) - 1
    (t0, A, c0, a0), (t1, B, c1, a1) = keys[i], keys[i + 1]
    s = (t - t0) / (t1 - t0)
    s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    if np.ndim(A) == 1:  # joint-space keyframe (7 arm angles)
        T = (1 - s) * A + s * B
    else:
        rot = Slerp([0, 1], Rot.from_matrix([A[:3, :3], B[:3, :3]]))(s).as_matrix()
        T = T_of(rot, (1 - s) * A[:3, 3] + s * B[:3, 3])
    attached = a1 if (c0 + s * (c1 - c0)) > 0.5 else a0
    return T, c0 + s * (c1 - c0), attached


def retime(keys, model, data, arm_q, T_pb, vmax, wmax):
    """Stretch each joint-space segment so the brush's peak speed (min-jerk peak = 1.875 x mean) stays under vmax m/s
    and wmax deg/s. Brush-space segments are checked directly."""
    def brush_T(k):
        if np.ndim(k) == 1:
            data.qpos[arm_q] = k
            mujoco.mj_kinematics(model, data)
            return S.body_T(model, data, S.PALM) @ T_pb
        return k
    out = [keys[0]]
    t_prev_new = keys[0][0]
    for (t0, A, c0, a0), (t1, B, c1, a1) in zip(keys[:-1], keys[1:]):
        Ts = [brush_T((1 - u) * A + u * B) if np.ndim(A) == 1 else None for u in np.linspace(0, 1, 21)]
        if Ts[0] is None:
            Ts = [A, B]
        dist = sum(np.linalg.norm(Ts[i + 1][:3, 3] - Ts[i][:3, 3]) for i in range(len(Ts) - 1))
        ang = sum(np.degrees(Rot.from_matrix(Ts[i][:3, :3].T @ Ts[i + 1][:3, :3]).magnitude()) for i in range(len(Ts) - 1))
        dur = max(t1 - t0, 1.875 * dist / vmax, 1.875 * ang / wmax)
        t_prev_new += dur
        out.append((t_prev_new, B, c1, a1))
    return out


def ik_palm(model, data, arm_q_adr, arm_dof, T_target, iters=60, damp=1e-3, tol=1e-4):
    """Damped least squares on the 7 arm joints so the palm body reaches T_target (pos + rot). Returns (pos err, rot err)."""
    bid = model.body(S.PALM).id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    lo = model.jnt_range[[model.dof_jntid[d] for d in arm_dof], 0]
    hi = model.jnt_range[[model.dof_jntid[d] for d in arm_dof], 1]
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        p, Rm = data.xpos[bid], data.xmat[bid].reshape(3, 3)
        ep = T_target[:3, 3] - p
        er = Rot.from_matrix(T_target[:3, :3] @ Rm.T).as_rotvec()
        if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 10 * tol:
            break
        mujoco.mj_jacBody(model, data, jp, jr, bid)
        J = np.vstack([jp[:, arm_dof], 0.3 * jr[:, arm_dof]])
        e = np.r_[ep, 0.3 * er]
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
        q = data.qpos[arm_q_adr] + np.clip(dq, -0.2, 0.2)
        data.qpos[arm_q_adr] = np.clip(q, lo, hi)
    mujoco.mj_kinematics(model, data)
    return float(np.linalg.norm(T_target[:3, 3] - data.xpos[bid])), \
        float(np.degrees(np.linalg.norm(Rot.from_matrix(T_target[:3, :3] @ data.xmat[bid].reshape(3, 3).T).as_rotvec())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='pick_lift')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--roll', type=float, default=270.0, help='brush roll in the grasp (deg)')
    ap.add_argument('--slide', type=float, default=0.02)
    ap.add_argument('--curl', type=float, default=0.8)
    ap.add_argument('--vmax', type=float, default=0.9, help='peak brush speed m/s (real takes: p95 0.5-0.9)')
    ap.add_argument('--wmax', type=float, default=350.0, help='peak brush turn rate deg/s (real: p95 ~300)')
    a = ap.parse_args()
    out = Path(a.out)
    rd = out / 'replay'
    for sub in ('rgb', 'right', 'depth_gt', 'masks_1', 'masks_2'):
        (rd / sub).mkdir(parents=True, exist_ok=True)
    (out / 'gt').mkdir(exist_ok=True)

    model, data, T_RC, _ = S.build(cube_xy=CUBE_XY)
    arm_j = S.arm_joint_ids(model)
    arm_q = np.array([model.jnt_qposadr[j] for j in arm_j])
    arm_d = np.array([model.jnt_dofadr[j] for j in arm_j])
    hand_d = np.array([model.jnt_dofadr[j] for j in S.hand_joint_ids(model)])
    home = np.asarray(stir_scene.IIWA_HOME_JOINT_POS)
    data.qpos[arm_q] = home
    S.set_hand_curl(model, data, a.curl)
    mujoco.mj_forward(model, data)
    T_pb = S.grasp_T_auto(model, data, a.roll, slide=a.slide)
    T_bp = np.linalg.inv(T_pb)

    keys = scenario(a.scenario)
    keys = retime(keys, model, data, arm_q, T_pb, a.vmax, a.wmax)
    data.qpos[arm_q] = home
    mujoco.mj_kinematics(model, data)
    print('keyframe times', [round(k[0], 2) for k in keys])
    n = int(round(keys[-1][0] * a.fps)) + 1
    sub = int(round(1 / (a.fps * model.opt.timestep)))
    ren = mujoco.Renderer(model, S.H, S.W)
    opt_brush = mujoco.MjvOption()
    opt_brush.geomgroup[:] = 0
    opt_brush.geomgroup[2] = 1
    model.geom_group[model.geom('brush_geom').id] = 2
    gid_b, gid_c = model.geom('brush_geom').id, model.geom('cube_geom').id
    Tinv = np.linalg.inv(T_RC)
    cam_brush, cam_cube, w_brush, w_cube, vis, pxs, att, ik_err = [], [], [], [], [], [], [], []
    np.savetxt(rd / 'cam_K.txt', S.K960)
    prev_q = data.qpos[arm_q].copy()
    brush_free = T_of(R_REST, [*BRUSH_REST_XY, TT + BRUSH_Y_HALF])
    writer = imageio.get_writer(out / 'preview.mp4', fps=a.fps, codec='libx264', macro_block_size=1)
    ts_lines = []
    # put the arm at frame 0 (multi-start IK), then settle the cube
    k0 = interp(keys, 0.0)[0]
    T0 = None if np.ndim(k0) == 1 else k0 @ T_bp
    best = (0, k0) if T0 is None else None
    for seed in range(8 if T0 is not None else 0):
        data.qpos[arm_q] = home + (0 if seed == 0 else np.random.RandomState(seed).uniform(-0.8, 0.8, 7))
        e = ik_palm(model, data, arm_q, arm_d, T0, iters=400)
        if best is None or e[0] + e[1] / 100 < best[0]:
            best = (e[0] + e[1] / 100, data.qpos[arm_q].copy())
    data.qpos[arm_q] = best[1]
    prev_q = best[1].copy()
    if keys[0][3]:
        mujoco.mj_kinematics(model, data)
        brush_free = S.body_T(model, data, S.PALM) @ T_pb
    S.set_free(model, data, 'brush_free', brush_free)
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    for i in range(n):
        t = i / a.fps
        T_b, curl, attached = interp(keys, t)
        if not attached and i > 0 and att and att[-1]:
            attached = True  # never drop once grasped
        S.set_hand_curl(model, data, curl)
        if np.ndim(T_b) == 1:
            data.qpos[arm_q] = T_b
            perr, rerr = 0.0, 0.0
        else:
            perr, rerr = ik_palm(model, data, arm_q, arm_d, T_b @ T_bp, iters=200 if i == 0 else 40)
        q_new = data.qpos[arm_q].copy()
        for k in range(sub):  # kinematic arm, physical cube
            data.qpos[arm_q] = prev_q + (k + 1) / sub * (q_new - prev_q)
            data.qvel[arm_d] = 0
            S.set_hand_curl(model, data, curl)
            data.qvel[hand_d] = 0
            mujoco.mj_kinematics(model, data)
            if attached:
                brush_free = S.body_T(model, data, S.PALM) @ T_pb
            S.set_free(model, data, 'brush_free', brush_free)
            mujoco.mj_step(model, data)
        prev_q = q_new
        data.qpos[arm_q] = q_new
        S.set_hand_curl(model, data, curl)
        if attached:
            brush_free = S.body_T(model, data, S.PALM) @ T_pb
        S.set_free(model, data, 'brush_free', brush_free)
        mujoco.mj_forward(model, data)
        Tb_w, Tc_w = S.body_T(model, data, 'brush'), S.body_T(model, data, 'cube')

        ren.update_scene(data, 'zed_left')
        rgb = ren.render().copy()
        ren.update_scene(data, 'zed_right')
        right = ren.render().copy()
        ren.enable_depth_rendering()
        ren.update_scene(data, 'zed_left')
        depth = ren.render().copy().astype(np.float32)
        ren.disable_depth_rendering()
        ren.enable_segmentation_rendering()
        ren.update_scene(data, 'zed_left')
        seg = ren.render().copy()
        ren.update_scene(data, 'zed_left', scene_option=opt_brush)
        seg_b = ren.render().copy()
        ren.disable_segmentation_rendering()
        is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        mb = is_geom & (seg[..., 0] == gid_b)
        mc = is_geom & (seg[..., 0] == gid_c)
        full_b = (seg_b[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg_b[..., 0] == gid_b)
        stem = f'{i:05d}'
        imageio.imwrite(rd / 'rgb' / f'{stem}.png', rgb)
        imageio.imwrite(rd / 'right' / f'{stem}.png', right)
        np.save(rd / 'depth_gt' / f'{stem}.npy', depth)
        if i == 0:
            imageio.imwrite(rd / 'masks_1' / f'{stem}.png', (mb * 255).astype(np.uint8))
            imageio.imwrite(rd / 'masks_2' / f'{stem}.png', (mc * 255).astype(np.uint8))
        ts_lines.append(f'{stem} {int(1_700_000_000e9 + t * 1e9)}')
        cam_brush.append(Tinv @ Tb_w)
        cam_cube.append(Tinv @ Tc_w)
        w_brush.append(Tb_w)
        w_cube.append(Tc_w)
        vis.append(mb.sum() / max(full_b.sum(), 1))
        pxs.append(int(mb.sum()))
        att.append(bool(attached))
        ik_err.append((perr, rerr))
        pv = rgb.copy()
        pv[mb] = (0.5 * pv[mb] + [0, 127, 0]).astype(np.uint8)
        writer.append_data(pv)
        if i % 30 == 0:
            print(f'{a.scenario} t={t:4.1f}s ik {perr * 1000:5.1f} mm {rerr:4.1f} deg  brush vis {vis[-1]:.2f} '
                  f'({pxs[-1]} px)  attached {attached}  cube z {Tc_w[2, 3]:.3f}', flush=True)
    writer.close()
    (rd / 'ts_ns.txt').write_text('\n'.join(ts_lines) + '\n')
    for k, v in dict(T_cam_brush=cam_brush, T_cam_cube=cam_cube, T_world_brush=w_brush, T_world_cube=w_cube,
                     vis_brush=vis, px_brush=pxs, attached=att, ik_err=ik_err).items():
        np.save(out / 'gt' / f'{k}.npy', np.asarray(v))
    np.savetxt(out / 'gt' / 'T_RC.txt', T_RC)
    json.dump(dict(scenario=a.scenario, fps=a.fps, n=n, roll=a.roll, slide=a.slide, curl=a.curl,
                   max_ik_mm=float(np.max([e[0] for e in ik_err]) * 1000),
                   max_ik_deg=float(np.max([e[1] for e in ik_err]))), open(out / 'gt' / 'info.json', 'w'), indent=1)
    print('wrote', out, 'frames', n, 'max IK err mm', round(np.max([e[0] for e in ik_err]) * 1000, 1))


if __name__ == '__main__':
    main()
