"""Reachable held-tool poses and the tool's visible fraction, for the 'tour_<seed>' scenarios of any task.
FPSIM_TASK=screw MUJOCO_GL=egl python search_poses.py N   -> pose_search_<task>.json"""
import sys, json, numpy as np, mujoco
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '.')
import sim_scene as S, gen_take as G
from realworld.common import stir_scene
N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
model, data, T_RC, _ = S.build()
gb = [g for g in range(model.ngeom) if model.geom(g).name.startswith('brush_geom')]
model.geom_group[gb] = 2
aq = np.array([model.jnt_qposadr[j] for j in S.arm_joint_ids(model)]); ad = np.array([model.jnt_dofadr[j] for j in S.arm_joint_ids(model)])
home = np.asarray(stir_scene.IIWA_HOME_JOINT_POS)
ren = mujoco.Renderer(model, 270, 480); ob = mujoco.MjvOption(); ob.geomgroup[:] = 0; ob.geomgroup[2] = 1
def vis():
    ren.enable_segmentation_rendering(); ren.update_scene(data, 'zed_left'); a = ren.render().copy()
    ren.update_scene(data, 'zed_left', scene_option=ob); b = ren.render().copy(); ren.disable_segmentation_rendering()
    va = (np.isin(a[..., 0], gb) & (a[..., 1] == 5)).sum(); vb = (np.isin(b[..., 0], gb) & (b[..., 1] == 5)).sum()
    return va / max(vb, 1), int(vb)
data.qpos[aq] = home; S.set_hand_curl(model, data, 0.8); mujoco.mj_forward(model, data)
Tpb = S.grasp_T_auto(model, data, 270, slide=0.0)
R0 = np.stack([[0, 0, -1.0], [1, 0, 0], [0, -1, 0]], 1) if S.TASK == 'screw' else np.eye(3)   # tool down / flat
rs = np.random.RandomState(0); res = []
for k in range(N):
    pos = [rs.uniform(-0.15, 0.2), rs.uniform(-0.95, -0.65), G.TT + rs.uniform(0.10, 0.32)]
    Rb = (Rot.from_rotvec(rs.normal(size=3) * 0.9) * Rot.from_matrix(R0)).as_matrix()
    T = G.T_of(Rb, pos); best = None
    for seed in range(3):
        data.qpos[aq] = home + (0 if seed == 0 else rs.uniform(-.8, .8, 7))
        e = G.ik_palm(model, data, aq, ad, T @ np.linalg.inv(Tpb), iters=200)
        if best is None or e[0] < best[0][0]: best = (e, data.qpos[aq].copy())
    if best[0][0] > 0.003: continue
    data.qpos[aq] = best[1]; S.set_hand_curl(model, data, 0.8); mujoco.mj_forward(model, data)
    S.set_free(model, data, 'brush_free', S.body_T(model, data, S.PALM) @ Tpb); mujoco.mj_forward(model, data)
    v, full = vis()
    res.append(dict(slide=0.0, pos=list(map(float, pos)), rotvec=Rot.from_matrix(Rb).as_rotvec().tolist(), q=best[1].tolist(), vis=float(v), full=full))
json.dump(res, open(f'pose_search_{S.TASK}.json', 'w'))
vs = np.array([r['vis'] for r in res]); print(S.TASK, 'reachable', len(res), 'vis pct', np.percentile(vs, [0, 10, 25, 50]).round(2))
