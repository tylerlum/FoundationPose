"""Score the tracker's ROS messages (recorded on the private master) against MuJoCo truth in the ROBOT frame.
python score_ros.py TAKE_DIR TOPICS.jsonl"""
import json, sys, collections
import numpy as np
from scipy.spatial.transform import Rotation as R
take, path = sys.argv[1], sys.argv[2]
st = {int(l.split()[1]): int(l.split()[0]) for l in open(f'{take}/replay/ts_ns.txt') if l.strip()}
Gb, Gc = np.load(f'{take}/gt/T_world_brush.npy'), np.load(f'{take}/gt/T_world_cube.npy')
msgs = [json.loads(l) for l in open(path)]
by = collections.defaultdict(list)
for m in msgs:
    by[m['topic']].append(m)
print({k: len(v) for k, v in by.items()})
def err(topic, G, sym=False):
    e_t, e_r = [], []
    for m in by[topic]:
        i = st.get(m['stamp'])
        if i is None:
            continue
        g = G[i]
        e_t.append(np.linalg.norm(np.array(m['p']) - g[:3, 3]))
        Rm = R.from_quat(m['q_xyzw'])
        d = R.from_matrix(g[:3, :3]).inv() * Rm
        e_r.append(np.degrees(min((d * s).magnitude() for s in R.create_group('O'))) if sym else np.degrees(d.magnitude()))
    e_t, e_r = np.array(e_t), np.array(e_r)
    return f'{topic}: n {len(e_t)} matched, origin err med {np.median(e_t)*1000:.1f} mm p90 {np.percentile(e_t,90)*1000:.1f} mm, ' \
           f'rot med {np.median(e_r):.1f} deg, frames > 5 cm: {int((e_t>0.05).sum())}'
print(err('/robot_frame/current_object_pose', Gb))
print(err('/robot_frame/current_object_pose_2', Gc, sym=True))
print(err('/perception/payload/pose_robot', Gc, sym=True))
stat = [json.loads(m['data']) if m['data'].startswith('{') else m['data'] for m in by['/perception/payload/status']]
valid = [s.get('valid') if isinstance(s, dict) else ('valid' in s and 'invalid' not in s) for s in stat]
print('payload status: ', len(stat), 'msgs, valid', sum(bool(v) for v in valid), '; example:', str(stat[len(stat)//2])[:200])
