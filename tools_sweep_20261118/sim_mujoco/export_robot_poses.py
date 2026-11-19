"""Object-1 poses of a tracker run in the robot frame, for another camera's --reseed_file.
python export_robot_poses.py RUN_DIR T_RC.txt OUT.npz"""
import sys, json, numpy as np
run, trc, out = sys.argv[1:4]
T_RC = np.loadtxt(trc)
P = np.load(f'{run}/poses_object_1.npy')
rows = [json.loads(l) for l in open(f'{run}/frame_log.jsonl')]
st, Tr = [], []
for i, r in enumerate(rows[:len(P)]):
    ns = r.get('object_capture_ns', [r['capture_ns']])[0] or r['capture_ns']
    st.append(int(ns)); Tr.append(T_RC @ P[i])
o = np.argsort(st)
np.savez(out, stamps=np.asarray(st, np.int64)[o], T_robot=np.asarray(Tr)[o])
print('wrote', out, len(st), 'poses')
