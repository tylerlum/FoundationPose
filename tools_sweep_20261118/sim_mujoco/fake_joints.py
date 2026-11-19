"""Fake robot for the hand check: publishes /iiwa/joint_states and /sharpa/joint_states (PRIVATE roscore only) from a
sim take's joints.npz, in step with the replayed frame (read from the tracker's pose stamps).
python fake_joints.py TAKE_DIR"""
import sys, time
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
J = np.load(f'{sys.argv[1]}/replay/joints.npz')
arm_names = [f'iiwa14_joint_{i}' for i in range(1, 8)]
hand_names = ['left_1_thumb_CMC_FE', 'left_thumb_CMC_AA', 'left_thumb_MCP_FE', 'left_thumb_MCP_AA', 'left_thumb_IP',
              'left_2_index_MCP_FE', 'left_index_MCP_AA', 'left_index_PIP', 'left_index_DIP', 'left_3_middle_MCP_FE',
              'left_middle_MCP_AA', 'left_middle_PIP', 'left_middle_DIP', 'left_4_ring_MCP_FE', 'left_ring_MCP_AA',
              'left_ring_PIP', 'left_ring_DIP', 'left_5_pinky_CMC', 'left_pinky_MCP_FE', 'left_pinky_MCP_AA',
              'left_pinky_PIP', 'left_pinky_DIP']
rospy.init_node('fake_robot_joints', anonymous=True)
pa = rospy.Publisher('/iiwa/joint_states', JointState, queue_size=1)
ph = rospy.Publisher('/sharpa/joint_states', JointState, queue_size=1)
cur = {'i': 0}
def on_pose(m):
    cur['i'] = int(round((m.header.stamp.to_nsec() - 1_700_000_000 * 10**9) / (1e9 / 30)))
rospy.Subscriber('/camera_frame/current_object_pose', PoseStamped, on_pose, queue_size=1)
r = rospy.Rate(200)
n = len(J['q_arm'])
while not rospy.is_shutdown():
    i = min(max(cur['i'] + 1, 0), n - 1)          # the frame the tracker is about to process
    for pub, names, q in ((pa, arm_names, J['q_arm'][i]), (ph, hand_names, J['q_hand'][i])):
        msg = JointState(); msg.header.stamp = rospy.Time.now(); msg.name = names; msg.position = list(map(float, q))
        pub.publish(msg)
    r.sleep()
