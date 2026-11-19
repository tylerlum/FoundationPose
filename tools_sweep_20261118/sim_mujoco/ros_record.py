"""Record the tracker's ROS output on a PRIVATE roscore (never the robot's). python ros_record.py OUT.jsonl
Subscribes to every PoseStamped topic under /robot_frame and /camera_frame, /perception/payload/{pose_robot,status},
and the quality array; writes one JSON line per message with receive time."""
import json
import sys
import time

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32MultiArray

out = open(sys.argv[1], 'w')
rospy.init_node('sim_tracker_recorder', anonymous=True)
subs = {}


def cb_pose(msg, topic):
    p, q = msg.pose.position, msg.pose.orientation
    out.write(json.dumps(dict(topic=topic, recv=time.time(), stamp=msg.header.stamp.to_nsec(),
                              p=[p.x, p.y, p.z], q_xyzw=[q.x, q.y, q.z, q.w])) + '\n')


def cb_str(msg, topic):
    out.write(json.dumps(dict(topic=topic, recv=time.time(), data=msg.data)) + '\n')


def cb_arr(msg, topic):
    out.write(json.dumps(dict(topic=topic, recv=time.time(), data=list(msg.data))) + '\n')


while not rospy.is_shutdown():
    for topic, typ in rospy.get_published_topics():
        if topic in subs:
            continue
        if typ == 'geometry_msgs/PoseStamped':
            subs[topic] = rospy.Subscriber(topic, PoseStamped, cb_pose, callback_args=topic, queue_size=50)
        elif typ == 'std_msgs/String' and topic.startswith('/perception'):
            subs[topic] = rospy.Subscriber(topic, String, cb_str, callback_args=topic, queue_size=50)
        elif typ == 'std_msgs/Float32MultiArray':
            subs[topic] = rospy.Subscriber(topic, Float32MultiArray, cb_arr, callback_args=topic, queue_size=50)
        else:
            continue
        print('subscribed', topic, typ, flush=True)
    out.flush()
    time.sleep(0.5)
