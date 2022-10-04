#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point

import sys
import signal

def signal_handler(signal, frame):
    print("Shutting down program.")
    sys.exit()

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    rospy.init_node('receiver_estimation_node', anonymous=True)
    rate = rospy.Rate(1)

    pub_receiver = rospy.Publisher('/receiver/position', Point, queue_size=1)

    receiver_position = Point()
    receiver_position.x = 0.0
    receiver_position.y = 0.0
    receiver_position.z = 0.0

    while not rospy.is_shutdown():

        print(rospy.get_name(), rospy.get_param(rospy.get_name() + '/cam_base_name'))

        # locate, todo

        pub_receiver.publish(receiver_position)

        rate.sleep()
