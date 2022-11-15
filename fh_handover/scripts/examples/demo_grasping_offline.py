#!/usr/bin/env python3
import rospy
import rospkg

import numpy as np
import moveit_msgs
import geometry_msgs
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotTrajectory

import fhMoveitUtils.moveit_utils as moveit

from fh_moveit_service.srv import moveitMoveToNamedSrv, moveitMoveToNamedSrvResponse
from fh_moveit_service.srv import moveitPlanToNamedSrv, moveitPlanToNamedSrvResponse
from fh_moveit_service.srv import moveitPlanFromPoseToPoseSrv, moveitPlanFromPoseToPoseSrvResponse
from fh_moveit_service.srv import moveitMoveToPoseSrv, moveitMoveToPoseSrvResponse
from fh_moveit_service.srv import moveitExecuteSrv, moveitExecuteSrvResponse
from fh_moveit_service.srv import moveitRobotStateSrv, moveitRobotStateSrvResponse
from fh_moveit_service.srv import moveitPlanToPoseSrv, moveitPlanToPoseSrvResponse
from fh_moveit_service.srv import moveitGetJointPositionAtNamed, moveitGetJointPositionAtNamedResponse
from fh_moveit_service.srv import moveitGripperCloseSrv, moveitGripperCloseSrvResponse
from fh_moveit_service.srv import moveitGripperOpenSrv, moveitGripperOpenSrvResponse

if __name__ == "__main__":

    rospy.init_node('aau_moveit_usage_example', anonymous=True)

    # Set planning parameters
    moveit.setMaxVelocityScalingFactor(0.2)
    moveit.setMaxAcceleratoinScalingFactor(0.2)
    moveit.setPlanningTime(5.0)
    moveit.setNumPlanningAttempts(50)

    # Open the gripper
    moveit.gripperOpen()

    # Move the robot to the predefined ready pose
    moveit.moveToNamed("home")
    moveit.moveToNamed("ready")
    
    # Plan a trajectory to the grasp pose found using tf_echo

    ## Define a pose (ROS message)

    pose = geometry_msgs.msg.Pose()
    pose.position.x = 0.423
    pose.position.y = 0.085
    pose.position.z = 0.007

    pose.orientation.x = 1
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 0    

    ## Compute the trajectory and execute it

    success, trajectory = moveit.planToPose(pose)
    print("Found trajectory: ", success)

    if success:
        moveit.executeTrajectory(trajectory)

    # Grasp the object

    moveit.grasp(width = 0.025, speed = 0.1, force = 140, epsilon_outer=0.01, epsilon_inner=0.01)

    # Move the grasped object 10 cm up

    pose.position.z += 0.1

    success, trajectory = moveit.planToPose(pose)
    print("Found trajectory: ", success)

    if success:
        moveit.executeTrajectory(trajectory)

    # Put it down again

    pose.position.z += -0.1

    success, trajectory = moveit.planToPose(pose)
    print("Found trajectory: ", success)

    if success:
        moveit.executeTrajectory(trajectory)

    # Release the object

    moveit.gripperOpen()

    ## Finally move back to home pose
    moveit.moveToNamed("home")
