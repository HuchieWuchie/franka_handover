#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Int32
import numpy as np
#from cameraService.cameraClient import CameraClient
from affordanceService.client import AffordanceClient
from orientation_service.srv import runOrientationSrv, runOrientationSrvResponse
from orientation_service.srv import setSettingsOrientationSrv, setSettingsOrientationSrvResponse
import fhUtils.msg_helper as msg_helper

class OrientationClient(object):
    """docstring for orientationClient."""

    def __init__(self):

        self.method = 0 # learned from observation

    def getOrientation(self, pcd_affordance):

        print("Waiting for orientation service...")
        rospy.wait_for_service("/computation/handover_orientation/get")
        orientationService = rospy.ServiceProxy("/computation/handover_orientation/get", runOrientationSrv)
        print("Connection to orientation service established!")

        #camClient = CameraClient()
        affClient = AffordanceClient(connected = False)

        pcd_points = np.asanyarray(pcd_affordance.points)
        pcd_colors = np.asanyarray(pcd_affordance.colors)

        if np.max(pcd_colors) <= 1:
            pcd_colors = pcd_colors * 255

        pcd_geometry_msg, pcd_color_msg = msg_helper.packPCD(pcd_points, pcd_colors, frame_id = "it_doesnt_matter")
        response = orientationService(pcd_geometry_msg, pcd_color_msg)

        current_orientation, current_translation, goal_orientation = msg_helper.unpackOrientation(response.current, response.goal)

        del response

        return current_orientation, current_translation, goal_orientation

    def setSettings(self, method):

        if method == 0 or method == 1:
            self.method = method

            rospy.wait_for_service("/computation/handover_orientation/set_settings")
            settingsService = rospy.ServiceProxy("/computation/handover_orientation/set_settings", setSettingsOrientationSrv)

            _ = settingsService(Int32(method))

        else:
            print("Invalid method")
