import sys
import rospy
from fh_sensors_camera.srv import *
import numpy as np
import cv2
from std_msgs.msg import Header, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import fhUtils.msg_helper as msg_helper

class CameraClient(object):
    """docstring for CameraClient."""

    def __init__(self, type = ""):
        self.type = type
        self.rgb = 0
        self.depth = 0
        self.uv = 0
        self.pointcloud = 0
        self.pointcloudColor = 0

        print(self.type)

        if self.type != "test" and self.type != "camera_robot" and self.type != "camera_shelf":
            raise Exception("Invalid type")

        self.base_service = "/sensors/" + self.type
        print(self.base_service)

        self.serviceNameCapture = self.base_service + "/capture"
        self.serviceNameRGB = self.base_service + "/rgb"
        self.serviceNameDepth = self.base_service + "/depth"
        self.serviceNameUV = self.base_service + "/pointcloud/static/uv"
        self.serviceNamePointcloud = self.base_service + "/pointcloud/static"

    def captureNewScene(self):
        """ Tells the camera service to update the static data """

        rospy.wait_for_service(self.serviceNameCapture)
        captureService = rospy.ServiceProxy(self.serviceNameCapture, capture)
        msg = capture()
        msg.data = True
        response = captureService(msg)

    def getRGB(self):
        """ Sets the self.rgb to current static rgb captured by camera """

        rospy.wait_for_service(self.serviceNameRGB)
        rgbService = rospy.ServiceProxy(self.serviceNameRGB, rgb)
        msg = rgb()
        msg.data = True
        response = rgbService(msg)
        img = np.frombuffer(response.img.data, dtype=np.uint8).reshape(response.img.height, response.img.width, -1)
        self.rgb = img
        return self.rgb


    def getDepth(self):
        """ Sets the self.depth to current static depth image captured by
        camera """

        rospy.wait_for_service(self.serviceNameDepth)
        depthService = rospy.ServiceProxy(self.serviceNameDepth, depth)
        msg = depth()
        msg.data = True
        response = depthService(msg)
        img = np.frombuffer(response.img.data, dtype=np.float16).reshape(response.img.height, response.img.width, -1)
        #img = np.frombuffer(response.img.data, dtype=np.uint8).reshape(response.img.height, response.img.width, -1)
        self.depth = img
        return self.depth

    def getUvStatic(self):
        """ Sets the self.uv to current static uv coordinates for translation
        from pixel coordinates to point cloud coordinates """

        rospy.wait_for_service(self.serviceNameUV)
        uvStaticService = rospy.ServiceProxy(self.serviceNameUV, uvSrv)
        msg = uvSrv()

        msg.data = True
        response = uvStaticService(msg)
        uv = msg_helper.unpackUV(response.uv)

        self.uv = uv
        return self.uv

    def getPointCloudStatic(self):
        """ sets self.pointcloud to the current static point cloud with geometry
        only """

        rospy.wait_for_service(self.serviceNamePointcloud)
        pointcloudStaticService = rospy.ServiceProxy(self.serviceNamePointcloud, pointcloud)

        msg = pointcloud()
        msg.data = True
        response = pointcloudStaticService(msg)
        self.pointcloud, self.pointcloudColor = msg_helper.unpackPCD(response.pc, response.color)

        return self.pointcloud, self.pointcloudColor
