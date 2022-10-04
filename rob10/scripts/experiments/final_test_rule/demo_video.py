#!/usr/bin/env python3
import sys
import copy
import rospy
import math
import numpy as np
import time
import open3d as o3d
import cv2
import random
import signal
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os
import argparse
import imageio

#import actionlib

import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Transform
import std_msgs.msg
from std_msgs.msg import Int8, Int16, MultiArrayDimension, MultiArrayLayout, Int32MultiArray, Float32MultiArray, Bool, Header
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2
from iiwa_msgs.msg import JointPosition

import rob9Utils.transformations as transform
from rob9Utils.graspGroup import GraspGroup
from rob9Utils.grasp import Grasp
import rob9Utils.moveit as moveit
from cameraService.cameraClient import CameraClient
from affordanceService.client import AffordanceClient
from grasp_service.client import GraspingGeneratorClient
from orientationService.client import OrientationClient
from locationService.client import LocationClient
from rob9Utils.visualize import visualizeGrasps6DOF, visualizeMasksInRGB, visualizeBBoxInRGB
import rob9Utils.iiwa
from rob9Utils.visualize import visualizeMasksInRGB, visualizeFrameMesh, createGripper, visualizeGripper, visualizeMeshInRGB, visualizeGraspInRGB
from rob9Utils.affordancetools import getPredictedAffordances, getAffordanceContours, getObjectAffordancePointCloud, getPointCloudAffordanceMask
from rob9Utils.utils import erodeMask, keepLargestContour, convexHullFromContours, maskFromConvexHull, thresholdMaskBySize, removeOverlapMask


from moveit_scripts.srv import *
from moveit_scripts.msg import *
from rob9.srv import graspGroupSrv, graspGroupSrvResponse

import time

def imgSubscriber(img_msg):
    global out_video
    #img_cam = br.imgmsg_to_cv2(msg, "bgr8")

    dtype, n_channels = np.uint8, 3
    dtype = np.dtype(dtype)
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

    img_buf = np.asarray(img_msg.data, dtype=dtype) if isinstance(img_msg.data, list) else img_msg.data

    if n_channels == 1:
        im = np.ndarray(shape=(img_msg.height, int(img_msg.step/dtype.itemsize)),
                        dtype=dtype, buffer=img_buf)
        im = np.ascontiguousarray(im[:img_msg.height, :img_msg.width])
    else:
        im = np.ndarray(shape=(img_msg.height, int(img_msg.step/dtype.itemsize/n_channels), n_channels),
                        dtype=dtype, buffer=img_buf)
        im = np.ascontiguousarray(im[:img_msg.height, :img_msg.width, :])

    # If the byte order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        im = im.byteswap().newbyteorder()

    img_cam = im

    out_video.append_data(img_cam[:, :, ::-1])

def signal_handler(signal, frame):
    print("Shutting down program.")
    sys.exit()

signal.signal(signal.SIGINT, signal_handler)


def callback(msg):
    global req_obj_id

    req_obj_id = msg.data[0]

def computeWaypoint(grasp, offset = 0.1):
    """ input:  graspsObjects   -   rob9Utils.grasp.Grasp() in world_frame
                offset          -   float, in meters for waypoint in relation to grasp
        output:
				waypoint		-   rob9Utils.grasp.Grasp()
    """

    world_frame = "world"
    ee_frame = "right_ee_link"

    waypoint = copy.deepcopy(grasp)

	# you can implement some error handling here if the grasp is given in the wrong frame
	#waypointWorld = Grasp().fromPoseStampedMsg(transform.transformToFrame(waypointCamera.toPoseStampedMsg(), world_frame))
	#graspWorld = Grasp().fromPoseStampedMsg(transform.transformToFrame(graspCamera.toPoseStampedMsg(), world_frame))

    # computing waypoint in camera frame
    rotMat = grasp.getRotationMatrix()
    offsetArr = np.array([[0.0], [0.0], [offset]])
    offsetCam = np.transpose(np.matmul(rotMat, offsetArr))[0]

    waypoint.position.x += -offsetCam[0]
    waypoint.position.y += -offsetCam[1]
    waypoint.position.z += -offsetCam[2]

    return waypoint

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate the affordance segmentation performance with a weighted F-measure')
    parser.add_argument('--save', dest='save',
                        help='If enabled, saves the various outputs into a folder.',
                        action='store_true')
    parser.add_argument('--visualize', dest='visualize',
                        help='If enabled, shows various blocking visualizations along the way.',
                        action='store_true')
    parser.add_argument('--move_robot', dest='move',
                        help='Enables physical movement of robot',
                        action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    global grasps_affordance, img, pcd, masks, bboxs, req_aff_id, req_obj_id, state, out_video, br

    args = parse_args()

    reset_gripper_msg = std_msgs.msg.Int16()
    reset_gripper_msg.data = 0
    activate_gripper_msg = std_msgs.msg.Int16()
    activate_gripper_msg.data = 1
    close_gripper_msg = std_msgs.msg.Int16()
    close_gripper_msg = 2
    open_gripper_msg = std_msgs.msg.Int16()
    open_gripper_msg.data = 3
    basic_gripper_msg = std_msgs.msg.Int16()
    basic_gripper_msg.data = 4
    pinch_gripper_msg = std_msgs.msg.Int16()
    pinch_gripper_msg.data = 5
    adjust_width_gripper_msg = std_msgs.msg.Int16()
    adjust_width_gripper_msg.data = 120 # 155 is those 8 cm
    increase_force_gripper_msg = std_msgs.msg.Int16()
    increase_force_gripper_msg.data = 30
    increase_speed_gripper_msg = std_msgs.msg.Int16()
    increase_speed_gripper_msg.data = 10


    print("Init")
    rospy.init_node('moveit_subscriber', anonymous=True)

    state = 1 # start at setup phase
    rate = rospy.Rate(10)
    br = CvBridge()
    pub_aff = rospy.Publisher('/outputs/img/affordance_prediction', Image,queue_size=1)
    pub_tool = rospy.Publisher('/outputs/img/object_to_pick', Image,queue_size=1)
    pub_pose = rospy.Publisher('/outputs/img/object_pose', Image,queue_size=1)
    pub_img_grasp = rospy.Publisher('/outputs/img/grasp', Image,queue_size=1)

    t_string = str(time.time()*1000)
    if args.save:
        if not os.path.exists(t_string):
            os.makedirs(t_string)

        out_video = imageio.get_writer(t_string + "/rgb_video.mp4", fps = 10)
        sub_img_cam = rospy.Subscriber('/sensors/realsense/rgb', Image, imgSubscriber)

    obj_labels = [14, 6, 16]
    cam_locations = ["camera_ready_3", "camera_ready_4", "camera_ready_4"]

    counter = 0
    while True:

        obj_lab = obj_labels[counter]
        cam_loc = cam_locations[counter]
        print(obj_lab, cam_loc)

        if state == 1:

            img_msg = br.cv2_to_imgmsg(np.zeros((720, 1280, 3), np.uint8))
            pub_aff.publish(img_msg)
            pub_tool.publish(img_msg)
            pub_pose.publish(img_msg)
            pub_img_grasp.publish(img_msg)



            # setup phase

            if args.move:
                set_ee = True
                if not rob9Utils.iiwa.setEndpointFrame():
                    set_ee = False
                print("STATUS end point frame was changed: ", set_ee)

                set_PTP_speed_limit = True
                if not rob9Utils.iiwa.setPTPJointSpeedLimits(0.2, 0.2):
                    set_PTP_speed_limit = False
                print("STATUS PTP joint speed limits was changed: ", set_PTP_speed_limit)

                set_PTP_cart_speed_limit = True
                if not rob9Utils.iiwa.setPTPCartesianSpeedLimits(0.2, 0.2, 0.2, 0.2, 0.2, 0.2):
                    set_PTP_cart_speed_limit = False
                print("STATUS PTP cartesian speed limits was changed: ", set_PTP_cart_speed_limit)

            #rospy.Subscriber('tool_id', Int8, callback)
            rospy.Subscriber('objects_affordances_id', Int32MultiArray, callback )

            pub_grasp = rospy.Publisher('iiwa/pose_to_reach', PoseStamped, queue_size=10)
            pub_waypoint = rospy.Publisher('iiwa/pose_to_reach_waypoint', PoseStamped, queue_size=10)
            pub_iiwa = rospy.Publisher('iiwa/command/JointPosition', JointPosition, queue_size=10 )
            gripper_pub = rospy.Publisher('iiwa/gripper_controller', Int16, queue_size=10, latch=True)
            display_trajectory_publisher = rospy.Publisher('iiwa/move_group/display_planned_path',
                                                moveit_msgs.msg.DisplayTrajectory,
                                                queue_size=20)
            # DO NOT REMOVE THIS SLEEP, it allows gripper_pub to establish connection to the topic
            #rospy.sleep(0.1)
            if args.move:

                rospy.sleep(2)

                gripper_pub.publish(reset_gripper_msg)
                rospy.sleep(0.1)
                gripper_pub.publish(activate_gripper_msg)
                rospy.sleep(0.1)
                gripper_pub.publish(open_gripper_msg)
                rospy.sleep(0.1)
                gripper_pub.publish(pinch_gripper_msg)
                rospy.sleep(0.1)
                gripper_pub.publish(adjust_width_gripper_msg)
                rospy.sleep(0.1)
                for i in range(12):
                    gripper_pub.publish(increase_force_gripper_msg)
                    rospy.sleep(0.1)
                    gripper_pub.publish(increase_speed_gripper_msg)
                    rospy.sleep(0.1)


                    result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)
            state_ready = moveit.getCurrentState()

            print("Services init")

            state = 2

        elif state == 2:

            img_msg = br.cv2_to_imgmsg(np.zeros((720, 1280, 3), np.uint8))
            pub_aff.publish(img_msg)
            pub_tool.publish(img_msg)
            pub_pose.publish(img_msg)
            pub_img_grasp.publish(img_msg)

            # Start AffNet-DR
            print("Segmenting affordance maps")
            aff_client = AffordanceClient()

            aff_client.start(GPU=True)

            state = 3

        elif state == 3:

            print("State 3")



            if args.move:
                result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed(cam_loc).joint_position.data)
            req_obj_id = -1
			# Capture sensor information

            img_msg = br.cv2_to_imgmsg(np.zeros((720, 1280, 3), np.uint8))
            pub_aff.publish(img_msg)
            pub_tool.publish(img_msg)
            pub_pose.publish(img_msg)
            pub_img_grasp.publish(img_msg)

            print("Camera is capturing new scene")
            cam = CameraClient()
            cam.captureNewScene()
            cloud, cloudColor = cam.getPointCloudStatic()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            pcd.colors = o3d.utility.Vector3dVector(cloudColor)

            cloud_uv = cam.getUvStatic()
            img = cam.getRGB()

            if args.save:
                o3d.io.write_point_cloud(os.path.join(t_string, str(counter) + "pcd.ply"), pcd)
                np.save(os.path.join(t_string, str(counter) +  "uv.npy"), cloud_uv)
                cv2.imwrite(os.path.join(t_string, str(counter) +  "img.png"), img)

            state = 4

        elif state == 4:
            # Analyze affordance

            _ = aff_client.run(img, CONF_THRESHOLD = 0.7)

            masks, labels, scores, bboxs = aff_client.getAffordanceResult()

            if args.save:
                np.save(os.path.join(t_string, str(counter) +  "masks.npy"), masks)
                np.save(os.path.join(t_string, str(counter) +  "labels.npy"), labels)
                np.save(os.path.join(t_string, str(counter) +  "scores.npy"), scores)
                np.save(os.path.join(t_string, str(counter) +  "bboxs.npy"), bboxs)

            print("Found the following objects, waiting for command: ")
            for (label, score) in zip(labels, scores):
                print(aff_client.OBJ_CLASSES[label], " - score: ", score, " - label: ", label)

            img_vis = visualizeBBoxInRGB(visualizeMasksInRGB(img, masks), labels, bboxs, scores)

            img_msg = br.cv2_to_imgmsg(img_vis)
            pub_aff.publish(img_msg)

            if args.visualize:
                cv2.imshow("masks", img_vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if args.save:
                cv2.imwrite(os.path.join(t_string, str(counter) +  "img_vis_raw.png"), img_vis)

            state = 5


        elif state == 5:

            req_obj_id = obj_lab
            state = 6

        elif state == 6:
            # Check user input

            try:
                obj_inst = np.where(labels == req_obj_id)[0][0]
                print("Attempting to pick up: ", aff_client.OBJ_CLASSES[labels[obj_inst]])
                state = 7
            except:
                print("Did not find requested object")
                req_obj_id = -1
                req_obj_id = -1
                state = 3

        elif state == 7:
            # post process affordance segmentation maps

            obj_inst_masks = masks[obj_inst]
            obj_inst_label = labels[obj_inst]
            obj_inst_bbox = bboxs[obj_inst]

            if args.save:
                np.save(os.path.join(t_string, str(counter) +  str(req_obj_id) + "_premasks.npy"), obj_inst_masks)
                np.save(os.path.join(t_string, str(counter) +  str(req_obj_id) + "_label.npy"), obj_inst_label)
                np.save(os.path.join(t_string, str(counter) +  str(req_obj_id) + "_bboxs.npy"), obj_inst_bbox)

            # Post process affordance predictions and compute point cloud affordance mask

            affordances_in_object = getPredictedAffordances(masks = obj_inst_masks, bbox = obj_inst_bbox)

            for aff in affordances_in_object:

                m_vis = np.zeros(obj_inst_masks.shape)
                masks = erodeMask(affordance_id = aff, masks = obj_inst_masks,
                                kernel = np.ones((3,3)))
                contours = getAffordanceContours(bbox = obj_inst_bbox, affordance_id = aff,
                                                masks = obj_inst_masks)


                if len(contours) > 0:
                    contours = keepLargestContour(contours)
                    hulls = convexHullFromContours(contours)

                    h, w = obj_inst_masks.shape[-2], obj_inst_masks.shape[-1]
                    if obj_inst_bbox is not None:
                        h = int(obj_inst_bbox[3] - obj_inst_bbox[1])
                        w = int(obj_inst_bbox[2] - obj_inst_bbox[0])

                    aff_mask = maskFromConvexHull(h, w, hulls = hulls)
                    _, keep = thresholdMaskBySize(aff_mask, threshold = 0.05)
                    if keep == False:
                        aff_mask[:, :] = False

                    if obj_inst_bbox is not None:
                        obj_inst_masks[aff, obj_inst_bbox[1]:obj_inst_bbox[3], obj_inst_bbox[0]:obj_inst_bbox[2]] = aff_mask
                        m_vis[aff, obj_inst_bbox[1]:obj_inst_bbox[3], obj_inst_bbox[0]:obj_inst_bbox[2]] = aff_mask
                    else:
                        obj_inst_masks[aff, :, :] = aff_mask
                        m_vis[aff,:,:] = aff_mask

            obj_inst_masks = removeOverlapMask(masks = obj_inst_masks)
            img_vis = visualizeMasksInRGB(img, obj_inst_masks)

            img_msg = br.cv2_to_imgmsg(img_vis)
            pub_tool.publish(img_msg)

            if args.visualize:
                cv2.imshow("masks", img_vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if args.save:
                np.save(os.path.join(t_string, str(counter) +  str(req_obj_id) + "_postmasks.npy"), obj_inst_masks)
                cv2.imwrite(os.path.join(t_string, str(counter) +  "img_vis.png"), img_vis)

            affordances_in_object = getPredictedAffordances(masks = obj_inst_masks, bbox = obj_inst_bbox)

            state = 8

        elif state == 8:
            # transform point cloud into world coordinate frame

            T, transl_cam_2_world, rot_mat_cam_2_world = transform.getTransform("ptu_camera_color_optical_frame", "world")

            pcd.transform(T)
            points = np.asanyarray(pcd.points)
            pcd_affordance = getObjectAffordancePointCloud(pcd, obj_inst_masks, uvs = cloud_uv)

            # Compute a downsampled version of the point cloud for collision checking
            # downsampling speeds up computation
            #pcd_downsample = pcd.voxel_down_sample(voxel_size=0.005)
            pcd_downsample = pcd.voxel_down_sample(voxel_size=0.01)

            state = 9

        elif state == 9:

            # Select affordance mask to compute grasps for
            observed_affordances = getPredictedAffordances(obj_inst_masks)

            success = []
            sampled_grasp_points = []
            for observed_affordance in observed_affordances:
                if observed_affordance in aff_client.functional_labels:
                    local_success, local_sampled_grasp_points = getPointCloudAffordanceMask(affordance_id = observed_affordance,
                                                    points = points, uvs = cloud_uv, masks = obj_inst_masks)
                    success.append(local_success)

                    if len(sampled_grasp_points) == 0:
                        sampled_grasp_points = local_sampled_grasp_points
                    else:
                        sampled_grasp_points = np.vstack((sampled_grasp_points, local_sampled_grasp_points))

            if args.save:
                o3d.io.write_point_cloud(os.path.join(t_string, str(counter) +  "pcd_world.ply"), pcd)
                o3d.io.write_point_cloud(os.path.join(t_string, str(counter) +  "pcd_affordance.ply"), pcd_affordance)

            if True in success:

                # computing goal pose of object in world frame and
                # current pose of object in world frame

                rotClient = OrientationClient()
                rotClient.setSettings(1)
                object_rot_mat_world, object_transl_world, goal_orientation_giver = rotClient.getOrientation(pcd_affordance) # we discard translation

                object_rot_quat_world = R.from_matrix(object_rot_mat_world).as_quat()
                object_pose_world = np.hstack((object_transl_world.flatten(), object_rot_quat_world))
                transform.visualizeTransform(transform.poseToTransform(object_pose_world), "Handover object")
                pcd_object_pose = visualizeFrameMesh(object_transl_world, object_rot_mat_world)

                #o3d.visualization.draw_geometries([pcd, pcd_object_pose])

                T_world_2_cam, _, _ = transform.getTransform("world", "ptu_camera_color_optical_frame")
                pcd_object_pose.transform(T_world_2_cam)

                img_obj_pose = visualizeMeshInRGB(pcd_object_pose, img)
                img_msg = br.cv2_to_imgmsg(img_obj_pose)
                pub_pose.publish(img_msg)

                if args.visualize:
                    cv2.imshow("Estimated object pose", img_obj_pose)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                if args.save:
                    np.save(os.path.join(t_string, str(counter) +  "object_pose_world.npy"), object_pose_world)
                    cv2.imwrite(os.path.join(t_string, str(counter) +  "img_obj_pose.png"), img_obj_pose)

                loc_client = LocationClient()
                goal_location_giver_original = loc_client.getLocation().flatten() # in givers frame

                _, _, rot_mat_giver_2_world = transform.getTransform("giver", "world")

                # run the grasp algorithm
                grasp_client = GraspingGeneratorClient()
                grasp_client.setSettings(0.05, -1.0, 1.0, # azimuth
                                        0.1, -0., 0., # polar
                                        0.0025, -0.005, 0.05) # depth
                grasps = grasp_client.run(sampled_grasp_points, pcd_downsample,
                                            "world", req_obj_id, -1, obj_inst)
                grasps.sortByScore()

                count_grasp = 0
                while count_grasp < len(grasps):

                    grasp = grasps[count_grasp]

                    print("=========================================")
                    print("Computing for grasp num: ", count_grasp + 1, " / ", len(grasps))

                    waypoint = computeWaypoint(grasp, offset = 0.1)
                    waypoint_msg = waypoint.toPoseMsg()
                    pub_waypoint.publish(waypoint.toPoseStampedMsg())
                    valid_waypoint, state_waypoint = moveit.getInverseKinematicsSolution(state_ready, waypoint_msg)

                    if valid_waypoint:
                        grasp_msg = grasp.toPoseMsg()
                        valid_grasp, state_grasp = moveit.getInverseKinematicsSolution(state_waypoint.solution, grasp_msg)

                        if valid_grasp:
                            pub_waypoint.publish(waypoint.toPoseStampedMsg())
                            pub_grasp.publish(grasp.toPoseStampedMsg())

                            rot_range = 7
                            x_range = 5

                            #for x_pos in range(x_range):
                                #for rot_num in range(rot_range):
                            x_pos = 0
                            while x_pos <= x_range:

                                rot_num = 0
                                while rot_num <= rot_range:

                                    # tools rotate around giver frame's x axis
                                    # contain objects rotate around giver frame's z-axis

                                    if obj_inst_label in [1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 17, 18, 19, 20]:
                                        rotation = R.from_euler("xyz", [0, 0, rot_num * (2 * math.pi / rot_range)]).as_matrix()
                                        rotation = R.from_euler("XYZ", [rot_num * (2 * math.pi / rot_range), 0, 0]).as_matrix()
                                    else:
                                        rotation = R.from_euler("xyz", [0, 0, rot_num * (2 * math.pi / rot_range)]).as_matrix()
                                    goal_orientation_giver = np.matmul(rotation, goal_orientation_giver)
                                    goal_rot_mat_world = np.matmul(rot_mat_giver_2_world, goal_orientation_giver)
                                    goal_rot_quat_world = R.from_matrix(goal_rot_mat_world).as_quat()

                                    goal_location_giver = np.zeros(3)
                                    goal_location_giver[0] = min(1.2, max(0.7 ,goal_location_giver_original[0]/2))
                                    goal_location_giver[0] += 0.2 - (0.4 / x_range) * x_pos
                                    goal_location_giver = np.reshape(goal_location_giver, (3, 1))
                                    goal_location_world = transform.transformToFrame(goal_location_giver, "world", "giver")
                                    goal_location_world = np.array([goal_location_world.pose.position.x, goal_location_world.pose.position.y, goal_location_world.pose.position.z])
                                    goal_location_world[2] = 1.2

                                    print(x_pos, rot_num)

                                    goal_pose_world = np.hstack((goal_location_world.flatten(), goal_rot_quat_world))
                                    #transform.visualizeTransform(transform.poseToTransform(goal_pose_world), "object_goal_pose")

                                    # Compute the homegenous 4x4 transformation matrices

                                    world_grasp_T = transform.poseStampedToMatrix(grasp.toPoseStampedMsg()) # grasp_pose_world
                                    world_centroid_T = transform.poseToMatrix(object_pose_world)
                                    world_centroid_T_goal = transform.poseToMatrix(goal_pose_world)

                                    # Compute an end effector pose that properly orients the grasped tool

                                    grasp_world_T = np.linalg.inv(world_grasp_T)
                                    grasp_centroid_T = np.matmul(grasp_world_T, world_centroid_T)

                                    centroid_grasp_T = np.linalg.inv(grasp_centroid_T)
                                    world_grasp_T_goal = np.matmul(world_centroid_T_goal, centroid_grasp_T)
                                    goal_q = transform.quaternionFromRotation(world_grasp_T_goal)

                                    # Create poseStamped ros message

                                    ee_goal_msg = geometry_msgs.msg.PoseStamped()
                                    ee_goal_msg.header.frame_id = "world"
                                    ee_goal_msg.header.stamp = rospy.Time.now()

                                    ee_pose = Pose()
                                    ee_pose.position.x = world_grasp_T_goal[0,3]
                                    ee_pose.position.y = world_grasp_T_goal[1,3]
                                    ee_pose.position.z = world_grasp_T_goal[2,3]

                                    ee_pose.orientation.x = goal_q[0]
                                    ee_pose.orientation.y = goal_q[1]
                                    ee_pose.orientation.z = goal_q[2]
                                    ee_pose.orientation.w = goal_q[3]

                                    ee_goal_msg.pose = ee_pose

                                    ee_tf = Transform()
                                    ee_tf.translation = ee_pose.position
                                    ee_tf.rotation = ee_pose.orientation


                                    valid_handover, state_handover = moveit.getInverseKinematicsSolution(state_ready, ee_pose)

                                    #print("Executing trajectory")
                                    if valid_handover:

                                        grasp_cam = copy.deepcopy(grasp)
                                        grasp_cam.transformToFrame("ptu_camera_color_optical_frame")
                                        img_vis_grasp = visualizeGraspInRGB(img, grasp_cam, opening = 0.12)

                                        img_msg = br.cv2_to_imgmsg(img_vis_grasp)
                                        pub_img_grasp.publish(img_msg)

                                        if args.visualize:

                                            cv2.imshow("Grasp in RGB image", img_vis_grasp)
                                            cv2.waitKey(0)
                                            cv2.destroyAllWindows()

                                        if args.save:
                                            cv2.imwrite(os.path.join(t_string, str(counter) +  "img_vis_grasp.png"), img_vis_grasp)
                                            np.save(os.path.join(t_string, str(counter) +  "grasp.npy"), grasp.toPoseArray())

                                        transform.visualizeTransform(ee_tf, "goal_EE_pose")

                                        if args.move:
                                            print("Moving to waypoint...")
                                            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)
                                            result = rob9Utils.iiwa.execute_ptp(state_waypoint.solution.joint_state.position[0:7])
                                            rospy.sleep(1)

                                            print("Moving to grasp pose...")
                                            result = rob9Utils.iiwa.execute_ptp(state_grasp.solution.joint_state.position[0:7])
                                            rospy.sleep(1)

                                            gripper_pub.publish(close_gripper_msg)
                                            rospy.sleep(1)

                                            print("I have grasped!")
                                            print("Moving to ready...")
                                            result = rob9Utils.iiwa.execute_ptp(state_waypoint.solution.joint_state.position[0:7])
                                            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)

                                            # Execute plan to handover pose
                                            result = rob9Utils.iiwa.execute_ptp(state_handover.solution.joint_state.position[0:7])
                                            rospy.sleep(2)

                                            gripper_pub.publish(open_gripper_msg)
                                            rospy.sleep(1)
                                            result = rob9Utils.iiwa.execute_ptp(moveit.getJointPositionAtNamed("ready").joint_position.data)
                                        print("Motion complete")
                                        state = 3

                                        x_pos = x_range +1
                                        rot_num = rot_range +1
                                        count_grasp = len(grasps)
                                        break

                                    rot_num += 1
                                x_pos += 1
                    count_grasp += 1

            state = 3
            counter += 1

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
