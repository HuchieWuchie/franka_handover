#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import open3d as o3d
from affordanceService.client import AffordanceClient
from fh_utils.msg import *
from fhUtils.graspGroup import GraspGroup
from fhUtils.grasp import Grasp
from fhUtils.affordancetools import getPredictedAffordances, getAffordanceColors
from geometry_msgs.msg import Pose



def visualizeGrasps6DOF(pointcloud, graspGroup, color = None):
    """ input:  pointcloud - open3d pointcloud
                graspGroup - rob9Utils.graspGroup GraspGroup()
    """

    """ Visualizes grasps in a point cloud, sorry for this monster of a
        function.
        Input:  num_to_visualize - number of highest scoring grasps to viz
        Output: None """

    # gripper geometry
    width, height, depth = 0.02, 0.02, 0.02
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    # make a list of open3d geometry
    o3d_gripper_geom = []
    for grasp in graspGroup.grasps:

        center = grasp.position.getVector(format="row")
        R = grasp.orientation.getRotationMatrix()
        score = grasp.score




        left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
        right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
        bottom = create_mesh_box(finger_width, width, height)
        tail = create_mesh_box(tail_length, finger_width, height)

        left_points = np.array(left.vertices)
        left_triangles = np.array(left.triangles)
        left_points[:,0] -= depth_base + finger_width
        left_points[:,1] -= width/2 + finger_width
        left_points[:,2] -= height/2

        right_points = np.array(right.vertices)
        right_triangles = np.array(right.triangles) + 8
        right_points[:,0] -= depth_base + finger_width
        right_points[:,1] += width/2
        right_points[:,2] -= height/2

        bottom_points = np.array(bottom.vertices)
        bottom_triangles = np.array(bottom.triangles) + 16
        bottom_points[:,0] -= finger_width + depth_base
        bottom_points[:,1] -= width/2
        bottom_points[:,2] -= height/2

        tail_points = np.array(tail.vertices)
        tail_triangles = np.array(tail.triangles) + 24
        tail_points[:,0] -= tail_length + finger_width + depth_base
        tail_points[:,1] -= finger_width / 2
        tail_points[:,2] -= height/2

        vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
        vertices = np.dot(R, vertices.T).T + center
        triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)

        colors = None
        if color == None:
            color_r, color_g, color_b = score, 0, 1 - score
            colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])
        else:
            color_r, color_g, color_b = color
            colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])


        gripper = o3d.geometry.TriangleMesh()
        gripper.vertices = o3d.utility.Vector3dVector(vertices)
        gripper.triangles = o3d.utility.Vector3iVector(triangles)
        gripper.vertex_colors = o3d.utility.Vector3dVector(colors)

        o3d_gripper_geom.append(gripper)

    #o3d.visualization.draw_geometries([pointcloud, *o3d_gripper_geom])

    return o3d_gripper_geom

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def create_plane(p1, p2, p3, p4):
    """ input:  p1  -   numpy array (x1, y1, z1)
                p2  -   numpy array (x2, y2, z2)
                p3  -   numpy array (x3, y3, z3)
                p4  -   numpy array (x4, y4, z4)
        output: box -   o3d mesh
    """ """not implemented yet !!!"""
    depth = 0.01 # in meters
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

# Bound selected points and visualize box in open3d
def viz_boundingbox(point_min, point_max):

    xmin = point_min[0]
    xmax = point_max[0]
    ymin = point_min[1]
    ymax = point_max[1]
    zmin = point_min[2]
    zmax = point_max[2]

    points_to_viz = [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],
                      [xmin, ymin, zmax], [xmax, ymin, zmax],
                      [xmin, ymax, zmax], [xmax, ymax, zmax]]
    lines_to_viz = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[1, 0, 0] for i in range(len(lines_to_viz))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_to_viz)
    line_set.lines = o3d.utility.Vector2iVector(lines_to_viz)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def viz_oriented_boundingbox(corners):

    b1,b2,b4,t1,t3,t4,t2,b3 = corners
    points_to_viz = [b1, b2, b4, t1, t3, t4, t2, b3]

    lines_to_viz = [[0, 1], [0, 2], [1,7], [7, 2],
                    [3, 6], [3, 5], [6, 4], [5, 4],
                    [0, 3], [1, 6], [7, 4], [2, 5]]

    colors = [[1, 0, 0] for i in range(len(lines_to_viz))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_to_viz)
    line_set.lines = o3d.utility.Vector2iVector(lines_to_viz)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def vizualizeLine(p1, p2):
    """ Input:  p1  -   numpy array (x, y, z)
                p2  -   numpy array (x, y, z)
        Output: lineset -   o3d.geometry.Lineset
    """
    points_to_viz = [p1, p2]

    lines_to_viz = [[0, 1]]

    colors = [[1, 0, 1] for i in range(len(lines_to_viz))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_to_viz)
    line_set.lines = o3d.utility.Vector2iVector(lines_to_viz)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def visualizeFrame(x, y, z, translation, size = 0.1):
    """ Input:  x   -   numpy array (x, y, z) or column vector
                y   -   numpy array (x, y, z) or column vector
                z   -   numpy array (x, y, z) or column vector
                translation -   numpy array (x, y, z) or column vector
                                the center point of the frame
                size        -   float, length of lines in meters
        output: lines   -   open3d.geometry.LineSet()
    """

    translation = translation.flatten()

    x = (x.flatten() * size) + translation
    y = (y.flatten() * size) + translation
    z = (z.flatten() * size) + translation
    points = [translation, x, y, z]
    lines_to_viz = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0 , 0], [0, 1, 0], [0, 0, 1]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines_to_viz)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def getMeshImage(mesh):
    """ Returns a mesh rendered in an RGB image.

        Input:  mesh    -   o3d.geometry.TriangleMesh()
        Output: img     -   numpy array (height, width, 3 channels)
    """

    rgb_image = None

    renderer_width = 1280
    renderer_height = 720

    # Intrinsic parameters of RealSense RGB stream

    renderer_fx = 910.724 #634.745
    renderer_fy = 908.896 #634.745
    renderer_cx = 642.414 #647.726
    renderer_cy = 360.67 #357.291

    intrinsics = o3d.camera.PinholeCameraIntrinsic(renderer_width, renderer_height,
                        renderer_fx, renderer_fy, renderer_cx, renderer_cy)
    extrinsics = np.eye(4)
    extrinsics[3,3] = 1.0 # 1.5

    renderer = o3d.visualization.rendering.OffscreenRenderer(renderer_width,
                    renderer_height)
    renderer.scene.set_background(np.array([0,0,0,0]))
    renderer.setup_camera(intrinsics, extrinsics)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit'
    material.point_size = 15

    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("pcd", mesh, material)

    img_mesh = np.asarray(renderer.render_to_image())
    img = cv2.cvtColor(img_mesh, cv2.COLOR_RGB2BGR)

    return img

def visualizeMeshInRGB(mesh, img):
    """ Pastes an open3d mesh onto an image, the mesh must already have
        size, location and orientation set before sending to function.

        Input:  mesh    -   o3d.geometry.TriangleMesh()
                img     -   numpy array (height, width, 3 channels)
        Output: img_vis -   numpy array (height, width, 3 channels)
    """
    img_mesh = getMeshImage(mesh)
    img_mesh = cv2.resize(img_mesh, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)

    img_vis = np.copy(img)

    img_vis[img_mesh[:,:, 0] > 30] = img_mesh[img_mesh[:,:, 0] > 30]
    img_vis[img_mesh[:,:, 1] > 30] = img_mesh[img_mesh[:,:, 1] > 30]
    img_vis[img_mesh[:,:, 2] > 30] = img_mesh[img_mesh[:,:, 2] > 30]

    return img_vis

def visualizeFrameMesh(translation, R, size = 0.1):
    """ Input:  translation -   numpy array (x, y, z) or column vector
                                the center point of the frame
                R    -  numpy array 3X3 rotation matrix
                size        -   float, length of lines in meters
        output: axisMesh   -   o3d.geometry.TriangleMesh()
    """

    axisMesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = size)
    axisMesh.rotate(R)
    axisMesh.translate(translation)

    return axisMesh



# Make points ready to visualize in open3d pointcloud with color
def viz_color_pointcloud(points, color):
    colors = [color for i in range(len(points))]
    points = np.array(points)
    pc_viz = o3d.geometry.PointCloud()
    pc_viz.colors = o3d.utility.Vector3dVector(colors)
    pc_viz.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    return pc_viz

def get_optimal_font_scale(text, width):
    """https://stackoverflow.com/questions/52846474/how-to-resize-text-for-cv2-puttext-according-to-the-image-size-in-opencv-python"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            if scale/10 > 1.0:
                return 1
            return scale/10
    return 1

def visualizeBBoxInRGB(im, labels, bboxs, scores):

    aff_client = AffordanceClient(connected = False)

    if bboxs is None:
        print("No bounding boxes to visualize")
        return 0
    if bboxs.shape[0] < 0:
        print("No bounding boxes to visualize")
        return 0

    img = im.copy()

    for box, label, score in zip(bboxs, labels, scores):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        ps = (box[0], box[1])
        pe = (box[2], box[3])
        color = (0, 0, 255)
        thickness = 2
        img = cv2.rectangle(img, ps, pe, color, thickness)

        text = str(aff_client.OBJ_CLASSES[label]) + " " + str(round(score, 2))
        width = int(box[2] - box[0])
        fontscale = get_optimal_font_scale(text, width)
        y_rb = int(box[3] + 40)
        img[int(box[3]):y_rb, int(box[0])-1:int(box[2])+2] = (0, 0, 255)
        img = cv2.putText(img, text, (box[0], int(y_rb-10)), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 255, 255), 2, 2)

    return img

def visualizeMasksInRGB(img, masks, colors = None):
    """ Input:
        img     -   np.array shape (h, w, c)
        masks   -   np.array bool, shape (N, affordances, h, w)

        Output:
        img     -   np.array shape (h, w, c) with masks overlayed
    """

    if len(masks.shape) <= 3:
        masks = np.reshape(masks, (-1, masks.shape[0], masks.shape[1], masks.shape[2]))

    if masks is None:
        print("No masks available to visualize")
        return 0
    if masks.shape[0] < 0:
        print("No masks available to visualize")
        return 0

    if colors is None:
        colors = getAffordanceColors()

    for i, color in enumerate(colors):
        r, g, b = color
        colors[i] = (b, g, r) # opencv uses BGR representation

    else:
        if len(colors) < masks.shape[1]:
            print("Amount of colors is less than amount of affordances")
            return 0

    full_mask = np.zeros(img.shape).astype(np.uint8)
    for obj_mask in np.flipud(masks):
        for count, affordance_mask in enumerate(obj_mask):
            if count >= 1:
                m = affordance_mask == 1
                full_mask[affordance_mask == 1] = colors[count]

    img = cv2.addWeighted(img, 1.0, full_mask, 0.7, 0)

    return img

def createGripper(opening = 0.08, translation = np.zeros(3), rotation = np.identity(3)):
    """ Creates a 3D representation of the Panda hand
        Input:
        opening:        - float, how open is the gripper given in meters.
        depth:          - float, how deep should the gripper grasp, meters
        translation     - np.array, (x, y, z) row or column vector
        rotation        - np.array, (3x3) rotation matrix

        Output:
        parts    - list [o3d.geometry.PointCloud()], a list of open3d
                            geometry
    """

    translation = translation.flatten()

    finger_width = 0.015 # x-axis, in meters 0.02
    finger_length = 0.02 # y-axis, in meters
    finger_height = 0.043 # z-axis
    #finger_offset_z = -0.01
    #finger_offset_z = 0.015
    #finger_offset_z = 0.001
    finger_offset_z = 0.005 # friday 7 oct used to be 0.01

    chasis_width = 0.045 # x-axis, in meters 0.03
    chasis_length = 0.21 # y-axis
    chasis_height = 0.08 # z-axis
    chasis = create_mesh_box(chasis_width, chasis_length, chasis_height,
                            dx = -chasis_width / 2, dy = -chasis_length / 2,
                            dz = -chasis_height -(finger_height)) # friday 7 oct what is -0.035 doing there?

    chasis_points = np.array(chasis.vertices)
    chasis_points = np.dot(rotation, chasis_points.T).T + translation

    chasis = o3d.geometry.PointCloud()
    chasis.points = o3d.utility.Vector3dVector(chasis_points)

    left_finger_points = np.array([

                            [-finger_length / 2.0, (finger_width / 2) -( opening / 2), finger_offset_z],
                            [finger_length / 2.0, (finger_width / 2) -( opening / 2), finger_offset_z],
                            [-finger_length / 2.0, (finger_width / 2) -( opening / 2), -finger_offset_z - finger_height],
                            [finger_length / 2.0, (finger_width / 2) -( opening / 2), -finger_offset_z - finger_height],

                            [-finger_length / 2.0, (-finger_width / 2) -(opening / 2), finger_offset_z],
                            [finger_length / 2.0, (-finger_width / 2) -(opening / 2), finger_offset_z],
                            [-finger_length / 2.0, (-finger_width / 2) -(opening / 2), -finger_offset_z - finger_height],
                            [finger_length / 2.0, (-finger_width / 2) -(opening / 2), -finger_offset_z - finger_height],


                            ])

    left_finger = o3d.geometry.PointCloud()
    left_finger_points = np.dot(rotation, left_finger_points.T).T + translation
    left_finger.points = o3d.utility.Vector3dVector(left_finger_points)

    right_finger_points = np.array([[-finger_length / 2.0, (finger_width / 2) + (opening / 2), finger_offset_z],
                            [finger_length / 2.0, (finger_width / 2) + (opening / 2), finger_offset_z],
                            [-finger_length / 2.0, (finger_width / 2) + (opening / 2), -finger_offset_z - finger_height],
                            [finger_length / 2.0, (finger_width / 2) + (opening / 2), -finger_offset_z - finger_height],

                            [-finger_length / 2.0, (-finger_width / 2) + ( opening / 2), finger_offset_z],
                            [finger_length / 2.0, (-finger_width / 2) + ( opening / 2), finger_offset_z],
                            [-finger_length / 2.0, (-finger_width / 2) + ( opening / 2), -finger_offset_z - finger_height],
                            [finger_length / 2.0, (-finger_width / 2) + ( opening / 2), -finger_offset_z - finger_height],
                            ])

    right_finger = o3d.geometry.PointCloud()
    right_finger_points = np.dot(rotation, right_finger_points.T).T + translation
    right_finger.points = o3d.utility.Vector3dVector(right_finger_points)

    parts = []
    parts.append(chasis)
    parts.append(left_finger)
    parts.append(right_finger)

    return parts



def visualizeGripper(parts, color = (1,0,0)):
    """
        Input:
        parts       - list[o3d.geometry.PointCloud()]

        Output:
        mesh        - o3d.geometry.TriangleMesh()
    """

    vertices_chasis = np.asanyarray(parts[0].points)

    triangles_chasis = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])


    vertices_left_finger = np.asanyarray(parts[1].points)
    triangles_left_finger = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    triangles_left_finger += 8

    vertices_right_finger = np.asanyarray(parts[2].points)
    triangles_right_finger = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    triangles_right_finger += 16

    vertices = np.concatenate([vertices_chasis, vertices_left_finger, vertices_right_finger], axis=0)
    triangles = np.concatenate([triangles_chasis, triangles_left_finger, triangles_right_finger], axis=0)

    hand = o3d.geometry.TriangleMesh()
    hand.vertices = o3d.utility.Vector3dVector(vertices)
    hand.triangles = o3d.utility.Vector3iVector(triangles)
    colors = [color for i in range(vertices.shape[0])]
    hand.vertex_colors = o3d.utility.Vector3dVector(colors)


    return hand

def visualizeGraspInRGB(img, grasp, opening = 0.08, color = (1,0,0)):
    """ Visualizes a grasp in an RGB image
        Input:
        img             - numpy matrix (H, W, 3)
        grasp           - rob9Utils.grasp Grasp()
        opening:        - float, how open is the gripper given in meters.
        color           - tuple 3, 0 to 1, rgb

        Output:
        img_vis         - numpy matrix (H, W, 3)
    """

    translation = grasp.position.getVector(format="row")
    rotation = grasp.orientation.getRotationMatrix()

    finger_width = 0.01 # x-axis, in meters 0.02
    finger_length = 0.04 # y-axis, in meters
    finger_height = 0.04 # z-axis
    finger_offset_z = 0.0025 # -0.0015, 0.025

    chasis_width = 0.02 # x-axis, in meters 0.03
    chasis_length = 0.18 # y-axis
    chasis_height = 0.05 # z-axis

    chasis = create_mesh_box(0.005, 0.005, finger_offset_z,
                            dx = -0.005 / 2, dy = -0.005 / 2,
                            dz = 0)

    chasis_points = np.array(chasis.vertices)
    chasis_points = np.dot(rotation, chasis_points.T).T + translation

    chasis = o3d.geometry.PointCloud()
    chasis.points = o3d.utility.Vector3dVector(chasis_points)

    left_finger_points = np.array([

                            [-chasis_width / 2.0, (finger_width / 2) -( opening / 2), finger_offset_z],
                            [chasis_width / 2.0, (finger_width / 2) -( opening / 2), finger_offset_z],
                            [-chasis_width / 2.0, (finger_width / 2) -( opening / 2), -finger_offset_z],
                            [chasis_width / 2.0, (finger_width / 2) -( opening / 2), -finger_offset_z],

                            [-chasis_width / 2.0, (-finger_width / 2) -(opening / 2), finger_offset_z],
                            [chasis_width / 2.0, (-finger_width / 2) -(opening / 2), finger_offset_z],
                            [-chasis_width / 2.0, (-finger_width / 2) -(opening / 2), -finger_offset_z],
                            [chasis_width / 2.0, (-finger_width / 2) -(opening / 2), -finger_offset_z],


                            ])

    left_finger = o3d.geometry.PointCloud()
    left_finger_points = np.dot(rotation, left_finger_points.T).T + translation
    left_finger.points = o3d.utility.Vector3dVector(left_finger_points)

    right_finger_points = np.array([[-chasis_width / 2.0, (finger_width / 2) + (opening / 2), finger_offset_z],
                            [chasis_width / 2.0, (finger_width / 2) + (opening / 2), finger_offset_z],
                            [-chasis_width / 2.0, (finger_width / 2) + (opening / 2), -finger_offset_z],
                            [chasis_width / 2.0, (finger_width / 2) + (opening / 2), -finger_offset_z],

                            [-chasis_width / 2.0, (-finger_width / 2) + ( opening / 2), finger_offset_z],
                            [chasis_width / 2.0, (-finger_width / 2) + ( opening / 2), finger_offset_z],
                            [-chasis_width / 2.0, (-finger_width / 2) + ( opening / 2), -finger_offset_z],
                            [chasis_width / 2.0, (-finger_width / 2) + ( opening / 2), -finger_offset_z],
                            ])

    right_finger = o3d.geometry.PointCloud()
    right_finger_points = np.dot(rotation, right_finger_points.T).T + translation
    right_finger.points = o3d.utility.Vector3dVector(right_finger_points)


    parts = []
    parts.append(left_finger)
    parts.append(right_finger)
    parts.append(chasis)


    vertices_left_finger = np.asanyarray(parts[0].points)
    triangles_left_finger = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    triangles_left_finger += 0

    vertices_right_finger = np.asanyarray(parts[1].points)
    triangles_right_finger = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    triangles_right_finger += 8

    vertices_chasis = np.asanyarray(parts[2].points)
    triangles_chasis = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    triangles_chasis += 16


    vertices = np.concatenate([vertices_chasis, vertices_left_finger, vertices_right_finger], axis=0)
    triangles = np.concatenate([triangles_chasis, triangles_left_finger, triangles_right_finger], axis=0)
    #vertices = np.concatenate([vertices_left_finger, vertices_right_finger], axis=0)
    #triangles = np.concatenate([triangles_left_finger, triangles_right_finger], axis=0)

    hand = o3d.geometry.TriangleMesh()
    hand.vertices = o3d.utility.Vector3dVector(vertices)
    hand.triangles = o3d.utility.Vector3iVector(triangles)
    colors = [color for i in range(vertices.shape[0])]
    hand.vertex_colors = o3d.utility.Vector3dVector(colors)

    img_mesh = getMeshImage(hand)
    img_mesh = cv2.resize(img_mesh, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
    img_mesh_gray = cv2.cvtColor(img_mesh, cv2.COLOR_BGR2GRAY)
    img_mesh_gray[img_mesh_gray > 10] = 255

    img_vis = np.copy(img)

    contours, hierarchy = cv2.findContours(img_mesh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    top_3_areas = [0, 0, 0]
    idxs = [0, 0, 0]
    for c_contour, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if contour_area > top_3_areas[2]:
            top_3_areas[2] = contour_area
            idxs[2] = c_contour

            if contour_area > top_3_areas[1]:
                top_3_areas[2] = top_3_areas[1]
                top_3_areas[1] = contour_area

                idxs[2] = idxs[1]
                idxs[1] = c_contour

                if contour_area > top_3_areas[0]:
                    top_3_areas[1] = top_3_areas[0]
                    top_3_areas[0] = contour_area

                    idxs[1] = idxs[0]
                    idxs[0] = c_contour

    idxs.sort()

    texts = ["Small finger", "Grasping point", "Big finger"]

    overall_contour = np.vstack([contours[idxs[0]], contours[idxs[1]], contours[idxs[2]]])
    rect = cv2.minAreaRect(overall_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img_vis = cv2.drawContours(img_vis, [box], 0, (0,0,0), 2)

    for i, idx in enumerate(idxs):

        x, y, w, h = cv2.boundingRect(contours[idx])
        font = cv2.FONT_HERSHEY_SIMPLEX

        if (1280 - (x + w)) > x:
            org = (x+w, y+int(h/2.0))
        else:
            org = (x-250, y +int(h/2.0))
        fontScale = 1
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        thickness = 2
        bg_thickness = thickness + 2

        img_vis = cv2.putText(img_vis, texts[i], org, font, fontScale, bg_color, bg_thickness, cv2.LINE_AA)
        img_vis = cv2.putText(img_vis, texts[i], org, font, fontScale, color, thickness, cv2.LINE_AA)

    img_vis[img_mesh[:,:, 0] > 30] = img_mesh[img_mesh[:,:, 0] > 30]
    img_vis[img_mesh[:,:, 1] > 30] = img_mesh[img_mesh[:,:, 1] > 30]
    img_vis[img_mesh[:,:, 2] > 30] = img_mesh[img_mesh[:,:, 2] > 30]



    return img_vis
