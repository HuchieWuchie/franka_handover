#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Header, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from fhUtils.graspGroup import GraspGroup, Grasp

def packPCD(geometry, color, frame_id = ""):
  """ Input:
      geometry    - np.array, contains x, y, z information
      color       - np.array, contains r, g, b information

      Output:
      msg_geometry - sensor_msgs.msg.PointCloud2
      msg_color    - std_msgs.Float32MultiArray
  """
  FIELDS_XYZ = [
      PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
      PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
      PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
  ]

  assert len(frame_id) != 0

  header = Header()
  header.stamp = rospy.Time.now()
  header.frame_id = frame_id

  msg_geometry = pc2.create_cloud(header, FIELDS_XYZ, geometry)
  msg_color = 0
  if color is not None:
      msg_color = Float32MultiArray()
      msg_color.data = color.astype(float).flatten().tolist()

  return msg_geometry, msg_color

def unpackPCD(msg_geometry, msg_color):
  """ Input:
      msg_geometry - sensor_msgs.msg.PointCloud2
      msg_color    - std_msgs.Float32MultiArray

      Output:
      geometry    - np.array, contains x, y, z information
      color       - np.array, contains r, g, b information
  """

  # Get cloud data from ros_cloud
  field_names = [field.name for field in msg_geometry.fields]
  geometry_data = list(pc2.read_points(msg_geometry, skip_nans=True, field_names = field_names))

  # Check empty
  if len(geometry_data)==0:
      print("Converting an empty cloud")
      return None, None

  geometry = [(x, y, z) for x, y, z in geometry_data ] # get xyz
  geometry = np.array(geometry)

  # get colors
  color = 0
  if msg_color is not None:
      color = np.asarray(msg_color.data)
      color = np.reshape(color, (-1,3)) / 255
      color = np.flip(color, axis=1)

  return geometry, color


def packUV(uv):
  """ packs the uv data into a Float32MultiArray

      Input:
      uv          -   np.array, int, shape (N, 2)

      Output:
      msg         -   std_msgs.msg.Float32MultiArray()
  """

  uvDim1 = MultiArrayDimension()
  uvDim1.label = "length"
  uvDim1.size = int(uv.shape[0] * uv.shape[1])
  uvDim1.stride = uv.shape[0]

  uvDim2 = MultiArrayDimension()
  uvDim2.label = "pair"
  uvDim2.size = uv.shape[1]
  uvDim2.stride = uv.shape[1]

  uvLayout = MultiArrayLayout()
  uvLayout.dim.append(uvDim1)
  uvLayout.dim.append(uvDim2)

  msg = Float32MultiArray()
  msg.data = uv.flatten().tolist()
  msg.layout = uvLayout

  return msg

def unpackUV(msg):
  """ unpacks the uv msg into a numpy array

      Input:
      msg         -   std_msgs.msg.Float32MultiArray()

      Output:
      uv          -   np.array, int, shape (N, 2)
  """

  rows = int(msg.layout.dim[0].size / msg.layout.dim[1].size)
  cols = int(msg.layout.dim[1].size)

  uv = np.array(msg.data).astype(int)
  uv = np.reshape(uv, (rows, cols))
  return uv

def packOrientation(current_transformation, goal_orientation):

    msg_current = Float32MultiArray()
    current_transformation = current_transformation.flatten().tolist()
    msg_current.data = current_transformation

    msg_goal = Float32MultiArray()
    goal_orientation = goal_orientation.flatten().tolist()
    msg_goal.data = goal_orientation

    return msg_current, msg_goal

def unpackOrientation(msg_current, msg_goal):
    current_transformation = np.asarray(msg_current.data).reshape((4,4))
    orientation = current_transformation[:3,:3]
    translation = current_transformation[:3,3]

    goal = np.asarray(msg_goal.data).reshape((3,3))
    return orientation, translation, goal

def packGrasps(poses, scores, frame_id, tool_id,
                                    affordance_id, obj_inst):

    grasps = []
    for pose, score in zip(poses, scores):

        grasp = Grasp(frame_id = frame_id)
        grasp.position.set(x = pose[0], y = pose[1], z = pose[2])
        grasp.orientation.setQuaternion(pose[3:])
        grasp.score = score
        grasp.tool_id = tool_id
        grasp.affordance_id = affordance_id
        grasp.setObjectInstance(obj_inst)

        grasp_msg = grasp.toGraspMsg()
        grasps.append(grasp_msg)

    return grasps
