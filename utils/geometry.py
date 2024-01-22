#!/usr/bin/env python3

from geometry_msgs.msg import Vector3
from math import sqrt, pow, acos
from typing import Tuple, List, Union
import tf
from math import radians
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped, Pose
import numpy as np
from spatialmath import SE3, UnitQuaternion
import spatialmath.base as smb
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class geometry:

	@staticmethod
	def mk_img(img: np.ndarray) -> Image:
		"""Convert a NumPy array to a ROS Image message."""
		bridge = CvBridge()
		image_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")
		return image_msg

	@staticmethod
	def mk_float64multiarray(l: list) -> Float64MultiArray:
		result = Float64MultiArray()
		result.data = l
		return result

	@staticmethod
	def mk_pose(pos: Union[np.ndarray,list] = [0,0,0], ori: Union[np.ndarray,SE3] = [1,0,0,0]) -> Pose:
		"""
		Create a gemetry_msg.msg.Pose.

		Parameters:
		- pos (np.ndarray): Translation vector [x, y, z].
		- ori (Union[np.ndarray, SE3]): Orientation, can be a rotation matrix, quaternion (w,x,y,z), or SE3 object.

		Returns:
		- gemetry_msg.msg.Pose: Pose.
		"""

		if isinstance(ori, list):
			ori = np.array(ori)

		# Convert ori to SE3 if it's already a rotation matrix or a quaternion
		if isinstance(ori, np.ndarray):
			if ori.shape == (3, 3):  # Assuming ori is a rotation matrix
				ori = SE3(smb.r2t(ori))
			elif ori.shape == (4,):  # Assuming ori is a quaternion
				ori = smb.r2t(UnitQuaternion(s=ori[0],v=ori[1:]).R)
			elif ori.shape == (3,):  # Assuming ori is rpy
				ori = SE3.Eul(ori, unit='rad')

		pose = Pose()
		pose.position.x = pos[0]
		pose.position.y = pos[1]
		pose.position.z = pos[2]
		ori = smb.r2q(ori.R)
		pose.orientation.w = ori[0]
		pose.orientation.x = ori[1]
		pose.orientation.y = ori[2]
		pose.orientation.z = ori[3]

		return pose

	@staticmethod
	def dot(a: Vector3, b: Vector3) -> float:
		"""perform a dot product (scalar product) of the two geometry_msgs.msg.Vector3 provided"""
		return a.x * b.x + a.y * b.y + a.z * b.z

	@staticmethod
	def pow(a: Vector3, b: float) -> Vector3:
		"""sets each element in a to the power of b, where a is a geometry_msgs.msg.Vector3 and b is a float"""
		return Vector3( pow(a.x,b), pow(a.y,b), pow(a.z,b) )

	@staticmethod
	def prod(a: Vector3, b: float) -> Vector3:
		"""multiply each element in a with b, where a is a geometry_msgs.msg.Vector3 and b is a float"""
		return Vector3( a.x*b, a.y*b, a.z*b )

	@staticmethod
	def l2(a:Vector3) -> float:
		"""returns the L2 norm of the vector e.g. euclidean distance sqrt( x² + y² + z² )"""
		return sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2))

	@staticmethod
	def l2_dist(a: Vector3, b: Vector3):
		"""Euclidean distance between 3D points in space"""
		return sqrt( pow(a.x - b.x,2) + pow(a.y - b.y,2) + pow(a.z - b.z,2) )

	@staticmethod
	def angle(a: Vector3, b: Vector3) -> float:
		"""returns the angle between the two vectors a and b"""
		return acos( ( geometry.dot(a,b) ) / (  geometry.l2(a) * geometry.l2(b)) )

	@staticmethod
	def flip(a: Vector3) -> Vector3:
		"""flips the direction of the vector a"""
		return Vector3( -1.0*a.x, -1.0*a.y, -1.0*a.z)

	@staticmethod
	def tup2vec(t: Tuple) -> Vector3:
		"""convert a tuple to a geometry.Vector3"""
		return Vector3(t[0], t[1], t[2])

	@staticmethod
	def vec2tup(v: Vector3) -> Tuple:
		"""convert a tuple to a geometry.Vector3"""
		return v.x, v.y, v.z
	
	def euler2quaternion(roll, pitch, yaw) -> Quaternion:
		"""
		Convert Euler angles to geometry quaternions.

		Parameters:
			roll (float): Roll angle in degrees.
			pitch (float): Pitch angle in degrees.
			yaw (float): Yaw angle in degrees.

		Returns:
			tuple: A tuple representing the quaternion (x, y, z, w).
		"""
		# Create a quaternion from Euler angles
		
		quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

		q = Quaternion()

		q.x = quaternion[0]
		q.y = quaternion[1]
		q.z = quaternion[2]
		q.w = quaternion[3]

		return q