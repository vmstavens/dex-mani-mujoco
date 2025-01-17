
import roboticstoolbox as rtb
import spatialmath as sm
import spatialmath.base as smb
import numpy as np
from spatialmath import (
    SE3, SO3, UnitQuaternion
)

from typing import Union

import mujoco as mj
from scipy.spatial.transform import Slerp

from spatialmath.base import trnorm

def quat_2_tf(quat:np.ndarray) -> SE3:
    return SE3(trnorm(UnitQuaternion(quat).matrix))

def pos_2_tf(pos:np.ndarray) -> SE3:
    return SE3(x = pos[0], y = pos[1], z = [2])

def tf_2_quat(tf: SE3) -> np.ndarray:
    q = UnitQuaternion(tf.R)
    return np.append([q.s],q.v)

def make_tf(pos: Union[np.ndarray,list] = [0,0,0], ori: Union[np.ndarray,SE3] = [1,0,0,0]) -> SE3:
    """
    Create a SE3 transformation matrix.

    Parameters:
    - pos (np.ndarray): Translation vector [x, y, z].
    - ori (Union[np.ndarray, SE3]): Orientation, can be a rotation matrix, quaternion, or SE3 object.

    Returns:
    - SE3: Transformation matrix.
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

    # Create SE3 object for the translation
    t = SE3(pos)
    # Combine translation and orientation
    T = SE3(t * ori)
    return T

def get_pose(name:str, model: mj.MjModel, data: mj.MjData) -> SE3:
    if name == "hand":
        pos, quat = data.mocap_pos, data.mocap_quat
        return make_tf(pos=pos,quat=quat)
    
    # Find the object ID by name
    obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)

    if obj_id == -1:
        raise ValueError(f"Object with name '{name}' not found.")

    # Get the object pose from the simulation data
    obj_pose = data.xpos[obj_id], data.xquat[obj_id]

    # Extract position and orientation
    pos = obj_pose[0]
    quat = obj_pose[1]

    return make_tf(pos=pos, quat=quat)

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion.

    Parameters:
        q (numpy.ndarray): Quaternion [w, x, y, z].

    Returns:
        Quaternion: Normalized quaternion.
    """
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Cannot normalize a quaternion with zero norm.")
    q = q / norm
    return q

def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Spherical Linear Interpolation (slerp) between two quaternions. [w,x,y,z]
    """

    dot_product = np.dot(q1, q2)

    # Ensure shortest path by flipping one quaternion
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product

    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta_0 = np.arccos(dot_product)
    
    # Handle the case where theta is close to zero
    if np.abs(theta_0 * alpha) < 1e-5:
        q_interpolated = (1.0 - alpha) * q1 + alpha * q2
    else:
        theta = theta_0 * alpha
        try:
            # Attempt to calculate slerp
            q_interpolated = (np.sin((1 - alpha) * theta) * q1 + np.sin(alpha * theta) * q2) / np.sin(theta)
        except ZeroDivisionError:
            # If np.sin(theta) is close to zero, use linear interpolation as a fallback
            q_interpolated = (1.0 - alpha) * q1 + alpha * q2

    return normalize_quaternion(q_interpolated)

def generate_pose_trajectory(start_pose: SE3, end_pose: SE3, n_steps:int ) -> SE3:
    poses = SE3()
    steps = np.linspace(0,1,n_steps)
    poses.append(start_pose)
    for i in steps:
        q_i = quaternion_slerp(
            q1    = tf_2_quat(start_pose), 
            q2    = tf_2_quat(end_pose), 
            alpha = i)

        p_i = start_pose.t + i * (end_pose.t - start_pose.t)

        # x = start_pose.position.x + i * (end_pose.position.x - start_pose.position.x)
        # y = start_pose.position.y + i * (end_pose.position.y - start_pose.position.y)
        # z = start_pose.position.z + i * (end_pose.position.z - start_pose.position.z)

        pose = make_tf(pos=p_i)
        # pose = quat_2_tf(q_i) @ make_tf(pos=p_i)
        poses.append( pose ) 
        # poses.append( list( np.append([x,y,z],interpolated_quaternion) ) )

    poses.pop(0)
    poses.append(end_pose)
    return poses

def rotate_x(T: SE3, angle: float, in_degrees: bool = False) -> SE3:
    """Rotate the T around the x-axis."""
    if in_degrees:
        angle = np.radians(angle)

    rotation_matrix = SO3.Rx(angle) * T.R
    rotated_position = rotation_matrix * T.t

    se3_object = SE3(rotation_matrix)

    # Return a new SE3 with the updated position and rotation
    return make_tf(pos=rotated_position, quat=[1,0,0,0])

def rotate_y(T:SE3, angle: float, in_degrees: bool = False) -> SE3:
    """Rotate the T around the y-axis."""
    if in_degrees:
        angle = np.radians(angle)

    rotation_matrix = SO3.Ry(angle)
    rotated_position = rotation_matrix * T.t

    # Return a new SE3 with the updated position and rotation
    return SE3(pos=rotated_position, rot=rotation_matrix)

def rotate_z(T: SE3, angle: float, in_degrees: bool = False) -> SE3:
    """Rotate the T around the z-axis."""
    if in_degrees:
        angle = np.radians(angle)

    rotation_matrix = SO3.Rz(angle)
    rotated_position = rotation_matrix * T.t

    # Return a new SE3 with the updated position and rotation
    return SE3(pos=rotated_position, rot=rotation_matrix)