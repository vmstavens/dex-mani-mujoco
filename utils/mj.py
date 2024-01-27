from typing import List, Dict, Union
import mujoco as mj
from math import degrees
import numpy as np
from spatialmath import SE3
import spatialmath.base as smb
from utils.rtb import make_tf
import spatialmath as sm


def get_actuator_names(model: mj.MjModel) -> List[str]:
    """
    Get a list of actuator names from the MuJoCo model.

    Returns:
    - List[str]: A list containing the names of all actuators in the MuJoCo model.
    """
    return [model.actuator(i).name for i in range(model.nu)]

def get_actuator_value(data: mj.MjData, actuator_name: str) -> float:
    """
    Retrieve the control value for a specific joint in the MuJoCo simulation.

    Parameters:
    - actuator_name (str): The name of the joint for which the control value is to be retrieved.

    Returns:
    - float: The control value of the specified joint.
    """
    return data.actuator(actuator_name).ctrl[0]

def set_actuator_value(data: mj.MjData, q:float, actuator_name:str) -> None:
    """
    Set the control value for a specific joint in the MuJoCo simulation.

    Parameters:
    - q (float): The control value to be set for the joint.
    - actuator_name (str): The name of the joint for which the control value is to be set.
    """
    data.actuator(actuator_name).ctrl = q

def get_actuator_range(model, actuator_name:str) -> List[float]:
    return model.actuator(actuator_name).ctrlrange

def get_joint_names(model: mj.MjModel) -> List[str]:
    return [model.joint(i).name for i in range(model.nu)]

def get_joint_value(data: mj.MjData, joint_name: str, rad: bool = True) -> float:
    return data.joint(joint_name).qpos[0] if rad else degrees(data.joint(joint_name).qpos[0])

def set_joint_value(data: mj.MjData, joint_name: str, q: float) -> None:
    data.joint(joint_name).qpos[0] = q

def get_joint_range(model, joint_name:str) -> List[float]:
    return model.joint(joint_name).range

def is_done_actuator(data: mj.MjData, joint_name:str ,actuator_name: str, epsilon:float = 1e-1) -> bool:
    q_joint = get_joint_value(data,joint_name)
    q_actuator = get_actuator_value(data,actuator_name)
    return True if abs(q_joint - q_actuator) < epsilon else False

def get_joint_actuator_diff(data: mj.MjData, joint_name:str ,actuator_name: str) -> List[float]:
    q_joint = get_joint_value(data,joint_name)
    q_actuator = get_actuator_value(data,actuator_name)
    return q_joint - q_actuator

def set_robot_pose(data: mj.MjData, model: mj.MjModel, robot_name:str, pose):
    pass

def get_object_pose(data: mj.MjData, object_name: str) -> SE3:
    """
    Get the current pose of an object in the MuJoCo simulation.

    Parameters:
    - data (mj.MjData): MuJoCo data object.
    - object_name (str): Name of the object whose pose needs to be retrieved.

    Returns:
    - smb.SE3: The current pose of the object as a spatialmath SE3 object.
    """
    # Retrieve the current position and quaternion orientation of the object
    xpos = data.body(object_name).xpos[:3]
    xquat = data.body(object_name).xquat

    # Convert quaternion to rotation matrix
    rotation_matrix = smb.q2r(xquat)

    # Create spatialmath SE3 object
    current_pose = make_tf(pos=xpos,ori=rotation_matrix)

    return current_pose


def set_object_pose(data: mj.MjData, model: mj.MjModel, object_name: str,
                    pos: List = [0.5, 0.5, 0.5],
                    ori: Union[np.ndarray, SE3] = [1, 0, 0, 0],
                    pose: Union[None, List[float], np.ndarray, SE3] = None) -> np.ndarray:
    """
    Set the pose of an object in the MuJoCo simulation.

    Parameters:
    - data (mj.MjData): MuJoCo data object.
    - model (mj.MjModel): MuJoCo model object.
    - object_name (str): Name of the object whose pose needs to be set.
    - pos (List): Position [x, y, z] of the object.
    - ori (Union[np.ndarray, SE3]): Orientation of the object as a quaternion [qw, qx, qy, qz] or SE3 object.
    - pose (Union[None, List[float], np.ndarray, SE3]): If provided, set the pose directly.

    Returns:
    - np.ndarray: The current pose of the object [x, y, z, qw, qx, qy, qz].
    """

    if pose is not None:
        if isinstance(pose, SE3):
            target_pose = pose
        else:
            # Assuming pose is a list or numpy array [x, y, z, qw, qx, qy, qz]
            target_pose = SE3(pose[:3], pose[3:])
    else:
        # Use the provided position and orientation
        target_pose = make_tf(pos=pos, ori=ori)

    # Set the pose
    data.body(object_name).xpos[:3] = target_pose.t
    data.body(object_name).xquat[:] = target_pose.R.q

    # Return the current pose
    current_pose = np.concatenate([data.body(object_name).xpos[:3], data.body(object_name).xquat])
    return current_pose
