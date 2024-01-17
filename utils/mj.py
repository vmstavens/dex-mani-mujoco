from typing import List, Dict, Union
import mujoco as mj
from math import degrees
import numpy as np
from spatialmath import SE3
import spatialmath.base as smb
from utils.rtb import make_tf


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

def set_object_pose(data: mj.MjData, model: mj.MjModel, object_name:str, 
            pos: List = [0.5,0.5,0.5], 
            ori: Union[np.ndarray,SE3] = [1,0,0,0], 
            pose: Union[None, List[float], np.ndarray, SE3] = None) -> None:

    if pose is not None:
        if isinstance(pose, SE3):
            target_pose = pose
        else:
            # Assuming pose is a list or numpy array [x, y, z, qw, qx, qy, qz]
            target_pose = SE3(pose[:3], pose[3:])
    else:
        # Use the provided position and orientation
        target_pose = make_tf(pos=pos, ori=ori)

    # print(data.body("flexcell"))
        # base
    # print(data)
    # print(data.mocap_pos[0,0])
    # data.mocap_pos[0,0] += 0.1
    # data.mocap_pos("arm")
    # data.mocap_pos[0,0] = 2
    # data.body("base").xpos[:3] = target_pose.t

def set_robot_pose(data: mj.MjData, model: mj.MjModel, robot_name:str, pose):
    pass