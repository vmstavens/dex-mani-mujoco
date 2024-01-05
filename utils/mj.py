from typing import List
import mujoco as mj
from math import degrees

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

def get_joint_names(model: mj.MjModel) -> List[str]:
    return [model.joint(i).name for i in range(model.nu)]

def get_joint_value(data: mj.MjData, joint_name: str, rad: bool = True) -> float:
    return data.joint(joint_name).qpos[0] if rad else degrees(data.joint(joint_name).qpos[0])

def set_joint_value(data: mj.MjData, joint_name: str, q: float) -> None:
    data.joint(joint_name).qpos[0] = q

def is_done_actuator(data: mj.MjData, joint_name:str ,actuator_name: str, epsilon:float = 1e-1) -> bool:
    q_joint = get_joint_value(data,joint_name)
    q_actuator = get_actuator_value(data,actuator_name)
    return True if abs(q_joint - q_actuator) < epsilon else False