from typing import List
import mujoco as mj

def get_actuator_names(model: mj.MjModel) -> List[str]:
    """
    Get a list of actuator names from the MuJoCo model.

    Returns:
    - List[str]: A list containing the names of all actuators in the MuJoCo model.
    """
    return [model.actuator(i).name for i in range(model.nu)]

def get_joint_value(data: mj.MjData, joint_name: str) -> float:
    """
    Retrieve the control value for a specific joint in the MuJoCo simulation.

    Parameters:
    - joint_name (str): The name of the joint for which the control value is to be retrieved.

    Returns:
    - float: The control value of the specified joint.
    """
    return data.actuator(joint_name).ctrl[0]

def set_joint_value(data: mj.MjData,q:float, joint_name:str) -> None:
    """
    Set the control value for a specific joint in the MuJoCo simulation.

    Parameters:
    - q (float): The control value to be set for the joint.
    - joint_name (str): The name of the joint for which the control value is to be set.
    """
    data.actuator(joint_name).ctrl = q