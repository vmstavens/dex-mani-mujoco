from typing import List
import mujoco as mj

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

def set_actuator_value(data: mj.MjData,q:float, actuator_name:str) -> None:
    """
    Set the control value for a specific joint in the MuJoCo simulation.

    Parameters:
    - q (float): The control value to be set for the joint.
    - actuator_name (str): The name of the joint for which the control value is to be set.
    """
    data.actuator(actuator_name).ctrl = q