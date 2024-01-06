import json
import numpy as np
import os
import warnings
import mujoco as mj
from typing import List, Dict
from dataclasses import dataclass
from spatialmath import SE3
from utils.rtb import make_tf

def read_config(json_filepath: str) -> Dict[str,Dict[str,List]]:
    result = {}
    try:
        with open(json_filepath, mode='r', encoding='utf-8') as jsonfile:
            # Try to load existing JSON content
            jsonobj = json.load(jsonfile)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is not a valid JSON, create an empty one
        jsonobj = {}
        with open(json_filepath, mode='w', encoding='utf-8') as jsonfile:
            json.dump(jsonobj, jsonfile)

    with open(json_filepath, mode='r', encoding='utf-8') as jsonfile:
        jsonobj = json.load(jsonfile)
        for cfg, finger_js in jsonobj.items():
            result[cfg] = {}
            for finger_name, ctrl_list in finger_js.items():
                if isinstance(ctrl_list,float):
                    result[cfg][finger_name] = ctrl_list
                else:
                    ctrl_transitions = [np.float32(ctrl) for ctrl in ctrl_list]
                    result[cfg][finger_name] = ctrl_transitions
    return result

def config_to_q(cfg:str, configs:Dict, actuator_names: List[str]) -> List[float]:
    try:
        cfg_json = configs[cfg]
        q = []
        for ac_name in actuator_names:
            q.append(cfg_json[ac_name])
        return q
    except KeyError:
        print("Wrong cfg string, try one of the following:")
        for k,v in configs.items():
            print(f"\t{k}")

def get_object_pose(object_name:str, model: mj.MjModel, data: mj.MjData) -> SE3:
    # Find the object ID by name
    obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_name)

    if obj_id == -1:
        raise ValueError(f"Object with name '{object_name}' not found.")

    # Get the object pose from the simulation data
    obj_pose = data.xpos[obj_id], data.xquat[obj_id]

    # Extract position and orientation
    pos = obj_pose[0]
    quat = obj_pose[1]

    pose = make_tf(pos=pos,ori=quat)

    return pose

@dataclass
class RobotConfig:
    def __init__(self,actuator_names: List[str], actuator_values: List[float], name:str = "placeholder") -> None:
        self._actuator_values = actuator_values
        self._actuator_names = actuator_names
        self._name = name
    @property
    def actuator_values(self) -> List:
        return self._actuator_values
    @property
    def actuator_names(self) -> List:
        return self._actuator_names
    @property
    def dict(self) -> Dict[str,List]:
        result = {}
        for i in range(len(self._actuator_values)):
            result[self._actuator_names[i]] = self._actuator_values[i]
        return result
    def __repr__(self) -> str:
        return json.dumps(self.dict, indent=1)
    def save(self, save_dir:str = "config/") -> None:
        """
        Save the RobotConfig instance as a JSON file in the specified directory.

        Parameters:
        - save_dir (str): The directory where the JSON file will be saved.

        Modifies:
        - Creates a JSON file in the specified directory containing the RobotConfig data.
        """
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Construct the file path
        file_path = os.path.join(save_dir, f"{self._name}_config.json")

        # Write the JSON data to the file
        with open(file_path, "w") as json_file:
            json.dump(self.dict, json_file, indent=1)

        print(f"RobotConfig saved successfully to: {file_path}")


def save_config(config_dir:str, config: RobotConfig,config_name:str = "placeholder") -> None:
        cfg_dict = None
        with open(config_dir, mode="r") as config_file:
            cfg_dict = json.load(config_file)
        with open(config_dir, mode="w") as config_file:
            if not (config_name in cfg_dict):
                cfg_dict[config_name] = config.dict
                json.dump(cfg_dict, config_file,indent=4)
            else:
                warnings.warn(f"config of name \"{config_name}\" already exsists in {config_dir}, overwriting old config...")
                del cfg_dict[config_name]
                cfg_dict[config_name] = config.dict
                json.dump(cfg_dict, config_file,indent=4)
        print(f"saved config \"{config_name}\" to config file {config_dir}")