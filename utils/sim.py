import json
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

def read_config(json_filepath: str) -> Dict[str,Dict[str,List]]:
    result = {}
    print(f"{json_filepath=}")
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

@dataclass
class RobotConfig:
    def __init__(self,joint_names, joint_values) -> None:
        self._joint_values = joint_values
        self._joint_names = joint_names
    @property
    def joint_values(self) -> List:
        return self._joint_values
    @property
    def joint_names(self) -> List:
        return self._joint_names
    @property
    def dict(self) -> Dict[str,List]:
        result = {}
        for i in range(len(self._joint_values)):
            result[self._joint_names[i]] = self._joint_values[i]
        return result
    def __repr__(self) -> str:
        return self.dict.__str__()