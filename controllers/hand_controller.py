import numpy as np
from typing import List, Dict, Union
import json
from utils.control import read_ctrl_limits, read_sign_transitions, generate_control_trajectory, read_config

class HandController:
    def __init__(self, ctrl_n_steps:int = 100, config_dir: str = "configs/hand.json") -> None:
        self._config_dir = config_dir
        self._cfgs = read_config(self._config_dir)
        self._n_steps = ctrl_n_steps
        self._traj = []
        self._i = 0

    @property
    def is_done(self) -> bool:
        return self._is_done()

    def _is_done(self) -> bool:
        return True if len(self._traj) == 0 else False

    def cfg_to_q(self, cfg:str) -> List:
        try:
            cfg_json = self._cfgs[cfg]
            q = cfg_json["wr"] + cfg_json["th"] + cfg_json["ff"] + cfg_json["mf"] + cfg_json["rf"] + cfg_json["lf"]
            return q
        except KeyError:
            print("Wrong cfg string, try one of the following:")
            for k,v in self._cfgs.items():
                print(f"\t{k}")

    def set_traj(self, start_ctrl: np.ndarray, end_ctrl: np.ndarray, n_steps = None) -> None:
        if n_steps is not None:
            self._traj = generate_control_trajectory(start_ctrl=start_ctrl, end_ctrl=end_ctrl,n_steps=n_steps)
        else:
            self._traj = generate_control_trajectory(start_ctrl=start_ctrl, end_ctrl=end_ctrl,n_steps=self._n_steps)

    def get_next_control(self) -> Union[List, None]:
        if self.is_done:
            return None
        return self._traj.pop(0)
    
# "open": [[0, 0, -0.9, 1.0, 0.0586, 0.161, 0.7, 0, 1.0, 0, 0, 1.0, 0, 0.2, 1.0, 0, 0, 0.2, 1.0, 0.0]]
#   "rest": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
