import mujoco as mj
import numpy as np
import roboticstoolbox as rtb
import os
import warnings

from typing import List, Union
from spatialmath import SE3

from utils.sim import (
    read_config, 
    RobotConfig
)

from utils.mj import (
    get_actuator_names,
    get_joint_value,
    set_joint_value
)


class ShadowHand:
    def __init__(self, model: mj.MjModel, data: mj.MjData, config_dir:str = "config/") -> None:
        self._model = model
        self._data = data
        self._N_ACTUATORS:int = 20
        self._traj = []
        self._HOME = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._name = "shadow_hand"
        self._config_dir = config_dir
        if os.path.isfile(self._config_dir):
            print(self._config_dir, type(self._config_dir))
            self._configs = read_config(self._config_dir)
        else:
            self._config_dir = "config/shadow_hand.json"
            warnings.warn(f"config_dir {self._config_dir} could not be found, using default config/shadow_hand.json")
            self._configs = read_config(self._config_dir)

    @property
    def n_actuators(self) -> int:
        return self._N_ACTUATORS
    
    @property
    def is_done(self) -> bool:
        return self._is_done()

    @property
    def trajectory(self) -> SE3:
        return self._traj

    def _is_done(self) -> bool:
        return True if len(self._traj) == 0 else False

    def get_q(self) -> RobotConfig:
        """
        Get the configuration of the hand's actuators in the MuJoCo simulation.

        Returns:
        - RobotConfig: An object containing joint values and names for the hand actuators.
        """
        hand_actuator_names = []
        hand_actuator_values = []
        for an in get_actuator_names(self._model):
            prefix = an.split("_")[0]
            if prefix == "rh" or prefix == "lh":
                hand_actuator_names.append(an)
        for han in hand_actuator_names:
            hand_actuator_values.append(get_joint_value(self._data, han))
        hc = RobotConfig(
            joint_values = hand_actuator_values,
            joint_names = hand_actuator_names
        )
        return hc

    def _set_q(self, q: Union[str,List]) -> None:
        hand_actuator_names = []
        for an in get_actuator_names(self._model):
            prefix = an.split("_")[0]
            if prefix == "rh" or prefix == "lh":
                hand_actuator_names.append(an)
        for i, han in enumerate(hand_actuator_names):
            set_joint_value(data=self._data, q=q[i], joint_name=han)

    def set_q(self, q: Union[str, List], n_steps: int = 10) -> None:
        """
        Set the trajectory for the hand's actuators in the MuJoCo simulation.

        Parameters:
        - q (Union[str, List]): Either a configuration string or a list of control values for the hand.

        Raises:
        - AssertionError: If the length of q does not match the expected number of hand actuators.

        Modifies:
        - Sets the trajectory for the hand controllers using the provided configuration.
        """
        if isinstance(q,str):
            q:list = self.cfg_to_q(q)
        assert len(q) == self.n_actuators, f"Length of q should be {self.n_actuators}, q had length {len(q)}"
        
        q0 = np.array(self.get_q().joint_values)
        qf = np.array(q)

        self._traj = rtb.jtraj(
            q0 = q0,
            qf = qf,
            t = n_steps
        ).q.tolist()

    def step(self) -> List:
        if self.is_done:
            return None
        self._set_q(self._traj.pop(0))

    def cfg_to_q(self, cfg:str) -> List:
        try:
            cfg_json = self._configs[cfg]
            q = cfg_json["wr"] + cfg_json["th"] + cfg_json["ff"] + cfg_json["mf"] + cfg_json["rf"] + cfg_json["lf"]
            return q
        except KeyError:
            print("Wrong cfg string, try one of the following:")
            for k,v in self._cfgs.items():
                print(f"\t{k}")

    def home(self) -> None:
        self.set_q(self._HOME)

    class ShadowFinger:
        def __init__(self) -> None:
            pass