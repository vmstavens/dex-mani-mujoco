import mujoco as mj
import numpy as np
import roboticstoolbox as rtb
import os
import json
import warnings

from typing import List, Union
from spatialmath import SE3

from utils.sim import (
    read_config, 
    save_config,
    config_to_q,
    RobotConfig
)

from utils.mj import (
    get_actuator_names,
    get_actuator_value,
    set_actuator_value
)


class ShadowHand:
    def __init__(self, model: mj.MjModel, data: mj.MjData, args, chirality: str = "rh") -> None:
        self._args = args
        self._model = model
        self._data = data
        self._N_ACTUATORS:int = 20
        self._traj = []
        self._HOME = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._name = "shadow_hand"
        self._chirality = chirality
        self._actuator_names = self._get_actuator_names()
        self._config_dir = self._get_config_dir()
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

    def _get_config_dir(self):
        self._config_dir = self._args.config_dir + self._name + ".json"
        if not os.path.exists(self._config_dir):
            os.makedirs(os.path.dirname(self._config_dir), exist_ok=True)
            warnings.warn(f"config_dir {self._config_dir} could not be found, create empty config")
        return self._config_dir

    def _get_actuator_names(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self._model):
            if self._chirality in ac_name:
                result.append(ac_name)
        return result

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
            hand_actuator_values.append(get_actuator_value(self._data, han))
        hc = RobotConfig(
            actuator_values = hand_actuator_values,
            actuator_names = hand_actuator_names
        )
        return hc

    def _set_q(self, q: Union[str,List]) -> None:
        hand_actuator_names = []
        for an in get_actuator_names(self._model):
            prefix = an.split("_")[0]
            if prefix == "rh" or prefix == "lh":
                hand_actuator_names.append(an)
        for i, han in enumerate(hand_actuator_names):
            set_actuator_value(data=self._data, q=q[i], actuator_name=han)

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
            q:list = self._cfg_to_q(q)
        assert len(q) == self.n_actuators, f"Length of q should be {self.n_actuators}, q had length {len(q)}"
        
        q0 = np.array(self.get_q().actuator_values)
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

    def _cfg_to_q(self, cfg:str) -> List:
        return config_to_q(
            cfg            = cfg, 
            configs        = self._configs, 
            actuator_names = self._actuator_names
        )

    def save_config(self, config_name:str = "placeholder") -> None:
        save_config(
            config_dir  = self._config_dir,
            config      = self.get_q(),
            config_name = config_name
        )

    def home(self) -> None:
        self.set_q(q = "home")

    class ShadowFinger:
        def __init__(self) -> None:
            pass