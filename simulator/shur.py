from utils.rtb import make_tf
from .shadow_hand import ShadowHand
from .ur10e import UR10e
from typing import List, Union, Dict
import os 
import warnings
import roboticstoolbox as rtb
import mujoco as mj
import math as m
import spatialmath as sm
import numpy as np
import json
import random

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
    set_actuator_value,
)

class SHUR:
    def __init__(self, model: mj.MjModel, data: mj.MjData, args) -> None:
        
        self._args = args
        self._model = model
        self._data = data
        self._name = "shur"

        UR_EE_TO_SH_WRIST_JOINTS = 0.21268             # m
        SH_WRIST_TO_SH_PALM      = 0.08721395775941231 # m
        
        self._ur10e       = UR10e(model, data, args)
        self._shadow_hand = ShadowHand(model, data, args)

        # self._robot = rtb.DHRobot(
        #     [
        #         rtb.RevoluteDH(d = 0.1807, alpha = m.pi / 2.0),         # J1
        #         rtb.RevoluteDH(a = -0.6127),                            # J2
        #         rtb.RevoluteDH(a = -0.57155),                           # J3
        #         rtb.RevoluteDH(d = 0.17415, alpha =  m.pi / 2.0),       # J4
        #         rtb.RevoluteDH(d = 0.11985, alpha = -m.pi / 2.0),       # J5
        #         rtb.RevoluteDH(d = 0.11655 + UR_EE_TO_SH_WRIST_JOINTS + SH_WRIST_TO_SH_PALM), # J6 + forearm
        #         # rtb.RevoluteDH(alpha = m.pi / 2),                       # WR1
        #         # rtb.RevoluteDH(alpha = m.pi / 2, offset= m.pi / 2),     # WR2
        #         # rtb.RevoluteDH(d = SH_WRIST_TO_SH_PALM),                # from wrist to palm
        #     ], name=self._name, base=sm.SE3.Trans(0,0,0)
        # )

        self._N_ACTUATORS:int = self.shadow_hand.n_actuators + self.ur10e.n_actuators
        self._actuator_names = self._get_actuator_names()
        self._config_dir = args.config_dir + self._name + ".json"
        if not os.path.exists(self._config_dir):
            os.makedirs(os.path.dirname(self._config_dir), exist_ok=True)
            warnings.warn(f"config_dir {self._config_dir} could not be found, create empty config")
        self._configs = read_config(self._config_dir)

    @property
    def shadow_hand(self) -> ShadowHand:
        return self._shadow_hand

    @property
    def ur10e(self) -> UR10e:
        return self._ur10e

    @property
    def is_done(self) -> bool:
        return True if (self.shadow_hand.is_done and self.ur10e.is_done) else False

    @property
    def n_actuators(self) -> int:
        return self._N_ACTUATORS

    def _get_actuator_names(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self._model):
            if self.ur10e._name in ac_name or self.shadow_hand._chirality in ac_name:
                result.append(ac_name)
        return result

    def get_q(self) -> RobotConfig:
        robot_actuator_names = []
        robot_actuator_values = []
        for an in get_actuator_names(self._model):
            prefix = an.split("_")[0]
            if prefix == "rh" or prefix == "lh" or prefix == "ur10e":
                robot_actuator_names.append(an)
        for han in robot_actuator_names:
            robot_actuator_values.append(get_actuator_value(self._data, han))
        rc = RobotConfig(
            actuator_values = robot_actuator_values,
            actuator_names = robot_actuator_names
        )
        return rc

    def _set_q(self, q: Union[str,List]) -> None:
        """
        Set the control values for the arm actuators in the MuJoCo simulation.

        This private method is responsible for updating the joint values of the arm actuators
        based on the provided control values. It iterates through the arm actuators' names,
        extracts the corresponding control values from the input list, and updates the MuJoCo
        data with the new joint values.

        Parameters:
        - q (Union[str, List]): Either a configuration string or a list of control values
        for the arm actuators.

        Raises:
        - AssertionError: If the length of q does not match the expected number of arm actuators.
        """
        robot_actuator_names = []
        for an in get_actuator_names(self._model):
            if ("ur10e" in an) or ("rh" in an) or ("lh" in an):
                robot_actuator_names.append(an)
        for i in range(len(robot_actuator_names)):
            set_actuator_value(data=self._data, q=q[i], actuator_name=robot_actuator_names[i])

    def set_q(self, q: Union[str, List, RobotConfig], n_steps: int = 10) -> None:
        """
        Set the control values for the arm actuators in the MuJoCo simulation.

        Parameters:
        - q (Union[str, List]): Either a configuration string or a list of control values for the arm.

        Raises:
        - AssertionError: If the length of q does not match the expected number of arm actuators.

        Modifies:
        - Sets the control values for the arm actuators in the MuJoCo simulation.
        """
        if isinstance(q,str):
            q:list = config_to_q(cfg=q,configs=self._configs,actuator_names=self._actuator_names)
        assert len(q) == self._N_ACTUATORS, f"Length of q should be {self._N_ACTUATORS}, q had length {len(q)}"
        
        q0 = np.array(self.get_q().actuator_values)
        qf = np.array(q)

        self._traj = rtb.jtraj(
            q0 = q0,
            qf = qf,
            t = n_steps
        ).q.tolist()

    def home(self) -> None:
        self.shadow_hand.home()
        self.ur10e.home()

    def step(self) -> None:
        if not self.ur10e.is_done:
            self.ur10e.step()
        if not self.shadow_hand.is_done:
            self.shadow_hand.step()

    def save_config(self, config_name:str = "placeholder") -> None:
        save_config(
            config_dir  = self._config_dir,
            config      = self.get_q(),
            config_name = config_name
        )