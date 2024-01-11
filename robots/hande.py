import mujoco as mj
import numpy as np
import roboticstoolbox as rtb
import os
import json
import warnings
from robots.base_robot import BaseRobot
from typing import List, Union, Tuple
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
    get_joint_value,
    is_done_actuator,
    get_joint_names
)


class HandE(BaseRobot):
    def __init__(self, model: mj.MjModel, data: mj.MjData, args) -> None:
        super().__init__()
        
        self._args = args

        self._model = model
        self._data = data
        self._actuator_names = self._get_actuator_names()
        self._joint_names    = self._get_joint_names()
        self._config_dir     = self._get_config_dir()
        self._configs        = self._get_configs()

        self._coupled_joints = self._get_coupled_joints()

        self._robot_handle = None
    @property
    def args(self):
        return self._args

    @property
    def mj_data(self) -> mj.MjData:
        return self._data
    
    @property
    def mj_model(self) -> mj.MjModel:
        return self._model

    @property
    def name(self) -> str:
        return "hand-e"

    def _set_robot_handle(self, robot_handle):
        self._robot_handle = robot_handle

    def _config_to_q(self, config: str) -> List[float]:
        return config_to_q(
                    cfg = config, 
                    configs = self._configs, 
                    actuator_names = self._actuator_names
                )

    def _get_coupled_joints(self) -> List[Tuple[str,str]]:
        return [ 
                (self.name + "_FFJ1",self.name + "_FFJ2"), 
                (self.name + "_MFJ1",self.name + "_MFJ2"), 
                (self.name + "_RFJ1",self.name + "_RFJ2"), 
                (self.name + "_LFJ1",self.name + "_LFJ2") 
                ]

    def _are_done_actuators(self) -> bool:
        joint_names = self._joint_names
        actuator_names = self._actuator_names
        joint_ids = [jn.split("_")[1] for jn in joint_names]

        actuator_checks = [
            is_done_actuator(self.mj_data, joint_names[i], an)
            for i, jid in enumerate(joint_ids)
            for an in actuator_names
            if jid in an
        ]

        coupled_joint_checks = [
            (get_joint_value(self.mj_data, tup[0]) + get_joint_value(self.mj_data, tup[1])) < 1e-1
            for tup in self._coupled_joints
        ]

        all_checks = actuator_checks + coupled_joint_checks

        return np.all(all_checks)

    def set_q(self, q : Union[str, List, RobotConfig]):
        if isinstance(q, str):
            q: List[float] = self._config_to_q(config=q)
        if isinstance(q, RobotConfig):
            q: List[float] = q.joint_values
        assert len(q) == self.n_actuators, f"Length of q should be {self.n_actuators}, q had length {len(q)}"

        qf = self._robot_handle._traj[-1].copy()

        qf[-self.n_actuators:] = self._clamp_q(q)

        self._robot_handle._traj.extend([qf])

    def set_ee_pose(
            self, 
            pos: List = [0.5, 0.5, 0.5], 
            ori: Union[np.ndarray,SE3] = [1, 0, 0, 0], 
            pose: Union[List[float], np.ndarray, SE3] = None, 
            solution_pool: int = 4) -> None:
        raise NotImplementedError(self.__class__.__name__ + ' cannot set ee pose for gripper') 

    def get_ee_pose(self) -> SE3:
        raise NotImplementedError(self.__class__.__name__ + ' cannot set ee pose for gripper') 