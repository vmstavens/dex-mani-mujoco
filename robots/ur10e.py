import mujoco as mj
import roboticstoolbox as rtb
import spatialmath as sm
import math as m
import numpy as np
import os
import warnings
import sys

from robots.base_robot import BaseRobot

from typing import List, Union, Dict

from spatialmath import SE3

from utils.mj import (
    get_actuator_names,
    get_joint_names,
    get_joint_value,
    get_actuator_value,
    set_actuator_value,
    is_done_actuator
)

from utils.rtb import (
    make_tf
)

from utils.sim import (
    read_config, 
    save_config,
    config_to_q,
    RobotConfig
)

class UR10e(BaseRobot):
    def __init__(self, model: mj.MjModel, data: mj.MjData, args) -> None:
        super().__init__()

        self._args = args

        self._robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d = 0.1807, alpha = m.pi / 2.0  , qlim=(-m.pi,m.pi)), # J1
                rtb.RevoluteDH(a = -0.6127                     , qlim=(-m.pi,m.pi)), # J2
                rtb.RevoluteDH(a = -0.57155                    , qlim=(-m.pi,m.pi)), # J3
                rtb.RevoluteDH(d = 0.17415, alpha =  m.pi / 2.0, qlim=(-m.pi,m.pi)), # J4
                rtb.RevoluteDH(d = 0.11985, alpha = -m.pi / 2.0, qlim=(-m.pi,m.pi)), # J5
                rtb.RevoluteDH(d = 0.11655                     , qlim=(-m.pi,m.pi)), # J6
            ], name=self.name, base=SE3.Rz(m.pi)                                     # base transform due to fkd ur standard
        )

        self._model = model
        self._data = data
        self._actuator_names = self._get_actuator_names()
        self._joint_names    = self._get_joint_names()
        self._config_dir     = self._get_config_dir()
        self._configs        = self._get_configs()

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
        return "ur10e"

    def _set_robot_handle(self, robot_handle):
        self._robot_handle = robot_handle

    def _config_to_q(self, config: str) -> List[float]:
        return config_to_q(
                    cfg = config, 
                    configs = self._configs, 
                    actuator_names = self._actuator_names
                )

    def set_q(self, q : Union[str, List, RobotConfig]):
        if isinstance(q, str):
            q: List[float] = self._config_to_q(config=q)
        if isinstance(q, RobotConfig):
            q: List[float] = q.joint_values
        assert len(q) == self.n_actuators, f"Length of q should be {self.n_actuators}, q had length {len(q)}"

        qf = self._robot_handle._traj[-1].copy()

        qf[:self.n_actuators] = self._clamp_q(q)

        self._robot_handle._traj.extend([qf])

    def set_ee_pose(self, 
            pos: List = [0.5,0.5,0.5], 
            ori: Union[np.ndarray,SE3] = [1,0,0,0], 
            pose: Union[None, List[float], np.ndarray, SE3] = None,
            solution_pool:int = 4,
            ) -> None:

        if pose is not None:
            if isinstance(pose, SE3):
                target_pose = pose
            else:
                # Assuming pose is a list or numpy array [x, y, z, qw, qx, qy, qz]
                target_pose = SE3(pose[:3], pose[3:])
        else:
            # Use the provided position and orientation
            target_pose = make_tf(pos=pos, ori=ori)

        q_sols = []
        for _ in range(solution_pool):
            q_sol, success, iterations, searches, residual = self._robot.ik_NR(Tep=target_pose)
            q_sols.append( (q_sol, success, iterations, searches, residual) )
        
        if not np.any([ x[1] for x in q_sols ]):
            raise ValueError(f"Inverse kinematics failed to find a solution to ee pose. Tried {solution_pool} times. [INFO]: \n\t{q_sol=}\n\t{success=}\n\t{iterations=}\n\t{searches=}\n\t{residual=}")
        
        d = sys.maxsize

        q0 = self._robot_handle._traj[-1].copy()[:self.n_actuators]
        qf = self._robot_handle._traj[-1].copy()

        for i in range(solution_pool):
            q_diff = q_sols[i][0] - q0
            q_dist = np.linalg.norm( q_diff )
            if q_dist < d:
                qf[:self.n_actuators] = self._clamp_q(q_sols[i][0])
                d = q_dist

        self._robot_handle._traj.extend([ qf ])

    def get_ee_pose(self) -> SE3:
        return SE3(self._robot.fkine(self.get_q().joint_values))