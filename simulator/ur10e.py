import mujoco as mj
import roboticstoolbox as rtb
import spatialmath as sm
import math as m
import numpy as np
import os
import warnings
import sys

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

class UR10e:
    def __init__(self, model: mj.MjModel, data: mj.MjData, args) -> None:
        
        self._args = args
        self._name = "ur10e"
        
        self._robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d = 0.1807, alpha = m.pi / 2.0),   # J1
                rtb.RevoluteDH(a = -0.6127),                      # J2
                rtb.RevoluteDH(a = -0.57155),                     # J3
                rtb.RevoluteDH(d = 0.17415, alpha =  m.pi / 2.0), # J4
                rtb.RevoluteDH(d = 0.11985, alpha = -m.pi / 2.0), # J5
                rtb.RevoluteDH(d = 0.11655),                      # J6
            ], name=self._name, base=SE3.Rz(m.pi)                 # base transform due to fkd ur standard
        )

        self._model = model
        self._data = data
        self._N_ACTUATORS:int = 6
        self._actuator_names = self._get_actuator_names()
        self._joint_names    = self._get_joint_names()
        self._config_dir     = self._get_config_dir()
        self._configs        = read_config(self._config_dir)

    @property
    def rtb_robot(self) -> rtb.DHRobot:
        return self._robot

    @property
    def n_actuators(self) -> int:
        return self._N_ACTUATORS

    @property
    def is_done(self) -> bool:
        return self._is_done()
    
    def _is_done(self) -> bool:
        return True if len(self._traj) == 0 else False

    def _get_actuator_names(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self._model):
            if self._name in ac_name:
                result.append(ac_name)
        return result

    def _get_joint_names(self):
        result = []
        for jt_name in get_joint_names(self._model):
            if self._name in jt_name:
                result.append(jt_name)
        return result

    def _get_config_dir(self):
        self._config_dir = self._args.config_dir + self._name + ".json"
        if not os.path.exists(self._config_dir):
            os.makedirs(os.path.dirname(self._config_dir), exist_ok=True)
            warnings.warn(f"config_dir {self._config_dir} could not be found, create empty config")
        return self._config_dir

    def get_q(self) -> RobotConfig:
        """
        Get the configuration of the arm's actuators in the MuJoCo simulation.

        Returns:
        - RobotConfig: An object containing joint values and names for the arm actuators.
        """
        arm_actuator_names = []
        arm_actuator_values = []
        for an in get_actuator_names(self._model):
            if "ur10e" in an:
                arm_actuator_names.append(an)
        for han in arm_actuator_names:
            arm_actuator_values.append(get_actuator_value(self._data, han))
        ac = RobotConfig(
            actuator_values = arm_actuator_values,
            actuator_names = arm_actuator_names
        )
        return ac

    def _set_q(self, q: List[float]) -> None:
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
        arm_actuator_names = []
        for an in get_actuator_names(self._model):
            prefix = an.split("_")[0]
            if prefix == "ur10e":
                arm_actuator_names.append(an)

        for i, aan in enumerate(arm_actuator_names):
            # d = 1e-4
            # while abs(get_actuator_value(self._data, aan) - q[i] > d):
            set_actuator_value(data=self._data, q=q[i], actuator_name=aan)

    def set_q(self, q: Union[str,List], n_steps: int = 10) -> None:
        if isinstance(q,str):
            q:list = self._cfg_to_q(q)
        assert len(q) == self._N_ACTUATORS, f"Length of q should be {self._N_ACTUATORS}, q had length {len(q)}"
        
        qf = np.array(q)

        
        if self.is_done:
            q0 = np.array(self.get_q().actuator_values)
        else:
            q0 = self._traj[-1]
        qf = np.array(q)
        new_traj = rtb.jtraj(
            q0 = q0,
            qf = qf,
            t = n_steps
        ).q.tolist()
        self._traj.extend(new_traj)

    def set_ee_pose(self, 
            pos: List = [0.5,0.5,0.5], 
            ori: Union[np.ndarray,SE3] = [1,0,0,0], 
            pose: Union[None, List[float], np.ndarray, SE3] = None,
            solution_pool:int = 4,
            n_steps:int = 2
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

        q0 = self.get_q().actuator_values if self.is_done else self._traj[-1]

        for i in range(solution_pool):
            q_diff = q_sols[i][0] - q0
            if np.linalg.norm( q_diff ) < d:
                qf = q_sols[i][0]
                d = np.linalg.norm( q_diff )

        new_traj = rtb.jtraj(
            q0 = q0,
            qf = qf,
            t = n_steps
        ).q.tolist()

        if self.is_done:
            self._traj = new_traj
        else:
            self._traj.extend(new_traj)

    def get_ee_pose(self) -> SE3:
        return SE3(self._robot.fkine(self.get_q().actuator_values))

    def _are_done_actuators(self) -> bool:
        for i,jn in enumerate(self._joint_names):
            if not is_done_actuator(self._data,joint_name=jn,actuator_name=self._actuator_names[i]):
                return False
        return True

    def step(self) -> None:
        self._set_q(self._traj[0])
        if not self._are_done_actuators():
            return
        
        self._set_q(self._traj.pop(0))

    def _cfg_to_q(self, cfg:str) -> List:
        return config_to_q(
            cfg            = cfg, 
            configs        = self._configs, 
            actuator_names = self._actuator_names
        )

    def home(self) -> None:
        self.set_q(q = "home")

    def save_config(self, config_name:str = "placeholder") -> None:
        save_config(
            config_dir  = self._config_dir,
            config      = self.get_q(),
            config_name = config_name
        )