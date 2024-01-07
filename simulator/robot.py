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
    get_joint_names,
    get_joint_value,
    is_done_actuator
)

from abc import ABC, abstractmethod, abstractproperty


class BaseRobot(ABC):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def args(self):
        pass

    @property
    def n_actuators(self) -> int:
        return len(self._get_actuator_names())

    @abstractproperty
    def mj_data(self) -> mj.MjData:
        pass

    @abstractproperty
    def mj_model(self) -> mj.MjModel:
        pass

    @abstractmethod
    def _config_to_q(self, config: str) -> List[float]:
        pass

    def set_q(self) -> None:
        pass

    def _set_q(self, q: Union[str,List]) -> None:
        robot_actuator_names = []
        for an in get_actuator_names(self.mj_model):
            if self.name in an:
                robot_actuator_names.append(an)
        for i in range(len(robot_actuator_names)):
            set_actuator_value(data=self.mj_data, q=q[i], actuator_name=robot_actuator_names[i])

    def home(self) -> None:
        self.set_q(q_robot = "home")

    def _get_actuator_names(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self.mj_model):
            if self.name in ac_name:
                result.append(ac_name)
        return result

    def _get_joint_names(self) -> List[str]:
        result = []
        for jt_name in get_joint_names(self.mj_model):
            if self.name in jt_name:
                result.append(jt_name)
        return result

    def are_done_actuators(self) -> bool:
        actuator_names = get_actuator_names(self.mj_model)
        joint_names = get_joint_names(self.mj_model)
        for i,jn in enumerate(joint_names):
            if not is_done_actuator(data = self.mj_data, joint_name = jn, actuator_name = actuator_names[i]):
                return False
        return True

    def get_q(self) -> RobotConfig:
        robot_actuator_names = []
        robot_actuator_values = []
        for an in get_actuator_names(self.mj_model):
            if an.split("_")[0] == self.name.split("_")[0]:
                robot_actuator_names.append(an)
        for ran in robot_actuator_names:
            robot_actuator_values.append(get_actuator_value(self.mj_data, ran))
        rc = RobotConfig(
            actuator_values = robot_actuator_values,
            actuator_names = robot_actuator_names
        )
        return rc

    def save_config(self, config_name:str = "placeholder") -> None:
        save_config(
            config_dir  = self.config_dir,
            config      = self.get_q(),
            config_name = config_name
        )

    def _get_config_dir(self) -> str:
        return self.args.config_dir + self.name + ".json"

    def _get_configs(self) -> Dict[str, Dict[str, float]]:
        if not os.path.exists(self.args.config_dir):
            os.makedirs(os.path.dirname(self.args.config_dir), exist_ok=True)
            warnings.warn(f"config_dir {self.args.config_dir} could not be found, create empty config")
        return read_config(self._get_config_dir())

class Robot(BaseRobot):
    def __init__(self, args, arm: BaseRobot = None, gripper: BaseRobot = None) -> None:
        super().__init__() 
        self._args           = args
        self._model          = arm.mj_model
        self._data           = arm.mj_data
        self._arm            = arm
        self._gripper        = gripper
        self._has_gripper    = True if gripper is not None else False
        self._arm_gripper    = True if arm     is not None else False
        self._joint_names    = self._get_joint_names()
        self._actuator_names = self._get_actuator_names()
        self._configs        = self._get_configs()
        self._traj           = [self.get_q().actuator_values]

    @property
    def arm(self) -> BaseRobot:
        return self._arm
    
    @property
    def gripper(self)-> BaseRobot:
        return self._gripper

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
        return self._arm.name if not self._has_gripper else self._arm.name + "_" + self._gripper.name

    @property
    def is_done(self) -> bool:
        return len(self._traj) == 0

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def n_actuators(self) -> int:
        return self._arm.n_actuators if not self._has_gripper else self._gripper.n_actuators + self._arm.n_actuators

    def _config_to_q(self, config: str) -> List[float]:
        return config_to_q(
                    cfg = config, 
                    configs = self._configs, 
                    actuator_names = self._actuator_names
                )

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
            if self.arm.name in an or self.gripper:
                robot_actuator_names.append(an)
        for i in range(len(robot_actuator_names)):
            set_actuator_value(data=self._data, q=q[i], actuator_name=robot_actuator_names[i])

    def set_q(self, q_robot: Union[str, List, RobotConfig] = None, q_arm: Union[str, List, RobotConfig] = None, q_gripper: Union[str, List, RobotConfig] = None, n_steps: int = 2) -> None:

        if q_robot is None and q_arm is None and q_gripper is None:
            warnings.warn(f"No q value provided to set_q(), returning...")
            return

        if q_robot is not None:
            if q_arm is not None:
                warnings.warn(f"A value was set for q, along with one for q_arm, q_arm is being ignored")
            if q_gripper is not None:
                warnings.warn(f"A value was set for q, along with one for q_gripper, q_gripper is being ignored")

        if self.is_done:
            qf = self.get_q().actuator_values
        else:
            qf = self._traj[-1].copy()

        if q_arm is not None:
            if isinstance(q_arm, str):
                q_arm: List[float] = self.arm._config_to_q(config=q_arm)
            if isinstance(q_arm, RobotConfig):
                q_arm: List[float] = q_arm.actuator_values
            assert len(q_arm) == self.arm.n_actuators, f"Length of q_arm should be {self.arm.n_actuators}, q_arm had length {len(q_arm)}"
            qf[:self.arm.n_actuators] = q_arm

        if q_gripper is not None:
            if isinstance(q_gripper, str):
                q_gripper: List[float] = self.gripper._config_to_q(config=q_gripper)
            if isinstance(q_gripper, RobotConfig):
                q_gripper: List[float] = q_gripper.actuator_values
            assert len(q_gripper) == self.gripper.n_actuators, f"Length of q_gripper should be {self.gripper.n_actuators}, q_gripper had length {len(q_gripper)}"
            qf[self.arm.n_actuators:] = q_gripper

        if q_robot is not None:
            if isinstance(q_robot, str):
                q_robot: List[float] = self._config_to_q(config=q_robot)
            if isinstance(q_robot, RobotConfig):
                q_robot: List[float] = q_robot.actuator_values
            assert len(q_robot) == self.n_actuators, f"Length of q_robot should be {self.n_actuators}, q_robot had length {len(q_robot)}"
            qf = q_robot

        self._traj.extend([qf])

    def step(self) -> None:
        self._set_q(self._traj[0])
        if not self.are_done_actuators():
            return
        self._set_q(self._traj.pop(0))

class UR10e(BaseRobot):
    def __init__(self, model: mj.MjModel, data: mj.MjData, args) -> None:
        super().__init__()
        
        self._args = args
        
        self._robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d = 0.1807, alpha = m.pi / 2.0),   # J1
                rtb.RevoluteDH(a = -0.6127),                      # J2
                rtb.RevoluteDH(a = -0.57155),                     # J3
                rtb.RevoluteDH(d = 0.17415, alpha =  m.pi / 2.0), # J4
                rtb.RevoluteDH(d = 0.11985, alpha = -m.pi / 2.0), # J5
                rtb.RevoluteDH(d = 0.11655),                      # J6
            ], name=self.name, base=SE3.Rz(m.pi)                  # base transform due to fkd ur standard
        )

        self._model = model
        self._data = data
        self._actuator_names = self._get_actuator_names()
        self._joint_names    = self._get_joint_names()
        self._config_dir     = self._get_config_dir()
        self._configs        = self._get_configs()

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

    @property
    def rtb_robot(self) -> rtb.DHRobot:
        return self._robot

    def _config_to_q(self, config: str) -> List[float]:
        return config_to_q(
                    cfg = config, 
                    configs = self._configs, 
                    actuator_names = self._actuator_names
                )

    def set_q():
        pass
class ShadowHand(BaseRobot):
    def __init__():

    # def set_q(self, q: Union[str,List], n_steps: int = 10) -> None:
    #     if isinstance(q,str):
    #         q:list = self._config_to_q(config=q)
    #     assert len(q) == self.n_actuators, f"Length of q should be {self.n_actuators}, q had length {len(q)}"
        
    #     qf = np.array(q)
        
    #     if self.is_done:
    #         q0 = np.array(self.get_q().actuator_values)
    #     else:
    #         q0 = self._traj[-1]

    #     new_traj = rtb.jtraj(
    #         q0 = q0,
    #         qf = qf,
    #         t = n_steps
    #     ).q.tolist()
    #     self._traj.extend(new_traj)

    # def set_ee_pose(self, 
    #         pos: List = [0.5,0.5,0.5], 
    #         ori: Union[np.ndarray,SE3] = [1,0,0,0], 
    #         pose: Union[None, List[float], np.ndarray, SE3] = None,
    #         solution_pool:int = 4,
    #         n_steps:int = 2
    #         ) -> None:

    #     if pose is not None:
    #         if isinstance(pose, SE3):
    #             target_pose = pose
    #         else:
    #             # Assuming pose is a list or numpy array [x, y, z, qw, qx, qy, qz]
    #             target_pose = SE3(pose[:3], pose[3:])
    #     else:
    #         # Use the provided position and orientation
    #         target_pose = make_tf(pos=pos, ori=ori)

    #     q_sols = []
    #     for _ in range(solution_pool):
    #         q_sol, success, iterations, searches, residual = self._robot.ik_NR(Tep=target_pose)
    #         q_sols.append( (q_sol, success, iterations, searches, residual) )
        
    #     if not np.any([ x[1] for x in q_sols ]):
    #             raise ValueError(f"Inverse kinematics failed to find a solution to ee pose. Tried {solution_pool} times. [INFO]: \n\t{q_sol=}\n\t{success=}\n\t{iterations=}\n\t{searches=}\n\t{residual=}")
        
    #     d = sys.maxsize

    #     q0 = self.get_q().actuator_values if self.is_done else self._traj[-1]

    #     for i in range(solution_pool):
    #         q_diff = q_sols[i][0] - q0
    #         if np.linalg.norm( q_diff ) < d:
    #             qf = q_sols[i][0]
    #             d = np.linalg.norm( q_diff )

    #     new_traj = rtb.jtraj(
    #         q0 = q0,
    #         qf = qf,
    #         t = n_steps
    #     ).q.tolist()

    #     if self.is_done:
    #         self._traj = new_traj
    #     else:
    #         self._traj.extend(new_traj)

    # def get_ee_pose(self) -> SE3:
    #     return SE3(self._robot.fkine(self.get_q().actuator_values))
