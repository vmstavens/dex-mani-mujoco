from utils.rtb import make_tf
from .shadow_hand import ShadowHand
from .ur10e import UR10e
from typing import List, Union, Dict, Tuple
import os 
import warnings
import roboticstoolbox as rtb
import mujoco as mj
import math as m
import spatialmath as sm
import numpy as np
import json
import sys
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
    get_joint_actuator_diff,
    get_joint_range,
    get_actuator_range,
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

    def _get_actuator_values(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self.mj_model):
            if self.name in ac_name:
                result.append( get_actuator_value(self.mj_data, ac_name) )
        return result

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

    def _get_joint_limits(self, joint_name:str) -> List[float]:
        return get_joint_range(self.mj_model, joint_name=joint_name)

    def _get_actuator_limits(self, actuator_name) -> List[float]:
        return get_actuator_range(self.mj_model, actuator_name=actuator_name)

    def _clamp_q(self, q: List[float]) -> List[float]:
        actuator_names = self._get_actuator_names()
        actuator_limits = [get_actuator_range(self.mj_model, jn) for jn in actuator_names]
        clamped_qs = []
        for i in range(len(q)):
            clamped_q = np.clip(a = q[i], a_min = actuator_limits[i][0], a_max = actuator_limits[i][1])
            clamped_qs.append(clamped_q)

        # for i in range(len(q)):
        #     print(f"\t{actuator_names[i]} {actuator_limits[i]}")

        return clamped_qs

    def _are_done_actuators(self) -> bool:
        actuator_names = self._get_actuator_names()
        joint_names    = self._get_joint_names()
        for i,jn in enumerate(joint_names):
            if not is_done_actuator(data = self.mj_data, joint_name = jn, actuator_name = actuator_names[i]):
                return False
        return True

    def get_q(self) -> RobotConfig:
        robot_joint_names = []
        robot_joint_values = []
        for an in get_joint_names(self.mj_model):
            if an.split("_")[0] == self.name.split("_")[0]:
                robot_joint_names.append(an)
        for ran in robot_joint_names:
            robot_joint_values.append(get_joint_value(self.mj_data, ran))
        rc = RobotConfig(
            joint_values = robot_joint_values,
            joint_names = robot_joint_names
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
        self._has_arm        = True if arm     is not None else False
        self._joint_names    = self._get_joint_names()
        self._actuator_names = self._get_actuator_names()
        self._configs        = self._get_configs()
        self._traj           = [self.get_q().joint_values]

        # to also give access to high level arm.set_q and gripper.set_q
        if self._has_gripper:
            self._gripper._set_robot_handle(self)
        if self._has_arm:
            self._arm._set_robot_handle(self)

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

    def _get_joint_names(self) -> List[str]:
        return get_joint_names(self.mj_model)

    def _clamp_q(self, q: List[float]) -> List[float]:
        actuator_names = self._get_actuator_names()
        actuator_limits = [get_actuator_range(self.mj_model,jn) for jn in actuator_names]
        clamped_qs = []
        for i in range(len(actuator_names)):
            clamped_q = np.clip(a = q[i], a_min = actuator_limits[i][0], a_max = actuator_limits[i][1])
            clamped_qs.append(clamped_q)
        return clamped_qs

    def _get_actuator_values(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self.mj_model):
            if self.arm.name in ac_name or self.gripper.name in ac_name:
                result.append( get_actuator_value(self.mj_data, ac_name))
        return result

    def _get_actuator_names(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self.mj_model):
            if self.arm.name in ac_name or self.gripper.name in ac_name:
                result.append(ac_name)
        return result

    def _config_to_q(self, config: str) -> List[float]:
        q = config_to_q(
                    cfg = config, 
                    configs = self._configs, 
                    actuator_names = self._actuator_names
                )
        return q

    def _set_q(self, q: Union[str,List]) -> None:
        robot_actuator_names = []
        actuator_names = get_actuator_names(self._model)
        for an in actuator_names:
            if self.arm.name in an or self.gripper:
                robot_actuator_names.append(an)
        for i in range(len(robot_actuator_names)):
            set_actuator_value(data=self._data, q=q[i], actuator_name=robot_actuator_names[i])

    def get_q(self) -> RobotConfig:
        robot_joint_names = []
        robot_joint_values = []
        for an in get_joint_names(self.mj_model):
            prefix = an.split("_")[0]
            if self.arm.name in prefix or self.gripper.name:
                robot_joint_names.append(an)
        for ran in robot_joint_names:
            robot_joint_values.append(get_joint_value(self.mj_data, ran))
        rc = RobotConfig(
            joint_values = robot_joint_values,
            joint_names = robot_joint_names
        )
        return rc

    def set_q(self, q_robot: Union[str, List, RobotConfig] = None, q_arm: Union[str, List, RobotConfig] = None, q_gripper: Union[str, List, RobotConfig] = None) -> None:

        if q_robot is None and q_arm is None and q_gripper is None:
            warnings.warn(f"No q value provided to set_q(), returning...")
            return

        if q_robot is not None:
            if q_arm is not None:
                warnings.warn(f"A value was set for q, along with one for q_arm, q_arm is being ignored")
            if q_gripper is not None:
                warnings.warn(f"A value was set for q, along with one for q_gripper, q_gripper is being ignored")

        if self.is_done:
            qf = self._get_actuator_values()
        else:
            qf = self._traj[-1].copy()

        if q_arm is not None:
            if isinstance(q_arm, str):
                q_arm: List[float] = self.arm._config_to_q(config=q_arm)
            if isinstance(q_arm, RobotConfig):
                q_arm: List[float] = q_arm.joint_values
            assert len(q_arm) == self.arm.n_actuators, f"Length of q_arm should be {self.arm.n_actuators}, q_arm had length {len(q_arm)}"
            qf[:self.arm.n_actuators] = self.arm._clamp_q(q_arm)

        if q_gripper is not None:
            if isinstance(q_gripper, str):
                q_gripper: List[float] = self.gripper._config_to_q(config=q_gripper)
            if isinstance(q_gripper, RobotConfig):
                q_gripper: List[float] = q_gripper.joint_values
            assert len(q_gripper) == self.gripper.n_actuators, f"Length of q_gripper should be {self.gripper.n_actuators}, q_gripper had length {len(q_gripper)}"
            qf[-self.gripper.n_actuators:] = self.gripper._clamp_q(q_gripper)

        if q_robot is not None:
            if isinstance(q_robot, str):
                q_robot: List[float] = self._config_to_q(config=q_robot)
            if isinstance(q_robot, RobotConfig):
                q_robot: List[float] = q_robot.joint_values
            assert len(q_robot) == self.n_actuators, f"Length of q_robot should be {self.n_actuators}, q_robot had length {len(q_robot)}"
            qf = self._clamp_q(q_robot)

        self._traj.extend([ qf ])

    def step(self) -> None:
        self.gripper._are_done_actuators()
        if self.arm._are_done_actuators() and self.gripper._are_done_actuators():
            # to ensure an element always exsists in _traj. otherwise 
            # gripper and arm cant be controlled independently
            if len(self._traj) > 1:
                self._traj.pop(0)
            self._set_q(self._traj[0])

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

        q0 = self._traj[-1]

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
        return SE3(self._robot.fkine(self.get_q().joint_values))

class ShadowHand(BaseRobot):
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
        return self._args.sh_chirality

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

