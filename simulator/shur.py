from utils.rtb import make_tf
from .shadow_hand import ShadowHand
from .ur10e import UR10e
from .robot import BaseRobot
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

class Robot(BaseRobot):
    def __init__(self, model: mj.MjModel, data: mj.MjData, args) -> None:
        self._args = args
        self._model = model
        self._data = data
        self._name = "robot"
        self._traj = []
        self._arm       = UR10e(model, data, args)
        self._gripper   = ShadowHand(model, data, args)
        self._joint_names = self._get_joint_names()
        self._actuator_names = self._get_actuator_names()

        UR_EE_TO_SH_WRIST_JOINTS = 0.21268             # m
        SH_WRIST_TO_SH_PALM      = 0.08721395775941231 # m

        self._N_ACTUATORS:int = self.gripper.n_actuators + self.arm.n_actuators
        self._config_dir = args.config_dir + self._name + ".json"
        if not os.path.exists(self._config_dir):
            os.makedirs(os.path.dirname(self._config_dir), exist_ok=True)
            warnings.warn(f"config_dir {self._config_dir} could not be found, create empty config")
        self._configs = read_config(self._config_dir)

    @property
    def gripper(self) -> ShadowHand:
        return self._gripper

    @property
    def arm(self) -> UR10e:
        return self._arm

    @property
    def is_done(self) -> bool:
        return self._is_done()

    def _is_done(self) -> bool:
        return len(self._traj) == 0

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def n_actuators(self) -> int:
        return self._N_ACTUATORS

    def _get_actuator_names(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self._model):
            if self.arm._name in ac_name or self.gripper._chirality in ac_name:
                result.append(ac_name)
        return result

    def _get_joint_names(self) -> List[str]:
        result = []
        for ac_name in get_joint_names(self._model):
            if self.arm._name in ac_name or self.gripper._chirality in ac_name:
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

    def _get_arm_q_end(self) -> List[float]:
        if self.gripper._are_done_actuators():
            return self.get_q().actuator_values[:self.arm.n_actuators]
        return self._traj[-1][:self.arm.n_actuators]

    def _get_gripper_q_end(self) -> List[float]:
        if self._are_done_actuators():
            return self.get_q().actuator_values
        return self._traj[-1][self.arm.n_actuators:]

    def set_q_arm(self, q: Union[str, List, RobotConfig], n_steps: int = 100) -> None:
        if isinstance(q, str):
            q: List = config_to_q(q, self.arm._configs, self.arm._actuator_names)
        if isinstance(q, RobotConfig):
            q: List = q.actuator_values
        assert len(q) == self.arm.n_actuators, f"Length of q should be {self.arm.n_actuators}, q had length {len(q)}"

        qf = np.array(q)
        q0 = np.array(self._get_arm_q_end())
        print(f"{qf=}")
        print(f"{q0=}")
        arm_q_traj = rtb.jtraj(
            q0 = q0,
            qf = qf,
            t = n_steps
        ).q.tolist()

        const_gripper_q_traj = [self._get_gripper_q_end() for _ in range(len(arm_q_traj))]

        self._traj.extend( np.concatenate( (arm_q_traj, const_gripper_q_traj), axis=1 ).tolist() )

    def set_q_gripper(self, q: Union[str, List, RobotConfig], n_steps: int = 100) -> None:
        pass
        # if isinstance(q, str):
        #     q: List = config_to_q(q, self.gripper._configs, self.gripper._actuator_names)
        # if isinstance(q, RobotConfig):
        #     q: List = q.actuator_values
        # assert len(q) == self.gripper.n_actuators, f"Length of q should be {self.gripper.n_actuators}, q had length {len(q)}"
        
        # if self.gripper.is_done:
        #     q0 = np.array(self.gripper.get_q().actuator_values)
        # else:
        #     q0 = self._traj[-1]
        # qf = np.array(q)
        
        # gripper_q_traj = rtb.jtraj(
        #     q0 = q0,
        #     qf = qf,
        #     t = n_steps
        # ).q.tolist()

        # arm_q      = self._traj[-1][:self.arm.n_actuators]
        # arm_q_traj = [ arm_q for _ in range(len(gripper_q_traj))]

        # print(arm_q_traj)
        # print(gripper_q_traj[-1])
        # arm_gripper_q_traj = np.concatenate((arm_q_traj, gripper_q_traj), axis=1).tolist()
        # for x in range(len(arm_gripper_q_traj)):
        #     print(arm_gripper_q_traj[x][:6])

        # self._traj.extend(arm_gripper_q_traj)

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

    def set_q(self, q: Union[str, List, RobotConfig], n_steps: int = 50) -> None:
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
            print(q)
        assert len(q) == self._N_ACTUATORS, f"Length of q should be {self._N_ACTUATORS}, q had length {len(q)}"
        
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
        # q0 = np.array(self.get_q().actuator_values)
        # qf = np.array(q)
        

        # self._traj = rtb.jtraj(
        #     q0 = q0,
        #     qf = qf,
        #     t = n_steps
        # ).q.tolist()

    def set_q_ur(self,q: Union[str, List, RobotConfig], n_steps: int = 10):

        if isinstance(q,str):
            q:list = self.arm._cfg_to_q(q)
        assert len(q) == self.arm.n_actuators, f"Length of q should be {self.arm.n_actuators}, q had length {len(q)}"
        
        if self.is_done:
            q0 = np.array(self.arm.get_q().actuator_values)
        else:
            q0 = self.arm._traj[-1]
        qf = np.array(q)

        new_traj = rtb.jtraj(
            q0 = q0,
            qf = qf,
            t = n_steps
        ).q.tolist()

        self._traj.extend(new_traj)

    def set_q_sh(self,q: Union[str, List, RobotConfig], n_steps: int = 10):
        pass

    def home(self) -> None:
        self.set_q(q = "home")

    def _are_done_actuators(self) -> bool:
        for i,jn in enumerate(self._joint_names):
            if not is_done_actuator(self._data,joint_name=jn,actuator_name=self._actuator_names[i]):
                return False
        return True

    def step(self) -> None:
        if not self.is_done:
            self._set_q(self._traj[0])
        
        if not self._are_done_actuators():
            return
        
        self._set_q(self._traj.pop(0))

        # if not self.ur10e.is_done:
        #     self.ur10e.step()
        # if not self.shadow_hand.is_done:
        #     self.shadow_hand.step()

    def save_config(self, config_name:str = "placeholder") -> None:
        save_config(
            config_dir  = self._config_dir,
            config      = self.get_q(),
            config_name = config_name
        )