import roboticstoolbox as rtb
import math as m
import spatialmath as sm
import numpy as np
import mujoco as mj
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple

from spatialmath import (
    SE3
)

from utils.mj import (
    get_actuator_names,
    get_joint_value,
    set_joint_value,
)

from utils.rtb import (
    make_tf
)

from utils.control import (
    read_config
)

@dataclass
class RobotConfig:
    def __init__(self,joint_names, joint_values) -> None:
        self._joint_values = joint_values
        self._joint_names = joint_names
    @property
    def joint_values(self) -> List:
        return self._joint_values
    @property
    def joint_names(self) -> List:
        return self._joint_names
    @property
    def dict(self) -> Dict[str,List]:
        result = {}
        for i in range(len(self._joint_values)):
            result[self._joint_names[i]] = self._joint_values[i]
        return result
    def __repr__(self) -> str:
        return self.dict.__str__()

class UR10e:
    def __init__(self, model: mj.MjModel, data: mj.MjData, config_dir:str = "config/ur10e.json") -> None:
        
        self._robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d = 0.1807, alpha = m.pi / 2.0),   # J1
                rtb.RevoluteDH(a = -0.6127),                      # J2
                rtb.RevoluteDH(a = -0.57155),                     # J3
                rtb.RevoluteDH(d = 0.17415, alpha =  m.pi / 2.0), # J4
                rtb.RevoluteDH(d = 0.11985, alpha = -m.pi / 2.0), # J5
                rtb.RevoluteDH(d = 0.11655),                      # J6
            ], name="ur10e", base=sm.SE3.Trans(0,0,0)
        )
        self._HOME   = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self._model = model
        self._data = data
        self.__N_ACTUATORS:int = 6
        self._traj = []
        self._config_dir = config_dir
        self._configs = read_config(self._config_dir)

    @property
    def rtb_robot(self) -> rtb.DHRobot:
        return self._arm

    @property
    def n_actuators(self) -> int:
        return self.__N_ACTUATORS

    @property
    def is_done(self) -> bool:
        return self._is_done()
    
    def _is_done(self) -> bool:
        return True if len(self._traj) == 0 else False

    def get_q(self) -> RobotConfig:
        """
        Get the configuration of the arm's actuators in the MuJoCo simulation.

        Returns:
        - RobotConfig: An object containing joint values and names for the arm actuators.
        """
        arm_actuator_names = []
        arm_actuator_values = []
        for an in get_actuator_names(self._model):
            prefix = an.split("_")[0]
            if prefix == "ur10e":
                arm_actuator_names.append(an)
        for han in arm_actuator_names:
            arm_actuator_values.append(get_joint_value(self._data, han))
        ac = RobotConfig(
            joint_values = arm_actuator_values,
            joint_names = arm_actuator_names
        )
        return ac

    def __set_q(self, q: Union[str,List]) -> None:
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
        for i, han in enumerate(arm_actuator_names):
            set_joint_value(data=self._data, q=q[i], joint_name=han)

    def set_q(self, q: Union[str,List], n_steps: int = 10) -> None:
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
            q:list = self.cfg_to_q(q)
        print(f"{q=}")
        assert len(q) == self.__N_ACTUATORS, f"Length of q should be {self.__N_ACTUATORS}, q had length {len(q)}"
        
        q0 = np.array(self.get_q().joint_values)
        qf = np.array(q)

        self._traj = rtb.jtraj(
            q0 = q0,
            qf = qf,
            t = n_steps
        ).q.tolist()

    def set_ee_pose(self, 
            pos: List = [0.5,0.5,0.5], 
            ori: Union[np.ndarray,SE3] = [1,0,0,0], 
            n_steps:int = 100
            ) -> None:
        print("start")
        print(self.get_ee_pose())
        print("end")
        print(make_tf(pos=pos,ori=ori))
        # print("n_steps =",n_steps)
        # self._traj = rtb.ctraj(
        cartesian_traj = rtb.ctraj(
            T0=self.get_ee_pose(),
            T1=make_tf(pos=pos,ori=ori),
            t=n_steps
        )
        print(cartesian_traj)
        for target in cartesian_traj:
            q_sol, success, iterations, searches, residual = self._robot.ik_NR(Tep=target)
            if not success:
                raise ValueError("Inverse kinematics failed to find a solution.")
            self._traj.append(q_sol)

    def get_ee_pose(self) -> SE3:
        return self._robot.fkine(self.get_q().joint_values)

    def step(self) -> None:
        if not self.is_done:
            self.__set_q(self._traj.pop(0))

    def cfg_to_q(self, cfg:str) -> List:
        try:
            cfg_json = self._configs[cfg]
            q = [
                    cfg_json["shoulder_pan"],
                    cfg_json["shoulder_liff"],
                    cfg_json["shoulder_elbow"],
                    cfg_json["wrist_1"],
                    cfg_json["wrist_2"],
                    cfg_json["wrist_3"]
                ]
            return q
        except KeyError:
            print("Wrong cfg string, try one of the following:")
            for k,v in self._configs.items():
                print(f"\t{k}")

    def home(self) -> None:
        self.set_q(self._HOME)

class ShadowHand:
    def __init__(self, model: mj.MjModel, data: mj.MjData, config_dir:str = "config/shadow_hand.json") -> None:
        self._model = model
        self._data = data
        self._N_ACTUATORS:int = 20
        self._traj = []
        self._HOME = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._config_dir = config_dir
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

    def __set_q(self, q: Union[str,List]) -> None:
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
        self.__set_q(self._traj.pop(0))

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

class SHUR:
    def __init__(self, model: mj.MjModel, data: mj.MjData) -> None:
        
        self._model = model
        self._data = data
        self._traj = []

        UR_EE_TO_SH_WRIST_JOINTS = 0.21268 # m
        SH_WRIST_TO_SH_PALM      = 0.08721395775941231 # m
        
        self._ur10e = UR10e(model, data)
        self._shadow_hand = ShadowHand(model, data)
        self._robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d = 0.1807, alpha = m.pi / 2.0),        # J1
                rtb.RevoluteDH(a = -0.6127),                         # J2
                rtb.RevoluteDH(a = -0.57155),                        # J3
                rtb.RevoluteDH(d = 0.17415, alpha =  m.pi / 2.0),      # J4
                rtb.RevoluteDH(d = 0.11985, alpha = -m.pi / 2.0),      # J5
                rtb.RevoluteDH(d = 0.11655 + UR_EE_TO_SH_WRIST_JOINTS), # J6 + forearm
                rtb.RevoluteDH(alpha = m.pi / 2),                      # WR1
                rtb.RevoluteDH(alpha = m.pi / 2, offset= m.pi / 2),      # WR2
                rtb.RevoluteDH(d = SH_WRIST_TO_SH_PALM),             # from wrist to palm
            ], name="shur", base=sm.SE3.Trans(0,0,0)
        )

    @property
    def shadow_hand(self) -> ShadowHand:
        return self._shadow_hand

    @property
    def ur10e(self) -> UR10e:
        return self._ur10e
    @property
    def is_done(self) -> bool:
        return self._is_done()

    def _is_done(self) -> bool:
        return True if (self.shadow_hand.is_done and self.ur10e.is_done) else False

    def get_next_control(self) -> List:
        return self._ur10e.get_next_control() + self._shadow_hand.get_next_control()

    def home(self) -> None:
        self.shadow_hand.home()
        self.ur10e.home()
    
    def step(self) -> None:
        if not self.ur10e.is_done:
            self.ur10e.step()
        if not self.shadow_hand.is_done:
            self.shadow_hand.step()