import mujoco as mj
import roboticstoolbox as rtb
import spatialmath as sm
import math as m
import numpy as np

from typing import List, Union

from spatialmath import SE3

from utils.mj import (
    get_actuator_names,
    get_joint_value,
    set_joint_value
)

from utils.rtb import (
    make_tf
)

from utils.sim import (
    read_config, 
    RobotConfig
)

class UR10e:
    def __init__(self, model: mj.MjModel, data: mj.MjData, args, config_dir:str = "config/ur10e.json") -> None:
        
        self._args = args
        
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
        self._N_ACTUATORS:int = 6
        self._traj = []
        self._config_dir = config_dir
        self._configs = read_config(self._config_dir)
        self._name = "ur10e"

    @property
    def rtb_robot(self) -> rtb.DHRobot:
        return self._arm

    @property
    def n_actuators(self) -> int:
        return self._N_ACTUATORS

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
        assert len(q) == self._N_ACTUATORS, f"Length of q should be {self._N_ACTUATORS}, q had length {len(q)}"
        
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

        cartesian_traj = rtb.ctraj(
            T0=self.get_ee_pose(),
            T1=make_tf(pos=pos,ori=ori),
            t=n_steps
        )

        for target in cartesian_traj:
            q_sol, success, iterations, searches, residual = self._robot.ik_NR(Tep=target)
            if not success:
                raise ValueError("Inverse kinematics failed to find a solution.")
            self._traj.append(q_sol)

    def get_ee_pose(self) -> SE3:
        return self._robot.fkine(self.get_q().joint_values)

    def step(self) -> None:
        if not self.is_done:
            self._set_q(self._traj.pop(0))

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