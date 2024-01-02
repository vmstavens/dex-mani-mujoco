import mujoco as mj
import mujoco.viewer
import mujoco
from mujoco.glfw import glfw
# from controllers.controller import Controller
# from controllers.expert import ExpertController
from controllers.hand_controller import HandController
from controllers.arm_controller import ArmController
from utils import control
import mujoco_py
from typing import Tuple, List, Optional, Union, Dict
import numpy as np
import time
import math as m
from math import pi
import sys
import pandas as pd
from threading import Thread, Lock
import roboticstoolbox as rtb
from utils.mj import (
    get_actuator_names,
    get_joint_value,
    set_joint_value
)
from spatialmath import (
    SE3, SO3, Quaternion, UnitQuaternion
    )
from scipy.spatial.transform import Slerp
from spatialmath.base import trnorm
from dataclasses import dataclass

import spatialmath as sm
import spatialmath.base as smb
from simulator.robots import SHUR
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'shadow_hand/utils')

from utils.rtb import (
    make_tf,
    quat_2_tf,
    tf_2_quat,
    pos_2_tf,
    get_pose,
    generate_pose_trajectory,
    rotate_x
)


from controllers.pose_controller import PoseController

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

class GLWFSim:
    def __init__(
        self,
        shadow_hand_xml_filepath: str,
        hand_controller: HandController,
        arm_controller: HandController,
        trajectory_steps: int,
        cam_verbose: bool,
        sim_verbose: bool
    ):

        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel
        self._model_file_path = shadow_hand_xml_filepath
        self._model = mj.MjModel.from_xml_path(filename=self._model_file_path)
        self._data = mj.MjData(self._model)
        self._camera = mj.MjvCamera()
        self._options = mj.MjvOption()

        self.N_ACTUATORS_HAND = 20
        self.N_ACTUATORS_ARM = 6
        self.N_ACTUATORS_ROBOT = self.N_ACTUATORS_HAND + self.N_ACTUATORS_ARM

        self._keyboard_pos_step = 0.05
        self.dt = 1.0 / 100.0

        # self._window = mujoco.viewer.launch_passive(self._model, self._data)
        self._window = mujoco.viewer.launch_passive(self._model, self._data,key_callback=self._keyboard_cb)
        self._scene = mj.MjvScene(self._model, maxgeom=10000)

        self._data_lock = Lock()
        self._pose_trajectory = []

        self._hand_controller = hand_controller
        self._arm_controller = arm_controller
        self._robot_controller = None
        self._trajectory_steps = trajectory_steps
        self._cam_verbose = cam_verbose
        self._sim_verbose = sim_verbose

        # ur tcp to wrist joints
        # UR_EE_TO_SH_WRIST_JOINTS = 0.21268 # m
        # SH_WRIST_TO_SH_PALM      = 0.08721395775941231 # m

        # self._arm = rtb.DHRobot(
        #     [
        #         rtb.RevoluteDH(d = 0.1807, alpha = pi / 2.0),        # J1
        #         rtb.RevoluteDH(a = -0.6127),                         # J2
        #         rtb.RevoluteDH(a = -0.57155),                        # J3
        #         rtb.RevoluteDH(d = 0.17415, alpha =  pi / 2.0),      # J4
        #         rtb.RevoluteDH(d = 0.11985, alpha = -pi / 2.0),      # J5
        #         rtb.RevoluteDH(d = 0.11655),
        #     ], name="ur10e", base=sm.SE3.Trans(0,0,0)
        # )
        # self._robot = rtb.DHRobot(
        #     [
        #         rtb.RevoluteDH(d = 0.1807, alpha = pi / 2.0),        # J1
        #         rtb.RevoluteDH(a = -0.6127),                         # J2
        #         rtb.RevoluteDH(a = -0.57155),                        # J3
        #         rtb.RevoluteDH(d = 0.17415, alpha =  pi / 2.0),      # J4
        #         rtb.RevoluteDH(d = 0.11985, alpha = -pi / 2.0),      # J5
        #         rtb.RevoluteDH(d = 0.11655 + UR_EE_TO_SH_WRIST_JOINTS), # J6 + forearm
        #         rtb.RevoluteDH(alpha = pi / 2),                      # WR1
        #         rtb.RevoluteDH(alpha = pi / 2, offset= pi / 2),      # WR2
        #         rtb.RevoluteDH(d = SH_WRIST_TO_SH_PALM),             # from wrist to palm
        #     ], name="shur", base=sm.SE3.Trans(0,0,0)
        # )

        # self._trajectory_iter = []
        # self._transition_history = []

        # # set home pose
        # self._HOME_ARM   = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        # self._HOME_HAND  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        # self._HOME_ROBOT = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # self._ZERO_ROBOT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

        # self._q_arm = np.zeros( self.N_ACTUATORS_ARM )
        # self._q_hand = np.zeros( self.N_ACTUATORS_HAND )
        # self._q_robot = np.zeros( self.N_ACTUATORS_ROBOT )

        # self.__set_home()
        # self.set_q_arm(q = self._HOME_ARM)

        self.shur = SHUR(self._model,self._data)
        self.shur.home()

        mj.set_mjcb_control(self._controller_fn)

    # def __set_home(self):
    #     self.__set_q_hand(self._HOME_HAND)
    #     self.__set_q_arm(self._HOME_ARM)

    def __set_q_hand(self, q: Union[str,List]) -> None:
        hand_actuator_names = []
        for an in get_actuator_names(self._model):
            prefix = an.split("_")[0]
            if prefix == "rh" or prefix == "lh":
                hand_actuator_names.append(an)
        for i, han in enumerate(hand_actuator_names):
            set_joint_value(data=self._data, q=q[i], joint_name=han)

    def get_q_hand(self) -> RobotConfig:
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

    def __set_q_arm(self, q: Union[str,List]) -> None:
        arm_actuator_names = []
        for an in get_actuator_names(self._model):
            prefix = an.split("_")[0]
            if prefix == "ur10e":
                arm_actuator_names.append(an)
        for i, han in enumerate(arm_actuator_names):
            set_joint_value(data=self._data, q=q[i], joint_name=han)

    def get_q_arm(self) -> RobotConfig:
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

    # traj methods #################################3

    def set_q_hand(self, q: Union[str, List]) -> None:
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
            q:list = self._hand_controller.cfg_to_q(q)
        assert len(q) == self.N_ACTUATORS_HAND, f"Length of q should be {self.N_ACTUATORS_HAND}, q had length {len(q)}"
        
        self._hand_controller.set_traj( 
            start_ctrl = self.get_q_hand().joint_values,
            end_ctrl = q
        )

    def set_q_arm(self, q: Union[str,List]) -> None:
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
            q:list = self._arm_controller.cfg_to_q(q)
        assert len(q) == self.N_ACTUATORS_ARM, f"Length of q should be {self.N_ACTUATORS_ARM}, q had length {len(q)}"
        
        self._arm_controller.set_traj( 
            start_ctrl = self.get_q_arm().joint_values,
            end_ctrl = q
        )

    def viewer_cb(self):
        with self._window as viewer:
            while viewer.is_running():
                step_start = time.time()
                viewer.sync()

                with self._data_lock:
                    mj.mjv_updateScene(
                        self._model,
                        self._data,
                        self._options,
                        None,
                        self._camera,
                        mj.mjtCatBit.mjCAT_ALL.value,
                        self._scene
                    )
                    mj.mj_step(m=self._model, d=self._data)

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    # pass
                    time.sleep(time_until_next_step)

    # # Handles keyboard button events to interact with simulator
    def _keyboard_cb(self, key):
        if key == glfw.KEY_K:
            print(" >>>>> KEY K <<<<<")
            print(" >>>>> verifying hand <<<<<")
            self.shur.shadow_hand.set_q(q = "grasp")
        elif key == glfw.KEY_O:
            print(" >>>>> KEY O <<<<<+")
            print(" >>>>> verifying arm <<<<<")
            self.shur.ur10e.set_q(q = "up")
        elif key == glfw.KEY_PERIOD:
            print(" >>>>> KEY . <<<<<")
            pos = [0.5, 0.5, 0.5]
            ori = self.shur.ur10e.get_ee_pose().R

            self.shur.ur10e.set_ee_pose(pos=pos, ori=ori)

        elif key == glfw.KEY_H:
            self.shur.home()
        elif key == glfw.KEY_J:
            print("doing nothing...")

    # Defines controller behavior
    def _controller_fn(self, model: mj.MjModel, data: mj.MjData) -> None:
        if not self.shur.is_done:
            self.shur.step()

    # Runs GLFW main loop
    def run(self):
        self.viewer_thrd = Thread(target=self.viewer_cb, daemon=True)
        self.viewer_thrd.start()
        print("run done...")
        input()
        print("DONE")