from numpy import ndarray
from utils.rtb import make_tf
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
import time

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from threading import Lock, Thread
from utils.geometry import geometry
from robots.base_robot import BaseRobot

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

class Robot(BaseRobot):
    def __init__(self, args, arm: BaseRobot = None, gripper: BaseRobot = None) -> None:
        super().__init__() 
        self._args           = args
        if arm is None:
            self._model = gripper.mj_model
            self._data = gripper.mj_data
        elif gripper is None:
            self._model = arm.mj_model
            self._data = arm.mj_data
        if arm is not None and gripper is not None:
            self._model = arm.mj_model
            self._data = arm.mj_data
        else:
            print("Warning! arm and gripper is None...")

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
            self._sub_gripper_q = rospy.Subscriber(f"mj/{self.gripper.name}_cmd_q", Float64MultiArray, callback=lambda q_msg: self.gripper.set_q(q=q_msg.data), buff_size=1)
            self._pub_gripper_q = rospy.Publisher(f"mj/{self.gripper.name}_q", Float64MultiArray, queue_size=1)
        if self._has_arm:
            self._arm._set_robot_handle(self)
            self._sub_arm_q = rospy.Subscriber(f"mj/{self.arm.name}_cmd_q", Float64MultiArray, callback=lambda q_msg: self.arm.set_q(q=q_msg.data), buff_size=1)
            self._pub_arm_q     = rospy.Publisher(f"mj/{self.arm.name}_q", Float64MultiArray, queue_size=1)
            self._sub_arm_ee    = rospy.Subscriber(f"mj/{self.arm.name}_cmd_q", Pose, callback=self._cmd_ee_callback, buff_size=1)
            self._pub_arm_ee    = rospy.Publisher(f"mj/{self.arm.name}_ee", Pose, queue_size=1)
        if self._has_arm and self._has_gripper:
            self._sub_robot_q   = rospy.Subscriber(f"mj/{self.name}_cmd_q", Float64MultiArray, callback=lambda q_msg: self.set_q(q=q_msg.data), buff_size=1)
            self._pub_robot_q   = rospy.Publisher(f"mj/{self.name}_q", Float64MultiArray, queue_size=1)

        self._trajectory_timeout = self._args.trajectory_timeout
        self._trajectory_time = 0.0 # s

        self._pub_lock = Lock()
        self._pub_thrd = Thread(target=self._pub_robot_info)
        self._pub_thrd.daemon = True
        self._pub_thrd.start()

    def _cmd_ee_callback(self, pose_msg: Pose):
        pos = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
        quat = [pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z]
        self.robot.arm.set_ee_pose(pos=pos, quat=quat)

    def _pub_robot_info(self):
        rate = rospy.Rate(self._args.robot_pub_freq)  # Set the publishing rate (1 Hz in this example)

        while not rospy.is_shutdown():
            with self._pub_lock:
                if self._has_arm:
                    self._pub_arm_q.publish(geometry.mk_float64multiarray(self.arm.get_q().joint_values))
                    arm_ee_pose = self.arm.get_ee_pose()
                    self._pub_arm_ee.publish(geometry.mk_pose(arm_ee_pose.t,ori=arm_ee_pose.R))
                if self._has_gripper:
                    self._pub_gripper_q.publish(geometry.mk_float64multiarray(self.gripper.get_q().joint_values))
                if self._has_arm and self._has_gripper:
                    self._pub_robot_q.publish(geometry.mk_float64multiarray(self.get_q().joint_values))

            rate.sleep()

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
        if self._has_arm and self._has_gripper:
            return self._arm.name + "_" + self._gripper.name
        elif self._has_arm and not self._has_gripper:
            return self._arm.name
        elif self._has_gripper and not self._arm:
            return self._gripper.name
        else:
            return ""

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
            if self._has_arm:
                if self.arm.name in ac_name:
                    result.append(ac_name)
            if self._has_gripper:
                if self.gripper.name in ac_name:
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
            if self.name:
                robot_actuator_names.append(an)
        for i in range(len(robot_actuator_names)):
            set_actuator_value(data=self._data, q=q[i], actuator_name=robot_actuator_names[i])

    def is_timout(self) -> bool:
        return (time.monotonic() - self._trajectory_time) > self._trajectory_timeout

    def get_q(self) -> RobotConfig:
        robot_joint_names = []
        robot_joint_values = []
        for an in get_joint_names(self.mj_model):
            if an == '':
                continue
            prefix = an.split("_")[0]
            if self.name:
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
        if ( self.arm._are_done_actuators() and self.gripper._are_done_actuators() ) or self.is_timout():
            # to ensure an element always exsists in _traj. otherwise 
            # gripper and arm cant be controlled independently
            self._trajectory_time = time.monotonic()
            self._set_q(self._traj[0])
            if len(self._traj) > 1:
                self._traj.pop(0)

    def set_ee_pose(
            self, 
            pos: List = [0.5, 0.5, 0.5], 
            ori: Union[np.ndarray,SE3] = [1, 0, 0, 0], 
            pose: Union[List[float], ndarray, SE3] = None, 
            solution_pool: int = 4) -> None:
        raise NotImplementedError(self.__class__.__name__ + ' cannot set ee pose for Robot') 

    def get_ee_pose(self) -> SE3:
        raise NotImplementedError(self.__class__.__name__ + ' cannot get ee pose for Robot')

    def home(self):
        if self._has_arm:
            self.arm.home()
        if self._has_gripper:
            self.gripper.home()