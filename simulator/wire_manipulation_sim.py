import mujoco as mj
import mujoco.viewer
import mujoco_py
from mujoco.glfw import glfw
from simulator.base_mujoco_sim import BaseMuJuCoSim
from robots import Robot, UR10e, HandE, Hand2F85
import numpy as np
import cv2
import pandas as pd
import json
import time
from datetime import datetime
import os
from utils.mj import (
    set_object_pose, 
    set_robot_pose,
    get_joint_names,
    get_object_pose,
)

from utils.rtb import make_tf

import rospy
from sensors import Camera, GelSightMini
import spatialmath as sm

class WiremanipulationSim(BaseMuJuCoSim):
    def __init__(self, args):
        self._args      = args
        self._model     = self._get_mj_model()
        self._data      = self._get_mj_data(self._model)
        self._camera    = self._get_mj_camera()
        self._options   = self._get_mj_options()
        self._window    = self._get_mj_window()
        self._scene     = self._get_mj_scene()
        self._pert      = self._get_mj_pertubation()
        self._data_lock = self._get_data_lock()
        self._context   = self._get_mj_context()

        rospy.init_node(self._args.sim_name)

        self._arm = UR10e(self._model, self._data, args)
        self._gripper = Hand2F85(self._model, self._data, args)
        self.robot = Robot(arm=self._arm, gripper=self._gripper, args=args)
        self.robot.home()

        self.cam_left = Camera(args=args, model=self._model, data=self._data, cam_name="cam_left" , live = True)
        self.cam_right = Camera(args=args, model=self._model, data=self._data, cam_name="cam_right", live = True)

        self.gs_left = GelSightMini(args=self._args, cam_name="cam_left")
        self.gs_right = GelSightMini(args=self._args, cam_name="cam_right")

        mj.set_mjcb_control(self.controller_callback)

    # Handles keyboard button events to interact with simulator
    def keyboard_callback(self, key):
        if key == glfw.KEY_SPACE:
            print("pressed space...")
            self.robot.home()
        elif key == glfw.KEY_COMMA:
            print("pressed comma...")
            rope_pose = get_object_pose(self._data, "rope")
            ee_pose = self.robot.arm.get_ee_pose()
            print("ee_pose =")
            print(ee_pose)

            base_pose = self.robot.arm.get_base_pose()
            print("base_pose =")
            print(base_pose)

            grasp_pose = rope_pose * sm.SE3.Tz(0.1)

            T_w_b = self.robot.arm.get_base_pose()
            print("T_w_b =")
            print(T_w_b)
            T_b_ee = self.robot.arm.get_ee_pose()
            print("T_b_ee =")
            print(T_b_ee)
            T_w_d = grasp_pose
            print("T_w_d =")
            print(T_w_d)

            
            T_ee_d = T_b_ee.inv() * T_w_b.inv() * T_w_d
            # T_ee_d = T_w_b.inv() * T_w_d * T_b_ee.inv()
            # temp = base_pose @ grasp_pose @ sm.SE3.Tz(0.1)
            print("T_ee_d =")
            print(T_ee_d)

            test_pose = make_tf(pos = [0.4, 0.4, 0.5], ori=ee_pose.R)
            print("test_pose =")
            print(test_pose)

            self.robot.arm.set_ee_pose(test_pose)
            # self.robot.arm.set_ee_pose(T_ee_d)
        elif key == glfw.KEY_PERIOD:
            print(self.robot.arm.get_ee_pose())


    # Defines controller behavior
    def controller_callback(self, model: mj.MjModel, data: mj.MjData) -> None:
        if not self.robot.is_done:
            self.robot.step()