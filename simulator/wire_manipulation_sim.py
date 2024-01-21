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
)

from sensors import Camera

class WiremanipulationSim(BaseMuJuCoSim):
    def __init__(self, args):
        self.args = args
        self._model   = self._get_mj_model()
        self._data    = self._get_mj_data(self._model)
        self._camera  = self._get_mj_camera()
        self._options = self._get_mj_options()
        self._window  = self._get_mj_window()
        self._scene   = self._get_mj_scene()
        self._pert    = self._get_mj_pertubation()
        self._context = self._get_mj_context()

        self._arm = UR10e(self._model, self._data, args)
        self._gripper = Hand2F85(self._model, self._data, args)
        self.robot = Robot(arm=self._arm, gripper=self._gripper, args=args)
        self.robot.home()

        mj.set_mjcb_control(self.controller_callback)

        self.cam = Camera(args=args, model=self._model, data=self._data, cam_name="cam")

    def _set_scene(self):
        pass

    # Handles keyboard button events to interact with simulator
    def keyboard_callback(self, key):
        if key == glfw.KEY_SPACE:
            # try:
            print("pressed space...")
            self.cam.shoot()

    # Defines controller behavior
    def controller_callback(self, model: mj.MjModel, data: mj.MjData) -> None:
        if not self.robot.is_done:
            self.robot.step()