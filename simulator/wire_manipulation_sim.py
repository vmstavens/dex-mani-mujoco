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
        self._gripper = HandE(self._model, self._data, args)
        self.robot = Robot(gripper=self._gripper, args=args)

        mj.set_mjcb_control(self.controller_callback)

        self.cam = Camera(args=args, model=self._model, data=self._data, cam_name="cam")

    def shoot(self):
        # self.cam.shoot()
        # pass
        camera = self._get_mj_camera(cam_name="cam")
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, 'cam')
        mj.mjv_updateScene(self._model, self._data, self._options, self._pert, camera, mj.mjtCatBit.mjCAT_ALL, self._scene)
        
        viewport = mj.MjrRect(0, 0, self._args.cam_width, self._args.cam_height)
        context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150)

        mj.mjr_render(viewport, self._scene, context)
        image = np.empty((self._args.cam_height, self._args.cam_width, 3), dtype=np.uint8)
        depth_hat_buf = np.empty((self._args.cam_height, self._args.cam_width, 1),dtype=np.float32)
        mj.mjr_readPixels(image, depth_hat_buf, viewport, context)

        # OpenGL renders with inverted y axis
        image         = image.squeeze()
        depth_hat_buf = depth_hat_buf.squeeze()
        
        extent = self._model.stat.extent
        near = self._model.vis.map.znear * extent
        far = self._model.vis.map.zfar * extent
        depth_hat_buf = np.flipud(near / (1 - depth_hat_buf * (1 - near / far)))

        cv2.imwrite("/home/vims/git/dex-mani-mujoco/simulator/test.png", cv2.cvtColor( image,cv2.COLOR_RGB2BGR ))
        cv2.imwrite("/home/vims/git/dex-mani-mujoco/simulator/test_d.png",depth_hat_buf)

    def _set_scene(self):
        pass

    # Handles keyboard button events to interact with simulator
    def keyboard_callback(self, key):
        if key == glfw.KEY_SPACE:
            # try:
            print("pressed space...")
            # self.shoot()
            i, f, T =self.cam.matrices
            print(f"{i=}")
            print(f"{f=}")
            print(f"{T=}")
            self.cam.shoot()


    # Defines controller behavior
    def controller_callback(self, model: mj.MjModel, data: mj.MjData) -> None:
        pass
        # if not self.robot.is_done:
        #     self.robot.step()