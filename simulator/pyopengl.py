import mujoco as mj
import mujoco.viewer
import mujoco
from mujoco.glfw import glfw
from controllers.controller import Controller
from utils import control
import mujoco_py
from typing import Tuple
import numpy as np
import time
import math as m
from math import pi
import sys
import pandas as pd
from typing import List, Optional
from threading import Thread, Lock
import roboticstoolbox as rtb
from spatialmath import (
    SE3, SO3, Quaternion, UnitQuaternion
    )
from scipy.spatial.transform import Slerp
from spatialmath.base import trnorm
from scipy.spatial.transform import Slerp

import spatialmath as sm
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

class GLWFSim:
    def __init__(
            self,
            shadow_hand_xml_filepath: str,
            hand_controller: Controller,
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

        self._keyboard_pos_step = 0.05
        self.dt = 1.0 / 100.0

        self._window = mujoco.viewer.launch_passive(self._model, self._data)
        # self._window = mujoco.viewer.launch_passive(self._model, self._data,key_callback=self._keyboard_cb)
        self._scene = mj.MjvScene(self._model, maxgeom=10000)

        self._pose_lock = Lock()
        self._data_lock = Lock()
        self._pose_trajectory = []

        self._hand_controller = hand_controller
        self._trajectory_steps = trajectory_steps
        self._cam_verbose = cam_verbose
        self._sim_verbose = sim_verbose

        self._arm = rtb.DHRobot(
            [
                rtb.RevoluteDH(d = 0.1807, alpha = pi / 2.0),   # J1
                rtb.RevoluteDH(a = -0.6127),                    # J2
                rtb.RevoluteDH(a = -0.57155),                   # J3
                rtb.RevoluteDH(d = 0.17415, alpha =  pi / 2.0),  # J4
                rtb.RevoluteDH(d = 0.11985, alpha = -pi / 2.0), # J5
                rtb.RevoluteDH(d = 0.11655),                    # J6
            ], name="ur10e", base=sm.SE3.Trans(0,0,0)
        )

        self._sign = ''
        self._trajectory_iter = iter([])
        self._transition_history = []

        # set home pose
        self.__HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # self.__HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.__set_q(q = self.__HOME)

        # self._init_controller()
        # mj.set_mjcb_control(self._controller_fn)

    def __set_q(self, q: list) -> None:
            for i in range(len(q)):
                self._data.qpos[i] = q[i]

    def launch_mujoco(self):
        with self._window as viewer:
            while viewer.is_running():
                step_start = time.time()

                mj.mj_step(m=self._model, d=self._data)
                mj.mjv_updateScene(
                    self._model,
                    self._data,
                    self._options,
                    None,
                    self._camera,
                    mj.mjtCatBit.mjCAT_ALL.value,
                    self._scene
                )
                # self._update()
                print("----")
                print(self._data.qpos[:7])
                print("----")
                print(self._data.qpos[0])
                self.__set_q(self.__HOME)

                if (self._data.qpos[0] > -1 ):
                    print("potential problem...")

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (time.time() - step_start)
                print("slep tim: ",time_until_next_step)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def _update(self):

        mj.mjv_updateScene(
                self._model,
                self._data,
                self._options,
                None,
                self._camera,
                mj.mjtCatBit.mjCAT_ALL.value,
                self._scene
            )

        # Step the simulation one last time to update with the final pose
        with self._data_lock:
            mj.mj_step(m=self._model, d=self._data)

        # # Render the final scene
        # mj.mjr_render(viewport=self._viewport, scn=self._scene, con=self._context)

        # # Swap OpenGL buffers (blocking call due to v-sync)
        # glfw.swap_buffers(window=self._window)

        # # Poll GLFW events
        # glfw.poll_events()

    # # Handles keyboard button events to interact with simulator
    def _keyboard_cb(self, key):
    # def _keyboard_cb(self, window, key, scancode, act, mods)
        # print("running")
        print(f"{key=} | {glfw.PRESS=}")
        # if key == glfw.PRESS:
        if key == glfw.KEY_BACKSPACE:
            with self._data_lock:
                mj.mj_resetData(self._model, self._data)
                mj.mj_forward(self._model, self._data)
            print("> reset simulation succeeded...")
        elif key == glfw.KEY_ESCAPE:
            self._terminate_simulation = True
            print("> terminated simulation...")
        elif key == glfw.KEY_UP:
            print("+x")
            with self._data_lock:
                self._data.mocap_pos[0, 0] += self._keyboard_pos_step
        elif key == glfw.KEY_DOWN:
            print("-x")
            with self._data_lock:
                self._data.mocap_pos[0, 0] -= self._keyboard_pos_step
        elif key == glfw.KEY_LEFT:
            print("+y")
            with self._data_lock:
                self._data.mocap_pos[0, 1] += self._keyboard_pos_step
        elif key == glfw.KEY_RIGHT:
            print("-y")
            with self._data_lock:
                self._data.mocap_pos[0, 1] -= self._keyboard_pos_step
        elif key == glfw.KEY_PERIOD:
            print("+z")
            with self._data_lock:
                self._data.mocap_pos[0, 2] += self._keyboard_pos_step
        elif key == glfw.KEY_COMMA:
            print("-z")
            with self._data_lock:
                self._data.mocap_pos[0, 2] -= self._keyboard_pos_step
            # print(self._data.mocap_quat)
        elif key == glfw.KEY_R:
            # get pose of robot

            n_steps = 10
            with self._data_lock:
                box_pose = get_pose("box", data=self._data, model=self._model)
                hand_pose = get_pose("hand", data=self._data, model=self._model)
            grasp_pose = box_pose.Tz(1.3)
            grasp_pose = rotate_x(grasp_pose, m.pi/2)
            
            print("box pose =") 
            print(box_pose)
            print("hand pose =") 
            print(hand_pose)
            print("grasp pose =")
            print(grasp_pose)

            test = generate_pose_trajectory(
                start_pose=hand_pose,
                end_pose=grasp_pose,
                n_steps=n_steps
            )
            self._pose_trajectory = test

            print(self._pose_trajectory)

        elif key == glfw.KEY_T:
            print("Pressed 2...")
        elif key == glfw.KEY_O:
            print("open...")
            self._sign = 'open'
            self._hand_controller.set_sign(sign=self._sign)
            # self._update()
        elif key == glfw.KEY_P:
            print("grasp...")
            self._sign = 'grasp'
            self._hand_controller.set_sign(sign=self._sign)
            # self._update()
        elif key == glfw.KEY_5:
            self._sign = 'yes'
            self._hand_controller.set_sign(sign=self._sign)
        elif key == glfw.KEY_6:
            self._sign = 'rock'
            self._hand_controller.set_sign(sign=self._sign)
        elif key == glfw.KEY_7:
            self._sign = 'circle'
            self._hand_controller.set_sign(sign=self._sign)

    # Initializes hand controller
    def _init_controller(self):
        # self._sign = 'order'
        self._sign = 'rest'
        self._hand_controller.set_sign(sign=self._sign)

    # Defines controller behavior
    def _controller_fn(self, model: mj.MjModel, data: mj.MjData) -> None:
        # as long as no new goal is defined, then we simply return (this function is called often)
        if self._hand_controller.is_done:
            return

        # Retrieves next trajectory control
        # trajectory is a non-empty list of joint configurations e.g. q
        next_ctrl = next(self._trajectory_iter, None)

        # Executes control if ctrl is not None else creates next trajectory between a control transition
        if next_ctrl is None:
            with self._data_lock:
                start_ctrl = data.ctrl
                end_ctrl = self._hand_controller.get_next_control(sign=self._sign)

            if end_ctrl is None:
                if self._sim_verbose:
                    print('Sign transitions completed')
            else:
                if self._sim_verbose:
                    print(f'New control transition is set from {start_ctrl} to {end_ctrl}')

                row = [self._sign, self._hand_controller.order - 1, end_ctrl]
                self._transition_history.append(row)

                # 100 step trajectory of the hand
                control_trajectory = control.generate_control_trajectory(
                    start_ctrl=start_ctrl,
                    end_ctrl=end_ctrl,
                    n_steps=self._trajectory_steps
                )

                # set the next control sequence equal to this control trajecotry
                self._trajectory_iter = iter(control_trajectory)

                if self._sim_verbose:
                    print('New trajectory is computed')
        else:
            # set the data control to the next in the 100 element long list of q configs '
            with self._data_lock:
                data.ctrl = next_ctrl

    # Runs GLFW main loop
    def run(self):
        print("etst")
        with self._window as viewer:
            print("etst2")
            while viewer.is_running():
                print("etst3")
                step_start = time.time()
                with self._data_lock:
                    print(self._data.qpos[:7])
                    print(self._data.qpos[0])
                    self.__set_q(self.__HOME)

                    if (abs(self._data.qpos[0]) < 1 ):
                        print("potential problem...")
                        # exit()

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                self._update()
        print("done")
        # self.mujoco_thrd = Thread(target=self.launch_mujoco, daemon=True)
        # self.mujoco_thrd.start()
        # input()
        # self._window = glfw.create_window(1200, 900, 'Shadow Hand Simulation', None, None)

        # while not glfw.window_should_close(window=self._window) and not self._terminate_simulation:
        #     time_prev = self._data.time

        #     # print("test = ",self.hand_pose)
        #     # self._set_pose(self.hand_pose)
        #     # self._update()
        #     while self._data.time - time_prev < 1.0/60.0:
        #         mj.mj_step(m=self._model, d=self._data)

        #     viewport_width, viewport_height = glfw.get_framebuffer_size(window=self._window)
        #     viewport = mj.MjrRect(left=0, bottom=0, width=viewport_width, height=viewport_height)

        #     if self._cam_verbose:
        #         print(
        #             f'Camera Azimuth = {self._camera.azimuth}, '
        #             f'Camera Elevation = {self._camera.elevation}, '
        #             f'Camera Distance = {self._camera.distance}, '
        #             f'Camera Lookat = {self._camera.lookat}'
        #         )

        #     # Update scene and render
        #     mj.mjv_updateScene(
        #         self._model,
        #         self._data,
        #         self._options,
        #         None,
        #         self._camera,
        #         mj.mjtCatBit.mjCAT_ALL.value,
        #         self._scene
        #     )
        #     mj.mjr_render(viewport=viewport, scn=self._scene, con=self._context)

        #     # swap OpenGL buffers (blocking call due to v-sync)
        #     glfw.swap_buffers(window=self._window)

        #     # process pending GUI events, call GLFW callbacks
            # glfw.poll_events()
        # glfw.terminate()
