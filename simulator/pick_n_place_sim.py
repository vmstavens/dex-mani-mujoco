import mujoco as mj
import mujoco.viewer
import mujoco_py
from mujoco.glfw import glfw
import time
import sys
from threading import Thread, Lock
from robots import Robot, ShadowHand, UR10e
from simulator.base_mujoco_sim import BaseMuJuCoSim
import math as m
from spatialmath import SE3

import spatialmath.base as smb

from utils.sim import (
    get_object_pose
)

from utils.rtb import (
    make_tf
)

from utils.mj import (
    get_joint_value,
    get_joint_names,
    is_done_actuator
)

class PickNPlaceSim(BaseMuJuCoSim):
    def __init__(self, args):
        self.args = args
        self._model   = self._get_mj_model()
        self._data    = self._get_mj_data(self._model)
        self._camera  = self._get_mj_camera()
        self._options = self._get_mj_options()
        self._window  = self._get_mj_window()
        self._scene   = self._get_mj_scene()

        self._ur10e = UR10e(self._model, self._data, args)
        self._rh = ShadowHand(self._model, self._data, args)

        self.robot = Robot(
            arm     = self._ur10e,
            gripper = self._rh,
            args    = args)
        self.robot.home()

        mj.set_mjcb_control(self.controller_callback)

    # # Handles keyboard button events to interact with simulator
    def keyboard_callback(self, key):
        box1_pose = get_object_pose("box1", model=self._model, data=self._data)
        box2_pose = get_object_pose("box2", model=self._model, data=self._data)
        if key == glfw.KEY_H:
            self.robot.home()

        elif key == glfw.KEY_COMMA:
            pass
        elif key == glfw.KEY_SPACE:
            print(" >>> initiated pick and place task <<<")
            q_pick_up = {
                "ur10e_shoulder_pan_joint": -0.9610317406481865,
                "ur10e_shoulder_lift_joint": -1.5502823078908237,
                "ur10e_elbow_joint": -2.091593176403119,
                "ur10e_wrist_1_joint": -2.655088563301384,
                "ur10e_wrist_2_joint": 0.6088429682430168,
                "ur10e_wrist_3_joint": 0.0005436264214320968
            }
            q_pick_up = [v for k,v in q_pick_up.items()]

            self.robot.arm.set_q(q = q_pick_up)

            T_w_grasp = make_tf(
                pos = box1_pose.t + [-0.3, 0-0.05, 0.06],
                ori = SE3.Ry(m.pi/2.0) * SE3.Rz(m.pi/2.0)
            )

            # self.robot.arm.set_ee_pose(T_w_pick_up)
            self.robot.gripper.set_q(q = "open")
            self.robot.arm.set_ee_pose(T_w_grasp)
            self.robot.gripper.set_q(q = "grasp")
            self.robot.arm.set_q(q = q_pick_up)

            T_w_box2 = make_tf(
                pos = box2_pose.t + [-0.3, 0.0-0.05, 0.3],
                # pos = box2_pose.t + [-0.3, 0-0.05, 0.06],
                ori = SE3.Ry(m.pi/2.0) * SE3.Rz(m.pi/2.0)
            )

            T_w_place = make_tf(
                pos = box2_pose.t + [-0.3, 0.0-0.05, 0.2],
                # pos = box2_pose.t + [-0.3, 0-0.05, 0.06],
                ori = SE3.Ry(m.pi/2.0) * SE3.Rz(m.pi/2.0)
            )

            self.robot.arm.set_ee_pose(T_w_box2)
            self.robot.arm.set_ee_pose(T_w_place)
            self.robot.gripper.set_q(q = "open")
            self.robot.arm.set_ee_pose(T_w_box2)


        elif key == glfw.KEY_J:
            print("doing nothing...")

    # Defines controller behavior
    def controller_callback(self, model: mj.MjModel, data: mj.MjData) -> None:
        if not self.robot.is_done:
            self.robot.step()

    # # Runs GLFW main loop
    # def run(self):
    #     self.viewer_thrd = Thread(target=self.viewer_callback, daemon=True)
    #     self.viewer_thrd.daemon = True
    #     self.viewer_thrd.start()
    #     input()
    #     print("done simulating...")