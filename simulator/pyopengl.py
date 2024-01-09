import mujoco as mj
import mujoco.viewer
import mujoco_py
from mujoco.glfw import glfw
import time
import sys
from threading import Thread, Lock
from simulator.robot import Robot, UR10e, ShadowHand
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

class GLWFSim:
    def __init__(self, args):

        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel
        self._scene_path = args.scene_path
        self._model = mj.MjModel.from_xml_path(filename=self._scene_path)
        self._data = mj.MjData(self._model)
        self._camera = mj.MjvCamera()
        self._options = mj.MjvOption()

        # self._window = mujoco.viewer.launch_passive(self._model, self._data)
        self._window = mujoco.viewer.launch_passive(self._model, self._data,key_callback=self._keyboard_cb)
        self._scene = mj.MjvScene(self._model, maxgeom=10000)

        self._data_lock = Lock()

        self._ur10e = UR10e(self._model, self._data, args)
        self._rh = ShadowHand(self._model, self._data, args)

        self.robot = Robot(
            arm     = self._ur10e,
            gripper = self._rh,
            args    = args)
        self.robot.home()

        mj.set_mjcb_control(self._controller_fn)

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
                    time.sleep(time_until_next_step)

    # # Handles keyboard button events to interact with simulator
    def _keyboard_cb(self, key):
        box1_pose = get_object_pose("box1", model=self._model, data=self._data)
        box2_pose = get_object_pose("box2", model=self._model, data=self._data)
        if key == glfw.KEY_H:
            self.robot.home()

        elif key == glfw.KEY_COMMA:
            pass
        elif key == glfw.KEY_SPACE:
            # self.robot.set_q(q_robot = "up")
            # self.robot.arm.set_q( q = "home")
            # self.robot.gripper.set_q(q="open")
            # self.robot.gripper.set_q(q="home")
            # self.robot.gripper.set_q(q="open")

            # self.robot.arm.set_q(q = "up")

            q_pick_up = {
                "ur10e_shoulder_pan_joint": -0.9610317406481865,
                "ur10e_shoulder_lift_joint": -1.5502823078908237,
                "ur10e_elbow_joint": -2.091593176403119,
                "ur10e_wrist_1_joint": -2.655088563301384,
                "ur10e_wrist_2_joint": 0.6088429682430168,
                "ur10e_wrist_3_joint": 0.0005436264214320968
            }
            q_pick_up = [v for k,v in q_pick_up.items()]

            # q_grasp = {
            #     "ur10e_shoulder_pan_joint": -0.9263504904976173,
            #     "ur10e_shoulder_lift_joint": -1.9130301564704193,
            #     "ur10e_elbow_joint": -2.3802542358484513,
            #     "ur10e_wrist_1_joint": -2.05555859077024,
            #     "ur10e_wrist_2_joint": 0.6403986739987306,
            #     "ur10e_wrist_3_joint": 2.778605863121308e-05
            # }
            # q_grasp = [v for k,v in q_grasp.items()]

            self.robot.arm.set_q(q = q_pick_up)
            # self.robot.gripper.set_q(q = "open")
            # self.robot.arm.set_q(q = q_grasp)
            # self.robot.gripper.set_q(q = "grasp")


            # T_w_pick_up = make_tf(
            #     pos = box1_pose.t + [-0.32,0 -0.0,0.35],
            #     ori = SE3.Ry(m.pi/2.0) * SE3.Rz(m.pi/2.0)
            # )

            T_w_grasp = make_tf(
                pos = box1_pose.t + [-0.3, 0-0.03, 0.06],
                ori = SE3.Ry(m.pi/2.0) * SE3.Rz(m.pi/2.0)
            )

            # self.robot.arm.set_ee_pose(T_w_pick_up)
            self.robot.gripper.set_q(q = "open")
            self.robot.arm.set_ee_pose(T_w_grasp)
            self.robot.gripper.set_q(q = "grasp")
            self.robot.arm.set_q(q = q_pick_up)
            # self.robot.shadow_hand.set_q(q = "open")
            # self.robot.ur10e.set_ee_pose(T_w_pick_up)
            # # palm_ori = self.robot.ur10e.get_ee_pose().R
            
            # T_w_grasp = make_tf(
            #     pos = box1_pose.t + [-0.3,0,0.1],
            #     ori = SE3.Ry(m.pi/2.0) * SE3.Rz(m.pi/2.0)
            # )
            # self.robot.arm.set_ee_pose(T_w_grasp)
            # self.robot.shadow_hand.set_q(q = "grasp")

            # self.robot.set_q("up")
            # self.robot.set_q("home")

            # demo
            # self.robot.ur10e.set_q("up")
            # self.robot.ur10e.set_q("home")


        elif key == glfw.KEY_J:
            print("doing nothing...")

    # Defines controller behavior
    def _controller_fn(self, model: mj.MjModel, data: mj.MjData) -> None:
        if not self.robot.is_done:
            print(self.robot.arm.get_q())
            self.robot.step()

    # Runs GLFW main loop
    def run(self):
        self.viewer_thrd = Thread(target=self.viewer_cb, daemon=True)
        self.viewer_thrd.daemon = True
        self.viewer_thrd.start()
        input()
        print("done simulating...")