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

from utils.helpers import (
    Vector3D,
    Pose
    )

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

# Quaternion order in MuJoCo w x y z

# TODO : pick up object (less mass on cube, more force in fingers, specify friction model) see examples Yuval Tassa

class GLWFSim:
    def __init__(
            self,
            shadow_hand_xml_filepath: str,
            hand_controller: Controller,
            trajectory_steps: int,
            cam_verbose: bool,
            sim_verbose: bool
    ):
        # with open(shadow_hand_xml_filepath, 'r') as file:
        #     xml_content = file.read()
        #     print(xml_content)
        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel
        self._model_file_path = shadow_hand_xml_filepath
        print(f"{self._model_file_path=}")
        self._model = mj.MjModel.from_xml_path(filename=shadow_hand_xml_filepath)
        print(f"{self._model=}")
        # self._model2 = mj.MjModel.from_xml_path(filename="/home/vims/git/dex-mani-mujoco/objects/universal_robots_ur10e/ur10e.xml")

        self._data = mj.MjData(self._model)
        self._camera = mj.MjvCamera()
        self._options = mj.MjvOption()

        self._keyboard_pos_step = 0.05

        self.dt = 1.0 / 100.0

        # self.grasp_pose = Pose(pos = self.box_pose.position, quat = self.hand_pose.quaternion).translate(delta_position=Vector3D(x=-0.3, y=0.0, z=0.3)).rotate(delta_orientation = rpy_to_quaternion(roll=m.pi)  )
        self._window = mujoco.viewer.launch_passive(self._model, self._data,key_callback=self._keyboard_cb)
        self._scene = mj.MjvScene(self._model, maxgeom=10000)

        self._i = 0

        self._pose_lock = Lock()
        self._data_lock = Lock()
        self._pose_trajectory = []
        # self._viewport_width, viewport_height = glfw.get_framebuffer_size(window=self._window)
        # self._context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150.value)

        self._hand_controller = hand_controller
        # self._pose_controller = PoseController(ctrl_limits=np.array([]))
        self._trajectory_steps = trajectory_steps
        self._cam_verbose = cam_verbose
        self._sim_verbose = sim_verbose


        # self._window = None
        # self._scene = None
        # self._context = None

        # self._mouse_button_left = False
        # self._mouse_button_middle = False
        # self._mouse_button_right = False
        # self._mouse_x_last = 0
        # self._mouse_y_last = 0
        # self._terminate_simulation = False

        self._sign = ''
        self._trajectory_iter = iter([])
        self._transition_history = []
        # self.hand_pose = get_pose("right_shadow_hand",model=self._model, data=self._data)

        
        # self._init_controller()
        # mj.set_mjcb_control(self._controller_fn)

    # def interpolate_poses(self, initial_pose, final_pose, num_steps):
    #     # Interpolate poses using spatialmath
    #     print(initial_pose)
    #     interpolated_poses = np.linspace(initial_pose.A, final_pose.A, num_steps)
    #     print(interpolated_poses[0])
    #     interpolated_se3_poses = [SE3(trnorm(x)) for x in interpolated_poses]
    #     return interpolated_se3_poses

    
    # def interpolate_poses(self, initial_pose, final_pose, num_steps):
    #     # Linear interpolation for position
    #     positions = np.linspace(initial_pose.t, final_pose.t, num_steps)

    #     # Spherical linear interpolation (slerp) for orientation
    #     initial_quaternion = UnitQuaternion(initial_pose.R)
    #     final_quaternion = UnitQuaternion(final_pose.R)

        

    #     orientations = UnitQuaternion.interpolate_slerp(initial_quaternion, final_quaternion, np.linspace(0, 1, num_steps))

    #     # Combine interpolated positions and orientations into SE3 poses
    #     interpolated_poses = [SE3(t=pos, R=orient.q2r()) for pos, orient in zip(positions, orientations)]

    #     return interpolated_poses

    @staticmethod
    def slerp(q1, q2, t):
        dot_product = np.dot(q1, q2)

        # Check if the quaternions are very close, use linear interpolation
        if abs(dot_product) > 0.99:
            result = (1 - t) * q1 + t * q2
            result /= np.linalg.norm(result)
            return result

        omega = np.arccos(dot_product)
        sin_omega = np.sin(omega)

        q_interpolated = (np.sin((1 - t) * omega) / sin_omega) * q1 + (np.sin(t * omega) / sin_omega) * q2
        return q_interpolated


    def interpolate_poses(self, initial_pose, final_pose, num_steps):
        # Linear interpolation for position
        positions = np.linspace(initial_pose.t, final_pose.t, num_steps)

        # Spherical linear interpolation (slerp) for orientation
        initial_quaternion = tf_2_quat(UnitQuaternion(initial_pose.R))
        final_quaternion   = tf_2_quat(UnitQuaternion(final_pose.R))

        slerp_interpolated_quaternions = [self.slerp(initial_quaternion, final_quaternion, t) for t in np.linspace(0, 1, num_steps)]

        # Convert interpolated quaternions to UnitQuaternion objects
        orientations = [UnitQuaternion(q) for q in slerp_interpolated_quaternions]

        poses = []

        print(orientations)
        for i,p in enumerate(positions):
            T = make_tf(quat=orientations[i]) @ make_tf(pos=p)
            poses.append(T)

        return poses


    def launch_mujoco(self):
        with self._window as viewer:
            while viewer.is_running():
                step_start = time.time()
                time_prev = self._data.time

                mj.mj_step(m=self._model, d=self._data)

                with self._pose_lock:
                    if self._pose_trajectory:
                        initial_pose = self._data.mocap_pos
                        initial_pose = make_tf(pos=self._data.mocap_pos, quat=self._data.mocap_quat)
                        final_pose = self._pose_trajectory.pop(0)  # Use the first pose in the list

                        # Interpolate poses smoothly
                        interpolated_poses = self.interpolate_poses(initial_pose, final_pose, num_steps=100)

                        for pose in interpolated_poses:
                            # with self._data_lock:
                            self._data.mocap_pos  = pose.t
                            self._data.mocap_quat = tf_2_quat(pose)

                            self._update()
                            # mj.mj_step(m=self._model, d=self._data)

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


    # @property
    # def transition_history(self) -> list:
    #     return self._transition_history

    # def _init_simulation(self):

        # self._init_world()
        # self._init_callbacks()
        # self._init_camera()


    # # Initializes world (simulation window)
    # def _init_world(self):
    #     glfw.init()
    #     self._window = glfw.create_window(1200, 900, 'Shadow Hand Simulation', None, None)
    #     glfw.make_context_current(window=self._window)
    #     glfw.swap_interval(interval=1)

    #     mj.mjv_defaultCamera(cam=self._camera)
    #     mj.mjv_defaultOption(opt=self._options)
    #     self._scene = mj.MjvScene(self._model, maxgeom=10000)
    #     self._context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150.value)

    #     self.set_pose(x = 1.0, y = 1.0, z = 1.0)

    # # Initializes keyboard & mouse callbacks for window navigation utilities
    # def _init_callbacks(self):
    #     glfw.set_key_callback(window=self._window,cbfun=self._keyboard_cb)
        # glfw.set_mouse_button_callback(window=self._window, cbfun=self._mouse_button_cb)
        # glfw.set_cursor_pos_callback(window=self._window, cbfun=self._mouse_move_cb)
        # glfw.set_scroll_callback(window=self._window, cbfun=self._mouse_scroll_cb)

    # # Initializes world camera (3D view)
    # def _init_camera(self):
    #     self._camera.azimuth = -180
    #     self._camera.elevation = -20
    #     self._camera.distance = 0.6
    #     self._camera.lookat = [0.37, 0, 0.02]

    def _set_position(self,pos: Vector3D) -> None:
        self._data.qpos[0] = pos.x
        self._data.qpos[1] = pos.y
        self._data.qpos[2] = pos.z

    def _set_quat(self, q: Quaternion) -> None:

        w = q.w
        x = q.x
        y = q.y
        z = q.z

        x = x if x is not None else 0.0
        y = y if y is not None else 0.0
        z = z if z is not None else 0.0
        w = w if w is not None else 0.0
        self._data.qpos[3] = w
        self._data.qpos[4] = x
        self._data.qpos[5] = y
        self._data.qpos[6] = z

    def _set_pose(self,pose: Pose) -> None:
        self._set_position(pos = pose.position)
        self._set_quat(q = pose.quaternion)

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

        # Render the final scene
        # mj.mjr_render(viewport=self._viewport, scn=self._scene, con=self._context)

        # Swap OpenGL buffers (blocking call due to v-sync)
        # glfw.swap_buffers(window=self._window)

        # Poll GLFW events
        # glfw.poll_events()

    # @property
    # def transition_history(self) -> List[List[Optional[float]]]:
    #     """Get the transition history."""
    #     return self._transition_history

    # def _interpolate_positions(self, current_pos: np.ndarray, target_pos: np.ndarray, alpha: float) -> np.ndarray:
    #     """Interpolate positions."""
    #     interpolated_pos = current_pos + alpha * (target_pos - current_pos)
    #     return interpolated_pos

    # def _interpolate_orientations(
    #         self,
    #         current_ori: Quaternion,
    #         target_ori: Quaternion,
    #         alpha: float
    #     ) -> Quaternion:
    #     """Interpolate orientations."""

    #     interpolated_quaternion = quaternion_slerp(current_ori, target_ori, alpha)
    #     return interpolated_quaternion

    # def _execute_trajectory(self, trajectory: List[Pose]) -> None:
    #     """Execute trajectory."""

    #     pos = {
    #         "x": [], "y": [], "z": [], 
    #         "des_x": [], "des_y": [], "des_z": [], 
    #         "qw" : [],"qx" : [],"qy" : [],"qz" : [],
    #         "des_qw" : [],"des_qx" : [],"des_qy" : [],"des_qz" : [] 
    #     }

    #     for target_pose in trajectory:

    #         self._set_pose( pose=target_pose )

    #         # self.hand_pose = get_pose("right_shadow_hand",model=self._model, data=self._data)

    #         pos["x"].append(self._data.qpos[0])
    #         pos["y"].append(self._data.qpos[1])
    #         pos["z"].append(self._data.qpos[2])

    #         pos["qw"].append(self._data.qpos[3])
    #         pos["qx"].append(self._data.qpos[4])
    #         pos["qy"].append(self._data.qpos[5])
    #         pos["qz"].append(self._data.qpos[6])

    #         pos["des_x"].append(target_pose.position.x)
    #         pos["des_y"].append(target_pose.position.y)
    #         pos["des_z"].append(target_pose.position.z)

    #         pos["des_qw"].append(target_pose.quaternion.w)
    #         pos["des_qx"].append(target_pose.quaternion.x)
    #         pos["des_qy"].append(target_pose.quaternion.y)
    #         pos["des_qz"].append(target_pose.quaternion.z)

    #         # Render the updated scene
    #         self._update()

    #     df = pd.DataFrame.from_dict(pos)
    #     df.to_csv("traj-2.csv")
    #     self.hand_pose = trajectory[-1]


    # def set_pose(self, x=None, y=None, z=None,
    #              qx=None, qy=None, qz=None, qw=None,
    #              roll=None, pitch=None, yaw=None,
    #              interpolation_time=1.0):

    #     current_pose = self._data.qpos[:7]
    #     current_pos = Vector3D.from_numpy(current_pose[:3])
    #     current_ori = Quaternion.from_numpy(current_pose[3:])

    #     pos = { "x":[], "y":[], "z":[], "des_x": [], "des_y": [], "des_z": [] }

    #     # Save the starting time
    #     start_time = time.time()
    #     # ctrl = start_ctrl + i*(end_ctrl - start_ctrl)/n_steps

    #     while time.time() - start_time < interpolation_time:
    #         # Compute interpolation factor (0 to 1) using ease-in-out function
    #         alpha = (time.time() - start_time) / interpolation_time
    #         # smoothstep = lambda x : x * x * (3 - 2 * x)
    #         # alpha = smoothstep((time.time() - start_time) / interpolation_time)

    #         # Interpolate positions
    #         if x is not None:
    #             self._data.qpos[0] = current_pos.x + alpha * (x - current_pos.x)
    #         if y is not None:
    #             self._data.qpos[1] = current_pos.y + alpha * (y - current_pos.y)
    #         if z is not None:
    #             self._data.qpos[2] = current_pos.z + alpha * (z - current_pos.z)

    #         pos["x"].append(self._data.qpos[0])
    #         pos["y"].append(self._data.qpos[1])
    #         pos["z"].append(self._data.qpos[2])

    #         pos["des_x"].append(current_pos.x + alpha * (x - current_pos.x))
    #         pos["des_y"].append(current_pos.y + alpha * (y - current_pos.y))
    #         pos["des_z"].append(current_pos.z + alpha * (z - current_pos.z))

            
    #         # Interpolate orientations using slerp
    #         if qw is not None:
    #             target_quaternion = Quaternion(w=qw,x=qx,y=qy,z=qz)
    #         else:
    #             # Convert roll-pitch-yaw to quaternion
    #             roll = roll if roll is not None else quaternion_to_rpy(current_ori)[0]
    #             pitch = pitch if pitch is not None else quaternion_to_rpy(current_ori)[1]
    #             yaw = yaw if yaw is not None else quaternion_to_rpy(q=current_ori)[2]
    #             target_quaternion = rpy_to_quaternion(roll, pitch, yaw)

    #         # q1 = Quaternion.from_numpy(current_ori)
    #         # q2 = Quaternion.from_numpy(target_quaternion)

    #         interpolated_quaternion = quaternion_slerp(current_ori, target_quaternion, alpha)
    #         self._data.qpos[3:7] = interpolated_quaternion.numpy()

    #         # Step the simulation
    #         mj.mj_step(m=self._model, d=self._data)

    #         # Render the updated scene
    #         viewport_width, viewport_height = glfw.get_framebuffer_size(window=self._window)
    #         viewport = mj.MjrRect(left=0, bottom=0, width=viewport_width, height=viewport_height)

    #         mj.mjv_updateScene(
    #             self._model,
    #             self._data,
    #             self._options,
    #             None,
    #             self._camera,
    #             mj.mjtCatBit.mjCAT_ALL.value,
    #             self._scene
    #         )
    #         mj.mjr_render(viewport=viewport, scn=self._scene, con=self._context)


    #         # Swap OpenGL buffers (blocking call due to v-sync)
    #         glfw.swap_buffers(window=self._window)

    #         # Poll GLFW events
    #         glfw.poll_events()

    #     # Set the final pose after the interpolation is complete
    #     if x is not None:
    #         self._data.qpos[0] = x
    #     if y is not None:
    #         self._data.qpos[1] = y
    #     if z is not None:
    #         self._data.qpos[2] = z
    #     if qw is not None:
    #         self._data.qpos[3] = qw
    #     if qy is not None:
    #         self._data.qpos[4] = qx
    #     if qz is not None:
    #         self._data.qpos[5] = qy
    #     if qw is not None:
    #         self._data.qpos[6] = qz

    #     # Step the simulation one last time to update with the final pose
    #     mj.mj_step(m=self._model, d=self._data)

    #     # Render the final scene
    #     mj.mjr_render(viewport=viewport, scn=self._scene, con=self._context)

    #     # Swap OpenGL buffers (blocking call due to v-sync)
    #     glfw.swap_buffers(window=self._window)

    #     # Poll GLFW events
    #     glfw.poll_events()
        
    #     df = pd.DataFrame.from_dict(pos)
    #     df.to_csv("test-linear-2.csv")

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

            # slerp_orientations:SO3 = SO3.interp(hand_tf, box_tf, np.linspace(0, 1, num_steps))
            # self._pose_trajectory = slerp_orientations
            # self._pose_trajectory = [hand_tf, box_tf]
            # for p in slerp_orientations:
                
            # self._data.mocap_quat = p.R

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
    # # Handles mouse-click events to move/rotate camera
    # def _mouse_button_cb(self, window, button, act, mods):
    #     self._mouse_button_left = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    #     self._mouse_button_middle = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    #     self._mouse_button_right = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    #     glfw.get_cursor_pos(window)

    # # Handles mouse-move callbacks to navigate camera
    # def _mouse_move_cb(self, window, xpos: int, ypos: int):
    #     dx = xpos - self._mouse_x_last
    #     dy = ypos - self._mouse_y_last
    #     self._mouse_x_last = xpos
    #     self._mouse_y_last = ypos

    #     if not (self._mouse_button_left or self._mouse_button_middle or self._mouse_button_right):
    #         return

    #     width, height = glfw.get_window_size(window=window)
    #     press_left_shift = glfw.get_key(window=window, key=glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    #     press_right_shift = glfw.get_key(window=window, key=glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    #     mod_shift = press_left_shift or press_right_shift

    #     if self._mouse_button_right:
    #         if mod_shift:
    #             action = mj.mjtMouse.mjMOUSE_MOVE_H
    #         else:
    #             action = mj.mjtMouse.mjMOUSE_MOVE_V
    #     elif self._mouse_button_left:
    #         if mod_shift:
    #             action = mj.mjtMouse.mjMOUSE_ROTATE_H
    #         else:
    #             action = mj.mjtMouse.mjMOUSE_ROTATE_V
    #     else:
    #         assert self._mouse_button_middle

    #         action = mj.mjtMouse.mjMOUSE_ZOOM

    #     mj.mjv_moveCamera(
    #         m=self._model,
    #         action=action,
    #         reldx=dx/height,
    #         reldy=dy/height,
    #         scn=self._scene,
    #         cam=self._camera
    #     )


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
        input()
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
        #     glfw.poll_events()
        # glfw.terminate()
