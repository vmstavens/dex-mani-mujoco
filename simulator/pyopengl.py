import mujoco as mj
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
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'shadow_hand/utils')

from utils.helpers import (
    Pose, 
    Vector3D, 
    Quaternion, 
    generate_pose_trajectory,
    quaternion_slerp,
    quaternion_to_rpy,
    rpy_to_quaternion,
    get_pose,
    normalize_quaternion
    )

from controllers.pose_controller import PoseController

# Quaternion order in MuJoCo w x y z

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
        self._model = mj.MjModel.from_xml_path(filename=shadow_hand_xml_filepath)

        self._hand_controller = hand_controller
        self._pose_controller = PoseController(ctrl_limits=np.array([]))
        self._trajectory_steps = trajectory_steps
        self._cam_verbose = cam_verbose
        self._sim_verbose = sim_verbose

        self._data = mj.MjData(self._model)
        self._camera = mj.MjvCamera()
        self._options = mj.MjvOption()

        self._window = None
        self._scene = None
        self._context = None

        self._mouse_button_left = False
        self._mouse_button_middle = False
        self._mouse_button_right = False
        self._mouse_x_last = 0
        self._mouse_y_last = 0
        self._terminate_simulation = False

        self._sign = ''
        self._trajectory_iter = iter([])
        self._trajectory_iter_pose = iter([])
        self._transition_history = []

        self._goal_pose = np.array([])

        self._init_simulation()
        self._init_controller()
        mj.set_mjcb_control(self._controller_fn)

    @property
    def transition_history(self) -> list:
        return self._transition_history

    def _init_simulation(self):
        self._init_world()
        self._init_callbacks()
        self._init_camera()

    # Initializes world (simulation window)
    def _init_world(self):
        glfw.init()
        self._window = glfw.create_window(1200, 900, 'Shadow Hand Simulation', None, None)
        glfw.make_context_current(window=self._window)
        glfw.swap_interval(interval=1)

        mj.mjv_defaultCamera(cam=self._camera)
        mj.mjv_defaultOption(opt=self._options)
        self._scene = mj.MjvScene(self._model, maxgeom=10000)
        self._context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150.value)

        self.set_pose(x = 1.0, y = 1.0, z = 1.0)

    # Initializes keyboard & mouse callbacks for window navigation utilities
    def _init_callbacks(self):
        glfw.set_key_callback(window=self._window, cbfun=self._keyboard_cb)
        glfw.set_mouse_button_callback(window=self._window, cbfun=self._mouse_button_cb)
        glfw.set_cursor_pos_callback(window=self._window, cbfun=self._mouse_move_cb)
        glfw.set_scroll_callback(window=self._window, cbfun=self._mouse_scroll_cb)

    # Initializes world camera (3D view)
    def _init_camera(self):
        self._camera.azimuth = -180
        self._camera.elevation = -20
        self._camera.distance = 0.6
        self._camera.lookat = [0.37, 0, 0.02]

    def _set_position(self,x = None,y = None,z = None) -> None:
        if x is not None:
            self._data.qpos[0] = x
        if y is not None:
            self._data.qpos[1] = y
        if z is not None:
            self._data.qpos[2] = z

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

    def _set_pose(self,x = None,y = None,z = None,roll = None, pitch = None, yaw = None):
        # if roll is None and pitch is None and yaw is None:
        roll = roll if roll is not None else 0.0
        pitch = pitch if pitch is not None else 0.0
        yaw = yaw if yaw is not None else 0.0
        q = rpy_to_quaternion(roll=roll,pitch=pitch,yaw=yaw)
        self._set_position(x=x, y=y, z=z)
        self._set_quat(q=q)
        print(f"Set pose to {x=}, {y=}, {z=}, {roll=}, {pitch=}, {yaw=}")

    def _set_pose2(self,pose: Pose) -> None:
        self._set_position(
            x = pose.position.x,
            y = pose.position.y,
            z = pose.position.z)
        # self._set_quat(q = pose.quaternion)

    def _update(self):
        viewport_width, viewport_height = glfw.get_framebuffer_size(window=self._window)
        viewport = mj.MjrRect(left=0, bottom=0, width=viewport_width, height=viewport_height)

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
        mj.mj_step(m=self._model, d=self._data)

        # Render the final scene
        mj.mjr_render(viewport=viewport, scn=self._scene, con=self._context)

        # Swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(window=self._window)

        # Poll GLFW events
        glfw.poll_events()

    @property
    def transition_history(self) -> List[List[Optional[float]]]:
        """Get the transition history."""
        return self._transition_history

    def _interpolate_positions(self, current_pos: np.ndarray, target_pos: np.ndarray, alpha: float) -> np.ndarray:
        """Interpolate positions."""
        interpolated_pos = current_pos + alpha * (target_pos - current_pos)
        return interpolated_pos

    def _interpolate_orientations(
            self,
            current_ori: Quaternion,
            target_ori: Quaternion,
            alpha: float
        ) -> Quaternion:
        """Interpolate orientations."""

        interpolated_quaternion = quaternion_slerp(current_ori, target_ori, alpha)
        return interpolated_quaternion

    def _execute_trajectory(self, trajectory: List[Pose]) -> None:
        """Execute trajectory."""

        pos = {
            "x": [], "y": [], "z": [], 
            "des_x": [], "des_y": [], "des_z": [], 
            "qw" : [],"qx" : [],"qy" : [],"qz" : [],
            "des_qw" : [],"des_qx" : [],"des_qy" : [],"des_qz" : [] 
        }

        for target_pose in trajectory:
            start_pose = get_pose("right_shadow_hand", model=self._model, data=self._data)
            orientation_start = start_pose.quaternion

            print(f"moving to {target_pose=}")

            self._set_pose2( pose=target_pose )

            pos["x"].append(self._data.qpos[0])
            pos["y"].append(self._data.qpos[1])
            pos["z"].append(self._data.qpos[2])

            pos["qw"].append(self._data.qpos[3])
            pos["qx"].append(self._data.qpos[4])
            pos["qy"].append(self._data.qpos[5])
            pos["qz"].append(self._data.qpos[6])

            pos["des_x"].append(target_pose.position.x)
            pos["des_y"].append(target_pose.position.y)
            pos["des_z"].append(target_pose.position.z)

            pos["des_qw"].append(target_pose.quaternion.w)
            pos["des_qx"].append(target_pose.quaternion.x)
            pos["des_qy"].append(target_pose.quaternion.y)
            pos["des_qz"].append(target_pose.quaternion.z)

            # Render the updated scene
            self._update()

        df = pd.DataFrame.from_dict(pos)
        df.to_csv("traj-2.csv")

    def set_pose(self, x=None, y=None, z=None,
                 qx=None, qy=None, qz=None, qw=None,
                 roll=None, pitch=None, yaw=None,
                 interpolation_time=1.0):

        current_pose = self._data.qpos[:7]
        current_pos = Vector3D.from_numpy(current_pose[:3])
        current_ori = Quaternion.from_numpy(current_pose[3:])

        pos = { "x":[], "y":[], "z":[], "des_x": [], "des_y": [], "des_z": [] }

        # Save the starting time
        start_time = time.time()
        # ctrl = start_ctrl + i*(end_ctrl - start_ctrl)/n_steps

        while time.time() - start_time < interpolation_time:
            # Compute interpolation factor (0 to 1) using ease-in-out function
            alpha = (time.time() - start_time) / interpolation_time
            # smoothstep = lambda x : x * x * (3 - 2 * x)
            # alpha = smoothstep((time.time() - start_time) / interpolation_time)

            # Interpolate positions
            if x is not None:
                self._data.qpos[0] = current_pos.x + alpha * (x - current_pos.x)
            if y is not None:
                self._data.qpos[1] = current_pos.y + alpha * (y - current_pos.y)
            if z is not None:
                self._data.qpos[2] = current_pos.z + alpha * (z - current_pos.z)

            pos["x"].append(self._data.qpos[0])
            pos["y"].append(self._data.qpos[1])
            pos["z"].append(self._data.qpos[2])

            pos["des_x"].append(current_pos.x + alpha * (x - current_pos.x))
            pos["des_y"].append(current_pos.y + alpha * (y - current_pos.y))
            pos["des_z"].append(current_pos.z + alpha * (z - current_pos.z))

            
            # Interpolate orientations using slerp
            if qw is not None:
                target_quaternion = Quaternion(w=qw,x=qx,y=qy,z=qz)
            else:
                # Convert roll-pitch-yaw to quaternion
                roll = roll if roll is not None else quaternion_to_rpy(current_ori)[0]
                pitch = pitch if pitch is not None else quaternion_to_rpy(current_ori)[1]
                yaw = yaw if yaw is not None else quaternion_to_rpy(q=current_ori)[2]
                target_quaternion = rpy_to_quaternion(roll, pitch, yaw)

            # q1 = Quaternion.from_numpy(current_ori)
            # q2 = Quaternion.from_numpy(target_quaternion)

            interpolated_quaternion = quaternion_slerp(current_ori, target_quaternion, alpha)
            self._data.qpos[3:7] = interpolated_quaternion.numpy()

            # Step the simulation
            mj.mj_step(m=self._model, d=self._data)

            # Render the updated scene
            viewport_width, viewport_height = glfw.get_framebuffer_size(window=self._window)
            viewport = mj.MjrRect(left=0, bottom=0, width=viewport_width, height=viewport_height)

            mj.mjv_updateScene(
                self._model,
                self._data,
                self._options,
                None,
                self._camera,
                mj.mjtCatBit.mjCAT_ALL.value,
                self._scene
            )
            mj.mjr_render(viewport=viewport, scn=self._scene, con=self._context)


            # Swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(window=self._window)

            # Poll GLFW events
            glfw.poll_events()

        # Set the final pose after the interpolation is complete
        if x is not None:
            self._data.qpos[0] = x
        if y is not None:
            self._data.qpos[1] = y
        if z is not None:
            self._data.qpos[2] = z
        if qw is not None:
            self._data.qpos[3] = qw
        if qy is not None:
            self._data.qpos[4] = qx
        if qz is not None:
            self._data.qpos[5] = qy
        if qw is not None:
            self._data.qpos[6] = qz

        # Step the simulation one last time to update with the final pose
        mj.mj_step(m=self._model, d=self._data)

        # Render the final scene
        mj.mjr_render(viewport=viewport, scn=self._scene, con=self._context)

        # Swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(window=self._window)

        # Poll GLFW events
        glfw.poll_events()
        
        df = pd.DataFrame.from_dict(pos)
        df.to_csv("test-linear-2.csv")

    # Handles keyboard button events to interact with simulator
    def _keyboard_cb(self, window, key: int, scancode, act: int, mods):
        
        if act == glfw.PRESS:
            if key == glfw.KEY_BACKSPACE:
                mj.mj_resetData(self._model, self._data)
                mj.mj_forward(self._model, self._data)
                print("> reset simulation succeeded...")

            elif key == glfw.KEY_ESCAPE:
                self._terminate_simulation = True
                print("> terminated simulation...")
            
            elif key == glfw.KEY_1:
                print(f"Pressed key 1...")
                
                box_pose  = get_pose("box",data=self._data, model=self._model)
                print(f"{box_pose=}")

                hand_pose = get_pose("right_shadow_hand",model=self._model, data=self._data)
                print(f"{hand_pose=}")

                # grasp_pose = Pose.copy(box_pose).lift(0.2).rot_x(90, in_degrees=True)
                # print(f"lifted box pose to {grasp_pose=}")


                print("generating traj...")
                pose_trajectory = generate_pose_trajectory(
                    start_pose = hand_pose,
                    end_pose = hand_pose.translate( delta_position = Vector3D(0.5,0,0) ),
                    n_steps=100
                )

                print("executing traj...")
                self._execute_trajectory(pose_trajectory)
                print("done...")

            elif key == glfw.KEY_2:
                pass

            elif key == glfw.KEY_3:
                print("grasping...")
                self._sign = 'grasp'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_4:
                self._sign = 'grasp'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_5:
                self._sign = 'yes'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_6:
                self._sign = 'rock'
                self._hand_controller.set_sign(sign=self._sign)
            elif key == glfw.KEY_7:
                self._sign = 'circle'
                self._hand_controller.set_sign(sign=self._sign)

    # Handles mouse-click events to move/rotate camera
    def _mouse_button_cb(self, window, button, act, mods):
        self._mouse_button_left = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._mouse_button_middle = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._mouse_button_right = glfw.get_mouse_button(window=window, button=glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        glfw.get_cursor_pos(window)

    # Handles mouse-move callbacks to navigate camera
    def _mouse_move_cb(self, window, xpos: int, ypos: int):
        dx = xpos - self._mouse_x_last
        dy = ypos - self._mouse_y_last
        self._mouse_x_last = xpos
        self._mouse_y_last = ypos

        if not (self._mouse_button_left or self._mouse_button_middle or self._mouse_button_right):
            return

        width, height = glfw.get_window_size(window=window)
        press_left_shift = glfw.get_key(window=window, key=glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        press_right_shift = glfw.get_key(window=window, key=glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = press_left_shift or press_right_shift

        if self._mouse_button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self._mouse_button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            assert self._mouse_button_middle

            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(
            m=self._model,
            action=action,
            reldx=dx/height,
            reldy=dy/height,
            scn=self._scene,
            cam=self._camera
        )

    # Zooms in/out with the camera inside the simulation world
    def _mouse_scroll_cb(self, window, xoffset: float, yoffset: float):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self._model, action, 0.0, -0.05*yoffset, self._scene, self._camera)

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
            # set the data control to the next in the 100 element long list of q configs 
            data.ctrl = next_ctrl

    # Runs GLFW main loop
    def run(self):
        while not glfw.window_should_close(window=self._window) and not self._terminate_simulation:
            time_prev = self._data.time

            # while self._data.time - time_prev < 1.0/100.0:
            #     mj.mj_step(m=self._model, d=self._data)

            viewport_width, viewport_height = glfw.get_framebuffer_size(window=self._window)
            viewport = mj.MjrRect(left=0, bottom=0, width=viewport_width, height=viewport_height)

            if self._cam_verbose:
                print(
                    f'Camera Azimuth = {self._camera.azimuth}, '
                    f'Camera Elevation = {self._camera.elevation}, '
                    f'Camera Distance = {self._camera.distance}, '
                    f'Camera Lookat = {self._camera.lookat}'
                )

            # Update scene and render
            mj.mjv_updateScene(
                self._model,
                self._data,
                self._options,
                None,
                self._camera,
                mj.mjtCatBit.mjCAT_ALL.value,
                self._scene
            )
            mj.mjr_render(viewport=viewport, scn=self._scene, con=self._context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(window=self._window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
        glfw.terminate()
