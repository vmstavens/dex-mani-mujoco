import mujoco as mj
import mujoco.viewer
import mujoco_py
from mujoco.glfw import glfw
from simulator.base_mujoco_sim import BaseMuJuCoSim
from robots import Robot, UR10e, HandE

from utils.mj import (
    set_object_pose, 
    set_robot_pose,
    get_joint_names,
    )

class TaskBoardSim(BaseMuJuCoSim):
    def __init__(self, args):
        self.args = args
        self._model   = self._get_mj_model()
        self._data    = self._get_mj_data(self._model)
        self._camera  = self._get_mj_camera()
        self._options = self._get_mj_options()
        self._window  = self._get_mj_window()
        self._scene   = self._get_mj_scene()

        self._set_scene()

        self._arm = UR10e(self._model, self._data, args)
        self._gripper = HandE(self._model, self._data, args)

        self.robot = Robot(arm=self._arm, gripper=self._gripper, args=args)

        mj.set_mjcb_control(self.controller_callback)

    def _set_scene(self):
        pass
        # flexcell dim (0.4, 0.6, 0.3) = (x, y, z)
        # set_robot_pose()
        # set_object_pose(data=self._data, model=self._model, object_name="flexcell",pos=[0.4, 0.6, 0.0])

    # Handles keyboard button events to interact with simulator
    def keyboard_callback(self, key):
        if key == glfw.KEY_SPACE:
            print(self._data.geom)
            # print(self._data.geom("task_board"))
            print(self._data.body("tb"))

            # set_object_pose(data=self._data, model=self._model, object_name="flexcell",pos=[0.4, 0.6, 0.0])


    # Defines controller behavior
    def controller_callback(self, model: mj.MjModel, data: mj.MjData) -> None:
        pass
        # if not self.robot.is_done:
        #     self.robot.step()