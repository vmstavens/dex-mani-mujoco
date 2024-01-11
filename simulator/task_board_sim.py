import mujoco as mj
import mujoco.viewer
import mujoco_py
from mujoco.glfw import glfw
from simulator.base_mujoco_sim import BaseMuJuCoSim
from robots import Robot, UR10e, HandE

class TaskBoardSim(BaseMuJuCoSim):
    def __init__(self, args):
        self.args = args
        self._model   = self._get_mj_model()
        self._data    = self._get_mj_data(self._model)
        self._camera  = self._get_mj_camera()
        self._options = self._get_mj_options()
        self._window  = self._get_mj_window()
        self._scene   = self._get_mj_scene()

        self._arm = UR10e(self._model, self._data, args)
        self._gripper = HandE(self._model, self._data, args)

        self.robot = Robot(arm=self._arm, gripper=self._gripper, args=args)

        mj.set_mjcb_control(self.controller_callback)

    # # Handles keyboard button events to interact with simulator
    def keyboard_callback(self, key):
        if key == glfw.KEY_SPACE:
            pass

    # Defines controller behavior
    def controller_callback(self, model: mj.MjModel, data: mj.MjData) -> None:
        return