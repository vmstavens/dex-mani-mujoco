import mujoco_py
import mujoco as mj
from mujoco.glfw import glfw
from simulator.base_mujoco_sim import BaseMuJuCoSim
import argparse

class TestSim(BaseMuJuCoSim):
    def __init__(self, args):
        self.args = args
        self._model   = self._get_mj_model()
        self._data    = self._get_mj_data(self._model)
        self._camera  = self._get_mj_camera()
        self._options = self._get_mj_options()
        self._window  = self._get_mj_window()
        self._scene   = self._get_mj_scene()

        mj.set_mjcb_control(self.controller_callback)

    # # Handles keyboard button events to interact with simulator
    def keyboard_callback(self, key):
        if key == glfw.KEY_SPACE:
            pass

    # Defines controller behavior
    def controller_callback(self, model: mj.MjModel, data: mj.MjData) -> None:
        return
    

def main():
    parser = argparse.ArgumentParser(description='MuJoCo simulation of dexterous manipulation, grasping and tactile perception.') 
    # parser.add_argument('--scene_path', type=str, default="objects/hand-e/scene_hand-e.xml")
    parser.add_argument('--scene_path', type=str, default="objects/ur10e_hand-e/ur10e_hand-e.xml")
    args, _ = parser.parse_known_args()
    ts = TestSim(args=args)
    ts.run()

if __name__ == "__main__":
    main()