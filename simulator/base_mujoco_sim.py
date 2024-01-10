
import mujoco as mj
from abc import abstractmethod, abstractproperty
from typing import Type
import mujoco.viewer
from mujoco.viewer import Handle
from threading import Lock, Thread
import time

class BaseMuJuCoSim:
    def __init__(self):
        pass

    @property
    @abstractmethod
    def args(self):
        self._args

    @args.setter
    def args(self, args) -> None:
        self._args = args

    @property
    def _scene_path(self) -> str:
        return self._args.scene_path

    @property
    def _data_lock(self) -> Lock:
        return Lock()

    @abstractmethod
    def controller_callback(self, model: mj.MjModel, data: mj.MjData) -> None:
        pass

    @abstractmethod
    def keyboard_callback(self, key):
        pass

    def viewer_callback(self):
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

    # Runs GLFW main loop
    def run(self):
        self.viewer_thrd = Thread(target=self.viewer_callback, daemon=True)
        self.viewer_thrd.daemon = True
        self.viewer_thrd.start()
        input()
        print("done simulating...")

    def _get_mj_model(self) -> mj.MjModel:
        return mj.MjModel.from_xml_path(filename=self._scene_path)
    
    def _get_mj_data(self, model) -> mj.MjData:
        return mj.MjData(model)
    
    def _get_mj_camera(self) -> mj.MjData:
        return mj.MjvCamera()
    
    def _get_mj_options(self) -> mj.MjvOption:
        return mj.MjvOption()
    
    def _get_mj_window(self) -> Handle:
        return mujoco.viewer.launch_passive(self._model, self._data,key_callback=self.keyboard_callback)
    
    def _get_mj_scene(self, maxgeom:int = 10000) -> mj.MjvScene:
        return mj.MjvScene(self._model, maxgeom=maxgeom)