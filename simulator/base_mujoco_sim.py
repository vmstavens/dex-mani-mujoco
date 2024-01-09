
import mujoco as mj
from abc import abstractmethod
from typing import Type
import mujoco.viewer
from mujoco.viewer import Handle
from threading import Lock

class BaseMuJuCoSim:
    def __init__(self):
        pass

    @property
    def _args(self):
        pass
    @property
    def _scene_path(self) -> str:
        pass
    @property
    def _mj_model(self) -> mj.MjModel:
        pass
    @property
    def _mj_data(self) -> mj.MjData:
        pass
    @property
    def _mj_camera(self) -> mj.MjvCamera:
        pass
    @property
    def _mj_options(self) -> mj.MjvOption:
        pass

    @property
    def _window(self) -> Handle:
        return mujoco.viewer.launch_passive(self._model, self._data,key_callback=self._keyboard_cb)

    @property
    def _scene(self) -> mj.MjvScene:
        return mj.MjvScene(self._mj_model, maxgeom=10000)

    @property
    def _data_lock(self) -> Lock:
        return Lock()

    @abstractmethod
    def controller_fn(self, model: mj.MjModel, data: mj.MjData) -> None:
        pass

    @abstractmethod
    def keyboard_callback(self, key):
        pass

    def viewer_callback(self) -> None:
        pass

    def run(self) -> None:
        pass