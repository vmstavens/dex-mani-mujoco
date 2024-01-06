
import os
import warnings
import mujoco as mj

from abc import ABC, abstractmethod, abstractproperty
from utils.sim import (
    read_config,
    save_config, 
    config_to_q,
    RobotConfig
)

from utils.mj import (
    get_actuator_names,
    get_actuator_value,
    set_actuator_value,
    get_joint_names,
    get_joint_value,
    is_done_actuator
)

from typing import List, Dict, Any

class BaseRobot(ABC):
    
    def save_config(self) -> None:
        pass

    def step(self) -> None:
        pass

    def _are_done_actuators(self)-> bool:
        for i,jn in enumerate(self.joint_names):
            if not is_done_actuator(self._data,joint_name=jn,actuator_name=self._actuator_names[i]):
                return False
        return True

    def set_q(self) -> None:
        pass

    @property
    def get_config_dir(self) -> str:
        self._config_dir = self.args.config_dir + self._name + ".json"
        if not os.path.exists(self._config_dir):
            os.makedirs(os.path.dirname(self._config_dir), exist_ok=True)
            warnings.warn(f"config_dir {self._config_dir} could not be found, create empty config")
        return self._config_dir

    @abstractproperty
    def joint_names(self) -> List[str]:
        pass
    
    @abstractproperty
    def configs(self) -> Dict[str,float]:
        pass
    
    @abstractproperty
    def config_dir(self) -> str:
        pass

    @abstractproperty
    def q(self) -> List[str]:
        pass

    @abstractproperty
    def args(self) -> Any:
        pass

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def model(self) -> mj.MjModel:
        pass

    @abstractproperty
    def model(self) -> mj.MjModel:
        pass

    @abstractproperty
    def data(self) -> mj.MjModel:
        pass