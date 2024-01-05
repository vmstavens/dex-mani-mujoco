

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

from typing import List

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

    @abstractproperty
    def joint_names(self) -> List[str]:
        pass