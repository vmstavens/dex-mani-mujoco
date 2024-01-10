from typing import List, Union, Dict
import os 
import warnings
import mujoco as mj
import numpy as np

from spatialmath import SE3

from utils.sim import (
    read_config,
    save_config, 
    RobotConfig
)

from utils.mj import (
    get_actuator_names,
    get_actuator_value,
    set_actuator_value,
    get_joint_names,
    get_joint_value,
    get_joint_range,
    get_actuator_range,
    is_done_actuator
)

from abc import ABC, abstractmethod, abstractproperty


class BaseRobot(ABC):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def args(self):
        pass

    @property
    def n_actuators(self) -> int:
        return len(self._get_actuator_names())

    @abstractproperty
    def mj_data(self) -> mj.MjData:
        pass

    @abstractproperty
    def mj_model(self) -> mj.MjModel:
        pass

    @abstractmethod
    def _config_to_q(self, config: str) -> List[float]:
        pass

    @abstractmethod
    def set_ee_pose(self, 
            pos: List = [0.5,0.5,0.5], 
            ori: Union[np.ndarray,SE3] = [1,0,0,0], 
            pose: Union[None, List[float], np.ndarray, SE3] = None,
            solution_pool:int = 4,
            ) -> None:
        pass

    @abstractmethod
    def get_ee_pose(self) -> SE3:
        pass

    def set_q(self) -> None:
        pass

    def _set_q(self, q: Union[str,List]) -> None:
        robot_actuator_names = []
        for an in get_actuator_names(self.mj_model):
            if self.name in an:
                robot_actuator_names.append(an)
        for i in range(len(robot_actuator_names)):
            set_actuator_value(data=self.mj_data, q=q[i], actuator_name=robot_actuator_names[i])

    def home(self) -> None:
        self.set_q(q_robot = "home")

    def _get_actuator_values(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self.mj_model):
            if self.name in ac_name:
                result.append( get_actuator_value(self.mj_data, ac_name) )
        return result

    def _get_actuator_names(self) -> List[str]:
        result = []
        for ac_name in get_actuator_names(self.mj_model):
            if self.name in ac_name:
                result.append(ac_name)
        return result

    def _get_joint_names(self) -> List[str]:
        result = []
        for jt_name in get_joint_names(self.mj_model):
            if self.name in jt_name:
                result.append(jt_name)
        return result

    def _get_joint_limits(self, joint_name:str) -> List[float]:
        return get_joint_range(self.mj_model, joint_name=joint_name)

    def _get_actuator_limits(self, actuator_name) -> List[float]:
        return get_actuator_range(self.mj_model, actuator_name=actuator_name)

    def _clamp_q(self, q: List[float]) -> List[float]:
        actuator_names = self._get_actuator_names()
        actuator_limits = [get_actuator_range(self.mj_model, jn) for jn in actuator_names]
        clamped_qs = []
        for i in range(len(q)):
            clamped_q = np.clip(a = q[i], a_min = actuator_limits[i][0], a_max = actuator_limits[i][1])
            clamped_qs.append(clamped_q)

        return clamped_qs

    def _are_done_actuators(self) -> bool:
        actuator_names = self._get_actuator_names()
        joint_names    = self._get_joint_names()
        for i,jn in enumerate(joint_names):
            if not is_done_actuator(data = self.mj_data, joint_name = jn, actuator_name = actuator_names[i]):
                return False
        return True

    def get_q(self) -> RobotConfig:
        robot_joint_names = []
        robot_joint_values = []
        for an in get_joint_names(self.mj_model):
            if an.split("_")[0] == self.name.split("_")[0]:
                robot_joint_names.append(an)
        for ran in robot_joint_names:
            robot_joint_values.append(get_joint_value(self.mj_data, ran))
        rc = RobotConfig(
            joint_values = robot_joint_values,
            joint_names = robot_joint_names
        )
        return rc

    def save_config(self, config_name:str = "placeholder") -> None:
        save_config(
            config_dir  = self.config_dir,
            config      = self.get_q(),
            config_name = config_name
        )

    def _get_config_dir(self) -> str:
        return self.args.config_dir + self.name + ".json"

    def _get_configs(self) -> Dict[str, Dict[str, float]]:
        if not os.path.exists(self.args.config_dir):
            os.makedirs(os.path.dirname(self.args.config_dir), exist_ok=True)
            warnings.warn(f"config_dir {self.args.config_dir} could not be found, create empty config")
        return read_config(self._get_config_dir())