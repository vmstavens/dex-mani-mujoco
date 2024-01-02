
from utils.control import read_config
from typing import List

class ArmController:
    def __init__(self, ctrl_n_steps:int = 100, config_dir: str = "configs/hand.json") -> None:
        self._config_dir = config_dir
        # self._cfgs = None
        self._cfgs = read_config(self._config_dir)
        self._ctrl_n_steps = ctrl_n_steps
        self._traj = []

    # @property
    # def is_done(self) -> bool:
    #     return self._is_done()
    
    # def _is_done(self) -> bool:
    #     return True if len(self._traj) == 0 else False
    
    # def cfg_to_q(self, cfg:str) -> List:
    #     try:
    #         cfg_json = self._cfgs[cfg]
    #         q = cfg_json["wr"] + cfg_json["ff"] + cfg_json["mf"] + cfg_json["rf"] + cfg_json["lf"] + cfg_json["th"]
    #         return q
    #         # return self._cfgs[cfg]
    #     except KeyError:
    #         print("Wrong cfg string, try one of the following:")
    #         for k,v in self._cfgs.items():
    #             print(f"\t{k}")