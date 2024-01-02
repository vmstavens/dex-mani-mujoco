import numpy as np
from controllers.controller import Controller
from typing import List, Dict 


class ExpertController(Controller):
    def __init__(self, ctrl_limits: List[np.ndarray], signs: Dict[str, List[np.ndarray]]= {'rest': (),'sign': (), 'order': ()}):
        super().__init__(ctrl_limits=ctrl_limits)

        self._signs = signs
        self._ctrl_transition_iter = None
        self._sign = "rest"

    def _set_sign(self, sign: str):
        # print(self._signs)
        self._sign = sign
        self._ctrl_transition_iter = iter(self._signs[sign])
    
    def sign_to_q(self, sign: str) -> np.ndarray:
        return self._signs[sign]

    def _get_next_control(self, sign: str, order: int) -> np.ndarray or None:
        print(self._ctrl_transition_iter)
        return next(self._ctrl_transition_iter, None)

    @property
    def sign(self) -> str:
        return self._sign