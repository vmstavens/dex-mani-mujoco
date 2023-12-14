import numpy as np
from controllers.controller import Controller
from typing import List, Dict 


class PoseController():
    def __init__(self, ctrl_limits: List[np.ndarray], signs: Dict[str, List[np.ndarray]]= {'rest': (),'sign': (), 'order': ()}):

        self._signs = signs
        self._ctrl_transition_iter = None
        self._ctrl_limits = ctrl_limits
        self._order = 0
        self._is_done = False
        self._NUM_OF_DOFS = 6

    def _set_sign(self, sign: str):
        self._ctrl_transition_iter = iter(self._signs[sign])

    def _get_next_control(self, sign: str, order: int) -> np.ndarray or None:
        return next(self._ctrl_transition_iter, None)

    @property
    def order(self) -> int:
        return self._order

    @property
    def is_done(self) -> bool:
        return self._is_done

    # Sets the behavior controller to specified sign
    def set_sign(self, sign: str):
        self._order = 0
        self._is_done = False

        self._set_sign(sign=sign)

    # Returns the next control of the transition. The positions are clipped according to control limits just in case
    def get_next_control(self, sign: str) -> np.ndarray or None:
        if self._is_done:
            return None

        next_ctrl = self._get_next_control(sign=sign, order=self._order)

        if next_ctrl is None:
            self._is_done = True
        else:
            assert next_ctrl.shape[0] == self._NUM_OF_DOFS

            self._order += 1

            # for i, (low, high) in enumerate(self._ctrl_limits):
            #     next_ctrl[i] = np.clip(a=next_ctrl[i], a_min=low, a_max=high)
        return next_ctrl
