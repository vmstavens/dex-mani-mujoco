#!/bin/python3
import gym
import time
import mujoco_py
from gym import spaces
import numpy as np
import mujoco

import sys
import os

import utils
from simulator import GLWFSim
from controllers.model import ModelController
from controllers.expert import ExpertController
from utils.control import read_sign_transitions, read_ctrl_limits

def main():

    xml_path = "objects/shadow_hand/scene_right.xml"
    signs_filepath = 'data/signs.json'
    ctrl_limits_filepath = "data/ctrl_limits.csv"

    # xml_path = "/home/vims/git/dex-mani-mujoco/mujoco_menagerie/shadow_hand/scene_right.xml"
    signs = read_sign_transitions(json_filepath=signs_filepath)
    ctrl_limits = read_ctrl_limits(csv_filepath=ctrl_limits_filepath)

    sim = GLWFSim(
        shadow_hand_xml_filepath=xml_path,
        hand_controller=ExpertController(ctrl_limits=ctrl_limits,signs=signs),
        trajectory_steps=10,
        cam_verbose=False,
        sim_verbose=True
    )
    sim.run()

if __name__ == "__main__":
    main()

# sign order
# "rest2": [[-1.57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
# [[W1, W2, Th5, Th4, Th3, Th2, Th1, FF4, FF3, FF2, FF1, MF4, MF3, MF2, MF1, RF4, RF3, RF2, RF1, LF, LF, LF, LF, LF]]