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
from typing import Union, Dict
import mujoco as mj
def main():

    # xml_path = "objects/universal_robots_ur10e/scene.xml"
    # xml_path = "objects/universal_robots_ur10e/scene.xml"
    # xml_path = "objects/scene.xml"
    xml_path = "objects/shadow_ur/scene.xml"
    # xml_path = "objects/shadow_hand/scene_right.xml"
    # xml_path = "/home/vims/git/dex-mani-mujoco/objects/universal_robots_ur10e/scene.xml"
    signs_filepath = 'data/signs.json'
    ctrl_limits_filepath = "data/ctrl_limits.csv"

    print(f"{xml_path=}")
    print(f"{signs_filepath=}")
    print(f"{ctrl_limits_filepath=}")

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