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
from controllers.hand_controller import HandController
from controllers.arm_controller import ArmController
from utils.control import read_sign_transitions, read_ctrl_limits
from typing import Union, Dict
import mujoco as mj
def main():

    scene_path = "objects/shadow_ur/scene.xml"
    hand_cfg_dir = "config/shadow_hand.json"
    arm_cfg_dir = "config/ur10e.json"

    print(f"{scene_path=}")
    print(f"{hand_cfg_dir=}")
    print(f"{arm_cfg_dir=}")

    sim = GLWFSim(
        shadow_hand_xml_filepath=scene_path,
        hand_controller=HandController(config_dir = hand_cfg_dir,ctrl_n_steps=10),
        arm_controller=HandController(config_dir = arm_cfg_dir,ctrl_n_steps=10),
        trajectory_steps=3,
        cam_verbose=False,
        sim_verbose=True
    )
    sim.run()

if __name__ == "__main__":
    main()