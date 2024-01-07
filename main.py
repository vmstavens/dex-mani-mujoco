#!/bin/python3
import mujoco_py
from simulator import GLWFSim

import spatialmath as sm

import argparse

def main():
    parser = argparse.ArgumentParser(description='MuJoCo simulation of dexterous manipulation, grasping and tactile perception.') 
    parser.add_argument('--scene_path', default="objects/shadow_ur/scene.xml")
    parser.add_argument('--config_dir', default="config/")
    parser.add_argument('--sh_chirality', default="rh")

    args, _ = parser.parse_known_args()

    print(" > Loaded configs:")
    for key, value in vars(args).items():
        print(f'\t{key:20}{value}')

    sim = GLWFSim(args)
    sim.run()

if __name__ == "__main__":
    main()