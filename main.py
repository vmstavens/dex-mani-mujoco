#!/usr/bin/env python
import mujoco_py
from simulator import BaseMuJuCoSim, PickNPlaceSim, TaskBoardSim, WiremanipulationSim

import spatialmath as sm

import argparse

def main():
    parser = argparse.ArgumentParser(description='MuJoCo simulation of dexterous manipulation, grasping and tactile perception.') 
    # parser.add_argument('--scene_path',         type=str,   default="objects/scenes/scene_task_board.xml")
    # parser.add_argument('--scene_path',         type=str,   default="objects/scenes/scene_pick_n_place.xml")
    parser.add_argument('--scene_path',         type=str,   default="objects/scenes/scene_wire_manipulation.xml")
    parser.add_argument('--config_dir',         type=str,   default="config/")
    parser.add_argument('--sh_chirality',       type=str,   default="rh")
    parser.add_argument('--trajectory_timeout', type=float, default=5.0, help="the trajectory execution timeout in seconds.")
    parser.add_argument('--sim_name',           type=str,   default="node_name", help="MuJoCo simulation ros node name.")
    parser.add_argument('--robot_pub_freq',     type=float, default=1.0, help="The publish frequency of robot information. If set to -1, publish at maximum speed")
    parser.add_argument('--camera_pub_freq',    type=float, default=1.0, help="The publish frequency of robot information. If set to -1, publish at maximum speed")
    parser.add_argument('--cam_width',          type=int,   default=640, help="camera image width.")
    parser.add_argument('--cam_height',         type=int,   default=480, help="camera image height.")

    args, _ = parser.parse_known_args()

    print(" > Loaded configs:")
    for key, value in vars(args).items():
        print(f'\t{key:20}{value}')

    scenes = ["scene_pick_n_place.xml", "scene_task_board.xml", "scene_wire_manipulation.xml"]
    sim_indx = [ int(i) for i,x in enumerate(scenes) if x in args.scene_path][0]
    simulations = [PickNPlaceSim, TaskBoardSim, WiremanipulationSim]

    sim:BaseMuJuCoSim = simulations[sim_indx](args)
    sim.run()

if __name__ == "__main__":
    main()  