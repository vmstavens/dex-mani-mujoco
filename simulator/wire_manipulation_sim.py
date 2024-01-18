import mujoco as mj
import mujoco.viewer
import mujoco_py
from mujoco.glfw import glfw
from simulator.base_mujoco_sim import BaseMuJuCoSim
from robots import Robot, UR10e, HandE, Hand2F85
import numpy as np
import cv2

from utils.mj import (
    set_object_pose, 
    set_robot_pose,
    get_joint_names,
)

from sensors import Camera

class WiremanipulationSim(BaseMuJuCoSim):
    def __init__(self, args):
        self.args = args
        self._model   = self._get_mj_model()
        self._data    = self._get_mj_data(self._model)
        self._camera  = self._get_mj_camera()
        self._options = self._get_mj_options()
        self._window  = self._get_mj_window()
        self._scene   = self._get_mj_scene()
        self._pert    = self._get_mj_pertubation()

        self._set_scene()

        self.cam = Camera(self._model, self._data, self._args, cam_name="cam")

        # self._arm = UR10e(self._model, self._data, args)
        self._gripper = HandE(self._model, self._data, args)

        self.robot = Robot(gripper=self._gripper, args=args)

        mj.set_mjcb_control(self.controller_callback)

    def _set_scene(self):
        pass

    # Handles keyboard button events to interact with simulator
    def keyboard_callback(self, key):
        if key == glfw.KEY_SPACE:
            self.cam.shoot()

    # Defines controller behavior
    def controller_callback(self, model: mj.MjModel, data: mj.MjData) -> None:
        pass
        # if not self.robot.is_done:
        #     self.robot.step()

    def test(self):
        # Render the simulated camera
        
        camera = self._get_mj_camera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, 'cam')
        # mj.mjv_updateScene(self._model, self._data, self._options, self._pert, camera, mj.mjtCatBit.mjCAT_ALL, self._scene)
        
        viewport = mj.MjrRect(0, 0, self._x_res, self._y_res)
        ctx = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150)
        # ctx = mujoco.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150)

        mj.mjr_render(viewport, self._scene, ctx)
        image = np.empty((self._y_res, self._x_res, 3), dtype=np.uint8)
        depth_hat_buf = np.empty((self._y_res, self._x_res, 1),dtype=np.float32)
        mj.mjr_readPixels(image, depth_hat_buf, viewport, ctx)

        # OpenGL renders with inverted y axis
        image         = np.flip(image, axis=0).squeeze()
        depth_hat_buf = np.flip(depth_hat_buf, axis=0).squeeze()

        cv2.imwrite("/home/vims/git/dex-mani-mujoco/simulator/test.png", cv2.cvtColor( image,cv2.COLOR_RGB2BGR ))
        cv2.imwrite("/home/vims/git/dex-mani-mujoco/simulator/test_d.png",depth_hat_buf)

        # target_mujoco_id = None
        # for vgeom in scn.geoms:
        # if vgeom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
        #     name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,vgeom.objid)
        #     if name == 'target':
        #     target_mujoco_id = vgeom.segid
        # assert target_mujoco_id is not None
        # else:
        #     renderer.update_scene(d, cam)