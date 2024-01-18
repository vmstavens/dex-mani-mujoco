import mujoco as mj
import numpy as np
import cv2
from typing import Tuple

class Camera:
	def __init__(self, model: mj.MjModel, data: mj.MjData, args, cam_name:str = "") -> None:
		
		self._args = args
		self._model = model
		self._data = data
		self._camera = mj.MjvCamera()
		self._options = mj.MjvOption()
		self._scene = mj.MjvScene(self._model, maxgeom=10_000)
		self._pert = mj.MjvPerturb()
		
		self._cam_name = cam_name
		self._camera.type = mj.mjtCamera.mjCAMERA_FIXED
		self._camera.fixedcamid = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_CAMERA, self._cam_name)
		
		self.x_res = self._args.cam_x_res
		self.y_res = self._args.cam_y_res

		self._viewport = mj.MjrRect(0, 0, self.x_res, self.y_res)

	def shoot(self,save_path:str = "test_img.png") -> Tuple[np.ndarray, np.ndarray]:

		context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150)
		mj.mjv_updateScene(self._model, self._data, self._options, self._pert, self._camera, mj.mjtCatBit.mjCAT_ALL, self._scene)
		mj.mjr_render(self._viewport, self._scene, context)

		image         = np.empty((self.y_res, self.x_res, 3), dtype=np.uint8)
		depth_hat_buf = np.empty((self.y_res, self.x_res, 1),dtype=np.float32)

		mj.mjr_readPixels(image, depth_hat_buf, self._viewport, context)

		# OpenGL renders with inverted y axis
		img         = image.squeeze()
		# image         = np.flip(image, axis=0).squeeze()
		dimg = depth_hat_buf.squeeze() * 255.0
		# depth_hat_buf = np.flip(depth_hat_buf, axis=0).squeeze()

		cv2.imwrite("/home/vims/git/dex-mani-mujoco/simulator/test.png", cv2.cvtColor( image, cv2.COLOR_RGB2BGR ))
		cv2.imwrite("/home/vims/git/dex-mani-mujoco/simulator/test_d.png",depth_hat_buf)
		return img, dimg
	
	def test(self):
		# very much testing
		# see: /home/vims/git/mujoco_camera_depth_test/camera_depth_test.py
		zfar  = self._model.vis.map.zfar * self._model.stat.extent
		znear = self._model.vis.map.znear * self._model.stat.extent
		# depth_hat = ogl_zbuf_inv(depth_hat_buf, znear, zfar)
		def opengl_depth_to_meters(depth_value, near, far):
			"""
			Convert OpenGL depth units to meters.

			Parameters:
			- depth_value: The depth value from OpenGL (typically in the range [0, 1]).
			- near: The near clipping plane value of the camera.
			- far: The far clipping plane value of the camera.

			Returns:
			- Depth in meters.
			"""
			depth_meters = (far - near) / (depth_value * (far / (near - far)) + near / (near - far))
			return depth_meters
