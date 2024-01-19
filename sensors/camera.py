import mujoco as mj
import numpy as np
import cv2
from typing import Tuple

from .cam_utils import ogl_zbuf_default_inv
import spatialmath as sm
# from dm_control.mujoco.engine import Physics
import dm_control.mujoco.engine as dmje

class Camera:
	def __init__(self, args, context, cam_name:str = "") -> None:

		self._args        = args
		self._cam_name    = cam_name
		self._scene_path  = self._args.scene_path
		self._model       = mj.MjModel.from_xml_path(filename=self._scene_path)
		self._data        = mj.MjData(self._model)
		self._scene       = mj.MjvScene(self._model, maxgeom=10_000)
		self._options     = mj.MjvOption()
		self._pertubation = mj.MjvPerturb()
		self._camera      = mj.MjvCamera()
		self._camera_id   = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_CAMERA, self._cam_name)
		self._camera.fixedcamid = self._camera_id
		self._camera.type = mj.mjtCamera.mjCAMERA_FIXED

		self._pertubation.active = 0
		self._pertubation.select = 0
		self._width        = self._args.cam_x_res
		self._height       = self._args.cam_y_res
		self._rect = mj.MjrRect(0, 0, self._width, self._height)

		self._img  = np.empty((self._height, self._width, 3), dtype=np.uint8)
		self._dimg = np.empty((self._height, self._width), dtype=np.float32)

		self._rgb_buffer = np.empty((self._height, self._width, 3), dtype=np.uint8)
		self._depth_buffer = np.empty((self._height, self._width), dtype=np.float32)
		self._context = context
		# self._context = mj.MjrContext(self._model)
		# self._context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150)

	@property
	def matrix(self):
		"""
		Returns the 3x4 camera matrix.
		For a description of the camera matrix see, e.g.,
		https://en.wikipedia.org/wiki/Camera_matrix.
		For a usage example, see the associated test.
		"""
		image, focal, rotation, translation = self.matrices()
		return image @ focal @ rotation @ translation

	def matrices(self) -> Tuple[sm.SE3,sm.SE3,sm.SE3]:
		"""Computes the component matrices used to compute the camera matrix.

		Returns:
		returns a tuple of [image matrix (not an image), focal matrix, extrinsic parameters SE3]
		"""
		camera_id = self._camera.fixedcamid
		if camera_id == -1:
			# If the camera is a 'free' camera, we get its position and orientation
			# from the scene data structure. It is a stereo camera, so we average over
			# the left and right channels. Note: we call `self.update()` in order to
			# ensure that the contents of `scene.camera` are correct.
			self.update()
			pos = np.mean([camera.pos for camera in self.scene.camera], axis=0)
			z = -np.mean([camera.forward for camera in self.scene.camera], axis=0)
			y = np.mean([camera.up for camera in self.scene.camera], axis=0)
			rot = np.vstack((np.cross(y, z), y, z))
			fov = self._model.vis.global_.fovy
		else:
			pos = self._data.cam_xpos[camera_id]
			rot = self._data.cam_xmat[camera_id].reshape(3, 3).T
			fov = self._model.cam_fovy[camera_id]

		# Translation matrix (4x4).
		translation = np.eye(4)
		translation[0:3, 3] = -pos
		# Rotation matrix (4x4).
		rotation = np.eye(4)
		rotation[0:3, 0:3] = rot
		# Focal transformation matrix (3x4).
		focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * self.height / 2.0
		focal_matrix = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
		# Image matrix (3x3).
		image_matrix = np.eye(3)
		image_matrix[0, 2] = (self.width - 1) / 2.0
		image_matrix[1, 2] = (self.height - 1) / 2.0

		T = sm.SE3()
		T.t = translation
		T.R = rotation

		return (image_matrix, focal_matrix, T)

	def shoot(self,save_path:str = "test_img.png") -> Tuple[np.ndarray, np.ndarray]:
		depth = True
		print(f"{depth=}")
		mj.mjr_readPixels(self._rgb_buffer if not depth else None,
						self._depth_buffer if depth else None, self._rect,
						self._context)
		# with self._context.gl.make_current() as ctx:
		# 	ctx.call(self._render_on_gl_thread, depth=True)
		# return
		print("0")
		# Render scene and text overlays, read contents of RGB or depth buffer.
		print("1")
		print("2")
		# Convert from [0 1] to depth in meters, see links below:
		# http://stackoverflow.com/a/6657284/1461210
		# https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
		extent = self._model.stat.extent
		near = self._model.vis.map.znear * extent
		far = self._model.vis.map.zfar * extent
		self._dimg = np.flipud(near / (1 - self._depth_buffer * (1 - near / far)))
		self._img = np.flipud(self._rgb_buffer)
		print("3")

		print("in shoot")
		# img = self._camera.render(  scene_option=self._options, depth=False)
		# print(img)
		# dimg = self._camera.render( scene_option=self._options, depth=True)

		# context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150)
		# mj.mjv_updateScene(self._model, self._data, self._options, self._pert, self._camera, mj.mjtCatBit.mjCAT_ALL, self._scene)
		# mj.mjr_render(self._viewport, self._scene, context)

		# image         = np.empty((self.y_res, self.x_res, 3), dtype=np.uint8)
		# depth_hat_buf = np.empty((self.y_res, self.x_res, 1), dtype=np.float32)

		# # depth_hat_buf = np.empty((self.y_res, self.x_res, 1),dtype=np.float32)

		# mj.mjr_readPixels(image, depth_hat_buf, self._viewport, context)

		# # OpenGL renders with inverted y axis
		# img         = image.squeeze()
		# # image         = np.flip(image, axis=0).squeeze()
		# dimg = depth_hat_buf.squeeze()
		
		# # depth is a float array, in meters.
		# depth = self._physics.render(depth=True)
		# # Shift nearest values to the origin.
		# depth -= depth.min()
		# # Scale by 2 mean distances of near rays.
		# depth /= 2*depth[depth <= 1].mean()
		# # Scale to [0, 255]
		# pixels = 255*np.clip(depth, 0, 1)
		# pixels = pixels.astype(np.uint8)

		# dimg = depth_hat_buf.squeeze() * 255.0

		# zfar  = self._model.vis.map.zfar * self._model.stat.extent
		# znear = self._model.vis.map.znear * self._model.stat.extent
		# dimg = ogl_zbuf_default_inv(depth_hat_buf, znear, zfar)
		# dimg = ogl_zbuf_inv(depth_hat_buf, znear, zfar)
		# depth_hat_buf = np.flip(depth_hat_buf, axis=0).squeeze()

		cv2.imwrite("/home/vims/git/dex-mani-mujoco/simulator/test.png", cv2.cvtColor( self._img, cv2.COLOR_RGB2BGR ))
		cv2.imwrite("/home/vims/git/dex-mani-mujoco/simulator/test_d.png",self._dimg)
		print("saved images...")
		return self._img, self._dimg
	
	def linearize_depth(self, depth):
		depth_img = self.z_near * self.z_far * self.extent / (self.z_far - depth * (self.z_far - self.z_near))
		return depth_img

	def set_camera_intrinsics(self, model, camera, viewport):
		fovy = model.cam_fovy[camera.fixedcamid] / 180 * np.pi / 2
		self.f = viewport.height / (2 * np.tan(fovy))
		self.cx = viewport.width / 2
		self.cy = viewport.height / 2

	def get_RGBD_buffer(self, model, viewport, context):
		self.color_buffer = np.zeros((viewport.height * viewport.width * 3,), dtype=np.uint8)
		self.depth_buffer = np.zeros((viewport.height * viewport.width,), dtype=np.float32)
		context.read_pixels(self.color_buffer, self.depth_buffer, viewport)
		
		self.extent = model.stat.extent
		self.z_near = model.vis.map.znear
		self.z_far = model.vis.map.zfar

		img_size = (viewport.width, viewport.height)
		self.color_image = cv2.cvtColor(self.color_buffer.reshape((img_size[1], img_size[0], 3), order='C'), cv2.COLOR_BGR2RGB)
		self.color_image = np.flipud(self.color_image)
		
		self.depth_image = self.linearize_depth(np.flipud(self.depth_buffer).reshape(img_size))
		return self.depth_buffer