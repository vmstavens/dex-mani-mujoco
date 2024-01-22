import mujoco as mj
import os
import numpy as np
from typing import Tuple
from datetime import datetime
from sensor_msgs.msg import Image
import cv2
import rospy
from threading import Lock, Thread
from utils.geometry import geometry
from cv_bridge import CvBridge

import mujoco.viewer as viewer

import time
import OpenGL.GL as gl

class Camera:
    """Class representing a camera in a Mujoco simulation.

    Note: This class assumes the use of the Mujoco physics engine and its Python bindings.
    """
    def __init__(self, args, model, data, data_lock, cam_name:str = "", save_dir="data/img/"):
        """Initialize Camera instance.

        Args:
        - args: Arguments containing camera width and height.
        - model: Mujoco model.
        - data: Mujoco data.
        - cam_name: Name of the camera.
        - save_dir: Directory to save captured images.
        """
        self._args = args
        self._cam_name = cam_name
        self._model = model
        self._data = data
        self._save_dir = save_dir + self._cam_name + "/"
        self._data_lock = data_lock

        self._width = self._args.cam_width
        self._height = self._args.cam_height

        self._options = mj.MjvOption()
        self._pert = mj.MjvPerturb()
        self._scene = mj.MjvScene(self._model, maxgeom=10_000)
        # self._context  = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150)
        # self._context  = mj.GLContext(self.width,self.heigth)
        self._context  = mj.MjrContext()
        # self._context = context
        # self._viewer = viewer.Handle(
        #     sim=self._model,
        #     cam=self._camera,
        #     opt=self._options,
        #     pert=self._pert,
        #     user_scn=self._scene)


        self._camera = mj.MjvCamera()
        self._camera.type = mj.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_CAMERA, self.name)
        
        self._viewport = mj.MjrRect(0, 0, self._width, self._height)

        self._img  = np.empty((self._height, self._width, 3), dtype=np.uint8)
        self._dimg = np.empty((self._height, self._width, 1),dtype=np.float32)

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        self._pub_rgb = rospy.Publisher(f"mj/{self.name}_img_rpg",Image,queue_size=1)
        self._pub_depth = rospy.Publisher(f"mj/{self.name}_img_depth", Image,queue_size=1)

        self._pub_lock = Lock()
        self._pub_thrd = Thread(target=self._pub_cam)
        self._pub_thrd.daemon = True
        self._pub_thrd.start()

    @property
    def context(self):
        return self._context
    
    @context.setter
    def context(self, new_context):
        print("updated context")
        self._context = new_context

    @property
    def heigth(self) -> int:
        """Return the height of the camera image."""
        return self._height
    
    @property
    def width(self) -> int:
        """Return the width of the camera image."""
        return self._width
    
    @property
    def save_dir(self) -> str:
        """Return the directory to save captured images."""
        return self._save_dir

    @property
    def name(self) -> str:
        """Return the name of the camera."""
        return self._cam_name

    @property
    def matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
        """Compute the component matrices for constructing the camera matrix.

        This property calculates and returns the image matrix, focal matrix, and translation matrix,
        essential for constructing the camera matrix. The camera matrix represents the transformation
        from 3D world coordinates to 2D image coordinates.

        If the camera is a 'free' camera (fixedcamid == -1), the position and orientation are obtained
        from the scene data structure. For stereo cameras, the left and right channels are averaged.
        Note: The method `self.update()` is called to ensure the correctness of `scene.camera` contents.

        If the camera is not 'free', the position, rotation, and field of view (fov) are extracted
        from the Mujoco data and model.

        Returns:
        A tuple containing the image matrix, focal matrix, and translation matrix of the camera.
        """

        camera_id = self._camera.fixedcamid
        if camera_id == -1:
            # If the camera is a 'free' camera, we get its position and orientation
            # from the scene data structure. It is a stereo camera, so we average over
            # the left and right channels. Note: we call `self.update()` in order to
            # ensure that the contents of `scene.camera` are correct.
            self.update()
            pos = np.mean([camera.pos for camera in self._scene.camera], axis=0)
            z = -np.mean([camera.forward for camera in self._scene.camera], axis=0)
            y = np.mean([camera.up for camera in self._scene.camera], axis=0)
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
        # homogeneous transformation matrix (4,4)
        T = translation @ rotation

        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * self._height / 2.0
        focal_matrix = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
        # Image matrix (3x3).
        img_matrix = np.eye(3)
        img_matrix[0, 2] = (self._width - 1) / 2.0
        img_matrix[1, 2] = (self._height - 1) / 2.0

        return (img_matrix, focal_matrix, T)

    def _pub_cam(self) -> None:
        rate = rospy.Rate(1)
        # rate = rospy.Rate(self._args.camera_pub_freq)  # Set the publishing rate (1 Hz in this example)
        while not rospy.is_shutdown():

            # get new images
            with self._pub_lock:
                self.shoot(autosave=False)
                bridge = CvBridge()
                image_msg_rgb = bridge.cv2_to_imgmsg(self._img, encoding="bgr8")
                image_msg_depth = bridge.cv2_to_imgmsg(self._dimg, encoding="32FC1")

                self._pub_rgb.publish( image_msg_rgb )
                self._pub_depth.publish( image_msg_depth )
                self.save("test")
            rate.sleep()

    def shoot(self, autosave: bool = True) -> None:
        """Captures a new image from the camera."""

        with self._data_lock:
            self._context = mj.MjrContext()
            # mj.mjv_updateScene(self._model, self._data, mj.MjvOption(), mj.MjvPerturb(), self._camera, mj.mjtCatBit.mjCAT_ALL, self._scene)
            mj.mjr_render(self._viewport, self._scene, self._context)
            mj.mjv_updateScene(self._model, self._data, self._options, self._pert, self._camera, mj.mjtCatBit.mjCAT_ALL, self._scene)
            # mj.mjr_render(self._viewport, self._scene, mj.MjrContext())
            # mj.mjr_readPixels(self._img, self._dimg, self._viewport, mj.MjrContext())
            print(type(self._img),type(self._dimg),type(self._viewport),type(self._context))
            mj.mjr_readPixels(self._img, self._dimg, self._viewport, self._context)
            # mj.mjr_drawPixels(self._img, self._dimg, self._viewport, self._context)
            print(np.mean(self._img), np.mean(self._dimg))

            temp_img = self._img.copy()
            temp_depth = self._dimg.copy()

            # OpenGL renders with inverted y axis
            temp_img  = temp_img.squeeze()
            temp_depth = temp_depth.squeeze()
            
            # in order to convert opengl units to meters
            extent = self._model.stat.extent
            near = self._model.vis.map.znear * extent
            far = self._model.vis.map.zfar * extent
            temp_depth = np.flipud(near / (1 - temp_depth * (1 - near / far)))

            self._dimg = temp_depth.copy()
            self._img = temp_img.copy()
            self.save("test")
            if autosave:
                self.save()

    def save(self, img_name:str = "") -> None:
        """Saves the captured image and depth information.

        Args:
        - img_name: Name for the saved image file.
        """
        if img_name == "":
            cv2.imwrite(self._save_dir + f"{datetime.now()}_rgb.png", cv2.cvtColor( self._img, cv2.COLOR_RGB2BGR ))
            cv2.imwrite(self._save_dir + f"{datetime.now()}_depth.png",self._dimg)
        else:
            cv2.imwrite(self._save_dir + f"{img_name}._rgb.png", cv2.cvtColor( self._img, cv2.COLOR_RGB2BGR ))
            cv2.imwrite(self._save_dir + f"{img_name}_depth.png",self._dimg)

