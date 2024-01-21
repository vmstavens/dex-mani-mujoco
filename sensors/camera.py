import mujoco as mj
import os
import numpy as np
from typing import Tuple
from datetime import datetime
from sensor_msgs.msg import Image
import cv2
import rospy
from threading import Lock, Thread

class Camera:
    """Class representing a camera in a Mujoco simulation.

    Note: This class assumes the use of the Mujoco physics engine and its Python bindings.
    """
    def __init__(self, args, model, data, cam_name:str = "", save_dir="data/img/"):
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

        self._width = self._args.cam_width
        self._height = self._args.cam_height

        self._options = mj.MjvOption()
        self._pert = mj.MjvPerturb()
        self._scene = mj.MjvScene(self._model, maxgeom=10_000)

        self._camera = mj.MjvCamera()
        self._camera.type = mj.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_CAMERA, self._cam_name)
        
        self._viewport = mj.MjrRect(0, 0, self._width, self._height)

        self._img  = np.empty((self._height, self._width, 3), dtype=np.uint8)
        self._dimg = np.empty((self._height, self._width, 1),dtype=np.float32)

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        rospy.init_node(self.name)

        self._pub_rpg = rospy.Publisher(f"mj/{self.name}_img_rpg",Image)
        self._pub_depth = rospy.Publisher(f"mj/{self.name}_img_depth", Image)
        # self._sub_shoot = rospy.Subscriber(f"mj/{self.name}_shoot", bool, callback=shoot_callback)

        self._pub_lock = Lock()
        self._pub_thrd = Thread(target=self._pub_robot_info)
        self._pub_thrd.daemon = True
        self._pub_thrd.start()

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
        rate = rospy.Rate(self._args.pub_freq)  # Set the publishing rate (1 Hz in this example)

        img = img_msg.data

    def shoot(self, autosave: bool = True) -> None:
        """Captures a new image from the camera."""
        self._context  = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150)
        mj.mjv_updateScene(self._model, self._data, self._options, self._pert, self._camera, mj.mjtCatBit.mjCAT_ALL, self._scene)
        mj.mjr_render(self._viewport, self._scene, self._context)
        mj.mjr_readPixels(self._img, self._dimg, self._viewport, self._context)

        # OpenGL renders with inverted y axis
        self._img  = self._img.squeeze()
        self._dimg = self._img.squeeze()
        
        # in order to convert opengl units to meters
        extent = self._model.stat.extent
        near = self._model.vis.map.znear * extent
        far = self._model.vis.map.zfar * extent
        self._dimg = np.flipud(near / (1 - self._dimg * (1 - near / far)))

        if autosave:
            self.save()

    def save(self, img_name:str = "") -> None:
        """Saves the captured image and depth information.

        Args:
        - img_name: Name for the saved image file.
        """
        if img_name == "":
            cv2.imwrite(self._save_dir + f"{datetime.now()}_rpg.png", cv2.cvtColor( self._img, cv2.COLOR_RGB2BGR ))
            cv2.imwrite(self._save_dir + f"{datetime.now()}_depth.png",self._dimg)
        else:
            cv2.imwrite(self._save_dir + f"{img_name}._rpg.png", cv2.cvtColor( self._img, cv2.COLOR_RGB2BGR ))
            cv2.imwrite(self._save_dir + f"{img_name}_depth.png",self._dimg)

