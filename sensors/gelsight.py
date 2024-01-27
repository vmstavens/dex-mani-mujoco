

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from threading import Lock, Thread
import cv2
import os
from .gelsight_utils import (
    gkern2D,
    gauss_noise,
    add_overlay,
    tangent
)

class GelSightMini:
    def __init__(self, args, cam_name:str) -> None:
                #  rgb_img_topic:str = "/mj/cam_left_img_rgb", 
                #  depth_img_topic:str = "/mj/cam_left_img_depth", 
                #  tac_img_topic:str = "/gelsight/tactile_image"
        
        self._args = args
        self._bridge = CvBridge()
        self._name = "gelsight"
        self._cam_name = cam_name

        self._pub_freq = self._args.gelsight_pub_freq
        self._pkg_path = os.path.dirname(os.path.abspath(__file__))
        self._background_img = cv2.imread(self._pkg_path + '/assets/background_gelsight2017.jpg')
        
        self._rgb_img_topic = f"/mj/{self._cam_name}_img_rgb"
        self._depth_img_topic = f"/mj/{self._cam_name}_img_depth"
        self._tac_img_topic = f"/mj/{self._cam_name}_img_gelsight"

        self._sub_rpg     = rospy.Subscriber(self._rgb_img_topic,   Image, self._get_rgb_img, queue_size=1)
        self._sub_depth   = rospy.Subscriber(self._depth_img_topic, Image, self._get_depth_img, queue_size=1)
        self._pub_tac_img = rospy.Publisher(self._tac_img_topic, Image, queue_size=1)

        self._rate = rospy.Rate(self._pub_freq)

        self._img: np.ndarray = None
        self._dimg: np.ndarray = None
        self._tac_img: np.ndarray = None

        # TODO : Fix these params
        # constants from: https://github.com/danfergo/gelsight_simulation/tree/master
        self._min_depth = 0.026  # distance from the image sensor to the rigid glass outer surface
        self._ELASTOMER_THICKNESS = 0.004 # m
        self._kernel_1_sigma = 7
        self._kernel_1_kernel_size = 21
        self._kernel_2_sigma = 9
        self._kernel_2_kernel_size = 52
        self._Ka = 0.8
        self._Ks = None
        self._Kd = None
        self._t = 3
        self._texture_sigma = 0.00001
        self._px2m_ratio = 5.4347826087e-05
        
        self._default_ks = 0.15
        self._default_kd = 0.5
        self._default_alpha = 5

        self._max_depth = self._min_depth + self._ELASTOMER_THICKNESS
        self._light_sources = [
            {'position': [0, 1, 0.25],  'color': (255, 255, 255), 'kd': 0.6, 'ks': 0.5},  # white, top
            {'position': [-1, 0, 0.25], 'color': (255, 130, 115), 'kd': 0.5, 'ks': 0.3},  # blue, right
            {'position': [0, -1, 0.25], 'color': (108, 82, 255),  'kd': 0.6, 'ks': 0.4},  # red, bottom
            {'position': [1, 0, 0.25],  'color': (120, 255, 153), 'kd': 0.1, 'ks': 0.1},  # green, left
        ]

        self._pub_lock = Lock()
        self._pub_thrd = Thread(target=self._pub_gelsight)
        self._pub_thrd.daemon = True
        self._pub_thrd.start()

    @property
    def name(self) -> str:
        return self._name

    @property
    def pub_freq(self) -> float:
        return self._pub_freq

    @property
    def img(self) -> np.ndarray:
        return self._img

    @property
    def depth_img(self) -> np.ndarray:
        return self._dimg

    @property
    def tac_img(self) -> np.ndarray:
        return self._tac_img

    def _get_rgb_img(self, img_msg: Image) -> None:
        self._img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    def _get_depth_img(self, img_msg: Image) -> None:
        img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="32FC1")
        img[np.isnan(img)] = np.inf
        self._dimg = img

    def _generate_gelsight_img(self, obj_depth, return_depth=False):
        # print('-----------> ', np.shape(obj_depth))
        # cv2.imwrite('object_depth.png', obj_depth)
        not_in_touch, in_touch = self.segments(obj_depth)
        protrusion_depth = self.protrusion_map(obj_depth, not_in_touch)
        elastomer_depth = self.apply_elastic_deformation(protrusion_depth, not_in_touch, in_touch)

        textured_elastomer_depth = gauss_noise(elastomer_depth, self._texture_sigma)

        out = self._Ka * self._background_img
        out = add_overlay(out, self.internal_shadow(protrusion_depth), (0.0, 0.0, 0.0))

        T = tangent(textured_elastomer_depth / self._px2m_ratio)
        # show_normalized_img('tangent', T)
        for light in self._light_sources:
            ks = light['ks'] if 'ks' in light else self._default_ks
            kd = light['kd'] if 'kd' in light else self._default_kd
            alpha = light['alpha'] if 'alpha' in light else self._default_alpha
            out = add_overlay(out, self.phong_illumination(T, light['position'], kd, ks, alpha), light['color'])

        kernel = gkern2D(3, 1)
        out = cv2.filter2D(out, -1, kernel)

        # cv2.imshow('tactile img', out)
        # cv2.imwrite('tactile_img.png', out)
        #
        if return_depth:
            return out, elastomer_depth
        return out

    def _pub_gelsight(self) -> None:
        while not rospy.is_shutdown():

            if self._img is None or self._dimg is None:
                continue

            # get new images
            with self._pub_lock:

                self._tac_img = self._generate_gelsight_img(self._dimg)

                # images to messages
                # gelsight_image: Image = self._bridge.cv2_to_imgmsg(self._dimg, encoding="bgr8")
                gelsight_image: Image = self._bridge.cv2_to_imgmsg(self._tac_img, encoding="bgr8")
                # gelsight_image: Image = self._bridge.cv2_to_imgmsg(self._tac_img, encoding="passthrough")


                # publish messages
                self._pub_tac_img.publish( gelsight_image )
                # self._pub_tac_img.publish( gelsight_image )

            self._rate.sleep()

    def phong_illumination(self, T: np.ndarray, source_dir: np.ndarray, kd: float, ks: float, alpha: float):
        """
        Apply Phong illumination model to calculate the reflected light intensity.

        The Phong reflection model combines diffuse and specular reflection components.

        Parameters:
        - T (numpy.ndarray): Surface normals of the object.
        - source_dir (numpy.ndarray): Direction vector of the light source.
        - kd (float): Diffuse reflection coefficient.
        - ks (float): Specular reflection coefficient.
        - alpha (float): Shininess or specular exponent.

        Returns:
        - numpy.ndarray: Resultant illumination intensity for each pixel.
        """
        # Calculate the dot product between surface normals and light source direction
        dot = np.dot(T, np.array(source_dir)).astype(np.float64)

        # Compute diffuse reflection component
        difuse_l = dot * kd
        difuse_l[difuse_l < 0] = 0.0  # Ensure non-negative values

        # Compute reflection vector R and the view vector V
        dot3 = np.repeat(dot[:, :, np.newaxis], 3, axis=2)
        R = 2.0 * dot3 * T - source_dir
        V = [0.0, 0.0, 1.0]

        # Compute specular reflection component
        spec_l = np.power(np.dot(R, V), alpha) * ks

        # Combine diffuse and specular components to get the final illumination
        return difuse_l + spec_l

    def apply_elastic_deformation_gauss(self, protrusion_depth, not_in_touch, in_touch):
        """
        Apply elastic deformation to an input depth map.

        Parameters:
        - protrusion_depth (numpy.ndarray): Input depth map.
        - not_in_touch (numpy.ndarray): Mask indicating areas not in touch.
        - in_touch (numpy.ndarray): Mask indicating areas in touch.

        Returns:
        - numpy.ndarray: Deformed depth map.
        """
        kernel = gkern2D(15, 7)
        deformation = self._max_depth - protrusion_depth

        for i in range(5):
            deformation = cv2.filter2D(deformation, -1, kernel)

        return 30 * -deformation * not_in_touch + (protrusion_depth * in_touch)

    def apply_elastic_deformation(self, protrusion_depth, not_in_touch, in_touch):
        """
        Apply a more complex version of elastic deformation to an input depth map.

        Parameters:
        - protrusion_depth (numpy.ndarray): Input depth map.
        - not_in_touch (numpy.ndarray): Mask indicating areas not in touch.
        - in_touch (numpy.ndarray): Mask indicating areas in touch.

        Returns:
        - numpy.ndarray: Deformed depth map.
        """

        protrusion_depth = - (protrusion_depth - self._max_depth)
        kernel = gkern2D(self._kernel_1_kernel_size, self._kernel_1_sigma)
        deformation = protrusion_depth

        deformation2 = protrusion_depth
        kernel2 = gkern2D(self._kernel_2_kernel_size, self._kernel_2_sigma)

        for i in range(self._t):
            deformation_ = cv2.filter2D(deformation, -1, kernel)
            r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
            deformation = np.maximum(r * deformation_, protrusion_depth)

            deformation2_ = cv2.filter2D(deformation2, -1, kernel2)
            r = np.max(protrusion_depth) / np.max(deformation2_) if np.max(deformation2_) > 0 else 1
            deformation2 = np.maximum(r * deformation2_, protrusion_depth)

        deformation_v1 = self.apply_elastic_deformation_gauss(protrusion_depth, not_in_touch, in_touch)

        for i in range(self._t):
            deformation_ = cv2.filter2D(deformation2, -1, kernel)
            r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
            deformation2 = np.maximum(r * deformation_, protrusion_depth)

        deformation_x = 2 * deformation - deformation2

        return self._max_depth - deformation_x

    def protrusion_map(self, original: np.ndarray, not_in_touch: np.ndarray) -> np.ndarray:
        """
        Generate a protrusion map based on the original depth map and regions that are not in touch.

        Parameters:
        - original (numpy.ndarray): Original depth map.
        - not_in_touch (numpy.ndarray): Binary map indicating regions not in touch.

        Returns:
        - numpy.ndarray: Protrusion map where values exceeding `max_depth` are clamped to `max_depth`.
        """
        protrusion_map = np.copy(original)
        protrusion_map[not_in_touch >= self._max_depth] = self._max_depth
        return protrusion_map

    def segments(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Segment the depth map into regions in touch and regions not in touch based on the max depth.

        Parameters:
        - depth_map (numpy.ndarray): Depth map.

        Returns:
        - Tuple[numpy.ndarray, numpy.ndarray]: Two binary maps representing regions not in touch and in touch, respectively.
        """
        not_in_touch = np.copy(depth_map)
        not_in_touch[not_in_touch < self._max_depth] = 0.0
        not_in_touch[not_in_touch >= self._max_depth] = 1.0

        in_touch = 1 - not_in_touch

        return not_in_touch, in_touch

    def internal_shadow(self, elastomer_depth: np.ndarray) -> np.ndarray:
        """
        Generate an internal shadow map based on the elastomer depth.

        Parameters:
        - elastomer_depth (numpy.ndarray): Elastomer depth map.

        Returns:
        - numpy.ndarray: Internal shadow map indicating regions of the elastomer that are shadowed.
        """
        elastomer_depth_inv = self._max_depth - elastomer_depth
        elastomer_depth_inv = np.interp(elastomer_depth_inv, (0, self._ELASTOMER_THICKNESS), (0.0, 1.0))
        return elastomer_depth_inv


import argparse

def main() -> None:

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
    parser.add_argument('--gelsight_pub_freq',  type=float, default=1.0, help="camera image height.")

    args, _ = parser.parse_known_args()

    print(" > Loaded configs:")
    for key, value in vars(args).items():
        print(f'\t{key:20}{value}')
    gs = GelSightMini(args=args)
    gs._pub_gelsight()
    rospy.spin()

if __name__ == "__main__":
    main()