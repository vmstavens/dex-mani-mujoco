
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

def cb_func(data: Image):
    cv_bridge = CvBridge()
    cv_image = cv_bridge.imgmsg_to_cv2(data, "passthrough")
    print("----")
    # print(cv_image)
    cv2.waitKey(100)
    print(np.mean(cv_image), np.max(cv_image), np.min(cv_image))
    cv_image = cv2.normalize(cv_image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow("test",  cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR ))
    # print(np.mean(cv_image))

def main():
    # [v] /mj/cam_left_img_depth
    # [v] /mj/cam_left_img_rgb
    # [v] /mj/cam_right_img_depth
    # [v] /mj/cam_right_img_rgb
    # [] /gelsight/tactile_image
    # [] make sure passive depth is equal for gazebo and mujoco

    rospy.init_node("test_sensor")
    sub = rospy.Subscriber("/mj/cam_right_img_depth",Image,callback=cb_func, queue_size=1)
    # sub = rospy.Subscriber("/mj/cam_left_img_depth",Image,callback=cb_func, queue_size=1)
    print("....")
    # sub = rospy.Subscriber("/mj/cam_left_img_rgb",Image,callback=cb_func, queue_size=1)
    rospy.spin()

if __name__ == "__main__":
    main()