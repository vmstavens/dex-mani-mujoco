
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

def cb_func(data: Image):
    cv_bridge = CvBridge()
    cv_image = cv_bridge.imgmsg_to_cv2(data, "passthrough")
    # print(cv_image)
    cv2.waitKey(1)
    # cv2.imshow("test",  cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR ))
    print(np.mean(cv_image))

def main():
    rospy.init_node("test_sensor")
    sub = rospy.Subscriber("/mj/cam_left_img_depth",Image,callback=cb_func, queue_size=1)
    # sub = rospy.Subscriber("/mj/cam_left_img_rgb",Image,callback=cb_func, queue_size=1)
    rospy.spin()

if __name__ == "__main__":
    main()