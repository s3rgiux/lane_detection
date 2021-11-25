#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import numpy as np
import cv2
import lanes
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time


class LaneDetect:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        print("created")
        
    def callback(self, data):
        print("entered")
        try:
            print("frame_conv")
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        print("frame")
        self.extract_and_print_lines(cv_image)
        print("finishe")


print("passed while")

def main(args):
    print("enetred main")
    obj_lane = LaneDetect()
    while not rospy.is_shutdown():
        time.sleep(0.02)
    #obj_lane.extract_and_print_lines()


if __name__ == '__main__':
    main(sys.argv)
