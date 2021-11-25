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

    def extract_and_print_lines(self, frame):
        #cap = image #cv2.VideoCapture("/home/keisuu/catkin_ws/src/lane_detection/scripts/test_video.mp4")
        #print("opened ", cap.isOpened())
        #while(cap.isOpened()):
        #frame = image
        canny_image = lanes.canny(frame)
        cropped_image = lanes.region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=10,maxLineGap=5)
        averaged_lines = lanes.average_slope_intercept(frame,lines)
        #print("***")
        #print("a0", averaged_lines[0])
        #print("a1", averaged_lines[1])
        line_image = lanes.display_lines(frame,averaged_lines)
        combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)    # Imposing the line_image on the original image

        cv2.imshow('Result',combo_image)
        #cv2.waitKey(1)

print("passed while")

def main(args):
    print("enetred main")
    obj_lane = LaneDetect()
    while not rospy.is_shutdown():
        #laser_merge_obj.pubStatus()
        #print("alive")
        time.sleep(0.02)
    #obj_lane.extract_and_print_lines()


if __name__ == '__main__':
    main(sys.argv)
