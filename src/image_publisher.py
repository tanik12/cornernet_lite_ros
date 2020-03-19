#!/usr/bin/env python
# # # -*- coding: utf-8 -*-
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import sys
import cv2

def camera_open():
    try:
        for num in range(10):
            cap = cv2.VideoCapture(num)
            if cap.isOpened():
                print("カメラデバイスが見つかりました。番号は" + str(num) + "番です。")
                return cap
    except:
        print("カメラデバイスが見つかりませんでした。")
        sys.exit()

def image_capture():
    rospy.init_node('pub_image_node')
    #rospy.init_node('publisher_image_sample', anonymous=True)
    pub = rospy.Publisher('image_data', Image, queue_size=10)
    # read image
    cap = camera_open()   
    cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # カメラ画像の縦幅を720に設定
    while not rospy.is_shutdown():
        if cap.isOpened()== False:
            print("Error!!! Camera don't open!")
            break    
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()
        if not ret:
            print("画像の取得に失敗しました。")
            continue

        # make bridge
        bridge = CvBridge()
        msg = bridge.cv2_to_imgmsg(frame, encoding="rgb8")
        
        pub.publish(msg)

if __name__ == '__main__':
    try:
        image_capture()
    except rospy.ROSInterruptException:
        pass
