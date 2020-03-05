import rospy
from std_msgs.msg import Float32MultiArray, UInt8MultiArray
import numpy as np

from cornernet_lite_ros.msg import object_info

def talker():

    obj_info = object_info()

    array_tlx = [502.25250244140625, 501.92431640625, 278.2569580078125, 23.033042907714844]
    array_tly = [139.4517364501953, 324.0793151855469, 277.4591369628906, 300.5567321777344]
    array_brx = [636.787353515625, 637.0100708007812, 435.83642578125, 131.19334411621094]
    array_bry = [189.5880889892578, 373.9225769042969, 333.73876953125, 341.2981262207031]
    array_trrafic_pval = [0.6367402672767639, 0.5848375558853149, 0.5164691209793091, 0.40139734745025635]
    array_color_label = [1, 0, 1, 2]
    array_Pedestrian_pval = [0.8, 0.3, 0.5, 0.7]

    obj_info.top_left_x = array_tlx
    obj_info.top_left_y = array_tly
    obj_info.bottom_right_x = array_brx
    obj_info.bottom_right_y = array_bry
    obj_info.traffic_pval = array_trrafic_pval
    obj_info.color_label = array_color_label
    obj_info.pedestrian_pval = array_Pedestrian_pval

    rospy.init_node("talker")
    pub = rospy.Publisher('/obj_info', object_info, queue_size=1)
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        pub.publish(obj_info)
        rate.sleep()

if __name__ == '__main__':
    talker()
    rospy.spin()



