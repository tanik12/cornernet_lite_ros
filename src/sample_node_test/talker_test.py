import rospy
from std_msgs.msg import Float32MultiArray, UInt8MultiArray
import numpy as np

from cornernet_lite_ros.msg import object_info, bbox, bboxes

def talker():

    test = [[245.64630126953125, 58.95439910888672, 468.280029296875, 124.51933288574219, 0.6187193393707275, "red", 0.888], 
            [380.8486022949219, 322.2468566894531, 468.5032958984375, 452.5641174316406, 0.4995148777961731, "blue", 0.888], 
            [380.8486022949219, 12.2468566894531, 454.36016845703125, 348.0196838378906, 0.46585899591445923, "rrr", 0.888]]

    #test222 = [[245.64630126953125, 58.95439910888672, 468.280029296875, 124.51933288574219, 0.6187193393707275, 1.0, 0.888]]

    mass_arr = []
    for item in test:
        obj_info = object_info()
        chunk_arr = []
        obj_info.xmin = item[0]
        obj_info.ymin = item[1]
        obj_info.xmax = item[2]
        obj_info.ymax = item[3]
        obj_info.probability = item[4]
        obj_info.color_class = item[5]
        obj_info.color_probability = item[6]

        chunk_arr.append(obj_info)
        bbox_info = bbox(bounding_box=chunk_arr)
        mass_arr.append(bbox_info)

    print("===========")
    print(mass_arr)
    bboxes_info = bboxes(bounding_boxes=mass_arr)

    rospy.init_node("talker")
    #pub = rospy.Publisher('/bboxes_info', bbox, queue_size=1) #importした時の名前
    pub2 = rospy.Publisher('/bboxes_info', bboxes, queue_size=1) #importした時の名前
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        pub2.publish(bboxes_info)
        rate.sleep()

if __name__ == '__main__':
    talker()
    rospy.spin()



