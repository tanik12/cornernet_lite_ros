import rospy
from cornernet_lite_ros.msg import object_info, bbox, bboxes

def callback(data):
    for box in data.bounding_boxes:
        obj_info = box.bounding_box[0]

        obj_class   = obj_info.object_class
        xmin        = obj_info.xmin
        ymin        = obj_info.ymin
        xmax        = obj_info.xmax
        ymax        = obj_info.ymax
        prob        = obj_info.probability
        color_class = obj_info.color_class
        color_obj   = obj_info.color_probability
        
        rospy.loginfo(
            "obj_name: {}, xmin: {}, xmax: {} ymin: {}, ymax: {}, prob: {}, color_class: {}, color_obj: {}".format(
                obj_class, xmin, xmax, ymin, ymax, prob, color_class, color_obj
            )
        )
        
def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/bboxes_info", bboxes, callback)
    rospy.spin()
        
if __name__ == '__main__':
    listener()