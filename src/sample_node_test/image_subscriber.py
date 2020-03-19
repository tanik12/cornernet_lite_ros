import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("image_data", Image, self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      cv2.imshow('Raw Frame', cv_image)
      cv2.waitKey(3)
    except CvBridgeError as e:
      print(e)

def main():
   ic = image_converter()
   rospy.init_node('image_converter', anonymous=True)
   try:
       rospy.spin()
   except KeyboardInterrupt:
       print("Shutting down")

   cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
