import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes

detector = CornerNet_Squeeze()
image    = cv2.imread("demo.jpg")

bboxes = detector(image)
print(bboxes)
image  = draw_bboxes(image, bboxes)
cv2.imwrite("demo_out.jpg", image)
