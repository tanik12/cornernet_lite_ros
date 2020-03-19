import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from core.detectors import CornerNet_Saccade
from core.vis_utils import draw_bboxes_org

detector = CornerNet_Saccade()
image    = cv2.imread("demo.jpg")

bboxes = detector(image)
image  = draw_bboxes_org(image, bboxes)
cv2.imwrite("demo_out.jpg", image)
