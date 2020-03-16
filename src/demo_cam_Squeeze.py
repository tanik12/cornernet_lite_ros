import rospy
from sensor_msgs.msg import Image

from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes, extract_specific_object

from cornernet_lite_ros.msg import object_info, bbox, bboxes
from cv_bridge import CvBridge, CvBridgeError

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

import glob
import os

from extract_color import extract_color_info
from  model import dir_check, load_model, inference
import numpy as np

class ObjectDetectionCornerNetLite:
    def __init__(self):
        self.detector = CornerNet_Squeeze()
        self.current_path = os.getcwd()
        self.model_dirpath = self.current_path + "/model"
        self.clf = load_model(self.model_dirpath)
        self.count = 1

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("image_data", Image, self.callback)
        #self.arg = sys.argv
        self.arg = "camera"
        self.data_path = "/home/gisen/Documents/own_dataset/traffic_light_dataset/traffic_light/*"
        self.imgs_path = self.load_color4train(self.data_path)
        self.save_flag = True

        self.trm_imges_dict    = {}
        self.bboxes_dict       = {}
        self.result            = {}

    def callback(self, data):
        #try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        #print(cv_image)
        #return
        self.run(self.arg, self.detector, cv_image)
                
        #except CvBridgeError as e:
        #    print(e)
    
    def load_color4train(self, data_path):
        imgs_path = glob.glob(data_path)
        return imgs_path
    
    def trimming(self, image, bboxes_traffic, bboxes_pdstrn):
        traffic_trm_imges = []
        pdstrn_trm_imges  = []
        trm_imges_dict    = {}

        if bboxes_traffic.shape[0] > 0:
            try:
                for bbox_traffic in bboxes_traffic:
                    x1 = int(bbox_traffic[0])
                    y1 = int(bbox_traffic[1])
                    x2 = int(bbox_traffic[2])
                    y2 = int(bbox_traffic[3])

                    trm_img = image[y1:y2,x1:x2]
                    traffic_trm_imges.append([trm_img])
            except:
                print("交通信号機のトリミングを試みましたが失敗しました")

        if bboxes_pdstrn.shape[0] > 0:
            try:
                for bbox_pdstrn in bboxes_pdstrn:
                    x1 = int(bbox_pdstrn[0])
                    y1 = int(bbox_pdstrn[1])
                    x2 = int(bbox_pdstrn[2])
                    y2 = int(bbox_pdstrn[3])
                    
                    trm_img = image[y1:y2,x1:x2]
                    pdstrn_trm_imges.append([trm_img])
            except:
                print("歩行者信号機のトリミングを試みましたが失敗しました")

        trm_imges_dict["traffic_signal"]    = traffic_trm_imges
        trm_imges_dict["pedestrian_signal"] = pdstrn_trm_imges
        
        bboxes_dict = {"traffic_signal":bboxes_traffic, "pedestrian_signal":bboxes_pdstrn}

        return trm_imges_dict, bboxes_dict
    
    def run(self, arg, detector, frame):
        print(arg)    
        pub = rospy.Publisher('/bboxes_info', bboxes, queue_size=1)

        if arg == "video" or arg == "camera":              
            image, bounding_boxes, bboxes_traffic, bboxes_pdstrn = self.obj_inference(detector, frame)

            self.trm_imges_dict, self.bboxes_dict = self.trimming(image, bboxes_traffic, bboxes_pdstrn)

            if len(self.trm_imges_dict["traffic_signal"]) + len(self.trm_imges_dict["pedestrian_signal"]) > 0:
                mass_list4pub = []
                print("信号機の数: ", len(self.bboxes_dict["traffic_signal"]), "歩行者信号機の数: ", len(self.bboxes_dict["pedestrian_signal"]))
                for obj_name in ["traffic_signal", "pedestrian_signal"]:
                    mass_list4draw = []
                
                    bboxes_arr = self.bboxes_dict[obj_name]
                    res_data = extract_color_info(self.trm_imges_dict[obj_name])
                    #print("(r, g, b, h, s, v): ", res_data[0][4]) #Debug用

                    for input_data, bbox_data in zip(res_data, bboxes_arr):
                        obj_info = object_info()
                        chunk_list = []

                        input_data = np.array(input_data[4])
                        pred, label_name = inference(input_data,  self.clf)

                        bbox_data = bbox_data.tolist()

                        #bboxes_info
                        obj_info.object_class      = obj_name
                        obj_info.xmin              = bbox_data[0]
                        obj_info.ymin              = bbox_data[1]
                        obj_info.xmax              = bbox_data[2]
                        obj_info.ymax              = bbox_data[3]
                        obj_info.probability       = bbox_data[4]
                        obj_info.color_class       = label_name
                        obj_info.color_probability = 9.99999 #後で確率値を出すようにする

                        chunk_list.append(obj_info)
                        bbox_info = bbox(bounding_box=chunk_list)
                        mass_list4pub.append(bbox_info)

                        chunk_list4draw = [bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3], bbox_data[4], label_name, 9.99999]
                        mass_list4draw.append(chunk_list4draw)
                    
                    bboxes_info = bboxes(bounding_boxes=mass_list4pub)
                    self.result[obj_name] = mass_list4draw

                image  = draw_bboxes(image, self.result)
            
                #print文はデバック用
                print(self.result)
                print("=======================")

                pub.publish(bboxes_info)            

                self.result = {}               
                #del result                        
    
            # 加工なし画像を表示する
            cv2.imshow('Raw Frame', image)

            # キー入力でqを押したら終了する
            k = cv2.waitKey(1)
            if k == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
        else:
            for img_path in self.imgs_path:
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path)
                image, bounding_boxes, _, _ = self.obj_inference(detector, img, self.count, image_name=img_name, flag=self.save_flag)
                self.count += 1
                break
        
    #Falseにするとトリミング画像を保存しない
    def obj_inference(self, detector, image, count=1, image_name=None, flag=False):
        bboxes_traffic = ""
        bboxes_pdstrn = ""
    
        bboxes = detector(image)
        if flag:
            bboxes_traffic, bboxes_pdstrn = extract_specific_object(image, bboxes, count, image_name=image_name, flag=flag)
        else:
            bboxes_traffic, bboxes_pdstrn = extract_specific_object(image, bboxes, count, image_name=image_name, flag=flag)
        
            ###image  = draw_bboxes(image, bboxes)
    
        return image, bboxes, bboxes_traffic, bboxes_pdstrn

if __name__ == "__main__":
    rospy.init_node("cornernet_ros")
    #args = sys.argv
    instance = ObjectDetectionCornerNetLite()
    #instance.run(instance.args[1], instance.detector)
    #instance.run("camera", instance.detector)

    rospy.spin()
