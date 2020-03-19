#!/usr/bin/env /home/gisen/.pyenv/versions/anaconda3-2019.10/envs/CornerNet_Lite/bin/python
import rospy
from sensor_msgs.msg import Image
from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes, extract_specific_object, trimming
from cornernet_lite_ros.msg import object_info, bbox, bboxes
from cv_bridge import CvBridge, CvBridgeError

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import glob
import os

from extract_color import extract_color_info              #色情報を抽出するための関数
from  model import dir_check, load_model, color_inference #色認識モデル関連の処理
import numpy as np

class ObjectDetectionCornerNetLite:
    def __init__(self):
        self.detector = CornerNet_Squeeze()
        self.current_path = os.getcwd()
        self.model_dirpath = self.current_path + "/src/cornernet_lite_ros/src/model" #rosrunでpathを解決するための一時的な対策
        #self.model_dirpath = self.current_path + "/model" #cornernet_lite_ros/src/ の中でdemo_cam_Squeeze.pyを実行したいのであればこれを使う。
        self.clf = load_model(self.model_dirpath)
        self.count = 1

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("image_data", Image, self.callback)
        self.arg = sys.argv
        self.data_path = "/home/gisen/Documents/own_dataset/traffic_light_dataset/traffic_light/*" #保存済みの車両・歩行者信号機の写真がある場所を指定。
        self.imgs_path = glob.glob(self.data_path)
        self.imshow_flag = False
        self.save_flag = False

        self.trm_imges_dict    = {}
        self.bboxes_dict       = {}
        self.result            = {}

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            self.run(self.arg, self.detector, cv_image)
        except CvBridgeError as e:
            print(e)

    #real time で物体認識処理をする箇所。Falseにするとトリミング画像を保存しない
    def obj_inference(self, detector, image, count=1, flag=False):
        bboxes_traffic = ""
        bboxes_pdstrn = ""
        bboxes = detector(image)

        #real timeでトリミング画像をsaveする場合はTrue。しない場合はFalse。
        #動くけど似たような写真を保存しないように修正する必要あり。色認識モデルを学習させる用のもの。
        if flag:
            #bboxes_traffic, bboxes_pdstrn = extract_specific_object(image, bboxes, count, flag=flag)
            _, _ = extract_specific_object(image, bboxes, self.count, flag=flag)
            self.count += 1
            return _, _, _, _
        else:
            bboxes_traffic, bboxes_pdstrn = extract_specific_object(image, bboxes, count, flag=flag)
            return image, bboxes, bboxes_traffic, bboxes_pdstrn
            ###image  = draw_bboxes(image, bboxes)

    #物体認識と色認識結果をpublishする処理
    #argに引数がなければデフォルトで"camera"を代入。
    def run(self, arg, detector, frame):
        if len(arg) == 1:
            arg.append("camera")
            
        pub = rospy.Publisher('/bboxes_info', bboxes, queue_size=1)

        if arg[1] == "camera":
            if self.save_flag:
                _, _, _, _ = self.obj_inference(detector, frame, flag=self.save_flag)
                return
            else:
                #現状、[変数名：bounding_boxes]を使っていないが将来的には使用するので残しておく。
                image, bounding_boxes, bboxes_traffic, bboxes_pdstrn = self.obj_inference(detector, frame, flag=self.save_flag)

            self.trm_imges_dict, self.bboxes_dict = trimming(image, bboxes_traffic, bboxes_pdstrn)

            #交通信号機や歩行者信号機が1つでもあったらpublishする。
            if len(self.trm_imges_dict["traffic_signal"]) + len(self.trm_imges_dict["pedestrian_signal"]) > 0:
                mass_list4pub = []
                print("信号機の数: ", len(self.bboxes_dict["traffic_signal"]), "歩行者信号機の数: ", len(self.bboxes_dict["pedestrian_signal"])) 
                for obj_name in ["traffic_signal", "pedestrian_signal"]:
                    mass_list4draw = []
                    
                    #bboxes_arrはndarrayであり、その中にはlistが格納されていることに注意。ここは後ほど修正。
                    #color_infoesはlistを入れ子にしている。
                    bboxes_arr = self.bboxes_dict[obj_name]
                    color_infoes = extract_color_info(self.trm_imges_dict[obj_name]) #色認識機に入れる入力
                    #print("(r, g, b, h, s, v): ", color_infoes[0][4]) #Debug用.row:bounding boxの何番目か, col:色情報

                    for input_data, bbox_data in zip(color_infoes, bboxes_arr):
                        obj_info = object_info()
                        chunk_list = []

                        input_data = np.array(input_data[4])
                        pred, label_name = color_inference(input_data,  self.clf)

                        bbox_data = bbox_data.tolist()

                        #bboxes_info
                        obj_info.object_class      = obj_name
                        obj_info.xmin              = bbox_data[0]
                        obj_info.ymin              = bbox_data[1]
                        obj_info.xmax              = bbox_data[2]
                        obj_info.ymax              = bbox_data[3]
                        obj_info.probability       = bbox_data[4]
                        obj_info.color_class       = label_name
                        obj_info.color_probability = 9.99999 #後で確率値を出すようにする.少々お待ちください。[変数名:pre]が入る。

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
    
            if self.imshow_flag:
                # 加工なし画像を表示する
                cv2.imshow('Raw Frame', image)
    
                # キー入力でqを押したら終了する
                k = cv2.waitKey(1)
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    sys.exit()

        #保存済みの画像をトリミングしてsaveする機能。色認識モデルを学習させる用のもの。
        elif arg[1] == "save":
            self.save_flag = True
            for img_path in self.imgs_path:
                img = cv2.imread(img_path)
                _, _, _, _ = self.obj_inference(detector, img, self.count, flag=self.save_flag)
                self.count += 1
        else:
            print("コマンドライン引数の第2引数は、camera or save のどれかを指定してください。")
            sys.exit()

if __name__ == "__main__":
    rospy.init_node("cornernet_lite_node")
    instance = ObjectDetectionCornerNetLite()
    rospy.spin()
