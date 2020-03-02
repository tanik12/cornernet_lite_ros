import cv2
import numpy as np
import os

#original code

#def draw_bboxes(image, bboxes, font_size=0.5, thresh=0.35, colors=None):
###def draw_bboxes(image, bboxes, font_size=0.5, thresh=0.35, colors=None):
###    """Draws bounding boxes on an image.
###
###    Args:
###        image: An image in OpenCV format
###        bboxes: A dictionary representing bounding boxes of different object
###            categories, where the keys are the names of the categories and the
###            values are the bounding boxes. The bounding boxes of category should be
###            stored in a 2D NumPy array, where each row is a bounding box (x1, y1,
###            x2, y2, score).
###        font_size: (Optional) Font size of the category names.
###        thresh: (Optional) Only bounding boxes with scores above the threshold
###            will be drawn.
###        colors: (Optional) Color of bounding boxes for each category. If it is
###            not provided, this function will use random color for each category.
###
###    Returns:
###        An image with bounding boxes.
###    """
###
###    image = image.copy()
###
###    for cat_name in bboxes:
###        keep_inds = bboxes[cat_name][:, -1] > thresh
###        cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
###        
###        if colors is None:
###            color = np.random.random((3, )) * 0.6 + 0.4
###            color = (color * 255).astype(np.int32).tolist()
###        else:
###            color = colors[cat_name]
###
###        for bbox in bboxes[cat_name][keep_inds]:
###            bbox = bbox[0:4].astype(np.int32)
###            if bbox[1] - cat_size[1] - 2 < 0:
###                cv2.rectangle(image,
###                    (bbox[0], bbox[1] + 2),
###                    (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
###                    color, -1
###                )
###                cv2.putText(image, cat_name,
###                    (bbox[0], bbox[1] + cat_size[1] + 2),
###                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
###                )
###            else:
###                cv2.rectangle(image,
###                    (bbox[0], bbox[1] - cat_size[1] - 2),
###                    (bbox[0] + cat_size[0], bbox[1] - 2),
###                    color, -1
###                )
###                cv2.putText(image, cat_name,
###                    (bbox[0], bbox[1] - 2),
###                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
###                )
###            cv2.rectangle(image,
###                (bbox[0], bbox[1]),
###                (bbox[2], bbox[3]),
###                color, 2
###            )
###    return image

def draw_bboxes(image, bboxes, font_size=0.5, thresh=0.35, colors=None):
    image = image.copy()
    name_dict = {"traffic_signal_blue":"ts_blue", "traffic_signal_red":"ts_red", "traffic_signal_yellow":"ts_yellow", "traffic_signal_unknown":"ts_unknown",
                 "pedestrian_signal_blue":"ps_blue", "pedestrian_signal_red":"ps_red", "pedestrian_signal_unknown":"ps_unknown"}
    
    for cat_name in bboxes:
        if len(bboxes[cat_name]) == 0:
            print(cat_name + "のbboxに関する情報がありませんでした。")
            continue

        if colors is None:
            color = np.random.random((3, )) * 0.6 + 0.4
            color = (color * 255).astype(np.int32).tolist()
        else:
            color = colors[cat_name]

        for bbox in bboxes[cat_name]:
            color_name = np.array(bbox).copy()
            bbox = np.array(bbox[0:4]).astype(np.int32)

            obj_name = name_dict[cat_name + "_" + color_name[5]]
            cat_size  = cv2.getTextSize(obj_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]

            if bbox[1] - cat_size[1] - 2 < 0:
                cv2.rectangle(image,
                    (bbox[0], bbox[1] + 2),
                    (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                    color, -1
                )
                cv2.putText(image, obj_name,
                    (bbox[0], bbox[1] + cat_size[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
                )
            else:
                cv2.rectangle(image,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2),
                    color, -1
                )
                cv2.putText(image, obj_name,
                    (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
                )
            cv2.rectangle(image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color, 2
            )
    return image

def extract_specific_object(image, bboxes, count=1, image_name=None, thresh=0.35, flag=False):
    if flag:
        img2 = image.copy()
        _, traffic_signal_dir, pedestrian_signal_dir = check_dir()
        exstract_list = ["traffic signal", "pedestrian signal"]

        for item in exstract_list:
            idx = bboxes[item][:, -1] > thresh
        
            #error処理もちゃんと入れること
            for bbox in bboxes[item][idx]:
                #intじゃないとエラーが出る。
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                
                trm_img = img2[y1:y2,x1:x2]
        
                try:
                    #書き出し
                    if item == "traffic signal":
                        cv2.imwrite(traffic_signal_dir + '/%06.f.jpg' % count, trm_img)
                    elif item == "pedestrian signal":
                        cv2.imwrite(pedestrian_signal_dir + '/%06.f.jpg' % count, trm_img)
                except:
                    print("書き込み失敗")
        return None, None
    else:
        img2 = image.copy()
        idx_traffic = bboxes["traffic signal"][:, -1] > thresh
        idx_pdstrn = bboxes["pedestrian signal"][:, -1] > thresh
        #error処理もちゃんと入れること
        bboxes_traffic = bboxes["traffic signal"][idx_traffic]
        bboxes_pdstrn = bboxes["pedestrian signal"][idx_pdstrn]

        return bboxes_traffic, bboxes_pdstrn            

def check_dir():
    current_path = os.getcwd()
    save_parent_dir = current_path + "/trim_img/"
    save_child_dir_1 = current_path + "/trim_img/traffic_signal"
    save_child_dir_2 = current_path + "/trim_img/pedestrian_signal"
    
    if not os.path.isdir(save_parent_dir):
        os.mkdir(save_parent_dir)
    if not os.path.isdir(save_child_dir_1):
        os.mkdir(save_child_dir_1)
    if not os.path.isdir(save_child_dir_2):
        os.mkdir(save_child_dir_2)

    return save_parent_dir, save_child_dir_1, save_child_dir_2