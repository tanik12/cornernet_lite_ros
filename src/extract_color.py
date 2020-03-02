import numpy as np
import glob
import os
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# 赤色の検出
def detect_red_color(img, hsv):
    tmp = np.array([])

    # 赤色のHSVの値域1
    hsv_min = np.array([0,144,153])
    hsv_max = np.array([8,194,252])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得 
    b = masked_img.T[0].flatten().mean()
    g = masked_img.T[1].flatten().mean()
    r = masked_img.T[2].flatten().mean()

    tmp = np.append(tmp, r)
    tmp = np.append(tmp, g)
    tmp = np.append(tmp, b)

    return mask, masked_img, tmp

# 青色の検出
def detect_blue_color(img, hsv):
    tmp = np.array([])

    # 青色のHSVの値域1
    hsv_min = np.array([84, 163, 108])
    hsv_max = np.array([104, 203, 155])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得 
    b = masked_img.T[0].flatten().mean()
    g = masked_img.T[1].flatten().mean()
    r = masked_img.T[2].flatten().mean()

    tmp = np.append(tmp, r)
    tmp = np.append(tmp, g)
    tmp = np.append(tmp, b)

    return mask, masked_img, tmp

# 緑色の検出
def detect_green_color(img, hsv):
    tmp = np.array([])

    # 色のHSVの値域1
    hsv_min = np.array([30, 64, 0])
    hsv_max = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得 
    b = masked_img.T[0].flatten().mean()
    g = masked_img.T[1].flatten().mean()
    r = masked_img.T[2].flatten().mean()

    tmp = np.append(tmp, r)
    tmp = np.append(tmp, g)
    tmp = np.append(tmp, b)

    return mask, masked_img, tmp

# 黄色の検出
def detect_yellow_color(img, hsv):
    tmp = np.array([])

    # 色のHSVの値域1
    hsv_min = np.array([20, 80, 10])
    hsv_max = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得 
    b = masked_img.T[0].flatten().mean()
    g = masked_img.T[1].flatten().mean()
    r = masked_img.T[2].flatten().mean()

    tmp = np.append(tmp, r)
    tmp = np.append(tmp, g)
    tmp = np.append(tmp, b)

    return mask, masked_img, tmp

def color_info(img):
    color_arr = np.array([])         

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得 
    b = img.T[0].flatten().mean()
    g = img.T[1].flatten().mean()
    r = img.T[2].flatten().mean()
    
    # BGRからHSVに変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # HSV平均値を取得
    # flattenで一次元化しmeanで平均を取得 
    h = hsv.T[0].flatten().mean()
    s = hsv.T[1].flatten().mean()
    v = hsv.T[2].flatten().mean()

    color_arr = np.append(color_arr, r)
    color_arr = np.append(color_arr, g)
    color_arr = np.append(color_arr, b)
    color_arr = np.append(color_arr, h)
    color_arr = np.append(color_arr, s)
    color_arr = np.append(color_arr, v)

    return color_arr, hsv, img 

def extract_color_info(images):
    feat_list = []
    hist_clr = np.array([])
    print(len(images))
    
    for img in images:
        tmpB = []
        img = img[0]

        try:
            img =  cv2.resize(img,(50, 50)) #あとで直す
        except:
            print("リサイズに失敗しました。")
            print("画像の中身：", img, "画像のshape", img.shape)

        try:
            color_arr, hsv, img = color_info(img)
            hist_r, hist_g, hist_b = color_hist(img)
        except:
            print("画像の読み込みに失敗しました。")
            continue
            #error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'ってエラーが出ることが多かった。
            #color_info関数のhsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)の箇所でエラーが起こっている。

        hist_clr = np.vstack((hist_r, hist_g))
        hist_clr = np.vstack((hist_clr, hist_b))
        red_mask, red_masked_img, avg_red_masked_img = detect_red_color(img, hsv)
        blue_mask, bule_masked_img, avg_blue_masked_img = detect_blue_color(img, hsv)
        green_mask, green_masked_img, avg_green_masked_img = detect_green_color(img, hsv)
        yellow_mask, yellow_masked_img, avg_yellow_masked_img = detect_yellow_color(img, hsv)
        
        ######
        #要検討
        #sum_array = avg_red_masked_img + avg_blue_masked_img + avg_green_masked_img + yellow_masked_img  
        sum_array = (avg_red_masked_img + (avg_blue_masked_img + avg_green_masked_img)/2 + avg_yellow_masked_img) / 3 
        ######
    
        tmpB.append(red_masked_img)
        tmpB.append(bule_masked_img)
        tmpB.append(green_masked_img)
        tmpB.append(yellow_masked_img)
        tmpB.append(color_arr)
        tmpB.append(sum_array)
        tmpB.append(hist_clr)

        feat_list.append(tmpB)
    
    return feat_list

#colorヒストグラムの取得
def color_hist(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    hist_r, bins = np.histogram(r.ravel(),256,[0,256])
    hist_g, bins = np.histogram(g.ravel(),256,[0,256])
    hist_b, bins = np.histogram(b.ravel(),256,[0,256])

    return hist_r, hist_g, hist_b

if __name__ == "__main__":
    img = cv2.imread("cut.jpg", cv2.COLOR_BGR2RGB)
    res_data = extract_color_info(img)

    print("res_data: ", res_data)
    print("res_data.shape: ", len(res_data))

    print("(r, g, b, h, s, v): ", res_data[4])
    ###cv2.imwrite("red_masked_img.png", res_data[0])
    ###cv2.imwrite("bule_masked_img.png", res_data[1])
    ###cv2.imwrite("green_masked_img.png", res_data[2])
    ###cv2.imwrite("yellow_masked_img.png", res_data[3])
