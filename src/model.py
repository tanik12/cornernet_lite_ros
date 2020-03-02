import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pandas as pd
import glob
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

label_dict = {0:"blue", 1:"red", 2:"yellow", 3:"unknown"}

def dir_check(model_dirpath):
    if os.path.exists(model_dirpath):
        print("Directory exists to save model file!!!")
    else:
        print("Directory did not exist to save model file...")
        sys.exit()

def load_model(model_dirpath):
    try:
        with open(model_dirpath + "/model.pickle", mode='rb') as fp:
            clf = pickle.load(fp)
            return clf
    except FileNotFoundError as e:
        print("Do not exist model file! Please make model file.", e)
        sys.exit()

def inference(x_train, model):
    x_train = x_train.reshape(1, -1)
    pred = model.predict(x_train)
    label_name = label_dict[pred[0]]
    
    return pred, label_name

if __name__ == "__main__":
    current_path = os.getcwd()
    model_dirpath = current_path + "/model"

    #test data
    test_x = np.array([64.4052, 85.112, 87.6772, 102.0968, 64.3176, 89.7904])
    test_y = np.array([0])
    pred_class = inference(test_x, model_dirpath)
    print("予想ラベル出力: ", pred_class)