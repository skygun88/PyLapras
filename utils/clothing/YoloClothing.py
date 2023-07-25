import os
import sys
import cv2
import glob
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# torch.cuda.empty_cache()
# print(device, '------------------------------------')

_DIRPATH = os.path.abspath(os.path.dirname(__file__))


class YoloClothing:
    def __init__(self, dataset='df2'):
        self.model, self.classes, self.colors = self.load_model(dataset)
        # print(self.classes)

    def load_model(self, dataset='df2'):

        #YOLO PARAMS
        yolo_df2_params = {   "model_def" : os.path.join(_DIRPATH, "yolo/df2cfg/yolov3-df2.cfg"),
        "weights_path" : os.path.join(_DIRPATH, "yolo/weights/yolov3-df2_15000.weights"),
        "class_path": os.path.join(_DIRPATH, "yolo/df2cfg/df2.names"),
        # "conf_thres" : 0.15,
        "conf_thres" : 0.2,
        # "conf_thres" : 0.4,
        "nms_thres" :0.2,
        "img_size" : 416,
        "device" : device}

        yolo_modanet_params = {   "model_def" : os.path.join(_DIRPATH, "yolo/modanetcfg/yolov3-modanet.cfg"),
        "weights_path" : os.path.join(_DIRPATH, "yolo/weights/yolov3-modanet_last.weights"),
        "class_path":os.path.join(_DIRPATH, "yolo/modanetcfg/modanet.names"),
        "conf_thres" : 0.2,
        "nms_thres" :0.2,
        "img_size" : 416,
        "device" : device}


        if dataset == 'df2': #deepfashion2
            yolo_params = yolo_df2_params
        elif dataset == 'modanet':
            yolo_params = yolo_modanet_params


        #Classes
        classes = load_classes(yolo_params["class_path"])

        #Colors
        cmap = plt.get_cmap("rainbow")
        colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])


        detectron = YOLOv3Predictor(params=yolo_params)
        return detectron, classes, colors


    def detect_clothes(self, img):
        # print(self.model)
        detections = self.model.get_detections(img)
        return detections

    def detect_clothes_class(self, img):
        detections = self.model.get_detections(img)
        return [(int(detection[5]), detection[4]) for detection in detections]

