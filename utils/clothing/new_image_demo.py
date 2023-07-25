import os
import sys
import cv2
import glob
import torch
from tqdm import tqdm
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

_DIRPATH = os.path.abspath(os.path.dirname(__file__))

def set_video_writer(cap, out_file_path, write_fps=15):
    return cv2.VideoWriter(out_file_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          write_fps,
                          (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

def load_model(dataset='df2'):

    #YOLO PARAMS
    yolo_df2_params = {   "model_def" : os.path.join(_DIRPATH, "yolo/df2cfg/yolov3-df2.cfg"),
    "weights_path" : os.path.join(_DIRPATH, "yolo/weights/yolov3-df2_15000.weights"),
    "class_path": os.path.join(_DIRPATH, "yolo/df2cfg/df2.names"),
    "conf_thres" : 0.5,
    "nms_thres" :0.4,
    "img_size" : 416,
    "device" : device}

    yolo_modanet_params = {   "model_def" : os.path.join(_DIRPATH, "yolo/modanetcfg/yolov3-modanet.cfg"),
    "weights_path" : os.path.join(_DIRPATH, "yolo/weights/yolov3-modanet_last.weights"),
    "class_path":os.path.join(_DIRPATH, "yolo/modanetcfg/modanet.names"),
    "conf_thres" : 0.5,
    "nms_thres" :0.4,
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


def detect_clothes(detectron, img):
    detections = detectron.get_detections(img)
    return detections

def detect_clothes_class(detectron, img):
    detections = detectron.get_detections(img)
    detections = [(detection[5], detection[4]) for detection in detections]
    return detections

if __name__ == '__main__':
    detectron, classes, colors = load_model()
    img_path = 'PyLapras/utils/clothing/tests/test5.jpg'


    if not os.path.exists(img_path):
        print('Img does not exists..')
        sys.exit()


    img = cv2.imread(img_path)
    detections = detect_clothes(detectron, img)
    detections = detect_clothes_class(detectron, img)

    
    if len(detections) != 0 :
        print(detections)
        # for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                
        #         print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))           

                
                # color = colors[int(cls_pred)]
                
                # color = tuple(c*255 for c in color)
                # color = (.7*color[2],.7*color[1],.7*color[0])       
                    
                # font = cv2.FONT_HERSHEY_SIMPLEX   
            
            
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
                
                # cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
                # y1 = 0 if y1<0 else y1
                # y1_rect = y1-25
                # y1_text = y1-5

                # if y1_rect<0:
                #     y1_rect = y1+27
                #     y1_text = y1+20
                # cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
                # cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)
                
                

                
    # cv2.imshow('Detections',img)
    # img_id = path.split('/')[-1].split('.')[0]
    # cv2.waitKey(0)
