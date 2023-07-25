import os
import sys
import cv2
import numpy as np
from image_enhancement import image_enhancement

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')

# from Utils.constant import *
# from Utils.path import *

from utils.detect.HumanDetector import HumanDetector
from utils.har.openpose.OpenPoseHAR import OpenPoseHAR
from utils.clothing.YoloClothing import YoloClothing
from utils.agender.Agender import AgeGenderEstimator
from utils.configure import *

class VisioinModel:
    def __init__(self) -> None:
        self.detect_model = HumanDetector()
        self.clothing_model = YoloClothing()
        self.har_model = OpenPoseHAR()
        self.agender_model = AgeGenderEstimator()
        

    def recognition(self, img: np.ndarray):
        result_act = -1
        result_clos = [0 for _ in range(CLO_NUM)]
        pose = [0 for _ in range(36)]
        clo_confs = [0 for _ in range(CLO_NUM)]
        har_conf = 0
        human_conf = 0
        face_img = None
        face_conf = 0

        result_age = -1
        result_gender = -1
        age_conf = np.zeros((101,), dtype=np.float32)
        gender_conf = 0.0


        img_copy: np.ndarray = img.copy()
        har_img_copy: np.ndarray = img.copy()
        w, h, _ = img_copy.shape

        ''' Find human bbox with face bbox'''
        detected = self.detect_model.detect_with_bbox(img_copy)
        # print(detected)
        ''' Recognize human activity based on pose '''
        har_result = self.har_model.inference(har_img_copy)
        _, result_act, _, pose_vector, har_conf = har_result
        pose = pose_vector[:36]

        clo_result = []
        for human in detected:
            body = human.get('person', None)
            face = human.get('head', None)

            bbox_size = 0
            if body != None:
                ''' Detect clothing types from human bbox '''
                human_img, human_bbox, human_conf = body

                x1, y1, x2, y2 = human_bbox
                bbox_size = (x2-x1)*(y2-y1)/(w*h)
                if bbox_size == 0:
                    human_conf = 0
                    continue

                human_img = image_enhancement.IE(human_img, 'HSV').FLH(1)
                clo_result = self.clothing_model.detect_clothes(human_img)

                for clo in clo_result:
                    cx1, cy1, cx2, cy2, conf, cls = clo
                    result_clos[int(cls)] = 1
                    clo_confs[int(cls)] = conf


            if face != None:
                ''' Crop face bbox '''
                face_img, face_bbox, face_conf = face
                x1, y1, x2, y2 = face_bbox
                face_bbox_size = (x2-x1)*(y2-y1)/(w*h)
                if face_bbox_size == 0:
                    face_img = None
                    face_conf = 0
                    continue
                gender_conf, age_conf, result_gender, result_age = self.agender_model.predict(face_img)


        return result_act, result_clos, pose, result_age, result_gender, har_conf, clo_confs, age_conf, gender_conf, human_conf, face_conf, face_img


    def recognition_from_path(self, path):
        img = cv2.imread(path)
        # print(img.shape)
        return self.recognition(img)


if __name__ == '__main__':
    model = VisioinModel()
    result = model.recognition_from_path('/SSD4TB/skygun/robot_code/PyLapras/Tester/video/20230403_223246/data/13/2.png')
    print(result)