import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pyagender.pyagender import PyAgender

_DIRPATH = os.path.abspath(os.path.dirname(__file__))
CASCADE_DIR = os.path.join(_DIRPATH, 'haarcascades')

def check_cascade(cascades, cascade_results):
    found_indxes = []
    for i, result in enumerate(cascade_results):
        if len(result) > 0:
            found_indxes.append(i)
    return [cascades[x] for x in found_indxes]

class AgeGenderEstimator:
    def __init__(self):
        self.model = PyAgender()
        self.cascades_fnames = os.listdir(CASCADE_DIR)
        self.cascade_files = [os.path.join(CASCADE_DIR, fname) for fname in self.cascades_fnames]
        
        self.cascades = [cv2.CascadeClassifier(cascade_file) for cascade_file in self.cascade_files]

    
    def detect_raw_ag(self, img: np.ndarray):
        test_img = PyAgender.aspect_resize(img, self.model.resnet_imagesize, self.model.resnet_imagesize)

        # predict ages and genders of the detected faces
        result = self.model.resnet.predict(np.array([test_img]))
        predicted_genders = result[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_age = result[1].dot(ages).flatten()[0]
        age_prob = result[1][0]
        gender_prob = predicted_genders[0][0] 
        predicted_gender = 0 if gender_prob < 0.5 else 1
        gender_prob = 1-predicted_genders[0][predicted_gender]

        return gender_prob, age_prob, predicted_gender, predicted_age
    

    def predict(self, img: np.ndarray):
        ''' Value when face is not detected '''
        face_is_here = False
        predicted_age = -1
        predicted_gender = -1
        age_prob = np.zeros((101,), dtype=np.float32)
        gender_prob = 0.0

        ''' Check there is whole face '''
        img_copy = img.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        eye_max = min(h, w)//10
        cascade_results = [
            cascade.detectMultiScale(gray, minNeighbors=0, scaleFactor=1.4, maxSize=(eye_max, eye_max)) if 'eye' in cname 
            else  cascade.detectMultiScale(gray, minNeighbors=0, scaleFactor=1.2)
            for cascade, cname in zip(self.cascades, self.cascades_fnames)
        ]
        face_is_here = True if sum([len(x) for x in cascade_results]) > 0 else False 

        ''' Age & Gender Estimation '''
        if face_is_here:
            # print(check_cascade(self.cascades_fnames, cascade_results))
            # print(cascade_results)
            # print(['eye' in x for x in self.cascades_fnames])
            # print(w, h, eye_max)
            img_copy = cv2.GaussianBlur(img_copy,(3,3),0)
            gender_prob, age_prob, predicted_gender, predicted_age = self.detect_raw_ag(img_copy)
            age_prob = age_prob*0.9
        
        return gender_prob, age_prob, predicted_gender, predicted_age


    def predict_from_path(self, fpath: str):
        img = cv2.imread(fpath)
        return self.predict(img)
    
# if __name__ == '__main__':
#     model = AgeGenderEstimator()
#     print(model.predict_from_path('/SSD4TB/skygun/robot_code/Result/20230403_223246/Face/10/5.png'))