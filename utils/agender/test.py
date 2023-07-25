import cv2
from pyagender.pyagender import PyAgender


if __name__ =='__main__':
    model = PyAgender()

    img = cv2.imread('/SSD4TB/skygun/vision/individual/faces/webcam/resolution_test/elder.jpeg')
    result = model.gender_age(img)
    print(result)