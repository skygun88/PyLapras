# -*- coding: UTF-8 -*-
import os
import sys
import cv2
import numpy as np
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import tensorflow as tf

# physical_devices = tf.config.list_physical_devices('GPU')

# tf.config.set_visible_devices(physical_devices[2:], 'GPU')
# tf.config.set_visible_devices([], 'GPU')

# physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize_with_label
from Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from Tracking.deep_sort.tracker import Tracker
from Tracking import generate_dets as gdet


def load_pretrain_model(model):
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('openpose')[0], 'openpose')
    input_width, input_height = 656, 368
    # input_width, input_height = 1280, 720
    dyn_graph_path = {
        'VGG_origin': os.path.join(file_path, "Pose/graph_models/VGG_origin/graph_opt.pb"),
        'mobilenet_thin': os.path.join(file_path, "Pose/graph_models/mobilenet_thin/graph_opt.pb")
    }
    graph_path = dyn_graph_path[model]
    if not os.path.isfile(graph_path):
        raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)

    return TfPoseVisualizer(graph_path, target_size=(input_width, input_height))

class OpenPoseHAR:
    def __init__(self):
        self.estimator = load_pretrain_model('VGG_origin')
        # self.estimator = load_pretrain_model('mobilenet_thin')
        # self.action_classifier = load_action_premodel('Action/training/lounge6.h5')
        self.action_classifier = load_action_premodel('Action/training/web_lounge2.h5')
        # self.action_classifier = load_action_premodel('Action/training/lounge3.h5')
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('openpose')[0], 'openpose')
        model_filename = os.path.join(file_path, 'Tracking/graph_model/mars-small128.pb')
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        self.metric = NearestNeighborDistanceMetric("cosine", 0.3, None)
        self.reset_tracker()

    def reset_tracker(self):
        self.tracker = Tracker(self.metric)

    def inference(self, img):
        # pose estimation
        # start = time.time()
        # with tf.device('/GPU:2'):
            
        # pose_time = time.time() - start
        # get pose info
        
        # recognize the action framewise

        # start = time.time()

        if tf.test.is_gpu_available():
            with tf.device('/GPU:1'):
                humans = self.estimator.inference(img.copy())
                pose = TfPoseVisualizer.draw_pose_rgb(img, humans)
                img, labels, confs = framewise_recognize_with_label(pose, self.action_classifier, self.encoder, self.tracker)
        else:
            humans = self.estimator.inference(img.copy())
            pose = TfPoseVisualizer.draw_pose_rgb(img, humans)
            img, labels, confs = framewise_recognize_with_label(pose, self.action_classifier, self.encoder, self.tracker)

        # har_time = time.time() - start

        # print(f'HAR time - Pose: {pose_time:.3f}, Har: {har_time:.3f} -----------')

        # height, width = img.shape[:2]
        # num_label = f"Human: {len(humans)}"
        # cv2.putText(img, num_label, (5, height-45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)        
        # print('labels:', labels)
        
        
        num_humans = len(humans)
        bboxes = pose[2]
        pose_vector = pose[4] if len(pose[4]) else [0 for _ in range(36)]
        pr = (len(pose_vector) - pose_vector.count(0))/len(pose_vector)

        label = -1 if len(labels) == 0 else labels[0]
        conf = 0 if len(confs) == 0 else confs[0]
        label = -1 if pr < 0.3 else label
        conf = 0 if pr < 0.3 else conf


        return num_humans, label, bboxes, pose_vector, conf

# if __name__ == '__main__':
#     model = OpenPoseHAR()
#     img = cv2.imread('/home/skygun/Dropbox/CDSN/Testbed/robot/robot_code/test_img2.png', cv2.IMREAD_COLOR)

#     for i in range(5):
#         img = cv2.imread('/home/skygun/Dropbox/CDSN/Testbed/robot/robot_code/test_img2.png', cv2.IMREAD_COLOR)
#         model.inference(img)
    
#     model.reset_tracker()
#     for i in range(5):
#         img = cv2.imread('/home/skygun/Dropbox/CDSN/Testbed/robot/robot_code/test_img2.png', cv2.IMREAD_COLOR)
#         model.inference(img)