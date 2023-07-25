import argparse
import time
from pathlib import Path
import os
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random


sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from util.datasets import letterbox
from models.experimental import attempt_load

from util.general import check_img_size, non_max_suppression, scale_coords
from util.torch_utils import select_device


class HumanDetector:
    def __init__(self):
        self.img_size = 640
        self.iou_thres = 0.45
        self.conf_thres = 0.5
        self.weights = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights/crowdhuman_yolov5m.pt')
        # self.device = 'cpu'
        
        self.device = '1' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'

        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once


    def detect(self, img):
        with torch.no_grad():
            im0 = img.copy()
            img = letterbox(im0, 640, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # print(img.shape)

            # Inference
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            results = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'

                        

                        if 'person' in label:
                            # padding = 10
                            padding = 5
                            # im0 = im0[max(int((xyxy[1].item())-padding), 0):int(xyxy[3].item())+padding, max(int(xyxy[0].item())-padding, 0):int(xyxy[2].item())+padding]
                            # results.append(im0)
                            x1, y1, x2, y2 = max(int(xyxy[1].item()-padding), 0), max(int(xyxy[0].item())-padding, 0), int(xyxy[3].item())+padding, int(xyxy[2].item())+padding
                            results.append(im0[x1:x2, y1:y2])
                            # print(results)
                            # print(xyxy)

                # Save results (image with detections)
                # cv2.imwrite('result.png', im0)
            return results

    def detect_with_bbox(self, img):
        with torch.no_grad():
            im0 = img.copy()
            img = letterbox(im0, 640, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # print(img.shape)

            # Inference
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            results = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    partial_dectection = {}
                    for *xyxy, conf, cls in reversed(det):
                        # label = f'{self.names[int(cls)]} {conf:.2f}'
                        label = f'{self.names[int(cls)]}'

                        # print(label)

                        # if 'person' == label:
                        #     # padding = 10
                        #     padding = 5
                        #     x1, y1, x2, y2 = max(int(xyxy[1].item()-padding), 0), max(int(xyxy[0].item())-padding, 0), int(xyxy[3].item())+padding, int(xyxy[2].item())+padding
                        #     # results.append([im0[x1:x2, y1:y2], [y1, x1, y2, x2]])
                        # else:
                        #     padding = 10

                        padding = 10 if 'person' == label else 15
                        x1, y1, x2, y2 = max(int(xyxy[1].item()-padding), 0), max(int(xyxy[0].item())-padding, 0), int(xyxy[3].item())+padding, int(xyxy[2].item())+padding
                        partial_dectection[label] = [im0[x1:x2, y1:y2], [y1, x1, y2, x2], conf.cpu().item()]
                            
                            # print(results)
                            # print(xyxy)

                    results.append(partial_dectection)
            
            # print(results)
                # Save results (image with detections)
                # cv2.imwrite('result.png', im0)
            return results


if __name__ == '__main__':
    imgs = ['imgs/00_3.png'] * 10

    detector = HumanDetector()

    for path in imgs:
        img = cv2.imread(path)    
        with torch.no_grad():
            result = detector.detect(img)
            cv2.imwrite('result.png', result[0])
