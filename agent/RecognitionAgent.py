import os
import cv2
import sys
import json
import time
import base64
import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from agent.LaprasAgent import LaprasAgent
from utils.configure import *
from utils.har.openpose.OpenPoseHAR import OpenPoseHAR
from utils.clothing.YoloClothing import YoloClothing
from map.n1map import N1Map, LoungeSensorMap


SHORT_SLEEVE_TOP = 0
LONG_SLEEVE_TOP = 1
SHORT_SLEEVE_OUTWEAR = 2
LONG_SLEEVE_OUTWEAR = 3
VEST = 4
SLING = 5
SHORTS = 6
TROUSERS = 7
SKIRTS = 8
SHORT_SLEEVE_DRESS = 9
LONG_SLEEVE_DRESS = 10
VEST_DRESS = 11
SLING_DRESS = 12


class RecognitionAgent(LaprasAgent):
    def __init__(self, agent_name='RecognitionAgent', place_name='Robot', sensor_place='N1Lounge8F'):
        super().__init__(agent_name, place_name)
        
        self.subscribe(f'{place_name}/context/RobotControlAgentOperatingStatus')

        self.create_timer(self.timer_callback, timer_period=1)
        # self.timer_cnt = 0
        ''' Modules '''
        self.har_model = OpenPoseHAR()
        self.clothing_model = YoloClothing()
        self.sensor_map: LoungeSensorMap = LoungeSensorMap()
        
        ''' For Recognition '''
        self.last_ts = 0
        self.last_alive = -1
        self.connected = False
        self.last_connected = False

        ''' For Robot control & Human detection '''
        self.robot_loc = [-1, -1]
        self.robot_r = -999
        self.robot_state = 'UNITITIALIZED'

        self.user_loc = -1, -1
        self.anchor = 0
        self.candiate_queue = []
        self.user_in_camera = False
        self.last_user_detected = -1
        self.bbox_loc = -1, -1
        self.bbox_size = -1, -1
        self.img_size = 848, 480
        self.img_center = self.img_size//2
        
        self.theta = 80
        self.ratio_coefficient = 0.1




        ''' Subscribe from robot '''
        self.robot_contexts = ['RobotDetectedImage', 'RobotControlAgentOperatingStatus', 'RobotX', 'RobotY', 'RobotStatus', 'RobotOrientation']
        for robot in self.robot_contexts:
            self.subscribe(f'{place_name}/context/{robot}')

        ''' Subscribe from motion sensors '''
        self.actvities, _, _ = self.sensor_map.get_activities()
        self.motions, _, _ = self.sensor_map.get_motions()

        for activity in self.actvities:
            self.subscribe(f'{sensor_place}/context/{activity}')
        for motion in self.motions:
            self.subscribe(f'{sensor_place}/context/{motion}')



        
    def timer_callback(self):      
        # self.timer_cnt += 1
        self.connected = self.check_connected()
        if self.last_connected == True and self.connected == False:
            self.har_model.reset_tracker()
        self.last_connected = self.connected

        if (time.time() - self.last_user_detected) < 10:
            self.user_in_camera = False
            self.bbox_loc = -1, -1


        latest = self.sensor_map.get_latest(5)
        filtered = list(filter(lambda x: (x[1] > 0) and ((time.time() - x[1]) < 30), latest))

        if len(filtered) > 0:
            self.candiate_queue = [self.sensor_map.get_loc(sensor[0]) for sensor in filtered]
        
        if self.connected and (self.robot_state == 'READY'):
            if (len(self.candiate_queue) > 0) and (self.user_in_camera == False):
                candidate_loc = self.candiate_queue[self.anchor]
                robot_loc = self.robot_loc

                user_angle = self.calculate_difference(robot_loc, candidate_loc)
                print(f'Angle difference: {user_angle}, robot: {self.robot_r}')

                ''' user detect '''
                rotational_angle = self.calculate_rotation(self.robot_r, user_angle)
                self.publish_func('robotAttend', arguments=[rotational_angle])
                ''' if there is no human, increase anchor value '''

            if self.user_in_camera:
                if abs(self.bbox_loc - self.img_center) > 100:
                    rotational_angle = (self.theta/2)*((self.img_center - self.bbox_loc)/self.img_center) 
                    self.publish_func('robotAttend', arguments=[int(rotational_angle)]) 

                self.estimate_distance(self.robot_r, self.bbox_size)



    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        msg_dict = json.loads(dict_string)
        
        if msg_dict.get('name') in self.robot_contexts:
            self.robot_response(msg_dict)
        else:
            self.sensor_response(msg_dict)


    
    def robot_response(self, msg_dict):
        context_name = msg_dict.get('name')
        timestamp = msg_dict.get('timestamp')

        if context_name == 'RobotControlAgentOperatingStatus':
            self.last_alive = msg_dict.get('timestamp')
        elif context_name == 'RobotDetectedImage':
            if self.connect:
                if timestamp - self.last_ts < 900:
                    return
                img_str = msg_dict['value']
                imgdata = base64.b64decode(img_str)
                cv_img = cv2.imdecode(np.array(np.frombuffer(imgdata, dtype=np.uint8)) , cv2.IMREAD_COLOR)
                
                num_humans, labels, bboxes = self.har_model.inference(cv_img)
                
                self.publish_context('humanDetected', value=len(bboxes), qos=1)
                if len(labels) > 0:
                    self.user_in_camera = True
                    self.last_user_detected = time.time()
                    self.bbox_loc = (2*bboxes[0][0] + bboxes[0][2])/2, (2*bboxes[0][1] + bboxes[0][3])/2
                    self.bbox_size = bboxes[0][2], bboxes[0][3]
                    # cv_img_cropeed = cv_img.copy()[bboxes[0][1]-30:bboxes[0][1]+bboxes[0][3]+30, bboxes[0][0]-30:bboxes[0][0]+bboxes[0][2]+30]
                    # detections = self.clothing_model.detect_clothes_class(cv_img_cropeed)
                    detections = self.clothing_model.detect_clothes_class(cv_img.copy())
                    print(detections)
                    processed_activity = self.activity_process(int(labels[0]))
                    processed_clothing = self.clothing_process(detections) if len(detections) > 0 else -1
                    
                    self.publish_context('detectedActivity', value=processed_activity, qos=1)
                    self.publish_context('detectedClothing', value=processed_clothing, qos=1)
                    print(f'Activity: {processed_activity} | Clothing: {processed_clothing}')

                self.last_ts = timestamp

        elif context_name == 'RobotX':
            self.robot_loc[0] = msg_dict.get('value')
        elif context_name == 'RobotY':
            self.robot_loc[1] = msg_dict.get('value')
        elif context_name == 'RobotOrientation':
            self.robot_r = msg_dict.get('value')
        elif context_name == 'RobotStatus':
            self.robot_state = msg_dict.get('value')

        else:
            print('wrong')


    def sensor_response(self, msg_dict):
        print(msg_dict)
        context_name = msg_dict.get('name')
        prev_state = self.sensor_map.get_sensor(context_name)
        state = msg_dict.get('timestamp')/1000
        value = True if msg_dict.get('value') == "True" else False

        if value == True:
            self.sensor_map.update_sensor(context_name, state)


    def calculate_difference(self, robot_loc, user_loc):
        np_robot = np.array(robot_loc)
        np_user = np.array(user_loc)
        
        vector = np_user - np_robot
        unit = np.array([1, 0])
        ang1 = np.arctan2(*vector[::-1])
        ang2 = np.arctan2(*unit[::-1])
        angle_360 = np.rad2deg((ang1 - ang2) % (2 * np.pi))
        result_angle = angle_360 if angle_360 <= 180 else angle_360 - 360
        return result_angle

    def calculate_rotation(self, robot_angle, user_angle):
        robot_360 = robot_angle if robot_angle > 0 else robot_angle + 360
        user_360 = user_angle if user_angle > 0 else user_angle + 360

        difference = user_360 - robot_360
        result_angle = difference if difference <= 180 else difference - 360

        return result_angle

    def estimate_distance(self, angle, bbox_size):
        x_ratio = bbox_size[0]/self.img_size[0]
        y_ratio = bbox_size[1]/self.img_size[1]

        mean_ratio = (x_ratio+y_ratio)/2

        mean_distance = self.ratio_coefficient/mean_ratio
        x_distance = self.ratio_coefficient/x_ratio
        y_distance = self.ratio_coefficient/y_ratio

        print(x_ratio, y_ratio, mean_ratio, x_distance, y_distance, mean_distance)


            
    def check_connected(self):
        now = int(time.time()*1000)
        connected = now - self.last_alive < 1000*15 
        return connected


    def activity_process(self, label):
        sits = [0]
        stands = [1]
        lie = [2]
        if label in sits:
            return 0
        elif label in stands:
            return 1
        elif label in lie:
            return 2
        else:
            return -1

    def clothing_process(self, detections):
        long_clothes = [LONG_SLEEVE_TOP, LONG_SLEEVE_OUTWEAR, LONG_SLEEVE_DRESS] # Long sleeve top, Long sleeve outwear, Long sleeve dress
        short_clothes = [SHORT_SLEEVE_TOP, SHORT_SLEEVE_OUTWEAR, VEST, SLING, SHORT_SLEEVE_DRESS, VEST_DRESS, SLING_DRESS]
        long_prob = 0
        short_prob = 0
        result = 0
        for detection in detections:
            if detection[0] in long_clothes:
                long_prob += detection[1]
            elif detection[0] in short_clothes:
                short_prob += detection[1]
        
        if long_prob > short_prob:
            result = 1
        elif long_prob == short_prob:
            result = -1
        return result
        

if __name__ == '__main__':
    client = RecognitionAgent()
    client.loop_forever()
    client.disconnect()
