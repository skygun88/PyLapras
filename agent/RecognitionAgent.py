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

class RecognitionAgent(LaprasAgent):
    def __init__(self, agent_name='RecognitionAgent', place_name='Robot'):
        super().__init__(agent_name, place_name)
        self.subscribe(f'{place_name}/context/RobotDetectedImage')
        self.subscribe(f'{place_name}/context/RobotControlAgentOperatingStatus', 2)

        self.create_timer(self.timer_callback, timer_period=1)
        self.timer_cnt = 0
        self.last_alive = -1
        self.connected = False
        self.last_connected = False
        self.model = OpenPoseHAR()
        
    def timer_callback(self):      
        # self.timer_cnt += 1
        self.connected = self.check_connected()
        if self.last_connected == True and self.connected == False:
            self.model.reset_tracker()
        self.last_connected = self.connected

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        msg_dict = json.loads(dict_string)
        context_name = msg_dict.get('name')
        
        if context_name == 'RobotControlAgentOperatingStatus':
            self.last_alive = msg_dict.get('timestamp')
        elif context_name == 'RobotDetectedImage':
            if self.connect:
                img_str = msg_dict['value']
                imgdata = base64.b64decode(img_str)
                cv_img = cv2.imdecode(np.array(np.frombuffer(imgdata, dtype=np.uint8)) , cv2.IMREAD_COLOR)
                
                num_humans, labels, bboxes = self.model.inference(cv_img)

                if len(labels) > 0:
                    self.publish_context('detectedActivity', value=int(labels[0]))
                # print(num_humans, labels, bboxes)
        else:
            print('wrong')
            
    def check_connected(self):
        now = int(time.time()*1000)
        connected = now - self.last_alive < 1000*15 
        return connected

if __name__ == '__main__':
    client = RecognitionAgent()
    client.loop_forever()
    client.disconnect()