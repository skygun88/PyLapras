import os
import cv2
import sys
import json
import time
import base64
import random
import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from agent.LaprasAgent import LaprasAgent
from utils.state import StateCollector
from utils.comfort import PMV

from utils.configure import *

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


class FakeRecognitionAgent(LaprasAgent):
    def __init__(self, agent_name='RecognitionAgent', place_name='Robot', sensor_place='N1Lounge8F'):
        super().__init__(agent_name, place_name)
        self.context_collector = StateCollector()
        self.sub_contexts = ['sensor0_Temperature', 'sensor1_Temperature', 'sensor0_Humidity', 'sensor1_Humidity', 'Aircon0Power', 'Aircon1Power']

        for context in self.sub_contexts:
            self.subscribe(f'{sensor_place}/context/{context}')

        self.clothing = random.choice([0, 1])
        self.activity = random.choice([0, 1])
        self.create_timer(self.timer_callback, timer_period=5)
        self.timer_cnt = 0
        self.sensor_place = sensor_place
        

        
    def timer_callback(self):      
        self.timer_cnt += 1
        print(self.timer_cnt)
        if self.timer_cnt < 36:
            self.publish_context('humanDetected', value=1, qos=1)
            self.publish_context('detectedActivity', value=0, qos=1)
            self.publish_context('detectedClothing', value=self.clothing, qos=1)
        else:
            self.publish_context('humanDetected', value=0, qos=1)

        if self.timer_cnt == 44:
            self.set_ac(random.choice([0, 1, 2]))

        if self.timer_cnt > 52:
            self.clothing = random.choice([0, 1])
            self.activity = random.choice([0, 1])
            self.timer_cnt = 0

    
    def set_ac(self, mode):
        print(f'set {mode}')
        if mode == 0:
            self.publish_func('StopAircon0', place=self.sensor_place)
            self.publish_func('StopAircon1', place=self.sensor_place)
        elif mode == 1:
            self.publish_func('StartAircon0', place=self.sensor_place)
            self.publish_func('StopAircon1', place=self.sensor_place)
        elif mode == 2:
            self.publish_func('StartAircon0', place=self.sensor_place)
            self.publish_func('StartAircon1', place=self.sensor_place)
        else:
            print('error occur in SET AC')
            sys.exit()

    
    def user_timer(self):
        curr_context = self.context_collector.get(self.sub_contexts)
        curr_state = self.preprocess_state(curr_context)
        curr_human_state = self.activity, self.clothing 

        if curr_human_state[2] == 0:
            if curr_state[2] > 1:
                self.publish_func('PowerDownAC', place='Robot')
            elif curr_state[2] < 1:
                self.publish_func('PowerUpAC', place='Robot')
        else:
            if curr_state[2] < 2:
                self.publish_func('PowerUpAC', place='Robot')

    def preprocess_state(self, context):
        sensor0_tem = context[0]
        sensor1_tem = context[1]
        sensor0_hum = context[2]
        sensor1_hum = context[3]
        ac0_power = context[4]
        ac1_power = context[5]

        temperature = (sensor0_tem+sensor1_tem)/2
        humidity = (sensor0_hum+sensor1_hum)/2
        ac_mode = ac0_power + ac1_power

        return [temperature, humidity, ac_mode]




    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        # print(f'Arrived message: {dict}')

        ''' Topic datas '''
        timestamp, publisher, type, name = dict.get('timestamp'), dict.get('publisher'), dict.get('type'), dict.get('name')
        value = dict.get('value')

        # dt = datetime.datetime.fromtimestamp(timestamp/1000)
        # print(dt, name, value)
        if name in self.sub_contexts:
            trainsition = self.context_collector.update_state(msg)

            

        

if __name__ == '__main__':
    client = FakeRecognitionAgent()
    client.loop_forever()
    client.disconnect()
