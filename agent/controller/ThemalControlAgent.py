import os 
import sys
import json
import time
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from agent import LaprasAgent
from utils.state import StateCollector
from utils.db import upload_replay, db_parser, preprocessing
from utils.dqn import DQN


INITIALIZING, READY, DETERMINING, ACTUATING, WAITING, TRAINING = 0, 1, 2, 3, 4
STATE_MAP = {
                INITIALIZING: 'INITIALIZING', 
                READY: 'READY', 
                DETERMINING: 'DETERMINING', 
                ACTUATING: 'ACTUATING', 
                WAITING: 'WAITING', 
                TRAINING: 'TRAINING'
            }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ThermalControlAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='ThermalControlAgent', place_name='N1Lounge8F', 
                    n_action=4, time_interval=60, epsilon=0.0,
                    weight=None):
        super().__init__(agent_name, place_name)
        self.status = INITIALIZING
        self.context_collector = StateCollector()
        self.human_context_collector = StateCollector() 
        self.n_action = n_action
        self.start_ts = self.curr_timestamp()
        self.epsilon = 0.0
        

        self.sub_contexts = ['sensor0_Temperature', 'sensor1_Temperature', 'sensor0_Humidity', 'sensor1_Humidity', 'Aircon0Power', 'Aircon1Power']
        ''' 
        Need to add contexts from robot
        '''
        self.sub_contexts = self.sub_contexts + ['Aircon0Temp', 'Aircon1Temp']
        self.sub_human_contexts = ['Activity', 'Outfit']


        # self.sub_funcs = ['StartAircon0', 'StartAircon1', 'StopAircon0', 'StopAircon1', 'tempUpAircon0', 'tempUpAircon1', 'tempDownAircon0', 'tempDownAircon1']
        self.history = None
        self.next_history = None
        self.user_detected = False

        for context in self.sub_contexts:
            self.subscribe(f'{place_name}/context/{context}')
        for context in self.sub_human_contexts:
            self.subscribe(f'{place_name}/context/{context}')

        self.create_timer(self.timer_callback, timer_period=time_interval)
        self.timer_cnt = 0


        # self.publish_context('AirconServiceAgentOperatingStatus', STATE_MAP[self.status], 2)

    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_MAP[self.status]}->{STATE_MAP[next_state]}')          
        self.status = next_state
        # self.publish_context('AirconServiceAgentOperatingStatus', STATE_MAP[self.status], 2)

    def timer_callback(self):
        print(f'[{self.agent_name}/{self.place_name}] Status: {STATE_MAP[self.status]}')
        curr_state = self.context_collector.get(self.sub_contexts)

        if self.status != INITIALIZING:
            self.collect_replay(curr_state)
            if self.status == DETERMINING:
                if self.history.shape[0] < 15:
                    self.history = np.append(self.history, np.array(curr_state).reshape(1, -1), axis=0)
                else:
                    self.history = np.append(self.history[1:, :], np.array(curr_state).reshape(1, -1), axis=0)

        if self.status == INITIALIZING:
            if not None in curr_state:
                self.transition(READY)
            else: 
                return
        elif self.status == READY:
            if self.user_detected:
                self.transition(next_state=DETERMINING)

        else:
            print('Error Occur')
            sys.exit()
    
        
        self.timer_cnt += 1

    '''
    State Definition
    1. Average Temperature Value: [~25.5, 25.5~26.5, 26.5~] => [0, 1, 2]
    2. Average Humidity Value: [~52, 52~56, 56~] => [0, 1, 2]
    3. A.C Mode: [OFF, 26, 24] => [0, 1, 2]
    4. OutFit: [Jacket OFF, Jacket ON] => [0, 1]
    '''
    def preprocess_state(self, context):
        sensor0_tem = context['sensor0_Temperature']
        sensor1_tem = context['sensor1_Temperature']
        sensor0_hum = context['sensor0_Temperature']
        sensor1_hum = context['sensor0_Temperature']




    def collect_replay(self, state):
        ts = self.curr_timestamp()
        print(ts, state)
        upload_replay(self.place_name, self.start_ts, ts, state)
    
    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        # print(f'Arrived message: {dict}')

        ''' Topic datas '''
        timestamp, publisher, type, name = dict.get('timestamp'), dict.get('publisher'), dict.get('type'), dict.get('name')
        value = dict.get('value')
        if name in self.sub_contexts:
            trainsition = self.state_collector.update_state(msg)

    def get_random_action(self):
        return random.randint(0, self.n_action-1)

    
    def get_action(self, state: torch.Tensor, epsilon: float):
        with torch.no_grad():
            output = self.model(state)
            if random.random() < epsilon: 
                action = random.randint(0, self.n_action-1)
            else:
                action = torch.argmax(output).item()
            max_q = torch.amax(output).item()

        return action, max_q
    
    def actuate(self, action):
        if action == 0:
            self.publish_func('StopAircon0')
            self.publish_func('StopAircon1')
        elif action == 1:
            self.publish_func('StartAircon0')
            self.publish_func('StopAircon1')

        elif action == 2:
            self.publish_func('StopAircon0')
            self.publish_func('StartAircon1')

        elif action == 3:
            self.publish_func('StartAircon0')
            self.publish_func('StartAircon1')



if __name__ == '__main__':
    client = ThermalControlAgent(agent_name='ThermalControlAgent')

    client.loop_forever()
    client.disconnect()