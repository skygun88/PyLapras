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
from collections import Counter
sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from agent import LaprasAgent
from utils.state import StateCollector
from utils.db import upload_replay2, db_parser, preprocessing
# from utils.dqn import DQN
from agent.controller.thermal_model import PPOAgent


INITIALIZING, READY, DETERMINING, ACTUATING, WAITING, TRAINING, RECORDING, MONITORING = 0, 1, 2, 3, 4, 5, 6, 7
STATE_MAP = {
                INITIALIZING: 'INITIALIZING', 
                READY: 'READY', 
                DETERMINING: 'DETERMINING', 
                ACTUATING: 'ACTUATING', 
                WAITING: 'WAITING', 
                TRAINING: 'TRAINING',
                RECORDING: 'RECORDING',
                MONITORING: 'MONITORING'
            }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RL_STATE_CLASSES = ['Temperature', 'Humidity', 'AC Mode', 'Human existence', 'Activity', 'Clothing']
STILL, DOWN, UP = 0, 1, 2
POSITIVE_REWARD = 1
NEGATIVE_REWARD = - POSITIVE_REWARD
SIT, STAND, LIE = 0, 1, 2
INNER, OUTER = 0, 1

class ThermalControlAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='ThermalControlAgent', place_name='N1Lounge8F', 
                    n_action=3, time_interval=10, model=None,
                    weight=None, fixed_clothing=None):
        super().__init__(agent_name, place_name)
        self.status = INITIALIZING
        self.context_collector = StateCollector()
        self.human_context_collector = StateCollector() 
        self.n_action = n_action
        self.start_ts = self.curr_timestamp()
        self.rl_agent = PPOAgent()
        self.weight = weight
        self.fixed_clothing = fixed_clothing

        if self.weight != None:
            if os.path.isfile(self.weight):
                self.rl_agent.load_model(self.weight)

        self.sub_contexts = ['sensor0_Temperature', 'sensor1_Temperature', 'sensor0_Humidity', 'sensor1_Humidity', 'Aircon0Power', 'Aircon1Power']
        self.sub_funcs = ['PowerUpAC', 'PowerDownAC']
        self.sub_human_contexts = ['humanDetected', 'detectedActivity', 'detectedClothing']


        self.history = None
        self.next_history = None
        self.user_detected = False
        self.human_exist_queue = [0]*5
        self.human_activity_queue = [-1]*10
        self.human_clothing_queue = [-1]*10

        self.last_state = []
        self.last_action = -1
        self.last_prob_a = []
        self.last_next_state = []
        self.user_feedback = -1
        self.episode_length = 0

        self.expected_ac_mode, self.anchor = -1, -1
        self.waiting_threshold = 60

        for context in self.sub_contexts:
            self.subscribe(f'{place_name}/context/{context}')
        for context in self.sub_human_contexts:
            self.subscribe(f'Robot/context/{context}', qos=0)
        for funcs in self.sub_funcs:
            self.subscribe(f'Robot/functionality/{funcs}')
        

        self.create_timer(self.timer_callback, timer_period=time_interval)
        self.timer_cnt = 0



    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_MAP[self.status]}->{STATE_MAP[next_state]}')          
        self.status = next_state

    def timer_callback(self):
        print(f'[{self.agent_name}/{self.place_name}] Status: {STATE_MAP[self.status]}')
        curr_context = self.context_collector.get(self.sub_contexts)

        if self.status != INITIALIZING:
            curr_state = self.preprocess_state(curr_context)
            curr_human_state = self.get_curr_human_state()

            print(curr_state, curr_human_state)
            self.collect_replay(curr_state, curr_human_state)


        if self.status == INITIALIZING:
            if not None in curr_context:
                self.transition(READY)
            else: 
                return
        elif self.status == READY:
            if curr_human_state[0] == 1 and curr_human_state[2] != -1:
                self.episode_length = 0
                self.transition(DETERMINING)

        elif self.status == DETERMINING:
            self.last_state = self.get_rl_state(curr_state, curr_human_state)
            self.last_action, self.last_prob_a = self.get_action(self.last_state)
            self.transition(ACTUATING)

        elif self.status == ACTUATING:
            self.expected_ac_mode, self.anchor = self.actuate(self.last_action, curr_state)
            self.transition(WAITING)

        elif self.status == WAITING:
            if curr_state[2] == self.expected_ac_mode:
                if (time.time() - self.anchor) > (self.waiting_threshold * 1000):
                    self.last_next_state = self.get_rl_state(curr_state, self.get_curr_human_state())
                    self.user_feedback = STILL
                    self.transition(RECORDING)

        elif self.status == RECORDING:
            ''' Memory for agent action '''
            reward = self.reward_function(self.last_action, self.user_feedback)
            self.rl_agent.add_memory(self.last_state, self.last_action, reward, self.last_next_state, 0, self.last_prob_a[self.last_action].item())
            self.episode_length += 1

            ''' Memory for user action '''
            if self.user_feedback != STILL:
                _, prob_a = self.get_action(self.last_next_state)
                self.rl_agent.add_memory(self.last_next_state, self.user_feedback, POSITIVE_REWARD, self.get_rl_state(curr_state, curr_human_state), 0, prob_a[self.user_feedback].item())
                self.episode_length += 1

            if (self.last_action == STILL) and (self.user_feedback == STILL):
                self.rl_agent.finish_path(self.episode_length)
                self.episode_length = 0
                self.transition(TRAINING)
            elif (self.last_action != STILL) and (self.user_feedback == STILL):
                self.transition(MONITORING)
            else:
                self.transition(DETERMINING)

        elif self.status == MONITORING:
            print('Waiting for context change')

        elif self.status == TRAINING:
            for _ in range(3):
                self.rl_agent.update_network()

            self.rl_agent.save_model(self.weight)


            '''  Reset '''
            self.human_exist_queue = [0]*5
            self.human_activity_queue = [-1]*10
            self.human_clothing_queue = [-1]*10

            self.transition(READY)

            
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


    def get_curr_human_state(self):
        existence = 1 if 1 in self.human_exist_queue else 0
        filtered_activities = list(filter(lambda x: x != -1, self.human_activity_queue))
        filtered_clothings = list(filter(lambda x: x != -1, self.human_clothing_queue))

        activity = Counter(filtered_activities).most_common(1)[0][0] if len(filtered_activities) > 0 else -1
        clothing = Counter(filtered_clothings).most_common(1)[0][0] if len(filtered_clothings) > 0 else -1
        if self.fixed_clothing != None:
            clothing = self.fixed_clothing

        return [existence, activity, clothing]


    def collect_replay(self, curr_state, curr_human_state):
        ts = self.curr_timestamp()
        upload_replay2(self.start_ts, ts, curr_state+curr_human_state, RL_STATE_CLASSES)
    
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
            if self.status == MONITORING:
                if trainsition == True:
                    self.transition(DETERMINING)

        elif name in self.sub_human_contexts:
            dt = datetime.datetime.fromtimestamp(timestamp/1000)
            print(dt, name, value)
            if name == 'humanDetected':
                self.update_human_exist(value)
            elif name == 'detectedActivity':
                self.update_human_activity(value)
            elif name == 'detectedClothing':
                self.update_human_clothing(value)
            else:
                print('error occur in on mesaage for human context')
                sys.exit()
        elif name in self.sub_funcs:
            if self.status == WAITING:
                curr_context = self.context_collector.get(self.sub_contexts)
                curr_state = self.preprocess_state(curr_context)
                self.last_next_state = self.get_rl_state(curr_state, self.get_curr_human_state())
                if name == 'PowerDownAC':
                    self.user_feedback = 1
                    self.actuate(1, curr_state)
                elif name == 'PowerUpAC':
                    self.user_feedback = 2
                    self.actuate(2, curr_state)
                else:
                    print('error occur in on mesaage for functionality')
                    sys.exit()
                self.transition(RECORDING)
        else:
            print('error occur in on message')
            sys.exit()

    def get_rl_state(self, state, human_state):
        combined_state = state + [human_state[2]]
        combined_state[0] = (min(max(combined_state[0], 25), 28)-25)/(28.0-25.0)
        combined_state[1] = (min(max(combined_state[1], 48), 60)-48)/(60.0-48.0)
        combined_state[2] = combined_state[2]/2.0
        combined_state[3] = combined_state[3]/1.0
        return combined_state

    def get_action(self, state):
        action, prob_a = self.rl_agent.get_action(state)
        return action, prob_a
    
    def actuate(self, action, curr_state):
        ac_mode = curr_state[2]
        expected_ac_mode = ac_mode
        if action == 1:
            if ac_mode == 2:
                self.publish_func('StopAircon1')
                expected_ac_mode -= 1
            elif ac_mode == 1:
                self.publish_func('StopAircon0')
                self.publish_func('StopAircon1')
                expected_ac_mode -= 1

        elif action == 2:
            if ac_mode == 0:
                self.publish_func('StartAircon0')
                expected_ac_mode += 1
            elif ac_mode == 1:
                self.publish_func('StartAircon0')
                self.publish_func('StartAircon1')
                expected_ac_mode + 1
        anchor = time.time()
        return expected_ac_mode, anchor


    def update_human_exist(self, detection_value):
        existence = 0
        if detection_value > 0:
            existence = 1
        self.human_exist_queue.append(existence)
        self.human_exist_queue.pop(0)

    def update_human_activity(self, value):
        self.human_activity_queue.append(value)
        self.human_activity_queue.pop(0)

    def update_human_clothing(self, value):
        if value == -1:
            return
        self.human_clothing_queue.append(value)
        self.human_clothing_queue.pop(0)

    
    def reward_function(self, action, feedback):
        reward = POSITIVE_REWARD
        if (action == STILL) and (feedback != STILL):
            reward = NEGATIVE_REWARD
        if (action == DOWN) and (feedback == UP):
            reward = NEGATIVE_REWARD
        if (action == UP) and (feedback == DOWN):
            reward = NEGATIVE_REWARD
        return reward
        



if __name__ == '__main__':
    client = ThermalControlAgent(agent_name='ThermalControlAgent', weight='PyLapras/resources/weight/thermal.pt')
    client = ThermalControlAgent(agent_name='ThermalControlAgent', weight='PyLapras/resources/weight/thermal.pt', fixed_clothing=0)

    client.loop_forever()
    client.disconnect()
