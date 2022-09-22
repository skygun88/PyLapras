import os
from symbol import pass_stmt 
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
from utils.comfort import PMV
from utils.configure import *


# INITIALIZING, READY, DETERMINING, ACTUATING, WAITING, TRAINING, RECORDING, MONITORING = 0, 1, 2, 3, 4, 5, 6, 7
INITIALIZING, READY, DETERMINING, ACTUATING, WAITING, TRAINING, RECORDING, USER_MONITORING, USER_ACTUATING, USER_RECORDING, USER_WAITING = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

STATE_MAP = {
                INITIALIZING: 'INITIALIZING', 
                READY: 'READY', 
                DETERMINING: 'DETERMINING', 
                ACTUATING: 'ACTUATING', 
                WAITING: 'WAITING', 
                TRAINING: 'TRAINING',
                RECORDING: 'RECORDING',
                USER_MONITORING: 'USER_MONITORING',
                USER_ACTUATING: 'USER_ACTUATING',
                USER_RECORDING: 'USER_RECORDING',
                USER_WAITING:'USER_WAITING'
            }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RL_STATE_CLASSES = ['Temperature', 'Humidity', 'AC Mode', 'Human existence', 'Activity', 'Clothing']
STILL, DOWN, UP = 0, 1, 2
POSITIVE_REWARD = 0.1
NEGATIVE_REWARD = - POSITIVE_REWARD
SIT, STAND, LIE = 0, 1, 2
INNER, OUTER = 0, 1

class ThermalControlAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='ThermalControlAgent', place_name='Robot', sensor_place='N1Lounge8F',
                    n_action=3, 
                    time_interval=5, 
                    weight=None, 
                    fixed_clothing=None, 
                    memory_path=None,
                    ambient=False
                ):
        super().__init__(agent_name, place_name)
        self.status = INITIALIZING
        self.context_collector = StateCollector()
        self.human_context_collector = StateCollector() 
        self.ambient = ambient
        self.n_action = n_action
        self.n_state = 3 if self.ambient else 5
        self.start_ts = self.curr_timestamp()
        self.rl_agent = PPOAgent(action_size=self.n_action, input_size=self.n_state)
        self.weight = weight
        self.fixed_clothing = fixed_clothing
        self.memory_path = memory_path
        
        if self.weight != None:
            if os.path.isfile(self.weight):
                self.rl_agent.load_model(self.weight)
        else:
            self.weight = os.path.join(WEIGHT_PATH, f'thermal_{self.start_ts}.pt')
        
        if self.memory_path != None:
            if os.path.isfile(self.memory_path):
                self.rl_agent.load_memory(self.memory_path)
        else:
            self.memory_path = os.path.join(WEIGHT_PATH, f'memory_{self.start_ts}.pickle')

        self.sub_contexts = ['sensor0_Temperature', 'sensor1_Temperature', 'sensor0_Humidity', 'sensor1_Humidity', 'Aircon0Power', 'Aircon1Power']
        self.sub_funcs = ['PowerUpAC', 'PowerDownAC']
        self.sub_human_contexts = ['humanDetected', 'detectedActivity', 'detectedClothing']


        self.user_detected = False
        self.human_exist_queue = [0]*5
        self.human_activity_queue = [-1]*20
        self.human_clothing_queue = [-1]*20

        self.episode = 0
        self.last_state = []
        self.last_action = -1
        self.last_prob_a = []
        self.last_next_state = []
        self.last_user_context = []
        self.user_feedback = -1
        self.episode_length = 0

        self.expected_ac_mode, self.anchor = -1, -1
        self.waiting_threshold = 30

        self.curr_agent_act = 0
        self.curr_human_act = 0


        for context in self.sub_contexts:
            self.subscribe(f'{sensor_place}/context/{context}')
        for context in self.sub_human_contexts:
            self.subscribe(f'{place_name}/context/{context}')
        for funcs in self.sub_funcs:
            self.subscribe(f'{place_name}/functionality/{funcs}')
        



        self.create_timer(self.timer_callback, timer_period=time_interval)
        self.timer_cnt = 0

        # self.create_timer(self.user_timer, timer_period=10)



    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_MAP[self.status]}->{STATE_MAP[next_state]}')          
        self.status = next_state

    def user_timer(self):
        if self.status == WAITING:
            curr_context = self.context_collector.get(self.sub_contexts)
            curr_state = self.preprocess_state(curr_context)
            curr_human_state = self.get_curr_human_state()

            if curr_human_state[2] == 0:
                if curr_state[2] > 1:
                    self.publish_func('PowerDownAC', place='Robot')
                elif curr_state[2] < 1:
                    self.publish_func('PowerUpAC', place='Robot')
            else:
                if curr_state[2] < 2:
                    self.publish_func('PowerUpAC', place='Robot')


        # if self.status == WAITING:
        #     curr_context = self.context_collector.get(self.sub_contexts)
        #     curr_state = self.preprocess_state(curr_context)
        #     curr_human_state = self.get_curr_human_state()

        #     cloth_index = 0.40 if curr_human_state[2] == 0 else 0.74 # socks, shoes, panaties, short sleeve shirt, trouser, jacket
        #     metabolism_rate = 1.1 # Sitting
        #     pmv = PMV(cloth_index, curr_state[0], curr_state[1], metabolism_rate).calculatePMV() 

        #     print(f'User\'s PMV: {pmv}', end='')
        #     boundary = 0.2
        #     if pmv < -0.4: # Cold discomfort - pmv < -0.5 
        #         if curr_state[2] > 0: 
        #             self.publish_func('PowerDownAC', place='Robot')
        #             print(' - Power Down', end='')
        #     elif pmv < -boundary: # -0.5 < pmv < -0.4
        #         if curr_state[2] > 1:
        #             self.publish_func('PowerDownAC', place='Robot')
        #             print(' - Power Down', end='')
        #     elif pmv > 0.4: # Hot discomfort - 0.5 < pmv 
        #         if curr_state[2] < 2:
        #             self.publish_func('PowerUpAC', place='Robot')
        #             print(' - Power Up', end='')
        #     elif pmv > boundary: # 0.4 < pmv < 0.5
        #         if curr_state[2] < 1:
        #             self.publish_func('PowerUpAC', place='Robot')
        #             print(' - Power up', end='')
        #     print()

            



    def timer_callback(self):
        print(f'[{self.agent_name}/{self.place_name}] Status: {STATE_MAP[self.status]}')
        curr_context = self.context_collector.get(self.sub_contexts)

        if self.status != INITIALIZING:
            curr_state = self.preprocess_state(curr_context)
            curr_human_state = self.get_curr_human_state()

            print(curr_state, curr_human_state, self.get_rl_state(curr_state, curr_human_state))
            
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
            self.last_action, self.last_prob_a = self.get_action(self.last_state, deterministic=True)
            self.last_user_context = curr_human_state[:]
            print(f'State: {self.last_state} -> Action Prob: {self.last_prob_a} -> Action: {self.last_action}')
            self.transition(ACTUATING)

        elif self.status == ACTUATING:
            self.expected_ac_mode, self.anchor = self.actuate(self.last_action, curr_state)
            
            self.transition(WAITING)
            time.sleep(5)

        elif self.status == WAITING:
            # if curr_state[2] == self.expected_ac_mode:
            #     if (time.time() - self.anchor) > self.waiting_threshold:
            #         self.last_next_state = self.get_rl_state(curr_state, self.get_curr_human_state())
            #         self.user_feedback = STILL
            #         self.transition(RECORDING)

            if (time.time() - self.anchor) > self.waiting_threshold:
                
                self.user_feedback = STILL
                self.transition(RECORDING)

            if curr_human_state[0] == 0:
                self.rl_agent.finish_path(self.episode_length)
                self.episode_length = 0
                self.transition(TRAINING)



        elif self.status == RECORDING:
            ''' Memory for agent action '''
            self.last_next_state = self.get_rl_state(curr_state, curr_human_state)
            reward = self.reward_function(self.last_action, self.user_feedback)
            

            ''' Memory for unchosen still action '''
            if (self.last_action != STILL) and (reward == POSITIVE_REWARD):
                self.rl_agent.add_memory(self.last_state, STILL, NEGATIVE_REWARD, self.last_state, 0, self.last_prob_a[STILL].item(), 0, self.last_user_context)
                self.episode_length += 1

            self.rl_agent.add_memory(self.last_state, self.last_action, reward, self.last_next_state, 0, self.last_prob_a[self.last_action].item(), 0, self.last_user_context)
            self.episode_length += 1


            if (self.last_action == STILL) or (reward == NEGATIVE_REWARD):
                ''' Transition to user loop '''
                self.last_state = self.get_rl_state(curr_state, curr_human_state)
                _, self.last_prob_a = self.get_action(self.last_state, deterministic=True)
                self.last_user_context = curr_human_state[:]
                self.transition(USER_MONITORING)
            else:
                self.transition(DETERMINING)

        elif self.status == USER_MONITORING:
            print('User actuation loop')

            ''' End user loop when user is gone or user preferred state is reached '''
            if curr_human_state[0] == 0:
                self.rl_agent.finish_path(self.episode_length)
                self.episode_length = 0
                self.transition(TRAINING)

            if (self.user_feedback == STILL):
                self.last_next_state = self.get_rl_state(curr_state, curr_human_state)
                self.rl_agent.add_memory(self.last_state, self.user_feedback, POSITIVE_REWARD, self.last_next_state, 0, self.last_prob_a[self.user_feedback].item(), 1, self.last_user_context)
                self.episode_length += 1
                self.rl_agent.finish_path(self.episode_length)
                self.episode_length = 0
                self.transition(TRAINING)

            
            else: # User feedback is not STILL
                ''' User actuation '''
                self.last_state = self.get_rl_state(curr_state, curr_human_state)
                _, self.last_prob_a = self.get_action(self.last_state, deterministic=True)
                self.last_user_context = curr_human_state[:]
                self.transition(USER_ACTUATING)

        
        elif self.status == USER_ACTUATING:
            self.expected_ac_mode, self.anchor = self.actuate(self.user_feedback, curr_state)
            self.transition(USER_RECORDING)
            time.sleep(5)

        elif self.status == USER_RECORDING:
            self.last_next_state = self.get_rl_state(curr_state, curr_human_state)
            reward = POSITIVE_REWARD

            self.rl_agent.add_memory(self.last_state, self.user_feedback, reward, self.last_next_state, 0, self.last_prob_a[self.user_feedback].item(), 1, self.last_user_context)
            self.episode_length += 1
            self.anchor = time.time()
            self.transition(USER_WAITING)
        
        elif self.status == USER_WAITING:
            if (time.time() - self.anchor) > self.waiting_threshold:
                self.user_feedback = STILL
                self.transition(USER_MONITORING)

        elif self.status == TRAINING:
            for _ in range(3):
                self.rl_agent.update_network()

            self.rl_agent.save_model(self.weight)
            self.rl_agent.save_memory(self.memory_path)

            self.episode += 1

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
            # if self.status == MONITORING:
            #     if trainsition == True:
            #         self.transition(DETERMINING)

        elif name in self.sub_human_contexts:
            dt = datetime.datetime.fromtimestamp(timestamp/1000)
            # print(dt, name, value)
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
            
            if name == 'PowerDownAC':
                self.user_feedback = DOWN
            elif name == 'PowerUpAC':
                self.user_feedback = UP
            else:
                print('error occur in on mesaage for functionality')
                sys.exit()
            if self.status == WAITING:
                self.transition(RECORDING)
            elif self.status == USER_WAITING:
                self.transition(USER_MONITORING)
        else:
            print('error occur in on message')
            sys.exit()

    def get_rl_state(self, state, human_state):
        if self.place_name == 'N1SeminarRoom825':
            ''' Seminar Room Setup '''
            t_low, t_high = 20, 27 
            h_low, h_high = 40, 60
        elif self.place_name == 'N1Lounge8F':
            ''' Lounge Set up '''
            t_low, t_high = 24, 28 
            h_low, h_high = 45, 65


        combined_state = state[:]
        combined_state[0] = (min(max(combined_state[0], t_low), t_high)-t_low)/(t_high-t_low)
        combined_state[1] = (min(max(combined_state[1], h_low), h_high)-h_low)/(h_high-h_low)
        combined_state[2] = combined_state[2]/2.0

        if self.ambient == False:
            combined_state.append(human_state[2]/1.0)
        
        return combined_state

    def get_action(self, state, deterministic=False):
        action, prob_a = self.rl_agent.get_action(state, deterministic)
        ''' Change unmeaningful action to STILL action '''
        if (action == 1) and (state[2] == 0):
            action = 0
        elif (action == 2) and (state[2] == 1):
            action = 0 
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
    # client = ThermalControlAgent(agent_name='ThermalControlAgent', weight='PyLapras/resources/weight/thermal.pt')
    client = ThermalControlAgent(
                # place_name='N1SeminarRoom825',
                # weight='PyLapras/resources/weight/thermal8.pt', 
                # memory_path='PyLapras/resources/weight/memory8.pickle',
                # ambient=True
            )
    # client = ThermalControlAgent(
    #             weight='PyLapras/resources/weight/thermal8.pt', 
    #             memory_path='PyLapras/resources/weight/memory8.pickle',
    #             ambient=False
    #         )
    # client = ThermalControlAgent(agent_name='ThermalControlAgent', weight='PyLapras/resources/weight/thermal.pt', fixed_clothing=0)

    client.loop_forever()
    client.disconnect()
