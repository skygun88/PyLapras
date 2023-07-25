import os
from symbol import pass_stmt 
import sys
import json
import time
import torch
import random
import datetime
import psutil

import pickle
import gzip

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
# from agent.controller.thermal_model import PPOAgent
from agent.controller.thermal_model_ddqn import DDQNAgent
from utils.comfort import PMV
from utils.configure import *
from utils.measurement import *


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
SIT, STAND = 0, 1
INNER, OUTER = 0, 1
MIN_TEM, MAX_TEM = 25, 28 


class ThermalControlAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='ThermalControlAgent', place_name='Robot', sensor_place='N1Lounge8F',
                    n_action=3, 
                    time_interval=1, 
                    weight_paths=None, 
                    fixed_clothing=None, 
                    memory_paths=None,
                    ambient=False
                ):
        super().__init__(agent_name, place_name)
        self.status = INITIALIZING

        pid = os.getpid()
        current_process = psutil.Process(pid)
        self.current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
        self.context_collector = StateCollector()
        self.human_context_collector = StateCollector() 
        self.ambient = ambient
        self.n_action = n_action
        self.n_state = 3 if self.ambient else 5
        self.start_ts = self.curr_timestamp()
        self.weight_paths = weight_paths
        self.memory_paths = memory_paths
        self.sensor_place = sensor_place

        if weight_paths == None:
            self.weights = {SIT: {INNER: "", OUTER: ""}, STAND: {INNER: "", OUTER: ""}}
            self.weight_paths = os.path.join(WEIGHT_PATH, f'weight_paths_{self.start_ts}.json')
        else:
            with open(weight_paths, "r") as f:
                self.weights = json.load(f)
                f.close()
        
        if memory_paths == None:
            self.memories = {SIT: {INNER: "", OUTER: ""}, STAND: {INNER: "", OUTER: ""}}
            self.memory_paths = os.path.join(WEIGHT_PATH, f'memories_paths_{self.start_ts}.json')
            
        else:
            with open(memory_paths, "r") as f:
                self.memories = json.load(f)
                f.close()

        
        self.rl_agents = {}
        for act in [SIT, STAND]:
            self.rl_agents[act] = {}
            for clo in [INNER, OUTER]:
                self.rl_agents[act][clo] = DDQNAgent(action_size=self.n_action, input_size=self.n_state)
                if self.weights[act][clo] != "":
                    if os.path.isfile(self.weights[act][clo]):
                        self.rl_agents[act][clo].load_model(self.weights[act][clo])
                else:
                    self.weights[act][clo] = os.path.join(WEIGHT_PATH, f'thermal_{act}_{clo}_{self.start_ts}.pt')
                
                if self.memories[act][clo] != "":
                    if os.path.isfile(self.memories[act][clo]):
                        self.rl_agents[act][clo].load_memory(self.memories[act][clo])
                else:
                    self.memories[act][clo] = os.path.join(WEIGHT_PATH, f'memory_{act}_{clo}_{self.start_ts}.pickle')
                    
        with open(self.weight_paths, 'w') as f:
            json.dump(self.weights, f)
            f.close()
        with open(self.memory_paths, 'w') as f:
            json.dump(self.memories, f)
            f.close()

        self.rl_agent = None
        self.weight = None
        self.memory = None
        self.fixed_clothing = fixed_clothing
        

        self.sub_contexts = ['sensor0_Temperature', 'sensor1_Temperature', 'sensor0_Humidity', 'sensor1_Humidity', 'AirconPower', 'AirconTemp']
        self.sub_funcs = ['TempUpAC', 'TempDownAC', 'TempStillAC']
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
        self.log_data = {}


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
        self.publish_context('AgentStatus', value=STATE_MAP[self.status])

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



    def timer_callback(self):
        
        curr_context = self.context_collector.get(self.sub_contexts)

        if self.status != INITIALIZING:
            curr_state = self.preprocess_state(curr_context)
            curr_human_state = self.get_curr_human_state()
            print(f'[{self.agent_name}/{self.place_name}] Status: {STATE_MAP[self.status]} | {curr_human_state}')
            # print(curr_state, curr_human_state, self.get_rl_state(curr_state, curr_human_state))
            
            self.collect_replay(curr_state, curr_human_state)

            


        if self.status == INITIALIZING:
            print(f'[{self.agent_name}/{self.place_name}] Status: {STATE_MAP[self.status]}')
            if not None in curr_context:
                self.transition(READY)
            else: 
                return
        elif self.status == READY:
            if curr_human_state[0] == 1 and curr_human_state[2] != -1:
                self.episode_length = 0
                self.curr_agent_act = 0
                self.curr_human_act = 0
                
                
                act, clo = curr_human_state[1], curr_human_state[2]
                self.rl_agent = self.rl_agents[act][clo]
                self.weight = self.weights[act][clo]
                self.memory = self.memories[act][clo]
                self.log_data[self.episode] = {}
                self.log_data[self.episode]['human_state'] = [act, clo]

                self.transition(DETERMINING)

        elif self.status == DETERMINING:
            self.last_state = self.get_rl_state(curr_state, curr_human_state)
            self.last_action, self.last_prob_a = self.get_action(self.last_state, deterministic=True)
            self.last_user_context = curr_human_state[:]

            if self.last_action != STILL:
                self.curr_agent_act += 1

            print(f'State: {self.last_state} -> Action Prob: {self.last_prob_a} -> Action: {self.last_action}')
            self.transition(ACTUATING)

        elif self.status == ACTUATING:
            self.expected_ac_mode, self.anchor = self.actuate(self.last_action, curr_state)
            time.sleep(10)
            self.transition(WAITING)

        elif self.status == WAITING:
            if (curr_human_state[1] != self.last_user_context[1]) or (curr_human_state[2] != self.last_user_context[2]):
                if self.episode_length > 0:
                    self.rl_agent.finish_path(self.episode_length)
                    self.episode_length = 0
                self.transition(TRAINING)

            if (time.time() - self.anchor) > self.waiting_threshold:
                
                self.user_feedback = STILL
                self.transition(RECORDING)

            if curr_human_state[0] == 0:
                if self.episode_length > 0:
                    self.rl_agent.finish_path(self.episode_length)
                    self.episode_length = 0
                self.transition(TRAINING)



        elif self.status == RECORDING:
            ''' Memory for agent action '''
            self.last_next_state = self.get_rl_state(curr_state, curr_human_state)
            reward = self.reward_function(self.last_action, self.user_feedback)
            
            print(f'Reward: {reward}')
            ''' Memory for unchosen still action '''
            if (self.last_action != STILL) and (reward == POSITIVE_REWARD):
                self.rl_agent.add_memory(self.last_state, STILL, NEGATIVE_REWARD, self.last_state, 0, 0, self.last_user_context, self.last_prob_a)
                self.episode_length += 1
                self.rl_agent.update_network()

            self.rl_agent.add_memory(self.last_state, self.last_action, reward, self.last_next_state, 0, 0, self.last_user_context, self.last_prob_a)
            self.episode_length += 1
            self.rl_agent.update_network()
        

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
                if self.episode_length > 0:
                    self.rl_agent.finish_path(self.episode_length)
                    self.episode_length = 0
                self.transition(TRAINING)

            if (curr_human_state[1] != self.last_user_context[1]) or (curr_human_state[2] != self.last_user_context[2]):
                if self.episode_length > 0:
                    self.rl_agent.finish_path(self.episode_length)
                    self.episode_length = 0
                self.transition(TRAINING)

            if (self.user_feedback == STILL):
                self.last_next_state = self.get_rl_state(curr_state, curr_human_state)
                self.rl_agent.add_memory(self.last_state, self.user_feedback, POSITIVE_REWARD, self.last_next_state, 0, 1, self.last_user_context, self.last_prob_a)
                self.episode_length += 1
                self.rl_agent.finish_path(self.episode_length)
                self.episode_length = 0
                self.transition(TRAINING)
            
            else: # User feedback is not STILL
                ''' User actuation '''
                self.curr_human_act += 1
                self.last_state = self.get_rl_state(curr_state, curr_human_state)
                _, self.last_prob_a = self.get_action(self.last_state, deterministic=True)
                self.last_user_context = curr_human_state[:]
                self.transition(USER_ACTUATING)

        
        elif self.status == USER_ACTUATING:
            self.expected_ac_mode, self.anchor = self.actuate(self.user_feedback, curr_state)
            time.sleep(10)
            self.transition(USER_RECORDING)
            

        elif self.status == USER_RECORDING:
            self.last_next_state = self.get_rl_state(curr_state, curr_human_state)
            reward = POSITIVE_REWARD

            self.rl_agent.add_memory(self.last_state, self.user_feedback, reward, self.last_next_state, 0, 1, self.last_user_context, self.last_prob_a)
            
            self.episode_length += 1
            self.rl_agent.update_network()
            self.anchor = time.time()
            self.transition(USER_WAITING)
        
        elif self.status == USER_WAITING:
            if (curr_human_state[1] != self.last_user_context[1]) or (curr_human_state[2] != self.last_user_context[2]):
                if self.episode_length > 0:
                    self.rl_agent.finish_path(self.episode_length)
                    self.episode_length = 0
                self.transition(TRAINING)

            if (time.time() - self.anchor) > self.waiting_threshold:
                self.user_feedback = STILL
                self.transition(USER_MONITORING)

        elif self.status == TRAINING:
            print(self.rl_agent.memory['state'])

            self.rl_agent.update_network()
            self.rl_agent.update_target()

            self.rl_agent.save_model(self.weight)
            self.rl_agent.save_memory(self.memory)

            self.save_log()

            self.episode += 1



            '''  Reset '''
            self.human_exist_queue = [0]*5
            self.human_activity_queue = [-1]*10
            self.human_clothing_queue = [-1]*10
            
            ''' Randomized '''
            random_ac = random.choice([0, 1, 2, 3, 4])
            self.set_ac(random_ac)

            print(f'Next AC - {random_ac}')
            time.sleep(30)

            self.transition(READY)

            
        else:
            print('Error Occur')
            sys.exit()
    
        
        self.timer_cnt += 1
        print(f'Memory in timer- {check_memory_usage():.3f} GB')


    def save_log(self):
        self.log_data[self.episode]['agent_act'] = self.curr_agent_act
        self.log_data[self.episode]['human_act'] = self.curr_human_act

        path = os.path.join(WEIGHT_PATH, f'log_{self.start_ts}.json')

        with open(path, 'w') as f:
            json.dump(self.log_data, f)
            f.close()


    '''
    State Definition
    1. Average Temperature Value: [~25.5, 25.5~26.5, 26.5~] => [0, 1, 2]
    2. Average Humidity Value: [~52, 52~56, 56~] => [0, 1, 2]
    3. A.C Mode: [OFF, 26, 24] => [0, 1, 2]
    4. OutFit: [Jacket OFF, Jacket ON] => [0, 1]
    '''
    # def preprocess_state(self, context):
    #     sensor0_tem = context[0]
    #     sensor1_tem = context[1]
    #     sensor0_hum = context[2]
    #     sensor1_hum = context[3]
    #     ac0_power = context[4]
    #     ac1_power = context[5]

    #     temperature = (sensor0_tem+sensor1_tem)/2
    #     humidity = (sensor0_hum+sensor1_hum)/2
    #     ac_mode = ac0_power + ac1_power

    #     return [temperature, humidity, ac_mode]

    def preprocess_state(self, context):
        sensor0_tem = context[0]
        sensor1_tem = context[1]
        sensor0_hum = context[2]
        sensor1_hum = context[3]
        ac_power = context[4]
        ac_temperature = context[5]

        temperature = (sensor0_tem+sensor1_tem)/2
        humidity = (sensor0_hum+sensor1_hum)/2
        ac_mode = 0
        
        if (ac_power == True) and (ac_temperature != None):
                ac_map = {28: 1, 27: 2, 26: 3, 25: 4}
                ac_mode = ac_map[ac_temperature]

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

    def set_ac(self, mode):
        print(f'set {mode}')
        ac_map = {1: 28, 2: 27, 3: 26, 4: 25}
        if mode == 0:
            self.publish_func('TurnOffAircon', place=self.sensor_place)
        elif mode in [1, 2, 3, 4] :
            self.publish_func('TurnOnAircon', arguments=[], place=self.sensor_place)
            set_point = ac_map[mode]
            self.publish_func('SetTempAircon', arguments=[set_point], place=self.sensor_place)
        else:
            print('error occur in SET AC')
            sys.exit()


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
            
            if name == 'TempDownAC':
                self.user_feedback = DOWN
            elif name == 'TempUpAC':
                self.user_feedback = UP
            elif name == 'TempStillAC': 
                self.user_feedback = STILL
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
        if self.sensor_place == 'N1SeminarRoom825':
            ''' Seminar Room Setup '''
            t_low, t_high = 20, 27 
            h_low, h_high = 40, 60
        elif self.sensor_place == 'N1Lounge8F':
            ''' Lounge Set up '''
            t_low, t_high = 24, 28 
            h_low, h_high = 40, 60


        combined_state = state[:]
        combined_state[0] = (min(max(combined_state[0], t_low), t_high)-t_low)/(t_high-t_low)
        combined_state[1] = (min(max(combined_state[1], h_low), h_high)-h_low)/(h_high-h_low)
        combined_state[2] = combined_state[2]/(MAX_TEM-MIN_TEM-1)

        if self.ambient == False:
            combined_state.append(human_state[1]/1.0)
            combined_state.append(human_state[2]/1.0)
        
        return combined_state

    def get_action(self, state, deterministic=False):
        action, prob_a = self.rl_agent.get_action(state)
        ''' Change unmeaningful action to STILL action '''
        if (action == DOWN) and (state[2] == 1):
            action = STILL
        elif (action == UP) and (state[2] == 0):
            action = STILL
        if self.rl_agent.memory['count'] < 10:
            action = STILL

        valid_action = action
        if self.last_action != None:
            if (action == UP) and (self.last_action == DOWN):
                valid_action = STILL
            elif (action == DOWN) and (self.last_action == UP):
                valid_action = STILL
        
        return valid_action, prob_a
    
    def actuate(self, action, curr_state):
        print(f'Actuation - {action}, {curr_state}')
        ac_mode = curr_state[2]
        
        expected_ac_mode = ac_mode

        if action == DOWN:
            if ac_mode < 1:
                self.publish_func('TurnOnAircon', place=self.sensor_place)
                expected_ac_mode += 1
            elif ac_mode < 4:
                self.publish_func('TempDownAircon', place=self.sensor_place)
                expected_ac_mode += 1
        elif action == UP:
            if ac_mode > 1:
                self.publish_func('TempUpAircon', place=self.sensor_place)
                expected_ac_mode -= 1
            elif ac_mode > 0:
                self.publish_func('TurnOffAircon', place=self.sensor_place)
                expected_ac_mode -= 1
            
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
