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

PYLAPRAS_PATH = os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras'
sys.path.append(PYLAPRAS_PATH)
from agent import LaprasAgent
from utils.state import StateCollector
from utils.db import upload_replay, db_parser, preprocessing
from utils.dqn import DQN

import requests


INITIALIZING, READY, COLLECTING, CONTROLLING, TRAIN = 0, 1, 2, 3, 4
STATE_MAP = {INITIALIZING: 'INITIALIZING', READY: 'READY', COLLECTING: 'COLLECTING', CONTROLLING: 'CONTROLLING', TRAIN: 'TRAIN'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HeaterServiceAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='HeaterServiceAgent', place_name='HomeIO', 
                    n_action=4, time_interval=60, mode='collect', epsilon=0.05,
                    weight=None):
        super().__init__(agent_name, place_name)
        self.status = INITIALIZING
        self.state_collector = StateCollector()        
        self.n_action = n_action
        self.start_ts = self.curr_timestamp()
        self.mode = mode
        self.epsilon = 0.05

        # self.sub_contexts = [   
        #                         'A_Temperature', 'A_Humidity', 'A_HeaterIntensity', 
        #                         'D_Temperature', 'D_Humidity', 'D_HeaterIntensity', 
        #                         # 'E_Temperature', 'E_Humidity', 'E_HeaterPower', 'E_HeaterIntensity'
        #                     ]
        
        self.sub_contexts = ['A_Temperature', 'D_Temperature', 'A_Humidity', 'D_Humidity', 'A_HeaterIntensity', 'D_HeaterIntensity']
        self.history = None
        self.next_history = None

        for context in self.sub_contexts:
            self.subscribe(f'{place_name}/context/{context}')


        self.create_timer(self.timer_callback, timer_period=time_interval)
        self.timer_cnt = 0

        if self.mode == 'control' or self.mode == 'train':
            self.model: DQN = DQN(self.n_action).to(device=device)
            self.target_model: DQN = DQN(self.n_action).to(device=device)
            self.target_model.load_state_dict(self.model.state_dict())

            if not weight == None:
                self.load_model(weight)
            self.target_model.eval()

        if self.mode == 'train':
            self.status = TRAIN

        # self.publish_context('AirconServiceAgentOperatingStatus', STATE_MAP[self.status], 2)

    def get_datetime(self):
        response = requests.get('https://localhost:9797/poll')
        # print(response.)

    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_MAP[self.status]}->{STATE_MAP[next_state]}')          
        self.status = next_state
        # self.publish_context('AirconServiceAgentOperatingStatus', STATE_MAP[self.status], 2)

    def timer_callback(self):
        # print(f'[{self.agent_name}/{self.place_name}] Status: {STATE_MAP[self.status]}')
        curr_state = self.state_collector.get(self.sub_contexts)

        if self.status != INITIALIZING and self.mode != 'train':
            if self.timer_cnt % 15 == 0:
                t1, t2, _, _, i1, i2 = curr_state
                print(f'[{self.start_ts}] [{t1:.2f}, {t2:.2f}, {i1:.1f}, {i2:.1f}]')
            self.collect_replay(curr_state)
            if self.status == CONTROLLING:
                if self.history.shape[0] < 15:
                    self.history = np.append(self.history, np.array(curr_state).reshape(1, -1), axis=0)
                else:
                    self.history = np.append(self.history[1:, :], np.array(curr_state).reshape(1, -1), axis=0)

        if self.status == INITIALIZING:
            if self.mode == 'train':
                self.transition(next_state=TRAIN)
            elif not None in curr_state:
                self.transition(READY)
                
            
            else: 
                return
        elif self.status == READY:
            if self.mode == 'collect':
                self.transition(next_state=COLLECTING)
            elif self.mode == 'control':
                self.transition(next_state=CONTROLLING)
                self.history = np.array(curr_state).reshape(1, -1)

            elif self.mode == 'train':
                self.transition(next_state=TRAIN)
            else:
                print('error occur')
                return 
        elif self.status == COLLECTING:
            if self.timer_cnt % 15 == 0:
                t1, t2, h1, h2, i1, i2 = curr_state
                if (t1+t2)/2 < 23:
                    action = 3
                elif (t1+t2)/2 < 24:
                  action = random.randint(1, 2)
                else:
                    action = 0
                 
                    
                # action = self.get_random_action()
                self.actuate(action)
        
        elif self.status == CONTROLLING:
            if self.timer_cnt % 15 == 0 and self.history.shape[0] == 15:
                dqn_state = preprocessing(self.history)
                tensor_state = torch.Tensor(dqn_state).unsqueeze(0).to(device=device)
                action, q_value = self.get_action(tensor_state, 0.05)
                print(action, q_value)
                self.actuate(action)
                # q_value_sum += q_value



        elif self.status == TRAIN:
            self.train_model_from_db()
            # sys.exit()
            # self.loop_stop()
            self.disconnect()
    
        
        self.timer_cnt += 1

    def load_model(self, path:str):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def train_model_from_db(self):
        replay_memory = db_parser(self.place_name, window_size=15, start=1650475275168)
        # max_epoch = 200000
        max_epoch = 500000
        # max_epoch = 1000000
        minibatch_size = 64
        lr = 1e-4
        gamma = 0.5
        optimizer = optim.RMSprop(self.model.parameters(), lr)
        # print(replay_memory[0][0].shape)

        for epoch in range(max_epoch):
            self.model.train()
            minibatch = random.sample(replay_memory, minibatch_size)
            states = torch.Tensor(np.stack([x[0] for x in minibatch], axis=0)).to(device=device)
            one_hot_actions = F.one_hot(torch.tensor([x[1] for x in minibatch]).to(device=device), num_classes=4)
            next_states = torch.Tensor(np.stack([x[2] for x in minibatch], axis=0)).to(device=device)
            rewards = torch.Tensor([x[3] for x in minibatch]).to(device=device)

            q_values = torch.sum(self.model(states)*one_hot_actions, 1)
            # ys = rewards + gamma*torch.amax(self.target_model(next_states).detach(), 1)
            # ys = rewards + gamma*torch.argmax(self.target_model(next_states).detach(), 1)
            ys = rewards + gamma*torch.argmax(self.model(next_states).detach(), 1)


            criterion = nn.SmoothL1Loss()
            loss = criterion(q_values, ys)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            self.model.eval()

            minibatch.clear()
            if epoch % 100 == 0:
                print(f'[{epoch}] loss: {loss.item()}')
                
            # if epoch % 5000 == 0:
            #     self.target_model.load_state_dict(self.model.state_dict())
            # if epoch % 5000==0:
            #     self.save_model(PYLAPRAS_PATH+'/resources/weight/heater.pt')
        
        self.save_model(PYLAPRAS_PATH+'/resources/weight/heater.pt')
        


    def collect_replay(self, state):
        ts = self.curr_timestamp()
        # t1, t2, h1, h2, i1, i2 = state
        # print(f'[{self.start_ts}] [{t1:.2f}, {t2:.2f}, {h1}, {h2}, {i1:.2f}, {i2:.2f}]')
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
            output = self.model(state.to(device))
            if random.random() < epsilon: 
                action = random.randint(0, self.n_action-1)
            else:
                action = torch.argmax(output).item()
            # max_q = torch.amax(output).item()
            max_q = torch.argmax(output).item()
            print(output)

        return action, max_q
    
    def actuate(self, action):
        if action == 0:
            self.publish_func('HeaterA_DOWN')
            self.publish_func('HeaterD_DOWN')
        elif action == 1:
            self.publish_func('HeaterA_UP')
            self.publish_func('HeaterD_DOWN')

        elif action == 2:
            self.publish_func('HeaterA_DOWN')
            self.publish_func('HeaterD_UP')

        elif action == 3:
            self.publish_func('HeaterA_UP')
            self.publish_func('HeaterD_UP')



if __name__ == '__main__':
    # client = HeaterServiceAgent(mode='collect', time_interval=0.1)
    # client = HeaterServiceAgent(mode='train', time_interval=1, weight=PYLAPRAS_PATH+'/resources/weight/heater.pt')
    # client = HeaterServiceAgent(mode='train', time_interval=1)

    client = HeaterServiceAgent(mode='control', time_interval=0.1, weight=PYLAPRAS_PATH+'/resources/weight/heater.pt')
    client.loop_forever()
    client.disconnect()