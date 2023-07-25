import os
import sys
import json
import time
import gzip
import random
import pickle
import datetime
import argparse
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from agent.LaprasAgent import LaprasAgent
from utils.configure import *

INITIALIZE, READY, ACCEPT, DETERMINE, EXECUTE, WAIT, TRAIN, RECOVER, DONE = 0, 1, 2, 3, 4, 5, 6, 7, 8
DISCONNECT, FAIL = -1, -2

STATE_DICT = {
    INITIALIZE: "INITIALIZE", 
    READY: "READY",
    ACCEPT: "ACCEPT",
    DETERMINE: "DETERMINE", 
    EXECUTE: 'EXECUTE',
    WAIT: "WAIT", 
    TRAIN: "TRAIN",
    RECOVER: "RECOVER", 
    DONE: "DONE", 
    DISCONNECT: "DISCONNECT", 
    FAIL: "FAIL"
}

N_AC_STATES = 4 # The number of aircon's state - 24, 25, 26, 27
N_FAN_STATES = 2 # The number of fan's state - ON/OFF
N_ACTIONS = 3 # The nunmber of actions of each agent - AC: STILL, UP, DOWN / Fan: STILL, ON, OFF 
ALPHA_init = 0.75
GAMMA = 0.25
EPSILON = 0.05
RESET_THRES = 0.6

AC_IDX, FAN_IDX = 0, 1
AC_24, AC_25, AC_26, AC_27 = 0, 1, 2, 3
FAN_OFF, FAN_ON = 0, 1
STILL, AC_UP, AC_DOWN = 0, 1, 2
STILL, FAN_TURN_ON, FAN_TURN_OFF = 0, 1, 2

''' Convert actual device state to RL state '''
def get_rl_state(ac_power: bool, ac_temp: int, fan_power: bool):
    ac_state = max(min(ac_temp-24, 3), 0) if ac_power == True else AC_27
    fan_state = 1 if fan_power == True else 0
    return [ac_state, fan_state]

''' Determine agent's action based on epsilon greedy method '''
def get_action(state: list, q_table: np.ndarray, epsilon: float):   
    if random.random() < epsilon:
        action = random.randrange(q_table.shape[2])
    else:
        policy = q_table[state[AC_IDX], state[FAN_IDX], :]
        action = np.argmax(policy)
    return action

''' Update Q-table based on Q-learning '''
def update_table(q_table: np.ndarray, state: list, action: int, next_state: list, reward: int, gamma: float, alpha: float):
    q_value = q_table[state[AC_IDX], state[FAN_IDX], action]
    next_q_value = np.max(q_table[next_state[AC_IDX], next_state[FAN_IDX], :])
    q_table[state[AC_IDX], state[FAN_IDX], action] = q_value + alpha * (reward + gamma*next_q_value - q_value) # Refer Q-learning update formula
    return q_table

class QlearningAgent(LaprasAgent):
    def __init__(self, 
        agent_name='QlearningAgent', 
        place_name='N1Lounge8F', 
        wait_time=1,
        ):
        super().__init__(agent_name, place_name)

        ''' Lapras Setup '''
        self.subscribe(f'{place_name}/context/AirconAlive', 0)
        self.subscribe(f'{place_name}/context/FanAlive', 0)
        self.subscribe(f'{place_name}/context/AirconPower', 0)
        self.subscribe(f'{place_name}/context/AirconTemp', 0)
        self.subscribe(f'{place_name}/context/FanPower', 0)
        self.subscribe(f'{place_name}/functionality/Qstart')
        self.subscribe(f'{place_name}/functionality/AirconFeedback')
        self.subscribe(f'{place_name}/functionality/FanFeedback')
        self.create_timer(self.timer_callback, timer_period=1)

        ''' initial setup '''
        self.wait_time = wait_time

        ''' Path setup '''
        self.result_dir = os.path.join(TESTER_PATH, 'video')
        self.now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")    

        self.state = INITIALIZE
        self.ac_power = None
        self.ac_temp = None
        self.fan_power = None
        self.ac_last_alive = -1
        self.fan_last_alive = -1
        self.timer_cnt = 0
        self.start_ts = time.time()
        self.accepted = False
        self.capacity = 5
        self.rl_state = [None, None]
        self.next_rl_state = [None, None]
        self.wait_start = -1
        self.curr_user = None

        ''' Q-table '''
        self.qtable_json_path = os.path.join(RESOURCE_PATH, 'qtable_paths.json')
        if os.path.isfile(self.qtable_json_path):
            self.qtable_paths = self.load_qtable_path()
        else:
            self.qtable_paths = dict()
            self.save_qtable_path()
        print(f'Current Q-tables: {self.qtable_paths}')

    def save_qtable_path(self):
        with open(self.qtable_json_path, 'w') as f:
            json.dump(self.qtable_paths, f)
            f.close()

    def load_qtable_path(self):
        with open(self.qtable_json_path, "r") as f:
            qtable_paths = json.load(f)
            f.close()
        return qtable_paths
    
    def load_qtable(self, user_name):
        qtable_path = self.qtable_paths.get(user_name, '')
        if qtable_path == '':
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            qtable_path = os.path.join(RESOURCE_PATH, f'Qtables/{now}.pickle')
            self.qtable_paths[user_name] = qtable_path
            self.save_qtable_path()
            qtables = [np.zeros((N_AC_STATES, N_FAN_STATES, N_ACTIONS)) for _ in range(2)]
            for i in range(N_AC_STATES):
                for j in range(N_FAN_STATES):
                    qtables[0][i, j, 0] = 1.0
                    qtables[1][i, j, 0] = 1.0
            self.save_qtable(user_name, qtables)
        else:
            with gzip.open(qtable_path,'rb') as f:
                qtables = pickle.load(f)
                f.close()
        return qtables

    def save_qtable(self, user_name, qtables):
        qtable_path = self.qtable_paths.get(user_name, '')
        if not qtable_path == '':
            with gzip.open(qtable_path, 'wb') as f:
                pickle.dump(qtables, f)
                f.close()
        else:
            print('save filename error')
            sys.exit()

    def timer_callback(self):
        self.timer_cnt += 1
        curr_ts = time.time()
        duration = curr_ts - self.start_ts

        self.connected = self.check_connected()
            
        if not self.connected:
            print('Conection Failed')
            return 

        if int(duration) % 2 == 0:
            print(f'[{STATE_DICT[self.state]}] AC: {self.ac_power}, {self.ac_temp}|{self.ac_last_alive} / Fan: {self.fan_power}|{self.fan_last_alive}')

        if self.state == READY:
            self.ready_action()
        elif self.state == ACCEPT:
            self.accept_action()
        elif self.state == DETERMINE:
            self.determine_action()
        elif self.state == EXECUTE:
            self.execute_action()
        elif self.state == WAIT:
            self.wait_action()
        elif self.state == TRAIN:
            self.train_action()
        elif self.state == RECOVER:
            self.recover_action()
        elif self.state == DONE:
            self.done_action()
        else:
            print(f'error occur - {STATE_DICT[self.state]}')
            self.disconnect()
            sys.exit()


    def ready_action(self):
        if self.accepted == True:
            self.accepted = False
            self.transition(ACCEPT)


    def accept_action(self):
        self.qtables = self.load_qtable(self.curr_user)
        self.reset_variable()
        print(self.qtables)
        self.transition(DETERMINE)
        

    def determine_action(self):
        ''' Action determination by Q-table '''
        self.rl_state = get_rl_state(self.ac_power, self.ac_temp, self.fan_power)
        self.AC_a = get_action(self.rl_state, self.qtables[AC_IDX], EPSILON)
        self.FAN_a = get_action(self.rl_state, self.qtables[FAN_IDX], EPSILON)
        print(f'State: {self.rl_state} | Actions: {self.AC_a}, {self.FAN_a}')

        self.next_rl_state = self.rl_state[:]
        self.real_AC_a, self.real_FAN_a = STILL, STILL
        ''' AC execution '''
        if self.AC_a == AC_UP:
            if (self.ac_power == True) and (self.ac_temp < 27):
                self.publish_func('TempUpAircon')
                self.next_rl_state[AC_IDX] += 1
                self.real_AC_a = self.AC_a
        elif self.AC_a == AC_DOWN:
            if self.ac_power == True:
                if self.ac_temp > 24:
                    self.publish_func('TempDownAircon')
                    self.next_rl_state[AC_IDX] -= 1
                    self.real_AC_a = self.AC_a
            else:
                self.publish_func('SetTempAircon', arguments=[26])
                self.next_rl_state[AC_IDX] -= 1
                self.real_AC_a = self.AC_a
        
        ''' Fan execution '''
        if self.FAN_a == FAN_TURN_ON:
            if self.fan_power == False:
                self.publish_func('FanOn')
                self.next_rl_state[FAN_IDX] = FAN_ON
                self.real_FAN_a = self.FAN_a
        elif self.FAN_a == FAN_TURN_OFF:
            if self.fan_power == True:
                self.publish_func('FanOff')
                self.next_rl_state[FAN_IDX] = FAN_OFF
                self.real_FAN_a = self.FAN_a

        self.transition(EXECUTE)


    def execute_action(self):
        print(self.rl_state, self.next_rl_state)
        curr_rl_state = get_rl_state(self.ac_power, self.ac_temp, self.fan_power)
        if self.next_rl_state == curr_rl_state:
            self.publish_last_actions()
            self.wait_start = time.time()
            self.AC_reward, self.FAN_reward = None, None
            self.transition(WAIT)


    def wait_action(self):
        curr_ts = time.time()

        ''' If user doesn't give feedback within waiting time -> consider user accept the last action '''
        if (curr_ts - self.wait_start) > self.wait_time:
            self.AC_reward = self.AC_reward if self.AC_reward != None else 1
            self.FAN_reward = self.FAN_reward if self.FAN_reward != None else 1
            self.transition(TRAIN)
        else:
            ''' When the agent get user feedbacks -> transit the state '''
            if (self.AC_reward != None) and (self.FAN_reward != None):
                self.transition(TRAIN)
            else:
                print(f'Remaining Time: {self.wait_time - (curr_ts - self.wait_start)} s')

    def train_action(self):
        ''' Update learning rate '''
        self.AC_alpha = self.AC_alpha / self.AC_t
        self.AC_correct = 1 if self.AC_reward == 1 else 0
        self.AC_acc_table.append(self.AC_correct)

        self.FAN_alpha = self.FAN_alpha / self.FAN_t
        self.FAN_correct = 1 if self.FAN_reward == 1 else 0
        self.FAN_acc_table.append(self.FAN_correct)

        print(f'Actions: ({self.AC_a}, {self.FAN_a}) | State transition: {self.rl_state} -> {self.next_rl_state} | Rewards: ({self.AC_reward}, {self.FAN_reward})')
        print(f'Time variable: {(self.AC_t, self.FAN_t)} | Learning rate: {(self.AC_alpha, self.FAN_alpha)} | Last 5 correctness: {(self.AC_acc_table, self.FAN_acc_table)}')

        self.qtables[AC_IDX] = update_table(self.qtables[AC_IDX], self.rl_state, self.AC_a, self.next_rl_state, self.AC_reward, GAMMA, self.AC_alpha)
        self.qtables[FAN_IDX] = update_table(self.qtables[FAN_IDX], self.rl_state, self.FAN_a, self.next_rl_state, self.FAN_reward, GAMMA, self.FAN_alpha)

        self.save_qtable(self.curr_user, self.qtables)


        print(self.qtables)


        ''' Update parameters '''
        self.AC_t += 1
        self.FAN_t += 1
        self.iteration += 1

        if sum(self.AC_acc_table)/len(self.AC_acc_table) < RESET_THRES:
            self.AC_t = 1
            self.AC_alpha = ALPHA_init
        if sum(self.FAN_acc_table)/len(self.FAN_acc_table) < RESET_THRES:
            self.FAN_t = 1
            self.FAN_alpha = ALPHA_init    


        ''' State transition branches '''
        if ((self.AC_a == STILL and self.AC_reward == 1) and (self.FAN_a == STILL and self.FAN_reward == 1)):
            ''' Scenario is done '''
            self.transition(DONE)

        else:

            if (self.AC_reward < 0) and (self.rl_state[AC_IDX] != self.next_rl_state[AC_IDX]):
                self.publish_func('SetTempAircon', arguments=[self.rl_state[AC_IDX]+24])
            else:
                self.rl_state[AC_IDX] = self.next_rl_state[AC_IDX]
            if (self.FAN_reward < 0) and (self.rl_state[FAN_IDX] != self.next_rl_state[FAN_IDX]):
                if self.rl_state[FAN_IDX] == FAN_ON:
                    self.publish_func('FanOn')
                else:
                    self.publish_func('FanOff')
            else:
                self.rl_state[FAN_IDX] = self.next_rl_state[FAN_IDX]
            
            self.transition(RECOVER)


    def recover_action(self):
        print(self.rl_state)
        curr_rl_state = get_rl_state(self.ac_power, self.ac_temp, self.fan_power)
        if self.rl_state == curr_rl_state:
            self.transition(DETERMINE)
    
    def done_action(self):
        self.publish_context('Qdone', value=self.curr_user)
        self.curr_user = None
        self.transition(READY)

    def on_message(self, client, userdata, msg):
        try:
            dict_string = str(msg.payload.decode("utf-8"))
            msg_dict = json.loads(dict_string)
        except Exception as e:
            print(f'Error occur when unpack mqtt msg - {e}')
            print(f'Tried msg : {msg}')
            return

        context_name = msg_dict.get('name')
        
        # Alive Message
        if context_name == 'AirconAlive':
            self.ac_last_alive = msg_dict.get('timestamp')

        elif context_name == 'FanAlive':
            self.fan_last_alive = msg_dict.get('timestamp')
        
        # Device state update
        elif context_name == 'AirconPower':
            self.ac_power = msg_dict.get('value')
            print('get_ac')
        elif context_name == 'AirconTemp':
            self.ac_temp = msg_dict.get('value')
        elif context_name == 'FanPower':
            self.fan_power = msg_dict.get('value')
            print('get_fan')

        # From FeedbackAgent
        elif context_name == 'Qstart':
            arguments = msg_dict['arguments']
            if self.state in [INITIALIZE, READY]:
                self.accepted = True
                self.curr_user = arguments[0]
        
        elif context_name == 'AirconFeedback':
            arguments = msg_dict['arguments']
            if self.state == WAIT:
                if (arguments[0] == self.curr_user) and (self.AC_reward == None):
                    self.AC_reward = 1 if arguments[1] == 'accept' else -1
        
        elif context_name == 'FanFeedback':
            arguments = msg_dict['arguments']
            if self.state == WAIT:
                if (arguments[0] == self.curr_user) and (self.FAN_reward == None):
                    self.FAN_reward = 1 if arguments[1] == 'accept' else -1

        else:  
            print('wrong')


    def check_connected(self):
        now = int(time.time()*1000)
        connected = ((now - self.ac_last_alive) < 1000*30) and ((now - self.fan_last_alive) < 1000*30)
        if connected and (self.state == INITIALIZE):
            self.transition(READY)
        return connected
    
    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_DICT[self.state]}->{STATE_DICT[next_state]}')                  
        self.state = next_state

    
    def reset_variable(self):
        self.iteration = 1
        self.AC_alpha, self.FAN_alpha = ALPHA_init, ALPHA_init 
        self.AC_a, self.AC_reward = None, None
        self.FAN_a, self.FAN_reward = None, None
        self.AC_acc_table, self.FAN_acc_table = deque(maxlen=self.capacity), deque(maxlen=self.capacity)
        self.AC_t, self.FAN_t = 1, 1

    def publish_last_actions(self):
        self.publish_context('LastAirconAction', value=int(self.real_AC_a))
        self.publish_context('LastFanAction', value=int(self.real_FAN_a))



    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QlearningAgent')
    parser.add_argument('-w', '--wait', type=int, default=60)


    args = parser.parse_args()

    wait_time = args.wait


    client = QlearningAgent(
        wait_time=wait_time,
    )
    client.loop_forever()
    client.disconnect()


