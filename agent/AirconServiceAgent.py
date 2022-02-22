import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import time
import random
import datetime
from agent import LaprasAgent
from utils.state import StateCollector
from utils.db import upload_replay

INITIALIZING, READY = 0, 1
STATE_MAP = {0: 'INITIALIZING', 1: 'READY'}


class AirconServiceAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='AirconServiceAgent', place_name='N1Lounge8F', n_action=4):
        super().__init__(agent_name, place_name)
        self.status = INITIALIZING
        self.state_collector = StateCollector()        
        self.n_action = n_action
        self.start_ts = self.curr_timestamp()
        # self.replay_memory = 
        self.sub_contexts = ['sensor0_Temperature', 'sensor1_Temperature', 'sensor0_Humidity', 'sensor1_Humidity', 'Aircon0Power', 'Aircon1Power']

        for context in self.sub_contexts:
            self.subscribe(f'N1Lounge8F/context/{context}')

        self.create_timer(self.timer_callback, timer_period=60)
        # self.publish_context('AirconServiceAgentOperatingStatus', STATE_MAP[self.status], 2)

    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_MAP[self.status]}->{STATE_MAP[next_state]}')          
        self.status = next_state
        # self.publish_context('AirconServiceAgentOperatingStatus', STATE_MAP[self.status], 2)

    def timer_callback(self):
        print(f'[{self.agent_name}/{self.place_name}] Status: {STATE_MAP[self.status]}')
        curr_state = self.state_collector.get(self.sub_contexts)

        if self.status == INITIALIZING:
            print(dict(zip(self.sub_contexts, curr_state)))

            if not None in curr_state:
                self.transition(READY)
            else: 
                return
        else:
            self.collect_replay(curr_state)
            if random.random() < 0.2:
                action = self.get_random_action()
                self.actuate(action)

            

    def collect_replay(self, state):
        ts = self.curr_timestamp()
        print(ts, state)
        upload_replay(self.start_ts, ts, state)
    
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

    def actuate(self, action):
        if action == 0:
            client.publish_func('StopAircon0')
            client.publish_func('StopAircon1')
        elif action == 1:
            client.publish_func('StartAircon0')
            client.publish_func('StopAircon1')

        elif action == 2:
            client.publish_func('StopAircon0')
            client.publish_func('StartAircon1')

        elif action == 3:
            client.publish_func('StartAircon0')
            client.publish_func('StartAircon1')



if __name__ == '__main__':
    client = AirconServiceAgent()
    client.loop_forever()
    client.disconnect()