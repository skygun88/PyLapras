import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import time
import datetime
from agent import LaprasAgent

INITIALIZING, READY = 0, 1
STATE_MAP = {0: 'INITIALIZING', 1: 'READY'}

class MonitorRobotAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='MonitorRobotAgent', place_name='N1Lounge8F'):
        super().__init__(agent_name, place_name)
        self.status = INITIALIZING
        # self.replay_memory = 
        self.sub_contexts = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']

        for context in self.sub_contexts:
            self.subscribe(f'N1Lounge8F/context/{context}')

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)

        ''' Topic datas '''
        timestamp, publisher, type, name = dict.get('timestamp'), dict.get('publisher'), dict.get('type'), dict.get('name')
        value = dict.get('value')

        print(timestamp, name, value)
    
if __name__ == '__main__':
    client = MonitorRobotAgent()
    client.loop_forever()
    client.disconnect()