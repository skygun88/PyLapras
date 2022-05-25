import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import time
import base64
import datetime
from PIL import Image
from io import BytesIO
from agent import LaprasAgent

INITIALIZING, READY, MOVING, ATTENDING, DETECTING, INFERRING = 0, 21, 22, 23, 24, 25
STATE_MAP = {INITIALIZING: 'INITIALIZING', READY: 'READY', MOVING: 'MOVING', ATTENDING: 'ATTENDING', DETECTING: 'DETECTING', INFERRING: 'INFERRING'}
SEAT_MAP = {'S01': (), 'S02': (), 'S03': (), 'S04': (), 'S05': (), 'S06': (), 'S07': (), 'S08': (), 'S09': (), 'S10': ()}

class MonitorRobotAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='MonitorRobotAgent', place_name='N1Lounge8F', timeout_thres=30):
        super().__init__(agent_name, place_name)
        self.status = INITIALIZING
        self.control_alive = False
        self.control_last_alive = 0
        self.timeout_thres = timeout_thres
        self.sub_contexts = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']
        self.curr_attention = None
        self.curr_image = None

        for context in self.sub_contexts:
            self.subscribe(f'{place_name}/context/{context}')

        self.create_timer(self.timer_callback, timer_period=1)
        self.subscribe(f'{place_name}/context/RobotControlAgentOperatingStatus', 2) # RobotControlAgent Alive        
        self.subscribe(f'{place_name}/context/RobotDetectedImage')
        self.publish_context('MonitorRobotAgentOperatingStatus', True, 2)

    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_MAP[self.status]}->{STATE_MAP[next_state]}')          
        self.status = next_state
        self.publish_context(f'{self.agent_name}Status', STATE_MAP[self.status], 2)

    def timer_callback(self):
        self.check_alive() # check inference manager and robot agent is alive
        print(f'[{self.agent_name}/{self.place_name}] Status: {STATE_MAP[self.status]} | ControlAgent: {self.control_alive}')

        if self.status == INITIALIZING:
            ''' State transition: INITIALIZING -> READY '''
            if self.control_alive:
                self.transition(next_state=READY)
                self.publish_func('undocking')
                self.publish_func('robotDetect')
            else:
                return
        elif self.status == READY:
            pass
        elif self.status == MOVING:
            pass
        elif self.status == ATTENDING:
            pass
        elif self.status == DETECTING:
            pass
        elif self.status == INFERRING:
            if self.curr_image == None:
                print('waiting to reeceive the detected image')
                return
            print('there is no user')

            self.transition(next_state=READY)
            self.curr_image = None
        else:
            print('Shuld not be here')
            sys.exit()


    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)

        ''' Topic datas '''
        timestamp, publisher, type, name = dict.get('timestamp'), dict.get('publisher'), dict.get('type'), dict.get('name')
        value = dict.get('value')

        if name == 'RobotControlAgentOperatingStatus':
            ''' RobotStatus Alive update '''
            if value == True:
                self.control_alive = True
                self.control_last_alive = timestamp
            return
 
        elif name == 'RobotDetectedImage':
            self.curr_image = Image.open(BytesIO(base64.b64decode(value)))
            return
        
        if self.status == INITIALIZING:
            pass
        elif self.status == READY:
            if name in self.sub_contexts:
                print(timestamp, name, value)
                if value == 'True':
                    self.curr_attention = SEAT_MAP[name]
                    self.transition(next_state=MOVING)
                    self.publish_func('robotMove', [self.curr_attention[0], self.curr_attention[1]])

        elif self.status == MOVING:
            if name == 'robotComplete' and value == 'move':
                self.transition(next_state=ATTENDING)
                self.publish_func('robotAttend', [self.curr_attention[2]])
        elif self.status == ATTENDING:
            if name == 'robotComplete' and value == 'attending':
                self.transition(next_state=DETECTING)
                self.publish_func('robotDetect')
        elif self.status == DETECTING:
            if name == 'robotComplete' and value == 'detecting':
                self.transition(next_state=INFERRING)
        elif self.status == INFERRING:
            pass
        else:
            print('Shuld not be here')
            sys.exit()

    def check_alive(self):
        curr_ts = self.curr_timestamp()
        # if curr_ts - self.inference_last_alive > self.timeout_thres*1000:
        #     self.inference_alive == False
        print(curr_ts, self.control_last_alive, self.timeout_thres*1000, curr_ts - self.control_last_alive > self.timeout_thres*1000, self.control_alive)
        if curr_ts - self.control_last_alive > self.timeout_thres*1000:
            self.control_alive = False

if __name__ == '__main__':
    client = MonitorRobotAgent()
    client.loop_forever()
    client.disconnect()