import os
import cv2
import sys
import json
import time
import base64
import platform
import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") if "Linux" in platform.platform() else None
from agent.LaprasAgent import LaprasAgent
from utils.configure import *

class WebFeedbackCollectorAgent(LaprasAgent):
    def __init__(self, agent_name='FeedbackCollectorAgent', place_name='Robot', sensor_place='N1Lounge8F'):
        super().__init__(agent_name, place_name)
        self.sensor_place = sensor_place
        # self.sub_contexts = ['sensor0_Temperature', 'sensor1_Temperature', 'sensor0_Humidity', 'sensor1_Humidity', 'Aircon0Power', 'Aircon1Power']
        self.sub_contexts = ['sensor0_Temperature', 'sensor1_Temperature', 'sensor0_Humidity', 'sensor1_Humidity', 'AirconPower', 'AirconTemp']
        for context in self.sub_contexts:
            self.subscribe(f'{self.sensor_place}/context/{context}')

        self.robot_contexts = ['AgentStatus']
        for context in self.robot_contexts:
            self.subscribe(f'{self.place_name}/context/{context}')

        self.tem0 = -1
        self.tem1 = -1
        self.hum0 = -1
        self.hum1 = -1
        self.ac_power = None
        self.ac_temperature = None
        self.tem_state = -1
        self.hum_state = -1
        self.ac_state = -1
        self.rl_state = "OFF"
        

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        msg_dict = json.loads(dict_string)
        context_name = msg_dict.get('name')
        
        if context_name == 'sensor0_Temperature':
            self.tem0 = msg_dict.get('value') 
        elif context_name == 'sensor1_Temperature':
            self.tem1 = msg_dict.get('value') 
        elif context_name == 'sensor0_Humidity':
            self.hum0 = msg_dict.get('value') 
        elif context_name == 'sensor1_Humidity':
            self.hum1 = msg_dict.get('value') 

        elif context_name == 'AirconPower':
            self.ac_power = msg_dict.get('value') 

        elif context_name == 'AirconTemp':
            self.ac_temperature = msg_dict.get('value') 
        elif context_name == 'AgentStatus':
            self.rl_state = msg_dict.get('value')

        else:
            print('wrong')
        self.update()


    def temp_up(self):
        self.publish_func('TempUpAC')

    def temp_down(self):
        self.publish_func('TempDownAC')
    def temp_still(self):
        self.publish_func('TempStillAC')


    def  update(self):
        if (self.tem0 != -1) and (self.tem1 != -1):
            self.tem_state = (self.tem0 + self.tem1) / 2
        if (self.hum0 != -1) and (self.hum1 != -1):
            self.hum_state = (self.hum0 + self.hum1) / 2
        if self.ac_power == False:
            self.ac_state = 0
        else:
            if self.ac_temperature != None:
                ac_map = {28: 1, 27: 2, 26: 3, 25: 4}
                self.ac_state = ac_map[self.ac_temperature]

        

    def set_ac(self, set_point):
        print(f'set {set_point}')
        self.publish_func('SetTempAircon', arguments=[set_point], place=self.sensor_place)
    
    def power_on(self):
        print(f'Turn ON')
        self.publish_func('TurnOnAircon', arguments=[], place=self.sensor_place)

    def power_off(self):
        print(f'Turn OFF')
        self.publish_func('TurnOffAircon', arguments=[], place=self.sensor_place)

    
        

if __name__ == '__main__':
    client = WebFeedbackCollectorAgent()
    client.loop_forever()
    client.disconnect()