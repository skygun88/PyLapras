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
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") if "Linux" in platform.platform() else None
from agent.LaprasAgent import LaprasAgent
from utils.configure import *

class FeedbackCollectorAgent(LaprasAgent):
    def __init__(self, gui, agent_name='FeedbackCollectorAgent', place_name='Robot', sensor_place='N1Lounge8F'):
        super().__init__(agent_name, place_name)
        self.sensor_place = sensor_place
        self.sub_contexts = ['sensor0_Temperature', 'sensor1_Temperature', 'sensor0_Humidity', 'sensor1_Humidity', 'Aircon0Power', 'Aircon1Power']
        for context in self.sub_contexts:
            self.subscribe(f'{self.sensor_place}/context/{context}')
        
        
        ''' For Preventing circuit import '''
        from Interface.FeedbackCollector import QFeedbackCollector
        self.gui: QFeedbackCollector = gui

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        msg_dict = json.loads(dict_string)
        context_name = msg_dict.get('name')
        
        if context_name == 'sensor0_Temperature':
            self.gui.update_temperature(tem1=msg_dict.get('value'))
        elif context_name == 'sensor1_Temperature':
            self.gui.update_temperature(tem2=msg_dict.get('value'))
        elif context_name == 'sensor0_Humidity':
            self.gui.update_humidity(hum1=msg_dict.get('value'))
        elif context_name == 'sensor1_Humidity':
            self.gui.update_humidity(hum2=msg_dict.get('value'))

        elif context_name == 'Aircon0Power':
            self.gui.update_aircon(ac1=msg_dict.get('value'))

        elif context_name == 'Aircon1Power':
            self.gui.update_aircon(ac2=msg_dict.get('value')) 

        else:
            print('wrong')


    def power_up(self):
        self.publish_func('PowerUpAC')

    def power_down(self):
        self.publish_func('PowerDownAC')


    def set_ac(self, mode):
        print(f'set {mode}')
        if mode == 0:
            self.publish_func('StopAircon0', place=self.sensor_place)
            self.publish_func('StopAircon1', place=self.sensor_place)
        elif mode == 1:
            self.publish_func('StartAircon0', place=self.sensor_place)
            self.publish_func('StopAircon1', place=self.sensor_place)
        elif mode == 2:
            self.publish_func('StartAircon0', place=self.sensor_place)
            self.publish_func('StartAircon1', place=self.sensor_place)
        else:
            print('error occur in SET AC')
            sys.exit()

    
        

if __name__ == '__main__':
    client = FeedbackCollectorAgent()
    client.loop_forever()
    client.disconnect()