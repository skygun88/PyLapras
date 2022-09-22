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

class MonnitTestAgent(LaprasAgent):
    def __init__(self, gui, agent_name='RobotTestAgent', place_name='Robot'):
        super().__init__(agent_name, place_name)
       
        ''' For Preventing circuit import '''
        # sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
        from Tester.RobotTester import QRobotTester
        self.gui: QRobotTester = gui
        actvities, _, _ = self.gui.sensor_map.get_activities()
        motions, _, _ = self.gui.sensor_map.get_motions()

        for activity in actvities:
            self.subscribe(f'N1Lounge8F/context/{activity}')
        for motion in motions:
            self.subscribe(f'N1Lounge8F/context/{motion}')
        
        self.create_timer(self.timer_callback, timer_period=1)

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        msg_dict = json.loads(dict_string)
        print(msg_dict, datetime.datetime.fromtimestamp(msg_dict.get('timestamp')/1000))
        context_name = msg_dict.get('name')
        prev_state = self.gui.sensor_map.get_sensor(context_name)
        state = msg_dict.get('timestamp')/1000
        value = True if msg_dict.get('value') == "True" else False

        if value == True:
            self.gui.sensor_map.update_sensor(context_name, state)
            if prev_state != state:
                self.gui.draw_points()

    def timer_callback(self):
        latest = self.gui.sensor_map.get_latest(5)
        # print(latest)
        filtered = list(filter(lambda x: (x[1] > 0) and ((time.time() - x[1]) < 30), latest))

        if len(filtered) > 0:
            locs = [self.gui.sensor_map.get_loc(sensor[0]) for sensor in filtered]
            locs = np.array(locs)
            mean_loc = np.mean(locs, axis=0)
            loc = locs[0, :]
            print(loc)
            self.gui.update_user(loc[0], loc[1])




       


    
    
        

if __name__ == '__main__':
    client = MonnitTestAgent()
    client.loop_forever()
    client.disconnect()