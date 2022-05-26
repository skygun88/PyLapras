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
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") if "Linux" in platform.platform() else None
from agent.LaprasAgent import LaprasAgent
from utils.configure import *

class RobotTestAgent(LaprasAgent):
    def __init__(self, gui, agent_name='RobotTestAgent', place_name='Robot'):
        super().__init__(agent_name, place_name)
        self.subscribe(f'{place_name}/context/RobotDetectedImage')
        self.subscribe(f'{place_name}/context/RobotControlAgentOperatingStatus', 2)
        self.subscribe(f'{place_name}/context/RobotX')
        self.subscribe(f'{place_name}/context/RobotY')
        self.subscribe(f'{place_name}/context/RobotStatus')
        self.subscribe(f'{place_name}/context/robotComplete')
        
        ''' For Preventing circuit import '''
        # sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
        from Tester.RobotTester import QRobotTester
        self.gui: QRobotTester = gui
        self.video_dir = os.path.join(TESTER_PATH, 'video')
        self.out_fname = 'out.mp4'
        self.timer_cnt = 0

        self.create_timer(self.timer_callback, timer_period=1)
        self.images = []
        self.ts = []
        self.record_flag = False
        
        ''' Robot Status '''
        self.robot_x, self.robot_y = -1, -1
        self.path = []
        self.robot_state = 'UNITITIALIZED'
        self.last_alive = -1
        self.initialized = False
        self.connected = False
        
    def timer_callback(self):      
        self.timer_cnt += 1
        self.connected = self.check_connected()

        if self.connected:
            if self.initialized == False:
                self.initialized = True
                self.gui.initRobot()
        
        else:
            if self.initialized == True:
                self.robot_x, self.robot_y = -1, -1
                self.last_alive = -1
                self.connected = False
        
        if self.initialized:
            self.gui.update_robot(self.connected, self.robot_x, self.robot_y, self.robot_state)

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        msg_dict = json.loads(dict_string)
        context_name = msg_dict.get('name')
        
        if context_name == 'RobotControlAgentOperatingStatus':
            self.last_alive = msg_dict.get('timestamp')
        elif context_name == 'RobotX':
            self.robot_x = msg_dict.get('value')
        elif context_name == 'RobotY':
            self.robot_y = msg_dict.get('value')
        elif context_name == 'RobotStatus':
            self.robot_state = msg_dict.get('value')
            print(self.robot_state)
        elif context_name == 'robotComplete':
            print(msg_dict['name'], msg_dict['value'])
        
        elif context_name == 'RobotDetectedImage':
            if self.initialized and self.connect:
                img_str = msg_dict['value']
                imgdata = base64.b64decode(img_str)
                cv_img = cv2.imdecode(np.array(np.frombuffer(imgdata, dtype=np.uint8)) , cv2.IMREAD_COLOR)
                if self.gui.cameraLabel.isVisible():
                    self.gui.c.cameraReceive.emit(cv_img)
                
                if self.record_flag == True:
                    self.images.append(cv_img)
                    self.ts.append(time.time())
                    
        else:
            print('wrong')

    def is_connected(self):
        return self.connected
            
    def check_connected(self):
        now = int(time.time()*1000)
        connected = self.robot_x > 0 and self.robot_y > 0 and now - self.last_alive < 1000*15 
        return connected
    
    def start_record(self):
        self.images.clear()
        self.ts.clear()
        self.record_flag = True

    def end_record(self):
        if self.record_flag == False:
            return
        self.record_flag = False
        images = self.images[:]
        ts = self.ts[:]
        runtime = ts[-1]-ts[0] if len(images) > 0 else 0

        if runtime > 0:
            height, width, _ = images[0].shape
            size = width, height
            fps = len(images) / runtime            
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = cv2.VideoWriter(os.path.join(self.video_dir, f'{now}.mp4'), cv2.VideoWriter_fourcc(*'FMP4'), fps, size)
            for frame in images:
                out.write(frame)
            out.release()

        self.ts.clear()
        self.images.clear()
    
    def dock(self):
        self.publish_func('docking', arguments=[])

    def undock(self):
        self.publish_func('undocking', arguments=[])

    def move(self, x, y):
        self.publish_func('robotMove', arguments=[x, y, 0])

    def rotate(self, angle):
        self.publish_func('robotAttend', arguments=[angle])
    
    
        

if __name__ == '__main__':
    client = RobotTestAgent()
    client.loop_forever()
    client.disconnect()