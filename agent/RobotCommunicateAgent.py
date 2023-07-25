import os
import cv2
import sys
import json
import time
import base64
import platform
import datetime
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") if "Linux" in platform.platform() else None
from agent.LaprasAgent import LaprasAgent
from utils.configure import *
from utils.angle import calculate_difference, calculate_rotation

INITIALIZE, READY, MOVE, WAIT, SAVE = 0, 1, 2, 3, 4
DISCONNECT, FAIL = -1, -2

STATE_DICT = {
    INITIALIZE: "INITIALIZE", 
    READY: "READY", 
    MOVE: "MOVE", 
    WAIT: "WAIT", 
    SAVE: "SAVE", 
    DISCONNECT: "DISCONNECT", 
    FAIL: "FAIL"
}

class RobotCommunicateAgent(LaprasAgent):
    def __init__(self, 
        agent_name='RobotCommunicateAgent', 
        place_name='Robot', 
        video_save=False, 
        wait_time=10,
        user_loc=None,
        user_info=None,
        ):
        super().__init__(agent_name, place_name)

        ''' Lapras Setup '''
        self.subscribe(f'{place_name}/context/RobotDetectedImage', 0)
        self.subscribe(f'{place_name}/context/RobotAlive', 1)
        self.subscribe(f'{place_name}/context/RobotX', 0)
        self.subscribe(f'{place_name}/context/RobotY', 0)
        self.subscribe(f'{place_name}/context/RobotOrientation', 0)
        self.subscribe(f'{place_name}/context/RobotStatus')
        self.subscribe(f'{place_name}/context/robotComplete')
        self.create_timer(self.timer_callback, timer_period=1)

        ''' initial setup '''
        self.user_info = user_info
        self.user_loc = user_loc
        self.wait_time = wait_time
        self.video_save = video_save

        ''' Path setup '''
        self.result_dir = os.path.join(TESTER_PATH, 'video')
        self.now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")     
        self.initialize_result()

        ''' Video Collect '''
        self.frame_idx= 0
        self.timer_cnt = 0
        self.start_ts = time.time()
        self.scene_frames = []
        self.temp_frames = []
        self.temp_latencies = []

        ''' Robot Status '''
        self.robot_x, self.robot_y = -1, -1
        self.robot_r = -999
        self.robot_state = 'UNITITIALIZED'
        self.last_alive = -1
        self.connected = False
        self.move_complete = False

        ''' Movement '''
        self.state = INITIALIZE
        self.target_x, self.target_y, self.target_r = -1, -1, -999
        self.robot_locs = self.load_robot_locs()
        self.loc_len = len(self.robot_locs)
        self.curr_idx = 0
        self.wait_start = 0
        

    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_DICT[self.state]}->{STATE_DICT[next_state]}')                  
        self.state = next_state

    def load_robot_locs(self):
        with open("/SSD4TB/skygun/robot_code/PyLapras/map/robot_loc.json", "r") as json_file:
            loc_dict = json.load(json_file)
            json_file.close()
        point_cnt = loc_dict['numOfPoint']
        return [(loc_dict['points'][f'{i}']['y'], loc_dict['points'][f'{i}']['x']) for i in range(point_cnt)]

    def initialize_result(self):
        self.curr_data_path = os.path.join(self.result_dir, f'{self.now}')
        self.data_dir_path = os.path.join(self.curr_data_path, 'data')
        self.meta_path = os.path.join(self.curr_data_path, 'metadata.csv')
        if not os.path.isdir(self.curr_data_path):
            os.makedirs(self.curr_data_path) 
        if not os.path.isdir(self.data_dir_path):
            os.makedirs(self.data_dir_path)
        meta_columns = ['start_dt', 'user_x', 'user_y', 'activity', 'clothing', 'age', 'gender']
        meta_data = [self.now, user_loc[0], user_loc[1]] + [self.user_info[x] for x in meta_columns[3:]]
        meta_df = pd.DataFrame(data=[meta_data], columns=meta_columns)
        meta_df.to_csv(self.meta_path, index=None)

    def move(self, x, y, z=0.0):
        self.publish_func('robotMove', arguments=[float(x), float(y), float(z)])

    def rotate(self, angle):
        self.publish_func('robotAttend', arguments=[float(angle)])

    def dock(self):
        self.publish_func('docking', arguments=[])

    def undock(self):
        self.publish_func('undocking', arguments=[])

    def timer_callback(self):
        self.timer_cnt += 1
        curr_ts = time.time()
        duration = curr_ts - self.start_ts

        self.connected = self.check_connected()
            
        if not self.connected:
            print('Conection Failed')
            return 

        if int(duration) % 5 == 0:
            print(f'[{STATE_DICT[self.state]}|{self.robot_state}] {self.robot_x:.2f}, {self.robot_y:.2f}, {self.robot_r:.2f}, {self.last_alive}')
            self.check_video_quality()

        if self.state == READY:
            self.ready_action()
        elif self.state == MOVE:
            self.move_action()
        elif self.state == WAIT:
            self.wait_action()
        elif self.state == SAVE:
            self.save_action()
        else:
            print(f'error occur - {STATE_DICT[self.state]}')
            self.disconnect()
            sys.exit()

    def ready_action(self):
        if self.curr_idx < self.loc_len:
            self.target_x, self.target_y = self.robot_locs[self.curr_idx]
            self.target_r = calculate_difference((self.target_x, self.target_y), self.user_loc)            

            print(self.curr_idx, self.target_x, self.target_y, self.target_r)
            self.move(self.target_x, self.target_y, self.target_r)
            self.transition(MOVE)
        else:
            self.disconnect()
            sys.exit()

    def move_action(self):
        ''' Check robot successfully arrive the target position'''
        print(f'Moving - ({self.robot_x:.2f}, {self.robot_y:.2f}) -> ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_r:.2f})')
        if self.move_complete == True:
            self.move_complete = False
            print('Moving is done')
            self.wait_start = time.time()
            self.scene_frames.clear()
            self.transition(WAIT)


    def wait_action(self):
        ''' Wait and Collect video frames on the current position'''
        if (time.time() - self.wait_start) < self.wait_time:
            return
        self.transition(SAVE)

    def save_action(self):
        ''' Save the collected video frames and return back to READY'''
        if self.video_save:
            ''' Save video data as MP4'''
            fps = self.frame_idx / self.wait_time
            video_path = os.path.join(self.data_dir_path, f'{self.curr_idx}.mp4')
            height, width, _ = self.scene_frames[0].shape
            size = (width,height)
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

            for frame in self.scene_frames:
                out.write(frame)
            out.release()
        else:
            ''' Save video data as images '''
            video_path = os.path.join(self.data_dir_path, f'{self.curr_idx}')
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            
            for idx, frame in enumerate(self.scene_frames):
                cv2.imwrite(os.path.join(video_path, f'{idx}.png'), frame)
             
        self.curr_idx += 1
        self.transition(READY)


    def on_message(self, client, userdata, msg):
        try:
            dict_string = str(msg.payload.decode("utf-8"))
            msg_dict = json.loads(dict_string)
        except Exception as e:
            print(f'Error occur when unpack mqtt msg - {e}')
            print(f'Tried msg : {msg}')

        context_name = msg_dict.get('name')
        curr_ts = self.curr_timestamp()
        send_ts = msg_dict['timestamp'] 
        # print(msg_dict)
        if context_name == 'RobotAlive':
            self.last_alive = msg_dict.get('timestamp')
        elif context_name == 'RobotX':
            self.robot_x = msg_dict.get('value')
        elif context_name == 'RobotY':
            self.robot_y = msg_dict.get('value')
        elif context_name == 'RobotOrientation':
            self.robot_r = msg_dict.get('value')
        elif context_name == 'RobotStatus':
            new_robot_state = msg_dict.get('value')
            self.robot_state = new_robot_state 
            # print(self.robot_state)
        elif context_name == 'robotComplete':
            # print(msg_dict['name'], msg_dict['value'])
            if msg_dict['value'] == 'move':
                self.move_complete = True
        
        elif context_name == 'RobotDetectedImage':
            img_str = msg_dict['value']
            imgdata = base64.b64decode(img_str)
            cv_img = cv2.imdecode(np.array(np.frombuffer(imgdata, dtype=np.uint8)) , cv2.IMREAD_COLOR)
            self.temp_frames.append(cv_img)
            if self.state == WAIT:
                self.scene_frames.append(cv_img)
            self.frame_idx += 1
            self.temp_latencies.append(curr_ts-send_ts)

        else:  
            print('wrong')



    
        

if __name__ == '__main__':
    client = RobotCommunicateAgent(
    )
    client.loop_forever()
    client.disconnect()


