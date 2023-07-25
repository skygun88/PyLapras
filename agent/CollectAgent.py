import os
import cv2
import sys
import json
import time
import base64
import datetime
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") if "Linux" in platform.platform() else None
from agent.LaprasAgent import LaprasAgent
from utils.configure import *
from utils.angle import calculate_difference, calculate_rotation
from utils.publish import *

INITIALIZE, DOCK, DOCKED, READY, MOVE, ROTATE, WAIT, SAVE = 0, 1, 2, 3, 4, 5, 6, 7
DISCONNECT, FAIL = -1, -2

STATE_DICT = {
    INITIALIZE: "INITIALIZE", 
    DOCK: "DOCK",
    DOCKED: "DOCKED",
    READY: "READY", 
    MOVE: "MOVE", 
    ROTATE: "ROTATE",
    WAIT: "WAIT", 
    SAVE: "SAVE", 
    DISCONNECT: "DISCONNECT", 
    FAIL: "FAIL"
}

class CollectAgent(LaprasAgent):
    def __init__(self, 
        agent_name='CollectAgent', 
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
        self.undock_request = False
        self.move_time = [0]*20
        self.move_start = 0

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
        meta_columns = ['start_dt', 'user_x', 'user_y', 'user_r', 'activity', 'clothing_top', 'clothing_bottom', 'gender', 'age']
        meta_data = [self.now, user_loc[0], user_loc[1], user_loc[2]] + [self.user_info[x] for x in meta_columns[4:]]
        meta_df = pd.DataFrame(data=[meta_data], columns=meta_columns)
        meta_df.to_csv(self.meta_path, index=None)


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
        elif self.state == ROTATE:
            self.rotate_action()
        elif self.state == WAIT:
            self.wait_action()
        elif self.state == SAVE:
            self.save_action()
        elif self.state == DOCK:
            self.dock_action()
        else:
            print(f'error occur - {STATE_DICT[self.state]}')
            self.disconnect()
            sys.exit()

    def ready_action(self):
        if self.curr_idx < self.loc_len:
            if self.robot_state == "DOCKED":
                robot_undock(self)
            elif self.robot_state == "UNDOCKING":
                return 
            elif self.robot_state == 'READY':
                self.target_x, self.target_y = self.robot_locs[self.curr_idx]
                # self.target_r = calculate_difference((self.target_x, self.target_y), self.user_loc)
                dummy_x = self.target_x + 0.1*(self.robot_x-self.target_x)/abs(self.robot_x-self.target_x)
                dummy_y = self.target_y + 0.1*(self.robot_y-self.target_y)/abs(self.robot_y-self.target_y)
                # print(dummy_x, dummy_y)
                self.target_r = calculate_difference((dummy_x, dummy_y), (self.user_loc[0], self.user_loc[1]))            

                print(self.curr_idx, self.target_x, self.target_y, self.target_r)
                self.move_start = time.time()
                robot_move(self, self.target_x, self.target_y, self.target_r)
                self.transition(MOVE)
            else:
                print('wrong robot state in READY (1)')
                self.disconnect()
                sys.exit()
        else:
            if self.robot_state == "READY":

                time_columns = ['robot_pos', 'movement_time']
                # meta_data = [self.now, user_loc[0], user_loc[1], user_loc[2]] + [self.user_info[x] for x in meta_columns[4:]]
                time_data = list(zip(list(range(20)), self.move_time))
                time_df = pd.DataFrame(data=time_data, columns=time_columns)
                time_path = os.path.join(self.curr_data_path, 'movement_time.csv')

                time_df.to_csv(time_path, index=None)


                self.target_x, self.target_y, self.target_r = poi_to_location['docking_station']
                
                robot_move(self, self.target_x, self.target_y, self.target_r)
                self.transition(MOVE)
            else:
                print('wrong robot state in READY (2)')
                self.disconnect()
                sys.exit()

    def move_action(self):
        ''' Check robot successfully arrive the target position'''
        print(f'Moving - ({self.robot_x:.2f}, {self.robot_y:.2f}) -> ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_r:.2f})')
        if self.move_complete == True:
            if self.curr_idx < self.loc_len:
                self.move_complete = False
                print('Moving is done')
                self.move_time[self.curr_idx] = time.time() - self.move_start
                self.wait_start = time.time()
                self.scene_frames.clear()
                self.transition(WAIT)
                # user_angle = calculate_difference((self.robot_x, self.robot_y), self.user_loc)
                # rotation_angle = calculate_rotation(self.robot_r, user_angle)
                # robot_rotate(self, rotation_angle)
                # self.transition(ROTATE)
            else:
                if self.robot_state == "READY":
                    self.move_complete = False
                    robot_dock(self)
                    self.transition(DOCK)

    def rotate_action(self):
        if self.move_complete == True:
            self.move_complete = False
            print('Rotation is done')
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
        frames = self.scene_frames[:]
        if self.video_save:
            ''' Save video data as MP4'''
            fps = len(frames) / self.wait_time
            video_path = os.path.join(self.data_dir_path, f'{self.curr_idx}.mp4')
            height, width, _ = frames[0].shape
            size = (width,height)
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

            for frame in frames:
                out.write(frame)
            out.release()
        else:
            ''' Save video data as images '''
            video_path = os.path.join(self.data_dir_path, f'{self.curr_idx}')
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            
            for idx, frame in enumerate(frames):
                cv2.imwrite(os.path.join(video_path, f'{idx}.png'), frame)
            
            self.scene_frames.clear()
        self.curr_idx += 1
        self.transition(READY)
    
    def dock_action(self):
        if self.robot_state in ("READY", "DOCKING"):
            return
        elif self.robot_state == "DOCKED":
            print('Robot docked successfully')
            self.disconnect()
            sys.exit()
        else:
            print('wrong robot state in READY (2)')
            self.disconnect()
            sys.exit()


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
            if msg_dict['value'] in ('move', 'attending'):
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


    def check_connected(self):
        now = int(time.time()*1000)
        connected = (self.robot_x > 0) and (self.robot_y > 0) and ((now - self.last_alive) < 1000*30) 
        if self.state == INITIALIZE:
            self.transition(READY)
        return connected
    

    def check_video_quality(self):
        mean_latency = sum(self.temp_latencies)/len(self.temp_latencies) if len(self.temp_latencies) > 0 else -1
        frames = len(self.temp_frames)
        temp_fps = frames/5
        print(f'Collected {self.frame_idx} frames | FPS: {temp_fps:.3f}, Latency: {mean_latency:.3f} ms')
        self.temp_frames.clear()
        self.temp_latencies.clear()

    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CollectAgent')
    parser.add_argument('-v', '--video', action='store_true', default=False)
    parser.add_argument('-w', '--wait', type=int, default=5)
    parser.add_argument('-a', '--activity', type=str, default='sit', choices=['sit', 'office', 'stand'])
    parser.add_argument('-s', '--sex', type=str, default='man', choices=['man', 'woman'])
    parser.add_argument('-g', '--age', type=int, default=26)
    parser.add_argument('-t', '--top', type=str, default='LONG'.lower(), choices=['SHORT'.lower(), 'LONG'.lower()])
    parser.add_argument('-b', '--bottom', type=str, default='TROUSERS'.lower(), choices=['TROUSERS'.lower(), 'SHORTS'.lower()])

    args = parser.parse_args()

    video_save = args.video
    wait_time = args.wait
    top = f'{args.top}_SLEEVE_TOP'.upper()
    bottom = args.bottom.upper()

    user_loc = (52.0, 47.8, 90)
    # user_loc = (52.1, 46.8, 270)
    # user_loc = (50.3, 44.0, 90)
    
    user_info = {
        "activity": args.activity,
        "clothing_top": top,
        "clothing_bottom": bottom,
        "gender": args.sex,
        "age": args.age
    }

    client = CollectAgent(
        video_save=video_save, 
        wait_time=wait_time,
        user_loc=user_loc,
        user_info=user_info
    )
    client.loop_forever()
    client.disconnect()


