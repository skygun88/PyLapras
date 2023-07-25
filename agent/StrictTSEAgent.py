import os
import cv2
import sys
import json
import time
import base64
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") if "Linux" in platform.platform() else None
from agent.LaprasAgent import LaprasAgent
from utils.configure import *
from utils.angle import calculate_difference, calculate_rotation, calculate_distance
from utils.publish import *
from utils.VisionModel import VisioinModel
from utils.A3C.meta_multi_A3C import Net
from utils.A3C.model_utils import *
from utils.comfort import *
# INITIALIZE, DOCK, DOCKED, READY, MOVE, ROTATE, WAIT, SAVE = 0, 1, 2, 3, 4, 5, 6, 7
INITIALIZE, DOCK, DOCKED, READY, ACCEPT, MOVE, ROTATE, RECOGNIZE, DETERMINE, SERVICE_DONE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

DISCONNECT, FAIL = -1, -2

STATE_DICT = {
    INITIALIZE: "INITIALIZE", 
    DOCK: "DOCK",
    DOCKED: "DOCKED",
    READY: "READY", 
    ACCEPT: 'ACCEPT',
    MOVE: "MOVE", 
    ROTATE: "ROTATE",
    RECOGNIZE: "RECOGNIZE", 
    DETERMINE: "DETERMINE", 
    SERVICE_DONE: "SERVICE_DONE",
    DISCONNECT: "DISCONNECT", 
    FAIL: "FAIL"
}
def clothing_str(clothings):
    detected_indexes = []
    for idx, detected in enumerate(clothings):
         if detected == 1:
             detected_indexes.append(idx)
    detected_labels = [CLO_MAP[idx] for idx in detected_indexes]
    return detected_labels

def determined_str(determined):
    act, clo, age, gen = determined
    clo_str = clothing_str(clo) if clo != None else None
    return f'{act}, {clo_str}, {age}, {gen}'
             

class TSEAgent(LaprasAgent):
    def __init__(self, 
        agent_name='TSEAgent', 
        place_name='Robot', 
        video_save=False, 
        wait_time=1,
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
        self.subscribe(f'{place_name}/functionality/UserMonitor')
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
        self.load_robot_actions()
        self.load_policy()

        # print(self.robot_locs)
        self.loc_len = len(self.robot_locs)
        self.curr_idx = 0
        self.wait_start = 0
        self.undock_request = False
        self.curr_rstate = None
        
        ''' Recognition '''
        
        self.vision_model = None
        self.input_config = (
            0,
            1,
            2,
            3,
        )
        
        self.rstate_queue = deque([], maxlen=3)
        self.progress = np.array([1, 1, 1, 1])

        self.determined = [None, None, None, None]
        self.curr_goal = None
        self.accepted = False
        self.started = False
        self.trajectory = []


    def load_robot_actions(self):
        self.robot_actions: pd.DataFrame = pd.read_csv('/SSD4TB/skygun/robot_code/PyLapras/map/robot_action.csv', header=0)

    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_DICT[self.state]}->{STATE_DICT[next_state]}')                  
        self.state = next_state

    def load_robot_locs(self):
        with open("/SSD4TB/skygun/robot_code/PyLapras/map/robot_loc.json", "r") as json_file:
            loc_dict = json.load(json_file)
            json_file.close()
        point_cnt = loc_dict['numOfPoint']
        return [(loc_dict['points'][f'{i}']['y'], loc_dict['points'][f'{i}']['x']) for i in range(point_cnt)]
        
    def load_policy(self):
        self.policies = []
        for i in range(3):
            self.policy_df: pd.DataFrame = pd.read_csv(f'/SSD4TB/skygun/robot_code/PyLapras/resources/trained_policy{i+1}.csv', sep=',')
            self.policy_pivots = [
                pd.pivot_table(
                    self.policy_df.loc[self.policy_df.target==i, :], 
                    index='robot_pos', 
                    columns='action', 
                    values='value', 
                    aggfunc='mean'
                ) 
                for i in range(4)
            ]
            self.policies.append([pivot.to_numpy() for pivot in self.policy_pivots])

    def find_nearest_position(self, robot_x, robot_y):
        distances = [(idx, calculate_distance((robot_x, robot_y), (x, y))) for idx, (x, y) in enumerate(self.robot_locs)]
        nearest = sorted(distances, key=lambda x: x[1])[0][0]
        return nearest

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

    def get_possible_rstate(self, rstate):
        curr_pos_information = self.robot_actions.loc[self.robot_actions.pos == rstate, :].iloc[0]
        return curr_pos_information[[ROBOT_ACTION_DICT[action] for action in ROBOT_ACTIONS]].to_numpy(dtype=int)


    def timer_callback(self):
        self.timer_cnt += 1
        curr_ts = time.time()
        duration = curr_ts - self.start_ts

        self.connected = self.check_connected()
        if self.vision_model == None:
            self.vision_model = VisioinModel()
            
        if not self.connected:
            print('Conection Failed')
            return 

        if int(duration) % 10 == 0:
            print(f'[{STATE_DICT[self.state]}|{self.robot_state}] RSTATE: {self.curr_rstate}, {self.robot_x:.2f}, {self.robot_y:.2f}, {self.robot_r:.2f}, {self.last_alive}')
            self.check_video_quality()

        if self.state == READY:
            self.ready_action()
        elif self.state == ACCEPT:
            self.accept_action()
        elif self.state == MOVE:
            self.move_action()
        elif self.state == ROTATE:
            self.rotate_action()
        elif self.state == RECOGNIZE:
            self.recognize_action()
        elif self.state == DETERMINE:
            self.determine_action()
        elif self.state == DOCK:
            self.dock_action()
        elif self.state == SERVICE_DONE:
            self.done_action()
        else:
            print(f'error occur - {STATE_DICT[self.state]}')
            self.disconnect()
            sys.exit()

    def ready_action(self):
        if self.robot_state == "DOCKED":
            robot_undock(self)
        elif self.robot_state == "UNDOCKING":
            return 
        elif self.robot_state == 'READY':
            if self.curr_rstate == None:
                self.curr_rstate = self.find_nearest_position(self.robot_x, self.robot_y)
                self.target_x, self.target_y = self.robot_locs[self.curr_rstate]
                self.target_r = 0
                print(self.curr_idx, self.target_x, self.target_y, self.target_r)
                robot_move(self, self.target_x, self.target_y, self.target_r)
                self.transition(MOVE)
            else:
                if self.accepted:
                    self.started = True
                    self.transition(ACCEPT)
        else:
            print('wrong robot state in READY (1)')
            self.disconnect()
            sys.exit()

    def accept_action(self):
        target_r = calculate_difference((self.robot_x, self.robot_y), (self.user_x, self.user_y))
        rotation_angle = calculate_rotation(self.robot_r, target_r)
        print(self.robot_r, target_r, rotation_angle)

        robot_rotate(self, int(rotation_angle))
        self.transition(ROTATE)

    def move_action(self):
        ''' Check robot successfully arrive the target position'''
        print(f'Moving - ({self.robot_x:.2f}, {self.robot_y:.2f}) -> ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_r:.2f})')
        if self.move_complete == True:
            self.move_complete = False
            if self.started: # Main Loop
                if np.sum(self.progress) > 0: # Not yet finished
                    print('Moving is done')
                    self.wait_start = time.time()
                    self.scene_frames.clear()
                    self.transition(RECOGNIZE)

                else:  # Finished
                    if self.robot_state == "READY":
                        self.started = False
                        self.progress = np.array([1, 1, 1, 1])
                        self.determined = [None, None, None, None]
                        self.curr_goal = None
                        self.accepted = False
                        self.transition(READY)
            else: # First fit in gird
                self.transition(READY)

    def rotate_action(self):
        if self.move_complete == True:
            self.move_complete = False
            print('Rotation is done')
            self.wait_start = time.time()
            self.scene_frames.clear()
            self.transition(RECOGNIZE)


    def recognize_action(self):
        ''' Wait and Collect video frames on the current position'''
        if (time.time() - self.wait_start) < self.wait_time:
            return
        frames = self.scene_frames[:]
        img = frames[-1]

        cv2.imwrite('/SSD4TB/skygun/robot_code/PyLapras/agent/test_img.png', img)
        result_act, result_clos, pose, result_age, result_gender, har_conf, clo_confs, age_conf, gender_conf, _, _, _ = self.vision_model.recognition(img)
        self.curr_preds = result_act, result_clos, result_age, result_gender
        self.curr_confs = har_conf, clo_confs, age_conf, gender_conf
        self.curr_pose = pose
        self.curr_img = img.copy()
        print(f'Recognized: {result_act}, {clothing_str(result_clos)}, {result_age}, {result_gender}')
        
        video_path = os.path.join(self.data_dir_path, f'{len(self.trajectory)}')
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
        
        for idx, frame in enumerate(frames):
            cv2.imwrite(os.path.join(video_path, f'{idx}.png'), frame)
        
        self.trajectory.append(self.curr_rstate)
        
        self.transition(DETERMINE)

    def determine_action(self):
        ''' Save the collected video frames and return back to READY'''
        # Trained 3 position
        if (self.user_x == 52.0) and (self.user_y == 47.8): 
            policy_idx = 0
        elif (self.user_x == 52.0) and (self.user_y == 46.8):
            policy_idx = 1
        elif (self.user_x == 50.3) and (self.user_y == 44.0):
            policy_idx = 2
        else:
            print('wrong_location')
    
        if self.curr_goal == None:
            for i in range(4):
                if self.progress[3-i] == 1:
                    self.curr_goal = 3-i
                    break
        policy = self.policies[policy_idx][self.curr_goal]
        robot_pos = self.curr_rstate
        a = np.argmax(policy[robot_pos, :])

        print(f'Progress : {self.progress}, Determined States: {determined_str(self.determined)}')
        print(f'Deterimend Action: {ROBOT_ACTION_DICT[a]} | Goal: {self.curr_goal}')
        
        a = DONE if a == STAY else a

        next_state = RECOGNIZE
        if a in [UP, DOWN, RIGHT, LEFT]:
            moved, next_rstate = self.get_next_rstate(self.curr_rstate, a)
            if moved:
                next_state = MOVE
                self.target_x, self.target_y = self.robot_locs[next_rstate]
                dummy_x = self.target_x + 0.1*(self.robot_x-self.target_x)/abs(self.robot_x-self.target_x)
                dummy_y = self.target_y + 0.1*(self.robot_y-self.target_y)/abs(self.robot_y-self.target_y)
                self.target_r = calculate_difference((dummy_x, dummy_y), (self.user_x, self.user_y))            

                print(next_rstate, self.target_x, self.target_y, self.target_r)
                robot_move(self, self.target_x, self.target_y, self.target_r)
                self.curr_rstate = next_rstate

        elif a == DONE:
            is_detected = self.is_detect_any(self.curr_goal, self.curr_preds)
            print(f'Determined - idx: {self.curr_goal}, value: {self.curr_preds[self.curr_goal]}')
            if is_detected: 
                self.determined[self.curr_goal] = self.curr_preds[self.curr_goal]
                self.progress[self.curr_goal] = 0
                self.curr_goal = None

                if np.sum(self.progress) == 0:
                    next_state = SERVICE_DONE
            
        elif a != STAY:
            print('Invalid Action selection')
            sys.exit()

        if next_state == RECOGNIZE:
            self.wait_start = time.time()

        self.scene_frames.clear()
        self.transition(next_state)
    
    def done_action(self):
        print(f'Recognized human states: {self.determined}')

        ''' Thermal Control based on detected human states '''
        control_actions = self.determine_thermal_control(self.determined)
        set_point = control_actions[0]
        fan_control = 'FanOn' if control_actions[1] else'FanOff'
        self.publish_func('SetTempAircon', arguments=[set_point], place='N1Lounge8F')
        self.publish_func(fan_control, arguments=[], place='N1Lounge8F')

        with open(os.path.join(self.data_dir_path, 'log.txt'), 'w') as f:
            save_str = '\n'.join([str(x) for x in self.trajectory])
            f.write(save_str)
            f.close()
        self.trajectory.clear()
        if self.robot_state == "READY":
            self.target_x, self.target_y, self.target_r = poi_to_location['docking_station']
            
            robot_move(self, self.target_x, self.target_y, self.target_r)
            self.transition(MOVE)

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
            return

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
        elif context_name == 'robotComplete':
            # print(msg_dict['name'], msg_dict['value'])
            if msg_dict['value'] in ('move', 'attending'):
                self.move_complete = True 
        elif context_name == 'UserMonitor':
            arguments = msg_dict['arguments']
            self.user_x = arguments[0]
            self.user_y = arguments[1]
            # self.transition(next_state=ACCEPT)
            self.accepted = True
        
        elif context_name == 'RobotDetectedImage':
            img_str = msg_dict['value']
            imgdata = base64.b64decode(img_str)
            cv_img = cv2.imdecode(np.array(np.frombuffer(imgdata, dtype=np.uint8)) , cv2.IMREAD_COLOR)
            self.temp_frames.append(cv_img)
            if self.state == RECOGNIZE:
                self.scene_frames.append(cv_img)
            self.frame_idx += 1
            self.temp_latencies.append(curr_ts-send_ts)

        else:  
            print('wrong')


    def check_connected(self):
        now = int(time.time()*1000)
        connected = (self.robot_x > 0) and (self.robot_y > 0) and ((now - self.last_alive) < 1000*30) 
        if connected and (self.state == INITIALIZE):
            self.transition(READY)
        return connected
    

    def check_video_quality(self):
        mean_latency = sum(self.temp_latencies)/len(self.temp_latencies) if len(self.temp_latencies) > 0 else -1
        frames = len(self.temp_frames)
        temp_fps = frames/5
        # print(f'Collected {self.frame_idx} frames | FPS: {temp_fps:.3f}, Latency: {mean_latency:.3f} ms')
        self.temp_frames.clear()
        self.temp_latencies.clear()

    def get_next_rstate(self, rstate: int, action: int)->Tuple[bool, int]:
        moved = False
        next_rstate = rstate
        if action != DONE:    
            curr_pos_information = self.robot_actions.loc[self.robot_actions.pos == rstate, :].iloc[0]
            next_rstate = int(curr_pos_information[ROBOT_ACTION_DICT[action]])
            moved = not (rstate == next_rstate)

        return moved, next_rstate

    def is_detect_any(self, goal, preds):
        detected = True
        if goal in [0, 2, 3]:
            detected = False if preds[goal] == -1 else True
        elif goal == 1:
            detected = False if np.sum(preds[goal]) == 0 else True
        else:
            print('InValid goal value is is_detect_any --------------')
            sys.exit()

        return detected
    
    def determine_thermal_control(self, human_states):
        candidate_tems = [24, 25, 26, 27]
        candidate_fans = [False, True]
        humidity = 60
        comforts = []
        options = []
        har, clothing, age, gender = human_states
        clo_ensemble = clothing_analysis(clothing) 
        for tem in candidate_tems:
            for fan in candidate_fans:
                TS = estimate_thermal_comfort(tem, humidity, fan, har, clo_ensemble, age, gender)
                comforts.append(TS)
                options.append((tem, fan, TS))
        abs_comforts = [abs(x) for x in comforts]
        optimal_value = min(abs_comforts)
        optimal_idx = abs_comforts.index(optimal_value)
        optimal_option = options[optimal_idx] 
        return optimal_option
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSEAgent')
    parser.add_argument('-v', '--video', action='store_true', default=False)
    parser.add_argument('-w', '--wait', type=int, default=1)
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
    # user_loc = (52.1, 46.8, 90)
    # user_loc = (50.3, 44.0, 90)
    
    user_info = {
        "activity": args.activity,
        "clothing_top": top,
        "clothing_bottom": bottom,
        "gender": args.sex,
        "age": args.age
    }

    client = TSEAgent(
        video_save=video_save, 
        wait_time=wait_time,
        user_loc=user_loc,
        user_info=user_info
    )
    client.loop_forever()
    client.disconnect()


