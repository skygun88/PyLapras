import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from utils.configure import *

'''
(255, 255, 255, 255) - White  : Allowed   ->  0
(150, 150, 150, 255) - Gray   : Forbidden ->  1
(0,   0,   0,   255) - Black  : Wall      ->  2
Image: 1000x1000 -> Pose axis: 100x100
'''

class N1Map:
    def __init__(self):
        self.map, self.map_img = self.load()
        self.robot_locs = self.load_robot(n_points=4)
        # print(self.robot_locs)

    def load(self):
        df = pd.read_csv(os.path.join(MAP_PATH, 'map.csv'), header=None, index_col=None)
        img = Image.open(os.path.join(MAP_PATH, 'current_map.png'))
        return df.to_numpy().transpose(), img

    def load_robot(self, n_points=5):
        with open(os.path.join(MAP_PATH, 'loc.json'), 'r') as f:
            robot_json = json.load(f)
            f.close()
        robot_start = np.array(robot_json['start'])
        robot_end = np.array(robot_json['end'])

        unit_vector = (robot_end-robot_start)/(n_points-1)
        robot_locs = np.stack([robot_start + unit_vector*i for i in range(n_points)])
        return robot_locs



    def is_reachable(self, curr_x, curr_y, target_x, target_y):
        pass

    def is_valid_location(self, x, y):
        x_idx, y_idx = self.pose_to_index(x, y)
        allowed = np.vectorize(lambda x: x == 0)
        # print(self.map[x_idx-1:x_idx+2, y_idx-1:y_idx+2])
        return allowed(self.map[x_idx-1:x_idx+2, y_idx-1:y_idx+2]).sum() == 9

    def is_allowed(self, x, y):
        return self.map[x, y] == 0

    def is_forbidden(self, x, y):
        return self.map[x, y] == 1

    def is_wall(self, x, y):
        return self.map[x, y] == 2

    def pose_to_index(self, x, y):
        return min(int(x*10), 999), min(int(y*10), 999)

class LoungeSensorMap:
    def __init__(self) -> None:
        # with open(os.path.join(MAP_PATH, 'lounge_monnit.json'), 'r') as f:
        #     sensor_json = json.load(f)
        #     f.close()
        with open(os.path.join(MAP_PATH, 'lounge_expected.json'), 'r') as f:
            sensor_json = json.load(f)
            f.close()
        self.activity_locs = sensor_json['Activity']
        self.motion_locs = sensor_json['Motion']

        # self.activity_names = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']
        self.activity_names = [
            'S01', 
            'S02', 
            'S03', 
            'S04', 
            'S05', 
            'S06', 
            'S07', 
            'S08', 
            'S09', 
            'S10'
            ]
        self.motion_names = [
            'M01', 
            'M02', 
            'M03', 
            'M04', 
            'M07', 
            'M08'
            ]

        self.activity_last = dict(zip(self.activity_names, [0]*len(self.activity_names)))
        self.motion_last = dict(zip(self.motion_names, [0]*len(self.motion_names)))

    def get_activities(self):
        locs = [self.activity_locs[sensor] for sensor in self.activity_names]
        return self.activity_names, locs, self.activity_last

    def get_motions(self):
        locs = [self.motion_locs[sensor] for sensor in self.motion_names]
        return self.motion_names, locs, self.motion_last

    def update_sensor(self, sensor_name, state):
        if sensor_name in self.activity_names:
            self.activity_last[sensor_name] = state
        elif sensor_name in self.motion_names:
            self.motion_last[sensor_name] = state

    def get_sensor(self, sensor_name):
        state = -1
        if sensor_name in self.activity_names:
            state = self.activity_last[sensor_name]
        elif sensor_name in self.motion_names:
            state = self.motion_last[sensor_name]
        return state

    def get_latest(self, number=1):
        return_number = 1 if number == 0 else number

        coupled_names = self.activity_names + self.motion_names
        coupled_last = [self.activity_last[x] for x in self.activity_names] + [self.motion_last[x] for x in self.motion_names]
        coupled = list(map(lambda x, y: [x, y], coupled_names, coupled_last))
        sorted_coupled = sorted(coupled, key=lambda x: x[1], reverse=True)
        return sorted_coupled[:return_number]

    def get_loc(self, sensor_name):
        sensor_name = sensor_name.upper()
        if sensor_name in self.activity_names:
            loc = self.activity_locs[sensor_name]
        elif sensor_name in self.motion_names:
            loc = self.motion_locs[sensor_name]
        return loc
    



if __name__ == '__main__':
    map = N1Map()
    print(map.is_valid_location(49.29, 48.3))