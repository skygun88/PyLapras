import os
import sys
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

    def load(self):
        df = pd.read_csv(os.path.join(MAP_PATH, 'map.csv'), header=None, index_col=None)
        img = Image.open(os.path.join(MAP_PATH, 'current_map.png'))
        return df.to_numpy().transpose(), img

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

if __name__ == '__main__':
    map = N1Map()
    print(map.is_valid_location(49.29, 48.3))