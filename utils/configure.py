import os

ROOT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0], 'PyLapras')
AGENT_PATH = os.path.join(ROOT_PATH, 'agent')
RESOURCE_PATH = os.path.join(ROOT_PATH, 'resources')
MAP_PATH = os.path.join(ROOT_PATH, 'map')
TESTER_PATH = os.path.join(ROOT_PATH, 'Tester')
WEIGHT_PATH = os.path.join(RESOURCE_PATH, 'weight')

poi_to_location = {'p0': (49.9553, 50.5662, 90), 
                   'p1': (47.7419, 50.6098, -90),
                   'p2': (47.4419, 54.4498, 0),
                   'p3': (31.6619, 52.7698, 0), 
                   'docking_station': (49.6886, 47.5623, -177), 
                   'lab': (24.6907, 53.3298, -87), 
                   'elevator': (42.9943, 53.9842, -90),
                   'seminar_room': (17.7158, 52.577, 182)
                   }

AC_MODE = {
            0: "OFF", 
            1: "Low",
            2: "High"
        }

CLO_NUM = 13
SHORT_SLEEVE_TOP = 0
LONG_SLEEVE_TOP = 1
SHORT_SLEEVE_OUTWEAR = 2
LONG_SLEEVE_OUTWEAR = 3
VEST = 4
SLING = 5
SHORTS = 6
TROUSERS = 7
SKIRTS = 8
SHORT_SLEEVE_DRESS = 9
LONG_SLEEVE_DRESS = 10
VEST_DRESS = 11
SLING_DRESS = 12

CLO_MAP = {
    SHORT_SLEEVE_TOP: 'SHORT_SLEEVE_TOP',
    LONG_SLEEVE_TOP: 'LONG_SLEEVE_TOP', 
    SHORT_SLEEVE_OUTWEAR: 'SHORT_SLEEVE_OUTWEAR',
    LONG_SLEEVE_OUTWEAR: 'LONG_SLEEVE_OUTWEAR',
    VEST: 'VEST',
    SLING: 'SLING',
    SHORTS: 'SHORTS',
    TROUSERS: 'TROUSERS',
    SKIRTS: 'SKIRTS',
    SHORT_SLEEVE_DRESS: 'SHORT_SLEEVE_DRESS',
    LONG_SLEEVE_DRESS: 'LONG_SLEEVE_DRESS',
    VEST_DRESS: 'VEST_DRESS',
    SLING_DRESS: 'SLING_DRESS'
}

''' Location '''
ROBOT_LOCS = [i for i in range(20)]

SIT, OFFICE, STAND = 0, 1, 2
# SHORT, LONG = 0, 1
ACTIVITIES = [SIT, OFFICE, STAND]
ACTIVITY_DICT = {'sit': SIT, 'office': OFFICE, 'stand': STAND}

''' Robot Control '''

STAY, UP, DOWN, RIGHT, LEFT, DONE = 0, 1, 2, 3, 4, 5 
ROBOT_ACTION_DICT = {
    STAY: 'STAY',
    UP: 'UP',
    DOWN: 'DOWN',
    RIGHT: 'RIGHT',
    LEFT: 'LEFT',
    DONE: 'DONE'
}
ROBOT_ACTIONS = [STAY, UP, DOWN, RIGHT, LEFT, DONE]


# PENALTY = -0.5
AGE_THRES = 5
GENDER_THRES = 0.5

# N_IN_ACT = 3
N_IN_ACT = 1
N_IN_CLO = 13
N_IN_AGE = 101
# N_IN_GEN = 2
N_IN_GEN = 1