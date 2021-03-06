import os

ROOT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0], 'PyLapras')
AGENT_PATH = os.path.join(ROOT_PATH, 'agent')
RESOURCE_PATH = os.path.join(ROOT_PATH, 'resources')
MAP_PATH = os.path.join(ROOT_PATH, 'map')
TESTER_PATH = os.path.join(ROOT_PATH, 'Tester')

poi_to_location = {'p0': (49.9553, 50.5662), 
                   'p1': (47.7419, 50.6098),
                   'p2': (47.4419, 54.4498),
                   'p3': (31.6619, 52.7698), 
                   'docking_station': (49.6886, 47.5623), 
                   'lab': (24.6907, 53.3298), 
                   'elevator': (42.9943, 53.9842),
                   'seminar_room': (17.7158, 52.577)}


