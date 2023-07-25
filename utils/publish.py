import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from agent.LaprasAgent import LaprasAgent


def robot_move(agent: LaprasAgent, x, y, z=0.0):
    agent.publish_func('robotMove', arguments=[float(x), float(y), float(z)])

def robot_rotate(agent: LaprasAgent, angle):
    agent.publish_func('robotAttend', arguments=[int(angle)])

def robot_dock(agent: LaprasAgent):
    agent.publish_func('docking', arguments=[])

def robot_undock(agent: LaprasAgent):
    agent.publish_func('undocking', arguments=[])