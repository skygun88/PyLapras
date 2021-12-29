import os 
import sys
import json
import time
import datetime
import LaprasAgent

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

''' 0: Initializing, 1: Ready, 12: Patrolling, 13: Observing, 14: Inferring, 15: Actuating, 16: Returning '''
INITIALIZING, READY, PATROLLING, OBSERVING, INFERRING, ACTUATING, RETURNING = 0, 1, 12, 13, 14, 15, 16
LOUNGE, WAITING = 'p0', 'elevator'
STATE_MAP = {0: 'INITIALIZING', 1: 'READY', 12: 'PATROLLING', 13: 'OBSERVING', 14: 'INFERRING', 15: 'ACTUATING', 16: 'RETURNING'}

'''
    PatrolAgent
    *  service agent for night patrol the N1 Building 8F
    * Input: 
        agent_name - Name of the lapras agent (Default: PatrolAgent)
        place_name - The place of lapras agent (Default: N1Lounge8F)
        timeout_thres - The threshold of time difference between current time and last recevied alive topic (Default: 10 (sec))
        schedules - The list of scheduled time to start patrol ex. [{start: (22, 0), end: (23, 0)}]
'''
class AmbientAgents(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='AmbientAgents', place_name='N1Lounge8F'):
        super().__init__(agent_name=agent_name, place_name=place_name)
        ''' Agent States '''
       
        self.turnOffAllDevices()    
        # self.create_timer(self.timer_callback, timer_period=1)

    def turnOffAllDevices(self):
        name_list = ["TurnOffAllLights", "TurnOffFan", "StopAircon0", "StopAircon1", "StopAircon0", "StopAircon1"]
        # name_list = ["TurnOnAllLights", "TurnOnFan", "TurnOnFanRotation", "StartAircon0", "StartAircon1"]
        for name in name_list:
            self.publish_func(name)
    
    def timer_callback(self):
        # self.publish_context('Brightness', 30.8753)
        self.turnOffAllDevices()
    

if __name__ == '__main__':
    client = AmbientAgents(agent_name='AmbientAgents', place_name='N1Lounge8F')
    client.loop_forever()
    client.disconnect()
