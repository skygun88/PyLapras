import os 
import sys
import json
import time
import datetime
import LaprasAgent

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

''' 0: Initializing, 1: Ready, 2: Moving, 3: Observing, 4: Rotating, 5: Capturing '''
INITIALIZING, READY, MOVING, OBSERVING, ROTATING, CAPTURING = 0, 1, 2, 3, 4, 5
LOUNGE, WAITING = 'p0', 'elevator'
'''
    PatrolAgent
    *  service agent for night patrol the N1 Building 8F
    * Input: 
        agent_name - Name of the lapras agent (Default: PatrolAgent)
        place_name - The place of lapras agent (Default: N1Lounge8F)
        timeout_thres - The threshold of time difference between current time and last recevied alive topic (Default: 10 (sec))
        schedules - The list of scheduled time to start patrol ex. [{start: (22, 0), end: (23, 0)}]
'''
class RobotControlAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='RobotControlAgent', place_name='N1Lounge8F', timeout_thres=20):
        super().__init__(agent_name=agent_name, place_name=place_name)
        ''' Agent States '''
        self.status = INITIALIZING
        self.timeout_thres = timeout_thres
        self.communicator_ready, self.video_ready, self.pose_ready = -False, False, False # Image collector, Pose collector status 
        # self.communicator = "ROSCOmmunicator"
        self.image_queue = []
        self.pose = -1, -1

        self.create_timer(self.timer_callback, timer_period=1)
        self.subscribe('N1Lounge8F/functionality/robotMove', 2) # RobotControlAgent Alive
        self.subscribe('N1Lounge8F/functionality/observe', 2) # InferenceManaer Alive
    
    def timer_callback(self):
        print(f'[{self.agent_name}/{self.place_name}] Status: {self.status}')
        
        if self.status == INITIALIZING:
            ''' State transition: INITIALIZING -> READY '''
            if self.communicator_ready and self.video_ready and self.pose_ready:
                self.status = READY
            else:
                print('Waiting for ')
                return
        
        elif self.status == READY:
            pass
        
        elif self.status == MOVING:
            print('Waiting for robot to arrive Lounge')
            pass

        elif self.status == OBSERVING:
            print('Waiting for robot to capture images')
            pass

        elif self.status == ROTATING:
            print('Waiting for inference manager to detect people')
            pass

        elif self.status == CAPTURING: 
            pass

        else:
            print('Fobbiden line')
            raise('Why code is in here')

        if self.status != INITIALIZING:
            self.publish_context('robotStatus', True, 2) # Send alive message

        # print(f'[{self.agent_name}/{self.place_name}] State transition occur')
        return

    ''' Subscription Callback '''
    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        print(f'Arrived message: {dict}')

        ''' Topic datas '''
        timestamp, publisher, type, name = dict.get('timestamp'), dict.get('iPatrolAgent'), dict.get('type'), dict.get('name')
        if type != 'functionality':
            print('wrong Topic type')
            return

        arguments = dict.get('arguments')

        if self.status == INITIALIZING:
            return

        elif self.status == READY:
            return
        
        elif self.status == MOVING:
            return

        elif self.status == OBSERVING:
            return

        elif self.status == ROTATING:
            return

        elif self.status == CAPTURING: 
            return

        else:
            print('Fobbiden line')
            raise('Why code is in here')
            


if __name__ == '__main__':
    client = RobotControlAgent(agent_name='RobotControlAgent', place_name='N1Lounge8F')
    client.loop_forever()
    client.disconnect()
