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
'''
    PatrolAgent
    *  service agent for night patrol the N1 Building 8F
    * Input: 
        agent_name - Name of the lapras agent (Default: PatrolAgent)
        place_name - The place of lapras agent (Default: N1Lounge8F)
        timeout_thres - The threshold of time difference between current time and last recevied alive topic (Default: 10 (sec))
        schedules - The list of scheduled time to start patrol ex. [{start: (22, 0), end: (23, 0)}]
'''
class PatrolAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='PatrolAgent', place_name='N1Lounge8F', timeout_thres=20, schedules=[]):
        super().__init__(agent_name=agent_name, place_name=place_name)
        ''' Agent States '''
        self.status = INITIALIZING
        self.inference_alive, self.control_alive = False, False # Whether InferenceMagner & ControlAgent is alive
        self.inference_last_alive, self.control_last_alive = 0, 0 # Timestampe of the last alive topics
        self.timeout_thres = timeout_thres
        self.schedules = list(map(lambda x: {**x, 'done': False}, schedules))
        self.curr_schedule = -1 # idx of current schedule

        
        self.create_timer(self.timer_callback, timer_period=1)
        self.subscribe('N1Lounge8F/context/RobotStatus', 2) # RobotControlAgent Alive
        self.subscribe('N1Lounge8F/context/InferenceManagerStatus', 2) # InferenceManaer Alive
        self.subscribe('N1Lounge8F/context/robotComplete', 2) # Move (p0, elevator), Observe
        self.subscribe('N1Lounge8F/context/detectedHuman', 2) # Detection Result
    
    def timer_callback(self):
        self.check_alive() # check inference manager and robot agent is alive
        print(f'[{self.agent_name}/{self.place_name}] Status: {self.status} | InferenceManager: {self.inference_alive}, ControlAgent: {self.control_alive}')
        
        if self.status == INITIALIZING:
            ''' State transition: INITIALIZING -> READY '''
            if self.inference_alive and self.control_alive:
                self.status = READY
            else:
                print('Waiting for ControlAgent and InferenceManager is ready')
                return
        
        elif self.status == READY:
            ''' State transition: READY -> PATROLLING '''
            is_scheduled, schedule_idx = self.check_schedule()
            '''
            Need to implement about communication between PatrolAgent and AmbientAgent
            '''
            if is_scheduled == True:
                self.status = PATROLLING
                self.curr_schedule = schedule_idx
                self.publish_func('robotMove', [LOUNGE]) # Send robot to move lounge
            else:
                print('Waiting for robot to arrive Lounge')
                return
        
        elif self.status == PATROLLING:
            print('Waiting for robot to arrive Lounge')
            return

        elif self.status == OBSERVING:
            print('Waiting for robot to capture images')
            return

        elif self.status == INFERRING:
            print('Waiting for inference manager to detect people')
            return

        elif self.status == ACTUATING: 
            '''
            No implementation about communication with ambient agent - need to be change
            '''
            print('Waiting for all devices to turn off')
            self.status = RETURNING 
            self.publish_func('robotMove', [WAITING]) 

        elif self.status == RETURNING:
            print('Waiting for robot to return waiting place')
            return

        else:
            print('Fobbiden line')
            raise('Why code is in here')

        print(f'[{self.agent_name}/{self.place_name}] State transition occur')
        return

    def check_schedule(self):
        now = datetime.datetime.now()
        for i, schedule in enumerate(self.schedules):
            start_dt = datetime.datetime(now.year, now.month, now.day, schedule['start'][0], schedule['start'][1])
            end_dt = datetime.datetime(now.year, now.month, now.day, schedule['end'][0], schedule['end'][1])
            if end_dt < start_dt:
                end_dt = end_dt + datetime.timedelta(days=1)
            
            print(now, start_dt, end_dt, schedule)
            if now - start_dt >= datetime.timedelta(0) and now - end_dt <= datetime.timedelta(0):
                if schedule['done'] == False:
                    return True, i
        return False, -1

    ''' Subscription Callback '''
    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        print(f'Arrived message: {dict}')

        ''' Topic datas '''
        timestamp, publisher, type, name = dict.get('timestamp'), dict.get('iPatrolAgent'), dict.get('type'), dict.get('name')
        if type != 'context':
            print('wrong Topic')
            return

        value = dict.get('value')
        
        if name == 'RobotStatus':
            ''' RobotStatus Alive update '''
            if value == True:
                self.control_alive = True
                self.control_last_alive = timestamp
            return
        
        elif name == 'InferenceManagerStatus':
            ''' RobotStatus Alive update '''
            if value == True:
                self.inference_alive = True
                self.inference_last_alive = timestamp
            return
        
        elif name == 'robotComplete':
            if self.status == PATROLLING:
                ''' State transition: PATROLLING -> OBSERVING '''
                if value == 'move':
                    self.status = OBSERVING
                    self.publish_func('observe') # Request RobotControlAgent to observe Lounge
                else:
                    print('Forbidden line')
            elif self.status == OBSERVING:
                ''' State transition: OBSERVING -> INFERRING '''
                if value == 'observe':
                    self.status = INFERRING
                    self.publish_func('inference', arguments=['humandetection']) # Request InferenceManager to inference
                else:
                    print('Forbidden line')
            elif self.status == RETURNING:
                ''' State transition: RETURNING -> READY '''
                if value == 'move':
                    self.status = READY # Scenario closed
                    self.schedules[self.curr_schedule]['done'] = True
                    self.curr_schedule = -1
                else:
                    print('Forbidden line')

        elif name == 'detectedHumans':
            if self.status == INFERRING: # Get inference results
                ''' State transition: INFERRING -> ACTUATING or RETURNING'''
                if value > 0:
                    self.status = RETURNING
                    self.publish_func('robotMove', [WAITING]) # Request RobotControlAgent to waiting place
                else: # value == 0
                    self.status = ACTUATING
                    self.publish_func('turnOffDevices') # Request AmbientAgents to turn off all devices



    def check_alive(self):
        curr_ts = self.curr_timestamp()
        if curr_ts - self.inference_last_alive > self.timeout_thres*1000:
            self.inference_alive == False
        if curr_ts - self.control_last_alive > self.timeout_thres*1000:
            self.control_alive == False
            


if __name__ == '__main__':
    schedules = [
        {
            'start': (17, 20),
            'end': (18, 00),
        }
    ]
    client = PatrolAgent(agent_name='PatrolAgent', place_name='N1Lounge8F', schedules=schedules)
    client.loop_forever()
    client.disconnect()
