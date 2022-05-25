import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import time
import datetime
from agent import LaprasAgent



''' 0: Initializing, 1: Ready, 12: Patrolling, 13: Observing, 14: Inferring, 15: Actuating, 16: Returning, 17: Docking, 18: Undocking '''
INITIALIZING, READY, PATROLLING, OBSERVING, INFERRING, ACTUATING, RETURNING, DOCKING, UNDOCKING = 0, 1, 12, 13, 14, 15, 16, 17, 18
# LOUNGE, WAITING = 'p0', 'elevator'
LOUNGE, WAITING = 'p0', 'docking_station'
STATE_MAP = {0: 'INITIALIZING', 1: 'READY', 12: 'PATROLLING', 13: 'OBSERVING', 14: 'INFERRING', 15: 'ACTUATING', 16: 'RETURNING', 17: 'DOCKING', 18: 'UNDOCKING'}

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
    def __init__(self, agent_name='PatrolAgent', place_name='N1Lounge8F', timeout_thres=30, schedules=[]):
        super().__init__(agent_name=agent_name, place_name=place_name)
        ''' Agent States '''
        self.status = INITIALIZING
        self.inference_alive, self.control_alive = True, False # Whether InferenceMagner & ControlAgent is alive
        self.inference_last_alive, self.control_last_alive = 0, 0 # Timestampe of the last alive topics
        self.timeout_thres = timeout_thres
        self.schedules = list(map(lambda x: {**x, 'done': False}, schedules))
        self.curr_schedule = -1 # idx of current schedule
        self.patrol_need = False
        self.curr_light = -1
        
        self.create_timer(self.timer_callback, timer_period=1)
        self.subscribe(f'{place_name}/context/RobotControlAgentOperatingStatus', 2) # RobotControlAgent Alive
        self.subscribe(f'{place_name}/context/inferenceManagerStatus', 2) # InferenceManaer Alive
        self.subscribe(f'{place_name}/context/robotComplete', 2) # Move (p0, elevator), Observe
        self.subscribe(f'{place_name}/context/detectedhumans', 2) # Detection Result
        self.subscribe(f'{place_name}/context/LightGroup1', 2)
        self.publish_context('PatrolAgentStatus', STATE_MAP[self.status], 2)

    def transition(self, next_state):
        print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_MAP[self.status]}->{STATE_MAP[next_state]}')          
        self.status = next_state
        self.publish_context(f'{self.agent_name}Status', STATE_MAP[self.status], 2)

    def timer_callback(self):
        self.check_alive() # check inference manager and robot agent is alive
        print(f'[{self.agent_name}/{self.place_name}] Status: {STATE_MAP[self.status]} | InferenceManager: {self.inference_alive}, ControlAgent: {self.control_alive}')

        if self.status == INITIALIZING:
            ''' State transition: INITIALIZING -> READY '''
            if self.inference_alive and self.control_alive:
                self.transition(next_state=READY)
            else:
                return
        
        elif self.status == READY:
            ''' State transition: READY -> PATROLLING '''
            is_scheduled, schedule_idx = self.check_schedule()
            '''
            Need to implement about communication between PatrolAgent and AmbientAgent
            '''
            if is_scheduled == True and self.curr_light == 1:
                print(f'[{self.agent_name}/{self.place_name}] Scheduled Time: {is_scheduled} & Light: {self.curr_light}')
                print('Request Robot to move to Lounge')
                self.transition(next_state=UNDOCKING)
                self.curr_schedule = schedule_idx
                self.publish_func('undocking') # Send robot to move lounge
            else:
                print(f'Not scheduled time or Lounge is turned off - Light: {self.curr_light == 1}')
                return
        
        elif self.status == UNDOCKING:
            print('Waiting for robot to undock')
            return

        elif self.status == DOCKING:
            print('Waiting for robot to dock')
            return
        
        elif self.status == PATROLLING:
            print('Waiting for robot to arrive Lounge')
            return

        elif self.status == OBSERVING:
            # self.publish_func('inference', arguments=['humandetection'])
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
            # self.status = RETURNING 
            # print(f'[{self.agent_name}/{self.place_name}] State transition: {STATE_MAP[ACTUATING]}->{STATE_MAP[RETURNING]}')
            # self.publish_func('robotMove', [WAITING]) 
            return

        elif self.status == RETURNING:
            print('Waiting for robot to return waiting place')
            return

        else:
            print('Fobbiden line')
            raise('Why code is in here')

        # self.publish_context('PatrolAgentOperatingStatus', True, 2)
        return

    def check_schedule(self):
        now = datetime.datetime.now()
        for i, schedule in enumerate(self.schedules):
            start_dt = datetime.datetime(now.year, now.month, now.day, schedule['start'][0], schedule['start'][1])
            end_dt = datetime.datetime(now.year, now.month, now.day, schedule['end'][0], schedule['end'][1])
            if end_dt < start_dt:
                end_dt = end_dt + datetime.timedelta(days=1)
            
            # print(now, start_dt, end_dt, schedule)
            if now - start_dt >= datetime.timedelta(0) and now - end_dt <= datetime.timedelta(0):
                if schedule['done'] == False:
                    return True, i
        return False, -1

    ''' Subscription Callback '''
    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        # print(f'Arrived message: {dict}')

        ''' Topic datas '''
        timestamp, publisher, type, name = dict.get('timestamp'), dict.get('publisher'), dict.get('type'), dict.get('name')
        
        value = dict.get('value')
        # print(f'[{name}: {value}] Message is arrived from {publisher}')

        if name == 'RobotControlAgentOperatingStatus':
            ''' RobotStatus Alive update '''
            if value == True:
                self.control_alive = True
                self.control_last_alive = timestamp
            return
        
        elif name == 'inferenceManagerStatus':
            ''' RobotStatus Alive update '''
            if value == True:
                self.inference_alive = True
                self.inference_last_alive = timestamp
            return
        
        elif name == 'robotComplete':
            print(f'[{self.agent_name}/{self.place_name}] {publisher} - {name}: {value}')
            if self.status == PATROLLING:
                ''' State transition: PATROLLING -> OBSERVING '''
                if value == 'move':
                    self.transition(next_state=OBSERVING)
                    self.publish_func('observe') # Request RobotControlAgent to observe Lounge
                else:
                    print('Forbidden line')
            elif self.status == OBSERVING:
                ''' State transition: OBSERVING -> INFERRING '''
                if value == 'observe':
                    self.transition(next_state=INFERRING)
                    self.publish_func('inference', arguments=['humandetection']) # Request InferenceManager to inference
                else:
                    print('Forbidden line')
            elif self.status == RETURNING:
                ''' State transition: RETURNING -> READY '''
                if value == 'move':
                    self.transition(next_state=DOCKING)
                    self.publish_func('docking')

                else:
                    print('Forbidden line')

            elif self.status == UNDOCKING:
                ''' State transition: UNDOCKING -> PATROLLING '''
                if value == 'undocking':
                    self.transition(next_state=PATROLLING) # Scenario closed
                    self.publish_func('robotMove', [LOUNGE]) # Send robot to move lounge
                else:
                    print('Forbidden line')
            
            elif self.status == DOCKING:
                ''' State transition: DOCKING -> READY '''
                if value == 'docking':
                    self.transition(next_state=READY) # Scenario closed
                    self.schedules[self.curr_schedule]['done'] = True
                    self.curr_schedule = -1
                else:
                    print('Forbidden line')
                

        elif name == 'detectedhumans':
            if self.status == INFERRING: # Get inference results
                ''' State transition: INFERRING -> ACTUATING or RETURNING'''
                print(f'[{self.agent_name}/{self.place_name}] {publisher} - Human Detection: {value}')
                if value == True:
                    self.transition(next_state=RETURNING)
                    self.publish_func('robotMove', [WAITING]) # Request RobotControlAgent to waiting place
                else: # value == 0
                    self.transition(next_state=ACTUATING)
                    self.turnOffAllDevices()
            
        elif name == 'LightGroup1':
            self.curr_light = 1 if value == 'On' else 0
            if self.status == ACTUATING:
                if value == 'Off':
                    self.transition(next_state=RETURNING)
                    self.publish_func('robotMove', [WAITING])


    def check_alive(self):
        curr_ts = self.curr_timestamp()
        # if curr_ts - self.inference_last_alive > self.timeout_thres*1000:
        #     self.inference_alive == False
        print(curr_ts, self.control_last_alive, self.timeout_thres*1000, curr_ts - self.control_last_alive > self.timeout_thres*1000, self.control_alive)
        if curr_ts - self.control_last_alive > self.timeout_thres*1000:
            self.control_alive = False
            
    def turnOffAllDevices(self):
        # name_list = ["TurnOffAllLights", "TurnOffFan", "StopAircon0", "StopAircon1", "StopAircon0", "StopAircon1"]
        name_list = ["TurnOffAllLights", "TurnOffFan", "StopAircon0", "StopAircon1"]
        # name_list = ["TurnOnAllLights", "TurnOnFan", "TurnOnFanRotation", "StartAircon0", "StartAircon1"]
        for name in name_list:
            print(f'Turn off {name.replace("TurnOff", "").replace("Stop", "")}')
            self.publish_func(name)
    

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    with open('resources/N1Lounge8F/patrolagent.json', 'r') as f:
        configs = json.loads(f.read())
        f.close()
    
    agent_name = configs['agent_name'] 
    place_name = configs['place_name'] 
    arguments = configs['arguments'] 
    schedules = arguments['schedules']

    # print(agent_name, place_name, arguments)
    client = PatrolAgent(agent_name=agent_name, place_name=place_name, schedules=schedules)
    client.loop_forever()
    client.disconnect()
