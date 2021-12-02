import os 
import sys
import json
import time
import LaprasAgent

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

''' 0: Initializing, 11: Ready, 12: Patrolling, 13: Observing, 14: Rotating, 15: Inferring, 16: Actuating, 17: Returning '''
INITIALIZING, READY, PATROLLING, OBSERVING, ROTATING, INFERRING, ACTUATING, RETURNING = 0, 1, 12, 13, 14, 15, 16, 17

class PatrolAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='PatrolAgent', place_name='N1Lounge8F', timeout_thres=10):
        super().__init__(agent_name=agent_name, place_name=place_name)
        ''' Agent States '''
        self.status = INITIALIZING
        self.control_ready = False
        self.inference_alive, self.control_alive = False, False # Whether InferenceMagner & ControlAgent is alive
        self.inference_last_alive, self.control_last_alive = 0, 0 # Timestampe of the last alive topics
        self.timeout_thres = timeout_thres

        self.connect()
        self.create_timer(self.timer_callback, timer_period=3)
        self.subscribe('N1Lounge8F/functionality/DetectHuman', 2)
        
        

    
    def timer_callback(self):
        self.check_alive() # check inference manager and robot agent is alive
        print(f'[{self.agent_name}/{self.place_name}] Status: {self.status} | InferenceManager: {self.inference_alive}, ControlAgent: {self.control_alive}')

        ''' State transition: INITIALIZING -> READY '''
        if self.status == INITIALIZING:
            if self.inference_alive and self.control_alive and self.control_ready:
                self.status = READY
        
        


    ''' Subscription Callback '''
    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        print(f'Arrived message: {dict}')

        ''' Topic datas '''
        timestamp = dict.get('timestamp')
        publisher = dict.get('iPatrolAgent')
        type = dict.get('type')
        name = dict.get('name')
        if type == 'functionality':
            args = dict.get('arguments')
        elif type == 'context':
            value = dict.get('value')
            
        else:
            print('wrong subscription')
            return


    def check_alive(self):
        curr_ts = self.curr_timestamp()
        if curr_ts - self.inference_last_alive > self.timeout_thres*1000:
            self.inference_alive == False
        if curr_ts - self.control_last_alive > self.timeout_thres*1000:
            self.control_alive == False
        


if __name__ == '__main__':
    client = PatrolAgent(agent_name='PatrolAgent', place_name='N1Lounge8F')
    client.loop_forever()
    client.disconnect()
