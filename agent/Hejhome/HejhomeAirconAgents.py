import os 
import sys
import json
import time
import requests
import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from agent import LaprasAgent
from utils.configure import * 

''' 
Subscribe - ON/OFF, UP/DOWN, Fan UP/DOWN
Publish - Aircon State - Power [ON|OFF] / SetPoint [25-28] / FanSpeed [1|2|3]
'''

MIN_TEM, MAX_TEM = 23, 28 

class HejhomeAirconAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='HejhomeAirconAgent', place_name='N1Lounge8F'):
        super().__init__(agent_name=agent_name, place_name=place_name)
        ''' Agent States '''
        self.power = None
        self.temperature = None
        self.mode = None
        self.fanspeed = None   
        self.api_path = os.path.join(RESOURCE_PATH, 'api_data.json')
        
        with open(self.api_path, 'r') as f:
            self.api_info = json.load(f)
            f.close()

        self.create_timer(self.timer_callback, timer_period=1)
        self.timer_cnt = 0

        ''' Subscribe the control message '''
        self.sub_list = ['TurnOffAircon', 'TurnOnAircon', 'TempUpAircon', 'TempDownAircon', 'SetTempAircon']
        for name in self.sub_list:
            self.subscribe(f'{place_name}/functionality/{name}')
        self.func_list = [self.turn_off_aircon, self.turn_on_aircon, self.temp_up_aircon, self.temp_down_aircon, self.set_temp_aircon]
        self.sub_funcs = dict(zip(self.sub_list, self.func_list))

    def timer_callback(self):
        print(f'Power: {self.power} | Temperature: {self.temperature} | Mode: {self.mode} | FanSpeed: {self.fanspeed}')
        

        if self.timer_cnt % 30 == 0:
            self.update()
        

        ''' Calibrate Aircon state against central control '''
        if self.timer_cnt % 60*3 == 0:
            if self.power == True:
                self.set_aircon(power=self.power, temperature=self.temperature)
            else:
                self.set_aircon(power=self.power)

        self.timer_cnt += 1
        if self.timer_cnt % 2 == 0:
            self.publish_context('AirconAlive', True)
        
        
        return


    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        msg_dict = json.loads(dict_string)

        ''' Topic datas '''
        timestamp, publisher, type, name = msg_dict.get('timestamp'), msg_dict.get('publisher'), msg_dict.get('type'), msg_dict.get('name')
        arguments = msg_dict.get('arguments')
        
        ''' Actuate Aircon '''
        print(f"Request - {name} | {arguments}")
        self.sub_funcs[name](arguments)

        return 

    def turn_on_aircon(self, dummy):
        self.set_aircon(power=True)
        self.update()
        return

    def turn_off_aircon(self, dummy):
        self.set_aircon(power=False)
        self.update()
        return

    def temp_up_aircon(self, dummy):
        curr_temperature = self.temperature
        curr_power = self.power
        if (curr_temperature == None) or (curr_power != True):
            return
        
        if (curr_temperature < MAX_TEM):
            new_tempeature = curr_temperature + 1
            self.set_aircon(power=curr_power, temperature=new_tempeature)
            self.update()

        return
        

    def temp_down_aircon(self, dummy):
        curr_temperature = self.temperature
        curr_power = self.power
        if (curr_temperature == None) or (curr_power != True):
            return
        
        if (curr_temperature > MIN_TEM):
            new_tempeature = curr_temperature - 1
            self.set_aircon(power=curr_power, temperature=new_tempeature)
            self.update()

        return

    def set_temp_aircon(self, arguments):
        temperature = arguments[0]
        print(temperature)
        if (temperature < MIN_TEM) or (temperature > MAX_TEM):
            return 
        self.set_aircon(power=True, temperature=temperature)
        self.update()
        return

    def update(self):
        transition = self.update_state()
        if transition == True:
            self.publish_aircon_state()
            print('Transition Occur')        
        return


    def update_state(self):
        ACCESS_TOKEN = self.api_info['ACCESS_TOKEN']
        AIRCON_ID = self.api_info['IR_AIRCON_ID']
        
        url = f"https://goqual.io/openapi/device/{AIRCON_ID}"

        payload = json.dumps({})

        headers = {
            'Authorization': f'Bearer {ACCESS_TOKEN}'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        state_map = json.loads(response.text)

        power = True if state_map['deviceState']['power']  == '켜짐' else False
        temperature = int(state_map['deviceState']['temperature'])
        mode = int(state_map['deviceState']['mode'])
        fanspeed = int(state_map['deviceState']['fanSpeed'])

        transition = (self.power != power) or (self.temperature != temperature) or (self.mode != mode) or (self.fanspeed != fanspeed)
        self.power = power
        self.temperature = temperature
        self.mode = mode
        self.fanspeed = fanspeed

        return transition
    
    def publish_aircon_state(self):
        name_list = ["AirconPower", "AirconTemp"]
        value_list = [self.power, self.temperature]

        for name, value in zip(name_list, value_list):
            self.publish_context(name, value, retain=True)

        return
    
    def set_aircon(self, power=None, temperature=None, mode=None, speed=None):
        ACCESS_TOKEN = self.api_info['ACCESS_TOKEN']
        AIRCON_ID = self.api_info['IR_AIRCON_ID']
        
        url = f"https://goqual.io/openapi/control/{AIRCON_ID}"

        payload_dict = {'requirments': {}}
        
        if (power !=None) and (power in [True, False]):
            payload_dict['requirments']['power'] = "true" if power == True else "false"
        if (temperature !=None) and (temperature in range(MIN_TEM, MAX_TEM+1)):
            payload_dict['requirments']['temperature'] = temperature
        if (mode !=None) and (mode in range(0, 4+1)):
            payload_dict['requirments']['mode'] = str(mode)
        if (speed !=None) and (speed in range(0, 3+1)):
            payload_dict['requirments']['fanSpeed'] = str(speed)

        payload = json.dumps(payload_dict)

        headers = {
            'Authorization': f'Bearer {ACCESS_TOKEN}',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)


        return response.status_code
        



if __name__ == '__main__':
    strict = True
    strict = False
    client = HejhomeAirconAgent(agent_name='HejhomeAirconAgent', place_name='N1Lounge8F')
    client.loop_forever()
    client.disconnect()
