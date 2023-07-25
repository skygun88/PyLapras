import os 
import sys
import json
import time
import requests
import datetime
#import psutil
#import rospy
#from sensor_msgs.msg import JoinState

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from agent import LaprasAgent
from utils.configure import *

max_speed, min_speed= 3, 1
max_deg, min_deg, mid_deg= 180, 0, 90
class HejhomeFanAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='HejhomeFanAgent', place_name='N1Lounge8F'):
        super().__init__(agent_name=agent_name, place_name=place_name)
        self.power=False
        self.fanSpeed=None
        self.swing=None
        self.api_path=os.path.join(RESOURCE_PATH, 'api_data.json')

        with open(self.api_path, 'r') as fs:
            self.api_info=json.load(fs)
            fs.close()

        self.create_timer(self.timeCallback, timer_period=1)
        self.tcnt=0
        self.fan_list=["FanOn","FanOff", "HighSpeedFan", "LowSpeedFan"]
        for nm in self.fan_list:
            self.subscribe(f'{place_name}/functionality/{nm}')
        self.fan_func_list=[self.fan_on, self.fan_off, self.high_speed, self.low_speed]
        self.sub_funcs=dict(zip(self.fan_list, self.fan_func_list))

    def timeCallback(self):
        print(f'Power: {self.power} | Speed: {self.fanSpeed} | Rotation: {self.swing}')
        #self.update()

        # if self.tcnt%5==0:
        #     if self.power==True:
        #           self.set_fan(power=self.power, speed=self.fanSpeed)
                            
        #     else:
        #         self.set_fan(power=self.power)

        self.tcnt+=1
        if self.tcnt % 2 == 0:
            self.publish_context('FanAlive', True)

        return

    def on_message(self, client, userdata, msg):
        dict_string=str(msg.payload.decode("utf-8"))
        msg_dict=json.loads(dict_string)

        name, type, publisher, timestamp= msg_dict.get("name"), msg_dict.get("type"), msg_dict.get("publisher"), msg_dict.get("timestamp")
        arguments=msg_dict.get("arguments")

        print(self.sub_funcs)
        print(f'Call: {name} | {arguments}')
        self.sub_funcs[name](arguments)
        return

    def fan_on(self, dummy):
        self.power = True
        self.set_fan(power=True)
        
        #self.update()
        return
    def fan_off(self, dummy):
        self.power = False
        self.set_fan(power=False)
        
        #self.update()
        return


    def high_speed(self, dummy):
        curr_speed=min_speed
        if(curr_speed==None):
            return
        if(curr_speed<max_speed):
            self.set_fan(fanSpeed=curr_speed+1)
            return
   # def mid_speed(self, dummy):
    #    curr_speed=self.fanSpeed
     #   if(curr_speed==None):
      #      return
       # if(curr_speed==max_speed-1):
        #    self.set_fan(fanSpeed=curr_speed)
         #   return

    def low_speed(self, dummy):
        curr_speed=max_speed
        if(curr_speed==None):
            return
        if(curr_speed>=min_speed):
            self.set_fan(fanSpeed=curr_speed-1)
            return


   # def set_rotation_speed(self, dummy):
    #    curr_rotate=mid_deg
        #curr_power=self.power
     #   if(curr_rotate==None):
      #      return
       # if (curr_rotate>min_deg)or(curr_rotate<max_deg):
        #    self.set_fan(power=True, swing=curr_rotate)
         #   self.update()
        #return
    #def hold_fan(self):
        #curr_rotate=self.swing()
        #curr_power=self.power()
        #if(curr_rotate==None)or(curr_power!=True):
            #return
        #if(curr_rotate<min_deg)or(curr_rotate==mid_deg)or(curr_rotate>max_deg):
            #self.set_fan(power=True, rotate=curr_rotate)
            #self.update()
        #return
    #def set_rotation(self, arguments):
        #fan_rotate=arguments[0]
        #if(fan_rotate>min_deg)or(fan_rotate==mid_deg)or(fan_rotate<=max_deg):
         #   return
        #self.set_fan(power=True, rotate=fan_rotate)
        #self.update()
        #return
    #def update(self):
       # state_change=self.update_state()
        #if state_change==True:
         #   self.publish_fan_state()
          #  print("Change occurred")
        #return
   # def update_state(self):
       # ACCESS_TOKEN=self.api_info["ACCESS_TOKEN"]
        #FAN_ID=self.api_info["IR_FAN_ID"]

        #url = f"https://goqual.io/openapi/control/{FAN_ID}"

        #payload = json.dumps({
             
         # "requirements": {
    	  #"power": "true",
    	  #"swing": "90",
          #"fanSpeed": "3"
           #}
         #})
        #headers = {
       #'Authorization': f'Bearer{FAN_ID} ',
       #'Content-Type': 'application/json;charset-UTF-8'
      #}

       # response = requests.request("GET", url, headers=headers, data=payload)    
        #state_map=json.loads(response.text)
        #power=True if state_map['requirements']['power']=='전원'else False
        #fanspeed=int(state_map['requirements']['fanspeed'])
        #swing=int(state_map['requirements']['rotate'])

        #state_change=(self.power!=power)or(self.fanspeed!=fanspeed)or(self.swing!=swing)
        #self.power=power
        #self.fanspeed=fanspeed
        #self.swing=swing
        #return state_change
    def publish_fan_info(self):
        names=["FanPower"]
        values=[self.power]
        # for nm in zip(names):
        self.publish_context(names[0], values[0], retain=True)
        return

    def set_fan(self, power=None, fanSpeed=None, swing=None):
        ACCESS_TOKEN=self.api_info["ACCESS_TOKEN"]
        FAN_ID=self.api_info["IR_FAN_ID"]

        url = f"https://goqual.io/openapi/control/{FAN_ID}"

        payload_dict={"requirments":{}}
        if (power!=None) and (power in [True, False]):
            payload_dict["requirments"]["value"]="TurnON" if power==True else "TurnOFF"
        elif (fanSpeed!=None)and(fanSpeed in range(min_speed, max_speed+1)):
            payload_dict["requirments"]["value"]= 'WindSpeed'
        elif (swing!=None)and(swing in range(min_deg, max_deg+1)):
            payload_dict["requirments"]["value"]= 'Rotate'

        # print(payload_dict)


        payload = json.dumps(payload_dict)
        print(payload)
        headers = {
        'Authorization': f'Bearer{ACCESS_TOKEN}',
        'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)

        print(response.text)

        self.publish_fan_info()


if __name__ == '__main__':
    strict = True
    strict = False
    client = HejhomeFanAgent(agent_name='HejhomeFanAgent', place_name='N1Lounge8F')
    client.loop_forever()
    client.disconnect()






            


                




