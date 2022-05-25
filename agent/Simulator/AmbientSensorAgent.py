import os 
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('agent')[0]+'utils/HomeIO')


from agent import LaprasAgent

from utils.HomeIO import clr
from utils.HomeIO.mmap import memory_map
from utils.HomeIO.SimulDatatime import sdt_to_dt
clr.AddReference('EngineIO')
# from utils.HomeIO.EngineIO import *

# import clr
# clr.AddReference('EngineIO')
from EngineIO import *

from datetime import datetime


class AmbientSensorAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='AmbientSensorAgent', place_name='HomeIO', 
                    time_interval=60, room='A', simul_time_interval=60):
        super().__init__(agent_name, place_name)
        self.simul_time_interval = simul_time_interval
        self.room = room # A, D, E
        self.temperature_mem = memory_map[self.room]['Temperature']
        self.humidity_mem = memory_map['Common']['Humidity']
        self.datetime_mem = memory_map['Common']['Datetime']
        self.last_temperature = self.get_temperature()
        self.last_humidity = self.get_humidity()
        self.last_datetime = self.get_datatime()

        self.create_timer(self.timer_callback, timer_period=time_interval)
        self.timer_cnt = 0

    def get_temperature(self):
        MemoryMap.Instance.Update()
        return MemoryMap.Instance.GetFloat(self.temperature_mem, MemoryType.Memory).Value - 273 # Kelvin

    def get_humidity(self):
        MemoryMap.Instance.Update()
        return MemoryMap.Instance.GetFloat(self.humidity_mem, MemoryType.Memory).Value

    def get_datatime(self):
        MemoryMap.Instance.Update()
        sdt = MemoryMap.Instance.GetDateTime(self.datetime_mem, MemoryType.Memory).Value
        return sdt_to_dt(sdt)


    def timer_callback(self):
        curr_datetime = self.get_datatime()
        
        
        if (curr_datetime - self.last_datetime).total_seconds() < self.simul_time_interval:
            return 
        
        curr_temperature = self.get_temperature()
        curr_humidity = self.get_humidity()

        if self.timer_cnt % 60 == 0:
            print(f'[{self.agent_name}|{self.room}] T: {curr_temperature:.4f}')
        
        self.publish_context(f'{self.room}_Temperature', curr_temperature, 2)
        self.publish_context(f'{self.room}_Humidity', curr_humidity, 2)

        # self.last_temperature = curr_temperature
        # self.last_humidity = curr_humidity
        self.last_datetime = curr_datetime
        
        self.timer_cnt += 1



if __name__ == '__main__':
    client = AmbientSensorAgent(time_interval=0.002, room='A', simul_time_interval=60)
    client.loop_forever()
    client.disconnect()

    print(client.agent_name)
