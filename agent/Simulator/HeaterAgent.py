import os 
import sys
import json

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



class HeaterAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='HeaterAgent', place_name='HomeIO', 
                    time_interval=60, room='A', simul_time_interval=60):
        super().__init__(agent_name, place_name)
        self.simul_time_interval = simul_time_interval
        self.room = room # A, D, E
        self.intensity_mem = memory_map[self.room]['Heater']['intensity']
        self.power_mem = memory_map[self.room]['Heater']['power']
        self.energy_mem = memory_map[self.room]['Heater']['energy']
        self.datetime_mem = memory_map['Common']['Datetime']

        self.last_intensity = self.get_intensity()
        self.last_power = self.get_power()
        # self.last_energy = self.get_energy()
        self.stepsize = 2.0
        self.last_datetime = self.get_datatime()
        # print(self.intensity_mem, self.power_mem, self.energy_mem)

        self.sub_contexts = [f'Heater{self.room}_ON', f'Heater{self.room}_OFF', f'Heater{self.room}_UP', f'Heater{self.room}_DOWN']
        for context in self.sub_contexts:
            self.subscribe(f'{place_name}/functionality/{context}')

        self.create_timer(self.timer_callback, timer_period=time_interval)
        self.timer_cnt = 0

    def on(self):
        device = MemoryMap.Instance.GetBit(self.power_mem, MemoryType.Output)
        device.Value = True
        MemoryMap.Instance.Update()
        return self.get_power()

    def off(self):
        device = MemoryMap.Instance.GetBit(self.power_mem, MemoryType.Output)
        device.Value = False
        MemoryMap.Instance.Update()
        return self.get_power()

    def intensity_up(self):
        device = MemoryMap.Instance.GetFloat(self.intensity_mem, MemoryType.Output)
        if device.Value == 0.001:
            device.Value = self.stepsize
        else:
            device.Value = min(device.Value + self.stepsize, 10.0)
        MemoryMap.Instance.Update()
        return self.get_intensity()

    def intensity_down(self):
        device = MemoryMap.Instance.GetFloat(self.intensity_mem, MemoryType.Output)
        device.Value = max(device.Value - self.stepsize, 0.001)
        MemoryMap.Instance.Update()
        return self.get_intensity()

    def get_power(self):
        MemoryMap.Instance.Update()
        return MemoryMap.Instance.GetBit(self.power_mem, MemoryType.Output).Value

    def get_intensity(self):
        MemoryMap.Instance.Update()
        return MemoryMap.Instance.GetFloat(self.intensity_mem, MemoryType.Output).Value

    def get_energy(self):
        MemoryMap.Instance.Update()
        return MemoryMap.Instance.GetFloat(self.energy_mem, MemoryType.Memory).Value

    def get_datatime(self):
        MemoryMap.Instance.Update()
        sdt = MemoryMap.Instance.GetDateTime(self.datetime_mem, MemoryType.Memory).Value
        return sdt_to_dt(sdt)


    def timer_callback(self):
        curr_datetime = self.get_datatime()
        
        if (curr_datetime - self.last_datetime).total_seconds() < self.simul_time_interval:
            return 
        # print(curr_datetime)
        if self.timer_cnt % 60 == 0:
            self.last_intensity = self.get_intensity()
            self.publish_context(f'{self.room}_HeaterIntensity', self.last_intensity, 2)

        # print(self.get_energy())
        self.last_datetime = curr_datetime
        self.timer_cnt += 1

    
    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        
        ''' Topic datas '''
        timestamp, publisher, type, name = dict.get('timestamp'), dict.get('publisher'), dict.get('type'), dict.get('name')
        value = dict.get('value')
        
        print(f'[{self.agent_name}|{self.room}] {name}')
        
        if name == f'Heater{self.room}_ON':
            self.last_power = self.on()
            self.publish_context(f'{self.room}_HeaterPower', 'ON', 2)

        elif name == f'Heater{self.room}_OFF':
            self.last_power = self.off()
            self.publish_context(f'{self.room}_HeaterPower', 'OFF', 2)

        elif name == f'Heater{self.room}_UP':
            self.last_intensity = self.intensity_up()
            self.publish_context(f'{self.room}_HeaterIntensity', self.last_intensity, 2)

        elif name == f'Heater{self.room}_DOWN':
            self.last_intensity = self.intensity_down()
            self.publish_context(f'{self.room}_HeaterIntensity', self.last_intensity, 2)

        else:
            print('Wrong Context subscribed')
            return


if __name__ == '__main__':
    client = HeaterAgent(time_interval=0.002, room='A', simul_time_interval=60)
    client.loop_forever()
    client.disconnect()

    print(client.agent_name)
