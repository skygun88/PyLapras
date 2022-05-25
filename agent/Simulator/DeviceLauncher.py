import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')

from agent.Simulator import AmbientSensorAgent, HeaterAgent
from multiprocessing import Process



def start_client(device, room, time_interval, simul_time_interval):
    if device == 'heater':
        client = HeaterAgent.HeaterAgent(time_interval=time_interval, room=room, simul_time_interval=simul_time_interval)
    else:
        client = AmbientSensorAgent.AmbientSensorAgent(time_interval=time_interval, room=room, simul_time_interval=simul_time_interval)
    client.loop_forever()
    client.disconnect()

if __name__ == '__main__':
    
    clients = []
    processes = []

    clients.append(['ambient', 'A', 0.1, 60])
    clients.append(['heater', 'A', 0.1, 60])
    clients.append(['ambient', 'D', 0.1, 60])
    clients.append(['heater', 'D', 0.1, 60])
    
    for client in clients:
        p = Process(target=start_client, args=(client[0], client[1], client[2], client[3]))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    