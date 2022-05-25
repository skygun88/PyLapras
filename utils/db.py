import os
import re
import sys
import json
import torch
import numpy as np
import pymongo as pm
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd


PYLAPRAS_PATH = os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras/'
RESOURCE_PATH = PYLAPRAS_PATH + 'resources/'
sys.path.append(PYLAPRAS_PATH)
from utils.comfort import PMV
# from utils.dqn import DQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def upload_replay(place, start_ts, ts, state):
    ''' Connect DB '''
    json_file = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/resources/db_info.json'
    with open(json_file, 'r') as f:
        db_info = json.load(f)
        f.close()

    client = pm.MongoClient(db_info['address'])
    client.data.authenticate(db_info['autheticate']['name'], db_info['autheticate']['pw'], mechanism=db_info['autheticate']['mechanism'])
    db = client.data
    place_to_db = {'N1Lounge8F': db.N1Lounge8F_replay, 'N1SeminarRoom825': db.N1SeminarRoom825_replay, 'HomeIO': db.HomeIO_replay}
    data = place_to_db[place]

    ''' Upload Data document '''
    data.insert_one({   
        'start': start_ts,
        'timestamp': ts, 
        'state': state})
        
    client.close()

# def preprocessing(ts_state):
#     max_tem, min_tem, max_hum = 30, 15, 20
#     tem_preprocess = np.vectorize(lambda x: (max(min(x, max_tem), min_tem) - min_tem)/(max_tem-min_tem))
#     hum_preprocess = np.vectorize(lambda x: min(x, max_hum)/max_hum)
#     result = np.zeros_like(ts_state)
#     result[:, 0:2] = tem_preprocess(ts_state[:, 0:2])
#     result[:, 2:4] = hum_preprocess(ts_state[:, 2:4])
#     result[:, 4:6] = ts_state[:, 4:6]
#     return result.T

def preprocessing(ts_state):
    max_tem, min_tem, max_hum = 25, 10, 100
    tem_preprocess = np.vectorize(lambda x: (max(min(x, max_tem), min_tem) - min_tem)/(max_tem-min_tem))
    hum_preprocess = np.vectorize(lambda x: min(x, max_hum)/max_hum)
    result = np.zeros_like(ts_state)
    result[:, 0:2] = tem_preprocess(ts_state[:, 0:2])
    # result[:, 2:4] = hum_preprocess(ts_state[:, 2:4])
    result[:, 2:4] = ts_state[:, 2:4]
    result[:, 4:6] = ts_state[:, 4:6]/10
    return result.T

def db_parser(place, window_size=15, start=None):
    ''' Connect DB '''
    json_file = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/resources/db_info.json'
    with open(json_file, 'r') as f:
        db_info = json.load(f)
        f.close()

    start_dt = 0 if start == None else start-1

    client = pm.MongoClient(db_info['address'])
    client.data.authenticate(db_info['autheticate']['name'], db_info['autheticate']['pw'], mechanism=db_info['autheticate']['mechanism'])
    db = client.data
    place_to_db = {'N1Lounge8F': db.N1Lounge8F_replay, 'N1SeminarRoom825': db.N1SeminarRoom825_replay, 'HomeIO': db.HomeIO_replay}
    data = place_to_db[place]

    queries = [[d['start'], d['timestamp'], d['state']] for d in \
        data.find({'start': {'$gt': start_dt}}).sort('timestamp')]
    
    
    
    episode_starts = sorted(list(set([query[0] for query in queries])))
    episodes = []

    for i, episode_start in enumerate(episode_starts):
        episodes.append(list(map(lambda x: x[2], filter(lambda x: x[0] == episode_start, queries))))

    replay_memory = []
    for e, episode in enumerate(episodes):
        if len(episode) < window_size*2+1:
            continue

        for i in range((len(episode)//window_size)-1):
            state, next_state = np.array(episode[i*window_size:(i+1)*window_size]), np.array(episode[(i+1)*window_size:(i+2)*window_size])
            actions = np.around(np.mean((np.around(next_state[:, 4:], 1)>np.around(state[:, 4:], 1))+(np.around(next_state[:, 4:], 1)==10), axis=0))
            
            # on_time = np.sum(actions)*15
            on_time = np.sum(np.around(next_state[:, 4:], 1))          
            
            energy_param = 0.002

            action_map = [[0, 2], [1, 3]]
            action = action_map[int(actions[0])][int(actions[1])]

            mean_tem, mean_hum = np.mean(next_state[:, 0:2]), np.mean(next_state[:, 2:4])
            # pmv = PMV(clothing_index=0.7, metabolism_index=1.0, temperature=mean_tem, humidity=mean_hum).calculatePMV()
            # pmv = PMV(clothing_index=0.36, metabolism_index=1.8, temperature=mean_tem, humidity=mean_hum).calculatePMV()
            pmv = PMV(clothing_index=0.5, metabolism_index=1.0, temperature=mean_tem, humidity=mean_hum).calculatePMV()

            reward = -abs(pmv) if abs(pmv) > 0.5 else 0.5
            if reward > 0:
                reward = reward - energy_param*on_time

            state = preprocessing(state)
            next_state = preprocessing(next_state)

            # if i < 100:
            if e == len(episodes)-1:
                print(mean_tem, pmv, reward, actions, np.mean(next_state[4:, :], axis=1))
                

            replay_memory.append([state, action, next_state, reward])


    print(len(queries), len(replay_memory))
    return replay_memory



def replay_analysis(place, start_dt, task_info):
    json_file = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/resources/db_info.json'
    with open(json_file, 'r') as f:
        db_info = json.load(f)
        f.close()

    client = pm.MongoClient(db_info['address'])
    client.data.authenticate(db_info['autheticate']['name'], db_info['autheticate']['pw'], mechanism=db_info['autheticate']['mechanism'])
    db = client.data
    place_to_db = {'N1Lounge8F': db.N1Lounge8F_replay, 'N1SeminarRoom825': db.N1SeminarRoom825_replay, 'HomeIO': db.HomeIO_replay}
    data = place_to_db[place]

    queries = [d['state'] for d in \
        data.find({'start': start_dt}).sort('timestamp')]
    
    # print(len(queries))
    # print(queries[:10])
    # for x in queries[:10]:
    #     print(x)
    
    total_time = min(len(queries), 7200)
        
    tems = [(x[0]+x[1])/2 for x in queries[-total_time:]]
    # pmvs = [abs(PMV(clothing_index=0.67, metabolism_index=1.3, temperature=x, humidity=0.54).calculatePMV()) for x in tems]
    # pmvs = [abs(PMV(clothing_index=0.36, metabolism_index=1.8, temperature=x, humidity=0.54).calculatePMV()) for x in tems]
    pmvs = [abs(PMV(clothing_index=task_info['clothing_index'], metabolism_index=task_info['metabolism_index'], temperature=x, humidity=0.54).calculatePMV()) for x in tems]
    energies = [x[4]*200+x[5]*175 for x in queries[-total_time:]]
    
    # plt.plot(pmvs)
    # plt.plot(energies)
    # plt.plot(tems)
    # plt.show()
    
    print(sum(tems)/total_time, sum(pmvs)/total_time, sum(energies)/total_time)
    time_len = 60*3
    # mean_tems = [0]*time_len
    # mean_pmvs = [0]*time_len
    # mean_energies = [0]*time_len
    # for i in range(len(tems)):
    #     idx = i%time_len
    #     mean_tems[idx] += tems[i]/(7200/time_len)
    #     mean_pmvs[idx] += pmvs[i]/(7200/time_len)
    #     mean_energies[idx] += energies[i]/(7200/time_len)
    #     print(idx, i)
        
    avg_tems = np.array(tems).reshape([-1, time_len])
    avg_pmvs = np.array(pmvs).reshape([-1, time_len])
    avg_energies = np.array(energies).reshape([-1, time_len])
    
    mean_tems = np.mean(avg_tems, axis=0)
    mean_pmvs = np.mean(avg_pmvs, axis=0)
    mean_energies = np.mean(avg_energies, axis=0)
    
    std_tems = np.std(avg_tems, axis=0) / np.sqrt(np.size(avg_tems, axis=1))
    std_pmvs = np.std(avg_pmvs, axis=0) / np.sqrt(np.size(avg_pmvs, axis=1))
    std_energies = np.std(avg_energies, axis=0) / np.sqrt(np.size(avg_energies, axis=1))
                
    # result_data = {'temperature': tems, 'pmv': pmvs, 'energy': energies}
    result_data = {'temperature': mean_tems, 
                   'pmv': mean_pmvs, 
                   'energy': mean_energies, 
                   'temperature_std': std_tems,
                   'pmv_std': std_pmvs,
                   'energy_std': std_energies
                   }
    
    
    return pd.DataFrame(result_data)

if __name__ == '__main__':
    TASK_PATH = RESOURCE_PATH + 'tasks.json'
    with open(TASK_PATH, 'r') as f:
        tasks_info = json.load(f)
        f.close()
    
    # db_parser('N1Lounge8F', window_size=15)
    # db_parser('N1SeminarRoom825', window_size=1)
    
    # db_parser('HomeIO', window_size=15, start=1650475275168)
    
    start_dt = 1650525780547 # RL1 - leisure
    rl_result = replay_analysis('HomeIO', start_dt, task_info=tasks_info['Leisure'])
    
    start_dt = 1650535640376 # RL2 - Excercise
    rl_result2 = replay_analysis('HomeIO', start_dt, task_info=tasks_info['Excercise'])
    
    start_dt = 1650543494514 # RL3 - Rest
    rl_result3 = replay_analysis('HomeIO', start_dt, task_info=tasks_info['Rest'])
    
    start_dt = 1650528304403 # Rule 1
    rule_result = replay_analysis('HomeIO', start_dt, task_info=tasks_info['Leisure'])
    
    # start_dt = 1650528304403 # Rule 2
    # rule_result2 = replay_analysis('HomeIO', start_dt, task_info=tasks_info['Excercise'])
    
    # start_dt = 1650528304403 # Rule 2
    # rule_result3 = replay_analysis('HomeIO', start_dt, task_info=tasks_info['Rest'])
    
    # tems = pd.DataFrame({'RL - Leisure': rl_result['temperature'], 'Rule - Leisure': rule_result['temperature']}, columns=['label', 'temperature'])
    
    plot_list = [
        [rule_result, 'Rule - Leisure'],
        [rl_result, 'Proposed - Leisure'], 
        [rl_result2, 'Proposed - Excercise'],
        [rl_result3, 'Proposed - Rest']
    ]
    
    # target = 'energy'
    # for r, l in plot_list:
    #     plt.plot(range(len(r[target])), r[target], label=l)
    #     plt.fill_between(range(len(r[target])), r[target]-r[target+'_std'], r[target]+r[target+'_std'], alpha=0.3)
    # plt.ylim(0, 4000)
    # plt.xlim(0, len(rl_result[target]))
    # plt.xlabel('time (min)')
    # plt.ylabel('Energy Consumption (W)')
    # plt.legend()
    # plt.show()
    
    target = 'pmv'
    for r, l in plot_list:
        plt.plot(range(len(r[target])), r[target], label=l)
        plt.fill_between(range(len(r[target])), r[target]-r[target+'_std'], r[target]+r[target+'_std'], alpha=0.3)
    plt.ylim(0, 0.5)
    plt.xlim(0, len(rl_result[target]))
    plt.xlabel('time (min)')
    plt.ylabel('PMV violation')
    plt.legend()
    plt.show()
    
    
    # target = 'pmv'
    # plt.plot(rule_result[target], label='Rule - Leisure')
    # plt.plot(rule_result2[target], label='Rule - Excercise')
    # plt.plot(rule_result3[target], label='Rule - Rest')
    # plt.ylim(0, 1)
    # plt.xlim(0, 60*24)
    # plt.xlabel('time (min)')
    # plt.ylabel('PMV violation')
    # plt.legend()
    # plt.show()
    
    # fig, ax = plt.subplots()

    # target = 'pmv'
    # ax.boxplot([rl_result[target], rule_result[target]], sym="b*")
    # plt.xticks([1, 2], ['RL - Leisure', 'Rule - Leisure'])
    # plt.ylabel('PMV violation')
    # plt.show()


    # target = 'energy'
    # ax.boxplot([rl_result[target], rule_result[target]], sym="b*")
    # plt.xticks([1, 2], ['RL - Leisure', 'Rule - Leisure'])
    # plt.ylabel('PMV violation')
    # plt.show()
