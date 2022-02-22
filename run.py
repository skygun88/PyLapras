import os 
import sys
import json
import importlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from agent import *


def run():
    print(sys.argv, len(sys.argv))
    if len(sys.argv) < 4:
        print(f'Need {4-len(sys.argv)} arguments - ex) python start LaprasAgent N1Lounge8F/patrolagent.json')
        sys.exit()
    
    command = sys.argv[1]
    target = sys.argv[2]
    config_path = f'resources/{sys.argv[3]}'

    with open(config_path, 'r') as f:
        configs = json.loads(f.read())
        f.close()
    
    agent_class_name = configs['agent_class_name']
    agent_name = configs['agent_name'] 
    place_name = configs['place_name'] 
    arguments = configs['arguments'] 

    print(command, target, config_path)
    print(agent_class_name, agent_name, place_name, arguments)
    # print(sys.modules[__name__])
    # for key, value in sys.modules.items():
    #     print(key, '|', value)
    # print(str_to_class(agent_class_name))
    agent_class = importlib.import_module(f'agent.{agent_class_name.split()}')
    print(agent_class)

# def str_to_class(classname):
#     return getattr(sys.modules['agent'], classname)

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    run()