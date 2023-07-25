import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import pickle
import gzip
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentConfig:
    # Learning
    gamma = 0.99 # It can be
    # gamma = 0.5 # It can be
    update_freq = 1
    k_epoch = 3
    learning_rate = 0.02
    # learning_rate = 0.010
    lmbda = 0.95
    eps_clip = 0.2
    v_coef = 1
    entropy_coef = 0.01

    # Memory
    memory_size = 4000
    min_memory = 8
    batch_size = 8

class DQNAgent(AgentConfig):
    def __init__(self, action_size=3, input_size=4):
        self.action_size = action_size  # Still/UP/DOWN
        self.policy_network = DQN(action_size=self.action_size, input_size=input_size).to(device)
        self.target_policy_network = DQN(action_size=self.action_size, input_size=input_size).to(device)
        self.target_policy_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch, gamma=0.999)
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'terminal': [], 'count': 0,
            'user_acted': [], # to indicate action done by user
            'user_context': [], # to indicate user context when system only depends ambient sensor
        }



    def get_action(self, current_state):
        with torch.no_grad():
            q = self.policy_network.forward(torch.FloatTensor(current_state).to(device))

            action = torch.argmax(q).item()
 
        return action, q


    def update_network(self):
        if self.memory['count'] > self.min_memory:          
            sampled_indexes = np.array(random.sample(range(self.memory['count']), self.batch_size))

            mini_state = torch.Tensor(np.array(self.memory['state'])[sampled_indexes]).to(device=device)
            mini_one_hot_action = F.one_hot(torch.tensor(np.array(self.memory['action'])[sampled_indexes]).to(device=device), num_classes=self.action_size).reshape(self.batch_size, self.action_size)
            mini_reward = torch.Tensor(np.array(self.memory['reward'])[sampled_indexes]).to(device=device)
            mini_next_state = torch.Tensor(np.array(self.memory['next_state'])[sampled_indexes]).to(device=device)
            mini_terminal = torch.Tensor(np.array(self.memory['terminal'])[sampled_indexes]).to(device=device)
            
            q_values = torch.sum(self.policy_network.forward(mini_state)*mini_one_hot_action, 1)
            ys = mini_reward + mini_terminal*self.gamma*(torch.amax(self.target_policy_network.forward(mini_next_state).detach(), 1).reshape(self.batch_size, 1))

            criterion = nn.SmoothL1Loss()


            self.loss = criterion(q_values, ys.reshape(self.batch_size))
            # print(q_values.shape, ys.shape, mini_reward.shape)

            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def add_memory(self, s, a, r, next_s, t, user_acted, user_context):
        if self.memory['count'] < self.memory_size:
            self.memory['count'] += 1
        else:
            self.memory['state'] = self.memory['state'][1:]
            self.memory['action'] = self.memory['action'][1:]
            self.memory['reward'] = self.memory['reward'][1:]
            self.memory['next_state'] = self.memory['next_state'][1:]
            self.memory['terminal'] = self.memory['terminal'][1:]
            self.memory['user_acted'] = self.memory['user_acted'][1:]
            self.memory['user_context'] = self.memory['user_context'][1:]

        self.memory['state'].append(s)
        self.memory['action'].append([a])
        self.memory['reward'].append([r])
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append([1 - t])
        self.memory['user_acted'].append(user_acted)
        self.memory['user_context'].append(user_context)


    

    def finish_path(self, length):
        self.memory['terminal'][-1] = [0] # Mark terminal of the episode


    def save_model(self, path:str):
        torch.save(self.policy_network.state_dict(), path)

    def load_model(self, path:str):
        self.policy_network.load_state_dict(torch.load(path))
    
    def update_target(self):
        self.target_policy_network.load_state_dict(self.policy_network.state_dict())
    
    def save_memory(self, path:str):
        with gzip.open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, path:str):
        with gzip.open(path,'rb') as f:
            self.memory = pickle.load(f)


class DQN(nn.Module):
    def __init__(self, action_size=3, input_size=4):
        super(DQN, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.hidden_size = 3 # Original - 24
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_pi = nn.Linear(self.hidden_size, self.action_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.fc3_pi(x)
        