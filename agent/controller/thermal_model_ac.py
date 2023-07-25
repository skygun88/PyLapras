import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import pickle
import gzip
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentConfig:
    # Learning
    gamma = 0.99 # It can be
    # gamma = 0.5 # It can be
    update_freq = 1
    k_epoch = 3
    learning_rate = 0.01
    # learning_rate = 0.010
    lmbda = 0.95
    eps_clip = 0.2
    v_coef = 1
    entropy_coef = 0.01

    # Memory
    memory_size = 4000

class ACAgent(AgentConfig):
    def __init__(self, action_size=3, input_size=4):
        self.action_size = action_size  # Still/UP/DOWN
        self.policy_network = MlpPolicy(action_size=self.action_size, input_size=input_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch, gamma=0.999)
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'terminal': [], 'count': 0,
            'user_acted': [], # to indicate action done by user
            'user_context': [], # to indicate user context when system only depends ambient sensor
        }

    def get_action(self, current_state, deterministic=False):
        with torch.no_grad():
            prob_a = self.policy_network.pi(torch.FloatTensor(current_state).to(device))
            if deterministic:
                action = torch.argmax(prob_a).item()

            else:
                action = torch.distributions.Categorical(prob_a).sample().item()
        return action, prob_a


    def update_network(self):
        memory_size = self.memory['count']
        state = torch.Tensor(np.array(self.memory['state'])).to(device=device)
        one_hot_action = F.one_hot(torch.tensor(np.array(self.memory['action'])).to(device=device), num_classes=self.action_size).reshape(memory_size, self.action_size)
        reward = torch.Tensor(np.array(self.memory['reward'])).to(device=device)
        next_state = torch.Tensor(np.array(self.memory['next_state'])).to(device=device)
        terminal = torch.Tensor(np.array(self.memory['terminal'])).to(device=device)
        
        action_prob = torch.sum(self.policy_network.pi(state)*one_hot_action, 1)
        log_prob = torch.log(action_prob+1e-5)

        values = self.policy_network.v(state)
        next_values = self.policy_network.v(next_state)

        target = reward + terminal*self.gamma*(next_values.reshape(memory_size, 1))
        advantage = (target - values).detach()
        actor_loss = torch.sum(-log_prob*advantage)

        critic_loss = torch.sum(0.5 * torch.square(target.detach() - values))

        self.loss = 0.4 * actor_loss + critic_loss



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
    
    def save_memory(self, path:str):
        with gzip.open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, path:str):
        with gzip.open(path,'rb') as f:
            self.memory = pickle.load(f)


class MlpPolicy(nn.Module):
    def __init__(self, action_size=3, input_size=4):
        super(MlpPolicy, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.hidden_size = 3 # Original - 24
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_pi = nn.Linear(self.hidden_size, self.action_size)
        self.fc3_v = nn.Linear(self.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def pi(self, x):
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3_pi(x)
        
        return self.softmax(x)

    def v(self, x):
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3_v(x)
        return x