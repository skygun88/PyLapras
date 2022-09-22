import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import pickle
import gzip

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

class PPOAgent(AgentConfig):
    def __init__(self, action_size=3, input_size=4):
        self.action_size = action_size  # Still/UP/DOWN
        self.policy_network = MlpPolicy(action_size=self.action_size, input_size=input_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch, gamma=0.999)
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': [], 'count': 0,
            'advantage': [], 'td_target': torch.FloatTensor([]),
            'user_acted': [], # to indicate action done by user
            'user_context': [], # to indicate user context when system only depends ambient sensor
        }


    def new_random_game(self):
        self.env.reset()
        action = self.env.action_space.sample()
        screen, reward, terminal, info = self.env.step(action)
        return screen, reward, action, terminal

    def get_action(self, current_state, deterministic=False):
        with torch.no_grad():
            prob_a = self.policy_network.pi(torch.FloatTensor(current_state).to(device))
            if deterministic:
                action = torch.argmax(prob_a).item()

                # sorted = torch.argsort(prob_a)
                # certification = (prob_a[sorted[-1]] /  prob_a[sorted[-2]]) 
                # # print(certification)
                # if certification < 1.5:
                #     action = 0

            else:
                action = torch.distributions.Categorical(prob_a).sample().item()
        return action, prob_a


    def update_network(self):
        # get ratio
        pi = self.policy_network.pi(torch.FloatTensor(self.memory['state']).to(device))
        new_probs_a = torch.gather(pi, 1, torch.tensor(self.memory['action']).to(device))
        old_probs_a = torch.FloatTensor(self.memory['action_prob']).to(device)
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))

        # surrogate loss
        surr1 = ratio * torch.FloatTensor(self.memory['advantage']).to(device)
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * torch.FloatTensor(self.memory['advantage']).to(device)
        pred_v = self.policy_network.v(torch.FloatTensor(self.memory['state']).to(device))
        v_loss = 0.5 * (pred_v - self.memory['td_target']).pow(2)  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy]).to(device)
        self.loss = (-torch.min(surr1, surr2).to(device) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def add_memory(self, s, a, r, next_s, t, prob, user_acted, user_context):
        if self.memory['count'] < self.memory_size:
            self.memory['count'] += 1
        else:
            self.memory['state'] = self.memory['state'][1:]
            self.memory['action'] = self.memory['action'][1:]
            self.memory['reward'] = self.memory['reward'][1:]
            self.memory['next_state'] = self.memory['next_state'][1:]
            self.memory['terminal'] = self.memory['terminal'][1:]
            self.memory['action_prob'] = self.memory['action_prob'][1:]
            self.memory['advantage'] = self.memory['advantage'][1:]
            self.memory['td_target'] = self.memory['td_target'][1:]
            self.memory['user_acted'] = self.memory['user_acted'][1:]
            self.memory['user_context'] = self.memory['user_context'][1:]

        self.memory['state'].append(s)
        self.memory['action'].append([a])
        self.memory['reward'].append([r])
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append([1 - t])
        self.memory['action_prob'].append(prob)
        self.memory['user_acted'].append(user_acted)
        self.memory['user_context'].append(user_context)

    def td_update_network(self, length):
        ''' memory in this episode'''
        temp_memory = self.temp_curr_path(length)

        ''' memory before this episode '''
        old_state = self.memory['state'][:-length]
        old_action = self.memory['action'][:-length]
        old_action_prob = self.memory['action_prob'][:-length]
        old_td_target = self.memory['td_target'][:-length]
        old_advantage = self.memory['advantage'][:-length]

        state = old_state + temp_memory['state']
        action = old_action + temp_memory['action']
        action_prob = old_action_prob + temp_memory['action_prob']
        td_target = torch.cat((old_td_target.to(device), temp_memory['td_target'].to(device)), dim=0)
        advantage = old_advantage + temp_memory['advantage']


        # get ratio
        pi = self.policy_network.pi(torch.FloatTensor(state).to(device))
        new_probs_a = torch.gather(pi, 1, torch.tensor(action).to(device))
        old_probs_a = torch.FloatTensor(action_prob).to(device)
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))

        # surrogate loss
        surr1 = ratio * torch.FloatTensor(advantage).to(device)
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * torch.FloatTensor(advantage).to(device)
        pred_v = self.policy_network.v(torch.FloatTensor(state).to(device))
        v_loss = 0.5 * (pred_v - td_target).pow(2)  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy]).to(device)
        self.loss = (-torch.min(surr1, surr2).to(device) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def temp_curr_path(self, length):
        temp_memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': [], 'count': 0,
            'advantage': [], 'td_target': torch.FloatTensor([]),
            'user_acted': [], # to indicate action done by user
            'user_context': [], # to indicate user context when system only depends ambient sensor
        }
        state = self.memory['state'][-length:]
        action = self.memory['action'][-length:]
        reward = self.memory['reward'][-length:]
        next_state = self.memory['next_state'][-length:]
        terminal = self.memory['terminal'][-length:]
        action_prob = self.memory['action_prob'][-length:]

        td_target = torch.FloatTensor(reward).to(device) + \
                    self.gamma * self.policy_network.v(torch.FloatTensor(next_state).to(device)) * torch.FloatTensor(terminal).to(device)
        delta = td_target - self.policy_network.v(torch.FloatTensor(state).to(device))
        delta = delta.detach().cpu().numpy()

        # get advantage
        advantages = []
        adv = 0.0

        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        temp_memory['state'] = state
        temp_memory['action'] = action
        temp_memory['reward'] = reward
        temp_memory['next_state'] = next_state
        temp_memory['terminal'] = terminal
        temp_memory['action_prob'] = action_prob
        temp_memory['td_target'] = td_target.data
        temp_memory['advantage'] = advantages

        return temp_memory

    

    def finish_path(self, length):
        state = self.memory['state'][-length:]
        reward = self.memory['reward'][-length:]
        next_state = self.memory['next_state'][-length:]
        self.memory['terminal'][-1] = [0] # Mark terminal of the episode
        terminal = self.memory['terminal'][-length:]

        td_target = torch.FloatTensor(reward).to(device) + \
                    self.gamma * self.policy_network.v(torch.FloatTensor(next_state).to(device)) * torch.FloatTensor(terminal).to(device)
        delta = td_target - self.policy_network.v(torch.FloatTensor(state).to(device))
        delta = delta.detach().cpu().numpy()

        # get advantage
        advantages = []
        adv = 0.0
        

        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        if self.memory['td_target'].shape == torch.Size([1, 0]):
            self.memory['td_target'] = td_target.data
        else:
            self.memory['td_target'] = torch.cat((self.memory['td_target'].to(device), td_target.data.to(device)), dim=0)
        self.memory['advantage'] += advantages

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