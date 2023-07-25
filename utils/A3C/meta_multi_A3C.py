import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
torch.set_printoptions(precision=3)

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self, s_dim, a_dim, isinvDM=False, isMgCl=False, isMeta=True, isImg=True, isVec=True):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.isinvDM = isinvDM
        self.isMgCl = isMgCl
        self.isMeta = isMeta
        self.isImg = isImg
        self.isVec = isVec

        self.epsilon = 0.01
        self.start_entropy = 0.1
        self.end_entropy = 0.01
        self.entropy_beta = self.start_entropy
        self.meta_entropy = 0.01
        self.entropy_converge = 0.999

        img_in_dimension = self.s_dim[0][3]
        img_in_channel = self.s_dim[0][1]
        n_human_state = 4
        vec_in = self.s_dim[1]
        vec_fc_o = 128
        c1, k1, s1 = 16, 8, 4
        c2, k2, s2 = 32, 4, 2
        fc_o = 128
        concat1_o = vec_fc_o+fc_o
        lstm_o = 256
        policy_o = self.a_dim
        value_o = 1

        meta_in_dimension = n_human_state + (policy_o+value_o)*n_human_state 
        meta_fc1_o = 96
        meta_fc2_o = 128
        meta_out_o = 4 
        meta_value_o = 1

        img_x_dimension = img_in_dimension
        img_x_dimension = (img_x_dimension-k1)//s1 + 1
        img_x_dimension = (img_x_dimension-k2)//s2 + 1

        dist_o = 8 # ~1.0, 1.0~1.5, 1.5~2.0, 2.0~2.5, 2.5~3.0, 3.0~3.5, 3.5~4.0, 4.0~
        angle_0 = 360//15 # 1 per 360

        if self.isImg and self.isVec:
            self.conv1 = nn.Conv2d(in_channels=img_in_channel, out_channels=c1, kernel_size=k1, stride=s1)        
            self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k2, stride=s2)
            self.fc = nn.Linear(img_x_dimension*img_x_dimension*c2, fc_o)

            self.vec_fc1 = nn.Linear(vec_in, vec_fc_o)
            self.vec_fc2 = nn.Linear(vec_fc_o, vec_fc_o)

            ''' invDM - predict distance & angle '''
            if self.isinvDM:
                self.invDM_fc = nn.Linear(vec_fc_o, vec_fc_o)
                self.invDM_dist = nn.Linear(vec_fc_o, dist_o)
                self.invDM_angle = nn.Linear(vec_fc_o, angle_0)

            self.img_feature_extractor = nn.Sequential(
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.ReLU(),
                nn.Flatten(),
                self.fc,
                nn.ReLU()
            )

            self.vec_feature_extractor = nn.Sequential(
                self.vec_fc1,
                nn.ReLU(),
                self.vec_fc2,
                nn.ReLU()
            )
            self.forward = self.all_forward

        elif self.isImg:
            fc_o = 32
            concat1_o = fc_o
            self.entropy_converge = 0.9999
            self.end_entropy = 0.03
            # fc_o = concat1_o
            self.conv1 = nn.Conv2d(in_channels=img_in_channel, out_channels=c1, kernel_size=k1, stride=s1)        
            self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k2, stride=s2)
            self.fc = nn.Linear(img_x_dimension*img_x_dimension*c2, fc_o)

            self.img_feature_extractor = nn.Sequential(
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.ReLU(),
                nn.Flatten(),
                self.fc,
                nn.ReLU()
            )
            self.forward = self.img_forward

        elif self.isVec:
            concat1_o = vec_fc_o
            self.vec_fc1 = nn.Linear(vec_in, vec_fc_o)
            self.vec_fc2 = nn.Linear(vec_fc_o, vec_fc_o)

            self.vec_feature_extractor = nn.Sequential(
                self.vec_fc1,
                nn.ReLU(),
                self.vec_fc2,
                nn.ReLU()
            )
            self.forward = self.vec_forward

        else:
            print('error - wrong model input')
            sys.exit()

        self.lstm = nn.LSTMCell(concat1_o, lstm_o)

        ''' Worker '''
        self.p_har = nn.Linear(lstm_o, policy_o)
        self.v_har = nn.Linear(lstm_o, value_o)

        self.p_clo = nn.Linear(lstm_o, policy_o)
        self.v_clo = nn.Linear(lstm_o, value_o)

        self.p_age = nn.Linear(lstm_o, policy_o)
        self.v_age = nn.Linear(lstm_o, value_o)

        self.p_gen = nn.Linear(lstm_o, policy_o)
        self.v_gen = nn.Linear(lstm_o, value_o)

        ''' Meta-controller '''
        self.meta_fc1 = nn.Linear(concat1_o, meta_fc1_o)
        self.meta_fc2 = nn.Linear(meta_fc1_o+meta_in_dimension, meta_fc2_o)
        self.meta_p = nn.Linear(meta_fc1_o+meta_in_dimension, meta_out_o)
        self.meta_v = nn.Linear(meta_fc1_o+meta_in_dimension, meta_value_o)

        self.apply(weights_init)

        self.policy_value_int(self.p_har, self.v_har)
        self.policy_value_int(self.p_clo, self.v_clo)
        self.policy_value_int(self.p_age, self.v_age)
        self.policy_value_int(self.p_gen, self.v_gen)
        self.policy_value_int(self.meta_p, self.meta_v)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
        self.distribution = torch.distributions.Categorical
        self.train()

        

    def policy_value_int(self, pnet, vnet):
        pnet.weight.data = normalized_columns_initializer(
            pnet.weight.data, 0.01)
        pnet.bias.data.fill_(0)
        vnet.weight.data = normalized_columns_initializer(
            vnet.weight.data, 1.0)
        vnet.bias.data.fill_(0)

    def entropy_update(self):
        self.entropy_beta = max(self.entropy_beta*self.entropy_converge, self.end_entropy)

    def img_forward(self, s_i, s_v, hs, cs, pg):
        f_si = self.img_feature_extractor(s_i)
        h_s, c_s = self.lstm(f_si, (hs, cs))

        ''' Robot controller pathway '''
        har_logits = self.p_har(h_s)
        har_value = self.v_har(h_s)
        har_result = har_logits, har_value

        clo_logits = self.p_clo(h_s)
        clo_value = self.v_clo(h_s)
        clo_result = clo_logits, clo_value

        age_logits = self.p_age(h_s)
        age_value = self.v_age(h_s)
        age_result = age_logits, age_value

        gen_logits = self.p_gen(h_s)
        gen_value = self.v_gen(h_s)
        gen_result = gen_logits, gen_value

        ''' Meta controller pathway '''
        meta_h1 = self.meta_fc1(f_si)
        meta_concat = torch.cat([pg, har_logits, har_value, clo_logits, clo_value, age_logits, age_value, gen_logits, gen_value, meta_h1], dim=1)
        meta_concat = torch.relu(meta_concat)
        meta_h2 = self.meta_fc2(meta_concat)
        
        meta_logits = self.meta_p(meta_h2)
        meta_value = self.meta_v(meta_h2)
        meta_result = meta_logits, meta_value

        return har_result, clo_result, age_result, gen_result, h_s, c_s, meta_result

    def vec_forward(self, s_i, s_v, hs, cs, pg):
        f_vi = self.vec_feature_extractor(s_v)
        h_s, c_s = self.lstm(f_vi, (hs, cs))

        ''' Robot controller pathway '''
        har_logits = self.p_har(h_s)
        har_value = self.v_har(h_s)
        har_result = har_logits, har_value

        clo_logits = self.p_clo(h_s)
        clo_value = self.v_clo(h_s)
        clo_result = clo_logits, clo_value

        age_logits = self.p_age(h_s)
        age_value = self.v_age(h_s)
        age_result = age_logits, age_value

        gen_logits = self.p_gen(h_s)
        gen_value = self.v_gen(h_s)
        gen_result = gen_logits, gen_value

        ''' Meta controller pathway '''
        meta_h1 = self.meta_fc1(f_vi)
        meta_concat = torch.cat([pg, har_logits, har_value, clo_logits, clo_value, age_logits, age_value, gen_logits, gen_value, meta_h1], dim=1)
        meta_concat = torch.relu(meta_concat)
        meta_h2 = self.meta_fc2(meta_concat)
        
        meta_logits = self.meta_p(meta_h2)
        meta_value = self.meta_v(meta_h2)
        meta_result = meta_logits, meta_value

        return har_result, clo_result, age_result, gen_result, h_s, c_s, meta_result


    def all_forward(self, s_i, s_v, hs, cs, pg):
        f_si = self.img_feature_extractor(s_i)
        f_vi = self.vec_feature_extractor(s_v)
        concat1 = torch.cat([f_vi, f_si], dim=1)
        h_s, c_s = self.lstm(concat1, (hs, cs))

        ''' Robot controller pathway '''
        har_logits = self.p_har(h_s)
        har_value = self.v_har(h_s)
        har_result = har_logits, har_value

        clo_logits = self.p_clo(h_s)
        clo_value = self.v_clo(h_s)
        clo_result = clo_logits, clo_value

        age_logits = self.p_age(h_s)
        age_value = self.v_age(h_s)
        age_result = age_logits, age_value

        gen_logits = self.p_gen(h_s)
        gen_value = self.v_gen(h_s)
        gen_result = gen_logits, gen_value

        ''' Meta controller pathway '''
        meta_h1 = self.meta_fc1(concat1)
        # print(pg.shape, har_logits.shape, har_value.shape, meta_h1.shape)
        meta_concat = torch.cat([pg, har_logits, har_value, clo_logits, clo_value, age_logits, age_value, gen_logits, gen_value, meta_h1], dim=1)
        meta_concat = torch.relu(meta_concat)
        meta_h2 = self.meta_fc2(meta_concat)
        
        meta_logits = self.meta_p(meta_h2)
        meta_value = self.meta_v(meta_h2)
        meta_result = meta_logits, meta_value

        return har_result, clo_result, age_result, gen_result, h_s, c_s, meta_result

    
    def choose_action_with_goal(self, s_i, s_v, hs, cs, g, pg, mask=None):
        self.eval()
        forward_results = self.forward(s_i, s_v, hs, cs, pg)
        h_s, c_s = forward_results[4], forward_results[5]
        logits, _ = forward_results[g]
        prob = F.softmax(logits, dim=1).data

        print(prob)
        if mask != None:
            prob = prob*mask
        m = self.distribution(prob)

        if random.random() < self.epsilon:
            action = np.int64(random.randint(0, self.a_dim-1))
        else:            
            # action = m.sample().numpy()[0] if prob.device == 'cpu' else m.sample().cpu().numpy()[0]
            print(torch.argmax(prob).numpy())
            action = torch.argmax(prob).numpy().item() if prob.device == 'cpu' else torch.argmax(prob).cpu().numpy().item()
            action = int(action)

        return action, h_s, c_s
    

    def choose_action(self, s_i, s_v, hs, cs, pg, mask=None):
        self.eval()
        forward_results = self.forward(s_i, s_v, hs, cs, pg)
        h_s, c_s = forward_results[4], forward_results[5]
        meta_logits, _ = forward_results[6]
        meta_prob = F.softmax(meta_logits, dim=1).data
        meta_prob = meta_prob*pg

        try:
            meta_m = self.distribution(meta_prob)
            g = meta_m.sample().numpy()[0] if meta_prob.device == 'cpu' else meta_m.sample().cpu().numpy()[0]
        except:
            meta_m = self.distribution(pg)
            g = meta_m.sample().numpy()[0] if meta_prob.device == 'cpu' else meta_m.sample().cpu().numpy()[0]


        logits, _ = forward_results[g]
        prob = F.softmax(logits, dim=1).data
        print(prob)
        if mask != None:
            prob = prob*mask
        m = self.distribution(prob)

        if random.random() < self.epsilon:
            action = np.int64(random.randint(0, self.a_dim-1))
        else:            
            # action = m.sample().numpy()[0] if prob.device == 'cpu' else m.sample().cpu().numpy()[0]
            # print(torch.argmax(prob))
            print(torch.argmax(prob).numpy())
            # print(torch.argmax(prob).numpy()[0])
            action = torch.argmax(prob).numpy().item() if prob.device == 'cpu' else torch.argmax(prob).cpu().numpy().item()
            action = int(action)
        return action, h_s, c_s, g


    def loss_func(self, s_i, a, v_t, s_v, hs, cs, g, pg):
        self.train()
        forward_results = self.forward(s_i, s_v, hs, cs, pg)
        logits, values = forward_results[g]
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        log_probs = m.log_prob(a) 
        entropy = m.entropy()
        exp_v = log_probs * td.detach().squeeze()
        a_loss = -exp_v
        a_loss -= self.entropy_beta*entropy
        total_loss = (0.5*c_loss + a_loss).sum()
        
        
        
        self.entropy_update()
        # if self.isinvDM:
        #     total_loss += self.invDM(s, a)
        return total_loss

    def meta_loss_func(self, s_i, a, v_t, s_v, hs, cs, pg):
        self.train()
        
        f_si = self.img_feature_extractor(s_i).detach()
        f_vi = self.vec_feature_extractor(s_v).detach()
        concat1 = torch.cat([f_vi, f_si], dim=1)
        h_s, c_s = self.lstm(concat1, (hs, cs))
        h_s = h_s.detach()

        ''' Robot controller pathway '''
        har_logits = self.p_har(h_s).detach()
        har_value = self.v_har(h_s).detach()

        clo_logits = self.p_clo(h_s).detach()
        clo_value = self.v_clo(h_s).detach()

        age_logits = self.p_age(h_s).detach()
        age_value = self.v_age(h_s).detach()

        gen_logits = self.p_gen(h_s).detach()
        gen_value = self.v_gen(h_s).detach()

        ''' Meta controller pathway '''
        meta_h1 = self.meta_fc1(concat1)
        # print(pg.shape, har_logits.shape, har_value.shape, meta_h1.shape)
        meta_concat = torch.cat([pg, har_logits, har_value, clo_logits, clo_value, age_logits, age_value, gen_logits, gen_value, meta_h1], dim=1)
        # meta_concat = torch.relu(meta_concat)
        meta_concat = torch.tanh(meta_concat)
        meta_h2 = self.meta_fc2(meta_concat)
        
        meta_logits = self.meta_p(meta_h2)
        meta_values = self.meta_v(meta_h2)
        
        # meta_logits = meta_logits*pg
        meta_prob = F.softmax(meta_logits, dim=1)
        meta_m = self.distribution(meta_prob)


        td = v_t - meta_values
        c_loss = td.pow(2)
        
        log_probs = meta_m.log_prob(a) 
        # pg_prob = F.softmax(pg, dim=1)

        entropy = meta_m.entropy()
        # similar_loss = F.cross_entropy(meta_prob, pg_prob)
        exp_v = log_probs * td.detach().squeeze()
        a_loss = -exp_v
        a_loss -= self.meta_entropy*entropy
        # total_loss = (0.5*c_loss + a_loss + 0.01*similar_loss).sum()
        total_loss = (0.5*c_loss + a_loss).sum()
        # total_loss = (c_loss + a_loss).sum()
        
        return total_loss
    
    def invDM_foward(self, s_i):
        f_si = self.img_feature_extractor(s_i)

        h_i = self.invDM_fc(f_si)
        h_i = F.relu(h_i)

        dist_logits = self.invDM_dist(h_i)
        angle_logits = self.invDM_angle(h_i)

        return dist_logits, angle_logits

    
    def invDM(self, s_i, dist, theta):
        dist_logits, angle_logits = self.invDM_foward(s_i)
        invDM_loss1 = F.cross_entropy(dist_logits, dist)
        invDM_loss2 = F.cross_entropy(angle_logits, theta)

        invDM_loss = invDM_loss1 + invDM_loss2

        return invDM_loss