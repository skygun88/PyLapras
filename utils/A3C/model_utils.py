"""
Functions that use multiple times
"""
import os
import sys
import cv2
import numpy as np
import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from utils.configure import *

# from Utils.constant import *

# def img_preprocess(img):
#     img_copy = img
#     ori_h, ori_w, _ = img_copy.shape
#     img_copy = img_copy[:, (ori_w//2)-(ori_h//2):(ori_w//2)+(ori_h//2), :]
#     img_copy = cv2.resize(img_copy, dsize=(84, 84), interpolation=cv2.INTER_AREA)/255
#     img_copy = np.transpose(img_copy, (2, 0 ,1))
#     return img_copy

# def img_preprocess(img):
#     img_copy = img
#     img_copy = np.transpose(img_copy, (2, 0 ,1))/255
#     return img_copy

def img_preprocess(img, gray=False):
    img_copy = img.copy()
    ori_h, ori_w, _ = img_copy.shape
    img_copy = img_copy[:, (ori_w//2)-(ori_h//2):(ori_w//2)+(ori_h//2), :]
    img_copy = cv2.resize(img_copy, dsize=(84, 84), interpolation=cv2.INTER_AREA)
    if gray:
        img_copy = np.expand_dims(cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY), axis=0)/255
    else:
        img_copy = np.transpose(img_copy, (2, 0 ,1))/255
    
    return img_copy

# def img_preprocess(img, gray=False):
#     img_copy = img
#     if gray:
#         img_copy = np.expand_dims(cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY), axis=0)/255
#     else:
#         img_copy = np.transpose(img_copy, (2, 0 ,1))/255
#     return img_copy

def clo_preprocess(vec_inpur, cr_conf):
    np_vector = np.concatenate([vec_inpur, cr_conf])
    return np_vector

def age_preprocess(vec_inpur, ag_conf):
    np_vector = np.concatenate([vec_inpur, ag_conf])
    return np_vector

def har_preprocess(vec_inpur, ar_conf):
    np_vector = np.concatenate([vec_inpur, np.array([ar_conf])])
    return np_vector

def gen_preprocess(vec_inpur, gd_conf):
    np_vector = np.concatenate([vec_inpur, np.array([gd_conf])])
    return np_vector


def preprocess(state, input_config, gray=False):
    ar_pos, ar_conf, cr_conf, ag_conf, gd_conf, full_img = state[:6]
    confs = [ar_conf, cr_conf, ag_conf, gd_conf]
    img_input = img_preprocess(full_img, gray=gray)
    vec_preprocesses = [har_preprocess, clo_preprocess, age_preprocess, gen_preprocess]
    vec_input = ar_pos
    for goal in input_config:
        vec_input = vec_preprocesses[goal](vec_input, confs[goal])
    return img_input, vec_input

def dist_preprocess(dist_list):
    dist_classes = [int(min(max((x//0.5)-1, 0.0), 7.0)) for x in dist_list]
    return np.array(dist_classes)

def angle_preprocess(angle_list):
    angle360s = [min(x, 360) if x > 0 else min(x + 360, 360) for x in angle_list]
    angle_classes = [int(min(x//15, (360//15)-1)) for x in angle360s]
    return np.array(angle_classes)

def v_wrap(np_array, dtype=np.float32, device='cpu'):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array).to(device)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

def push_and_pull_with_meta(env, trajectory, opt, lnet, gnet, done, s_i_, s_v_, pg_, bsi, bsv, ba, br, h_s, c_s, bhs, bcs, bpg, gamma, device='cpu'):
    v_s_ = 0. # terminal
    if not done:
        v_s_ = lnet.forward(
            v_wrap(s_i_[None, :], device=device), 
            v_wrap(s_v_[None, :], device=device), 
            h_s, 
            c_s, 
            v_wrap(pg_[None, :], device=device)
        )[6][-1].data
        v_s_ = v_s_.numpy()[0, 0] if device == 'cpu' else v_s_.cpu().numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.meta_loss_func(
        v_wrap(np.stack(bsi, axis=0), device=device),
        v_wrap(np.array(ba), dtype=np.int64, device=device) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba), device=device),
        v_wrap(np.array(buffer_v_target)[:, None], device=device),
        v_wrap(np.stack(bsv, axis=0), device=device),
        torch.cat(bhs, axis=0),
        torch.cat(bcs, axis=0),
        v_wrap(np.stack(bpg, axis=0), device=device),
    )



    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lnet.parameters(), 50)
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

def push_and_pull_multi(env, trajectory, opt, lnet, gnet, done, s_i_, s_v_, pg_, bsi, bsv, ba, br, h_s, c_s, bhs, bcs, g, bpg, gamma, device='cpu', input_config=(0,1,2,3)):
    v_s_ = 0. # terminal
    if not done:
        if device == 'cpu':
            # print('hi', pg_.shape)
            v_s_ = lnet.forward(
                v_wrap(s_i_[None, :], device=device), 
                v_wrap(s_v_[None, :], device=device), 
                h_s, 
                c_s,
                v_wrap(pg_[None, :], device=device)
            )[g][-1].data.numpy()[0, 0]
        else:
            v_s_ = lnet.forward(
                v_wrap(s_i_[None, :], device=device), 
                v_wrap(s_v_[None, :], device=device),
                h_s, 
                c_s,
                v_wrap(pg_[None, :], device=device)
            )[g][-1].data.cpu().numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bsi, axis=0), device=device),
        v_wrap(np.array(ba), dtype=np.int64, device=device) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba), device=device),
        v_wrap(np.array(buffer_v_target)[:, None], device=device),
        v_wrap(np.stack(bsv, axis=0), device=device),
        torch.cat(bhs, axis=0),
        torch.cat(bcs, axis=0),
        g,
        v_wrap(np.stack(bpg, axis=0), device=device),
    )

    if lnet.isinvDM:
        dist_ang = [env.calculate_distance_angle(rstate) for rstate in trajectory][:-1]
        dists = [x[0] for x in dist_ang]
        angs = [x[1] for x in dist_ang]
        

        dist_processed = dist_preprocess(dists)
        angle_processed = angle_preprocess(angs)
    
        invDM_loss = lnet.invDM(
            v_wrap(np.stack(bsi, axis=0), device=device),
            v_wrap(dist_processed, dtype=np.int64, device=device),
            v_wrap(angle_processed, dtype=np.int64, device=device)
        )
    
        # loss += 0.5*invDM_loss
        loss += invDM_loss

    # if done and lnet.isMgCl:
    if (not done) and lnet.isMgCl:
    # if lnet.isMgCl:
        other_goals = list(input_config)
        other_goals.pop(g)
        ba[-1] = np.array(DONE)
        for goal in other_goals:
            reward = env.reward_funcs[goal](DONE, env.curr_trues, env.curr_preds)
            if reward > 0:
                v_s_ = 0.
                buffer_v_target = []
                br[-1] = 1
                for r in br[::-1]:    # reverse buffer r
                    v_s_ = r + gamma * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()  
                
                mgcl_loss = lnet.loss_func(
                    v_wrap(np.stack(bsi, axis=0), device=device),
                    v_wrap(np.array(ba), dtype=np.int64, device=device) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba), device=device),
                    v_wrap(np.array(buffer_v_target)[:, None], device=device),
                    v_wrap(np.stack(bsv, axis=0), device=device),
                    torch.cat(bhs, axis=0),
                    torch.cat(bcs, axis=0),
                    goal,
                    v_wrap(np.stack(bpg, axis=0), device=device),
                )

                # loss += 0.5*mgcl_loss
                loss += mgcl_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lnet.parameters(), 50)
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

def push_and_pull(env, trajectory, opt, lnet, gnet, done, s_i_, s_v_, bsi, bsv, ba, br, h_s, c_s, bhs, bcs, gamma, device='cpu'):
    if done:
        v_s_ = 0.               # terminal
    else:
       
        if device == 'cpu':
            v_s_ = lnet.forward(
                v_wrap(s_i_[None, :], device=device), 
                v_wrap(s_v_[None, :], device=device), 
                h_s, 
                c_s
            )[-1].data.numpy()[0, 0]
        else:
            v_s_ = lnet.forward(
                v_wrap(s_i_[None, :], device=device), 
                v_wrap(s_v_[None, :], device=device),
                h_s, 
                c_s
            )[-1].data.cpu().numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bsi, axis=0), device=device),
        v_wrap(np.array(ba), dtype=np.int64, device=device) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba), device=device),
        v_wrap(np.array(buffer_v_target)[:, None], device=device),
        v_wrap(np.stack(bsv, axis=0), device=device),
        torch.cat(bhs, axis=0),
        torch.cat(bcs, axis=0),
    )







    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lnet.parameters(), 50)
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())



def record(global_ep, global_ep_r, ep_r, res_queue, name, writer=None):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.2f" % global_ep_r.value,
    )
    if writer != None:
        writer.add_scalar('global_ep_r', global_ep_r.value, global_ep.value)