from importlib.resources import path
from unicodedata import name
import gym
import os, sys
sys.path.append('/home/zp/deeplearning/spinningup_project')
import torch
import numpy as np

# 全局函数，把字典形式的obs转换为array的state
def obs2state(obs, key_list=['observation', 'desired_goal']):
    '''
    把原生obs，也就是dict{'observation':obs, 'achieved_goal':ag, 'desired_goal':g}
    转换成 array(cat(obs, g))
    把除了 'achieved_goal' 的其他键的值cat起来
    返回 [obs_dim+gaol_dim, ] 形状的array
    '''
    s = np.concatenate(([obs[key] for key in key_list]))
    return s

def load_normer_ac(path):
    normer, ac  = torch.load(path, map_location=torch.device('cpu'))
    return normer, ac 

def test(normer, ac, env):
    ep_reward_list = []
    test_success_all = []
    for j in range(20):
        obs = env.reset()
        ep_reward = 0
        success = []
        for i in range(50):

            s = obs2state(obs)
            s = normer.normalize(s)  # 必须要归一化！因为训练时候输入都是归一化的，测试时候不归一化肯定不行！
            s = torch.as_tensor(s, dtype=torch.float32)
            a = ac.act(s)
            
            env.render()

            obs, r, done, info = env.step(a)   # 直接更新obs
            success.append(info['is_success'])
            
            ep_reward += r
        ep_reward_list.append(ep_reward)

        testepret = ep_reward
        testsuccess = success[-1]
        test_success_all.append(testsuccess)
        print('TestEpret=', testepret)
        print('Testsuccess=',testsuccess)

    test_success_all = np.array(test_success_all)
    
    print(test_success_all.mean())

   
if __name__ == "__main__":
    env = gym.make('FetchPush-v1')   # 这个环境必须要和train的环境对应哈
    normer, ac = load_normer_ac('data_Goalenv/2022-03-22_Goalenv_exp_DDPGHER_FetchPush-v1/2022-03-22_15-07-22-Goalenv_exp_DDPGHER_FetchPush-v1_s123/ac_norm_model.pth')
    test(normer, ac, env)