import sys
sys.path.append('/home/zp/deeplearning/spinningup_project')
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam

import spinup.goal_env_alogos.DDPG_HER.core as core
from spinup.goal_env_alogos.Offpolicy.baseoffpolicy import OffPolicy

class DDPGHER(OffPolicy):
    # 直接输入 __init__自动继承！之前在offPolicy所有定义好的self.xxx都不需要再写了
    # 继承就是覆写，之前的Offpolicy里面self的量是根据之前的类里的init传进去的，现在这些self量都会根据 这个 DDPGHER的init传进去！
    def __init__(
                 self, 
                 act_dim, obs_dim, a_bound,            # 在goal环境里，obs_dim 是 observation_dim + goal_dim
                 actor_critic=core.MLPActorCritic, 
                 ac_kwargs=dict(), 
                 seed=0, replay_size=int(1e6), 
                 gamma=0.9, polyak=0.99, pi_lr=0.001, q_lr=0.001, 
                 batch_size=256, 
                 act_noise=0.1, target_noise=0.2,
                 noise_clip=0.5, policy_delay=2, 
                 goal_selection_strategy='future', n_sampled_goal=4, 
                 action_l2=0.0, clip_return=None, 
                 device=None
                 ):
        super(DDPGHER, self).__init__(act_dim, obs_dim, a_bound, 
                                      actor_critic=core.MLPActorCritic, 
                                      ac_kwargs=ac_kwargs, 
                                      seed=seed, replay_size=replay_size, 
                                      gamma=gamma, polyak=polyak, pi_lr=pi_lr, q_lr=q_lr, 
                                      batch_size=batch_size, 
                                      act_noise=act_noise, target_noise=target_noise, 
                                      noise_clip=noise_clip, policy_delay=policy_delay, 
                                      goal_selection_strategy=goal_selection_strategy, n_sampled_goal=n_sampled_goal, 
                                      action_l2=action_l2, clip_return=clip_return, 
                                      device=device
                                      )
        # 建立 ddpg 的ac
        self.ac = actor_critic(obs_dim=self.obs_dim, act_dim=self.act_dim, act_bound=self.a_bound).to(self.device)  # 现在这个device取决于创建DDPGHER实例时传进去的设置
        self.ac_targ = deepcopy(self.ac).to(self.device)

        # 冻结所有的目标参数
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        # 显示有多少需要训练的数，这里不用logger！直接print
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q])
        print('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

        # 设置优化器
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

    # 定义计算损失的函数
    def compute_loss_pi(self, data):
        '''
        data = {'obs'=[batch_size, observation_dim + goal_dim],
                'obs2'=[batch_size, observation_dim + goal_dim],
                'act'=[batch_size, act_dim],
                'rew'=[batch_size,],
                'done'=[batch_size,]}
        '''
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        
        return -q_pi.mean()
    
    def compute_loss_q(self, data):
        '''
        data = {'obs'=[batch_size, observation_dim + goal_dim],
                'obs2'=[batch_size, observation_dim + goal_dim],
                'act'=[batch_size, act_dim],
                'rew'=[batch_size,],
                'done'=[batch_size,]}
        '''
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.ac.q(o, a)
        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)
            # Target policy smoothing 相比原本的ddpg，加了这一步！ target_noise是为了平滑输出
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.a_bound, self.a_bound)
            # Target Q-values
            q_pi_targ = self.ac_targ.q(o2, a2)

            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()
        loss_info = dict(QVals=q.cpu().detach().numpy())   # 形状 [N,]
        return loss_q, loss_info
    
    
    def update(self, batch_size=100):  
        '''
        更新步骤, 直接在update步骤里面用buffer进行sample得到data
        data = {'obs'=[batch_size, observation_dim + goal_dim],
                'obs2'=[batch_size, observation_dim + goal_dim],
                'act'=[batch_size, act_dim],
                'rew'=[batch_size,],
                'done'=[batch_size,]}
        '''
        data = self.buffer.sample_batch(batch_size)

        # 先对ac.q网络进行1步优化
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # 在原始的DDPG上进行了改进，用了延迟更新pi的方法
        if self.learn_step % self.policy_delay == 0:
            # 冻结ac.q网络，接下来更新策略pi的时候不要更改ac.q
            for p in self.ac.q.parameters():
                p.requires_grad = False
        
            # 接下来对ac.pi进行优化
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # 优化完 ac.pi，解冻ac.q
            for p in self.ac.q.parameters():
                p.requires_grad = True
            
            # 最后， 更新target的两个网络参数，软更新 用了自乘操作，节省内存
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
        self.learn_step += 1  # 每update一次，learn_step+1

        return loss_q, loss_info['QVals']


