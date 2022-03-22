import sys
sys.path.append('/home/zp/deeplearning/spinningup_project')
import numpy as np
import torch
import copy
import pickle
from spinup.goal_env_alogos.Offpolicy.normalizer import StateNorm
from spinup.goal_env_alogos.Offpolicy.memory import ReplayBuffer

class OffPolicy:
    '''
    所有离线RL算法在goal环境下的基础类，
    包含了处理goal环境的Her采样方法以及存储网络等功能
    self.ac还有优化器，还有learn函数需要在具体的算法中继承过来重写！
    '''
    def __init__(self,
                 act_dim, obs_dim, a_bound,
                 actor_critic=None,
                 ac_kwargs=dict(), seed=0,
                 replay_size=int(1e6),
                 gamma=0.99,
                 polyak=0.995,
                 pi_lr=1e-3,
                 q_lr=1e-3,
                 batch_size=256,
                 act_noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 goal_selection_strategy='future',
                 n_sampled_goal=4,
                 action_l2=1.0,
                 clip_return=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
                 ):
        # torch 设置
        torch.manual_seed(seed)
        self.device = device
        self.learn_step = 0       # 记录learn步数的计数器

        # 超参数
        self.obs_dim = obs_dim                  # 在goal环境里，obs_dim 是 observation_dim + goal_dim。 
                                                # 代码中所有的obs也都是指 cat(observation,desired_goal)
        self.act_dim = act_dim                  # 普通的act_dim
        self.a_bound = a_bound                  # act的高度，上下界，一个浮点数
        self.policy_delay = policy_delay
        self.action_noise = act_noise
        self.gamma = gamma
        self.replay_size = replay_size
        self.polyak = polyak
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        # 共享动作空间的信息
        ac_kwargs['action_space'] = a_bound
        self.ac_kwargs = ac_kwargs

        # 设置Her
        self.goal_selection_strategy = goal_selection_strategy
        self.n_sampled_goal = n_sampled_goal

        # 设置归一化器
        self.normer = StateNorm(size=self.obs_dim)
        
        # 设置loss计算时候的超参数
        self.action_l2 = action_l2
        self.clip_return = clip_return

        # 设置buffer
        self.buffer = ReplayBuffer(self.obs_dim,   # 在goal环境里，obs_dim 是 observation_dim + goal_dim
                                   self.act_dim, 
                                   self.replay_size, 
                                   self.device)    # 一个简单的FIFObuffer
    
    def get_action(self, state, noise_scale=0):
        '''
        输入的state是 observation+desired_goal 的concat
        noise_scale 是一个布尔值，默认是0， 也可以输入1
        如果noise_scale 不为0，那么就取它为act_noise
        a + 正态分布噪声
        get_action 本身就利用了normer进行 state的归一化了！
        '''
        # 先把输入的state进行归一化
        s = self.normer.normalize(state)
        if not noise_scale:                   # 如果noise_scale 不是0， 那么把noise_scale 赋值为action_noise
            noise_scale = self.action_noise
        s_cuda = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = self.ac.act(s_cuda)
        a += noise_scale * np.random.randn(self.act_dim)  # 形状为act_dim的标准正态分布噪声
        return np.clip(a, -self.a_bound, self.a_bound)
    
    def convert_dict_to_array(self, obs_dict,
                              exclude_key=['achieved_goal']):
        '''
        把原生obs，也就是dict{'observation':obs, 'achieved_goal':ag, 'desired_goal':g}
        转换成 array(cat(obs, g))
        把除了 'achieved_goal' 的其他键的值cat起来
        返回 [obs_dim+gaol_dim, ] 形状的array
        '''
        obs_array = np.concatenate([obs_dict[key] for key, _ in obs_dict.items() if key not in exclude_key])
        return obs_array
    

    # HER 处理数据=============================================================================================
    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        '''
        给长度为T的列表episode_trans， 一个目前的transition的索引0-T-1的整数
        episode_trans 是一个长度为 T 的列表，
        episode_trans 的每一个元素 
        trans 是一个长度为 6 的列表： 
            [0]: 1个原生 obs, dict{'observation':obs, 'achieved_goal':ag, 'desired_goal':g}
            [1]: 1个action
            [2]: 1个奖励r
            [3]: 1个原生 obs_next, dict{'observation':obs, 'achieved_goal':ag, 'desired_goal':g}
            [4]: 1个布尔值 done
            [5]: 1个info， dict{'is_success': 0.0}
        
        先得到一个idx  [目前的transition的索引 - T) 之间的数
        然后根据idx取一个 trans:一个长度为 6 的列表
                [0]: 1个原生 obs, dict{'observation':obs, 'achieved_goal':ag, 'desired_goal':g}
                [1]: 1个action
                [2]: 1个奖励r
                [3]: 1个原生 obs_next, dict{'observation':obs, 'achieved_goal':ag, 'desired_goal':g}
                [4]: 1个布尔值 done
                [5]: 1个info， dict{'is_success': 0.0}
        然后另 ag = 列表的[0]位置原生obs 字典的 achieved_goal
        '''
        if self.goal_selection_strategy == 'future':
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == 'final':
            selected_transition = episode_transitions[-1]
        else:
            raise ValueError('没有这种采样策略！')
        ag = selected_transition[0]['achieved_goal']
        return ag
    
    def _sample_achieved_goals(self, episod_transitions, transition_idx, n_sample_goal=4):
        '''
        利用上一个函数，采集出包含多个 ag 的列表 ags
        '''
        ags = [self._sample_achieved_goal(episod_transitions, transition_idx) for _ in range(n_sample_goal)]
        return ags

    def HER_save_episode(self, episode_trans, reward_func):
        '''
        在每一个步数为T的小回合结束时调用该函数
        把长度为T的列表episode_trans 经HER处理，然后存入buffer中
        episode_trans 是一个长度为 T 的列表，
        episode_trans 的每一个元素 
        trans 是一个长度为 6 的列表： 
            [0]: 1个原生 obs, dict{'observation':obs, 'achieved_goal':ag, 'desired_goal':g}
            [1]: 1个action
            [2]: 1个奖励r
            [3]: 1个原生 obs_next, dict{'observation':obs, 'achieved_goal':ag, 'desired_goal':g}
            [4]: 1个布尔值 done
            [5]: 1个info， dict{'is_success': 0.0}
        reward_func 是一个计算奖励的函数，输入两个goal，返回稀疏奖励或者返回密集奖励，fetch环境默认是稀疏奖励，也就是说返回-1或者1
        '''
        # 第一步，更新normer
        ep_obs = np.array([
                np.concatenate((trans[0]['observation'], trans[0]['desired_goal']))
                for trans in episode_trans
            ])  # 得到一个array 形状为 [T, observation_dim + goal_dim] 的 ep_obs， 只用来更新nomer，没有别的用处
        
        self.normer.update(v=ep_obs)      # 用得到的 ep_obs 来更新一下normer中的mean和std, 这个最后是要保存的！

        # 第二步，开始HER sample然后存入buffer
        for transition_idx, transition in enumerate(episode_trans):
            '''
            transition_idx:0,1,2...50
            transition: 对应上述idx，[1个原生obs, 1个action, 1个r, 1个原生obs_next, done]
            循环里面1步，存进 (buffer 1 + n_samples) 组 [arr_obs, action, r, arr_obs_next, done]
            '''
            # 循环终止条件
            if (transition_idx == len(episode_trans)-1) and (self.goal_selection_strategy=='future'):
                break

            # 对raw_obs和raw_obs_next进行字典转合并数组操作
            raw_obs, action, reward, raw_next_obs, done, info = copy.deepcopy(transition)  # raw_obs: dict{'observation':obs, 'achieved_goal':ag, 'desired_goal':g}
            obs_arr, next_obs_arr = map(self.convert_dict_to_array, (raw_obs, raw_next_obs))  # map函数传入一个函数和一个可迭代对象！把对象中的每一项都用函数处理！
            # obs_arr [observation_dim + goal_dim,],   next_obs_arr [observation_dim + goal_dim,]

            # 对array进行归一化
            obs_arr = self.normer.normalize(v=obs_arr)
            next_obs_arr = self.normer.normalize(v=next_obs_arr)
            # obs_arr [observation_dim + goal_dim,],   next_obs_arr [observation_dim + goal_dim,]

            # 把归一化之后的 1份 obs_arr, next_obs_arr 和动作等传入buffer
            self.buffer.store(obs_arr, action, reward, next_obs_arr, done)

            # 利用目前的 transition_idx 和 全部的 episod_trans 采样出一个包含多个ag的列表， 作为新的 desired_goals
            # [(goal_dim,), (goal_dim,), ...]
            sampled_goals = self._sample_achieved_goals(episode_trans, transition_idx, n_sample_goal=self.n_sampled_goal)

            # 对每一个 新的 desired_goals， 把transition拿过来复制一下，产生一个有新的desired goals的transition
            # 存 n_samples 个 条目 进入到 buffer
            for new_goal in sampled_goals:
                # 对raw_obs和raw_obs_next进行置换并且进行字典转合并数组操作
                #-------------------------------------------------------------------------------------
                raw_obs, action, reward, raw_next_obs, done, info = copy.deepcopy(transition)

                # 置换, 并计算新的reward
                raw_obs['desired_goal'] = new_goal
                raw_next_obs['desired_goal'] = new_goal
                # 计算新的奖励，下一个状态已经到达的位置，和new_goal
                reward = reward_func(raw_next_obs['achieved_goal'], new_goal, info)
                # 保证done一直为False
                done = False

                # 对被置换后的raw_obs和raw_obs_next进行字典转合并数组操作
                obs_arr, next_obs_arr = map(self.convert_dict_to_array, (raw_obs, raw_next_obs))
                #-----------------------------------------------------------------------------------------
                # obs_arr [observation_dim + goal_dim,],   next_obs_arr [observation_dim + goal_dim,]

                # 对array进行归一化
                obs_arr = self.normer.normalize(v=obs_arr)
                next_obs_arr = self.normer.normalize(v=next_obs_arr)
                # obs_arr [observation_dim + goal_dim,],   next_obs_arr [observation_dim + goal_dim,]

                # 把归一化之后的 1份 obs_arr, next_obs_arr 和动作等传入buffer
                self.buffer.store(obs_arr, action, reward, next_obs_arr, done)
    # HER 处理数据部分结束=============================================================================================
    
    def test_agent(self, args, env, n=5, logger=None, obs2state=None):
        '''
        利用现有的agent测试环境，并记录信息
        '''
        ep_reward_list = []
        for j in range(n):
            obs = env.reset()
            ep_reward = 0
            success = []

            for i in range(args.n_steps):
                s = obs2state(obs)
                a = self.get_action(s)
                # 归一化操作在get_action内部完成！
                obs, r, done, info = env.step(a)
                success.append(info['is_success'])
                ep_reward += r
            
            if logger:
                logger.store(TestEpRet=ep_reward)
                logger.store(TestSuccess=success[-1])  # 返回本回合最后时刻的success标志，代表本回合是否完成！
            
            ep_reward_list.append(ep_reward)
        mean_ep_reward = np.mean(np.array(ep_reward_list))
        if logger:
            return mean_ep_reward, logger
        else:
            return mean_ep_reward

    def save_net_and_norm(self, save_path):
        '''
        保存下来现在的 nomer 和 ac
        '''
        model_path = save_path + '/ac_norm_model.pth'
        torch.save([self.normer, self.ac], model_path)
        print('save normer and ac to: ', model_path)
