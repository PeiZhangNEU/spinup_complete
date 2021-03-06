# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/deeplearning/spinningup_project')
import numpy as np
import torch
import gym
import time
import spinup.alogos.ddpg.core as core
from spinup.alogos.ddpg.ddpg import *

def train(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         delayup=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, update_times=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1, use_gpu=False):
        '''
        训练函数，传入agent和env进行训练。主循环。
        '''
        # 训练时可以选择使用GPU或者是CPU
        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                print("这台设备不支持gpu，自动调用cpu")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')

        agent = ddpg(env_fn, actor_critic=actor_critic, ac_kwargs=ac_kwargs, 
                     replay_size=replay_size, gamma=gamma, 
                     delayup=delayup, pi_lr=pi_lr, q_lr=q_lr, 
                     num_test_episodes=num_test_episodes, max_ep_len=max_ep_len,
                     logger_kwargs=logger_kwargs, device=device)
        env = env_fn()

        # 随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 准备开始主循环
        total_steps = steps_per_epoch * epochs   # 和on-policy不同，off-policy的算法用总步数进行训练，不分epoch了
        start_time = time.time()
        o, ep_ret, ep_len = env.reset(), 0, 0

        # 主循环,就一个循环，不需要on policy的那种麻烦的终止条件。
        
        for t in range(total_steps): 
            # 在开始步数到达之前，只使用随机动作；达到开始步数之后，使用pi得到的加噪声的动作
            if t > start_steps:
                a = agent.get_action(o, act_noise)
            else:
                a = env.action_space.sample()
            
            # 执行环境交互
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # 如果因为运行到末尾出现done，把done置为False
            d = False if ep_len == max_ep_len else d

            # 存数据到buffer
            agent.buffer.store(o, a, r, o2, d)

            # 非常重要，更新状态
            o = o2

            # 如果 d 或者 回合结束，则重置状态和计数器
            if d or (ep_len == max_ep_len):
                agent.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0
            
            # 如果达到了更新步数，之后每隔50步就update50次
            if t >= update_after and t % update_every == 0:
                for _ in range(update_times):
                    batch = agent.buffer.sample_batch(batch_size)
                    agent.update(data=batch)
            
            # 打印以及存储模型，以及测试模型
            if (t+1) % steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch   # 除完向下取整

                # 存储模型
                if (epoch % save_freq == 0) or (epoch == epochs):
                    agent.logger.save_state({'env':env}, None)
                
                # 测试目前的表现
                agent.test_agent()

                # 打印
                agent.logger.log_tabular('Epoch', epoch)
                agent.logger.log_tabular('EpRet', with_min_and_max=True)
                agent.logger.log_tabular('TestEpRet', with_min_and_max=True)
                agent.logger.log_tabular('EpLen', average_only=True)
                agent.logger.log_tabular('TestEpLen', average_only=True)
                agent.logger.log_tabular('TotalEnvInteracts', t)
                agent.logger.log_tabular('QVals', with_min_and_max=True)
                agent.logger.log_tabular('LossPi', average_only=True)
                agent.logger.log_tabular('LossQ', average_only=True)
                agent.logger.log_tabular('Time', time.time()-start_time)
                agent.logger.dump_tabular()


if __name__ == '__main__':
    # 设置超参数
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)  # 一共训练了3e6次
    # parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--exp_name', type=str, default='ddpg_pendulum')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # 执行训练过程
    train(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs, use_gpu=True)