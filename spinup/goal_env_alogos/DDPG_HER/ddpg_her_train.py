import numpy as np
import gym
import os, sys
sys.path.append('/home/zp/deeplearning/spinningup_project')
import torch
import time
from spinup.goal_env_alogos.DDPG_HER.ddpg_her import DDPGHER
# 下面引入的是该Up写的新的logx的文件，比较好用
from spinup.goal_env_alogos.spinup_utils.logx import setup_logger_kwargs, colorize
from spinup.goal_env_alogos.spinup_utils.logx import EpochLogger
from spinup.goal_env_alogos.spinup_utils.print_logger import Logger

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

def train(net, env, args):
    '''
    用args存储需要用到的参数，简洁
    '''
    # 初始化logger
    exp_name = args.exp_name + '_' + args.RL_name + '_' + args.env_name
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name,
                                        seed=args.seed,
                                        output_dir=args.output_dir + '/')
    logger = EpochLogger(**logger_kwargs)
    sys.stdout = Logger(logger_kwargs['output_dir'] + 'print.log', sys.stdout)
    logger.save_config(locals(), __file__)

    # 开始主循环
    # start running
    start_time = time.time()
    for i in range(args.n_epochs):
        for c in range(args.n_cycles):
            obs = env.reset()
            episode_trans = []
            # 把字典形式的obs转为array[observation_dim+goal_dim,]
            s = obs2state(obs)
            ep_reward = 0
            episode_time = time.time()
            success = []
            for j in range(args.n_steps):
                a = net.get_action(s, noise_scale=args.noise_ps)

                # 贪婪策略，如过随机数小于随机限度，就产生纯随机动作
                if np.random.random() < args.random_eps:
                    a = np.random.uniform(low=-net.a_bound,
                                          high=net.a_bound,
                                          size=net.act_dim)
                a = np.clip(a, -net.a_bound, net.a_bound)
                obs_next, r, done, info = env.step(a)
                success.append(info["is_success"])

                # 把字典形式的obs转为array[observation_dim+goal_dim,]
                s_ = obs2state(obs_next)

                # 防止gym中的最大step会返回done=True
                done = False if j == args.n_steps - 1 else done
                # episode trans列表
                episode_trans.append([obs, a, r, obs_next, done, info])

                # 同步更新array的obs和原生obs
                s = s_
                obs = obs_next

                ep_reward += r
            # 在一个回合结束，利用Her_save把transitions存入buffer中, 在存入的时候，用这个回合的数据对normer进行了更新！
            net.HER_save_episode(episode_trans=episode_trans,
                                 reward_func=env.compute_reward)
            logger.store(EpRet=ep_reward)

            # 优化网络40次
            for _ in range(40):
                lossq, qVals = net.update(args.batch_size)
                logger.store(QVals=qVals)
            # 如果这次cycle的 回合success大于0，就打印一下
            if 0.0 < sum(success) < args.n_steps:
                print("epoch:", i,
                      "\tep:", c,
                      "\tep_rew:", ep_reward,
                      "\ttime:", np.round(time.time()-episode_time, 3),
                      '\tdone:', sum(success))

        # 每一个epoch结束，测试并存储一下网络
        # 把logger传到test函数里面，存入TestEpRet 和 TestSuccess
        test_ep_reward, logger = net.test_agent(args=args,
                                                env=env,
                                                n=10,
                                                logger=logger,
                                                obs2state=obs2state,
                                                )
        logger.store(TestEpRet=test_ep_reward)

        # 测试完就存储一下网络
        net.save_net_and_norm(logger_kwargs["output_dir"])

        logger.log_tabular('Epoch', i)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('QVals', with_min_and_max=True)
        # 这两个是test函数返回的logger存储的
        logger.log_tabular('TestEpRet', average_only=True)
        logger.log_tabular('TestSuccess', average_only=True)   # 因为是average了，所以成功得到的是成功率！

        logger.log_tabular('TotalEnvInteracts', i * args.n_cycles * args.n_steps + c * args.n_steps + j + 1)
        logger.log_tabular('TotalTime', time.time() - start_time)
        logger.dump_tabular()
    

    print(colorize("the experience %s is end" % logger.output_file.name,
                   'green', bold=True))


if __name__ == '__main__':
    # 设置超参数, train 和 net 需要的所有参数都在这里
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='FetchPush-v1')
    parser.add_argument('--exp_name', type=str, default='Goalenv_exp')
    parser.add_argument('--RL_name', type=str, default='DDPGHER')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--n_epochs', type=int, default=500, help='the number of epochs to train the agent')
    parser.add_argument('--n_cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n_steps', type=int, default=50)                                      # 不同环境的默认回合数不一样，这个需要注意，可以通过环境得到
    parser.add_argument('--batch_size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--output_dir', type=str, default='data_Goalenv')


    parser.add_argument('--noise_ps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random_eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--action_l2', type=float, default=1.0, help='l2 reg')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')

    args = parser.parse_args()

    # 先把train需要的输入整出来
    env = gym.make(args.env_name)
    env.seed(args.seed)
    np.random.seed(args.seed)

    # 直接获取env的各项
    s_dim = env.observation_space.spaces['observation'].shape[0] + \
            env.observation_space.spaces['desired_goal'].shape[0]
    act_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    net = DDPGHER(act_dim=act_dim,
                  obs_dim=s_dim,
                  a_bound=a_bound,
                  action_l2=args.action_l2,
                  gamma=args.gamma,
                  seed=args.seed,
                  device=device)

    train(net, env, args)

    
