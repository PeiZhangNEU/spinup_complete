import numpy as np


class StateNorm:
    '''
    将传入的 obs 和 goal 的cat后的量一起归一化
    这是一种离线归一化的方法，记录下来数据的mean和std，用来对一批数据进行归一化，
    下一批数据来了再更新归一化的参数
    '''
    def __init__(self, size, eps=1e-2, default_clip_range=5):
        '''
        self.size = observation_dim + goal_dim
        '''
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.count = np.zeros(1, np.float32)

        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    # update the parameters of the normalizer
    # normer 一旦初始化，里面的元素都不会清零，会一直叠加，根据大数定理，到最后，这个normer的mean和std代表整个空间的mean和std
    def update(self, v):
        '''
        v = shape [T, observation_dim + goal_dim] 的 ep_obs
        '''
        v = v.reshape(-1, self.size)
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count += v.shape[0]

        self.mean = self.sum / self.count
        self.std = np.sqrt(np.maximum(np.square(self.eps),
                                      (self.sumsq / self.count) - np.square(
            self.sum / self.count)))
        # print("mean:", self.mean)
        # print("std:", self.std)

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range

        return np.clip((v - self.mean) / self.std,
                       -clip_range, clip_range)

