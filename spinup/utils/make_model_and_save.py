import os 
import torch

def make_model_and_save(model, dirs, name):
    '''用来保存模型pt到特定位置
    输入：
    model实例， 父目录路径， 模型名字
    policy.pi   'model_view/sac/' 'actor.pt'
    '''
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model, dirs + name)
