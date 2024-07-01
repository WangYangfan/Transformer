import os
import torch
import torch.nn as nn
import random
import numpy as np

def seed_environment(seed):
    """ 设置整个环境的随机种子 """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return

def check_model(model: nn.Module):
    """ 检查模型梯度 """
    # print(model.parameters())
    for name, param in model.named_parameters():
        print(
            "name:", name,
            "grad_requires:", param.requires_grad,
            # "weight:", param.data,
            # "grad_value：", param.grad
            "device:", param.device,
        )
    return

def check_config(config: dict):
    """ 检查配置 """
    assert config['device'] is not None
    assert config['name'] is not None

    for path_name in ['log_path', 'cache_path', 'output_path']:
        new_path = config[path_name]
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    for key, value in config.items():
        print(key, value, type(value))
    return
