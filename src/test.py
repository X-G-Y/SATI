import os
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict
from data_loader import get_loader
from testsolver import Solver

import torch
import torch.nn as nn
from torch.nn import functional as F


if __name__ == '__main__':
    
    # Setting random seed
    random_name = str(random())
    random_seed = 336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    #print(train_config)

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)


    folder = "/home/s22xjq/SATI/src/checkpoints_无敌！"
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=False)

    # Build the model
    solver.build()

    solver.eval(folder, mode="test", to_print=True)
