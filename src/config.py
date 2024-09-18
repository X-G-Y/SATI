import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

# path to a pretrained word embedding file
word_emb_path = '/home/s22xjq/SATI/glove.840B.d.txt'
assert(word_emb_path is not None)


username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('/home/s22xjq/CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam, 'NAdam':optim.NAdam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh, "sigmoid" :nn.Sigmoid}


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[self.data.lower()]
        self.sdk_dir = sdk_dir
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        # self.data_dir = self.dataset_dir.joinpath(self.mode)
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)

    # Bert
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)
    parser.add_argument('--use_domain', type=str2bool, default=True)
    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    #parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--name', type=str, default=f"bestF1")
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=50)
    #try: use early stop in mosei
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--start_saving', type=float, default=0.4)

    parser.add_argument('--diff_weight', type=float, default=0.4)
    parser.add_argument('--sim_weight', type=float, default=1.0)
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--jsd_weight', type=float, default=1)



    parser.add_argument('--learning_rate', type=float, default=1.7e-05)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    

    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='relu')
    
    
    #my
    parser.add_argument('--queries', type=int, default='64')
    #layers of self-attention
    parser.add_argument('--layers', type=int, default=2)

    # Model
    parser.add_argument('--model', type=str,
                        default='SATI', help='one of {SATI, }')

    # Data
    parser.add_argument('--data', type=str, default='mosi')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    print(kwargs.data)
    if kwargs.data == "mosi":
        kwargs.num_classes = 1
        kwargs.batch_size = 64
        kwargs.use_domain = True
        kwargs.start_saving = 0.4
        kwargs.patience = 100
    elif kwargs.data == "mosei":
        kwargs.num_classes = 1
        kwargs.batch_size = 16
        kwargs.use_domain = True
        kwargs.start_saving = 0.0
        kwargs.patience = 5
    elif kwargs.data == "ur_funny":
        kwargs.num_classes = 2
        kwargs.batch_size = 32
    else:
        print("No dataset mentioned")
        exit()

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
