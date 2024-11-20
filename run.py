import torch
import numpy as np

import time
import os

import pickle

from torch.utils.data import DataLoader
from trainer import Trainer

class Config(object):
    def __init__(self) -> None:
        self.run_name = "test"
        self.start_epoch = 0
        self.max_epoch = 100
        self.lr_max = 1e-4
        self.lr_min = 1e-5
        self.lr_decay = pow((self.lr_min / self.lr_max), 1 / self.max_epoch)

        self.device = torch.device('cuda:0')


        self.train_batch_size = 32
        self.save_interval = 5
        self.log_interval = 30

        self.max_token_length = 50
        self.max_norm = 1.0
        self.seed = 42

        self.img_dir = # todo

        # for test
        self.training = True
        self.load_path = None # todo
        self.test_img_dir = # todo

def main():
    config = Config()

    # set seed
    np.random.rand(config.seed)
    torch.manual_seed(config.seed)

    if config.training:
        config.run_name = "{}_{}".format(config.run_name, time.strftime("%Y%m%dT%H%M%S"))





    # figure out trainer
    trainer = Trainer(config, tokenizer = None) # todo

    if config.load_path is not None:
        trainer.load(config.load_path)
