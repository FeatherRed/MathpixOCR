import torch
import numpy as np

import time
import os

import pickle
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from runner import Runner
from dataloader import create_dataloader, create_testloader

import torchvision.transforms as transforms
from tokenizer import Tokenizer


class Config(object):
    def __init__(self) -> None:
        self.run_name = "test"
        self.start_epoch = 0
        self.max_epoch = 100
        self.lr_max = 1e-3
        self.lr_min = 1e-4
        self.lr_decay = pow((self.lr_min / self.lr_max), 1 / self.max_epoch)

        self.device = torch.device('cuda:0')

        self.lr_encoder = self.lr_max
        self.lr_decoder = self.lr_max

        self.train_ratio = 0.9  # valid_ratio = 1 - train_ratio
        self.train_batch_size = 32
        self.valid_batch_size = 1000
        self.test_batch_size = 1000
        self.save_interval = 5
        self.log_interval = 30
        self.validate_period = 5
        self.batch_size = 32
        self.save_period = 5
        self.log_period = 30
        self.max_token_length = 50
        self.max_norm = 1.0
        self.seed = 42

        self.img_dir = "datasets/mini/images"
        self.pkl_dir = "datasets/mini/pkls"
        self.output_dir = "output/"

        # for test
        self.training = True
        self.test_img_dir = "datasets/test/images"  # todo
        self.test_ids = [i for i in range(104011, 105061)]  # 0 ~ 1049

        self.load_path = None # 'output/test_20241128T104341/save_model/'
        self.load_model = None # 'epoch-25.pt'
        self.load_name = None # 'test_20241128T104341
        self.max_length = 200


def main():
    config = Config()

    # set seed
    np.random.rand(config.seed)
    torch.manual_seed(config.seed)

    tb_logger = None

    if config.training:
        config.run_name = "{}_{}".format(config.run_name, time.strftime("%Y%m%dT%H%M%S"))
        config.output_dir = config.output_dir + config.run_name + '/'
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        tb_logger = SummaryWriter(os.path.join("tensorboard"), config.run_name)
    else:
        # only test
        config.output_dir = config.output_dir + config.load_name + '/'

    transform = transforms.Compose([
        transforms.Resize((40, 240)),  # Resize image to a fixed size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalize image
    ])

    if config.training:
        dataloader, valid_dataloader, vocab = create_dataloader(img_dir = config.img_dir,
                                                                pkl_dir = config.pkl_dir,
                                                                train_ratio = config.train_ratio,
                                                                train_batch_size = config.train_batch_size,
                                                                valid_batch_size = config.valid_batch_size,
                                                                transform = transform
                                                                )
        tokenizer = Tokenizer(vocab)
        # 保存 tokenizer

        print(f'debug: max length: {dataloader.dataset.max_length}')
        # print(f'debug: max length: {valid_dataloader.dataset.max_length}')
        with open(config.output_dir + 'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f, -1)
            print(f"Successfully save tokenizer!")

        test_dataloader = create_testloader(config.test_img_dir, ids = config.test_ids,
                                            test_batch_size = config.test_batch_size,
                                            transform = transform)

        # figure out trainer
        trainer = Runner(config, tokenizer = tokenizer)  # todo

        if config.load_path is not None:
            trainer.load(config.load_path)

        trainer.train(dataloader, valid_dataloader, test_dataloader, tb_logger = tb_logger)
    else:
        # test
        # load tokenizer
        with open(config.output_dir + 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        tester = Runner(config = config, tokenizer = tokenizer)
        # load model
        tester.load(config.load_path + config.load_model)

        test_dataloader = create_testloader(config.test_img_dir, ids = config.test_ids,
                                            test_batch_size = config.test_batch_size,
                                            transform = transform)
        tester.test(test_dataloader, config.output_dir + 'test/', 1) # 1表示 test


if __name__ == "__main__":
    main()
