import torch
import numpy as np
import pickle
import time
import torch.nn as nn
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *


from model.transformer import model_transformer

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# todo 修改一下 和model有接口 然后模型都在model文件夹里 这里不会有其他的东西


class Trainer(object):
    def __init__(self, config, tokenizer) -> None:
        self.config = config


        # if 模型一

        self.model = model_transformer(config, tokenizer)

        self.learning_step = 0


    def train(self, dataloader, eval_dataloader, test_dataloader, tb_logger = None):
        max_epoch = self.config.max_epoch
        start_epoch = self.config.start_epoch # if load model todo

        # training
        for epoch in range(start_epoch, max_epoch):
            print(f"Training epoch [{epoch}/{max_epoch}]...")
            epoch_loss_list = [] # store loss per epoch
            # todo 从dataloader中取东西
            # todo desc中还得加入学习率啥的，这里还没加入
            for bs_img, bs_caption, caption_length in tqdm(dataloader, desc = f"run_name:{self.config.run_name}-epoch[{epoch}/{max_epoch}]")
                bs_loss = self.model.train_batch(bs_img, bs_caption, caption_length)
                self.learning_step += 1
                epoch_loss_list.append(bs_loss)

                # todo tensorboard
            print(f"Loss of epoch [{epoch}/{max_epoch}] is {np.mean(epoch_loss_list)}, current learning steps:{self.learning_step}")

            # todo save model

            self.model.lr_scheduler.step()
        #todo save loss

    def save_model(self, save_dir, epoch):
        print("Saving model...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 这里保存的是整个类
        # 可以选择保存 encoder、decoder、optimizer的状态
        with open(save_dir + f"epoch-{epoch}.pkl", 'wb') as f:
            pickle.dump(self.model, f, -1)

        torch.save(
            {
                'encoder': self.model.encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'optimizer': self.model.optimizer.state_dict()
            },
            os.path.join(save_dir, f'epoch-{epoch}.pt')
        )
        print("Successfully save model...")

    def load(self, path):
        # 这里用的是导入state
        state_dict = torch.load(path)
        self.model.encoder.load_state_dict(state_dict['encoder'])
        self.model.decoder.load_state_dict(state_dict['decoder'])
        self.model.optimizer.load_state_dict(state_dict['optimizer'])
        print("Successfully load model")
