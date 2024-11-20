import torch
import numpy as np
import pickle
import time
import torch.nn as nn
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class Trainer(object):
    def __init__(self, config, tokenizer) -> None:
        self.config = config

        # todo encoder decoder
        self.encoder = nn.Linear(1, 1)
        self.decoder = nn.Linear(1, 1)

        self.tokenizer = tokenizer

        # todo optimizer

        self.optimizer = torch.optim.Adam(
            [{"params": self.encoder.parameters(), 'lr': self.config.lr_encoder}] +
            [{"params": self.decoder.parameters(), 'lr': self.config.lr_decoder}]
        )

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR()


        # input cuda

        # todo


        # figure out criterion
        self.criterion = nn.CrossEntropyLoss().to(self.config.device)

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
                bs_loss = self.train_batch(bs_img, bs_caption, caption_length)
                self.learning_step += 1
                epoch_loss_list.append(bs_loss)

                # todo tensorboard
            print(f"Loss of epoch [{epoch}/{max_epoch}] is {np.mean(epoch_loss_list)}, current learning steps:{self.learning_step}")

            # todo save model

            self.lr_scheduler.step()
        #todo save loss

    def train_batch(self, image, caption, length):
        # todo 开启训练

        device = self.config.device
        image = image.to(device)
        encoded_caption, origin_idxs = self.tokenizer.encoder(caption)

        encoded_caption = torch.FloatTensor(encoded_caption).to(device)

        encoded_image = self.encoder(image)

        # 解码
        scores, decoded_caption, decoded_length, sort_idx = self.decoder(encoded_image, encoded_caption, length)

        target_idxs = origin_idxs[sort_idx][:, 1:]

        scores = pack_padded_sequence(scores, decoded_length.cpu(), batch_first = True).data
        targets = pack_padded_sequence(target_idxs, decoded_length.cpu(), batch_first = True).data

        assert not torch.any(torch.isnan(scores)), "Nan happen in scores!!"
        assert not torch.any(torch.isnan(targets)), "Nan happen in targets!!"

        loss = self.criterion(scores, targets.long())

        assert not torch.isnan(loss), "Nan happen in loss!!!"

        self.optimizer.zero_grad()
        loss.backward()

        # 梯度截断
        clip_grad_norms(self.optimizer.param_groups, max_norm = self.config.max_norm)

        self.optimizer.step()

        return loss.detach().cpu()


    def save_model(self, save_dir, epoch):
        print("Saving model...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 这里保存的是整个类
        # 可以选择保存 encoder、decoder、optimizer的状态
        with open(save_dir + f"epoch-{epoch}.pkl", 'wb') as f:
            pickle.dump(self, f, -1)

        torch.save(
            {
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            os.path.join(save_dir, f'epoch-{epoch}.pt')
        )
        print("Successfully save model...")

    def load(self, path):
        # 这里用的是导入state
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        print("Successfully load model")