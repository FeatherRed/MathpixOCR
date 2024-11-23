import torch
import numpy as np
import time
import os
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import clip_grad_norms
from model.basic_model import Basic_Model

class model_transformer(Basic_Model):
    def __init__(self, config, tokenizer):
        super(model_transformer, self).__init__(config, tokenizer)

        self.config = config
        self.tokenizer = tokenizer

        self.device = self.config.device


        # figure encoder decoder
        self.encoder = nn.Linear(1, 1) # todo
        self.decoder = nn.Linear(1, 1) # todo

        # figure optimizer

        self.optimizer = torch.optim.Adam(
            [{"params": self.encoder.parameters(), 'lr': self.config.lr_encoder}] +
            [{"params": self.decoder.parameters(), 'lr': self.config.lr_decoder}]
        )

        # figure lr_scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay, last_epoch = -1,)

        # figure out criterion
        self.criterion = nn.CrossEntropyLoss().to(self.config.device)

    def train_batch(self, image, caption, length):

        device = self.device
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

    def get_parameter_number(self):
        # encoder and decoder
        pass

        # total_num = sum(p.numel() for p in self.parameters())
        # trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


