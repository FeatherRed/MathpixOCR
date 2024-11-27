import math

import torch
import numpy as np
import time
import os
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import clip_grad_norms
from model.basic_model import Basic_Model
from model.nets import Encoder4transformer as Encoder, Decoder_transformer as Decoder
class model_transformer(Basic_Model):
    def __init__(self, config, tokenizer):
        super(model_transformer, self).__init__(config, tokenizer)

        self.config = config
        self.tokenizer = tokenizer

        self.device = self.config.device


        # figure encoder decoder
        # Encoder 和 Decoder初步
        # self.encoder = CNN_Encoder(feature_dim=512)  # 假设 CNN 输出特征维度为 512
        # self.decoder = RNN_Decoder(
        #     feature_dim=512,
        #     embed_dim=256,
        #     hidden_dim=512,
        #     vocab_size=tokenizer.vocab_size
        # )

        self.encoder = Encoder()
        self.decoder = Decoder(tokenizer, config)

        # figure optimizer

        self.optimizer = torch.optim.Adam(
            [{"params": self.encoder.parameters(), 'lr': self.config.lr_encoder}] +
            [{"params": self.decoder.parameters(), 'lr': self.config.lr_decoder}]
        )

        # figure lr_scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay, last_epoch = -1,)

        # figure out criterion
        self.criterion = nn.CrossEntropyLoss().to(self.config.device)


        # move to device
        device = self.config.device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def train_batch(self, image, caption, length):
        self.decoder.train()
        self.encoder.train()

        device = self.device
        image = image.to(device)
        encoded_caption, target_idxs = self.tokenizer.encode(caption)

        encoded_caption = torch.FloatTensor(encoded_caption).to(device)
        target_idxs = torch.tensor(target_idxs, dtype = torch.long).to(device)
        length = length.to(device)
        encoded_image = self.encoder(image)

        # 解码
        scores, decoded_caption, decoded_length, sort_idx = self.decoder(encoded_image, encoded_caption, length)

        targets = target_idxs[sort_idx][:, 1:]

        scores = pack_padded_sequence(scores, decoded_length.cpu(), batch_first = True).data
        targets = pack_padded_sequence(targets, decoded_length.cpu(), batch_first = True).data.squeeze()

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


# 定义 CNN Encoder
class CNN_Encoder(nn.Module):
    def __init__(self, feature_dim):
        super(CNN_Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

    def forward(self, images):
        # 输入：图像 (batch_size, 3, height, width)
        features = self.cnn(images)  # 输出特征 (batch_size, feature_dim, h, w)
        return features


# 定义 RNN Decoder
class RNN_Decoder(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions, lengths):
        # Embedding
        embeddings = self.embedding(captions)  # (batch_size, max_length, embed_dim)

        # RNN 输入：拼接 CNN 特征和 Embedding
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed_inputs)  # RNN 输出
        outputs = self.fc(packed_outputs.data)  # 全连接层输出词分布
        return outputs

    def generate(self, encoder_out, temperature = 0, top_p = 1):
        bs = encoder_out.shape[0]
        device = encoder_out.device

        pass