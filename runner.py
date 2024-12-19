import torch
import numpy as np
import pickle
import time
import torch.nn as nn
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *
from benchmark import *
from tensorboardX import SummaryWriter
from model.model import model_lstm

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# todo 修改一下 和model有接口 然后模型都在model文件夹里 这里不会有其他的东西


class Runner(object):
    def __init__(self, config, tokenizer) -> None:
        self.config = config


        # if 模型一

        self.model = model_lstm(config, tokenizer)
        self.tokenizer = tokenizer
        self.learning_step = 0

        # 打印一下 model
        print(get_parameter_number(self.model.decoder))
        print(get_parameter_number(self.model.encoder))


    def train(self, dataloader, eval_dataloader, test_dataloader, tb_logger = None):
        max_epoch = self.config.max_epoch
        start_epoch = self.config.start_epoch # if load model todo


        self.loss_list = []
        # training
        for epoch in range(start_epoch, max_epoch + 1):
            print(f"Training epoch [{epoch}/{max_epoch}]...")
            epoch_loss_list = [] # store loss per epoch
            # todo 从dataloader中取东西
            # todo desc中还得加入学习率啥的，这里还没加入
            for bs_img, bs_caption, caption_length in tqdm(dataloader, desc = f"run_name:{self.config.run_name}-epoch[{epoch}/{max_epoch}]"):
                bs_loss, predicts_string = self.model.train_batch(bs_img, bs_caption, caption_length)
                self.learning_step += 1
                epoch_loss_list.append(bs_loss)
                self.loss_list.append(bs_loss)
                if tb_logger is not None and self.learning_step % self.config.log_interval == 0:
                    acc = exact_match_score(bs_caption, predicts_string) / 100.0
                    tb_logger.add_scalar("train/loss", bs_loss, self.learning_step)
                    tb_logger.add_scalar("train/accuracy", acc, self.learning_step)
            print(f"Loss of epoch [{epoch}/{max_epoch}] is {np.mean(epoch_loss_list)}, current learning steps:{self.learning_step}")

            # todo save model
            if epoch % self.config.save_interval == 0:
                self.save_model(self.config.output_dir + "save_model/", epoch)
                # 验证集验证一波
                self.eval(eval_dataloader, epoch, tb_logger)
                # self.test(test_dataloader, self.config.output_dir + 'test/', epoch)
                # todo
                if epoch >= 50:
                    self.test(test_dataloader, self.config.output_dir + 'test/', epoch)
                    # 再测测试集

            self.model.lr_scheduler.step()
        tb_logger.close()

        # save loss
        loss_path = self.config.output_dir
        with open(f"{loss_path}/loss.pt", 'wb') as f:
            torch.save(self.loss_list, f)


    def save_model(self, save_dir, epoch):
        print("Saving model...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # # 这里保存的是整个类
        # # 可以选择保存 encoder、decoder、optimizer的状态
        # with open(save_dir + f"epoch-{epoch}.pkl", 'wb') as f:
        #     pickle.dump(self.model, f, -1)

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

    # only for eval
    def eval(self, dataloader, epoch, tb_logger = None):
        self.model.encoder.eval()
        self.model.decoder.eval()

        refers = []
        hypos = []
        device = self.config.device
        with torch.no_grad():
            for imgs, refer, length in tqdm(dataloader, desc = "Evaluating..."):
                imgs = imgs.to(device)
                encoded_imgs = self.model.encoder(imgs)

                predicts = self.model.decoder.generate(encoder_out = encoded_imgs, temperature = 0, top_p = 0.25)

                predicts_string = self.tokenizer.decode(predicts)

                refers.extend(refer)
                hypos.extend(predicts_string)
        score1, score2, score3, t_score = total_score(refers, hypos)
        print(f'eval scores: BLEU Score:{score1}, Edit Distance Score:{score2}, Exact Match Score:{score3}')

        if tb_logger is not None:
            tb_logger.add_scalar('eval/score1(BLEU)', score1, epoch)
            tb_logger.add_scalar('eval/score2', score2, epoch)
            tb_logger.add_scalar('eval/score3(Match)', score3, epoch)
            tb_logger.add_scalar('eval/total_score', t_score, epoch)

    def test(self, dataloader, output_path, epoch):
        self.model.encoder.eval()
        self.model.decoder.eval()

        device = self.config.device
        save_list = []
        save_ids = []
        with torch.no_grad():
            for ids, imgs in tqdm(dataloader, desc = "Testing..."):
                imgs = imgs.to(device)
                encoded_imgs = self.model.encoder(imgs)

                predicts = self.model.decoder.generate(encoder_out = encoded_imgs)
                predicts_string = self.tokenizer.decode(predicts)
                save_list.extend(predicts_string)
                save_ids.extend(ids)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # 有序号
        with open(output_path + f"test_out_id-{epoch}.txt", 'w') as f:
            for id, strs in zip(save_ids, save_list):
                f.write(f"{id.item()}" + ": " + strs + '\n')

        # 无序号
        with open(output_path + f"test_out-{epoch}.txt", 'w') as f:
            for str in save_list:
                f.write(str[8: len(str) - 6] + '\n')
