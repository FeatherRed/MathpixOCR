import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
from utils import sample_top_p
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=6, stride=6)

        # self.fc1 = nn.Linear(10 * 56, 48)
        # self.relu3 = nn.ReLU()

        # self.fc2 = nn.Linear(1024, 512)
        # self.relu4 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        # print(f'out:{out.shape}')
        # out = out.view(out.size(0), out.size(1), -1)
        # print(f'out:{out.shape}')
        # out = self.fc1(out)
        # out = self.relu3(out)

        # out = self.fc2(out)
        # out = self.relu4(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self.make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 64, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x : [bs, 3, w, h]
        out = self.conv1(x) # [bs, in_channels, w, h]
        out = self.bn1(out) # [bs, in_channels, w, h]
        out = F.relu(out) # [bs, in_channels, w, h]

        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out) # [bs, in_channels, w, h]
        out = self.layer2(out) # [bs, in_channels, w // 2, h // 2]
        out = self.layer3(out) # [bs, in_channels, w // 2 // 2, h // 2 // 2]
        out = self.layer4(out)  # [bs, in_channels, w // 2 // 2 // 2, h // 2 // 2 // 2]
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)

        # [bs, 3, w, h] -----> [bs, 64, w // 8, h // 8]

        return out


class Encoder4lstm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.resnet = torchvision.models.resnet50(pretrained=True)
        # self.fc = nn.Linear(1000, 256)
        # self.relu = nn.LeakyReLU()
        # self.cnn = CNN()
        self.cnn = ResNet(BasicBlock, [1, 1, 1, 1])
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

    def forward(self, image):
        encoded_out = None
        encoded_out = self.cnn(image)
        encoded_out = encoded_out.view(encoded_out.size(0), encoded_out.size(1), -1)
        encoded_out = encoded_out.permute(0, 2, 1)

        out_lstm, (h, c) = self.lstm(encoded_out)
        # print(f'debug: outlstm: {out_lstm.shape}')
        # only tune the fc network

        return out_lstm[:, -1]


class Decoder_lstm(nn.Module):
    def __init__(self, tokenizer, config) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.output_size = tokenizer.vocab_size
        self.max_length = config.max_length
        self.config = config
        self.input_size = 64
        self.hidden_size = 256
        self.num_layer = 1
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layer,
            batch_first=True)
        self.output_net = nn.Linear(self.hidden_size, self.output_size)
        self.embedding_net = nn.Linear(100, self.input_size)

        self.init_h_net = nn.Linear(self.input_size, self.hidden_size)
        self.init_c_net = nn.Linear(self.input_size, self.hidden_size)

    def forward(self, encoder_out, encoded_captions=None, caption_lengths=None):
        bs = encoder_out.shape[0]
        device = encoder_out.device
        # print(f'debug: device in train {device}')

        # sort by the length for the ease of paralleling
        # print(f'debug: captions length:{caption_lengths}')
        caption_lengths, sort_idx = torch.sort(caption_lengths, dim=-1, descending=True)
        # print(f'debug: after captions length:{caption_lengths}')

        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]

        # h_0 = torch.zeros(1, bs, self.hidden_size).to(device)
        # c_0 = torch.zeros_like(h_0)

        h = self.init_h_net(encoder_out).unsqueeze(0)
        c = self.init_c_net(encoder_out).unsqueeze(0)

        # print(f'debug: h:{h.shape}, c:{c.shape}')
        embeded_captions = self.embedding_net(encoded_captions)

        decoder_length = (caption_lengths - 1)
        # print(f'debug: decoder length:{decoder_length}')

        predicts = (torch.ones((bs, max(decoder_length), self.output_size)) * -1).to(device)

        for t in range(max(decoder_length)):
            working_index = sum([l > t for l in decoder_length])
            x_in = embeded_captions[:working_index, t].unsqueeze(1)
            h = h[:, :working_index]
            c = c[:, :working_index]
            # print(f'debug: x_in:{x_in.shape}')
            out_lstm, (h, c) = self.lstm(x_in, (h, c))
            out_lstm = self.output_net(out_lstm).squeeze(1)
            # no need to softmax
            # out_lstm = torch.softmax(out_lstm, -1)
            # print(f'debug: out_lstm:{out_lstm.shape}, predict:{predicts[:working_index, t].shape}')
            predicts[:working_index, t] = out_lstm

        return predicts, encoded_captions, decoder_length, sort_idx

    def generate(self, encoder_out, temperature=0, top_p=1):
        bs = encoder_out.shape[0]
        device = encoder_out.device

        cur_len = 0

        h = self.init_h_net(encoder_out).unsqueeze(0)
        c = self.init_c_net(encoder_out).unsqueeze(0)

        working_index = torch.arange(bs).to(device)

        # init selected tokens
        max_length = self.max_length
        selected_tokens = torch.zeros((bs, max_length), dtype=torch.long).to(device)
        selected_tokens[:, 0] = self.tokenizer.start_token_index

        selected_embeded_tokens = torch.zeros((bs, max_length, self.input_size)).to(device)
        # print(f'dubug: sel: {self.tokenizer.word_vectors[selected_tokens[:,0].cpu()].shape}')
        selected_embeded_tokens[:, 0] = self.embedding_net(
            torch.FloatTensor(self.tokenizer.word_vectors[selected_tokens[:, 0].cpu()]).to(device))

        while working_index.shape[0] > 0:
            x_in = selected_embeded_tokens[working_index, cur_len].unsqueeze(1)
            # x_in = (2 * selected_tokens[working_index, cur_len] - self.output_size) / self.output_size
            # print(f'debug: x_in:{x_in.shape}')
            # x_in = self.embedding_net(x_in).unsqueeze(1)
            # print(f'debug: x_in:{x_in.shape}')

            out_lstm, (h, c) = self.lstm(x_in, (h, c))
            out_lstm = self.output_net(out_lstm).squeeze(1)

            if temperature > 0:
                probs = torch.softmax(out_lstm / temperature, -1)
                select_tok = sample_top_p(probs, top_p).squeeze_(-1)
            else:
                select_tok = torch.argmax(out_lstm, -1)
            # print(f'debug: select_tok{select_tok}')
            selected_tokens[working_index, cur_len + 1] = select_tok
            selected_embeded_tokens[working_index, cur_len + 1, :] = self.embedding_net(
                torch.FloatTensor(self.tokenizer.word_vectors[select_tok.cpu()]).to(device))

            not_end_index = self.tokenizer.not_end(select_tok)
            # print(f'debug: index:{not_end_index}')
            working_index = working_index[not_end_index]
            # print(f'debug: working_idx:{working_index}')
            h = h[:, not_end_index]
            c = c[:, not_end_index]

            cur_len += 1

            if cur_len == max_length - 2:
                selected_tokens[working_index, cur_len + 1] = self.tokenizer.end_token_index
                break

        return selected_tokens.cpu().long().squeeze()


