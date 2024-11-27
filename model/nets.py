import torch
import torch.nn as nn
import math
import torchvision
from utils import sample_top_p
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size = 1,
                    stride = stride,
                    bias = False,
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
    def __init__(self, block, num_blocks, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(
            3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self.make_layer(block, 32, num_blocks[0], stride = 1)
        self.layer2 = self.make_layer(block, 64, num_blocks[1], stride = 2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride = 2)
        self.layer4 = self.make_layer(block, 64, num_blocks[3], stride = 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 6, stride = 6)

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


class Encoder4transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = ResNet(BasicBlock, [1, 1, 1, 1])

    def forward(self, image):
        encoded_out = self.resnet(image)
        encoded_out = encoded_out.view(encoded_out.size(0), encoded_out.size(1), -1)
        encoded_out = encoded_out.permute(0, 2, 1)
        return encoded_out


# todo: 加入位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x, pos = None):
        if pos is None:
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len]
            return x
        else:
            x = x + self.pe[:, pos]
            return x


class Decoder_transformer(nn.Module):
    def __init__(self, tokenizer, config) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.output_size = tokenizer.vocab_size

        self.embed_size = 256

        self.input_size = 64
        self.pe_encoder = PositionalEncoding(d_model = self.embed_size, max_len = 150)

        self.num_layer = 2
        self.num_head = 2
        self.max_length = config.max_length

        self.word_size = 100
        self.embedding_net = nn.Linear(self.word_size, self.embed_size)
        self.pe = PositionalEncoding(d_model = self.embed_size, max_len = self.max_length)

        self.input_to_encoded = nn.Linear(self.input_size, self.embed_size)

        decoder_layers = nn.TransformerDecoderLayer(d_model = self.embed_size, nhead = self.num_head,
                                                    batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers = self.num_layer)

        # output_net
        self.output_net = nn.Linear(self.embed_size, self.output_size)

    # no mask
    def forward(self, encoder_out, encoded_captions = None, caption_lengths = None):
        bs = encoder_out.shape[0]
        device = encoder_out.device
        # print(f'debug: device in train {device}')

        # sort by the length for the ease of paralleling
        # print(f'debug: captions length:{caption_lengths}')
        caption_lengths, sort_idx = torch.sort(caption_lengths, dim = -1, descending = True)
        # print(f'debug: after captions length:{caption_lengths}')

        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]
        # print(f'debug: enout:{encoder_out.shape}')
        decoder_memory = self.input_to_encoded(encoder_out)  # bs, 49, embed_size

        # 位置编码
        decoder_memory = self.pe_encoder(decoder_memory)

        # print(f'debug: decoder_memory:{decoder_memory.shape}')
        scale = math.sqrt(self.output_size)
        # print(f'debug: h:{h.shape}, c:{c.shape}')
        assert not torch.any(torch.isnan(encoded_captions)), "Nan happen in en_cap!!!"
        embeded_captions = self.embedding_net(encoded_captions[:, :-1])
        assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in em_cap1!!!"
        embeded_captions = self.pe(embeded_captions)
        assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in em_cap2!!!"

        decoder_length = (caption_lengths - 1)
        # print(f'debug: decoder length:{decoder_length}')

        predicts = (torch.ones((bs, max(decoder_length), self.output_size)) * -1).to(device)

        for t in range(max(decoder_length)):
            working_index = sum([l > t for l in decoder_length])
            decoder_memory = decoder_memory[:working_index]
            decoder_tgt = embeded_captions[:working_index, :t + 1]
            # print(f'debug: decoder_tgt:{decoder_tgt.shape}')
            out_decoder = self.decoder(decoder_tgt, decoder_memory)  # bs, t+1, embed_size

            assert not torch.any(torch.isnan(decoder_memory)), "Nan happen in memory!!!"
            assert not torch.any(torch.isnan(decoder_tgt)), "Nan happen in tgt!!!"
            assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder!!!"

            out_decoder = self.output_net(out_decoder[:, -1])
            assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder1!!!"
            # print(f'debug: out_lstm:{out_lstm.shape}, predict:{predicts[:working_index, t].shape}')
            predicts[:working_index, t] = out_decoder

        # Generate a mask for parallel decoding
        # mask = nn.Transformer().generate_square_subsequent_mask(max(decoder_length)).to(device)
        # print(f'debug: mask:{mask.shape}')

        # out_decoder = self.decoder(embeded_captions, decoder_memory, tgt_mask=mask)  # bs, seq_len, embed_size
        # out_decoder = self.output_net(out_decoder)
        # print(f'debug: length:{max(decoder_length)}, out_decoder:{out_decoder.shape}')
        # predicts[:, :out_decoder.shape[1]] = out_decoder
        # assert not torch.any(torch.isnan(decoder_memory)), "Nan happen in memory!!!"
        # assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in tgt!!!"
        # assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder!!!"

        return predicts, encoded_captions, decoder_length, sort_idx

    # mask
    # def forward(self, encoder_out, encoded_captions = None, caption_lengths = None):
    #     bs = encoder_out.shape[0]
    #     device = encoder_out.device
    #     # print(f'debug: device in train {device}')

    #     # sort by the length for the ease of paralleling
    #     # print(f'debug: captions length:{caption_lengths}')
    #     caption_lengths, sort_idx = torch.sort(caption_lengths, dim=-1, descending=True)
    #     # print(f'debug: after captions length:{caption_lengths}')

    #     encoder_out = encoder_out[sort_idx]
    #     encoded_captions = encoded_captions[sort_idx].unsqueeze(-1)

    #     decoder_memory = self.input_to_encoded(encoder_out) # bs, 49, embed_size

    #     # print(f'debug: decoder_memory:{decoder_memory.shape}')

    #     # print(f'debug: h:{h.shape}, c:{c.shape}')
    #     assert not torch.any(torch.isnan(encoded_captions)), "Nan happen in en_cap!!!"
    #     embeded_captions = self.embedding_net(encoded_captions[:, :-1] / math.sqrt(self.output_size))
    #     assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in em_cap1!!!"
    #     embeded_captions = self.pe(embeded_captions)
    #     assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in em_cap2!!!"

    #     decoder_length = (caption_lengths - 1)
    #     # print(f'debug: decoder length:{decoder_length}')

    #     predicts = (torch.ones((bs, max(decoder_length), self.output_size)) * -1).to(device)

    #     # for t in range(max(decoder_length)):
    #     #     working_index = sum([l > t for l in decoder_length])
    #     #     decoder_memory = decoder_memory[:working_index]
    #     #     decoder_tgt = embeded_captions[:working_index, :t+1]
    #     #     # print(f'debug: decoder_tgt:{decoder_tgt.shape}')
    #     #     out_decoder = self.decoder(decoder_tgt, decoder_memory) # bs, t+1, embed_size

    #     #     assert not torch.any(torch.isnan(decoder_memory)), "Nan happen in memory!!!"
    #     #     assert not torch.any(torch.isnan(decoder_tgt)), "Nan happen in tgt!!!"
    #     #     assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder!!!"

    #     #     out_decoder = self.output_net(out_decoder[:, -1])
    #     #     assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder1!!!"
    #     #     # print(f'debug: out_lstm:{out_lstm.shape}, predict:{predicts[:working_index, t].shape}')
    #     #     predicts[:working_index, t] = out_decoder

    #     # Generate a mask for parallel decoding
    #     mask = nn.Transformer().generate_square_subsequent_mask(max(decoder_length)).to(device)
    #     print(f'debug: mask:{mask.shape}')

    #     out_decoder = self.decoder(embeded_captions, decoder_memory, tgt_mask=mask)  # bs, seq_len, embed_size
    #     out_decoder = self.output_net(out_decoder)
    #     print(f'debug: length:{max(decoder_length)}, out_decoder:{out_decoder.shape}')
    #     predicts[:, :out_decoder.shape[1]] = out_decoder
    #     assert not torch.any(torch.isnan(decoder_memory)), "Nan happen in memory!!!"
    #     assert not torch.any(torch.isnan(embeded_captions)), "Nan happen in tgt!!!"
    #     assert not torch.any(torch.isnan(out_decoder)), "Nan happen in out_decoder!!!"

    #     return predicts, encoded_captions, decoder_length, sort_idx

    # interface for test (generation)
    # todo: 在encode处除了个scale
    def generate(self, encoder_out, temperature = 0, top_p = 1):
        bs = encoder_out.shape[0]
        device = encoder_out.device

        scale = math.sqrt(self.output_size)

        working_index = torch.arange(bs).to(device)

        encoded_input = self.input_to_encoded(encoder_out)

        # 位置编码
        encoded_input = self.pe_encoder(encoded_input)

        # init selected tokens
        max_length = self.max_length
        selected_tokens = torch.zeros((bs, max_length), dtype = torch.long).to(device)
        selected_tokens[:, 0] = self.tokenizer.start_token_index

        selected_embeded_tokens = torch.zeros((bs, max_length, self.embed_size)).to(device)
        # print(f'dubug: sel: {self.tokenizer.word_vectors[selected_tokens[:,0].cpu()].shape}')
        selected_embeded_tokens[:, 0] = self.pe(
            self.embedding_net(torch.FloatTensor(self.tokenizer.word_vectors[selected_tokens[:, 0].cpu()]).to(device)),
            pos = 0)

        cur_len = 1
        while working_index.shape[0] > 0:
            decoder_memory = encoded_input[working_index]
            decoder_tgt = selected_embeded_tokens[working_index, :cur_len]
            # print(f'debug: memo:{decoder_memory.shape}, tgt:{decoder_tgt.shape}')
            out_decoder = self.decoder(decoder_tgt, decoder_memory)

            out_decoder = self.output_net(out_decoder[:, -1])
            if temperature > 0:
                probs = torch.softmax(out_decoder / temperature, dim = -1)
                next_token = sample_top_p(probs, top_p).squeeze_(-1)
                # print(f'debug: next_token:{next_token.shape}')
            else:
                next_token = torch.argmax(out_decoder, dim = -1)
            selected_tokens[working_index, cur_len] = next_token
            selected_embeded_tokens[working_index, cur_len, :] = self.pe(
                self.embedding_net(torch.FloatTensor(self.tokenizer.word_vectors[next_token.cpu()]).to(device)),
                pos = cur_len)
            # print(f'debug: se:{selected_tokens.shape}, seem:{selected_embeded_tokens.shape}')

            not_end_index = self.tokenizer.not_end(next_token)
            # print(f'debug: index:{not_end_index}')
            working_index = working_index[not_end_index]
            # print(f'debug: widx:{working_index}')
            cur_len += 1
            if cur_len == max_length - 1:
                selected_tokens[working_index, cur_len] = self.tokenizer.end_token_index
                break
        return selected_tokens.cpu().long().squeeze()

