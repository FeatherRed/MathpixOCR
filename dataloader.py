import os
import pickle
import sys
import time
import random
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from gensim.models import Word2Vec
from tokenizer import Tokenizer


class PKLDataset(Dataset):
    def __init__(self, img_folder = None, all_data = None, transform = None):  # 两种构造方式
        """
        Args:
            pkl_file (str): 存储 .pkl 文件的目录路径。
            transform (callable, optional): 用于处理数据的转换函数。
        """
        self.img_folder = img_folder
        self.transform = transform
        self.all_data = all_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 数据索引。
        Returns:
            dict: 包含图像和标签的数据。
        """

        img_id = self.all_data[idx]['ID']
        img_path = os.path.join(self.img_folder, f'{img_id}.png')
        img = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform is not None:
            img = self.transform(img)

        label = self.all_data[idx]['label']

        # 给 label 加标记
        label = '<start>' + ' ' + label + ' ' + '<end>'
        label_length = len(label.split())

        return img, label, label_length

    # def __load_pkl(self):
    #     pkl_files = os.listdir(self.pkl_file)
    #     all_data = []
    #     T0 = time.perf_counter()
    #     for file_name in pkl_files:
    #         batch_file = os.path.join(self.pkl_file, file_name)
    #         with open(batch_file, 'rb') as f:
    #             data = pickle.load(f)
    #             all_data.extend(data)
    #     T1 = time.perf_counter()
    #     print(f"读取时间 {T1 - T0} s")
    #     return all_data

    def build_vocab(self):
        captions = ['<start> ' + data['label'] + ' <end>' for data in self.all_data]

        tokenized_captions = [caption.split() for caption in captions]
        self.max_length = max([len(caption) for caption in tokenized_captions])

        vocab = Word2Vec(tokenized_captions, vector_size = 100, window = 5, min_count = 1, workers = 4, seed = 42)

        return vocab

# only image
class TestDataset(Dataset):
    def __init__(self, img_folder, ids, transform = None):
        self.img_folder = img_folder
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.img_folder, f"{img_id}.png")
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img_id, img

# def create_dataloader(directory, batch_size = 32, shuffle = True, num_workers = 0, transform = None):
#     """
#     Args:
#         directory (str): 数据集目录路径。
#         batch_size (int): 批量大小。
#         shuffle (bool): 是否随机打乱数据。
#         num_workers (int): 使用的子进程数。
#         transform (callable, optional): 数据预处理函数。
#     Returns:
#         DataLoader: 数据加载器。
#     """
#     dataset = PKLDataset(directory, transform = transform)
#     dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
#     return dataloader

def split_list(data, train_ratio = 0.9, seed = None):
    if seed is not None:
        random.seed(seed)

    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    train_size = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:train_size]
    valid_data = shuffled_data[train_size:]

    return train_data, valid_data


def load_pkl(pkl_dir):
    all_data = []
    pkl_files = os.listdir(pkl_dir)
    T0 = time.perf_counter()
    for file_name in pkl_files:
        batch_file = os.path.join(pkl_dir, file_name)
        with open(batch_file, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    T1 = time.perf_counter()
    print(f"读取时间 {T1 - T0} s")
    return all_data


def create_dataloader(img_dir, pkl_dir, train_ratio = 0.9, train_batch_size = 32, valid_batch_size = 1000,
                      transform = None):
    # todo config
    all_data = load_pkl(pkl_dir)
    # dataset = PKLDataset(img_folder = img_dir, pkl_file = pkl_dir, transform = transform)

    trainlist, validlist = split_list(all_data, train_ratio = train_ratio, seed = 42)
    trainset = PKLDataset(img_folder = img_dir, transform = transform, all_data = trainlist)
    validset = PKLDataset(img_folder = img_dir, transform = transform, all_data = validlist)
    vocab = trainset.build_vocab()
    # 创建数据加载器
    train_loader = DataLoader(trainset, batch_size = train_batch_size, shuffle = True)
    valid_loader = DataLoader(validset, batch_size = valid_batch_size, shuffle = False)

    return train_loader, valid_loader, vocab

def create_testloader(img_dir, ids, test_batch_size = 1000, transform = None):
    testset = TestDataset(img_dir, ids, transform)
    test_loader = DataLoader(testset, batch_size = test_batch_size, shuffle = False)
    return test_loader


if __name__ == '__main__':
    img_folder = "datasets/train/images"
    pkl_dir = "datasets/train/pkls"
    test_img_folder = 'datasets/test/images'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to a fixed size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalize image
    ])

    # train_loader, valid_loader, vocab = create_dataloader(img_folder, pkl_dir, transform = transform)
    # tokenizer = Tokenizer(vocab)
    test_id = [104011, 105060]
    ids = [i for i in range(test_id[0], test_id[1] + 1)] # 0 ~ 1049
    testset = TestDataset(test_img_folder, ids, transform)
    (id, img) = testset[1049]
    print(id)
    print(img.shape)

