import os
import pickle
import sys
import time
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from gensim.models import Word2Vec
from tokenizer import Tokenizer


class PKLDataset(Dataset):
    def __init__(self, img_folder, pkl_file, transform = None):
        """
        Args:
            pkl_file (str): 存储 .pkl 文件的目录路径。
            transform (callable, optional): 用于处理数据的转换函数。
        """
        self.img_folder = img_folder
        self.pkl_file = pkl_file
        self.transform = transform
        # self.file_list = [os.path.join(pkl_file, f) for f in os.listdir(pkl_file) if f.endswith('.pkl')]
        self.all_data = self.__load_pkl()

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

    def __load_pkl(self):
        pkl_files = os.listdir(self.pkl_file)
        all_data = []
        T0 = time.perf_counter()
        for file_name in pkl_files:
            batch_file = os.path.join(self.pkl_file, file_name)
            with open(batch_file, 'rb') as f:
                data = pickle.load(f)
                all_data.extend(data)
        T1 = time.perf_counter()
        print(f"读取时间 {T1 - T0} s")
        return all_data

    def build_vocab(self):
        captions = ['<start> ' + data['label'] + ' <end>' for data in self.all_data]

        tokenized_captions = [caption.split() for caption in captions]
        self.max_length = max([len(caption) for caption in tokenized_captions])

        vocab = Word2Vec(tokenized_captions, vector_size = 100, window = 5, min_count = 1, workers = 4, seed = 42)

        return vocab


def create_dataloader(directory, batch_size = 32, shuffle = True, num_workers = 0, transform = None):
    """
    Args:
        directory (str): 数据集目录路径。
        batch_size (int): 批量大小。
        shuffle (bool): 是否随机打乱数据。
        num_workers (int): 使用的子进程数。
        transform (callable, optional): 数据预处理函数。
    Returns:
        DataLoader: 数据加载器。
    """
    dataset = PKLDataset(directory, transform = transform)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return dataloader

def create_dataset(img_dir, pkl_dir, train_ratio = 0.9, batch_size = 32, transform = None):
    # todo config
    dataset = PKLDataset(img_folder = img_dir, pkl_file = pkl_dir, transform = transform)

    vocab = dataset.build_vocab()

    train_size = int(len(dataset) * train_ratio)
    valid_size = len(dataset) - train_size

    trainset, validset = random_split(dataset, [train_size, valid_size])

    # 创建数据加载器
    train_loader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(validset, batch_size = batch_size, shuffle = False)

    return train_loader, valid_loader, vocab


if __name__ == '__main__':
    # pkl_path = "data_process/MyDataset/batch_"
    # pkls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # all_data = []
    # T0 = time.perf_counter()
    # for i in pkls:
    #     with open(pkl_path + str(i) + ".pkl", 'rb') as f:
    #         data = pickle.load(f)
    #         all_data = all_data + data
    # T1 = time.perf_counter()
    # print((T1 - T0) * 1000)
    # memory_size = sys.getsizeof(all_data)
    # print(f"Memory size of data: {memory_size} bytes")
    '''
    data -> [list]
    
    '''
    img_folder = "data_process/Datasets/images"
    pkl_dir = "data_process/MyDataset"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to a fixed size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalize image
    ])

    train_loader, valid_loader, vocab = create_dataset(img_folder, pkl_dir, transform = transform)
    tokenizer = Tokenizer(vocab)
    for data in train_loader:
        (imgs, caps, caplens) = data
        encoded_caps, target_idxs = tokenizer.encode(caps)

        encoded_caps = torch.FloatTensor(encoded_caps)
        target_idxs = torch.tensor(target_idxs, dtype = torch.long)
        text = tokenizer.decode(target_idxs)
        print(text)

