import os
import pickle
from torch.utils.data import Dataset, DataLoader
import torch


class PKLDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (str): 存储 .pkl 文件的目录路径。
            transform (callable, optional): 用于处理数据的转换函数。
        """
        self.directory = directory
        self.transform = transform
        self.file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 数据索引。
        Returns:
            dict: 包含图像和标签的数据。
        """
        file_path = self.file_list[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 假设数据包含 'image' 和 'label' 两部分
        image = data['image']
        label = data['label']

        if self.transform:
            image = self.transform(image)

        return {"image": torch.tensor(image, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long)}


def create_dataloader(directory, batch_size=32, shuffle=True, num_workers=0, transform=None):
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
    dataset = PKLDataset(directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader