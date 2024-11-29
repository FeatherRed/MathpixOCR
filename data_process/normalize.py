import torch
import numpy as np
import sys
sys.path.append("..")
from dataloader import *
def getStat(train_loader):
    print('Compute mean and variance for training data.')

    pixels = []
    for X, _, _ in train_loader:
        X = np.array(X) / 255.0
        pixels.append(X.reshape(-1, 3))
    means = np.mean(pixels, axis = 0)
    stds = np.std(pixels, axis = 0)
    return means, stds

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 假设你已经有一个 Dataloader: dataloader
# 如果没有，可以这样初始化：
# transform = transforms.Compose([transforms.ToTensor()])
# dataset = datasets.ImageFolder("path_to_your_dataset", transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

def calculate_mean_std(dataloader):
    """
    计算数据集的 mean 和 std
    Args:
        dataloader: 数据加载器，输出形状为 [1, 3, w, h]
    Returns:
        mean, std: 数据集的均值和标准差 (每通道)
    """
    mean = torch.zeros(3)  # 对于 RGB 图像
    std = torch.zeros(3)
    total_pixels = 0

    for batch in dataloader:
        # 提取图像数据（如果是 tuple，通常第一个是图像，第二个是标签）
        images = batch[0]  # 形状 [1, 3, w, h]
        # 移除 batch 维度并转为 [3, w * h]
        images = images.squeeze(0)  # [3, w, h]
        total_pixels += images.shape[1] * images.shape[2]  # 累积像素数
        mean += images.view(3, -1).mean(dim=1)  # 累加每通道的均值
        std += images.view(3, -1).std(dim=1)  # 累加每通道的标准差

    mean /= len(dataloader)  # 求全局均值
    std /= len(dataloader)  # 求全局标准差
    return mean, std

# 使用示例
# mean, std = calculate_mean_std(dataloader)
# print(f"Mean: {mean}, Std: {std}")


if __name__ == '__main__':
    pkl_dir = '../datasets/train/pkls'
    img_dir = '../datasets/train/images'
    all_data = load_pkl(pkl_dir)

    transform = transforms.Compose([
        # transforms.Resize((40, 240)),  # Resize image to a fixed size
        transforms.ToTensor(),  # Convert image to tensor
        # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalize image
    ])
    '''
    Compute mean and variance for training data.
([nan, 0.9090973, 0.9090973], [0.20551637, 0.20606166, 0.20590003])
    '''
    trainset = PKLDataset(img_folder = img_dir, all_data = all_data, transform = transform)
    train_loader = DataLoader(trainset, batch_size = 1, shuffle = True)
    print(calculate_mean_std(train_loader))

    '''
    (tensor([0.9087, 0.9083, 0.9103]), tensor([0.2206, 0.2212, 0.2211]))
    '''