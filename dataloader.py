import os
import pickle
import sys
import time

from torch.utils.data import Dataset, DataLoader
import torch


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
        file_path = self.file_list[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 假设数据包含 'image' 和 'label' 两部分
        image = data['image']
        label = data['label']

        if self.transform:
            image = self.transform(image)

        return {"image": torch.tensor(image, dtype = torch.float32),
                "label": torch.tensor(label, dtype = torch.long)}

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

    pkl_dir = "data_process/MyDataset"
    Mydataloader = PKLDataset(pkl_dir)
    print(len(Mydataloader))
