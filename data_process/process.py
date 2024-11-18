import os
import pickle
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
import numpy as np


def FMM_func(vocab_list: list[str], sentence: str) -> list[str]:
    """
    正向最大匹配（FMM）

    :param vocab_list: 词典列表
    :param sentence: 待分词的句子
    :return: 分词后的结果列表
    """
    max_len = len(max(vocab_list, key=len))  # 词典中最长词的长度
    start = 0
    token_list = []
    while start < len(sentence):
        index = min(start + max_len, len(sentence))  # 确保索引不越界
        for _ in range(max_len):
            token = sentence[start:index]
            if token in vocab_list or len(token) == 1:  # 单字或匹配词典
                token_list.append(token)
                start = index
                break
            index -= 1
    return token_list


def process_file(file_path: Path, image_dir: Path, image_suffix: str, vocab: list[str]) -> dict:
    """
    处理单个标签文件

    :param file_path: 标签文件路径
    :param image_dir: 图像文件目录
    :param image_suffix: 图像文件后缀
    :param vocab: 词汇表
    :return: 包含标签和图片信息的字典
    """
    ID = file_path.stem  # 文件名（去除后缀）

    empty_dict = {"ID": ID, "label": ""}

    # 跳过缺失的图片文件
    image_path = image_dir / f"{ID}{image_suffix}"
    if not image_path.exists():
        return empty_dict

    # 读取标签文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用 FMM 分词
    token_list = FMM_func(vocab, content)
    token_list = [token for token in token_list if token.strip()]  # 移除空格和空字符串

    new_content = ' '.join(token_list)

    # 过滤无效标签
    if (
        any(token not in vocab for token in token_list if token.strip())
        or "error mathpix" in content
        or len(new_content.splitlines()) > 1
    ):
        return empty_dict

    # 返回正确的字典
    return {
        "ID": ID,
        "label": new_content,
        "image": np.array(Image.open(image_path)),
    }


def process_labels(input_label_dir: Path, output_dir: Path, image_dir: Path, vocab_file: Path):
    """
    主函数，处理所有标签文件并保存结果

    :param input_label_dir: 标签文件目录
    :param output_dir: 输出目录
    :param image_dir: 图像文件目录
    :param vocab_file: 词汇表文件路径
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取标签文件列表
    label_file_list = list(input_label_dir.iterdir())

    # 自动获取图像后缀
    image_suffix = next(image_dir.iterdir()).suffix

    # 读取词汇表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = f.read().split()

    dict_list_final = []
    counter_removed = 0
    counter_total = 0
    counter_batch = 0

    with Pool() as pool:
        dict_list = pool.imap_unordered(
            lambda file_path: process_file(file_path, image_dir, image_suffix, vocab),
            label_file_list,
            chunksize=10,
        )

        for label_image_dictionary in tqdm(dict_list, total=len(label_file_list)):
            counter_total += 1
            if label_image_dictionary["label"]:
                dict_list_final.append(label_image_dictionary)
                # 每处理 10,000 个标签，写入一次文件
                if len(dict_list_final) % 10000 == 0:
                    batch_file = output_dir / f"batch_{counter_batch}.pkl"
                    with open(batch_file, 'wb') as f:
                        pickle.dump(dict_list_final, f)
                    # np.save(batch_file, dict_list_final, allow_pickle=True)
                    print(f"写入部分数据到: {batch_file}")
                    counter_batch += 1
                    dict_list_final = []
            else:
                counter_removed += 1

        # 写入剩余数据
        if dict_list_final:
            batch_file = output_dir / f"batch_{counter_batch}.pkl"
            with open(batch_file, 'wb') as f:
                pickle.dump(dict_list_final, f)
            # np.save(batch_file, dict_list_final, allow_pickle=True)

        print(
            f"""
            总结:
            - 原始标签数量: {counter_total}
            - 被移除的标签数量: {counter_removed}
            - 有效标签数量: {counter_total - counter_removed}
            - 输出路径: {output_dir}
            """
        )


if __name__ == '__main__':
    # 设置路径变量
    input_label_dir = Path('./data_process/original_dataset/labels')
    image_dir = Path('./data_process/original_dataset/images')
    output_dir = Path('./datasets//MyDataset')
    vocab_file = Path('./data_process/vocab.txt')

    # 调用主函数
    process_labels(input_label_dir, output_dir, image_dir, vocab_file)
