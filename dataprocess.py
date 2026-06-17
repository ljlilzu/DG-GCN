import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
#from utils import normalize
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scipy.sparse as sp
import json
import os
import random
from collections import defaultdict
from sklearn.utils import shuffle
from itertools import chain
# 解析索引文件的函数
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
#处理cora/citeseer/pubmed数据集，不平衡设置
def process_data(dataset,im_class_num=3,im_ratio=0.1,):
    names = ['y', 'ty', 'ally','x', 'tx', 'allx','graph']
    objects = []
    for i in range(len(names)):
        with open("D:/PycharmProjects/SS-GNN-main/data/cache/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    y, ty, ally, x, tx, allx, graph = tuple(objects)
    print(graph)
    test_idx_reorder = parse_index_file("D:/PycharmProjects/SS-GNN-main/data/cache/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    print(features)
    f = open('D:/PycharmProjects/SS-GNN-main/data/{}/{}.adj'.format(dataset, dataset), 'w+')
    for i in range(len(graph)):
        adj_list = graph[i]
        for adj in adj_list:
            f.write(str(i) + '\t' + str(adj) + '\n')
    f.close()

    edge_file = open('D:/PycharmProjects/SS-GNN-main/data/{}/{}.edge'.format(dataset, dataset), 'w+')
    for i in range(len(graph)):
        adj_list = graph[i]
        for adj in adj_list:
            edge_file.write(f"{i} {adj}\n")
    edge_file.close()

    label_list = []
    for i in labels:
        label = np.where(i == np.max(i))[0][0]
        label_list.append(label)
    np.savetxt('D:/PycharmProjects/SS-GNN-main/data/{}/{}.label'.format(dataset, dataset), np.array(label_list), fmt='%d')
    np.savetxt('D:/PycharmProjects/SS-GNN-main/data/{}/{}.test'.format(dataset, dataset), np.array(test_idx_range), fmt='%d')
    np.savetxt('D:/PycharmProjects/SS-GNN-main/data/{}/{}.feature'.format(dataset, dataset), features, fmt='%f')


#随机划分数据集
import numpy as np
from sklearn.model_selection import train_test_split
import os
#对三个常规数据集进行划分，cora/citeseer/pubmed
def split_data(dataset, n_samples, n_splits=10):
    """
    根据给定的数据集名称和样本数量按指定比例生成10组 train.txt、val.txt 和 test.txt 文件。

    参数:
    - dataset: 数据集的名称，例如 'cora'。
    - n_samples: 数据集中样本的总数量。
    - n_splits: 需要生成的随机划分数量，默认为10。
    """
    # 生成一个包含所有索引的列表
    indices = np.arange(n_samples)
    # 按照 48%, 32%, 20% 的比例进行训练集、验证集和测试集划分
    train_size = 0.25
    val_size = 0.25
    test_size = 0.50
    # 指定保存路径
    save_dir = os.path.join('D:/PycharmProjects/SS-GNN-main/data_geom', dataset)
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，创建目录
    # 设置随机种子
    np.random.seed(random_seed)
    for split_id in range(n_splits):
        # 每次划分前重新设置随机状态
        random_state = random_seed + split_id
        # 首先划分测试集
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        # 然后从剩余的部分划分训练集和验证集
        val_ratio = val_size / (train_size + val_size)  # 在剩下的 80% 中，验证集占 32/80 = 0.4
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio, random_state=random_state
        )
        # 保存到文件，使用当前划分的编号作为前缀
        np.savetxt(os.path.join(save_dir, f'{split_id}train.txt'), train_indices, fmt='%d')
        np.savetxt(os.path.join(save_dir, f'{split_id}val.txt'), val_indices, fmt='%d')
        np.savetxt(os.path.join(save_dir, f'{split_id}test.txt'), test_indices, fmt='%d')

        print(f"Generated {split_id}train.txt, {split_id}val.txt, and {split_id}test.txt for dataset: {dataset}")

#对cora/citeseer/pubmed数据集进行划分(随机下采样）
import os
import random
import torch


def split_data_arti(dataset, im_ratio=0.5, n_splits=5, random_seed=42):
    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Define dataset-specific configurations
    if dataset == 'cora':
        num_classes = 7
        im_class_num = 3  # Last 3 classes as minority classes
    elif dataset == 'citeseer':
        num_classes = 6
        im_class_num = 3  # Last 3 classes as minority classes
    elif dataset == 'pubmed':
        num_classes = 3
        im_class_num = 2  # Last 2 classes as minority classes
    else:
        raise ValueError("Unknown dataset!")

    # Load labels from dataset
    label_path = f'D:/PycharmProjects/DG-GCN/data_geom/{dataset}'
    label_file = os.path.join(label_path, f'{dataset}.label')
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    with open(label_file, 'r') as file:
        labels = [int(line.strip()) for line in file.readlines()]
    labels = torch.LongTensor(labels)

    total_samples = len(labels)
    save_dir = os.path.join('D:/PycharmProjects/DG-GCN/data_geom', dataset)
    os.makedirs(save_dir, exist_ok=True)

    for split_id in range(n_splits):
        print(f"Creating split {split_id + 1}/{n_splits}...")
        # Step 1: Randomly shuffle and split the dataset
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)

        # Determine majority and minority classes
        minority_classes = list(range(num_classes - im_class_num, num_classes))
        majority_classes = list(set(range(num_classes)) - set(minority_classes))

        train_idx = []
        val_idx = []
        test_idx = []

        # Split data by class
        for cls in range(num_classes):
            class_indices = [idx for idx in all_indices if labels[idx] == cls]
            random.shuffle(class_indices)  # Shuffle class indices for random selection

            # Minority classes: Adjust training size according to im_ratio
            if cls in minority_classes:
                train_size = int(20 * im_ratio)
                train_idx += class_indices[:train_size]
                val_idx += class_indices[train_size:train_size + 25]
                test_idx += class_indices[train_size + 25:train_size + 80]
            else:
                # Majority classes: Fixed size of 20 samples for training
                train_idx += class_indices[:20]
                val_idx += class_indices[20:45]  # 25 samples for validation
                test_idx += class_indices[45:100]  # 55 samples for testing

        # Ensure that the training, validation, and test sets are randomized
        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        # Save splits to files
        with open(os.path.join(save_dir, f"{split_id}train.txt"), "w") as f_train:
            for idx in train_idx:
                f_train.write(f"{idx}\n")
        with open(os.path.join(save_dir, f"{split_id}val.txt"), "w") as f_val:
            for idx in val_idx:
                f_val.write(f"{idx}\n")
        with open(os.path.join(save_dir, f"{split_id}test.txt"), "w") as f_test:
            for idx in test_idx:
                f_test.write(f"{idx}\n")

        print(f"Split {split_id + 1}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

'''划分数据集'''
split_data_arti("cora",im_ratio=0.5, n_splits=3,random_seed=42)