import argparse
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models_homo import SimpleMLP
from sklearn.metrics import f1_score
import os
from config import Config
import time
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from torch import nn
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform

def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3,
                   edge_gen_strategy="proposed", k=5, debug_edge_count=False):

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    adj_new_rows = []
    same_class_edge_count = 0  # 统计同类节点的边数
    total_edge_count = 0  # 统计总生成的边数
    if len(labels) in [11701, 10312]:
        num_per_class_list = [torch.sum(labels == i).item() for i in range(c_largest + 1)]
        classes_to_sample = sorted(range(c_largest + 1), key=lambda x: num_per_class_list[x])[:im_class_num]
    else:
        classes_to_sample = range(c_largest - im_class_num + 1, c_largest + 1)

    original_idx_train = idx_train.clone()
    original_num_nodes = adj.shape[0] if adj is not None else embed.shape[0]

    # 只用原图嵌入来生成边，避免把新合成节点也纳入候选
    embed_orig = embed[:original_num_nodes, :].clone()

    total_added_nodes = 0

    for i in classes_to_sample:
        chosen = original_idx_train[(labels == i)[original_idx_train]]

        if chosen.shape[0] == 0:
            continue

        num = int(chosen.shape[0] * portion)

        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for _ in range(c_portion):
            chosen_batch = chosen[:num]

            if chosen_batch.shape[0] == 0:
                continue

            chosen_embed = embed_orig[chosen_batch, :]

            if chosen_batch.shape[0] == 1:
                idx_neighbor_local = np.array([0])
            else:
                distance = squareform(pdist(chosen_embed.cpu().detach()))
                np.fill_diagonal(distance, distance.max() + 100)
                idx_neighbor_local = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed_orig[chosen_batch, :] + \
                        (chosen_embed[idx_neighbor_local, :] - embed_orig[chosen_batch, :]) * interp_place

            # 先为这一批新节点生成邻接行
            new_rows_batch = []
            for local_k in range(len(chosen_batch)):
                chosen_node = chosen_batch[local_k]
                neighbor_node = chosen_batch[idx_neighbor_local[local_k]]

                if edge_gen_strategy == "proposed":
                    new_row,same_class_edge_count = generate_edges_proposed(
                        embed=embed,
                        labels=labels,
                        chosen_node=chosen_node,
                        neighbor_node=neighbor_node,
                        new_embed_k=new_embed[local_k],
                        adj=adj
                    )
                elif edge_gen_strategy == "full_connection":
                    new_row ,same_class_edge_count= generate_edges_full_connection(
                        chosen_node=chosen_node,
                        labels=labels,
                        neighbor_node=neighbor_node,
                        adj=adj
                    )
                else:
                    raise ValueError(f"Unknown edge generation strategy: {edge_gen_strategy}")

                new_rows_batch.append(new_row)

                # 计算同类节点连接的比例
                same_class_edge_count += same_class_edge_count
                total_edge_count += 1

                if debug_edge_count:
                    print(f"[{edge_gen_strategy}] new node in class {i}, local_k={local_k}, "
                          f"edge_num={int(new_row.sum().item())}")

            # 检查：这一批新节点数 == 新邻接行数
            assert len(new_rows_batch) == chosen_batch.shape[0], \
                f"len(new_rows_batch)={len(new_rows_batch)} != chosen_batch.shape[0]={chosen_batch.shape[0]}"

            # 再统一扩展 embed / labels / idx_train
            new_labels = labels.new(torch.Size((chosen_batch.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen_batch.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            adj_new_rows.extend(new_rows_batch)
            total_added_nodes += chosen_batch.shape[0]
    if adj is not None:
        actual_added_nodes = embed.shape[0] - original_num_nodes
        actual_adj_rows = len(adj_new_rows)

        #print(f"[DEBUG] original_num_nodes={original_num_nodes} "
              #f"actual_added_nodes={actual_added_nodes}, ",
              #f"adj_new_rows={actual_adj_rows}")

        assert actual_added_nodes == total_added_nodes, \
            f"embed新增节点数 {actual_added_nodes} != total_added_nodes {total_added_nodes}"

        assert actual_added_nodes == actual_adj_rows, \
            f"embed新增节点数 {actual_added_nodes} != adj_new_rows行数 {actual_adj_rows}"

        if actual_adj_rows == 0:
            return embed, labels, idx_train, adj

        adj_new = torch.stack(adj_new_rows, dim=0)  # [M, N]
        new_total_nodes = embed.shape[0]

        new_adj = adj.new(torch.Size((new_total_nodes, new_total_nodes))).fill_(0.0)
        new_adj[:original_num_nodes, :original_num_nodes] = adj
        new_adj[original_num_nodes:, :original_num_nodes] = adj_new
        new_adj[:original_num_nodes, original_num_nodes:] = adj_new.t()
        return embed, labels, idx_train, new_adj
    else:
        return embed, labels, idx_train


def generate_edges_proposed(embed,labels, chosen_node, neighbor_node, new_embed_k, adj):
    """
    1. 合成节点先连接两个源节点
    2. 再分别考察两个源节点的一阶邻居
    3. 若合成节点与某邻居的相似度 > 该邻域平均相似度，则连边
    返回：
        new_row: shape [N_original]
    """
    new_row = torch.zeros(adj.shape[1], device=adj.device)

    # 与两个源节点连接
    new_row[chosen_node] = 1.0
    new_row[neighbor_node] = 1.0

    # 源节点1的一阶邻居
    neighbors_chosen = adj[chosen_node, :].nonzero(as_tuple=True)[0]
    same_class_edge_count = 0  # 统计与合成节点同类的邻居节点数
    if neighbors_chosen.numel() > 0:
        similarity_chosen = F.cosine_similarity(
            new_embed_k.unsqueeze(0), embed[neighbors_chosen, :], dim=-1
        )
        mean_similarity_chosen = similarity_chosen.mean()
        for neighbor, sim in zip(neighbors_chosen, similarity_chosen):
            if sim > mean_similarity_chosen:
                new_row[neighbor] = 1.0

                # 判断这条边是否是同类节点之间的连接
                if labels[chosen_node] == labels[neighbor]:
                    same_class_edge_count += 1

    # 源节点2的一阶邻居
    neighbors_neighbor = adj[neighbor_node, :].nonzero(as_tuple=True)[0]
    if neighbors_neighbor.numel() > 0:
        similarity_neighbor = F.cosine_similarity(
            new_embed_k.unsqueeze(0), embed[neighbors_neighbor, :], dim=-1
        )
        mean_similarity_neighbor = similarity_neighbor.mean()
        for neighbor, sim in zip(neighbors_neighbor, similarity_neighbor):
            if sim > mean_similarity_neighbor:
                new_row[neighbor] = 1.0
                if labels[neighbor_node] == labels[neighbor]:
                    same_class_edge_count += 1


    return new_row,same_class_edge_count


def generate_edges_full_connection(chosen_node,labels, neighbor_node, adj):
    """
    Full Connection:
    合成节点连接到两个源节点 + 两个源节点的所有一阶邻居
    返回：
        new_row: shape [N_original]
    """
    new_row = torch.zeros(adj.shape[1], device=adj.device)
    same_class_edge_count = 0

    # 连接两个源节点本身
    new_row[chosen_node] = 1.0
    new_row[neighbor_node] = 1.0

    # 连接两个源节点的全部一阶邻居
    neighbors_chosen = adj[chosen_node, :].nonzero(as_tuple=True)[0]
    neighbors_neighbor = adj[neighbor_node, :].nonzero(as_tuple=True)[0]

    if neighbors_chosen.numel() > 0:
        new_row[neighbors_chosen] = 1.0
        # 判断这条边是否是同类节点之间的连接
        for neighbor in neighbors_chosen:
            if labels[chosen_node] == labels[neighbor]:
                same_class_edge_count += 1
    if neighbors_neighbor.numel() > 0:
        new_row[neighbors_neighbor] = 1.0
        # 判断这条边是否是同类节点之间的连接
        for neighbor in neighbors_chosen:
            if labels[chosen_node] == labels[neighbor]:
                same_class_edge_count += 1

    return new_row,same_class_edge_count
