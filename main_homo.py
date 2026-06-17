from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
import itertools
import csv
from utils import *
from oversampling import recon_upsample
from models_homo import Encoder, MLP, MLPDecoder, HLAGNN
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import os
import argparse
import math
from config import Config
import time
import oversampling
from torch.utils.data import DataLoader, TensorDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 过采样..........

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

def train(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    adj, sadj = load_graph_homo(config)
    features, labels, idx_train, idx_val, idx_test = load_data(config)
    adj = torch.tensor(adj.todense(), dtype=torch.float32).to(device)
    features, labels,idx_train, idx_val, idx_test = features.to(device), labels.to(device), idx_train.to(device), idx_val.to(device), idx_test.to(device)
    # 如果用GraphSAGE，则需要将邻接矩阵转换为边索引
    edge_index = adj.nonzero().clone().detach().t().contiguous()
    # 根据类别数计算每个类别的权重
    # 计算每个类别的样本数
    class_counts = torch.bincount(labels[idx_train])
    class_weights = 1.0 / (class_counts.float() + 1e-6)  # 使用样本数的倒数作为权重
    class_weights = class_weights / class_weights.sum()  # 归一化权重
    class_weights = class_weights.to(device)  # 确保权重和数据在同一设备上

    weight = features.new((labels.max().item() + 1)).fill_(1)
    weight[-im_class_num:] = 1 + args.portion
    # 1.生成嵌入
    model_Encoder = Encoder(nfeat=config.fdim,
                            nhid=config.nhid1,
                            nembed=config.nhid2,
                            dropout=config.dropout).to(device)
    optimizer_encoder = optim.Adam(model_Encoder.parameters(),
                                   lr=0.001, weight_decay=0.0005)
    # 2.还原特征
    model_Decoder = MLPDecoder(input_dim=config.nhid2,
                               hidden_dim=config.nhid1,
                               output_dim=config.fdim,
                               dropout=config.dropout).to(device)
    optimizer_decoder = optim.Adam(model_Decoder.parameters(),
                                   lr=0.001, weight_decay=0.0005)

    model_MLP = MLP(n_feat=config.fdim,
                    n_hid=config.nhid2,
                    nclass=config.class_num,
                    dropout=config.dropout).to(device)

    optimizer_mlp = optim.Adam(model_MLP.parameters(),
                               lr=0.001,
                               weight_decay=0.0002)
    # 增加计时
    start = time.time()

    # 1.预训练Encoder和Decoder，确保生成的嵌入表示和解码后的特征表示有效
    for i in range(50):
        model_Encoder.train()
        model_Decoder.train()
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        embed = model_Encoder(features, edge_index)
        features_new = model_Decoder(embed)
        # 使用特征重构损失来预训练
        loss_recon = F.mse_loss(features_new, features)  # 重构损失
        loss_recon.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()
        #print(f"Pre-training epoch {i + 1}, reconstruction loss: {loss_recon.item():.4f}")
    # 生成并更新采样节点的特征，使用 recon_feat_loss 进行优化
    embed = model_Encoder(features, edge_index)
    # 采样得到的平衡图
    # 使用不同的边生成策略进行过采样
    embed, labels_new, idx_train_new, adj_new = oversampling.recon_upsample(embed, labels, idx_train,
                                                                            adj=adj.detach().to_dense(),
                                                                            portion=args.portion,
                                                                            im_class_num=im_class_num,
                                                                            edge_gen_strategy=edge_gen_strategy, k=k)

    # 获取解码器还原的所有节点特征（包括原始节点和新合成节点）
    features_new_from_decoder = model_Decoder(embed)  # 解码器还原的所有 2738 个节点的特征矩阵

    # 创建 features_new：拼接原始特征与新合成节点的特征
    features_new = torch.cat((features, features_new_from_decoder[config.n_nodes:, :]), dim=0)
    #print(features_new.shape)
    #print(adj_new.shape)
    # 初始化最佳验证和测试准确率
    mlp_acc_val_best = 0
    mlp_best_test = 0

    # 预训练soft labels，训练MLP软标签
    for i in range(30):
        model_MLP.train()
        optimizer_mlp.zero_grad()  # 梯度清零
        output = model_MLP(features_new)
        loss = F.nll_loss(output[idx_train_new], labels_new[idx_train_new],weight=class_weights)
        acc = accuracy(output[idx_train_new], labels_new[idx_train_new])
        loss.backward(retain_graph=True)
        optimizer_mlp.step()
        model_MLP.eval()  # 评估模式
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        if acc_val > mlp_acc_val_best:
            mlp_acc_val_best = acc_val
            mlp_best_test = acc_test
        print(f"epoch: {i + 1:4d}, loss: {loss.item(): .4f}, acc: train={acc.item(): .4f}, "f"valid={acc_val.item(): .4f}  test={acc_test.item(): .4f}")
    print(f"best acc={mlp_best_test:.4f}")

    # 测试1-hop neighbors
    # bi_adj = si_adj

    # 初始化
    feat_dim = config.fdim
    idx_train_unmasked = idx_train_new
    feat_new = features_new  # 存储特征矩阵副本

    # 4.最终GCN的定义模型
    model_HLAGNN = HLAGNN(nfeat=config.fdim,
                          adj=adj_new,
                          nhid1=config.nhid1,
                          nhid2=config.nhid2,
                          nclass=config.class_num,
                          n_nodes=features_new.shape[0],
                          dropout=config.dropout,
                          device=device).to(device)

    optimizer_HLAGNN = optim.Adam(model_HLAGNN.parameters(),
                                  lr=config.lr,
                                  weight_decay=config.weight_decay)

    best_acc_val_HLAGNN = 0
    best_f1 = 0
    best = 0  # 整个训练过程的测试准确率
    best_test = 0  # 验证集上最好的测试准确率
    best_bacc_test = 0  # 测试集上最好的balanced accuracy
    best_f1_test = 0  # 测试集上最好的F1-score
    best_auc_test = 0  # 测试集上最好的AUC-ROC

    for i in range(config.epochs):
        torch.cuda.empty_cache()  # 每轮训练开始前释放缓存
        model_Encoder.train()
        model_Decoder.train()
        model_HLAGNN.train()
        model_MLP.train()

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        optimizer_HLAGNN.zero_grad()
        optimizer_mlp.zero_grad()
        embed = model_Encoder(features, edge_index)
        # features = (features - features.mean()) / (features.std() + 1e-6)  # 标准化嵌入特征
        # 使用不同的边生成策略进行过采样
        embed, labels_new, idx_train_new, adj_new = oversampling.recon_upsample(embed, labels, idx_train,
                                                                                adj=adj.detach().to_dense(),
                                                                                portion=args.portion,
                                                                                im_class_num=im_class_num,
                                                                                edge_gen_strategy=edge_gen_strategy,
                                                                                k=k)

        features_new_from_decoder = model_Decoder(embed)  # 解码器还原的所有 2738 个节点的特征矩阵
        # 创建 features_new：拼接原始特征与新合成节点的特征
        features_new = torch.cat((features, features_new_from_decoder[config.n_nodes:, :]), dim=0)
        si_adj = adj_new.clone()  # 对si_adj的修改不会影响adj
        bi_adj = adj_new.mm(adj_new)  # 计算adj的二阶邻接矩阵
        # 邻接矩阵改良，k-hop neighbors
        bi_adj = si_adj + bi_adj  # 公式（17）中的高阶
        bi_adj[bi_adj > 0] = 1


        # 输出soft label和预测结果
        output = model_MLP(features_new)
        out, adj_mask, emb = model_HLAGNN(features_new, si_adj, bi_adj, output)
        # 计算loss
        loss_mlp = F.nll_loss(output[idx_train_unmasked], labels_new[idx_train_unmasked])
        loss_recon = F.mse_loss(features, features_new_from_decoder[:config.n_nodes, :])  # Feature reconstruction loss
        loss_gnn = F.nll_loss(out[idx_train_unmasked], labels_new[idx_train_unmasked],weight=class_weights)
        loss =10*loss_recon + loss_mlp + loss_gnn

        acc = accuracy(out[idx_train_unmasked], labels_new[idx_train_unmasked])
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()
        optimizer_HLAGNN.step()
        optimizer_mlp.step()
        # 评估模型
        model_HLAGNN.eval()
        model_MLP.eval()

        # 计算在验证集和测试集上的准确率
        acc_val = accuracy(out[idx_val], labels[idx_val])
        acc_test = accuracy(out[idx_test], labels[idx_test])
        # 计算并保存验证集上的预测结果
        y_true_val = labels[idx_val].cpu().numpy()
        y_pred_val = out[idx_val].argmax(dim=1).cpu().numpy()

        y_true_test = labels[idx_test].cpu().numpy()
        y_pred_test = out[idx_test].argmax(dim=1).cpu().numpy()
        # 计算bAcc
        bacc_val = balanced_accuracy_score(y_true_val, y_pred_val)
        bacc_test = balanced_accuracy_score(y_true_test, y_pred_test)
        # 计算F1-score
        f1_val = f1_score(y_true_val, y_pred_val, average='macro')
        f1_test = f1_score(y_true_test, y_pred_test, average='macro')
        # 计算AUC-ROC
        y_pred_prob_val = F.softmax(out[idx_val], dim=1).detach().cpu().numpy()
        y_pred_prob_test = F.softmax(out[idx_test], dim=1).detach().cpu().numpy()
        # 计算每个类的 AUC，针对多类问题，`average='macro'` 会计算每个类的 AUC 并取平均
        auc_val = roc_auc_score(y_true_val, y_pred_prob_val, multi_class='ovr', average='macro')
        auc_test = roc_auc_score(y_true_test, y_pred_prob_test, multi_class='ovr', average='macro')
        if acc_val > best_acc_val_HLAGNN:
            best_acc_val_HLAGNN = acc_val
            best_test = acc_test
            best_bacc_test = bacc_test
            best_f1_test = f1_test
            best_auc_test = auc_test
        if best < acc_test:
            best = acc_test

        print(f"epoch: {i + 1:4d}, loss: {loss.item() - loss_mlp.item(): .4f}, acc: train={acc.item(): .4f} "
              f"valid={acc_val.item(): .4f}  test={acc_test.item(): .4f}")

    end = time.time()

    print(f"Elapsed Time={end - start}(s)")
    print("Best accuracy:", best_test.item())

    return best_test.item() * 100, best_bacc_test * 100, best_f1_test * 100, best_auc_test * 100

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset",
                       default="cora", type=str, required=False)
    parse.add_argument("-m", "--mask_rate",
                       help="masked labeled data for training", default=0.5, type=float, required=False)

    parse.add_argument("--use_labels",
                       help="use labels for propagation, default: false", action="store_true", required=False)
    parse.add_argument("--label_reuse",
                       help="reuse soft labels for propagation, default: false", action="store_true", required=False)

    parse.add_argument("-p", "--portion", help="portion of new nodes",
                       default=1.0, type=float, required=False)
    args = parse.parse_args()
    portion = args.portion
    config_file = "./config/" + str(args.dataset) + ".ini"
    config = Config(config_file)

    # 乱七八糟，到处都有flag配置，后面需要统一
    cuda = True
    use_seed = True  # True

    device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    # 固定少数类
    if args.dataset in ["cora", "citeseer"]:
        im_class_num = 3
    elif args.dataset =='blogcatalog':
        im_class_num = 14
    elif args.dataset == 'wiki-cs':
        im_class_num = 5
    else:
        print("no this dataset: {args.dataset}")
    results = []
    bacc_results = []
    f1_results = []
    auc_results = []
    for split_id in range(3):
        if use_seed:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if cuda:
                torch.cuda.manual_seed(config.seed)

        # 打印配置文件
        config.update_path(split_id)
        print(vars(config))
        best_test, best_bacc_test, best_f1_test, best_auc_test = train(args, config)
        results.append(best_test)
        bacc_results.append(best_bacc_test)
        f1_results.append(best_f1_test)
        auc_results.append(best_auc_test)

    print("Accuracy results:", results)
    print("F1-score results:", f1_results)
    print("AUC-ROC results:", f1_results)

    calc_mean_sd(results, metric_names=['accuracy'])
    calc_mean_sd(f1_results, metric_names=['f1_score'])
    calc_mean_sd(auc_results, metric_names=['auc_score'])


