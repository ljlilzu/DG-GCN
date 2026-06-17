import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.functional import cosine_similarity


# 实现了图卷积层，对输入的节点特征矩阵进行卷积操作，并通过邻接矩阵进行传播
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):  # in_features, out_features指定输入和输出的特征维度
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):  # input是节点特征矩阵，adj是邻接矩阵
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

#注意力机制融合特征和标签相似度矩阵
class AttentionMechanism(nn.Module):
    def __init__(self, n_nodes, device='cuda'):
        super(AttentionMechanism, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        self.q = nn.Parameter(torch.randn(1, n_nodes).to(device))
        self.W_L = nn.Parameter(torch.randn(n_nodes, n_nodes).to(device))
        self.b_L = nn.Parameter(torch.randn(n_nodes,n_nodes).to(device))
        self.W_H = nn.Parameter(torch.randn(n_nodes, n_nodes).to(device))
        self.b_H = nn.Parameter(torch.randn(n_nodes,n_nodes).to(device))

    def forward(self, L, H):
        # 计算 \omega_L^i
        omega_L = self.q @ torch.tanh(self.W_L @ L.t() + self.b_L)
        # 计算 \omega_H^i
        omega_H = self.q @ torch.tanh(self.W_H @ H.t() + self.b_H)

        # 计算 softmax
        alpha_L = F.softmax(omega_L, dim=1)
        alpha_H = F.softmax(omega_H, dim=1)

        # 组合矩阵
        S = alpha_L * L + alpha_H * H
        return S
#自适应融合特征和标签相似度矩阵
class AdaptiveFeatureFusing(nn.Module):
    def __init__(self, d_feat, d_sim,device='cuda'):
        super(AdaptiveFeatureFusing, self).__init__()
        self.d_feat = d_feat  # Dimension of feat_matrix
        self.d_sim = d_sim  # Dimension of sim_matrix
        self.device = device
        self.linear_layer = nn.Linear(d_feat + d_sim, 2).to(device)

    def forward(self, feat_matrix, sim_matrix):
        feat_matrix = feat_matrix.to(self.device)
        sim_matrix = sim_matrix.to(self.device)
        # Concatenate the feature matrices
        concatenated_features = torch.cat((feat_matrix, sim_matrix), dim=1)

        # Apply the learnable weight layer
        weights = self.linear_layer(concatenated_features)
        weights = F.softmax(weights, dim=1)

        # Compute the final representation
        H_F = weights[:, 0].unsqueeze(1) * feat_matrix + weights[:, 1].unsqueeze(1) * sim_matrix

        return H_F


class GraphConvolution_homo(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, adj, out_features, bias=True,device='cuda',dropout=0.5):
        super(GraphConvolution_homo, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = torch.nn.Linear(in_features, out_features)  # 线性层，目的是在嵌入空间计算特征相似度，和标签相似度在一个量纲
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_bi = Parameter(
            torch.FloatTensor(in_features, out_features))
        self.w = Parameter(torch.FloatTensor(1))
        # 定义可学习的融合系数 alpha
        self.alpha = Parameter(torch.FloatTensor(1).to(device))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # 这是论文中公式11的T
        self.register_buffer("LPA_weight", adj.clone().to(device))
        self.register_buffer("identity", torch.eye(adj.size(0)).to(device))
        # 确保 adj 是最新的
        self.adj = adj.clone().to(self.device)
        self.n_nodes = self.adj.size(0)  # 强制更新 n_nodes
        self.attention = AttentionMechanism(n_nodes=self.n_nodes, device=self.device)
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv_bi = 1. / math.sqrt(self.weight_bi.size(1))
        self.weight_bi.data.uniform_(-stdv_bi, stdv_bi)
        self.w.data.uniform_(0.5, 1)
        #self.alpha.data.fill_(0.5)  # 初始化 alpha 为 0.5
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, bi_adj, output,eta=0.5,fusion_mode="equal"):
        """
            input: node feature matrix
            adj: original adjacency matrix
            bi_adj: A_k
            output: soft labels
            fusion_mode:
            - "sl_only":     S = S^L
            - "sf_only":     S = S^F
            - "weighted":    S = eta * S^L + (1-eta) * S^F
            - "equal":       S = 0.5 * (S^L + S^F)

        """
        input = input.to(self.device)
        adj = adj.to(self.device)
        bi_adj = bi_adj.to(self.device)
        output = output.to(self.device)
        # 特征相似度矩阵S^F
        input_norm = F.softmax(input, dim=1)
        feat_matrix = torch.matmul(input_norm, input_norm.t())

        # output是MLP的输出，下面两行对应公式（4）
        output = output.exp()
        # 求出, 即标签相似度S^L
        sim_matrix = torch.matmul(output, output.t())
        if fusion_mode =="sl_only":
            fused_matrix =sim_matrix
        elif fusion_mode =="sf_only":
            fused_matrix = feat_matrix
        elif fusion_mode =="weighted":
            fl_fuser = AdaptiveFeatureFusing(feat_matrix.size(1), sim_matrix.size(1), device=self.device)
            fused_matrix = fl_fuser(feat_matrix, sim_matrix)
        elif fusion_mode == "equal":
            fused_matrix = torch.add(feat_matrix, sim_matrix)*0.5
        #注释掉下面这行，即去掉S，同等权重聚合所有邻居
        bi_adj = torch.mul(bi_adj.to(self.device), fused_matrix)  # 只加权相似度高的邻居
        #print(bi_adj)
        # 公式（18）\hat{D}^{-1}
        with torch.no_grad():
            bi_row_sum = torch.sum(bi_adj, dim=1, keepdim=True)
            # np.power(rowsum, -1).flatten()
            bi_r_inv = torch.pow(bi_row_sum, -1).flatten()
            bi_r_inv[torch.isinf(bi_r_inv)] = 0.  # 防止出现inf
            bi_r_mat_inv = torch.diag(bi_r_inv)

        # 公式（18），计算\hat{D}^{-1}A_k*H
        bi_adj = torch.matmul(bi_r_mat_inv, bi_adj)

        # 公式（18）Z^(l-1)W_e^{l}
        support = torch.mm(input, self.weight)

        # 公式（10）Z^(l-1)W_n^{l}
        support_bi = torch.mm(input, self.weight_bi)

        # 目标节点表示（下为特征矩阵）
        identity = torch.eye(adj.shape[0]).to(self.device)
        output = torch.spmm(identity, support)

        # 邻居节点表示（下为特征矩阵）
        output_bi = torch.spmm(bi_adj, support_bi)

        # 公式（18）求得Z^(l)
        output = output + torch.mul(self.w, output_bi)

        if self.bias is not None:
            return output + self.bias, sim_matrix
        else:
            return output, sim_matrix

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
