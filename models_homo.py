import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphConvolution_homo
from torch.nn.parameter import Parameter
import torch
import math
from torch_geometric.nn import SAGEConv  # 使用PyTorch Geometric的SAGEConv层

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x)

    def get_emb(self, x, adj):
        return F.relu(self.gc1(x, adj)).detach()

# 定义GraphSAGE编码器生成节点嵌入，随后再采样
class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Encoder, self).__init__()
        self.sage1 = SAGEConv(nfeat, nhid)
        self.sage2 = SAGEConv(nhid, nhid)  # 增加中间层
        self.sage3 = SAGEConv(nhid, nembed)  # 增加一个新的层
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

#定义一个简单的mlp解码特征维度
class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLPDecoder, self).__init__()

        # 第一层，input_dim 到 hidden_dim 的转换
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # 第二层，hidden_dim 到 hidden_dim // 2 的转换
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # Batch Normalization
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        # 输出层，hidden_dim // 2 到 output_dim 的转换
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        # Dropout
        self.dropout = dropout

    def forward(self, x):
        # 第一层 + BatchNorm + ReLU + Dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        # 第二层 + BatchNorm + ReLU + Dropout
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        # 输出层
        x = self.fc3(x)
        return x


class GCN_homo(nn.Module):
    def __init__(self, nfeat, adj, nhid, out, dropout, device):
        super(GCN_homo, self).__init__()
        self.device = device  # 保存设备信息
        self.gc1 = GraphConvolution_homo(nfeat, adj, nhid,device=device)
        self.gc2 = GraphConvolution_homo(nhid, adj, nhid,device=device)
        self.gc3 = GraphConvolution_homo(nhid, adj, out,device=device)
        self.dropout = dropout


    def forward(self, x, adj, bi_adj, output,eta=0.5,fusion_mode="equal"):
        x, mask = self.gc1(x, adj, bi_adj, output,eta=0.5)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training = self.training)
        # x_2 = F.relu(self.gc2(x, adj, bi_adj, output))
        # x_2 = F.dropout(x_2, self.dropout, training=self.training)
        x_3, mask = self.gc3(x, adj, bi_adj, output,eta=0.5)
        #return torch.cat((x, x_2), dim=1)
        return x_3, mask


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class MLP(nn.Module):
    def __init__(self, n_feat, n_hid, nclass, dropout):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid),  # 增加隐藏层
            nn.ReLU(),
            nn.Linear(n_hid, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.mlp(x)

    def get_emb(self, x):
        return self.mlp[0](x).detach()

class SimpleMLP(nn.Module):
    def __init__(self, n_feat, n_hid, n_embed):
        super(SimpleMLP, self).__init__()
        # 定义 MLP 的层
        self.layer1 = nn.Linear(n_feat, n_hid)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # 激活函数 ReLU
        self.layer2 = nn.Linear(n_hid, n_embed)  # 隐藏层到输出层

    def forward(self, x):
        x = self.layer1(x)  # 输入到第一个线性层
        x = self.relu(x)  # 应用 ReLU 激活函数
        x = self.layer2(x)  # 输出到投影层
        return x

class HLAGNN(nn.Module):
    def __init__(self, nfeat, adj, nclass, nhid1, nhid2, n_nodes, dropout, device):
        super(HLAGNN, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        self.adj = adj.clone().to(device)  # **初始时用原始邻接矩阵**
        self.GCN1 = GCN_homo(nfeat, self.adj, nhid1, nhid2, dropout, device=device)
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)).to(device))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        ).to(device)

    def update_adj(self, new_adj):
        """**动态更新邻接矩阵**"""
        self.adj = new_adj.clone().to(self.device)  # **更新 HLAGNN 的邻接矩阵**
        self.GCN1.adj = self.adj  # **更新 GCN 层的邻接矩阵**
        print(f"[DEBUG] Updated HLAGNN adj.shape={self.adj.shape}")  # Debug 信息

    def forward(self, x, adj, bi_adj, output,eta=0.5,fusion_mode="equal"):
        """前向传播，仍然使用 `GCN1` 里的 `adj`"""
        emb, adj_mask = self.GCN1(x, adj, bi_adj, output,eta=eta)
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        out = self.MLP(emb)
        return out, adj_mask, emb

