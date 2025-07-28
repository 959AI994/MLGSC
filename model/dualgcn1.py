from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.gcn import GCN
from model.attention import Attention

class DualGCN(nn.Module):
    def __init__(self, config, num_classes) -> None:
        super().__init__()
        nhid1, nhid2, nout, dropout = 512, 1024, 512, 0.5
        self.drop1 = config['drop1']
        # 从配置中获取参数 drop2 的值
        self.drop2 = config['drop2']
        # 这里仅保留nfeat2的计算
        nfeat2 = config['num_pc'] * (config['size'] ** 2)

        # 原始编码器
        self.gcn = GCN(nfeat2, nhid1, nhid2, nout, dropout)
        # 新建一个动量编码器，目的是实现MoCo模型
        self.momentum_gcn = deepcopy(self.gcn)

        # 加入注意力机制
        self.attention = Attention(nout)


    # 更新动量编码器的权重
    def update_momentum_encoder(self, beta=0.999):
        for param_q, param_k in zip(self.gcn.parameters(), self.momentum_gcn.parameters()):
            param_k.data = param_k.data * beta + param_q.data * (1. - beta)

    # 数据增强的几种方式
    @staticmethod
    # 采用新方式进行数据增强
    def add_random_edges(adj, add_ratio=0.5):
        num_nodes = adj.size(0)
        num_add = int(num_nodes * add_ratio)

        non_edges = torch.nonzero(adj == 0).numpy()
        selected_edges = non_edges[np.random.choice(non_edges.shape[0], num_add, replace=False)]

        adj_added = adj.clone()
        adj_added[selected_edges[:, 0], selected_edges[:, 1]] = 1
        return adj_added

    @staticmethod
    def drop_random_edges(adj, drop_ratio=0.5):
        num_nodes = adj.size(0)
        num_drop = int(num_nodes * drop_ratio)

        edges = torch.nonzero(adj == 1).numpy()
        selected_edges = edges[np.random.choice(edges.shape[0], num_drop, replace=False)]

        adj_dropped = adj.clone()
        adj_dropped[selected_edges[:, 0], selected_edges[:, 1]] = 0
        return adj_dropped

    @staticmethod
    def drop_random_nodes(adj, drop_ratio=0.2):
        """
        Randomly drop nodes and their associated edges.
        """
        num_nodes = adj.size(0)
        num_drop = int(num_nodes * drop_ratio)

        drop_indices = np.random.choice(num_nodes, num_drop, replace=False)
        keep_indices = list(set(range(num_nodes)) - set(drop_indices))

        adj_dropped = adj[keep_indices, :][:, keep_indices]

        return adj_dropped, keep_indices

    @staticmethod
    def subgraph_sampling(adj, sampling_ratio=0.8):
        """
        Randomly sample a subgraph.
        """
        num_nodes = adj.size(0)
        num_sample = int(num_nodes * sampling_ratio)

        sampled_indices = np.random.choice(num_nodes, num_sample, replace=False)
        subgraph_adj = adj[sampled_indices, :][:, sampled_indices]

        return subgraph_adj, sampled_indices

    def forward(self, spe_feat, adj2):
        '''
        ## 输入空谱特征和邻接矩阵
        ---
        return: 增强图的卷积结果
        '''
        emb = self.gcn(F.dropout(spe_feat, self.drop1), F.dropout(adj2, self.drop1))
  
        diff = self.momentum_gcn(F.dropout(spe_feat, self.drop2), F.dropout(adj2, self.drop2))

        return emb, diff

    # def forward(self, spe_feat, adj2):
    #     '''
    #     输入空谱特征和邻接矩阵
    #     ---
    #     return: 增强图的卷积结果
    #     '''
    #     adj_view1 = self.add_random_edges(adj2.clone())  # 使用clone确保原始邻接矩阵不被修改
    #     adj_view1 = self.drop_random_edges(adj_view1)
    #
    #     if self.training:  # Only apply data augmentation during training
    #         choice = np.random.choice([0, 1])
    #         if choice == 0:
    #             adj_view2, indices = self.drop_random_nodes(adj2.clone())
    #             spe_feat_view2 = spe_feat[indices, :]
    #         else:
    #             adj_view2, indices = self.subgraph_sampling(adj2.clone())
    #             spe_feat_view2 = spe_feat[indices, :]
    #     else:
    #         adj_view2 = adj2
    #         spe_feat_view2 = spe_feat
    #   
    #     emb1 = self.gcn(F.dropout(spe_feat, self.drop1), F.dropout(adj_view1, self.drop1))
    #     diff1 = self.gcn(F.dropout(spe_feat, self.drop2), F.dropout(adj_view1, self.drop2))
    #
    #     emb2 = self.momentum_gcn(F.dropout(spe_feat_view2, self.drop1), F.dropout(adj_view2, self.drop1))
    #     diff2 = self.momentum_gcn(F.dropout(spe_feat_view2, self.drop2), F.dropout(adj_view2, self.drop2))
    #
    #     return emb1, emb2,diff1, diff2


    def embs(self, spe_feat, adj2):
        # 调用GCN层，得到嵌入结果
        emb = self.gcn(spe_feat, adj2)
        # 通过注意力层得到最终的嵌入表示
        embs, att = self.attention(emb.unsqueeze(dim=1))
        return embs

    def loss_func(self, emb, diff):
        return self.gcn.loss(emb, diff)

    # def loss_func(self, emb1, emb2, diff1, diff2):
    #     return ((self.gcn.loss(emb1, diff1) + self.momentum_gcn.loss(emb2, diff2)) / 2) # +self.gcn.contrastive_loss(emb1,emb2)

