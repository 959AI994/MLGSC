from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.gcn import GCN
from model.attention import Attention
from model.attentionPool import AttentionPool

class DualGCN(nn.Module):
    def __init__(self, config, num_classes) -> None:
        super().__init__()
        nhid1, nhid2, nout, dropout = 512, 1024, 512, 0.5      # 这几个参数一般不用动
        
        self.drop1 = config['drop1']
        
        self.drop2 = config['drop2']

        nfeat1 = config['num_pc'] * (2 * config['num_openings_closings'] + 1)
        nfeat2 = config['num_pc'] * (config['size'] ** 2)
        self.gcn1 = GCN(nfeat1, nhid1, nhid2, nout, dropout)
        self.gcn2 = GCN(nfeat2, nhid1, nhid2, nout, dropout)

        self.attention = Attention(nout)
    
        self.attention_pool = AttentionPool(nout)  

        self.alpha_raw = nn.Parameter(torch.Tensor(1))
        self.beta_raw = nn.Parameter(torch.Tensor(1))
        self.gamma_raw = nn.Parameter(torch.Tensor(1))

        nn.init.uniform_(self.alpha_raw, 0.30, 0.40)
        nn.init.uniform_(self.beta_raw, 0.35, 0.45)
        nn.init.uniform_(self.gamma_raw, 0.05, 0.10)

    def get_constrained_sigmas(self):
        # Apply sigmoid activation function
        sigmoid_alpha = torch.sigmoid(self.alpha_raw)
        sigmoid_beta = torch.sigmoid(self.beta_raw)
        sigmoid_gamma = torch.sigmoid(self.gamma_raw)

        # Scale to specified range without normalization
        # alpha, beta: 0.3 ~ 0.4
        alpha = 0.10 * sigmoid_alpha + 0.300
        beta = 0.10 * sigmoid_beta + 0.350
        # gamma: 0.052 ~ 0.067
        gamma = 0.05 * sigmoid_gamma + 0.050

        return alpha, beta, gamma

    def forward(self, emp_feat, spe_feat, adj1, adj2):
        '''
        ## 输入emb特征和空谱特征和两个邻接矩阵矩阵
        ---
        return: 增强图的卷积结果
        '''
        emb1 = self.gcn1(F.dropout(emp_feat, self.drop1), F.dropout(adj1, self.drop1))
        diff1 = self.gcn1(F.dropout(emp_feat, self.drop2), F.dropout(adj1, self.drop2))

        emb2 = self.gcn2(F.dropout(spe_feat, self.drop1), F.dropout(adj2, self.drop1))
        diff2 = self.gcn2(F.dropout(spe_feat, self.drop2), F.dropout(adj2, self.drop2))

        pooled_emb1 = self.attention_pool(emb1)
        pooled_emb2 = self.attention_pool(emb2)


        return emb1, emb2, diff1, diff2, pooled_emb1, pooled_emb2
    

    def embs(self, emp_feat, spe_feat, adj1, adj2):
    
        emb1 = self.gcn1(emp_feat, adj1)

        emb2 = self.gcn2(spe_feat, adj2)

        embs = torch.stack([emb1, emb2], dim=1)
        embs, att = self.attention(embs)
        return embs

    def loss_func(self, emb1, emb2, diff1, diff2, pooled_emb1, pooled_emb2):

        alpha, beta, gamma = self.get_constrained_sigmas()

        Lnode1 = (self.gcn1.loss(emb1, diff1) + self.gcn2.loss(emb2, diff2)) / 2
        Lnode2 = (self.gcn1.loss(emb1, emb2) + self.gcn1.loss(diff1, diff2)) / 2
        Lgraph = self.gcn1.graph_contrastive_loss(pooled_emb1, pooled_emb2)

        total_loss = alpha * Lnode1 + beta * Lnode2 + gamma * Lgraph

        return total_loss


