import torch
import torch.nn as nn
import numpy as np

from model.gcn import GCN
from model.attention import Attention
from model.compgcn import CompGCN


class DualGCN(nn.Module):
    def __init__(self, config, num_classes) -> None:
        super().__init__()
        nout, dropout = 512, 0.5      # 这几个参数一般不用动
        # 两个分支的输入大小（nfeat）不同
        nfeat1 = config['num_pc'] * (2 * config['num_openings_closings'] + 1)
        nfeat2 = config['num_pc'] * (config['size'] ** 2)
        self.gcn1 = CompGCN(nfeat1, nout, dropout)
        self.gcn2 = CompGCN(nfeat2, nout, dropout)
        # self.gcn1 = GCN(nfeat1, nhid1, nhid2, nout, dropout)
        # self.gcn2 = GCN(nfeat2, nhid1, nhid2, nout, dropout)
        self.attention = Attention(nout)
        self.MLP = nn.Sequential(
            # nn.Linear(nout + nfeat1 + nfeat2, nout),
            nn.Linear(nout, num_classes),
            nn.LogSoftmax(dim=1)
        )       # 分类器

        self.adj1 = None
        self.adj2 = None
        self.diff1 = None
        self.diff2 = None

    def forward(self, emp_feat, spe_feat, adj1, adj2, diff1, diff2):
        '''
        ## 输入emb特征和空谱特征和四个矩阵
        ---
        return: 分类结果, embbing
        '''
        if self.adj1 is None:   # 需要对四个矩阵进行变形以匹配对比学习的模型
            adj1 = torch.from_numpy(adj1).float().to(emp_feat.device)
            adj2 = torch.from_numpy(adj2).float().to(emp_feat.device)
            diff1 = torch.from_numpy(diff1).float().to(emp_feat.device)
            diff2 = torch.from_numpy(diff2).float().to(emp_feat.device)
            self.adj1 = torch.unsqueeze(adj1, 0)
            self.adj2 = torch.unsqueeze(adj2, 0)
            self.diff1 = torch.unsqueeze(diff1, 0)
            self.diff2 = torch.unsqueeze(diff2, 0)
        emp_feat = torch.unsqueeze(emp_feat, 0)
        spe_feat = torch.unsqueeze(spe_feat, 0)
        idx = np.random.permutation(emp_feat.size(1))                        # 打乱feature
        logits1, __, __ = self.gcn1(emp_feat, emp_feat[:, idx, :], self.adj1, self.diff1)
        logits2, __, __ = self.gcn2(spe_feat, spe_feat[:, idx, :], self.adj2, self.diff2)
        emb1, __ = self.gcn1.embed(emp_feat, self.adj1, self.diff1, False)
        emb2, __ = self.gcn2.embed(spe_feat, self.adj2, self.diff2, False)

        emb = torch.stack([emb1[0], emb2[0]], dim=1)
        emb, att = self.attention(emb)

        # emb = torch.cat([emb, emp_feat, feat], dim=1)
        # output = self.MLP(emb)
        return None, emb, logits1, logits2

