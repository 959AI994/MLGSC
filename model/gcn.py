import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from model.attention import Attention

class GraphConvolution(Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
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

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nout, dropout, tau=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.bn1 = nn.BatchNorm1d(nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.bn2 = nn.BatchNorm1d(nhid2)
        self.gc3 = GraphConvolution(nhid2, nout)
        self.fc1 = torch.nn.Linear(nhid1, nhid2)
        self.fc2 = torch.nn.Linear(nhid2, nout)
        self.tau = tau
        self.dropout = dropout

        self.attention_pool = Attention(nout)  

    def forward(self, x, adj):
        x = self.gc1(x, adj)  
        x = F.relu(self.bn1(x))  
        x = self.gc2(x, adj)  
        x = F.relu(self.bn2(x))  
        x = F.dropout(x, self.dropout, training = self.training) 
        x = self.gc3(x, adj) 
        return x      
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:  
        z = F.elu(self.fc1(z)) 
        return self.fc2(z)  

    def graph_projection(self, z: torch.Tensor) -> torch.Tensor:
        # indian_pines数据集
        # fc1 = nn.Linear(4391, 1024)
        # PaviaU数据集
        fc1 = nn.Linear(4391, 1024)
        fc2 = nn.Linear(1024, 512)
        z = F.elu(fc1(z))  
        return fc2(z)  

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):  
        z1 = F.normalize(z1)  
        z2 = F.normalize(z2)  
        return torch.mm(z1, z2.t()) 

    def graph_sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=0)
        z2 = F.normalize(z2, dim=0)
        return torch.dot(z1, z2)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):  
        f = lambda x: torch.exp(x / self.tau)  
        refl_sim = f(self.sim(z1, z1))  
        between_sim = f(self.sim(z1, z2))  
      
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def graph_level_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):  
        f = lambda x: torch.exp(x / self.tau)  
        refl_sim = f(self.graph_sim(z1, z1))  
        between_sim = f(self.graph_sim(z1, z2))  
        return -torch.log(between_sim / (refl_sim + between_sim - 1))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor): 
          z1 = self.projection(z1)   
          z2 = self.projection(z2)   
          loss1 = self.semi_loss(z1, z2)  
          loss2 = self.semi_loss(z2, z1)  
          return (loss1 + loss2).mean() / 2   
  
    def pool(self, z: torch.Tensor)-> torch.Tensor:
        return z.mean(dim=1)

    def max_pool(self, z: torch.Tensor) -> torch.Tensor:
        return z.max(dim=1)[0]

    def attention_pool(self, z: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.attention_pool(z)
        return pooled
    
    def graph_contrastive_loss(self, pooled_emb1, pooled_emb2):
        graph_emb1 = pooled_emb1 
        graph_emb2 = pooled_emb2

        loss1 = self.graph_level_semi_loss(graph_emb1, graph_emb2)  
        loss2 = self.graph_level_semi_loss(graph_emb2, graph_emb1)  

        return (loss1 + loss2).mean() / 2  

    def contrastive_loss(self, emb1, emb2):

        return self.semi_loss(emb1, emb2)
    