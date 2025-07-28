import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, node_features_dim):
        super(AttentionPool, self).__init__()
        # 使用一个线性层来计算注意力分数
        self.attention_weights = nn.Linear(node_features_dim, 1)

    def forward(self, node_features):
        # 计算注意力分数
        attention_scores = self.attention_weights(node_features)
        attention_scores = F.softmax(attention_scores, dim=0)

        # 加权聚合节点特征
        graph_embedding = torch.sum(attention_scores * node_features, dim=0)
        return graph_embedding
