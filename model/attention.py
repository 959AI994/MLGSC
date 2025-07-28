import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # 使用项目网络计算注意力权重
        w = self.project(z)
        # 对权重应用softmax函数，使其和为1。
        beta = torch.softmax(w, dim=1)
        # 使用注意力权重与原始输入进行加权，计算注意力加权和。
        # 返回加权和和注意力权重。
        return (beta * z).sum(1), beta