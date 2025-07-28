import torch
import torch.nn as nn

# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


class CompGCN(nn.Module):
    def __init__(self, n_in, n_h, dropout=0.5):
        super(CompGCN, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.dropout = dropout

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff):
        h_1 = self.gcn1(seq1, adj)
        # h_1 = nn.functional.dropout(h_1)
        c_1 = torch.mean(h_1, dim=1)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff)
        c_2 = torch.mean(h_2, dim=1)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj)
        h_4 = self.gcn2(seq2, diff)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse):
        h_1 = self.gcn1(seq, adj, sparse)
        c = torch.mean(h_1, dim=1)

        h_2 = self.gcn2(seq, diff, sparse)
        return h_1 + h_2, c

# 这个类就起到一个最后分类的作用
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret