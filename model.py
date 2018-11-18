import torch.nn as nn
import torch
import numpy as np


class Net(nn.Module):
    def __init__(self, vocab_sz, dim):
        super(Net, self).__init__()
        self.fe = nn.Embedding(vocab_sz, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        fe_a = self.fe(xs[:, 0])
        fe_b = self.fe(xs[:, 1])

        dis_dot = self.batch_dot(fe_a, fe_b).view(xs.size()[0])
        dis_cos = dis_dot / self.norm(fe_a) / self.norm(fe_b)
        dis = (self.sigmoid(dis_cos) + 1) / 2

        return dis

    def norm(self, a):
        return (a ** 2).sum(dim=1) ** 0.5

    def batch_dot(self, a, b):
        batch_sz = a.size()[0]
        dim = a.size()[1]
        return torch.bmm(a.view(batch_sz, 1, dim), b.view(batch_sz, dim, 1)).view(batch_sz)
