import torch.nn as nn
import torch
import numpy as np


class Net(nn.Module):
    def __init__(self, vocab_len, dim):
        super(Net, self).__init__()
        tpt = torch.Tensor(np.random.randn(vocab_len, dim))
        self.fe_0 = nn.Parameter(tpt, requires_grad=True)
        self.fe_1 = nn.Parameter(tpt, requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        dis = torch.zeros([xs.size()[0]])
        for idx, x in enumerate(xs):
            fe_a = self.fe_0[int(x[0])]
            fe_b = self.fe_1[int(x[1])]

            dis_cur = torch.dot(fe_a, fe_b)
            dis_cur = self.sigmoid(dis_cur)

            dis[idx] = dis_cur
        return dis
