import numpy as np
import torch
import math
import pickle


def gen_A(num_classes, t, adj_file):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(D, A), D)
    return adj


class AveragePrecisionMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = torch.FloatTensor()
        self.targets = torch.LongTensor()

    def add(self, output, target):
        output = output.cpu()
        target = target.cpu()

        if self.scores.numel() == 0:
            self.scores = output.clone()
            self.targets = target.clone()
        else:
            self.scores = torch.cat([self.scores, output])
            self.targets = torch.cat([self.targets, target])

    def overall(self):
        scores = self.scores.numpy()
        targets = self.targets.numpy()
        Ng = (targets == 1).sum(axis=0)
        Np = (scores >= 0).sum(axis=0)
        Nc = ((targets == 1) & (scores >= 0)).sum(axis=0)

        Np[Np == 0] = 1
        OP = Nc.sum() / Np.sum()
        OR = Nc.sum() / Ng.sum()
        OF1 = (2 * OP * OR) / (OP + OR + 1e-6)

        CP = (Nc / Np).mean()
        CR = (Nc / Ng).mean()
        CF1 = (2 * CP * CR) / (CP + CR + 1e-6)

        return OP, OR, OF1, CP, CR, CF1
