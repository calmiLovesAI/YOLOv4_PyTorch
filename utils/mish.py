import torch
import torch.nn as nn
import torch.nn.functional as F


def mish(x):
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)