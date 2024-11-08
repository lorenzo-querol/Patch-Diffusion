import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_norm(n_filters, norm):
    if norm is None or norm.lower() == "none":
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    else:
        return Identity()
