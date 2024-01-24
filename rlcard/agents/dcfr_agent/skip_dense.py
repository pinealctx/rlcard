import torch.nn as nn

from utils import he_normal


class SkipDense(nn.Module):
    """Dense Layer with skip connection in PyTorch."""

    def __init__(self, units, activation="leakyrelu"):
        super(SkipDense, self).__init__()
        self.hidden = nn.Linear(units, units)
        # Using He initialization (also known as Kaiming initialization)
        he_normal(self.hidden.weight, activation)

    def forward(self, x):
        return self.hidden(x) + x
