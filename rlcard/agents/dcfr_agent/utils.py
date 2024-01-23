import torch.nn as nn


def he_normal(tensor, activation):
    if activation == "leaky_relu":
        nn.init.kaiming_normal_(tensor, a=0.2, nonlinearity='leakyrelu')
    else:
        nn.init.kaiming_normal_(tensor, nonlinearity='relu')
