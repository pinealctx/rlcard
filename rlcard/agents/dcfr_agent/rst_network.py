import torch.nn as nn

from .skip_dense import SkipDense
from .utils import he_normal


class RstNetwork(nn.Module):
    """Implements the Rst network as an MLP.
    """

    def __init__(self,
                 input_size,
                 layers,
                 output_size,
                 activation='leakyrelu',
                 **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('Unsupported activation function: {}'.format(activation))

        self.mlp = nn.Sequential()
        self.input_layer = nn.Linear(input_size, layers[0])
        he_normal(self.input_layer.weight, activation)
        self.mlp.add_module('input', nn.Sequential(self.input_layer, self.activation))

        self.hidden = []
        prev_units = layers[0]
        for units in layers[:-1]:
            if prev_units == units:
                self.hidden.append(SkipDense(units, activation))
            else:
                self.hidden.append(nn.Linear(prev_units, units))
                he_normal(self.hidden[-1].weight, activation)
            prev_units = units

        for i, layer in enumerate(self.hidden):
            fc = nn.Sequential(layer, self.activation)
            self.mlp.add_module('h{}'.format(i), fc)

        self.normalization = nn.LayerNorm(prev_units)
        self.mlp.add_module('norm', self.normalization)

        self.last_layer = nn.Linear(prev_units, layers[-1])
        he_normal(self.last_layer.weight, activation)
        self.mlp.add_module('last', nn.Sequential(self.last_layer, self.activation))

        self.out_layer = nn.Linear(layers[-1], output_size)
        self.mlp.add_module('out', self.out_layer)
