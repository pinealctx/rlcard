import torch

from torch import nn
from torch.multiprocessing import Lock


class SkipDense(nn.Module):
    """Dense Layer with skip connection in PyTorch."""

    def __init__(self, units, activation="leakyrelu"):
        super(SkipDense, self).__init__()
        self.hidden = nn.Linear(units, units)
        # Using He initialization (also known as Kaiming initialization)
        he_normal(self.hidden.weight, activation)

    def forward(self, x):
        return self.hidden(x) + x


class RstNet(nn.Module):
    """Implements the Rst network as an MLP.
    """

    def __init__(self,
                 input_size,
                 layers,
                 output_size,
                 activation='leakyrelu',
                 lr=0.0001,
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

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def train_model(self, state, target):
        self.optimizer.zero_grad()
        prediction = self.forward(state)
        loss = self.loss_function(prediction, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class SoftMaxRstNet(RstNet):
    def __init__(
            self,
            input_size,
            output_size,
            mlp_layers,
            activation='leakyrelu',
            lr=0.0001,
    ):
        super().__init__(input_size, mlp_layers, output_size, activation, lr)
        self.softmax = nn.Softmax(dim=-1)
        self.lock = Lock()
        self.loss_function = nn.KLDivLoss(reduction='batchmean')

    def forward(self, inputs):
        state = inputs
        x = self.mlp(state)
        x = self.softmax(x)
        return x


class SoftMaxNet(nn.Module):
    """Implements the SoftMax network as an MLP.
    """

    def __init__(self,
                 input_size,
                 layers,
                 output_size,
                 activation='leakyrelu',
                 lr=0.0001,
                 **kwargs):
        super().__init__(**kwargs)


def he_normal(tensor, activation):
    if activation == "leaky_relu":
        nn.init.kaiming_normal_(tensor, a=0.2, nonlinearity='leakyrelu')
    else:
        nn.init.kaiming_normal_(tensor, nonlinearity='relu')
