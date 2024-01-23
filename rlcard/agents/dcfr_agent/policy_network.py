import torch.nn as nn

from skip_dense import SkipDense
from utils import he_normal


class PolicyNetwork(nn.Module):
    """Implements the policy network as an MLP.

    Implements the policy network as a MLP with skip connections in adjacent
    layers with the same number of units, except for the last hidden connection
    where a layer normalization is applied.
    """

    def __init__(self,
                 input_size,
                 policy_network_layers,
                 num_actions,
                 activation='leakyrelu',
                 **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        self._num_actions = num_actions
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('Unsupported activation function: {}'.format(activation))

        self.softmax = nn.Softmax(dim=-1)

        self.hidden = []
        prevunits = 0
        for units in policy_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units, activation))
            else:
                self.hidden.append(nn.Linear(prevunits, units))
                he_normal(self.hidden[-1].weight, activation)
            prevunits = units
        self.normalization = nn.LayerNorm(prevunits)
        self.lastlayer = nn.Linear(prevunits, policy_network_layers[-1])
        he_normal(self.lastlayer.weight, activation)

        self.out_layer = nn.Linear(policy_network_layers[-1], num_actions)
        self.model = nn.Sequential()
        for i, layer in enumerate(self.hidden):
            fc = nn.Sequential(layer, self.activation)
            self.model.add_module('h{}'.format(i), fc)
        self.model.add_module('norm', self.normalization)
        self.model.add_module('last', nn.Sequential(self.lastlayer, self.activation))
        self.model.add_module('out', self.out_layer)

    def forward(self, inputs):
        """Applies Policy Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Action probabilities
        """
        x, mask = inputs

        # 使用nn.Sequential执行层
        x = self.model(x)

        # mask处理
        mask = mask.float()
        x = x * mask
        x = x - (1 - mask) * 1e20

        # soft max
        x = self.softmax(x)
        return x
