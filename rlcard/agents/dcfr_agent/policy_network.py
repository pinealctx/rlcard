import torch.nn as nn

from .rst_network import RstNetwork


class PolicyNetwork(RstNetwork):
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
        super().__init__(input_size, policy_network_layers, num_actions, activation, **kwargs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        """Applies Policy Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Action probabilities
        """
        x, mask = inputs
        # 使用nn.Sequential执行层
        x = self.mlp(x)
        # mask处理
        mask = mask.bool()
        x.masked_fill_(~mask, float('-inf'))
        # soft max
        x = self.softmax(x)
        return x
