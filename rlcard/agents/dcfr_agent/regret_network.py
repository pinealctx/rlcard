from rst_network import RstNetwork


# Use pytorch to reimplement RegretNetwork as up comment
class RegretNetwork(RstNetwork):
    """Implements the advantage network as an MLP.

    Implements the advantage network as an MLP with skip connections in
    adjacent layers with the same number of units, except for the last hidden
    connection where a layer normalization is applied.
    """

    def __init__(self,
                 input_size,
                 regret_network_layers,
                 num_actions,
                 activation='leakyrelu',
                 **kwargs):
        super().__init__(input_size, regret_network_layers, num_actions, activation, **kwargs)

    def forward(self, inputs):
        """Applies Policy Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Cumulative regret for each info_state action
        """
        x, mask = inputs
        # 使用nn.Sequential执行层
        x = self.mlp(x)
        # mask处理
        x = mask * x
        return x

