import torch.nn as nn

from skip_dense import SkipDense
from utils import he_normal


'''
class AdvantageNetwork(tf.keras.Model):
  """Implements the advantage network as an MLP.

  Implements the advantage network as an MLP with skip connections in
  adjacent layers with the same number of units, except for the last hidden
  connection where a layer normalization is applied.
  """

  def __init__(self,
               input_size,
               adv_network_layers,
               num_actions,
               activation='leakyrelu',
               **kwargs):
    super().__init__(**kwargs)
    self._input_size = input_size
    self._num_actions = num_actions
    if activation == 'leakyrelu':
      self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
    elif activation == 'relu':
      self.activation = tf.keras.layers.ReLU()
    else:
      self.activation = activation

    self.hidden = []
    prevunits = 0
    for units in adv_network_layers[:-1]:
      if prevunits == units:
        self.hidden.append(SkipDense(units))
      else:
        self.hidden.append(
            tf.keras.layers.Dense(units, kernel_initializer='he_normal'))
      prevunits = units
    self.normalization = tf.keras.layers.LayerNormalization()
    self.lastlayer = tf.keras.layers.Dense(
        adv_network_layers[-1], kernel_initializer='he_normal')

    self.out_layer = tf.keras.layers.Dense(num_actions)

  @tf.function
  def call(self, inputs):
    """Applies Policy Network.

    Args:
        inputs: Tuple representing (info_state, legal_action_mask)

    Returns:
        Cumulative regret for each info_state action
    """
    x, mask = inputs
    for layer in self.hidden:
      x = layer(x)
      x = self.activation(x)

    x = self.normalization(x)
    x = self.lastlayer(x)
    x = self.activation(x)
    x = self.out_layer(x)
    x = mask * x

    return x
'''

# Use pytorch to reimplement AdvantageNetwork as up comment
class AdvantageNetwork(nn.Module):
    """Implements the advantage network as an MLP.

    Implements the advantage network as an MLP with skip connections in
    adjacent layers with the same number of units, except for the last hidden
    connection where a layer normalization is applied.
    """

    def __init__(self,
                 input_size,
                 adv_network_layers,
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

        self.hidden = []
        prevunits = 0
        for units in adv_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units, activation))
            else:
                self.hidden.append(nn.Linear(prevunits, units))
                he_normal(self.hidden[-1].weight, activation)
            prevunits = units
        self.normalization = nn.LayerNorm(prevunits)
        self.lastlayer = nn.Linear(prevunits, adv_network_layers[-1])
        he_normal(self.lastlayer.weight, activation)

        self.out_layer = nn.Linear(adv_network_layers[-1], num_actions)
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
            Cumulative regret for each info_state action
        """
        x, mask = inputs

        # 使用nn.Sequential执行层
        x = self.model(x)

        # mask处理
        x = mask * x

        return x

