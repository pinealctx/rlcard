import os
import torch
import numpy as np

from torch import nn
from torch.multiprocessing import Lock
from rlcard.utils.utils import remove_illegal
from .omc_cfr_agent import OSMCCFRAgent


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


class PolicyNet(RstNet):
    def __init__(
            self,
            state_shape,
            action_shape,
            mlp_layers=[512, 512, 512, 512, 512, 512, 512, 512],
            activation='leakyrelu',
            lr=0.0001,
    ):
        super().__init__(state_shape, mlp_layers, action_shape, activation, lr)
        self.softmax = nn.Softmax(dim=-1)
        self.lock = Lock()

    def forward(self, inputs):
        state = inputs
        x = self.mlp(state)
        x = self.softmax(x)
        return x


class DeepOSMCCFRAgent(OSMCCFRAgent):
    def __init__(self,
                 env,
                 max_lru_size,
                 device_num=0,
                 batch_size=64,
                 epochs=1,
                 process_id=0,
                 activation='leakyrelu',
                 lr=0.0001,
                 model_path='./deep_omc_cfr_model'):
        super().__init__(env, max_lru_size, model_path)

        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.epochs = epochs
        self.process_id = process_id
        self.average_policy_net = PolicyNet(self.env.state_shape[0][0], self.env.num_actions,
                                            activation=activation,
                                            lr=lr).to(self.device)
        self.average_policy_net.share_memory()

    def train(self):
        super().train()
        self.train_model()

    def train_model(self):
        if len(self.average_policy) < self.batch_size:
            return
        for epoch in range(self.epochs):
            states, targets = zip(*list(self.average_policy.items()))
            states = [np.frombuffer(state, dtype=np.float64) for state in states]
            states = np.array(states)
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            targets = np.array(targets)
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
            self.average_policy_net.lock.acquire()
            try:
                for i in range(0, len(states), self.batch_size):
                    inputs = states[i:i + self.batch_size]
                    labels = targets[i:i + self.batch_size]
                    self.average_policy_net.train_model(inputs, labels)
            finally:
                self.average_policy_net.lock.release()

    def save(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        file_name = "deep_omc_cfr_model_{}_{}.pth".format(self.process_id, self.iteration)
        self.save_model(os.path.join(self.model_path, file_name))

    def load(self):
        if not os.path.exists(self.model_path):
            return
        file_name = "deep_omc_cfr_model_{}.pth".format(self.process_id)
        self.load_model(os.path.join(self.model_path, file_name))

    def eval_step(self, state):
        """use self policy network to predict action"""
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        action_probs = self.eval_action_probs(obs, legal_actions)
        action = np.random.choice(len(action_probs), p=action_probs)
        info = {'probs': {state['raw_legal_actions'][i]: float(action_probs[i])
                          for i in range(len(state['legal_actions']))}}

        return action, info

    def eval_action_probs(self, obs, legal_actions):
        """ Get action probabilities from the policy network """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_probs = self.average_policy_net(obs_tensor).detach().cpu().numpy()
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def save_model(self, path):
        """Save the model to the specified path."""
        torch.save({
            'average_net_state_dict': self.average_policy_net.state_dict(),
        }, path)

    def load_model(self, path):
        """Load the model from the specified path."""
        checkpoint = torch.load(path)
        self.average_policy_net.load_state_dict(checkpoint['average_net_state_dict'])


def he_normal(tensor, activation):
    if activation == "leaky_relu":
        nn.init.kaiming_normal_(tensor, a=0.2, nonlinearity='leakyrelu')
    else:
        nn.init.kaiming_normal_(tensor, nonlinearity='relu')

