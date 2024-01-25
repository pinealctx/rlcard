import torch
import torch.nn as nn
import numpy as np

from cachetools import LRUCache
from rlcard.agents import CFRAgent
from rlcard.agents.dcfr_agent import RstNetwork
from rlcard.utils.utils import remove_illegal


class DCFRAgent(CFRAgent):
    def __init__(self, env, max_lru_size, model_path='./deep_cfr_model'):
        super().__init__(env, model_path)

        # 固定大小的字典
        self.policy = LRUCache(maxsize=max_lru_size)
        self.average_policy = LRUCache(maxsize=max_lru_size)
        self.regrets = LRUCache(maxsize=max_lru_size)
        self.policy_buff = [None] * max_lru_size
        self.policy_buff = self.policy_buff[:0]

    def update_policy(self):
        super().update_policy()


class PolicyNet(RstNetwork):
    def __init__(
            self,
            state_shape,
            action_shape,
            mlp_layers = [512, 512, 512, 512, 512, 512, 512, 512],
    ):
        super().__init__(state_shape, mlp_layers, action_shape)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        state = inputs
        x = self.mlp(state)
        x = self.softmax(x)
        return x


class DCFRActor(object):
    def __init__(
            self,
            env,
            state_shape,
            action_shape,
            lr=0.0001,
            max_lru_size=1000000,
            mlp_layers = [512, 512, 512, 512, 512, 512, 512, 512],
            device="0",
    ):
        self.agent = DCFRAgent(env, max_lru_size)
        self.device = 'cuda:' + device if device != "cpu" else "cpu"
        self.net = PolicyNet(state_shape, action_shape, mlp_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.action_shape = action_shape

    def eval_step(self, state):
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        probs = self.action_probs(obs, legal_actions)
        action = np.random.choice(len(probs), p=probs)

        info = {'probs': {state['raw_legal_actions'][i]: float(probs[i])
                          for i in range(len(state['legal_actions']))}}

        return action, info

    def share_memory(self):
        self.net.share_memory()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def action_probs(self, obs, legal_actions):
        """ Get action probabilities from the policy network """
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_probs = self.net(obs_tensor).detach().numpy()
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def load_policy_dict(self, policy_dict):
        self.net.load_state_dict(policy_dict)

    def state_dict(self):
        return self.net.state_dict()

    def set_device(self, device):
        self.device = device


class DCFRModel(object):
    def __init__(
            self,
            state_shape,
            action_shape,
            multi_num=16,
            lr=0.0001,
            mlp_layers = [512, 512, 512, 512, 512, 512, 512, 512],
            device="0",
    ):
        self.actors = []
        for _ in range (multi_num):
            actor = DCFRActor(
                state_shape,
                action_shape,
                lr,
                mlp_layers,
                device
            )
            self.actors.append(actor)

    def share_memory(self):
        for actor in self.actors:
            actor.share_memory()

    def eval(self):
        for actor in self.actors:
            actor.eval()

    def parameters(self, index):
        return self.actors[index].parameters()

    def get_actor(self, index):
        return self.actors[index]

    def get_agents(self):
        return self.actors
