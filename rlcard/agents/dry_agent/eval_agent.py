import numpy as np
import torch

from rlcard.utils.utils import remove_illegal


class EvalAgent(object):
    def __init__(self, average_policy_net, device):
        self.device = device
        self.average_policy_net = average_policy_net

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
