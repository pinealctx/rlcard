import os.path

import torch
import torch.nn as nn

from advantage_network import AdvantageNetwork
from policy_network import PolicyNetwork


class DCFRAgent(object):
    """ Deep CFR Agent
    """

    def __init__(self, env, device, model_path):
        """ Initilize Agent

        Args:
            env (Env): Env class
        """
        self.env = env
        self.device = 'cuda:' + device if device != "cpu" else "cpu"
        self.model_path = model_path
        self.iteration = 0

        # One policy network for each player
        self.policy_network = PolicyNetwork(env.state_shape,
                                            env.policy_net_layers,
                                            env.num_actions,
                                            env.activation).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=env.policy_lr)
        self.policy_loss = nn.MSELoss()

        # Each advantage network for each player
        self.advantage_networks = nn.ModuleList([
            AdvantageNetwork(env.state_shape,
                             env.adv_net_layers,
                             env.num_actions,
                             env.activation).to(self.device)
            for _ in range(env.num_players)])
        self.advantage_optimizers = [
            torch.optim.Adam(adv.parameters(), lr=env.adv_lr)
            for adv in self.advantage_networks]
        self.advantage_losses = [nn.MSELoss() for _ in range(env.num_players)]

        # training hyperparameters
        self.num_iterations = env.num_iterations
        self.num_traversals = env.num_traversals
        self.batch_size_policy = env.batch_size_policy
        self.steps_policy = env.steps_policy
        self.batch_size_adv = env.batch_size_adv
        self.steps_adv = env.steps_adv
        self.save_adv_per_iter = env.save_adv_per_iter

    def train(self):
        for _ in range (self.num_iterations):
            for player in range(self.env.num_players):
                for _ in range (self.num_traversals):
                    self.env.reset()
                    self.traverse_tree(player)
                advantage_loss = self.train_advantage_network(player)
                policy_loss = self.train_policy_network()
                if self.iteration % self.save_adv_per_iter == 0:
                    torch.save(self.advantage_networks[player].state_dict(),
                               os.path.join(self.model_path, 'adv_{}_{}.pth'.format(player, self.iteration)))
                    torch.save(self.policy_network.state_dict(),
                               os.path.join(self.model_path, 'policy_{}.pth'.format(self.iteration)))
                self.iteration += 1
                print('\rIteration {} Player {} Advantage Loss {} Policy Loss {}'.format(
                    self.iteration, player, advantage_loss, policy_loss), end='')

    def traverse_tree(self, player_id):
        if self.env.is_over():
            return self.env.get_payoffs()

        cur_player = self.env.get_player_id()
        if cur_player == player_id:
            # 当前玩家节点
            legal_actions = self.env.get_legal_actions()
            info_state = self.env.get_state(cur_player)
            policy = self.policy_network.forward(info_state, legal_actions)

