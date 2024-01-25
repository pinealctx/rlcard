import os.path
import pickle
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from .advantage_network import AdvantageNetwork
from .policy_network import PolicyNetwork
from .data_set import AdvantageDataset, PolicyDataset


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
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=env.policy_learning_rate)
        self.policy_loss = nn.MSELoss()

        # Each advantage network for each player
        self.advantage_networks = nn.ModuleList([
            AdvantageNetwork(env.state_shape,
                             env.adv_net_layers,
                             env.num_actions,
                             env.activation).to(self.device)
            for _ in range(env.num_players)])
        self.advantage_optimizers = [
            torch.optim.Adam(adv_net.parameters(), lr=env.advantage_learning_rate)
            for adv_net in self.advantage_networks]
        self.advantage_losses = [nn.MSELoss() for _ in range(env.num_players)]

        # training hyperparameters
        """
        self.num_iterations = env.num_iterations
        self.num_traversals = env.num_traversals
        self.policy_batch_size = env.policy_batch_size
        self.advantage_batch_size = env.advantage_batch_size
        self.policy_steps = env.policy_steps
        self.save_policy_per_iter = env.save_policy_per_iter
        self.advantage_steps = env.advantage_steps
        self.save_advantage_per_iter = env.save_adv_per_iter
        """

    def train(self):
        for _ in range(self.env.num_iterations):
            for player in range(self.env.num_players):
                for _ in range(self.env.num_traversals):
                    self.env.reset()
                    self.traverse_tree(player)
                advantage_loss = self.train_advantage_network(player)
                policy_loss = self.train_policy_network()

                if self.iteration % self.env.save_advantage_per_iter == 0:
                    torch.save(self.advantage_networks[player].state_dict(),
                               os.path.join(self.model_path, 'adv_{}_{}.pth'.format(player, self.iteration)))
                if self.iteration % self.env.save_policy_per_iter == 0:
                    torch.save(self.policy_network.state_dict(),
                               os.path.join(self.model_path, 'policy_{}.pth'.format(self.iteration)))
                self.iteration += 1
                print('Iteration {} Player {} Advantage Loss {} Policy Loss {}\n'.format(
                    self.iteration, player, advantage_loss, policy_loss))

    def traverse_tree(self, player_id):
        if self.env.is_over():
            return self.env.get_payoffs()

        cur_player = self.env.get_player_id()
        state = self.env.get_state(cur_player)
        if cur_player == player_id:
            # 计算advantage
            advantages = self.get_advantages(cur_player)
            # regret matching
            strategy = self.regret_matching(advantages)

            # 计算expectation of payoff
            exp_payoff = np.zeros(self.env.num_actions)
            for action in self.env.action_space:
                self.env.step(action)
                exp_payoff[action] = self.traverse_tree(player_id)
                self.env.step_back()

            ev = np.dot(strategy, exp_payoff)
            regrets = exp_payoff - ev

            # 存储advantage
            self.store_advantage(state, regrets, cur_player)
        else:
            # 存储strategy
            policy = self.get_policy(state)
            action = np.random.choice(self.env.num_actions, p=policy)
            self.env.step(action)

        return ev

    def train_advantage_network(self, player_id):
        advantage_dataset = AdvantageDataset(self.advantage_buffers[player_id])
        advantage_loader = DataLoader(advantage_dataset,
                                      batch_size=self.env.policy_batch_size,
                                      shuffle=True)
        for _ in range(self.env.steps_adv):
            for batch in advantage_loader:
                state, advantage, iteration, mask = batch
                advantage_pred = self.advantage_networks[player_id](state, mask)
                loss = self.advantage_losses[player_id](advantage_pred, advantage)

                self.advantage_optimizers[player_id].zero_grad()
                loss.backward()
                self.advantage_optimizers[player_id].step()

        return loss.item()

    def train_policy_network(self):
        policy_dataset = PolicyDataset(self.policy_buffers)
        policy_loader = DataLoader(policy_dataset, batch_size=self.env.policy_batch_size, shuffle=True)
        for _ in range(self.env.steps_policy):
            for batch in policy_loader:
                state, action_prob, iteration, mask = batch
                action_probs_pred = self.policy_network(state, mask)
                loss = self.policy_loss(action_probs_pred, action_prob)

                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

        return loss.item()

    def store_advantage(self, state, advantage, player_id):
        self.advantage_buffers[player_id].append((state, advantage, self.iteration))

    def get_advantages(self, player_id):
        state = self.env.get_state(player_id)
        advantage = self.advantage_networks[player_id](state)
        return advantage

    def get_policy(self, state):
        action_probs = self.policy_network(state)
        return action_probs

    def save(self):
        # 创建保存目录
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

            # 保存策略网络
        policy_net_path = os.path.join(self.model_path, 'policy_net.pth')
        torch.save(self.policy_net.state_dict(), policy_net_path)

        # 保存优势网络
        for i, net in enumerate(self.advantage_nets):
            advantage_net_path = os.path.join(self.model_path, f'advantage_net_{i}.pth')
            torch.save(net.state_dict(), advantage_net_path)

        # 其他训练状态
        params = {
            'iteration': self.iteration
        }
        params_path = os.path.join(self.model_path, 'params.pickle')
        pickle.dump(params, open(params_path, 'wb'))

    def load(self):
        # 加载策略网络
        policy_net_path = os.path.join(self.model_path, 'policy_net.pth')
        if os.path.exists(policy_net_path):
            self.policy_net.load_state_dict(torch.load(policy_net_path))

        # 加载优势网络
        for i in range(self.env.num_players):
            advantage_net_path = os.path.join(self.model_path, f'advantage_net_{i}.pth')
            if os.path.exists(advantage_net_path):
                self.advantage_nets[i].load_state_dict(torch.load(advantage_net_path))

        # 其他训练状态
        params_path = os.path.join(self.model_path, 'params.pickle')
        if os.path.exists(params_path):
            params = pickle.load(open(params_path, 'rb'))
            self.iteration = params['iteration']
