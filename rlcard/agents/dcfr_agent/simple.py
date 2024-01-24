import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)  # 使用Softmax来输出概率分布
        )

    def forward(self, x):
        return self.layers(x)


def remove_illegal(action_probs, legal_actions):
    """ Remove illegal actions and normalize """
    probs = np.zeros_like(action_probs)
    probs[legal_actions] = action_probs[legal_actions]
    probs /= probs.sum()
    return probs


class DeepCFRAgent():
    def __init__(self, env, input_size, model_path='./deep_cfr_model'):
        self.env = env
        self.model_path = model_path

        self.policy_network = MLP(input_size, env.num_actions)
        self.advantage_networks = {player_id: MLP(input_size, env.num_actions) for player_id in range(env.num_players)}

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.advantage_optimizers = {player_id: optim.Adam(network.parameters(), lr=0.001) for player_id, network in self.advantage_networks.items()}

        self.iteration = 0

    def train(self, num_iterations):
        for _ in range(num_iterations):
            self.iteration += 1
            for player_id in range(self.env.num_players):
                self.env.reset()
                probs = np.ones(self.env.num_players)
                self.traverse_tree(probs, player_id)

            if self.iteration % self.env.update_every == 0:
                self.update_policy()

    def traverse_tree(self, probs, player_id):
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        state_utility = np.zeros(self.env.num_players)

        obs, legal_actions = self.env.get_state(current_player)
        action_probs = self.action_probs(obs, legal_actions)

        for action in legal_actions:
            new_probs = probs.copy()
            new_probs[current_player] *= action_probs[action]

            # 遍历子状态
            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            state_utility += action_probs[action] * utility

            if current_player == player_id:
                self.update_advantage_network(obs, action, utility, player_id)

        return state_utility

    def update_policy(self):
        for obs in self.env.observation_space:
            action_probs = self.regret_matching(obs)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            predicted_probs = self.policy_network(obs_tensor)
            loss = -torch.sum(torch.log(predicted_probs) * action_probs)  # 交叉熵损失函数
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

    def regret_matching(self, obs):
        regret = self.get_regret(obs)
        positive_regret_sum = sum(r for r in regret if r > 0)

        # action_probs = np.zeros(self.env.num_actions)
        if positive_regret_sum > 0:
            action_probs = np.maximum(0, regret) / positive_regret_sum
        else:
            action_probs = np.ones(self.env.num_actions) / self.env.num_actions

        return torch.tensor(action_probs, dtype=torch.float32)

    def action_probs(self, obs, legal_actions):
        """ Get action probabilities from the policy network """
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_probs = self.policy_network(obs_tensor).detach().numpy()
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def eval_step(self, state):
        """ Given a state, predict action based on the current policy """
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        probs = self.action_probs(obs, legal_actions)
        action = np.random.choice(len(probs), p=probs)

        info = {'probs': {state['raw_legal_actions'][i]: float(probs[i])
                          for i in range(len(state['legal_actions']))}}

        return action, info

    def get_regret(self, obs, player_id):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        advantage_values = self.advantage_networks[player_id](obs_tensor)
        regret = advantage_values - torch.max(advantage_values)
        return regret.detach().numpy()

    def update_advantage_network(self, obs, action, utility, player_id):
        # 将观测转换为张量
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        # 计算当前的advantage值
        current_advantages = self.advantage_networks[player_id](obs_tensor)
        # 计算损失。这里的损失是实际效用与预测advantage之间的平方差
        loss = (utility - current_advantages[action]) ** 2
        # 反向传播和优化
        self.advantage_optimizers[player_id].zero_grad()
        loss.backward()
        self.advantage_optimizers[player_id].step()

    def save(self):
        """ Save model """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.policy_network.state_dict(), os.path.join(self.model_path, 'policy_network.pth'))
        # Save the advantage networks
        for player_id, network in self.advantage_networks.items():
            torch.save(network.state_dict(), os.path.join(self.model_path, f'advantage_network_{player_id}.pth'))

    def load(self):
        """ Load model """
        if not os.path.exists(self.model_path):
            return
        self.policy_network.load_state_dict(torch.load(os.path.join(self.model_path, 'policy_network.pth')))
        # Load the advantage networks
        for player_id in self.advantage_networks:
            path = os.path.join(self.model_path, f'advantage_network_{player_id}.pth')
            self.advantage_networks[player_id].load_state_dict(torch.load(path))

# 使用示例
# env = ...  # 创建环境
# input_size = ... # 设置输入尺寸
# agent = DeepCFRAgent(env, input_size)
# agent.train()
# agent.save()
