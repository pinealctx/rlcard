import numpy as np

from cachetools import LRUCache
from rlcard.agents import CFRAgent


class OSMCCFRAgent(CFRAgent):

    def __init__(self,
               env,
               max_lru_size,
               model_path='./omc_cfr_model'):
        super().__init__(env, model_path)
        # 固定大小的字典
        self.policy = LRUCache(maxsize=max_lru_size)
        self.average_policy = LRUCache(maxsize=max_lru_size)
        self.average_policy_pool = LRUCache(maxsize=max_lru_size//10)
        self.regrets = LRUCache(maxsize=max_lru_size)

    def traverse_tree(self, probs, player_id):
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()

        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)
        obs, legal_actions = self.get_state(current_player)
        action_probs = self.action_probs(obs, legal_actions, self.policy)

        if current_player == player_id:
            # Sample an action according to the current strategy
            action = np.random.choice(legal_actions, p=action_probs[legal_actions])
            new_probs = probs.copy()
            new_probs[current_player] *= action_probs[action]

            # Keep traversing the child state
            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            state_utility += action_probs[action] * utility
            action_utilities[action] = utility

            # Update regrets and strategy along the path of the sampled outcome
            player_prob = probs[current_player]
            counterfactual_prob = (np.prod(probs[:current_player]) *
                                    np.prod(probs[current_player + 1:]))
            player_state_utility = state_utility[current_player]

            if obs not in self.regrets:
                self.regrets[obs] = np.zeros(self.env.num_actions)
            if obs not in self.average_policy:
                self.average_policy[obs] = np.zeros(self.env.num_actions)

            regret = counterfactual_prob * (action_utilities[action][current_player]
                                            - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += self.iteration * player_prob * action_probs[action]
            self.average_policy_pool[obs] = self.average_policy[obs]

        else:
            # Choose the action that leads to the current information set
            action = np.argmax(action_probs)
            new_probs = probs.copy()
            new_probs[current_player] *= action_probs[action]

            # Keep traversing the child state
            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            state_utility += action_probs[action] * utility

        return state_utility
