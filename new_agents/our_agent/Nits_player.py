import random
from rlcard.games.limitholdem.utils import compare_hands
import numpy as np
import rlcard
from rlcard.models.model import Model
from Probability_player import calculate_win_rate, get_remaining_cards
import torch


# 糊涂虫（Nits）：翻牌前激进的参与，翻牌后变得仅在有好牌时加注

# 更改起手逻辑与后续逻辑

class NitsAgent(object,torch.nn.Module):

    def __init__(self, num_actions):

        super(NitsAgent, self).__init__()

        self.num_actions = num_actions

        self.use_raw = True


    @staticmethod
    def step(state):

        hands = state["raw_obs"]["hand"]
        legal_actions = state["raw_legal_actions"]
        community_cards = state["raw_obs"]["public_cards"]

        win_rate = calculate_win_rate(hands, community_cards, 2000)

        if len(community_cards) == 0:
            if win_rate >= 0.5:
                choice = 'raise'
            elif win_rate > 0.2 and win_rate < 0.5:
                choice = 'call'
            else:
                choice = 'check'

        if len(community_cards) == 3 or len(community_cards) == 4 or len(community_cards) == 5:
            if win_rate >= 0.75:
                choice = 'raise'
            elif win_rate > 0.4 and win_rate < 0.6:
                choice = 'call'
            else:
                choice = 'check'

        if choice in legal_actions:

            return choice
        else:
            if choice == 'raise':
                if 'call' in legal_actions:
                    return 'call'
                else:
                    return 'check'

            elif choice == 'check':

                return 'fold'

            else:  # choice == 'call'
                if 'check' in legal_actions:
                    return 'check'
                else:
                    return 'fold'



    def eval_step(self, state):

        return self.step(state), []



