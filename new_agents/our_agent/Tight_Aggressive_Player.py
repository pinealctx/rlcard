import random
from rlcard.games.limitholdem.utils import compare_hands
import numpy as np
import rlcard
from rlcard.models.model import Model
from Probability_player import calculate_win_rate, get_remaining_cards


# 紧凶型玩家，标准玩家，根据他人的操作简单调整策略

class TightAggressiveAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, num_actions):

        self.num_actions = num_actions

        self.use_raw = True


    @staticmethod
    def step(state):

        hands = state["raw_obs"]["hand"]
        legal_actions = state["raw_legal_actions"]
        community_cards = state["raw_obs"]["public_cards"]

        win_rate = calculate_win_rate(hands, community_cards, 2000)

        num_game = 0 if len(community_cards) == 0 else len(community_cards) - 2
        raise_num = state["raw_obs"]["raise_nums"][num_game]

        if raise_num == 0:
            win_rate = win_rate + 0.1
        elif win_rate < 0.75:
            win_rate = win_rate - 0.1


        if len(community_cards) == 0:
            if win_rate >= 0.6:
                choice = 'raise'
            elif win_rate > 0.3 and win_rate < 0.6:
                choice = 'call'
            else:
                choice = 'check'

        if len(community_cards) == 3:
            if win_rate >= 0.6:
                choice = 'raise'
            elif win_rate > 0.3 and win_rate < 0.6:
                choice = 'call'
            else:
                choice = 'check'

        if len(community_cards) == 4:
            if win_rate >= 0.7:
                choice = 'raise'
            elif win_rate > 0.3 and win_rate < 0.7:
                choice = 'call'
            else:
                choice = 'check'

        if len(community_cards) == 5:
            if win_rate >= 0.7:
                choice = 'raise'
            elif win_rate > 0.4 and win_rate < 0.7:
                choice = 'call'
            else:
                choice = "check"

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