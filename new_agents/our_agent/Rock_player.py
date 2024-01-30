import random
from rlcard.games.limitholdem.utils import compare_hands
import numpy as np
import rlcard
from rlcard.models.model import Model
from Probability_player import calculate_win_rate, get_remaining_cards



# 岩石类玩家（Rocks）：不是在等待好牌，而是在等待最好的牌

class RockAgent(object):
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

        if win_rate >= 0.8:
            choice = 'raise'
        elif win_rate > 0.6 and win_rate < 0.8:
            choice = 'call'
        else:
            choice = 'check'

        if choice in legal_actions:

            return choice
        else:
            if choice == 'raise' :
                if 'call' in legal_actions:
                    return 'call'
                else:
                    return 'check'

            elif choice == 'check':

                return 'fold'

            else:   # choice == 'call'
                if 'check' in legal_actions:
                    return 'check'
                else:
                    return 'fold'

    def eval_step(self, state):

        return self.step(state), []