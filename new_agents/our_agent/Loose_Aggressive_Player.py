from .Probability_player import calculate_win_rate

# 松凶型玩家，给人一种疯狂的感觉，但是实际有明确的策略性

# 降低前两轮的fold阈值与raise阈值
# 后两轮逐步恢复阈值

class LooseAggressiveAgent(object):
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

        if len(community_cards) == 0:
            if win_rate >= 0.5:
                choice = 'raise'
            elif win_rate > 0.1 and win_rate < 0.5:
                choice = 'call'
            else:
                choice = 'check'

        if len(community_cards) == 3:
            if win_rate >= 0.5:
                choice = 'raise'
            elif win_rate > 0.1 and win_rate < 0.5:
                choice = 'call'
            else:
                choice = 'check'

        if len(community_cards) == 4:
            if win_rate >= 0.6:
                choice = 'raise'
            elif win_rate > 0.3 and win_rate < 0.6:
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