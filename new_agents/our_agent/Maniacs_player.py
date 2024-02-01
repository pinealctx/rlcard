from .Probability_player import calculate_win_rate


# 疯狂玩家：玩的十分松，喜欢诈胡且经常诈胡
# 几乎不会fold，只有在后两轮且手牌极差的情况下才会fold

class ManiacAgent(object):
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
            else:
                choice = 'call'

        if len(community_cards) == 3:
            if win_rate >= 0.5:
                choice = 'raise'
            else:
                choice = 'call'

        if len(community_cards) == 4:
            if win_rate >= 0.5:
                choice = 'raise'
            elif win_rate > 0.3 and win_rate < 0.5:
                choice = 'call'
            else:
                choice = 'check'

        if len(community_cards) == 5:
            if win_rate >= 0.6:
                choice = 'raise'
            elif win_rate > 0.3 and win_rate < 0.6:
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