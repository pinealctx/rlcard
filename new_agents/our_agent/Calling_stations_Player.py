# 跟注站：莫的感情的跟注机器，长久地待在底池而很少表现出激进度，即使当他们有很好的牌时也是如此。

class CallingStationAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, num_actions):

        self.num_actions = num_actions

        self.use_raw = True


    @staticmethod
    def step(state):

        legal_actions = state["raw_legal_actions"]

        choice = "call"

        if choice in legal_actions:

            return choice

        else:
            if 'check' in legal_actions:
                return 'check'
            elif 'raise' in legal_actions:
                return 'raise'
            else:
                return 'fold'



    def eval_step(self, state):

        return self.step(state), []