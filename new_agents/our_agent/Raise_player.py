import numpy as np


class RaiseAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, num_actions):

        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):

        choice = np.random.choice([0,1])

        return choice

    def eval_step(self, state):

        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1 / len(state['legal_actions'])

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in
                         range(len(state['legal_actions']))}

        return self.step(state), info
