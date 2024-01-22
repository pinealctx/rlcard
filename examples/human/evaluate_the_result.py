''' A toy example of playing against a random agent on Limit Hold'em
'''
import os

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils.utils import print_card
from rlcard.models.limitholdem_rule_models import LimitholdemRuleAgentV1 as LimitholdemRuleAgent


def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
        print(1)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
        print(2)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
        print(3)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
        print(4)

    return agent


# 加载环境

env = rlcard.make('limit-holdem')

# 加载各类agent区域

model_path = r'D:\rlcard\examples\experiments\dmc_result\limit-holdem\0_192585600.pth'

# 随机agent

agent_0 = RandomAgent(num_actions=env.num_actions)

agent_1 = load_model(model_path)

agent_2 = load_model(model_path)

agent_3 = LimitholdemRuleAgent()

# Make environment

# agent_0 = RandomAgent(num_actions=env.num_actions)
env.set_agents([
    agent_0,
    agent_3,
])

print(">> Limit Hold'em random agent")

turns = 1000

agent_1_score = 0

agent_2_score = 0

while (turns):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    if len(trajectories[0]) != 0:
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record) + 1):
            """
            if action_record[-i][0] == state['current_player']:
                break
            """
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('=============     Random Agent    ============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
        agent_1_score += 1
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        agent_2_score += 1
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')
    turns -= 1

print('===============     Final     ===============')

print('Agent0: {}   Agent1: {}'.format(agent_1_score, agent_2_score))

# input("Press any key to continue...")
