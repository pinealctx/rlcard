
import rlcard
from rlcard.agents import LimitholdemHumanAgent as HumanAgent
from rlcard.utils.utils import print_card

from new_agents.our_agent.Raise_player import RaiseAgent
from new_agents.our_agent.Random_player import RandomAgent
from new_agents.our_agent.Probability_player import ProbabilityAgent
from new_agents.our_agent.Nits_player import NitsAgent
from new_agents.our_agent.Rock_player import RockAgent
from new_agents.our_agent.Tight_Aggressive_Player import TightAggressiveAgent
from new_agents.our_agent.Weak_Tight_Players import WeakTightAgent
from new_agents.our_agent.Maniacs_player import ManiacAgent
from new_agents.our_agent.Loose_Aggressive_Player import LooseAggressiveAgent
from new_agents.our_agent.Calling_stations_Player import CallingStationAgent
from new_agents.our_agent.Donkeys_Player import DonkeyAgent

# Make environment
env = rlcard.make('limit-holdem')
human_agent = HumanAgent(env.num_actions)
agent_0 = RaiseAgent(num_actions=env.num_actions)
agent_1 = RandomAgent(num_actions=env.num_actions)
agent_2 = ProbabilityAgent(num_actions=env.num_actions)

agent_3 = NitsAgent(num_actions=env.num_actions)
agent_4 = RockAgent(num_actions=env.num_actions)
agent_5 = TightAggressiveAgent(num_actions=env.num_actions)
agent_6 = WeakTightAgent(num_actions=env.num_actions)
agent_7 = ManiacAgent(num_actions=env.num_actions)
agent_8 = LooseAggressiveAgent(num_actions=env.num_actions)
agent_9 = CallingStationAgent(num_actions=env.num_actions)
agent_10 = DonkeyAgent(num_actions=env.num_actions)


env.set_agents([
    human_agent,
    agent_10
])

print(">> Limit Hold'em random agent")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    if len(trajectories[0]) != 0:
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            """
            if action_record[-i][0] == state['current_player']:
                break
            """
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     Agent      ===============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")
