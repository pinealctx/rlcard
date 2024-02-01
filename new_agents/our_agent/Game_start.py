from tqdm import tqdm
import rlcard

from new_agents.our_agent.Nits_player import NitsAgent
from new_agents.our_agent.Rock_player import RockAgent
from new_agents.our_agent.Tight_Aggressive_Player import TightAggressiveAgent
from new_agents.our_agent.Weak_Tight_Players import WeakTightAgent
from new_agents.our_agent.Maniacs_player import ManiacAgent
from new_agents.our_agent.Loose_Aggressive_Player import LooseAggressiveAgent
from new_agents.our_agent.Calling_stations_Player import CallingStationAgent
from new_agents.our_agent.Donkeys_Player import DonkeyAgent
import pandas as pd

encode_card = {"SA": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4, "S6": 5, "S7": 6, "S8": 7, "S9": 8, "ST": 9, "SJ": 10, "SQ": 11, "SK": 12, "HA": 13, "H2": 14, "H3": 15, "H4": 16, "H5": 17, "H6": 18, "H7": 19, "H8": 20, "H9": 21, "HT": 22, "HJ": 23, "HQ": 24, "HK": 25, "DA": 26, "D2": 27, "D3": 28, "D4": 29, "D5": 30, "D6": 31, "D7": 32, "D8": 33, "D9": 34, "DT": 35, "DJ": 36, "DQ": 37, "DK": 38, "CA": 39, "C2": 40, "C3": 41, "C4": 42, "C5": 43, "C6": 44, "C7": 45, "C8": 46, "C9": 47, "CT": 48, "CJ": 49, "CQ": 50, "CK": 51}


def init_the_game(test_agent,enemy_agent):
    # Make environment
    env = rlcard.make('limit-holdem')

    # human_agent = HumanAgent(env.num_actions)
    # RaiseAgent(num_actions=env.num_actions)
    # RandomAgent(num_actions=env.num_actions)
    # ProbabilityAgent(num_actions=env.num_actions)

    agents = [
        NitsAgent(num_actions=env.num_actions),
        RockAgent(num_actions=env.num_actions),
        TightAggressiveAgent(num_actions=env.num_actions),
        WeakTightAgent(num_actions=env.num_actions),
        ManiacAgent(num_actions=env.num_actions),
        LooseAggressiveAgent(num_actions=env.num_actions),
        CallingStationAgent(num_actions=env.num_actions),
        DonkeyAgent(num_actions=env.num_actions)
    ]

    a = NitsAgent(num_actions=env.num_actions)

    env.set_agents([
        agents[test_agent],
        agents[enemy_agent]
    ])

    return env




def encode_the_action(action):
    if action == "raise":
        return 1
    elif action == "call":
        return 2
    elif action == "check":
        return 3
    elif action == "fold":
        return 4

def encode_the_action_list(actions):

    encoding = [[]] * 4
    encoding[0] = []
    encoding[1] = []
    encoding[2] = []
    encoding[3] = []

    current_stage = 0

    for action in actions:

        if action[1] == 'raise':
            encoding[current_stage].append(1)

        elif action[1] == 'call':
            if encoding[0] == []:
                encoding[current_stage].append(2)
            else:
                encoding[current_stage].append(2)
                current_stage += 1

        elif action[1] == 'check':
            if encoding[current_stage] != []:
                if encoding[current_stage][-1] == 3 or encoding[current_stage][-1] == 2:
                    encoding[current_stage].append(3)
                    current_stage += 1
            else:
                encoding[current_stage].append(3)

        elif action[1] == 'fold':
            encoding[current_stage].append(4)
            break

    encoding[0] += [-1] * (6 - len(encoding[0]))
    encoding[1] += [-1] * (6 - len(encoding[1]))
    encoding[2] += [-1] * (6 - len(encoding[2]))
    encoding[3] += [-1] * (6 - len(encoding[3]))
    encodings = encoding[0] + encoding[1] + encoding[2] + encoding[3]

    return encodings,current_stage


def main(turns, test_agent):

    start_player_list = []
    full_action_list = []
    final_state_list = []
    our_card_list = []
    enemy_card_list = []
    public_card_list = []
    label_list = []

    for i in tqdm(range(8)):

        env = init_the_game(test_agent, i)

        for _ in tqdm(range(turns)):

            trajectories, payoffs = env.run(is_training=False)

            our_agent = trajectories[0][-1]
            enemy_agent = trajectories[1][-1]

            our_raw_obs = our_agent["raw_obs"]
            enemy_raw_obs = enemy_agent["raw_obs"]

            start_player = our_agent["action_record"][0][0]
            action_list, final_state = encode_the_action_list(our_agent["action_record"])

            if final_state == 4:
                final_state =3

            our_card = our_raw_obs["hand"]
            our_encode_card = [encode_card[our_card[0]], encode_card[our_card[1]]]
            enemy_card = enemy_raw_obs["hand"]
            enemy_encode_card = [encode_card[enemy_card[0]], encode_card[enemy_card[1]]]

            public_card = our_raw_obs["public_cards"]
            public_card = [encode_card[public_card[i]] for i in range(len(public_card))]
            public_card += [-1] * (5 - len(public_card))

            start_player_list.append(start_player)
            full_action_list.append(action_list)
            final_state_list.append(final_state)
            our_card_list.append(our_encode_card)
            enemy_card_list.append(enemy_encode_card)
            public_card_list.append(public_card)
            label_list.append(test_agent)

        df = pd.DataFrame({"start_player": start_player_list, "full_action": full_action_list, "final_state": final_state_list, "our_card": our_card_list, "enemy_card": enemy_card_list, "public_card": public_card_list , "label": label_list})
        df.to_pickle(r"D:\RLcard\data\data_of_agent_" + str(test_agent) + "_vs_agent_" + str(i) + ".pkl")


if __name__ == "__main__":

    turns = 10000
    for i in range(8):
        main(turns, i)











