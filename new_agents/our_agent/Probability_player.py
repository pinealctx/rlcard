import random
from rlcard.games.limitholdem.utils import compare_hands


def calculate_win_rate(player1_hand, community_cards, num_trials=1000):
    wins = 0

    for _ in range(num_trials):

        remain_community_cards = random.sample(get_remaining_cards(player1_hand+community_cards), 5-len(community_cards))

        all_community_cards = community_cards + remain_community_cards
        # 补全玩家1的底牌
        player2_hand = random.sample(get_remaining_cards(player1_hand + all_community_cards), 2)

        # 拼接玩家手牌与公共牌
        player_1_all_cards = player1_hand + all_community_cards
        player_2_all_cards = player2_hand + all_community_cards

        # 比较两名玩家的手牌
        winner = compare_hands([player_1_all_cards, player_2_all_cards])

        # 统计玩家1的胜利次数
        if winner[0] == 1:
            wins += 1

    # 计算胜率
    win_rate = wins / num_trials

    return win_rate

# 获取剩余的牌
def get_remaining_cards(used_cards):
    all_cards = ['CA', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'CT', 'CJ', 'CQ', 'CK',
                 'DA', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'DT', 'DJ', 'DQ', 'DK',
                 'HA', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'HT', 'HJ', 'HQ', 'HK',
                 'SA', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'ST', 'SJ', 'SQ', 'SK'
                ]

    remaining_cards = [card for card in all_cards if card not in used_cards]

    return remaining_cards


# 基准的全看概率的agent

class ProbabilityAgent(object):
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



if __name__ == "__main__":
    # 测试计算胜率函数
    player1_hand = ['CJ', 'CQ']
    community_cards = ["C3"]
    win_rate = calculate_win_rate(player1_hand, community_cards, 2000)

    print("玩家1胜率：", win_rate)

