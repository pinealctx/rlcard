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

# 测试计算胜率函数
player1_hand = ['S7', 'C7']
community_cards = []
win_rate = calculate_win_rate(player1_hand, community_cards, 10000)

print("玩家1胜率：", win_rate)
