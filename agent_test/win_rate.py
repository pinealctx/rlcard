import random
import itertools
import argparse
from datetime import datetime

import pandas as pd

from new_agents.our_agent.Probability_player import get_remaining_cards, calculate_win_rate as calculate_win_rate_p2
from rlcard.games.limitholdem.cardcmp import CardComb, whole_cards
from rlcard.games.limitholdem.utils import compare_hands

seq_idx = [
    (0, 64),  # 0
    (64, 128),  # 1
    (128, 192),  # 2
    (192, 256),  # 3
    (256, 320),  # 4
    (320, 384),  # 5
    (384, 448),  # 6
    (448, 512),  # 7
    (512, 576),  # 8
    (576, 640),  # 9
    (640, 704),  # 10
    (704, 768),  # 11
    (768, 832),  # 12
    (832, 896),  # 13
    (896, 960),  # 14
    (960, 1024),  # 15
    (1024, 1088),  # 16
    (1088, 1152),  # 17
    (1152, 1216),  # 18
    (1216, 1280),  # 19
    (1280, 1326),  # 20
]


def calculate_win_rate_quick(player1_hand, community_card, player_num=2, num_trials=2000):
    wins = 0
    comm_sample_count = 5 - len(community_card)
    other_p_s_count = (player_num - 1) * 2
    left_card = get_remaining_cards(player1_hand + community_card)
    for _ in range(num_trials):
        sample_cards = random.sample(left_card, comm_sample_count + other_p_s_count)
        all_community_cards = community_card + sample_cards[:comm_sample_count]
        p_others = [sample_cards[i:i + 2] for i in range(comm_sample_count, comm_sample_count + other_p_s_count, 2)]
        # now join the hands and community cards
        # card_all is list of all user's hand + community cards
        # [player_hand+community_card, player_hand+community_card, ...]
        card_all = [player1_hand + all_community_cards] + [p + all_community_cards for p in p_others]
        win_infos = compare_hands(card_all)
        if win_infos[0] == 1:
            wins += 1
    return wins / num_trials


def calculate_win_rate_slow(p1_hand, community_card, player_num=2, num_trials=2000):
    wins = 0
    comm_sample_count = 5 - len(community_card)
    other_p_s_count = (player_num - 1) * 2
    left_card = get_remaining_cards(p1_hand + community_card)

    for _ in range(num_trials):
        sample_cards = random.sample(left_card, comm_sample_count + other_p_s_count)
        all_community_cards = community_card + sample_cards[:comm_sample_count]
        p_others = [sample_cards[i:i + 2] for i in range(comm_sample_count, comm_sample_count + other_p_s_count, 2)]
        wins += compare_hands_slow(p1_hand, all_community_cards, p_others)
    return wins / num_trials


def compare_hands_slow(p1, community_card, p_others):
    p1_value = CardComb().add_cards(p1).add_cards(community_card).count_value()
    p_other_values = [CardComb().add_cards(p).add_cards(community_card).count_value() for p in p_others]
    return 1 if all(p1_value >= p2 for p2 in p_other_values) else 0


def save_before_flop_win_rate(player_num=2, num_trials=2000):
    """calculate win rate of all possible hands before flop
    save it to pandas dataframe
    """
    print('start time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    p1_hands = list(itertools.combinations(whole_cards, 2))
    win_rate = []
    for p1_hand in p1_hands:
        win_rate.append({
            'hands': p1_hand,
            'community_card': [],
            'win_rate': calculate_win_rate_quick(list(p1_hand), [], player_num, num_trials),
        })

    # save to pandas dataframe
    df = pd.DataFrame(win_rate)
    df.to_csv('before_flop_win_rate{}.csv'.format(player_num), index=False)


def save_win_rate(idx: int, player_num=2, num_trials=2000, random_sample_count=3000):
    """calculate win rate of all possible hands after flop
    save it to pandas dataframe
    """
    print('start time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    p1_hands = list(itertools.combinations(whole_cards, 2))
    win_rate = []
    start_idx, end_idx = seq_idx[idx]
    p1_hands = p1_hands[start_idx:end_idx]
    for index, p1_hand in enumerate(p1_hands):
        p1_hand_list = list(p1_hand)
        left_cards = get_remaining_cards(p1_hand_list)
        # random sample community cards, it could be 3 cards or 4 cards or 5 cards
        for _ in range(random_sample_count):
            n = random.randint(3, 5)
            community_card = random.sample(left_cards, n)
            win_rate.append({
                'hands': p1_hand_list,
                'community_card': community_card,
                'win_rate': calculate_win_rate_quick(p1_hand_list, community_card, player_num, num_trials),
            })
        print('current time:{} finish {}/{}'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), index + 1, len(p1_hands)))

    # save to pandas dataframe
    df = pd.DataFrame(win_rate)
    df.to_csv('flop_win_rate{}-{}-{}.csv'.format(player_num, start_idx, end_idx), index=False)


def compare_win_rate(cards: str, player_num=2, num_trials=2000):
    """compare win rate of two hands
    """
    cards = cards.split(',')
    p1_hand = cards[:2]
    community_card = cards[2:]

    calculate_win_rate_with_time(calculate_win_rate_slow, p1_hand, community_card, player_num, num_trials)
    calculate_win_rate_with_time(calculate_win_rate_quick, p1_hand, community_card, player_num, num_trials)
    old = calculate_win_rate_p2(p1_hand, community_card, num_trials)
    print('old of compare two players:', old)


def calculate_win_rate_with_time(fn, p1_hand, community_card, player_num=2, num_trials=2000):
    time1 = datetime.now()
    win_rate = fn(p1_hand, community_card, player_num, num_trials)
    time2 = datetime.now()
    print('func name:{} win rate current:{}, use time:{}:'.format(fn.__name__, win_rate, time2 - time1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate win rate before flop")
    parser.add_argument(
        '--player_num',
        type=int,
        default=2,
        help='number of players'
    )
    parser.add_argument(
        '--num_trials',
        type=int,
        default=2000,
        help='number of trials'
    )
    parser.add_argument(
        '--step',
        type=str,
        default='before_flop',
        help='before_flop/flop/compare',
    )
    parser.add_argument(
        '--sample_count',
        type=int,
        default=4000,
        help='number of samples for each hand'
    )
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='index of hands, 0-20, each index has 64 hands'
    )
    parser.add_argument(
        '--cards',
        type=str,
        default='',
        help="cards sequences to indicate player's hand and community cards, e.g. 'CA,C5,DA,DK,S5'"
    )

    args = parser.parse_args()
    if args.step == 'before_flop':
        save_before_flop_win_rate(args.player_num, args.num_trials)
    elif args.step == 'flop':
        save_win_rate(args.index, args.player_num, args.num_trials, args.sample_count)
    elif args.step == 'compare':
        compare_win_rate(args.cards, args.player_num, args.num_trials)
