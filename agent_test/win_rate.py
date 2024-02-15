import random
import itertools
import argparse
from datetime import datetime

import pandas as pd

from new_agents.our_agent.Probability_player import get_remaining_cards
from rlcard.games.limitholdem.cardcmp import CardComb, whole_cards


def calculate_win_rate(p1_hand, community_card, player_num=2, num_trials=2000):
    wins = 0
    comm_sample_count = 5 - len(community_card)
    other_p_s_count = (player_num - 1) * 2
    left_card = get_remaining_cards(p1_hand + community_card)

    for _ in range(num_trials):
        sample_cards = random.sample(left_card, comm_sample_count + other_p_s_count)
        all_community_cards = community_card + sample_cards[:comm_sample_count]
        p_others = [sample_cards[i:i + 2] for i in range(comm_sample_count, comm_sample_count + other_p_s_count, 2)]
        wins += compare_hands(p1_hand, all_community_cards, p_others)
    return wins / num_trials


def compare_hands(p1, community_card, p_others):
    p1_value = CardComb().add_cards(p1).add_cards(community_card).count_value()
    p_other_values = [CardComb().add_cards(p).add_cards(community_card).count_value() for p in p_others]
    return 1 if all(p1_value >= p2 for p2 in p_other_values) else 0


def save_before_flop_win_rate(player_num=2, num_trials=2000):
    """calculate win rate of all possible hands before flop
    save it to pandas dataframe
    """
    p1_hands = list(itertools.combinations(whole_cards, 2))
    win_rate = []
    for p1_hand in p1_hands:
        win_rate.append({
            'hands': p1_hand,
            'win_rate': calculate_win_rate(list(p1_hand), [], player_num, num_trials),
        })

    # save to pandas dataframe
    df = pd.DataFrame(win_rate)
    df.to_csv('before_flop_win_rate{}.csv'.format(player_num), index=False)


def save_win_rate(start_idx: int, end_idx: int, player_num=2, num_trials=2000, random_sample_count=3000):
    """calculate win rate of all possible hands after flop
    save it to pandas dataframe
    """
    p1_hands = list(itertools.combinations(whole_cards, 2))
    win_rate = []
    p1_hands = p1_hands[start_idx:end_idx]
    for index, p1_hand in enumerate(p1_hands):
        left_cards = get_remaining_cards(list(p1_hand))
        # random sample community cards, it could be 3 cards or 4 cards or 5 cards
        for _ in range(random_sample_count):
            n = random.randint(3, 5)
            community_card = random.sample(left_cards, n)
            win_rate.append({
                'hands': p1_hand,
                'community_card': community_card,
                'win_rate': calculate_win_rate(list(p1_hand), community_card, player_num, num_trials),
            })
        print('current time:{} finish {}/{}'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), index + 1, len(p1_hands)))

    # save to pandas dataframe
    df = pd.DataFrame(win_rate)
    df.to_csv('flop_win_rate{}-{}-{}.csv'.format(player_num, start_idx, end_idx), index=False)


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
        help='before_flop/flop/turn/river',
    )
    parser.add_argument(
        '--sample_count',
        type=int,
        default=4000,
        help='number of samples for each hand'
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='start index of hands'
    )
    parser.add_argument(
        '--end_index',
        type=int,
        default=1326,
        help='end index of hands'
    )

    args = parser.parse_args()
    if args.step == 'before_flop':
        save_before_flop_win_rate(args.player_num, args.num_trials)
    elif args.step == 'flop':
        save_win_rate(args.start_index, args.end_index, args.player_num, args.num_trials, args.sample_count)
