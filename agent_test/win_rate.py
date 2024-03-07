import random
import itertools
import argparse
import pandas as pd
import logging

from datetime import datetime
from new_agents.our_agent.Probability_player import get_remaining_cards, calculate_win_rate as calculate_win_rate_p2
from rlcard.games.limitholdem.cardcmp import CardComb, whole_cards
from rlcard.games.limitholdem.utils import compare_hands

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# split 0-1326 hands into seq_idx, each index has 13 hands
seq_idx = [(i * 10, (i + 1) * 10) for i in range(132)] + [(1320, 1326)]

hand_combs = list(itertools.combinations(whole_cards, 2))


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
    logging.info('start save_before_flop_win_rate, play num:{}, num trials:{}'.format(player_num, num_trials))
    p1_hands = list(itertools.combinations(whole_cards, 2))
    win_rate = []
    for index, p1_hand in enumerate(p1_hands):
        win_rate.append({
            'hands': p1_hand,
            'community_card': [],
            'win_rate': calculate_win_rate_quick(list(p1_hand), [], player_num, num_trials),
        })
        logging.info('finish {}/{}'.format(index + 1, len(p1_hands)))

    # save to pandas dataframe
    df = pd.DataFrame(win_rate)
    df.to_parquet('before_flop_win_rate{}.parquet'.format(player_num), index=False)


def save_flop_win_rate(idx: int, player_num=2, num_trials=2000):
    """calculate win rate of all possible hands after flop
    save it to pandas dataframe
    """

    def flop(left_cards: list):
        """Generate flop cards"""
        for combo in itertools.combinations(left_cards, 3):
            yield list(combo)

    save_win_rate_with_gen(idx, flop, player_num, num_trials)


def save_turn_win_rate(idx: int, player_num=2, num_trials=2000, random_sample_count=20000):
    """calculate win rate of all possible hands after turn
    save it to pandas dataframe
    """

    def turn(left_cards: list):
        """Generate turn cards"""
        for _ in range(random_sample_count):
            yield random.sample(left_cards, 4)

    save_win_rate_with_gen(idx, turn, player_num, num_trials)


def save_river_win_rate(idx: int, player_num=2, num_trials=2000, random_sample_count=100000):
    """calculate win rate of all possible hands after river
    save it to pandas dataframe
    """

    def river(left_cards: list):
        """Generate river cards"""
        for _ in range(random_sample_count):
            yield random.sample(left_cards, 5)

    save_win_rate_with_gen(idx, river, player_num, num_trials)


def save_win_rate_with_gen(idx: int, generator, player_num=2, num_trials=2000):
    """calculate win rate of all possible hands with community card generated func
    save it to pandas dataframe
    """
    start_idx, end_idx = seq_idx[idx]
    logging.info('start sav rate of {}, player num: {}, idx:{}，{}-{}'.
                 format(generator.__name__, player_num, idx, start_idx, end_idx))
    win_rate = []
    p1_hands = hand_combs[start_idx:end_idx]
    count = 0
    for index, p1_hand in enumerate(p1_hands):
        p1_hand_list = list(p1_hand)
        left_cards = get_remaining_cards(p1_hand_list)
        for community_card in generator(left_cards):
            win_rate.append({
                'hands': p1_hand_list,
                'community_card': community_card,
                'win_rate': calculate_win_rate_quick(p1_hand_list, community_card, player_num, num_trials),
            })
            count += 1
            if count % 1000 == 0:
                logging.info('simulate count:{}'.format(count))
        logging.info('finish {}/{}'.format(index + 1, len(p1_hands)))

    # save to pandas dataframe
    df = pd.DataFrame(win_rate)
    df.to_parquet('win_rate-{}-{}-{:0>5d}-{:0>5d}.parquet'.format(generator.__name__, player_num, start_idx, end_idx),
                  index=False)


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


def calculate_win_rate_with_time(fn, p1_hand, community_card, player_num=2, num_trials=2000, count=1):
    time1 = datetime.now()
    for _ in range(count):
        win_rate = fn(p1_hand, community_card, player_num, num_trials)
    time2 = datetime.now()
    print('func name:{} win rate current:{}, use time:{}:'.format(fn.__name__, win_rate, (time2 - time1) / count))


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
        help='compare/before_flop/flop/turn/river',
    )
    parser.add_argument(
        '--turn_sample_count',
        type=int,
        default=50000,
        help='number of samples for turn community cards'
    )
    parser.add_argument(
        '--river_sample_count',
        type=int,
        default=150000,
        help='number of samples for river community cards'
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
    if args.step == 'compare':
        compare_win_rate(args.cards, args.player_num, args.num_trials)
    elif args.step == 'before_flop':
        save_before_flop_win_rate(args.player_num, args.num_trials)
    elif args.step == 'flop':
        save_flop_win_rate(args.index, args.player_num, args.num_trials)
    elif args.step == 'turn':
        save_turn_win_rate(args.index, args.player_num, args.num_trials, args.turn_sample_count)
    elif args.step == 'river':
        save_river_win_rate(args.index, args.player_num, args.num_trials, args.river_sample_count)
