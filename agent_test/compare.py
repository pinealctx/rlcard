import random
import argparse
import logging

from agent_test.win_rate import compare_hands
from rlcard.games.limitholdem.cardcmp import CardComb, whole_cards
from agent_test.seven_card_encode import seven_cards_encode, load_cards_values
from agent_test.win_rate import calculate_win_rate_with_time, calculate_win_rate_quick, calculate_win_rate_slow
from new_agents.our_agent.Probability_player import get_remaining_cards

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
global_kv = None


def set_global_kv(kv):
    global global_kv
    global_kv = kv


def compare_players_card_use_value(hands):
    p1_value = CardComb().add_cards(hands[0]).count_value()
    pother_values = [CardComb().add_cards(ps).count_value() for ps in hands[1:]]
    return 1 if all(p1_value >= p_value for p_value in pother_values) else 0


def compare_players_card(hands):
    ws = compare_hands(hands)
    return ws[0]


def compare_players_card_use_hit(hands):
    p1 = global_kv[seven_cards_encode(hands[0])]
    pk = [global_kv[seven_cards_encode(hand)] for hand in hands[1:]]
    return 1 if all(p1 >= p for p in pk) else 0


def compare_players_card_with_others(p1, comm, others):
    p1 = global_kv[seven_cards_encode(p1 + comm)]
    pk = [global_kv[seven_cards_encode(hand + comm)] for hand in others]
    return 1 if all(p1 >= p for p in pk) else 0


def compare_players(player_num: int, num: int):
    if player_num < 2 or player_num > 10:
        raise ValueError("player_num must between 2 and 10")
    sample_count = player_num * 2 + 5
    count = 0
    for _ in range(num):
        cards = random.sample(whole_cards, sample_count)
        hands = [cards[i:i + 2] for i in range(0, player_num * 2, 2)]
        community = cards[player_num * 2:]
        hands = [hand + community for hand in hands]
        v1 = compare_players_card(hands)
        v2 = compare_players_card_use_value(hands)
        v3 = compare_players_card_use_hit(hands)
        if not (v1 == v2 and v2 == v3):
            logging.error("error hand:{} v1:{} v2:{} v3:{}".format(hands, v1, v2, v3))
            return
        count += 1
        if count % 100000 == 0:
            logging.info("count:{}".format(count))


def calculate_win_rate_use_table(p1_hand, community_card, player_num=2, num_trials=2000):
    wins = 0
    comm_sample_count = 5 - len(community_card)
    other_p_s_count = (player_num - 1) * 2
    left_card = get_remaining_cards(p1_hand + community_card)

    for _ in range(num_trials):
        sample_cards = random.sample(left_card, comm_sample_count + other_p_s_count)
        all_community_cards = community_card + sample_cards[:comm_sample_count]
        p_others = [sample_cards[i:i + 2] for i in range(comm_sample_count, comm_sample_count + other_p_s_count, 2)]
        wins += compare_players_card_with_others(p1_hand, all_community_cards, p_others)
    return wins / num_trials


def compare_win_rate_calculate_time(p1_hand, com_cards, player_num=2, num_trials=5000):
    calculate_win_rate_with_time(calculate_win_rate_use_table, p1_hand, com_cards, player_num, num_trials)
    calculate_win_rate_with_time(calculate_win_rate_quick, p1_hand, com_cards, player_num, num_trials)
    calculate_win_rate_with_time(calculate_win_rate_slow, p1_hand, com_cards, player_num, num_trials)


def before_flop_win_rate(player_num=2, num_trials=5000):
    p1_hands = random.sample(whole_cards, 2)
    com_cards = []
    compare_win_rate_calculate_time(p1_hands, com_cards, player_num, num_trials)


def flop_win_rate(player_num=2, num_trials=5000):
    p1_hands = random.sample(whole_cards, 2)
    com_cards = random.sample(whole_cards, 3)
    compare_win_rate_calculate_time(p1_hands, com_cards, player_num, num_trials)


def turn_win_rate(player_num=2, num_trials=5000):
    p1_hands = random.sample(whole_cards, 2)
    com_cards = random.sample(whole_cards, 4)
    compare_win_rate_calculate_time(p1_hands, com_cards, player_num, num_trials)


def river_win_rate(player_num=2, num_trials=5000):
    p1_hands = random.sample(whole_cards, 2)
    com_cards = random.sample(whole_cards, 5)
    compare_win_rate_calculate_time(p1_hands, com_cards, player_num, num_trials)


def main():
    parser = argparse.ArgumentParser("compare players")
    parser.add_argument(
        '--player-num',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num',
        type=int,
        default=1000000,
    )
    parser.add_argument(
        '--f',
        type=str,
        default="seven_cards_values.pkl",
        help='file name'
    )
    parser.add_argument(
        "--action",
        type=str,
        default="compare",
        help="compare/before_flop/flop/turn/river"
    )
    args = parser.parse_args()
    logging.info("args:{}".format(args))
    logging.info("load cards values from file:{}".format(args.f))
    kv = load_cards_values(args.f)
    set_global_kv(kv)
    logging.info("load cards values success")

    if args.action == "compare":
        compare_players(args.player_num, args.num)
    if args.action == "before_flop":
        before_flop_win_rate(args.player_num, args.sample)
    if args.action == "flop":
        flop_win_rate(args.player_num, args.sample)
    if args.action == "turn":
        turn_win_rate(args.player_num, args.sample)
    if args.action == "river":
        river_win_rate(args.player_num, args.sample)


if __name__ == '__main__':
    main()
