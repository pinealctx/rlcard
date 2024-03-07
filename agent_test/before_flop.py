import argparse
import logging
import itertools
import os
import multiprocessing
import pandas as pd

from rlcard.games.limitholdem.cardcmp import whole_cards
from agent_test.seven_card_table import load_cards_value_table, calculate_win_rate_use_table

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def calculate_win_rate_before_flop(idx, p1_hands_list, num_trials=30000):
    pd_data = []
    count = 0
    for p1_hands in p1_hands_list:
        win_rates = []
        p1_hand_as_list = list(p1_hands)
        for i in range(2, 11):
            win_rate = calculate_win_rate_use_table(p1_hand_as_list, [], player_num=i, num_trials=num_trials)
            win_rates.append(win_rate)
        # p1_hands is col1, win_rates is col2-11
        pd_data.append([p1_hand_as_list] + win_rates)
        count += 1
        if count % 8 == 0:
            logging.info("cpu{} process percent: {:.2f}%".format(idx, count / len(p1_hands_list) * 100))
    return pd_data


def handle_before_flop(idx, p1_hands_list, file_name='before_flop', num_trials=30000):
    logging.info("start handle_before_flop, idx:{}, num trials:{}".format(idx, num_trials))
    pd_data = calculate_win_rate_before_flop(idx, p1_hands_list, num_trials=num_trials)
    logging.info("save to parquet file: {}".format(file_name + str(idx) + '.parquet'))
    df = pd.DataFrame(pd_data)
    df.to_parquet(file_name + str(idx) + '.parquet', index=False)
    logging.info("handle_before_flop done, idx:{}".format(idx))


def main():
    parser = argparse.ArgumentParser("Calculate win rate of before flop")
    parser.add_argument('--num_trials', type=int, default=30000, help='number of trials')
    parser.add_argument('--f', type=str, default="seven_cards_values.pkl", help='file name')
    parser.add_argument('--s', type=str, default="before_flop", help='save file name')
    args = parser.parse_args()
    logging.info("args:{}".format(args))
    load_cards_value_table(args.f)
    hands_lst = list(itertools.combinations(whole_cards, 2))
    cpu_count = os.cpu_count()
    logging.info("cpu count:{}".format(cpu_count))
    chunk_size = len(hands_lst) // cpu_count
    chunk_size = chunk_size + 1 if len(hands_lst) % cpu_count != 0 else chunk_size
    logging.info("chunk size:{}".format(chunk_size))
    plist = []
    for i in range(cpu_count):
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < len(hands_lst) else len(hands_lst)
        hand_slice = hands_lst[start:end]
        p = multiprocessing.Process(target=handle_before_flop, args=(i, hand_slice, args.s, args.num_trials))
        plist.append(p)
        p.start()
    for p in plist:
        p.join()
    logging.info("all done")


if __name__ == '__main__':
    main()
