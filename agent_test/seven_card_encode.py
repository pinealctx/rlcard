import itertools
import pickle
import argparse
import time
import pandas as pd
import logging

from cykhash import Int64toInt64Map
from rlcard.games.limitholdem.cardcmp import CardComb, whole_cards, seven_cards_encode

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def handle_cards_values(cards_lst, file_name):
    kv = {}
    count = 0
    size = len(cards_lst)
    for cards in cards_lst:
        cs = list(cards)
        k = seven_cards_encode(cs)
        v = CardComb().add_cards(cs).count_value()
        kv[k] = v
        count += 1
        if count % 100000 == 0:
            logging.info("process percent: {:.2f}%".format(count / size * 100))
    # save to file
    with open(file_name, 'wb') as f:
        pickle.dump(kv, f)


def save_cards_2_parquets(cards_lst, file_name, chunk_size=20000000):
    size = len(cards_lst)
    chunk_items = size // chunk_size
    chunk_items = chunk_items + 1 if size % chunk_size != 0 else chunk_items
    count = 0
    for i in range(chunk_items):
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < size else size
        kvs = []
        for cards in cards_lst[start:end]:
            cs = list(cards)
            k = seven_cards_encode(cs)
            v = CardComb().add_cards(cs).count_value()
            kvs.append({'k': k, 'v': v})
            count += 1
            if count % 500000 == 0:
                logging.info("process percent: {:.2f}%".format(count / size * 100))
        # save to file
        df = pd.DataFrame(kvs)
        logging.info("save to parquet file: {}".format(file_name + str(i) + '.parquet'))
        df.to_parquet(file_name + str(i) + '.parquet', index=False)
        logging.info("save to parquet file: {} done".format(file_name + str(i) + '.parquet'))
    logging.info("process count: {}".format(count))


def load_cards_values(file_name, total_size=133784560, chunk_size=20000000):
    chunk_idx = total_size // chunk_size
    chunk_idx = chunk_idx + 1 if total_size % chunk_size != 0 else chunk_idx
    cykhash_map = Int64toInt64Map()
    for i in range(chunk_idx):
        logging.info("load parquet file: {}".format(file_name + str(i) + '.parquet'))
        df = pd.read_parquet(file_name + str(i) + '.parquet')
        logging.info("load parquet file: {} size:{}".format(file_name + str(i) + '.parquet', len(df)))
        for _, row in df.iterrows():
            cykhash_map[row['k']] = row['v']
        logging.info("load parquet file: {} done".format(file_name + str(i) + '.parquet'))
    return cykhash_map


def main():
    parser = argparse.ArgumentParser("handle cards values")
    parser.add_argument(
        '--f',
        type=str,
        default="seven_cards_values.pkl",
        help='file name'
    )
    parser.add_argument(
        '--action',
        type=str,
        default="save",
        help='save/snap/load'
    )
    args = parser.parse_args()
    if args.action == "save":
        logging.info("start to save cards values to parquet")
        hands_lst = list(itertools.combinations(whole_cards, 7))
        logging.info("combine hands list size is:{}".format(len(hands_lst)))
        save_cards_2_parquets(hands_lst, args.f)
    elif args.action == "snap":
        hands_lst = list(itertools.combinations(whole_cards, 7))
        handle_cards_values(hands_lst, args.f)
    elif args.action == "load":
        logging.info("start to load cards values from parquet")
        ds = load_cards_values(args.f)
        logging.info("dict size is:{}".format(len(ds)))
        time.sleep(120)


if __name__ == '__main__':
    main()
