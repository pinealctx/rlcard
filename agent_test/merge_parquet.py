import argparse
import os
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def merge_parquet_files(file_name):
    # list all parquet files such as "before_flop0.parquet/before_flo1.parquet/before_flo2.parquet"
    # list current directory
    files = os.listdir()
    # filter : file_name+number+.parquet, such as "before_flop0.parquet", despite the file_name is "before_flop.parquet"
    files = [f for f in files if f.startswith(file_name) and f.endswith('.parquet') and file_name != f.split('.')[0]]
    logging.info("parquet files:{}".format(files))
    # read all parquet files
    df_lst = [pd.read_parquet(f) for f in files]
    # concat all dataframes
    df = pd.concat(df_lst)
    logging.info("all item count is {}".format(len(df)))
    # save to parquet file
    logging.info("save to parquet file: {}".format(file_name + '.parquet'))
    df.to_parquet(file_name + '.parquet', index=False)
    logging.info("save to parquet file: {} done".format(file_name + '.parquet'))


def main():
    parser = argparse.ArgumentParser("merge parquet files")
    parser.add_argument('--f', type=str, default="before_flop", help='file name')
    args = parser.parse_args()
    logging.info("args:{}".format(args))
    merge_parquet_files(args.f)


if __name__ == '__main__':
    main()
