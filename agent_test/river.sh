#!/bin/bash

# 检查参数是否为空
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Error: 参数不能为空"
  exit 1
fi

# 检查参数是否为数字
re='^[0-9]+$'
if ! [[ $1 =~ $re ]] || ! [[ $2 =~ $re ]] || ! [[ $3 =~ $re ]]; then
   echo "Error: 参数必须为数字"
   exit 1
fi

export PYTHONPATH=$PWD/..:$PYTHONPATH

player_num=$1
start_index=$2
end_index=$3

for i in $(seq $start_index $end_index)
do
  log_str=$(printf "%d-%03d" $player_num $i)
  nohup python win_rate.py --step=river --player_num=$player_num --index=$i > river_$log_str.log 2>&1 &
done
