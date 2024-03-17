#!/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="456"

for seed in $seeds; do
  echo $seed
  echo $TASK
  python DiffOpt/surrogate_DKL.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --which_gpu 3
done
