# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="1469983670"
Coefficients="0"

for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    python DiffOpt/surrogate_DKL.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --which_gpu 4
  done
done
