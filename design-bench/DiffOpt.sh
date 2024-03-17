# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="1469983670"
Coefficients="1000"


for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    # python DiffOpt/DiffOpt.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --coefficient $Coefficient --which_gpu 0
    python DiffOpt/DiffOpt.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient  --strategy 'cons'  --surrogate 'dkl' --which_gpu 7 --suffix "max_ds_conditioning"
  done
done
