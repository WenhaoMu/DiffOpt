# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="456"
Coefficients="100"


for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    python design_baselines/diff/trainer_amend_fourier_Qtest.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 2 --suffix "max_ds_conditioning"
  done
done
