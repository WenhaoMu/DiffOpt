# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="123"
Coefficients="100"


for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    # python design_baselines/diff/trainer_amend_Qtest.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --coefficient $Coefficient --which_gpu 0
    python design_baselines/diff/trainer_amend_Qtest_t.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 7 --suffix "max_ds_conditioning"
  done
done
