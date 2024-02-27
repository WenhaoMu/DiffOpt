# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="123"
Exp_ks="0.1 0.01 0.05 0.08"
Coefficients="100"


for seed in $seeds; do
  for Coefficient in $Coefficients; do
    for Exp_k in $Exp_ks; do
      echo $seed
      echo $TASK
      echo $Coefficient
      python design_baselines/diff/trainer_amend_Qtest_exp_t.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 6 --suffix "max_ds_conditioning"  --exp_k $Exp_k
    done
  done
done
