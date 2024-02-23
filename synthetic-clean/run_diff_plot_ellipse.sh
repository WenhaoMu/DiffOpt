# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="1469983670"
Coefficients="-5"


for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    # python design_baselines/diff_branin/trainer_gui_constraint_ellipse.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --coefficient $Coefficient --which_gpu 6
    python design_baselines/diff_branin/trainer_gui_constraint_ellipse.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 6
  done
done
