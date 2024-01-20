#!/usr/bin/env bash

CONFIG="$1"
TASK="$2"
# seeds="123 234 345"
# seeds="123"
# seeds="234"
# seeds="345"
# seeds="456"
# seeds="567"
# temp="$3"
seeds="1469983670"
Coefficients="0 1 2 3 4 5 10 15 20 30 40 50 60 70 80 90 100 -1 -2 -3 -4 -5 -10 -15 -20 -30 -40 -50"

for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    # python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK
    # python design_baselines/diff/generation_plot.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --suffix "max_ds_conditioning"
    python design_baselines/diff/generation_mean_fourier.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 7 --suffix "max_ds_conditioning"
  done
done
