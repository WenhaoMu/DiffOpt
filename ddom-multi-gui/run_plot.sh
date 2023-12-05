# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
# seeds="123 234 345"
# seeds="234 345"
# temp="$3"
# seeds="123"
# seeds="345"
# seeds="456"
# seeds="1469983670"
seeds="1569983670 1469983670 314159"
# Coefficients="0 -10 -20 -30 -40 -50 -60 -70 -80 -90 -100 10 20 30 40 50 60 70 80 90 100"
# Coefficients="1 2 3 4 5 15 -1 -2 -3 -4 -5 -15"
# Coefficients="0"
# Coefficients="0 1 2 3 4 5 10 15 20 30 40 50 60 70 80 90 100"
# Coefficients="110 120 130 140 150 160 170 180 190 200"
Coefficients="200"


for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    # python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --coefficient $Coefficient --which_gpu 2
    # python design_baselines/diff/trainer_datadistance.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 0 --suffix "max_ds_conditioning"
    # python design_baselines/diff_multi/trainer_amend_multi.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --coefficient $Coefficient --which_gpu 0
    python design_baselines/diff_multi/trainer_amend_multi_plot_valid.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 0 --suffix "max_ds_conditioning"
    # python design_baselines/diff/trainer_fourier.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --suffix "max_ds_conditioning"
  done
done
