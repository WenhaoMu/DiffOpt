# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
# seeds="123 234 345"
# seeds="234 345"
# temp="$3"
# seeds="123"
# seeds="345"
# seeds="456"
seeds="1469983670"
Coefficients="-1"

for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    # python design_baselines/diff/trainer_noweight.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --coefficient $Coefficient --which_gpu 7
    python design_baselines/diff/trainer_noweight.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 7 --suffix "max_ds_conditioning"
    # python design_baselines/diff/trainer_noweight_fourier.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 7 --suffix "max_ds_conditioning"
    # python design_baselines/diff/trainer_fourier.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --suffix "max_ds_conditioning"
  done
done
