#!/usr/bin/env bash

CONFIG="$1"
TASK="$2"
# seeds="123 234 345"
seeds="123"
# seeds="456"
# seeds="678"
# seeds="123 234 345"
# seeds="234 345 456 567"

for seed in $seeds; do
  echo $seed
  echo $TASK
  # python design_baselines/diff_branin/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK
#   python design_baselines/diff_branin/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK 
  # python design_baselines/diff_branin/classifier.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK 
  python design_baselines/diff/datadistance.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --which_gpu 3
  # python design_baselines/diff/classifier_fourier.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK
done
