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
# seeds="42 98765 314159 271828 1469983670 1369983670 1569983670 1669983670"
# seeds="98765 314159 271828 1469983670 1369983670 1569983670 1669983670"
# Coefficients="0"
# seeds="42 98765 314159 271828 1469983670"
# seeds="1369983670 1569983670 1669983670"
# Coefficients="0 1 2 3 4 5 6 7 8 9 10 15 20 30 40 50 60 70 80 90 100 -1 -2 -3 -4 -5 -6 -10 -15"
# Coefficients="0 1 2 3 4 5 10 15 20 30 40 50 60 70 80 90 100"
Coefficients="0 1000000 2000000 3000000 4000000 5000000 10000000 15000000 20000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000 100000000"

for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    # python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --coefficient $Coefficient --which_gpu 2
    python diff/trainer_x0test_correct_Qtest_GP.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 2 --suffix "max_ds_conditioning"
    # python design_baselines/diff/trainer_fourier.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --suffix "max_ds_conditioning"
  done
done
