# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="1469983670"
# seeds="42 98765 314159 271828 1469983670"
# Coefficients1="1000 5000 10000 30000 50000 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000 1100000 1200000 1300000 1400000 1500000"
Coefficients1="100000 200000 "
Coefficients2="1000"
gpu=5


for seed in $seeds; do
  for Coefficient1 in $Coefficients1; do 
    for Coefficient2 in $Coefficients2; do 
      echo $seed
      echo $TASK
      # echo $Coefficient
      echo $Coefficient1 $Coefficient2
      echo $gpu
      python diff/Diffusion_DKL2.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient1 $Coefficient2 --which_gpu $gpu --suffix "max_ds_conditioning"
    done
  done
done
