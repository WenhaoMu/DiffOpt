# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="1469983670"
Coefficients1="900000"
Coefficients2="0"
gpu=2


for seed in $seeds; do
  for Coefficient1 in $Coefficients1; do 
    for Coefficient2 in $Coefficients2; do 
      echo $seed
      echo $TASK
      echo $Coefficient1 $Coefficient2
      echo $gpu
      python diff/Diffusion_DKL2.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient1 $Coefficient2 --which_gpu $gpu --suffix "max_ds_conditioning"
      # python diff/Diffusion_DKL2.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --coefficient $Coefficient1 $Coefficient2 --which_gpu $gpu 
    done
  done
done
