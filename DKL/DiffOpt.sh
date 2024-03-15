# !/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="1469983670"
Coefficients1="900000"
Exp_ks="0.1"
Coefficients2="0"
gpu=3


for seed in $seeds; do
  for Coefficient1 in $Coefficients1; do 
    for Coefficient2 in $Coefficients2; do 
      for Exp_k in $Exp_ks; do
        echo $seed
        echo $TASK
        echo $Coefficient1 $Coefficient2
        echo $gpu
        # python DiffOpt/Diffusion.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --coefficient $Coefficient1 $Coefficient2 --which_gpu $gpu 
        python DiffOpt/Diffusion.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient1 $Coefficient2 --which_gpu $gpu --exp_k $Exp_k --strategy 'exp' --suffix "max_ds_conditioning"
      done
    done
  done
done
