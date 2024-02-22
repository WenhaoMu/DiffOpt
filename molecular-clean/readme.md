This folder contains files for DiffOpt code for multi-objective molecular optimization problem.  
  
There are 6 .sh file used to run the program. `run_cla_dkl.sh`, `run_cla_fourier.sh`, `run_cla_mlp.sh` are respectively corresponding to run the training process of **dkl classifier**, **fourier gaussian embedding classifier**, **mlp classifier**. `run_diff_dkl.sh`, `run_diff_fourier.sh`, `run_diff_mlp.sh` are respectively corresponding to run the training and generating process of diffusion process.  
  
To run a program, you will use the corresponding task name and configs file. For example, to train a **classifier**, you may input:  
`./run_cla_dkl.sh ./configs/classifier_dkl.cfg compound`  
`./run_cla_fourier.sh ./configs/classifier_fourier.cfg compound`  
`./run_cla_dkl_mlp.sh ./configs/classifier_mlp.cfg compound`  
  
To train the **diffusion model**, you may input:  
`./run_diff_dkl.sh ./configs/score_diffusion_multi.cfg compound`  
  
To **generate solution**, you may input:  
`./run_diff_dkl.sh ./configs/score_diffusion_multi.cfg compound`  
`./run_diff_mlp.sh ./configs/score_diffusion_multi.cfg compound`  
`./run_diff_fourier.sh ./configs/score_diffusion_multi.cfg compound`  