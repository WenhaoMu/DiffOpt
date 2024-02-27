This folder contains files for DiffOpt code for design-bench problem with DKL classifier.  
  
There are 3 .sh file used to run the program. `cla.sh` is corresponding to run the training process of **DKL classifier**. `run_diff6.sh`, `run_diff7.sh` are corresponding to run the training and generating process of diffusion process with constant and exponential annealing strategy.  
  
To run a program, you will use the corresponding task name and configs file. For example, to train a **classifier**, you may input:  
`./cla.sh ./configs/classifier_ellipse.cfg superconductor`  
  
To train the **diffusion model**, you may input:  
`./run_diff6.sh ./configs/score_diffusion.cfg superconductor`  
  
To train the **diffusion model**, you may input:  
`./run_diff6.sh ./configs/score_diffusion.cfg superconductor`  