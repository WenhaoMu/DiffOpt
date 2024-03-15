This folder contains files for DiffOpt code for design-bench problem with DKL classifier.  
  
There are 2 .sh file used to run the program. `surrogate.sh` is corresponding to run the training process of **DKL classifier**. `DiffOpt.sh` is corresponding to run the training and generating process of diffusion process with constant and exponential annealing strategy.  , the first line command is used to train the model and the second line of the command is used to generate solution. You can change the parameter 'strategy' to used different kinds of annealing strategy.
  
To run a program, you will use the corresponding task name and configs file. For example, to train a **classifier**, you may input:  
`./surrogate.sh ./configs/surrogate.cfg superconductor`  
  
To train and evaluate the **diffusion model**, you may input:  
`./diffusion.sh ./configs/diffusion.cfg superconductor`  
  