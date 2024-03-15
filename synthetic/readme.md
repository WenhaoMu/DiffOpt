This folder contains files for DiffOpt code for synthetic task.  
  
There are 2 .sh file used to run the program. `surrogate.sh` is corresponding to run the training process of **mlp classifier**. `diffusion.sh` is corresponding to run the training and generating process of diffusion process, the first line command is used to train the model and the second line of the command is used to generate solution.
  
To run a program, you will use the corresponding task name and configs file. For example, to train a **classifier**, you may input:  
`./surrogate.sh ./configs/surrogate.cfg branin`  
  
To train and use the **diffusion model**, you may input:  
`./diffusion.sh ./configs/diffusion.cfg branin`  
  
