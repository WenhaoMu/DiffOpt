This folder contains files for DiffOpt code for multi-objective molecular optimization problem.  
  
There are 4 .sh file used to run the program. `surrogate_DKL.sh`, `surrogate_Fourier.sh`, `surrogate_MLP.sh` are respectively corresponding to run the training process of **DKL surrogate**, **fourier gaussian embedding surrogate**, **MLP surrogate**. `DiffOpt.sh` is used to run the training and generating process of diffusion process, the first line command is used to train the model and the second line of the command is used to generate solution. You can change the parameter 'surrogate' to used different kinds of surrogate.
  
To run a program, you will use the corresponding task name and configs file. For example, to train a **surrogate**, you may input:  
`./surrogate_DKL.sh ./configs/surrogate_dkl.cfg compound`  
`./surrogate_Fourier.sh ./configs/surrogate_fourier.cfg compound`  
`./surrogate_MLP.sh ./configs/surrogate_mlp.cfg compound`  
  
To train and use the **diffusion model**, you may input:  
`./DiffOpt.sh ./configs/diffusion.cfg compound`  
  
