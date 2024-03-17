This folder contains files for DiffOpt code for design-bench tasks.  
  
There are 4 .sh file used to run the program.`surrogate_DKL.sh`, `surrogate_Fourier.sh`, `surrogate_MLP.sh`, `surrogate_LSTM.sh` are respectively corresponding to run the training process of **DKL surrogate**, **fourier gaussian embedding surrogate**, **MLP surrogate**, **LSTM surrogate**. `DiffOpt.sh` is corresponding to run the training and generating process of diffusion process with constant and exponential annealing strategy.  , the first line command is used to train the model and the second line of the command is used to generate solution. You can change the parameter 'strategy' to used different kinds of annealing strategy, and change parameter 'surrogate' to use different surrogate models.
  
To run a program, you will use the corresponding task name and configs file. For example, to train a **surrogate model**, you may input:  
`./surrogate_LSTM.sh ./configs/surrogate_lstm.cfg superconductor`
`./surrogate_DKL.sh ./configs/surrogate_dkl.cfg superconductor`  
`./surrogate_Fourier.sh ./configs/surrogate_fourier.cfg superconductor`  
`./surrogate_MLP.sh ./configs/surrogate_mlp.cfg superconductor`    
  
To train and use the **diffusion model**, you may input:  
`./DiffOpt.sh ./configs/diffusion.cfg superconductor`  
  