This folder contains files for DiffOpt code for design-bench problem with DKL classifier.  
  
There are 9 .sh file used to run the program. `run_classifier.sh`, `run_classifier_fourier.sh`, `run_classifier_lstm.sh` are corresponding to run the training process of **MLP classifier**, **Fourier Guassian classifier**, **LSTM classifier**. `run_diff.sh`, `run_diff_exp.sh`, `run_diff_lstm.sh`, `run_diff_lstm_exp.sh`, `run_diff_fourier.sh`, `run_diff_fourier_exp.sh` are corresponding to run the training and generating process of diffusion process with **MLP classifier**, **LSTM classifier**, **Fourier Gaussian classifier** and constant or exponential annealing strategy.  
  
To run a program, you will use the corresponding task name and configs file. For example, to train a **classifier**, you may input:  
`./run_classifier_fourier.sh ./configs/classifier.cfg superconductor`  
  
To train the **diffusion model**, you may input:  
`./run_diff.sh ./configs/score_diffusion.cfg superconductor`  
  
To train the **diffusion model**, you may input:  
`./run_diff.sh ./configs/score_diffusion.cfg superconductor`  