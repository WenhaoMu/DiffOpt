This folder contains files for DiffOpt code for multi-objective synthetic problem.  
  
There are 2 .sh file used to run the program. `run_classifier.sh` is corresponding to run the training process of **mlp classifier**. `run_diff_plot_ellipse.sh` is corresponding to run the training and generating process of diffusion process.  
  
To run a program, you will use the corresponding task name and configs file. For example, to train a **classifier**, you may input:  
`./run_classifier.sh ./configs/classifier_ellipse.cfg branin`  
  
To train the **diffusion model**, you may input:  
`./run_diff_plot_ellipse.sh ./configs/score_diffusion_gui_ellipse.cfg branin`  
  
To **generate solution**, you may input:  
`./run_diff_plot_ellipse.sh ./configs/score_diffusion_gui_ellipse.cfg branin`  