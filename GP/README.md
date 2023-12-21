# D4CO
This folder contains file for classifier training and diffusion process on design-bench tasks. 

`cla.sh` file is the script to run the classifier training. `diff\DKL_train_regression_x0.py` is the file to train Deep Kernel Learning with Gaussian Process classifier in latent space (x0 in its name is a typo). `diff\DKL_model_regression.py` contains the DKL model. To run this script, for example, input:`./cla.sh ./configs/classifier.cfg superconductor`  

`run_diff4.sh` file is teh script to run the classifier-guided diffusion on design-bench tasks. `diff/Diffusion_DKL2.py` is the file of diffusion process. To run this script, for example, input: `./run_diff4.sh ./configs/score_diffusion.cfg superconductor`

`DKL_model` folder contains pretrained classifier model, for example named as `best_model_superconductor_x0.pth` (x0 is a typo). `DKL_solution` folder contains output of classifier-guided diffusion on design-bench tasks.
