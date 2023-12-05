This folder contains file about the training of classifer, diffusion process and molecular evaluation on drug design problem.

`run_diff.sh` file is a script for classifer training and diffusion process. To run classifier-guided diffusion, for example, input `./run_diff.sh ./configs/score_diffusion_multi.cfg compound`.

`/design_baselines/diff_multi/trainer_amend_multi_DKL_valid.py' is the file to do classifier-guided diffusion. The pretrained 3 classifier model corresponding 3 objective is saved in fold `DKL_model`. The output of diffusion is saved in `solution/compound_amend_multi_5.csv`. The output of x is save as `molecular_Test.py`, copy it to folder `design_baselines/diff_multi/molecular_discovery_problem` and run `generation_validtest.py` to get the validity of the generation of x.
