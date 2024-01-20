# MLP Fourier classifier

All models are in folder 'design_baselines/diff'. If you need to train MLP classifier, the file is 'classifier_model.py' and 'classifier.py'. If you need to train fourier classifier, the file is 'classifier_model_fourier.py' and 'classifier_fourier.py'. If you need to train MLP-x0 classifier, the file is 'classifier_model_x0.py' and 'classifier_x0.py'. 

The files start with 'trainer' is the diffusion model. 'trainer_amend_fourier.py' is the diffusion model with fourier classifier, 'trainer_amend_fourier_pro.py' has a projection from gradient to score function. 'trainer_amend.py' is the diffusion model with MLP classifier, 'trainer_amend_pro.py' has a projection. 'trainer_x0test_correct.py' is the diffusion model with x0 classifier. 'trainer_x0test_correct_pro.py' has a projection. All files end with 'Qtest' is the same as the files without it except the sample number of solution is 256 instead of 512.

The file 'ddom-copy6/run_classifier.sh', 'ddom-copy6/run_classifier_fourier.sh', 'ddom-copy6/run_classifier_x0.sh' can be used to trainer classifiers. You can use './run_classifier.sh ./configs/classifier.cfg superconductor' to train a new model.