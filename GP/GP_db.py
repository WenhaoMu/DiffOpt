import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

import json
import os
import random
import string
import uuid
import shutil
import pickle

import copy

from typing import Optional, Union
from pprint import pprint

import configargparse

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import matplotlib.pyplot as plt
import os

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    import design_bench

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset
    from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset

    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
    # from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pickle as pkl

import torch
from torch.utils.data import Dataset, DataLoader

# from nets import DiffusionTest, DiffusionScore
from diff.util import TASKNAME2TASK, configure_gpu, set_seed, get_weights
# from forward import ForwardModel

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

seed=1469983670
set_seed(seed)

task = design_bench.make(TASKNAME2TASK['superconductor'])
# task = design_bench.make(TASKNAME2TASK['chembl'])



task.map_normalize_x()
task.map_normalize_y()
dataset = task.dataset
dataset.subsample(max_samples=50)

# print(task.y)

# if task.is_discrete:
#     task.map_to_logits()

train_X = torch.from_numpy(task.x)
train_Y = torch.from_numpy(task.y)

print('TYPE: ', type(train_X))
print("Shape X: ", train_X.shape)
print("Shape Y: ", train_Y.shape)
print("X", train_X.max(), train_X.min())
print("Y", train_Y.max(), train_Y.min())

gp_init = SingleTaskGP(train_X, train_Y)
init_state_dict = copy.deepcopy(gp_init.state_dict())

gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll);

trained_state_dict = gp.state_dict()

is_trained = False
for (key1, param1), (key2, param2) in zip(init_state_dict.items(), trained_state_dict.items()):
    if not torch.equal(param1, param2):
        is_trained = True
        break

if is_trained:
    os.makedirs('classifier', exist_ok=True)
    # torch.save(gp.state_dict(), "./classifier/gp_super50.pth")
else:
    print("The model has not been trained!")

from botorch.acquisition import UpperConfidenceBound

UCB = UpperConfidenceBound(gp, beta=0.1)

from botorch.optim import optimize_acqf

# bounds = torch.stack([torch.zeros(86), torch.ones(86)])
# bounds = torch.stack([train_X.min(dim=0).values, train_X.max(dim=0).values])
lower_bound = -1000 * torch.ones(train_X.shape[1])
upper_bound = 1000 * torch.ones(train_X.shape[1])
bounds = torch.stack([lower_bound, upper_bound])


candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)

print("Candidate: ", candidate)
print("acq_value: ", acq_value)
