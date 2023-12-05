import json
import os
import random
import string
import uuid
import shutil
import pickle

import copy

import torch.nn as nn

from typing import Optional, Union
from pprint import pprint

import configargparse

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import matplotlib.pyplot as plt

from forward import ForwardModel
from classifier_model import Classifier

from DGP_model import DeepGaussianProcess,Layer

# from LBFGS import FullBatchLBFGS

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

from nets import DiffusionTest, DiffusionScore
from util import TASKNAME2TASK, configure_gpu, set_seed, get_weights
from forward import ForwardModel

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

args_filename = "args.json"
checkpoint_dir = "checkpoints"
wandb_project = "sde-flow"

import torch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood

class RvSDataset(Dataset):

    def __init__(self, task, x, y, w=None, device=None, mode='train'):
        self.task = task
        self.device = device
        self.mode = mode
        self.x = x
        self.y = y
        self.w = w

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        if self.w is not None:
            w = torch.tensor(self.w[idx])
        else:
            w = None
        '''
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
            if w is not None:
                w = w.to(self.device)
        '''
        if w is None:
            return x, y
        else:
            return x, y, w


def temp_get_super_y(task):
    y = task.y.reshape(-1)
    sorted_y_idx = np.argsort(y)

    super_task = design_bench.make(TASKNAME2TASK["superconductor"])
    super_task.map_normalize_y()

    super_y = super_task.y.reshape(-1)
    super_y = np.sort(super_y)
    super_y = super_y[sorted_y_idx]

    return super_y


def split_dataset(task, val_frac=None, device=None, temp=None):
    length = task.y.shape[0]
    shuffle_idx = np.arange(length)
    shuffle_idx = np.random.shuffle(shuffle_idx)

    if task.is_discrete:
        task.map_to_logits()
        x = task.x[shuffle_idx]
        x = x.reshape(x.shape[1:])
        x = x.reshape(x.shape[0], -1)
    else:
        x = task.x[shuffle_idx]

    # y = temp_get_super_y(task)
    y = task.y
    y = y[shuffle_idx]
    if not task.is_discrete:
        x = x.reshape(-1, task.x.shape[-1])
    y = y.reshape(-1, 1)
    # w = get_weights(y, base_temp=0.03 * length)
    # w = get_weights(y, base_temp=0.1)
    w = get_weights(y, temp=temp)

    # TODO: Modify
    # full_ds = DKittyMorphologyDataset()
    # y = (y - full_ds.y.min()) / (full_ds.y.max() - full_ds.y.min())

    print(w)
    print(w.shape)

    if val_frac is None:
        val_frac = 0

    val_length = int(length * val_frac)
    train_length = length - val_length

    train_dataset = RvSDataset(
        task,
        x[:train_length],
        y[:train_length],
        # None,
        w[:train_length],
        device,
        mode='train')
    val_dataset = RvSDataset(
        task,
        x[train_length:],
        y[train_length:],
        # None,
        w[train_length:],
        device,
        mode='val')

    return train_dataset, val_dataset



import torch
from torch.utils.data import DataLoader

def get_loaders(task, val_frac, device, batch_size, num_workers, temp):
    train_dataset, val_dataset = split_dataset(task, val_frac, device, temp)

    train_loader = DataLoader(train_dataset,
                              shuffle=True, 
                              num_workers=num_workers,
                              batch_size=batch_size)
    
    val_loader = DataLoader(val_dataset,
                            shuffle=False,  
                            num_workers=num_workers,
                            batch_size=batch_size)

    return train_loader, val_loader





def log_args(
    args: configargparse.Namespace,
    wandb_logger: pl.loggers.wandb.WandbLogger,
) -> None:
    """Log arguments to a file in the wandb directory."""
    wandb_logger.log_hyperparams(args)

    args.wandb_entity = wandb_logger.experiment.entity
    args.wandb_project = wandb_logger.experiment.project
    args.wandb_run_id = wandb_logger.experiment.id
    args.wandb_path = wandb_logger.experiment.path

    out_directory = wandb_logger.experiment.dir
    pprint(f"out_directory: {out_directory}")
    args_file = os.path.join(out_directory, args_filename)
    with open(args_file, "w") as f:
        try:
            json.dump(args.__dict__, f)
        except AttributeError:
            json.dump(args, f)

def train_one_epoch(model, likelihood, optimizer, train_X, train_Y):
    model.train()
    optimizer.zero_grad()

    output = model(train_X)

    loss = -likelihood.log_marginal(train_Y, output).mean()

    loss.backward()

    optimizer.step()

    return loss.item()



def validate(model, likelihood, val_X, val_Y):
    model.eval()
    with torch.no_grad():
        output = model(val_X)
        loss = -likelihood.log_marginal(val_Y, output).mean()
    return loss.item()

def test(model, likelihood, test_X, test_Y):
    model.eval()
    with torch.no_grad():
        output = model(test_X)
        loss = -likelihood.log_marginal(test_Y, output).mean()
    return loss.item()




def run_classifier(
    taskname: str,
    seed: int,
    wandb_logger: pl.loggers.wandb.WandbLogger,
    args,
    device=None,
):
    epochs = args.epochs
    max_steps = args.max_steps
    train_time = args.train_time
    hidden_size = args.hidden_size
    depth = args.depth
    learning_rate = args.learning_rate
    auto_tune_lr = args.auto_tune_lr
    dropout_p = args.dropout_p
    checkpoint_every_n_epochs = args.checkpoint_every_n_epochs
    checkpoint_every_n_steps = args.checkpoint_every_n_steps
    checkpoint_time_interval = args.checkpoint_time_interval
    batch_size = args.batch_size
    val_frac = args.val_frac
    use_gpu = args.use_gpu
    device = device
    num_workers = args.num_workers
    vtype = args.vtype
    T0 = args.T0
    normalise_x = args.normalise_x
    normalise_y = args.normalise_y
    debias = args.debias
    score_matching = args.score_matching

    set_seed(seed)
    task = design_bench.make(TASKNAME2TASK[taskname])
    if taskname != 'tf-bind-10':
        task = design_bench.make(TASKNAME2TASK[taskname])
    else:
        task = design_bench.make(TASKNAME2TASK[taskname],
                                 dataset_kwargs={"max_samples": 10000})
    if task.is_discrete:
        task.map_to_logits()
    if normalise_x:
        task.map_normalize_x()
    if normalise_y:
        task.map_normalize_y()

    output_device = torch.device("cuda:0")
    print("DEVICE:", device)
    n_devices = torch.cuda.device_count()
    
    train_loader, val_loader = get_loaders(task=task,
                                       val_frac=val_frac,
                                       device=output_device,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       temp=args.temp)
    
    train_X = torch.from_numpy(task.x)
    train_Y = torch.from_numpy(task.y)
    # inducing_points1 = torch.randn(train_X.shape[0], train_X.shape[1]).to(output_device)

    # Layers = Layer(inducing_points1, n_devices=n_devices, output_device=output_device)
    model = DeepGaussianProcess(train_X.shape, train_Y.shape[-1], n_devices=n_devices, output_device=output_device)
    model = model.to(output_device)
    # model = Layers.to(output_device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = nn.DataParallel(model)
    else:
        print("Single GPU")

    likelihood = GaussianLikelihood().to(output_device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = FullBatchLBFGS(model.parameters(), lr=0.1)
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        likelihood.train()
        train_loss = 0.0
        num_train = 0

        for train_X, train_Y,_ in train_loader:
            train_X = train_X.to(output_device)
            train_Y = train_Y.to(output_device)
            print("TRAIN DATA:", type(train_X), train_X.shape, type(train_Y), train_Y.shape)
            model.train()
            likelihood.train()
            optimizer.zero_grad()
            print("DEVICE OF TRAIN X: ", train_X.device)
            output = model(train_X)
            loss = -likelihood.log_marginal(train_Y, output).sum()
            # loss = -mll(output, train_Y).sum()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_train += train_X.shape[0]
        
        train_loss = train_loss / num_train

        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

        best_val_loss = float('inf')
        best_model_state = None
        model.eval()
        val_loss = 0.0
        num_val = 0
        for val_X, val_Y,_ in val_loader:
            val_X = val_X.to(device)
            val_Y = val_Y.to(device)
            print("VAL DATA:", type(val_X), val_X.shape, type(val_Y), val_Y.shape)
            # optimizer.zero_grad()
            with torch.no_grad():
                output = model(val_X)
                
                loss = -likelihood.log_marginal(val_Y, output).sum()
            
            # loss.backward()
            
            optimizer.step()
            val_loss += loss.item()
            num_train += val_X.shape[0]
        
        val_loss = val_loss / num_train
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state
            torch.save(model.state_dict(), os.path.join("DGP_models", "best_model.pth"))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    


    # is_trained = False
    # for (key1, param1), (key2, param2) in zip(init_state_dict.items(), trained_state_dict.items()):
    #     if not torch.equal(param1, param2):
    #         is_trained = True
    #         break

    # if is_trained:
    #     os.makedirs('classifier', exist_ok=True)
    #     torch.save(gp.state_dict(), "./classifier/gp_model.pth")
    # else:
    #     print("The model has not been trained!")


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    # configuration
    parser.add_argument(
        "--configs",
        default=None,
        required=False,
        is_config_file=True,
        help="path(s) to configuration file(s)",
    )
    parser.add_argument('--mode',
                        choices=['train', 'eval'],
                        default='train',
                        required=True)
    parser.add_argument('--task',
                        choices=list(TASKNAME2TASK.keys()),
                        required=True)
    # reproducibility
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help=
        "sets the random seed; if this is not specified, it is chosen randomly",
    )
    parser.add_argument("--condition", default=0.0, type=float)
    parser.add_argument("--lamda", default=0.0, type=float)
    parser.add_argument("--temp", default='90', type=str)
    parser.add_argument("--suffix", type=str, default="")
    # experiment tracking
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--score_matching", action='store_true', default=False)
    # training
    train_time_group = parser.add_mutually_exclusive_group(required=True)
    train_time_group.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="the number of training epochs.",
    )
    train_time_group.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help=
        "the number of training gradient steps per bootstrap iteration. ignored "
        "if --train_time is set",
    )
    train_time_group.add_argument(
        "--train_time",
        default=None,
        type=str,
        help="how long to train, specified as a DD:HH:MM:SS str",
    )
    parser.add_argument("--num_workers",
                        default=1,
                        type=int,
                        help="Number of workers")
    checkpoint_frequency_group = parser.add_mutually_exclusive_group(
        required=True)
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_epochs",
        default=None,
        type=int,
        help="the period of training epochs for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_steps",
        default=None,
        type=int,
        help="the period of training gradient steps for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_time_interval",
        default=None,
        type=str,
        help="how long between saving checkpoints, specified as a HH:MM:SS str",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        required=True,
        help="fraction of data to use for validation",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="place networks and data on the GPU",
    )
    parser.add_argument('--simple_clip', action="store_true", default=False)
    parser.add_argument("--which_gpu",
                        # default=0,
                        # type=int,
                        help="which GPU to use")
    parser.add_argument(
        "--normalise_x",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--normalise_y",
        action="store_true",
        default=False,
    )

    # i/o
    parser.add_argument('--dataset',
                        type=str,
                        choices=['mnist', 'cifar'],
                        default='mnist')
    parser.add_argument('--dataroot', type=str, default='~/.datasets')
    parser.add_argument('--saveroot', type=str, default='~/.saved')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--num_steps',
                        type=int,
                        default=1000,
                        help='number of integration steps for sampling')

    # optimization
    parser.add_argument('--T0',
                        type=float,
                        default=1.0,
                        help='integration time')
    parser.add_argument('--vtype',
                        type=str,
                        choices=['rademacher', 'gaussian'],
                        default='rademacher',
                        help='random vector for the Hutchinson trace estimator')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=1.)

    # model
    parser.add_argument(
        '--real',
        type=eval,
        choices=[True, False],
        default=True,
        help=
        'transforming the data from [0,1] to the real space using the logit function'
    )
    parser.add_argument(
        '--debias',
        action="store_true",
        default=False,
        help=
        'using non-uniform sampling to debias the denoising score matching loss'
    )

    # TODO: remove
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        help="learning rate for each gradient step",
    )
    parser.add_argument(
        "--auto_tune_lr",
        action="store_true",
        default=False,
        help=
        "have PyTorch Lightning try to automatically find the best learning rate",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=False,
        help="size of hidden layers in policy network",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=False,
        help="number of hidden layers in policy network",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        required=False,
        help="dropout probability",
        default=0,
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--beta_max",
        type=float,
        required=False,
        default=20.0,
    )
    args = parser.parse_args()

    wandb_project = "score-matching " if args.score_matching else "sde-flow"

    args.seed = np.random.randint(2**31 - 1) if args.seed is None else args.seed
    set_seed(args.seed + 1)
    device = configure_gpu(args.use_gpu, args.which_gpu)
    # device = configure_gpu(args.use_gpu)

    expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"

    
    if args.mode == 'train':
        if not os.path.exists(expt_save_path):
            os.makedirs(expt_save_path)
        wandb_logger = pl.loggers.wandb.WandbLogger(
            project=wandb_project,
            name=f"{args.name}_task={args.task}_{args.seed}",
            save_dir=expt_save_path)
        log_args(args, wandb_logger)
        run_classifier(
        # run_training_forward(
            taskname=args.task,
            seed=args.seed,
            wandb_logger=wandb_logger,
            args=args,
            device=device,
        )
