import json
import os
import random
import string
import uuid
import shutil
import pickle

from typing import Optional, Union
from pprint import pprint

import configargparse

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import csv

from models.DKL_model_regression import GPRegressionModel
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
import torch
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import gpytorch
import math
import tqdm
from lib.sdes import VariancePreservingSDE

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

from models.nets import DiffusionTest, DiffusionScore
from models.surrogate_mlp import Classifier as MLP_Classifier
from models.surrogate_fourier import Classifier as Fourier_Classifier
from models.util import TASKNAME2TASK, configure_gpu, set_seed, get_weights
from molecular_discovery_problem.problem import *

args_filename = "args.json"
checkpoint_dir = "checkpoints"
wandb_project = "sde-flow"


class Compound:

    def __init__(self):
        with open("./DiffOpt/data/chembl_encoded_valid.p", "rb") as f:
            data = pickle.load(f)
        with open("./DiffOpt/data/chembl_value_valid.p", "rb") as f:
            value = pickle.load(f)
        self.x = data
        self.y = value[:, [1, 2, 5]]

        self.mean_x = self.x.mean(axis=0)
        self.std_x = self.x.std(axis=0)

        self.mean_y = self.y.mean(axis=0)
        self.std_y = self.y.std(axis=0)

        self.is_x_normalized = False
        self.is_y_normalized = False

        self.is_discrete = False
        print("X shape: ", self.x.shape)
        print("Y shape: ", self.y.shape)


    def map_normalize_x(self):
        self.x = (self.x - self.mean_x) / self.std_x
        self.is_x_normalized = True

    def map_normalize_y(self):
        self.y = (self.y - self.mean_y) / self.std_y
        self.is_y_normalized = True

    def denormalize_y(self, y):
        return y * self.std_y + self.mean_y
    
    def denormalize_x(self, x):
        return x * self.std_x + self.mean_x


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

        if w is None:
            return x, y
        else:
            return x, y, w



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

    y = task.y
    y = y[shuffle_idx]
    if not task.is_discrete:
        x = x.reshape(-1, task.x.shape[-1])
    y = y.reshape(-1, task.y.shape[-1])
    w = get_weights(y[:,0:1], temp=temp) + get_weights(y[:,1:2], temp=temp) - get_weights(y[:,2:3], temp=temp)

    print(w)
    print("Wshape:", w.shape)

    if val_frac is None:
        val_frac = 0

    val_length = int(length * val_frac)
    train_length = length - val_length

    train_dataset = RvSDataset(
        task,
        x[:train_length],
        y[:train_length],
        w[:train_length],
        device,
        mode='train')
    val_dataset = RvSDataset(
        task,
        x[train_length:],
        y[train_length:],
        w[train_length:],
        device,
        mode='val')

    return train_dataset, val_dataset


class RvSDataModule(pl.LightningDataModule):

    def __init__(self, task, batch_size, num_workers, val_frac, device, temp):
        super().__init__()

        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.device = device
        self.train_dataset = None
        self.val_dataset = None
        self.temp = temp

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = split_dataset(
            self.task, self.val_frac, self.device, self.temp)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  num_workers=self.num_workers,
                                  batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size)
        return val_loader


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


def run_training(
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
    if taskname == 'compound':
        task = Compound() 
    elif taskname != 'tf-bind-10':
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


    if not score_matching:
        model = DiffusionTest(taskname=taskname,
                              task=task,
                              learning_rate=learning_rate,
                              hidden_size=hidden_size,
                              vtype=vtype,
                              beta_min=args.beta_min,
                              beta_max=args.beta_max,
                              simple_clip=args.simple_clip,
                              T0=T0,
                              debias=debias,
                              dropout_p=dropout_p)
    else:
        print("Score matching loss")
        model = DiffusionScore(taskname=taskname,
                               task=task,
                               learning_rate=learning_rate,
                               hidden_size=hidden_size,
                               vtype=vtype,
                               beta_min=args.beta_min,
                               beta_max=args.beta_max,
                               simple_clip=args.simple_clip,
                               T0=T0,
                               debias=debias,
                               dropout_p=dropout_p)

    # monitor = "val_loss" if val_frac > 0 else "train_loss"
    monitor = "elbo_estimator" if val_frac > 0 else "train_loss"
    checkpoint_dirpath = os.path.join(wandb_logger.experiment.dir,
                                      checkpoint_dir)
    checkpoint_filename = f"{taskname}_{seed}-" + "-{epoch:03d}-{" + f"{monitor}" + ":.4e}"
    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        monitor=monitor,
        filename=checkpoint_filename,
        save_last=True,  # save latest model
        save_top_k=1,  # save top model based on monitored loss
    )
    trainer = pl.Trainer(
        gpus=int(use_gpu),
        auto_lr_find=auto_tune_lr,
        max_epochs=epochs,
        max_steps=max_steps,
        max_time=train_time,
        logger=wandb_logger,
        progress_bar_refresh_rate=20,
        callbacks=[val_checkpoint_callback],
        track_grad_norm=2,  # logs the 2-norm of gradients
        limit_val_batches=1.0 if val_frac > 0 else 0,
        limit_test_batches=0,
    )

    data_module = RvSDataModule(task=task,
                                val_frac=val_frac,
                                device=device,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                temp=args.temp)
    trainer.fit(model, data_module)


@torch.no_grad()
def run_evaluate(
    taskname,
    seed,
    hidden_size,
    learning_rate,
    checkpoint_path,
    args,
    wandb_logger=None,
    device=None,
    normalise_x=False,
    normalise_y=False,
):
    set_seed(seed)
    task = Compound()
    problem = get_problem('molecular_discovery')
    
    if task.is_discrete:
        task.map_to_logits()
    if normalise_x:
        task.map_normalize_x()
    if normalise_y:
        task.map_normalize_y()


    if not args.score_matching:
        model = DiffusionTest.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            taskname=taskname,
            task=task,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            vtype=args.vtype,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            T0=args.T0,
            dropout_p=args.dropout_p)
    else:
        print("Score matching loss")
        model = DiffusionScore.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            taskname=taskname,
            task=task,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            vtype=args.vtype,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            T0=args.T0,
            dropout_p=args.dropout_p)

    model = model.to(device)
    model.eval()

    train_X = task.x
    train_Y = task.y
    train_X = torch.from_numpy(task.x)
    train_Y = torch.from_numpy(task.y)

    if not task.is_discrete:
        dim_x = task.x.shape[-1]
    else:
        dim_x = task.x.shape[-1] * task.x.shape[-2]


    if args.surrogate == 'dkl':
        def _get_classifier():
            classifier1 = GPRegressionModel(dim_x+1)
            classifier1.load_state_dict(torch.load('/localscratch/wmu30/ddom-multi-gui/DKL_model/best_model_{}_x0_1_chembl_nor_valid.pth'.format(taskname)))
            classifier2 = GPRegressionModel(dim_x+1)
            classifier2.load_state_dict(torch.load('/localscratch/wmu30/ddom-multi-gui/DKL_model/best_model_{}_x0_2_chembl_nor_valid.pth'.format(taskname)))
            classifier3 = GPRegressionModel(dim_x+1)
            classifier3.load_state_dict(torch.load('/localscratch/wmu30/ddom-multi-gui/DKL_model/best_model_{}_x0_3_chembl_nor_valid.pth'.format(taskname)))
            return classifier1, classifier2, classifier3
        
        def cond_fn(x, t, y=None):
            assert y is not None

            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                preds1 = classifier1(torch.cat([x_in,t],dim=1))
                predictive_means1, predictive_variances1 = preds1.mean, preds1.variance  
                preds2 = classifier2(torch.cat([x_in,t],dim=1))
                predictive_means2, predictive_variances2 = preds2.mean, preds2.variance  
                preds3 = classifier3(torch.cat([x_in,t],dim=1))
                predictive_means3, predictive_variances3 = preds3.mean, preds3.variance    
                mean_grad1, = torch.autograd.grad(predictive_means1.mean(), x_in, retain_graph=True)  
                variance_grad1, = torch.autograd.grad(predictive_variances1.mean(), x_in)
                mean_grad2, = torch.autograd.grad(predictive_means2.mean(), x_in, retain_graph=True)  
                variance_grad2, = torch.autograd.grad(predictive_variances2.mean(), x_in)
                mean_grad3, = torch.autograd.grad(predictive_means3.mean(), x_in, retain_graph=True)  
                variance_grad3, = torch.autograd.grad(predictive_variances3.mean(), x_in)
                return mean_grad1, mean_grad2, mean_grad3
            
    elif args.surrogate == 'mlp':
        def _get_classifier():
            checkpoint_path1 = f"experiments/{taskname}/MLP_1/1469983670_128/wandb/latest-run/files/checkpoints/last.ckpt"
            classifier1 = MLP_Classifier.load_from_checkpoint(checkpoint_path=checkpoint_path1,taskname=taskname,task=task,)
            checkpoint_path2 = f"experiments/{taskname}/MLP_2/1469983670_128/wandb/latest-run/files/checkpoints/last.ckpt"
            classifier2 = MLP_Classifier.load_from_checkpoint(checkpoint_path=checkpoint_path2,taskname=taskname,task=task,)
            checkpoint_path3 = f"experiments/{taskname}/MLP_3/1469983670_128/wandb/latest-run/files/checkpoints/last.ckpt"
            classifier3 = MLP_Classifier.load_from_checkpoint(checkpoint_path=checkpoint_path3,taskname=taskname,task=task,)
            return classifier1, classifier2, classifier3

        def cond_fn(x, t, y=None):
            assert y is not None
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                y_pred1 = classifier1.mlp(x_in, t) 
                grad_y_pred_1 = torch.autograd.grad(y_pred1.sum(), x_in)[0]
                y_pred2 = classifier2.mlp(x_in, t)  
                grad_y_pred_2 = torch.autograd.grad(y_pred2.sum(), x_in)[0]
                y_pred3 = classifier3.mlp(x_in, t)  
                grad_y_pred_3 = torch.autograd.grad(y_pred3.sum(), x_in)[0]
                return grad_y_pred_1, grad_y_pred_2, grad_y_pred_3
            
    elif args.surrogate == 'fourier':
        def _get_classifier():
            checkpoint_path1 = f"experiments/{taskname}/fourier_1/1469983670_128/wandb/latest-run/files/checkpoints/last.ckpt"
            classifier1 = Fourier_Classifier.load_from_checkpoint(checkpoint_path=checkpoint_path1,taskname=taskname,task=task,)
            checkpoint_path2 = f"experiments/{taskname}/fourier_2/1469983670_128/wandb/latest-run/files/checkpoints/last.ckpt"
            classifier2 = Fourier_Classifier.load_from_checkpoint(checkpoint_path=checkpoint_path2,taskname=taskname,task=task,)
            checkpoint_path3 = f"experiments/{taskname}/fourier_3/1469983670_128/wandb/latest-run/files/checkpoints/last.ckpt"
            classifier3 = Fourier_Classifier.load_from_checkpoint(checkpoint_path=checkpoint_path3,taskname=taskname,task=task,)
            return classifier1, classifier2, classifier3

    
        def cond_fn(x, t, y=None):
            assert y is not None
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                y_pred1 = classifier1.mlp(t, x_in)  
                grad_y_pred_1 = torch.autograd.grad(y_pred1.sum(), x_in)[0]
                y_pred2 = classifier2.mlp(t, x_in) 
                grad_y_pred_2 = torch.autograd.grad(y_pred2.sum(), x_in)[0]
                y_pred3 = classifier3.mlp(t, x_in) 
                grad_y_pred_3 = torch.autograd.grad(y_pred3.sum(), x_in)[0]
                return grad_y_pred_1, grad_y_pred_2, grad_y_pred_3


    def heun_sampler(sde, x_0, ya, num_steps, lmbd=0., keep_all_samples=True):
        device = sde.gen_sde.T.device
        batch_size = x_0.size(0)
        ndim = x_0.dim() - 1
        T_ = sde.gen_sde.T.cpu().item()
        delta = T_ / num_steps
        ts = torch.linspace(0, 1, num_steps + 1) * T_

        xs = []
        x_t = x_0.detach().clone().to(device)
        t = torch.zeros(batch_size, *([1] * ndim), device=device)
        t_n = torch.zeros(batch_size, *([1] * ndim), device=device)
        with torch.no_grad():
            for i in range(num_steps):
                t.fill_(ts[i].item())
                if i < num_steps - 1:
                    t_n.fill_(ts[i + 1].item())
                mu = sde.gen_sde.mu(t, x_t, lmbd=lmbd, gamma=args.gamma)
                sigma = sde.gen_sde.sigma(t, x_t, lmbd=lmbd)

                # Add gradient
                gradient1, gradient2, gradient3 = cond_fn(x_t, t, ya)
                mu = (

                    mu.float() + args.coefficient *gradient1.float() - args.coefficient *gradient2.float() + args.coefficient *gradient3.float()
                )

                x_t = x_t + delta * mu + delta**0.5 * sigma * torch.randn_like(
                    x_t
                )  # one step update of Euler Maruyama method with a step size delta
                # Additional terms for Heun's method
                if i < num_steps - 1:
                    mu2 = sde.gen_sde.mu(t_n,
                                         x_t,
                                        #  ya,
                                         lmbd=lmbd,
                                         gamma=args.gamma)
                    sigma2 = sde.gen_sde.sigma(t_n, x_t, lmbd=lmbd)
                    x_t = x_t + (sigma2 -
                                 sigma) / 2 * delta**0.5 * torch.randn_like(x_t)

                if keep_all_samples or i == num_steps - 1:
                    xs.append(x_t.cpu())
                else:
                    pass
        return xs

    def euler_maruyama_sampler(sde,
                               x_0,
                               ya,
                               num_steps,
                               lmbd=0.,
                               keep_all_samples=True):
        """
        Euler Maruyama method with a step size delta
        """
        # init
        device = sde.gen_sde.T.device
        batch_size = x_0.size(0)
        ndim = x_0.dim() - 1
        T_ = sde.gen_sde.T.cpu().item()
        delta = T_ / num_steps
        ts = torch.linspace(0, 1, num_steps + 1) * T_

        # sample
        xs = []
        x_t = x_0.detach().clone().to(device)
        t = torch.zeros(batch_size, *([1] * ndim), device=device)
        with torch.no_grad():
            for i in range(num_steps):
                t.fill_(ts[i].item())
                mu = sde.gen_sde.mu(t, x_t, lmbd=lmbd, gamma=args.gamma)
                sigma = sde.gen_sde.sigma(t, x_t, lmbd=lmbd)

                # Add gradient
                gradient = cond_fn(x_t, t, ya)
                mu = (
                    mu.float() + 0*gradient.float()
                )

                x_t = x_t + delta * mu + delta**0.5 * sigma * torch.randn_like(
                    x_t
                )  # one step update of Euler Maruyama method with a step size delta
                if keep_all_samples or i == num_steps - 1:
                    xs.append(x_t.cpu())
                else:
                    pass
        return xs

    num_steps = args.num_steps
    num_samples = 1000
    lmbds = [args.lamda]

    # use the max of the dataset instead
    args.condition = task.y.max()
    # save to file
    expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}_noreweight"
    # expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"
    assert os.path.exists(expt_save_path)

    alias = uuid.uuid4()
    run_specific_str = f"{num_samples}_{num_steps}_{args.condition}_{args.gamma}_{args.beta_min}_{args.beta_max}_{args.suffix}_{alias}"
    save_results_dir = os.path.join(
        expt_save_path, f"wandb/latest-run/files/results/{run_specific_str}/")
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    assert os.path.exists(save_results_dir)

    symlink_dir = os.path.join(expt_save_path,
                               f"wandb/latest-run/files/results/latest-run")

    if os.path.exists(symlink_dir):
        os.unlink(symlink_dir)
    os.symlink(run_specific_str, symlink_dir)

    # sample and plot
    designs = []
    for lmbd in lmbds:
        if not task.is_discrete:
            x_0 = torch.randn(num_samples, task.x.shape[-1],
                              device=device)  # init from prior
        else:
            x_0 = torch.randn(num_samples,
                              task.x.shape[-1] * task.x.shape[-2],
                              device=device)  # init from prior

        y_ = torch.ones(num_samples).to(device) * args.condition
        classifier1, classifier2, classifier3 = _get_classifier()
        classifier1.to(x_0.device)
        classifier2.to(x_0.device)
        classifier3.to(x_0.device)
        xs = heun_sampler(model,
                          x_0,
                          y_,
                          num_steps,
                          lmbd=lmbd,
                          keep_all_samples=False)  # sample

        ctr = 0
        for sol in xs:
            ctr += 1
            print(sol.shape)
            if not sol.isnan().any():
                designs.append(sol.cpu().numpy())
                if args.surrogate == 'dkl':
                    os.makedirs('./DiffOpt/molecular_discovery_problem/Result_DKL', exist_ok=True)
                    torch.save(sol, f'./DiffOpt/molecular_discovery_problem/Result_DKL/molecular_{args.coefficient}_{num_samples}.pt')
                elif args.surrogate == 'mlp':
                    os.makedirs('./DiffOpt/molecular_discovery_problem/Result_MLP', exist_ok=True)
                    torch.save(sol, f'./DiffOpt/molecular_discovery_problem/Result_MLP/molecular_{args.coefficient}_{num_samples}.pt')
                if args.surrogate == 'fourier':
                    os.makedirs('./DiffOpt/molecular_discovery_problem/Result_Fourier', exist_ok=True)
                    torch.save(sol, f'./DiffOpt/molecular_discovery_problem/Result_Fourier/molecular_{args.coefficient}_{num_samples}.pt')
            else:
                print("No solution!")

    designs = np.concatenate(designs, axis=0)
    print(designs.shape)
   



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
    parser.add_argument('--surrogate',
                        choices=['dkl', 'mlp', 'fourier'],
                        default='dkl',
                        required=True)
    parser.add_argument('--task',
                        choices=list(TASKNAME2TASK.keys())+ ["branin"]+ ["compound"],
                        required=True)
    parser.add_argument("--coefficient",
                        type=float,
                        default=0,
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
        default=False,
        help="place networks and data on the GPU",
    )
    parser.add_argument('--simple_clip', action="store_true", default=False)
    parser.add_argument("--which_gpu",
                        default=0,
                        type=int,
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
    parser.add_argument('--batch_size', type=int, default=64)
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
    
    os.makedirs(f"./experiments/{args.task}/{args.name}/{args.seed}_noreweight", exist_ok=True)
    expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}_noreweight"
    # expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"

    if args.mode == 'train':
        if not os.path.exists(expt_save_path):
            os.makedirs(expt_save_path)
        wandb_logger = pl.loggers.wandb.WandbLogger(
            project=wandb_project,
            name=f"{args.name}_task={args.task}_{args.seed}",
            save_dir=expt_save_path)
        log_args(args, wandb_logger)
        run_training(
            taskname=args.task,
            seed=args.seed,
            wandb_logger=wandb_logger,
            args=args,
            device=device,
        )
    elif args.mode == 'eval':
        checkpoint_path = os.path.join(
            expt_save_path, "wandb/latest-run/files/checkpoints/last.ckpt")
        run_evaluate(taskname=args.task,
                     seed=args.seed,
                     hidden_size=args.hidden_size,
                     args=args,
                     learning_rate=args.learning_rate,
                     checkpoint_path=checkpoint_path,
                     device=device,
                     normalise_x=args.normalise_x,
                     normalise_y=args.normalise_y)
    else:
        raise NotImplementedError
