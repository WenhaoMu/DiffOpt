import sys
import os
import math

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Optional, Tuple, Type


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


import numpy as np
import pytorch_lightning as pl

import torch
from torch import optim, nn, utils, Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

from models.util import TASKNAME2TASK

from lib.sdes import VariancePreservingSDE, PluginReverseSDE, ScorePluginReverseSDE
from models.unet import UNET_1D


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x

    

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Network(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=1024, fourier_dim_z=86, fourier_dim_t=1):
        super().__init__()
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.time_fourier = GaussianFourierProjection(embedding_size=fourier_dim_t, scale=1)
        self.state_fourier = GaussianFourierProjection(embedding_size=fourier_dim_t, scale=1)

        self.time_embed = nn.Sequential(
            nn.Linear(2 * fourier_dim_t, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.state_embed = nn.Sequential(
            nn.Linear(in_dim * 2 * fourier_dim_t, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def get_fourier(self, z):
        emb = []
        for i in range(z.shape[-1]):
            emb.append(self.state_fourier(z[..., i:i+1]))
        return torch.cat(emb, dim=-1)

    def forward(self, t, z):
        batch_size = z.shape[0]
        t = t.unsqueeze(dim=-1)
        # embed time and state
        t_fourier = self.time_fourier(t)
        t_emb = self.time_embed(t_fourier)
        state_fourier = self.get_fourier(z)
        state_emb = self.state_embed(state_fourier)

        x = t_emb + state_emb
        x = x + self.mlp1(x)
        x = self.mlp2(x)

        return x





class Classifier(pl.LightningModule):

    def __init__(
            self,
            taskname,
            task,
            hidden_size=1024,
            learning_rate=1e-3,
            beta_min=0.1,
            beta_max=20.0,
            dropout_p=0,
            simple_clip=False,
            activation_fn=Swish(),
            T0=1,
            debias=False,
            vtype='rademacher'):
        super().__init__()
        self.taskname = taskname
        self.task = task
        self.learning_rate = learning_rate
        self.dim_y = self.task.y.shape[-1]
        if not task.is_discrete:
            self.dim_x = self.task.x.shape[-1]
        else:
            self.dim_x = self.task.x.shape[-1] * self.task.x.shape[-2]
        self.dropout_p = dropout_p
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.simple_clip = simple_clip
        self.debias = debias
        self.T0 = T0
        self.vtype = vtype

        self.learning_rate = learning_rate


        self.mlp = Network(in_dim=self.dim_x,
                           out_dim=1,
                           fourier_dim_z=self.dim_x,
                           fourier_dim_t=1,
                           hidden_dim=hidden_size)
                        #    act=activation_fn)

        self.T = torch.nn.Parameter(torch.FloatTensor([self.T0]),
                                    requires_grad=False)

        self.inf_sde = VariancePreservingSDE(beta_min=self.beta_min,
                                             beta_max=self.beta_max,
                                             T=self.T)


    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = torch.optim.Adam(self.mlp.parameters(),
                                     lr=self.learning_rate)
        return optimizer

   

    def training_step(self, batch, batch_idx, log_prefix="train"):
        x, y, w = batch

        if self.debias:
            t_ = self.inf_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.inf_sde.sample(t_, x, return_noise=True)

        pred = self.mlp(t_.squeeze(), x_hat)
        loss = torch.nn.functional.mse_loss(pred, y)
        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss

