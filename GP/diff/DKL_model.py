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

from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from densenet_part import DenseNet


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, input_dims, output_dims, num_inducing=128):

        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )
        self.mean_module = gpytorch.means.ConstantMean()
        # self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    

class DKLModel(gpytorch.Module):
    def __init__(self, dim_x, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.grid_bounds = grid_bounds
        self.hidden_dim = 128
        self.gp_layer = GaussianProcessLayer(input_dims=32, output_dims=None)
        self.main = nn.Sequential(
            nn.Linear(dim_x , self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 32),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
        )  

    def forward(self, x):
        features = self.main(x)
        res = self.gp_layer(features)
        return res
