import math
import tqdm
import torch
import gpytorch
# from matplotlib import pyplot as plt

# Make plots inline
# %matplotlib inline

import urllib.request
import os
from scipy.io import loadmat
from math import floor

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)


# if not smoke_test and not os.path.isfile('../elevators.mat'):
#     print('Downloading \'elevators\' UCI dataset...')
#     urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')


if smoke_test:  # this is for running the notebook in our testing framework
    X, y = torch.randn(2000, 3), torch.randn(2000)
else:
    data = torch.Tensor(loadmat('elevators.mat')['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]


train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

print("TRAIN X: ", train_x.shape)
print("TRAIN Y: ", train_y.shape)

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))

# feature_extractor = LargeFeatureExtractor()

# class GPRegressionModel(gpytorch.models.ExactGP):
class GPRegressionModel(gpytorch.models.ApproximateGP):
        # def __init__(self, train_x, train_y, likelihood, data_dim):
        def __init__(self, train_x, train_y, likelihood, data_dim, num_inducing=500):
            inducing_points = torch.randn(num_inducing, data_dim)
            # super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
            super(GPRegressionModel, self).__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.feature_extractor = LargeFeatureExtractor(data_dim)

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood, data_dim)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

epoch = 2 if smoke_test else 60

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
# optimizer = torch.optim.Adam([
#     {'params': model.feature_extractor.parameters()},
#     {'params': model.covar_module.parameters()},
#     {'params': model.mean_module.parameters()},
#     {'params': model.likelihood.parameters()},
# ], lr=0.01)
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

# def train():
#     iterator = tqdm.notebook.tqdm(range(training_iterations))
#     for i in iterator:
#         # Zero backprop gradients
#         optimizer.zero_grad()
#         # Get output from model
#         output = model(train_x)
#         # Calc loss and backprop derivatives
#         loss = -mll(output, train_y)
#         loss.backward()
#         iterator.set_postfix(loss=loss.item())
#         optimizer.step()

def train(epoch):
    model.train()
    likelihood.train()
    train_loss = 0.0
    num_train = 0
    for epoch in range(1, epoch + 1):
        optimizer.zero_grad()
        output = model(train_x)
        # print(train_x.shape, train_y.shape)
        # print("Y: ", train_y.shape)
        print("OUTPUT: ", output.mean)
        loss = -mll(output, train_y)
        loss.backward()
        train_loss+=loss.item()
        optimizer.step()
        num_train += train_x.shape[0]
        train_loss = train_loss / num_train
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")


train(epoch)

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
    preds = model(test_x)

print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))