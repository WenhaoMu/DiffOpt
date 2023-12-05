import torch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood
from DGP_model_new import DeepGp
import os
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
from gpytorch.mlls import VariationalELBO, AddedLossTerm

# data
num_points = 100
x1 = torch.linspace(0, 1, num_points)
x2 = torch.linspace(0, 1, num_points)
train_X = torch.stack([x1, x2], dim=1)
train_Y = (train_X[:, 0] + train_X[:, 1]).unsqueeze(-1) + torch.randn(num_points, 1) * 0.05
print(train_X.shape, train_Y.shape)


# model = DeepGaussianProcess(train_X, train_Y)
model = DeepGp(train_X.shape)
if torch.cuda.is_available():
    model = model.cuda()
# likelihood = GaussianLikelihood()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_X.shape[-2]))
train_X = train_X.cuda()
train_Y = train_Y.cuda()
# model = model.cuda()
# likelihood = likelihood.cuda()

# print(train_X.device)

model.train()
# likelihood.train()
num_samples = 3 
num_epochs = 100
for epoch in range(num_epochs):
    with gpytorch.settings.num_likelihood_samples(num_samples):
        optimizer.zero_grad()
        output = model(train_X)
        # loss = -likelihood.log_marginal(train_Y, output).mean()
        # print(type(loss),loss.shape)
        loss = -mll(output, train_Y.shape[0])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

torch.save(model.state_dict(), os.path.join("DGP_model", "best_model_toy_new.pth"))
