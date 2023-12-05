import torch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood

class Layer(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class DeepGaussianProcess(GPyTorchModel):
    def __init__(self, train_x_shape, output_dim=1):
        super().__init__()

        inducing_points1 = torch.randn(train_x_shape[0], train_x_shape[1])  # for the first layer
        inducing_points2 = torch.randn(train_x_shape[0], output_dim)  # for the second layer
        self.layer1 = Layer(inducing_points1)
        self.layer2 = Layer(inducing_points2)
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        output1_dist = self.layer1(x)  
        output1 = output1_dist.mean  
        output2_dist = self.layer2(output1)
        output2 = output2_dist.mean  
        return output2_dist 

