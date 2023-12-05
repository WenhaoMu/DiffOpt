import torch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood
# from DGP_model import DeepGaussianProcess
import os

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
        # print('x: ',x.shape)
        # output1 = self.layer1(x).mean  # we'll use the mean as input to the next layer
        # # print('x1: ',output1.shape)
        # # output1 = self.layer1(x).mean.unsqueeze(-1)
        # print('x1: ',output1.shape)
        # output2 = self.layer2(output1)
        # print('x2: ',output2.shape)
        # return output2
        # print('x: ', x.shape)
        output1_dist = self.layer1(x) 
        output1 = output1_dist.mean   
        # print('x1: ', output1.shape)

        output2_dist = self.layer2(output1)
        output2 = output2_dist.mean  
        # print('x2: ', output2.shape)
        # print("type",type(output1_dist))
        return output2_dist 





# train_X = torch.linspace(0, 1, 100)
# train_Y = torch.sin(train_X * (2 * 3.14)) + torch.randn_like(train_X) * 0.2
# if len(train_X.shape) == 1:
#     train_X = train_X.unsqueeze(-1)

num_points = 100


x1 = torch.linspace(0, 1, num_points)
x2 = torch.linspace(0, 1, num_points)
train_X = torch.stack([x1, x2], dim=1)


train_Y = (train_X[:, 0] + train_X[:, 1]).unsqueeze(-1) + torch.randn(num_points, 1) * 0.05
print(train_X.shape, train_Y.shape)




# model = DeepGaussianProcess(train_X, train_Y)
model = DeepGaussianProcess(train_X.shape, train_Y.shape[-1])
likelihood = GaussianLikelihood()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    output = model(train_X)
    
    loss = -likelihood.log_marginal(train_Y, output).mean()
    # print(type(loss),loss.shape)

    loss.backward()

    optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

torch.save(model.state_dict(), os.path.join("DGP_models", "best_model_toy.pth"))
