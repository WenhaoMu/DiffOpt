from problem import *
import torch


problem = get_problem('molecular_discovery')
# problem.targets : stores embeddings for each protein target, of which there are 102
# problem.objectives : names of each of the 10 objectives

torch.manual_seed(42)

N = 1      # batch size
a = torch.randn(N, 32).cuda()       # compounds randomly sampled from latent space
c = problem.targets[torch.randperm(len(problem.targets))[:N]]       # randomly selected protein targets
# c = problem.targets[torch.randperm(len(problem.targets))[:1]]  
print('C: {}'.format(c.shape))


x = torch.cat((a, c), dim=1).requires_grad_(True)        # input to the surrogate model

print(problem.evaluate(x, 0).shape)     # output shape: (N, 10), each objective is in the range [0, 1]
print(problem.evaluate(x, 0))
print(a)





# print(problem.evaluate(x, 0))
print('-----------------------------')
y_pred = problem.evaluate(x, 0)
y_to_consider = y_pred[0,0]
print(y_to_consider)
grads = torch.autograd.grad(y_to_consider, x, retain_graph=True)[0]
print("GRAD: ",grads.shape)

grad_input2 = grads[:,64:96]

print("INPUT2: ",grad_input2.shape)