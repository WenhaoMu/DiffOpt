from problem import *
import torch
import os


problem = get_problem('molecular_discovery')
# problem.targets : stores embeddings for each protein target, of which there are 102
# problem.objectives : names of each of the 10 objectives

torch.manual_seed(42)

compounds_data = torch.zeros(10000, 32).cuda()
outputs_data = torch.zeros(10000, 3).cuda()

N = 1      # batch size

for i in range(10000):
    a = torch.randn(N, 32).cuda()       # compounds randomly sampled from latent space
    c = problem.targets[torch.randperm(len(problem.targets))[:N]]       # randomly selected protein targets
    # c = problem.targets[torch.randperm(len(problem.targets))[:1]]  
    print('C: {}'.format(c.shape))
    print('A: {}'.format(a.shape))


    x = torch.cat((a, c), dim=1).requires_grad_(True)        # input to the surrogate model

    # print(problem.evaluate(x, 0).shape)     # output shape: (N, 10), each objective is in the range [0, 1]
    # print(problem.evaluate(x, 0))
    output = problem.evaluate(x, 0)

    compounds_data[i] = a.squeeze()
    outputs_data[i, :3] = output.squeeze()[:3]


# os.makedirs('./data', exist_ok=True)
# torch.save(compounds_data, './data/compounds_data.pt')
# torch.save(outputs_data, './data/compounds_value.pt')