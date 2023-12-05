from problem import *
import torch
import os
from utils import z_to_smiles
from rdkit import Chem

problem = get_problem('molecular_discovery')
# problem.targets : stores embeddings for each protein target, of which there are 102
# problem.objectives : names of each of the 10 objectives

torch.manual_seed(42)

compounds_data = torch.zeros(10000, 32).cuda()
outputs_data = torch.zeros(10000, 3).cuda()

N = 1      # batch size
error_count = 0

for i in range(1000):
    a = torch.randn(N, 32).cuda()       # compounds randomly sampled from latent space
    c = problem.targets[torch.randperm(len(problem.targets))[:N]]       # randomly selected protein targets
    # c = problem.targets[torch.randperm(len(problem.targets))[:1]]  
    # print('C: {}'.format(c.shape))


    x = torch.cat((a, c), dim=1).requires_grad_(True)        # input to the surrogate model

    # print(problem.evaluate(x, 0).shape)     # output shape: (N, 10), each objective is in the range [0, 1]
    # print(problem.evaluate(x, 0))
    output = problem.evaluate(x, 0)
    # mol = Chem.MolFromSmiles('Cc1ccccc1')

    try:
        mol = z_to_smiles(a)
        # print(mol[0])
        try:
            mol = Chem.MolFromSmiles(mol[0])
        except Exception as a:
            print(f"在第 {i+1} 次执行时发生错误: {a}")
    except Exception as e:
        # 如果发生错误，增加错误计数
        error_count += 1
        # 你也可以打印错误消息或进行其他的错误记录
        print(f"在第 {i+1} 次执行时发生错误: {e}")
    # print(mol)
    # if mol is not None:
    #     # try:
    #         # 尝试对分子进行“清洗”，确保其化学结构的合理性
    #         # Chem.rdmolops.SanitizeMol(mol)
    #     print("The molecule is valid.")
    #     # except Exception as e:
    #         # print("The molecule is invalid:", e)
    # else:
    #     print("The SMILES string is invalid, and no molecule was created.")
    #     smile = z_to_smiles(a)

    compounds_data[i] = a.squeeze()
    outputs_data[i, :3] = output.squeeze()[:3]
print(f"在 1000 次执行中，共有 {error_count} 次报错。")

os.makedirs('./data', exist_ok=True)
# torch.save(compounds_data, './data/compounds_data.pt')
# torch.save(outputs_data, './data/compounds_value.pt')