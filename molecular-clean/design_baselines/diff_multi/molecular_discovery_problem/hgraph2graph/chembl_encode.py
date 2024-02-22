# Imports / Arg Parser / Functions
# Copied from generate.py preprocess.py and / or hgraph/hgnn.py

from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy
import random
from hgraph import *
import rdkit


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x.to('cuda') if type(x) is torch.Tensor else torch.tensor(x).to('cuda')
    tree_tensors = [make_tensor(x).long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', default='data/chembl/vocab.txt')
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', default='ckpt/chembl-pretrained/model.ckpt')

parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--nsample', type=int, default=10000)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()

# Parse Vocabuary File
vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
args.vocab = PairVocab(vocab)

# Test Compound to reconstruct
# smiles = ['C=Cc1cccc(C(=O)N2CC(c3ccc(F)cc3)C(C)(C)C2)c1']
smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]

print("\nINPUT SMILES: {0}\n".format(" ".join(smiles)))

# Convert SMILES String into MolGraph Tree / Graph Tensors
# (See preprocess.py)

print("SMILES ",type(smiles))
o = tensorize(smiles, args.vocab)
batches, tensors, all_orders = o

# Extract pieces we need
tree_tensors, graph_tensors = make_cuda(tensors)


# Load Checkpoint model
model = HierVAE(args)
model = model.cuda()

model.load_state_dict(torch.load(args.model, map_location=torch.device('cuda'))[0])
model.eval()


# Encode compound
root_vecs, tree_vecs, _, graph_vecs = model.encoder(tree_tensors, graph_tensors)

print("\nLATENT_EMBEDDING\n")
print(root_vecs)
print("\n")

# Unsure what this second step does / what the difference between
# the first and second root_vecs values are?
root_vecs, root_kl = model.rsample(root_vecs, model.R_mean, model.R_var, perturb=False)

print("\nLATENT_EMBEDDING_2\n")
print(root_vecs)
print("\n")


# Decode compound
decoded_smiles = model.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)


def z_to_smiles(z):
    return model.decoder.decode((z, z, z), greedy=True, max_decode_step=150)

print('------------------- start encoding chembl data -------------------')
random.seed(7)
chembl_encoded = numpy.zeros((10000,32))
count = 0
selected_items = []
# data = [x.strip("\r\n ").split() for x in open('./chembl_sample.txt')]
data = vocab = [x.strip("\r\n ").split() for x in open('data/chembl/all.txt')]

while count < 10000:
    try:
        i = random.choice(data)
        if i in selected_items:
            continue  

        o = tensorize(i, args.vocab)
        batches, tensors, all_orders = o

        tree_tensors, graph_tensors = make_cuda(tensors)
        root_vecs, tree_vecs, _, graph_vecs = model.encoder(tree_tensors, graph_tensors)
        root_vecs, root_kl = model.rsample(root_vecs, model.R_mean, model.R_var, perturb=False)
        try:
            mol_smile = z_to_smiles(root_vecs)

            chembl_encoded[count] = root_vecs.detach().cpu().numpy()

            count += 1
            selected_items.append(i)
            print(f'-----finish {count} times-----')
        except Exception as err:
            print(f"Error during decoding: {err}")

    except Exception as e:
        print(f"Error during encoding: {e}")

numpy.save('./chembl_encoded_valid.npy', chembl_encoded)

with open('chembl_sample_valid.txt', 'w') as f:
    for item in selected_items:
        f.write(' '.join(item) + '\n')
