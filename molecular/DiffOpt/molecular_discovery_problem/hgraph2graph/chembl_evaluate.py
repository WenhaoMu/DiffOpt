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

from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from tdc import Oracle

random.seed(7)

data = numpy.load('./chembl_encoded_valid.npy')
print(data.shape)
print(data[0])

SA_oracle = Oracle(name = 'SA')
DRD2_oracle = Oracle(name = 'DRD2')
JNK3_oracle = Oracle(name = 'JNK3')
GSK3B_oracle = Oracle(name = 'GSK3B')

value = numpy.zeros((10000,6))
count = 0
data = [x.strip("\r\n ").split() for x in open('./chembl_sample_valid.txt')]
for i in data:
    # print(i)
    mol = Chem.MolFromSmiles(i[0])
    logP = Descriptors.MolLogP(mol)
    qed = QED.qed(mol)
    sa_score = SA_oracle(i[0])
    drd2_score = DRD2_oracle(i[0])
    jnk3_score = JNK3_oracle(i[0])
    gsk3b_score = GSK3B_oracle(i[0])

    value[count,0] = logP
    value[count,1] = qed
    value[count,2] = sa_score
    value[count,3] = drd2_score
    value[count,4] = jnk3_score
    value[count,5] = gsk3b_score

    count += 1
    print(f'-----finish {count} times-----')

numpy.save('./chembl_value_valid.npy', value)
