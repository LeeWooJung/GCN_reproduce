import argparse

import numpy as np
import scipy.sparse as sp

from preprocessing import load_data, normalize, sparse_to_torchTensor
from model import GCN


parser = argparse.ArgumentParser()
parser.add_argument('--dname', type=str, default='cora', help='Name of dataset')
parser.add_argument('--dtype', type=str, default='citation', help='Type of dataset')
parser.add_argument('--iscuda', type=bool, default=True, help='Use cuda or not')
parser.add_argument('--seed', type=int, default=42, help='torch random seed')
parser.add_argument('--epochs', type=int, default=200, help='# of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability)')
parser.add_argument('--nhid', type=int, default=16, help='# of initial hidden units')

args = parser.parse_args()

# DataSet
dname = args.dname
dtype = args.dtype
# Training hyperparameters
iscuda = args.iscuda
seed = args.seed
epochs = args.epochs
lr = args.lr
wd = args.wd
dropout = args.dropout
nhid = args.nhid

# load dataset
A, X, y = load_data(dname,dtype)

nfeat = X.shape[1]
nclass = max(y).item()

# define model
model = GCN(nfeat, nhid, nclass, dropout)
output = model(X, A)

print(output)
