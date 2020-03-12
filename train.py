import argparse
import time

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim

from preprocessing import load_data, normalize, sparse_to_torchTensor
from model import GCN


parser = argparse.ArgumentParser()
parser.add_argument('--dname', type=str, default='cora', help='Name of dataset')
parser.add_argument('--dtype', type=str, default='citation', help='Type of dataset')
parser.add_argument('--seed', type=int, default=42, help='torch random seed')
parser.add_argument('--epochs', type=int, default=200, help='# of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability)')
parser.add_argument('--nhid', type=int, default=16, help='# of initial hidden units')

args = parser.parse_args()
device = ("cuda" if torch.cuda.is_available() else "cpu")

# DataSet
dname = args.dname
dtype = args.dtype
# Training hyperparameters
seed = args.seed
epochs = args.epochs
lr = args.lr
wd = args.wd
dropout = args.dropout
nhid = args.nhid

np.random.seed(seed)
torch.manual_seed(seed)
if device == "cuda":
	torch.cuda.manual_seed(seed)

# load dataset
A, X, y, train_idx, val_idx, test_idx = load_data(dname,dtype)

dfeat = X.shape[1]
nclass = y.shape[1] 

# define model & optimizer
model = GCN(dfeat, nhid, nclass, dropout)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = wd)

# Data to cuda?
model = model.to(device)
X = X.to(device)
A = A.to(device)
y = y.to(device)
train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

def train(epoch):
	t = time.time()
	model.train()
	optimizer.zero_grad()
	output = model(X,A)
	train_loss = F.nll_loss(output[train_idx], torch.max(y[train_idx],1)[1])
	train_loss.backward()
	optimizer.step()

	val_loss = F.nll_loss(output[val_idx], torch.max(y[val_idx],1)[1])
	# accruacy code
	print('Epoch: {:04d}, train_loss: {:.4f}, validation_loss: {:.4f}'
			.format(epoch+1,train_loss, val_loss))

for epoch in range(epochs):
	train(epoch)
