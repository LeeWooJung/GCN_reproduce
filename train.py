import time
import argparse

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim

from preprocessing import load_data
from utils import accuracy, stop_criterion
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
parser.add_argument('--window', type=int, default=10, help='# of window size to stop trianing')

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
window_size = args.window

np.random.seed(seed)
torch.manual_seed(seed)
if device == "cuda":
	torch.cuda.manual_seed(seed)

# load dataset
A, X, y, train_idx, val_idx, test_idx = load_data(dname,dtype)

dfeat = X.shape[1]
nclass = y.shape[1]
val_window = []

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

print("Start training the model.............")
start = time.time()
for epoch in range(epochs):
	model.train()
	optimizer.zero_grad()
	output = model(X,A)
	train_loss = F.nll_loss(output[train_idx], torch.max(y[train_idx],1)[1])
	train_accuracy = accuracy(output[train_idx], torch.max(y[train_idx],1)[1])
	train_loss.backward()
	optimizer.step()

	model.eval()
	output = model(X,A)
	val_loss = F.nll_loss(output[val_idx], torch.max(y[val_idx],1)[1])
	val_accuracy = accuracy(output[val_idx], torch.max(y[val_idx],1)[1])

	print('Epoch: {:04d}'.format(epoch+1),
	      'train loss: {:.4f}'.format(train_loss),
	      'train accuracy: {:.4f}%'.format(train_accuracy),
	      'validation loss: {:.4f}'.format(val_loss),
	      'validation accuracy: {:.4f}%'.format(val_accuracy))

	if epoch == epochs-1: continue
	val_window.append(val_loss)
	if len(val_window) >= window_size:
		if stop_criterion(val_window, window_size):
			print("*"*30, end = " ")
			print("Validation accuracy doesn't decrease anymore with window size : {:02d}".format(window_size))
			break
		del val_window[0]
end = time.time()
print("Total time to train model: {:.4f}s".format(end-start))
print("Optimization step is finished!")

#test
test_start = time.time()
model.eval()
output = model(X,A)
test_loss = F.nll_loss(output[test_idx], torch.max(y[test_idx],1)[1])
test_accuracy = accuracy(output[test_idx], torch.max(y[test_idx],1)[1])
test_end = time.time()
print("Test result: ",
      "Total time: {:.4f}s".format(test_end-test_start),
      "test loss: {:.4f}".format(test_loss),
      "test accuracy: {:.4f}".format(test_accuracy))
