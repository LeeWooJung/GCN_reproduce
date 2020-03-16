import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class ConvLayer(nn.Module):
	def __init__(self, inDim, outDim):
		super(ConvLayer, self).__init__()

		self.inDim = inDim
		self.outDim = outDim
		self.W = nn.Parameter(torch.randn(inDim,outDim))
		self.b = nn.Parameter(torch.zeros(outDim))
		self.setParam()

	def setParam(self):
		
		In, Out = tuple(self.W.size())
		absRange = np.sqrt(6.0/(In+Out))
		self.W.data.uniform_(-absRange,absRange)
		#absRange = np.sqrt(3.0/Out)
		#self.b.data.uniform_(absRange,absRange)

	def forward(self, x, adj):

		x = torch.sparse.mm(adj, x)
		x = torch.matmul(x, self.W) +  self.b

		return x

class GCN(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(GCN, self).__init__()

		self.gc1 = ConvLayer(nfeat, nhid)
		self.gc2 = ConvLayer(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, adj):

		x = F.relu(self.gc1(x, adj))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.gc2(x, adj)
		output = F.log_softmax(x, dim=1)

		return output
