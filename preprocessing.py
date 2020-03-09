import os
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp

from sklearn.preprocessing import LabelEncoder as LE

def normalize(matrix): # same with https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
	"""
		Row-normalize sparse matrix
	"""
	rowsum = np.array(matrix.sum(1))
	r_inv = np.power(rowsum,-1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	matrix = r_mat_inv.dot(matrix)
	return matrix

def labelEncoding(labels):
	le = LE()
	LE()
	labels = np.array(le.fit_transform(labels)) + 1
	return labels

def sparse_mx_to_torch_sparse_tensor(sparse_mx): # same with https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
	"""
		Convert a scipy sparse matrix to a torch sparse tensor.
	"""
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(
			np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)

def load_data(dname='cora', dtype='citation'):
	if dtype=='citation':
		dpath = os.path.join('data', dname, dname)
		linkPath = '.cites'
		dataPath = '.content'
		# read data
		data = np.genfromtxt(dpath + dataPath, delimiter='\t', dtype = np.dtype(str))
		nodeList = np.array(data[:,0], dtype=np.int32)
		features = sp.csr_matrix(data[:,1:-1], dtype=np.float32)
		labels = labelEncoding(data[:,-1])
		# read edges
		graph = nx.read_edgelist(dpath + linkPath, delimiter='\t', nodetype=int)
		adj = nx.to_scipy_sparse_matrix(graph, format = 'coo', dtype=np.float32, nodelist = nodeList.tolist())
		# row-sum normalize of features & adjacency matrix
		features = normalize(features)
		adj = normalize(adj + sp.eye(adj.shape[0]))
		# change the data to torch tensor
		features = torch.FloatTensor(np.array(features.todense()))
		labels = torch.LongTensor(labels)
		adj = sparse_mx_to_torch_sparse_tensor(adj)
		# index
		if dname == 'cora':
			idx_train = range(140)
			idx_val = range(200,500)
			idx_test = range(500,1500)

		idx_train = torch.LongTensor(idx_train)
		idx_val = torch.LongTensor(idx_val)
		idx_test = torch.LongTensor(idx_test)

		return adj, features, labels, idx_train, idx_val, idx_test
