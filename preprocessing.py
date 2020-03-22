import os
import torch

import numpy as np
import networkx as nx
import scipy.sparse as sp

def normalize(data):

	rowsum = np.array(data.sum(1))
	with np.errstate(divide='ignore'):
		inv_r = np.power(rowsum,-1).flatten()
	inv_r[np.isinf(inv_r)] = 0.
	data = sp.diags(inv_r).dot(data)

	return data

def sparse_to_torchTensor(matrix):

	matrix = sp.coo_matrix(matrix)
	values = torch.FloatTensor(matrix.data)
	indices = torch.LongTensor(np.vstack((matrix.row, matrix.col)))
	shape = matrix.shape	

	return torch.sparse.FloatTensor(indices, values, torch.Size(shape))

def load_data(dname='cora', dtype='citation'):
	"""
		Reference: https://github.com/tkipf/gae/blob/master/gae/input_data.py
	"""
	print('Loading {} dataset.........'.format(dname))
	if dtype == 'citation':
		candidate = ['x','y','tx','ty','allx','ally','graph']
		obj = []
		dpath = os.path.join('./data',dname)

		for name in candidate:
			with open("{}/ind.{}.{}".format(dpath,dname,name),'rb') as f:
				obj.append(np.load(f, allow_pickle=True,encoding='latin1'))
		x, y, tx, ty, allx, ally, graph = tuple(obj)

		test_index = []
		for line in open("{}/ind.{}.test.index".format(dpath,dname)):
			test_index.append(int(line.strip()))

		# sort the test index
		test_index_sorted = np.sort(test_index)

		if dname == 'citeseer':
			# fill zero vectors to isolated test nodes
			full_test_index = range(test_index_sorted[0], test_index_sorted[-1]+1)
			tx_ = sp.lil_matrix((len(full_test_index), x.shape[1]))
			tx_[test_index_sorted-min(test_index_sorted),:] = tx
			tx = tx_
			ty_ = np.zeros((len(full_test_index), y.shape[1]))
			ty_[test_index_sorted-min(test_index_sorted),:] = ty
			ty = ty_

		features = sp.vstack((allx, tx)).tolil()
		features[test_index,:] = features[test_index_sorted,:]
		features = normalize(features)
		features = torch.FloatTensor(features.todense())

		adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).astype(np.float32)
		adj = adj + sp.eye(adj.shape[0]) # Add self connections
		adj = normalize(adj)
		adj = sparse_to_torchTensor(adj)

		label = np.vstack((ally, ty))
		label[test_index,:] = label[test_index_sorted,:]
		label = torch.LongTensor(label)

		train_idx = range(x.shape[0])
		val_idx = range(x.shape[0], x.shape[0]+500)

		train_idx = torch.LongTensor(train_idx)
		val_idx = torch.LongTensor(val_idx)
		test_idx = torch.LongTensor(test_index_sorted)

	return adj, features, label, train_idx, val_idx, test_idx
