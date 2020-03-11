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

def reverseOneHot(label):

	label = np.where(label == 1)[1] + 1
	label = label.reshape(-1,1)

	return label

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
	if dtype == 'citation':
		candidate = ['x','tx','allx','graph']
		obj = []
		dpath = os.path.join('./data',dname)

		for name in candidate:
			with open("{}/ind.{}.{}".format(dpath,dname,name),'rb') as f:
				obj.append(np.load(f, allow_pickle=True,encoding='latin1'))
		x, tx, allx, graph = tuple(obj)

		test_index = []
		for line in open("{}/ind.{}.test.index".format(dpath,dname)):
			test_index.append(int(line.strip()))

		# sort the test index
		test_index_sorted = np.sort(test_index)

		if dname == 'citeseer':
			# fill zero vectors to isolated test nodes
			full_test_index = range(test_index_sorted[0], test_index_sorted[-1]+1)
			tx_ = sp.lil_matrix((len(full_test_index), x.shape[1]))
			tx = tx_

		features = sp.vstack((allx, tx)).tolil()
		features[test_index,:] = features[test_index_sorted,:]
		adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

		features = normalize(features)

		adj = adj.astype(np.float32)
		adj = adj + sp.eye(adj.shape[0]) # Add self connections
		adj = normalize(adj)

		train_label = np.load("{}/ind.{}.ally".format(dpath,dname), allow_pickle=True, encoding='latin1')
		test_label = np.load("{}/ind.{}.ty".format(dpath,dname), allow_pickle=True, encoding='latin1')

		train_label = reverseOneHot(train_label)
		test_label = reverseOneHot(test_label)

		label = np.vstack((train_label, test_label))
		label[test_index,:] = label[test_index_sorted,:]

		adj = sparse_to_torchTensor(adj)
		features = torch.FloatTensor(features.todense())
		label = torch.LongTensor(label)

	return adj, features, label
