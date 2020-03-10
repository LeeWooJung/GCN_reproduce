import os
import numpy as np
import scipy.sparse as sp
import networkx as nx

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

	return features, graph
