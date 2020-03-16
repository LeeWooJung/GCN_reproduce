import torch

def accuracy(output, label):
	
	output = torch.max(output,1)[1]
	label = label

	correct = (output == label).float().sum()
	acc = 100 * correct.item() / len(label)

	return acc

def stop_criterion(window, size):
	for i in range(1, 1+size):
		if window[-i] <  window[-i-1]:
			return False
	return True
