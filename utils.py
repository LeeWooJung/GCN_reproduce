import torch

def accuracy(output, label):
	
	output = torch.max(output,1)[1]
	label = label

	correct = (output == label).float().sum()
	acc = 100 * correct.item() / len(label)

	return acc
