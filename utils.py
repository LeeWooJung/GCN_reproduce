import torch

def accuracy(output, label):
	
	output = torch.max(output,1)[1]
	label = label

	correct = (output == label).float().sum()
	acc = 100 * correct.item() / len(label)

	return acc

def stop_criterion(accuracy, window):

	for i in range(1, window):
		if accuracy[i-1] > accuracy[i]:
			return False
	return True

