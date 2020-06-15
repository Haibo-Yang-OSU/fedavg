"""
optimizer class
"""
import torch.optim as optim

from absl import flags
FLAGS = flags.FLAGS

def get_optimizer(model):
	optimizer = optim.SGD(model.parameters(), lr=FLAGS.client_lr)
	return optimizer