"""
gradients compression methods
"""

import torch
from absl import flags

FLAGS = flags.FLAGS

def get_compression(delta):
	"""

	:param delta:
	:return:
	"""
	delta_temp = delta.clone()
	zero_ = torch.zeros_like(delta_temp)
	condition = torch.rand(delta_temp.size()) >= FLAGS.compress_factor
	if FLAGS.gpu:
		condition = condition.to(device='cuda')
	compressed_delta = torch.where(condition, delta_temp, zero_)
	return compressed_delta

