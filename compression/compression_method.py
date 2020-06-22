"""
gradients compression methods
"""

import torch
import copy
from absl import flags

FLAGS = flags.FLAGS
_COMPRESSOR_ = ['signSGD', 'random drop sparsifaction', 'topK sparsification']

def signSGD(delta):
	return torch.sign(delta)

def random_drop_sparsification(delta):
	zero_ = torch.zeros_like(delta)
	condition = torch.rand(delta.size()) >= FLAGS.compress_factor
	if FLAGS.gpu:
		condition = condition.to(device='cuda')
	compressed_delta = torch.where(condition, delta, zero_)
	return compressed_delta

def top_value_sparsification(delta):
	delta_temp = abs(delta)
	value, index = torch.topk(delta_temp, int(len(delta_temp) * FLAGS.compress_factor), largest=False)
	zeros_ = torch.zeros_like(index).type(torch.float)
	delta.put_(index, zeros_)
	return delta

def get_compression(delta):
	"""

	:param delta:
	:return:
	"""
	# delta_ = delta.clone()
	if FLAGS.compressor == 'none':
		return delta
	elif FLAGS.compressor == 'signSGD':
		return signSGD(delta)
	elif FLAGS.compressor == 'random_drop':
		delta_ = delta.clone()
		return random_drop_sparsification(delta_)
	elif FLAGS.compressor == 'topK':
		delta_ = delta.clone()
		return top_value_sparsification(delta_)
	else:
		raise ValueError("only support compressor: {}".format(_COMPRESSOR_))

