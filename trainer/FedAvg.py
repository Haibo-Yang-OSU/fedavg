"""
FedAvg class as trainer
"""

# To do: model, optimizer
# To do: client, server

import multiprocessing

from models import model
from trainer import client
from utils import read_write_data


from utils.torch_utils import get_flat_params_from, set_flat_params_to


import torch
import numpy as np
from absl import flags
import time
FLAGS = flags.FLAGS


# def multitrain(c, server_flat_params, round_num):
# 	delta, metric = c.local_train(server_flat_params, round_num=round_num)
# 	print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
# 	      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
# 	      "Delta: norm {:>.4f} ({:>.4f}->{:>.4f})| "
# 	      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
# 		round_num, metric['id'], i, FLAGS.clients_per_round,
# 		metric['grad_norm'], metric['grad_min'], metric['grad_max'],
# 		metric['delta_norm'], metric['delta_min'], metric['delta_max'],
# 		metric['train_loss'], metric['train_acc'] * 100, metric['time']))
# 	return delta, metric

class Server(object):
	"""
	server class
	"""

	def __init__(self, dataset):
		self.dataset = dataset

		self.model = model.get_model()
		self.move_model_to_gpu()
		# self.global_optimizer :  To do

		self.round_num = 0

		print('>>> Activate a server for training')

		self.clients = self.setup_clients()
		print('>>> Activate {} clients for training'.format(len(self.clients)))

		self.output_metric = read_write_data.Metrics()

	def move_model_to_gpu(self):
		if FLAGS.gpu:
			torch.cuda.set_device(FLAGS.device)
			torch.backends.cudnn.enabled = True
			self.model.cuda()
			print('>>> Server use gpu on device {}'.format(FLAGS.device))
		else:
			print('>>> Server do not use gpu')

	def setup_clients(self):
		"""
		Instantiates clients based on given train and test data
		Returns:
			all_clients: List of clients
		"""
		users, groups, train_data, test_data = self.dataset
		# print("Activate clients number: {}".format(len(users)))
		if len(groups) == 0:
			groups = [None for _ in users]

		all_clients = []
		for user, group in zip(users, groups):
			if isinstance(user, str) and len(user) >= 5:
				user_id = int(user[-5:])
			else:
				user_id = int(user)
			# self.all_train_data_num += len(train_data[user])
			data_frame = (user_id, group, train_data[user], test_data[user])
			c = client.Client(data_frame, self.round_num)
			all_clients.append(c)
		return all_clients

	def select_clients(self, seed=1, replace=False):
		"""Selects num_clients clients weighted by number of samples from possible_clients

		Args:
			1. seed: random seed
			2. num_clients: number of clients to select; default 20
				note that within function, num_clients is set to min(num_clients, len(possible_clients))

		Return:
			list of selected clients objects
		"""
		num_clients = min(FLAGS.clients_per_round, len(self.clients))
		np.random.seed(seed)
		return np.random.choice(self.clients, num_clients, replace=replace).tolist()


	def local_train(self, sample_clients, round_num):
		"""
		local train process for each client
		:return:
		delta: (num_sampels, delta)
		metric: list of dict of performance information
		"""

		deltas = []
		metrics = []
		server_flat_params = self.get_flat_model_params()

		# cores = multiprocessing.cpu_count()
		# pool = multiprocessing.Pool(processes=cores)
		# results = pool.starmap_async(multitrain, [(c, server_flat_params, round_num) for c in sample_clients]).get()
		# for delta, metric in results:
		# 	deltas.append(delta)
		# 	metrics.append(metric)

		for i, c in enumerate(sample_clients):
			delta, metric = c.local_train(server_flat_params, round_num=round_num)

			# print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
			# 	    "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
			#         "Delta: norm {:>.4f} ({:>.4f}->{:>.4f})| "
			# 	    "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
			# 		round_num, metric['id'], i, FLAGS.clients_per_round,
			# 		metric['grad_norm'], metric['grad_min'], metric['grad_max'],
			# 	    metric['delta_norm'], metric['delta_min'], metric['delta_max'],
			# 		metric['train_loss'], metric['train_acc'] * 100, metric['time']))

			deltas.append(delta)
			metrics.append(metric)
		return deltas, metrics


	def aggregate(self, deltas):
		"""
		aggregate delta from clients
		:return:
		"""
		num = 0
		previous_para = self.get_flat_model_params()
		averaged_delta = torch.zeros_like(previous_para)
		for num_samples, client_delta in deltas:
			num += 1
			averaged_delta += client_delta
		averaged_delta /= num
		# new_para = previous_para + FLAGS.server_lr * averaged_delta
		return averaged_delta


	def train(self):
		"""
		train process
		:return:
		"""
		print('----sampling {} clients per communication round \n'.format(FLAGS.clients_per_round))

		for round_num in range(FLAGS.num_round):

			self.test_latest_model_on_traindata(round_num)
			self.test_latest_model_on_testdata(round_num)

			sample_clients = self.select_clients(seed=round_num, replace=False)


			deltas, metrics = self.local_train(sample_clients, round_num)

			# aggregate clients delta and update the server model
			average_delta = self.aggregate(deltas)
			self.update_server_state(average_delta)

			# permance analysis of each clients
			# function of metrics

		self.test_latest_model_on_traindata(round_num)
		self.test_latest_model_on_testdata(round_num)

		self.output_metric.write()

	def test_latest_model_on_traindata(self, round_num):
		"""

		:param round_num:
		:return:
		"""
		begin_time = time.time()
		stats = self.local_test(use_eval_data=False)
		print('>>> Training info in Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} \n'.format(
			round_num, stats['acc'], stats['loss']))
		self.output_metric.add_training_stats(round_num, stats)

	def test_latest_model_on_testdata(self, round_num):
		"""

		:param round_num:
		:return:
		"""
		begin_time = time.time()
		stats = self.local_test(use_eval_data = True)
		print('>>> Testing info in Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} \n'.format(
			round_num, stats['acc'], stats['loss']))
		self.output_metric.add_test_stats(round_num, stats)


	def local_test(self, use_eval_data=True):
		assert self.model is not None
		flat_para = get_flat_params_from(self.model)

		num_samples = []
		tot_corrects = []
		losses = []
		for c in self.clients:
			tot_correct, num_sample, loss = c.local_test(flat_para, use_eval_data=use_eval_data)

			tot_corrects.append(tot_correct)
			num_samples.append(num_sample)
			losses.append(loss)

		# ids = [c.cid for c in self.clients]
		# groups = [c.group for c in self.clients]

		stats = {'acc': sum(tot_corrects) / sum(num_samples),
		         'loss': sum(losses) / sum(num_samples),
		         'num_samples': num_samples}

		return stats


	def update_server_state(self, average_delta):
		pre_para = self.get_flat_model_params()
		update_para = pre_para + FLAGS.server_lr * average_delta
		self.set_flat_model_params(update_para)


	def set_flat_model_params(self, flat_params):
		set_flat_params_to(self.model, flat_params)

	def get_flat_model_params(self):
		flat_params = get_flat_params_from(self.model)
		return flat_params.detach()


