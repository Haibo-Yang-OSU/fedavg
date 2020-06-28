"""
federated learning FedAvg with both client side and server side optimizer
"""

import os
import sys

from absl import flags
from absl import logging

import torch

from utils import read_write_data
from trainer import FedAvg




def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def read_options():
    """
    read all the experiment parameter
    :return: None
    """
    flags.DEFINE_string('client_optimizer', 'sgd', 'name of the client optimizer.')
    flags.DEFINE_float('client_lr', 0.1, 'learning rate.')
    # flags.DEFINE_enum('client_lr_schedule', 'constant', ['constant', 'exp_decay', 'inv_lin_decay', 'inv_sqrt_decay'],
                      # 'learning rate schedule of client.')
    # flags.DEFINE_integer('client_decay_epochs', 25, 'Number of epochs before decaying the learning rate.')
    # flags.DEFINE_float('client_lr_decay', 0.1, 'How much to decay the learning rate by at each stage.')
    flags.DEFINE_integer('client_batch_size', 64, 'Size of batches for training and eval.')

    flags.DEFINE_string('server_optimizer', 'sgd', 'name of the optimizer.')
    flags.DEFINE_float('server_lr', 0.01, 'learning rate.')
    # flags.DEFINE_enum('server_lr_schedule', 'constant', ['constant', 'exp_decay', 'inv_lin_decay', 'inv_sqrt_decay'],
                      # 'learning rate schedule of server.')
    # flags.DEFINE_integer('server_decay_epochs', 25, 'Number of epochs before decaying the learning rate.')
    # flags.DEFINE_float('server_lr_decay', 0.1, 'How much to decay the learning rate by at each stage.')

    flags.DEFINE_integer('clients_per_round', 100, 'Number of clients for each communication round.')
    flags.DEFINE_integer('num_epochs', 5, 'Number of epochs to local train.')
    flags.DEFINE_integer('num_round', 2, 'Number of communication round.')

    flags.DEFINE_enum('compressor', 'none', ['none', 'signSGD', 'random_drop', 'topK', 'unbiased_drop'], 'Which model to use for classification.')
    flags.DEFINE_float('compress_factor', 0.0, 'gradients compression factor')

    # flags.DEFINE_string('model', 'cnn', 'name of the model.')
    flags.DEFINE_enum('model', 'cnn', ['lenet', 'cnn', 'ccnn', 'resnet18'], 'Which model to use for classification.')

    flags.DEFINE_string(
        'dataset', 'mnist',
        'Dataset name. Root name of the output directory.')
    flags.DEFINE_string(
        'dataset_name', 'all_data_1_digits_2_niid',
        'name of the data file for different partition for non-iid data')

    flags.DEFINE_integer('device', 0, 'CUDA device.')
    flags.DEFINE_bool('gpu', True, 'GPU or not, default to use GPU')

    flags.DEFINE_bool('error_feedback', False, 'use error feedback or not')


    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    if not torch.cuda.is_available():
        FLAGS.gpu = None


def main():
    read_options()
    FLAGS = flags.FLAGS

    # data_inf is a tuple (client_id, grops, training_data, test_data
    data_inf = read_write_data.read_data()
    print('>>> Read data completed')

    print('model_{}_client_lr_{}_server_lr_{}_clients_per_round_{}_local_epochs_{}_num_round_{}_compressor_{}_compression_{}_feedback_{}'.format(
        FLAGS.model, FLAGS.client_lr, FLAGS.server_lr, FLAGS.clients_per_round,
        FLAGS.num_epochs, FLAGS.num_round, FLAGS.compressor, FLAGS.compress_factor, FLAGS.error_feedback
    ))
    fedavg_server = FedAvg.Server(data_inf)
    fedavg_server.train()


if __name__ == '__main__':
    main()










