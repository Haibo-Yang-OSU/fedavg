import pickle
import json
import numpy as np
import os
import time
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image

from absl import flags
from absl import logging


__all__ = ['mkdir', 'read_data', 'Metrics', "MiniDataset"]
FLAGS = flags.FLAGS


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def read_data():
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    train_data_file = os.path.join('./Data', FLAGS.dataset, 'train', (FLAGS.dataset_name + '.pkl'))
    test_data_file = os.path.join('./Data', FLAGS.dataset, 'test', (FLAGS.dataset_name + '.pkl'))

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read data from:', train_data_file, test_data_file)

    with open(train_data_file, 'rb') as inf:
        cdata = pickle.load(inf)
    clients.extend(cdata['users'])
    if 'hierarchies' in cdata:
        groups.extend(cdata['hierarchies'])
    train_data.update(cdata['user_data'])

    for cid, v in train_data.items():
        train_data[cid] = MiniDataset(v['x'], v['y'])


    with open(test_data_file, 'rb') as inf:
        cdata = pickle.load(inf)
    test_data.update(cdata['user_data'])

    for cid, v in test_data.items():
        test_data[cid] = MiniDataset(v['x'], v['y'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target


class Metrics(object):
    def __init__(self, name=''):
        num_rounds = FLAGS.num_round

        # Statistics in training procedure
        self.loss_on_train_data = [0] * num_rounds
        self.acc_on_train_data = [0] * num_rounds

        # Statistics in test procedure
        self.loss_on_eval_data = [0] * num_rounds
        self.acc_on_eval_data = [0] * num_rounds

        self.result_path = mkdir(os.path.join('./result', FLAGS.dataset, FLAGS.dataset_name))
        self.exp_name = 'model_{}_client_lr_{}_server_lr_{}_clients_per_round_{}_local_epochs_{}_num_round_{}_compressor_{}_compression_{}'.format(
	        FLAGS.model, FLAGS.client_lr, FLAGS.server_lr, FLAGS.clients_per_round,
	        FLAGS.num_epochs, FLAGS.num_round, FLAGS.compressor, FLAGS.compress_factor
        )

        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)

    def add_training_stats(self, round_i, train_stats):

        self.loss_on_train_data[round_i] = train_stats['loss']
        self.acc_on_train_data[round_i] = train_stats['acc']
        self.train_writer.add_scalar('train_loss', train_stats['loss'], round_i)
        self.train_writer.add_scalar('train_acc', train_stats['acc'], round_i)

    def add_test_stats(self, round_i, eval_stats):
        self.loss_on_eval_data[round_i] = eval_stats['loss']
        self.acc_on_eval_data[round_i] = eval_stats['acc']
        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def write(self):
        test_dir = os.path.join(self.result_path, self.exp_name, 'test_accuracy.txt')
        train_dir_loss = os.path.join(self.result_path, self.exp_name, 'training_loss.txt')
        train_dir_acc = os.path.join(self.result_path, self.exp_name, 'training_acc.txt')
        np.savetxt(str(train_dir_loss), self.loss_on_train_data)
        np.savetxt(str(train_dir_acc), self.acc_on_train_data)
        np.savetxt(str(test_dir), self.acc_on_eval_data)
        print('----write result finished.')
