"""
client class
"""

import time
from torch.utils.data import DataLoader
from absl import flags
import logging

from utils.flops_counter import get_model_complexity_info
from models import model
from optimizer import optimizers
from compression import compression_method

from utils.torch_utils import get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch

FLAGS = flags.FLAGS
criterion = nn.CrossEntropyLoss()

class Client(object):
    """Base class for all local clients

    Outputs of gradients or local_solutions will be converted to np.array
    in order to save CUDA memory.
    """

    def __init__(self):
        self.logger = logging.getLogger('main.client')
        self.logger.info("Initial a client.")

    def reset(self, dataset, errorfeedback, round_num):
        self.cid = dataset[0]
        self.train_data = dataset[1]
        self.test_data = dataset[2]
        self.model = model.get_model()
        self.move_model_to_gpu()
        self.error = errorfeedback #torch.zeros_like(self.get_flat_model_params())
        self.num_epochs = FLAGS.num_epochs
        self.round_num = round_num
        self.lr = FLAGS.client_lr
        # self.lr = self.get_lr()
        self.optimizer = optimizers.get_optimizer(self.model)

        self.train_dataloader = DataLoader(self.train_data, batch_size=FLAGS.client_batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=FLAGS.client_batch_size, shuffle=False)

        self.input_shape = model.get_input_info()['input_shape']

        self.logger.debug("reset client %d", self.cid)

        # Setup local model and evaluate its statics
        # self.flops, self.params_num, self.model_bytes = \
        #     get_model_complexity_info(self.model, self.input_shape)

    def move_model_to_gpu(self):
        if FLAGS.gpu:
            torch.backends.cudnn.enabled = True
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                self.debug("Client %d use data parallel, Availabel GPUs: %d", self.cid, torch.cuda.device_count())
            # torch.cuda.set_device(FLAGS.device)
                self.model.to(FLAGS.device)
            else:
                self.model.to(FLAGS.device)
                self.logger.debug("Client %d use one GPU: %d", self.cid, FLAGS.device)
        else:
            self.logger.debug("Client %d use one CPU", self.cid)


    def get_lr(self):
        # To do: return lr based on round_num
        if FLAGS.client_lr_schedule == 'constant':
            return FLAGS.client_lr
        elif FLAGS.client_lr_schedule == 'stepLR':
            return FLAGS.client_lr * (FLAGS.client_lr_decay ** (FLAGS.round_num // FLAGS.client_decay_epochs))
        else:
            raise ValueError("Donot support the learning rate schedule.")


    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)


    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()


    def local_train(self, flat_params, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local delta= latest model - previous model, rather than local model
            2. Return Dict contain
                2.1 Statistic Dict
                    2.1.1: bytes_write: number of bytes transmitted
                    2.1.2: bytes_read: number of bytes received
                    2.1.3: comp: number of FLOPs executed in training process
                    2.1.4: running time for this communiction round
                2.2 Parameter Dict
                    latest model parameter's norm, max, min
                2.3 Delta Dict
                    delta's norm, max, min
                2.4 Loss Dict
                    training loss/accuracy: list of each local epoch
        """
        # updata clients' model based on server's model
        t0 = time.time()
        self.set_flat_model_params(flat_params)
        self.previous_model = self.get_flat_model_params()

        self.model.train()
        train_loss = 0
        train_acc = 0
        train_total = 0
        for epoch in range(self.num_epochs):
            train_loss = train_acc = train_total = 0
            for batch_idx, (x, y) in enumerate(self.train_dataloader):
                if FLAGS.gpu:
                    x, y = x.cuda(device=FLAGS.device), y.cuda(device=FLAGS.device)

                self.optimizer.zero_grad()
                pred = self.model(x)

                if torch.isnan(pred.max()):
                    raise ValueError(">>> gradients blow up for Nan.")

                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
            # train_loss.append(train_loss_ep)
            # train_acc.append(train_acc_ep)
            # train_total.append(train_total_ep)

        local_new_model = self.get_flat_model_params()
        return_delta = local_new_model - self.previous_model

        if FLAGS.error_feedback:
            return_delta_temp = return_delta + self.error
            return_delta = compression_method.get_compression(return_delta_temp)
            self.error = return_delta_temp - return_delta
        else:
            # return_delta_temp = return_delta
            return_delta = compression_method.get_compression(return_delta)

        return_dict = {}

        # bytes_w = self.model_bytes
        # comp = self.num_epochs * train_total * self.flops
        # bytes_r = self.model_bytes
        # stats_dict = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
        #          "comp": comp,  "time": round(end_time - begin_time, 2)}
        # return_dict.update(stats_dict)
        #
        stats_dict = {'id': self.cid, "time": round(time.time() - t0, 2)}
        return_dict.update(stats_dict)
        #
        param_dict = {"grad_norm": torch.norm(local_new_model).item(),
                      "grad_max": local_new_model.max().item(),
                      "grad_min": local_new_model.min().item()}
        return_dict.update(param_dict)

        delta_dict = {"delta_norm": torch.norm(return_delta).item(),
                      "delta_max": return_delta.max().item(),
                      "delta_min": return_delta.min().item()}
        return_dict.update(delta_dict)

        loss_dict = {"train_loss": train_loss/train_total,
                     "train_acc": train_acc/train_total}
        return_dict.update(loss_dict)

        error_dict = {"error_norm": torch.norm(self.error).item(),
                      "error_max": self.error.max().item(),
                      "error_min": self.error.min().item()}
        return_dict.update(error_dict)


        return (len(self.train_data), return_delta), return_dict, self.error


    def local_test(self, flat_params, use_eval_data=True):
        """Test current model on local eval data

        Returns:
            1. tot_correct: total # correct predictions
            2. test_samples: int
        """
        if use_eval_data:
            dataloader, dataset = self.test_dataloader, self.test_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data

        tot_correct, loss = self.local_test_from(flat_params, dataloader)

        return tot_correct, len(dataset), loss


    def local_test_from(self, flat_params, test_dataloader):
        self.set_flat_model_params(flat_params)
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                if FLAGS.gpu:
                    x, y = x.cuda(device=FLAGS.device), y.cuda(device=FLAGS.device)

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss

