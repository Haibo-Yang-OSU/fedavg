import os

local_lrs = [0.1, 0.01]
# global_lrs = [1, 10]
global_lrs = [0.1, 0.01]
num_users_sels = [100]
local_ep_epoches = [(1, 100), (10, 100)]
datasets = ['all_data_1_digits_1_niid', 'all_data_1_digits_10_niid',
           'all_data_1_digits_5_niid', 'all_data_1_digits_2_niid']

# coms = [0.5, 0.8, 0.9, 0.95]
# compressors = ['signSGD', 'random_drop', 'topK']
compressors = ['signSGD']
CR = 0.0
models = 'cnn'
data = 'fmnist'

for local_lr in local_lrs:
	for global_lr in global_lrs:
		for num_users_sel in num_users_sels:
			for local_ep_epoch in local_ep_epoches:
				for dataset in datasets:
					for compressor in compressors:
						suffix = "python3 run_fl.py" + " --dataset=" + data + " --dataset_name=" + str(dataset) \
							+ " --model=" + str(models) + " --compressor=" + compressor + " --compress_factor=" + str(CR) \
							+ " --num_round=" + str(local_ep_epoch[1]) \
							+ " --clients_per_round=" + str(num_users_sel) \
							+ " --num_epochs=" + str(local_ep_epoch[0]) \
							+ " --client_lr=" + str(local_lr) + " --server_lr=" + str(global_lr)
						os.system(suffix)
