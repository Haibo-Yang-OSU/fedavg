import os

local_lrs = [0.1, 0.001]
global_lrs = [1, 10]
num_users_sels = [10, 100]
local_ep_epoches = [(1, 100), (5, 100), (10, 100)]
datasets = ['all_data_0_digits_1_niid', 'all_data_0_digits_10_niid',
           'all_data_0_digits_5_niid', 'all_data_0_digits_2_niid']

coms = [0.5, 0.8, 0.9, 0.95]

# local_lrs = [0.01]
# global_lrs = [1]
# num_users_sels = [10]
# local_ep_epoches = [(1, 20)]
# datasets = ['mnist_all_data_0_digits_1_niid.pkl']

# models = 'logistic'
models = '2nn'

for local_lr in local_lrs:
	for global_lr in global_lrs:
		for num_users_sel in num_users_sels:
			for local_ep_epoch in local_ep_epoches:
				for dataset in datasets:
					for com in coms:
						suffix = "python3 run_fl.py " + "--dataset_name=" + str(dataset) \
							+ " --model=" + str(models) + " --compress_factor=" + str(com) \
							+ " --num_round=" + str(local_ep_epoch[1]) \
							+ " --clients_per_round=" + str(num_users_sel) \
							+ " --num_epochs=" + str(local_ep_epoch[0]) \
							+ " --client_lr=" + str(local_lr) + " --server_lr=" + str(global_lr)
						os.system(suffix)
