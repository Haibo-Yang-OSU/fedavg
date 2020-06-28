"""
mnist non-iid generator
"""
import torch
import torchvision
import numpy as np
import os
import copy
import pickle

def digits_split(features, labels, balance = True):
    # list of 10 with each a list of training data for one digit
    mnist_digits = []
    for number in range(10):
        idx = labels == number
        mnist_digits.append(features[idx])

    if balance:
        minNumber = min([len(dig) for dig in mnist_digits])
        for number in range(10):
            mnist_digits[number] = mnist_digits[number][:minNumber-1]

        print(">>> Data is balanced")
    data_size = [len(dig) for dig in mnist_digits]
    print("size of data list: ", data_size)
    return mnist_digits, np.array(data_size)

def assign_data(traindata_list, testdata_list, digit_num, repeated_train, repeated_test):
    train_X = [[] for _ in range(WORKERS)]
    train_Y = [[] for _ in range(WORKERS)]
    test_X = [[] for _ in range(WORKERS)]
    test_Y = [[] for _ in range(WORKERS)]
    print("begin assign data")
    print("repeating number: {}, {}".format(repeated_train, repeated_test))
    for worker in range(WORKERS):
        print("samples in train dataset: {}".format([len(v) for v in traindata_list]))
        available = choose_digit(traindata_list, digit_num, repeated_train)
        print("worker id: {}, selected digits: {}".format(worker, available))
        for i in available:
            for _ in range(repeated_train):
                train_X[worker].append(traindata_list[i][-1])
                traindata_list[i] = traindata_list[i][:-1]
                train_Y[worker] += [i]
            for _ in range(repeated_test):
                test_X[worker].append(testdata_list[i][-1])
                testdata_list[i] = testdata_list[i][:-1]
                test_Y[worker] += [i]
    return train_X, train_Y, test_X, test_Y

def choose_digit(data_list, digit_num, repeat_num):
    available_digit = []
    for i, aList in enumerate(data_list):
        if len(aList) > repeat_num:
            available_digit.append(i)

    # try:
    # 	lst = np.random.choice(available_digit, digit_num, replace=False).tolist()
    # except:
    # 	raise ValueError("random choose digits failure")

    try:
        lst = np.random.choice(available_digit, digit_num, replace=False).tolist()
    except:
        indx = [len(v) for v in data_list]
        lst = available_digit
        while (len(lst) < digit_num):
            max_indx = indx.index(max(indx))
            lst += [max_indx]
            indx[max_indx] -= 1
        # print(available_digit)
    # print('choose digits:', lst)
    return lst

def save_data(train_X, train_y, test_X, test_y, digit_num):
    # Setup directory for train/test data
    print('>>> Begin saving data.')
    image = 1 if IMAGE_DATA else 0
    train_path = '{}/train/all_data_{}_digits_{}_niid.pkl'.format(DATASET_FILE, image, digit_num)
    test_path = '{}/test/all_data_{}_digits_{}_niid.pkl'.format(DATASET_FILE, image, digit_num)

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 100 users
    for i in range(WORKERS):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile)

        print('>>> Save data.')


cpath = os.path.dirname(__file__)
DATASET_FILE = os.path.join(cpath, 'data_temp', 'fmnist')

trainset = torchvision.datasets.FashionMNIST(DATASET_FILE, download=True, train=True, transform=torchvision.transforms.ToTensor())
testset = torchvision.datasets.FashionMNIST(DATASET_FILE, download=True, train=False,transform=torchvision.transforms.ToTensor())

trainX_np = trainset.train_data.numpy()
trainY_np = trainset.train_labels.numpy()
testX_np = testset.test_data.numpy()
testY_np = testset.test_labels.numpy()

WORKERS = 100
DIGITS = [1, 2, 5, 10]
USER_DIGIT_SIZE = 10
IMAGE_DATA = True # image for CNN
SAVE = True
np.random.seed(6)
# print(trainset.train_data.size())


train_features, train_list_size = digits_split(trainX_np, trainY_np)
test_features, test_list_size = digits_split(testX_np, testY_np)
train_size = min(train_list_size)
test_size = min(test_list_size)


# print(type(train_features), train_features[0][0].shape)
for digit in DIGITS:
    repeated_train = int(train_size * 10 / (WORKERS * digit))
    repeated_test = int(test_size * 10 / (WORKERS * digit))

    temp_train = copy.deepcopy(train_features)
    temp_test = copy.deepcopy(test_features)
    train_X, train_Y, test_X, test_Y = assign_data(temp_train, temp_test, digit, repeated_train, repeated_test)
    save_data(train_X, train_Y, test_X, test_Y, digit)
