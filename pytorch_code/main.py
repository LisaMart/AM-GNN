#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--heads', type=int, default=8, help='number of attention heads')  # новое добавление - количество голов для многоголового внимания
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/sample') # changed default diginetica for sample
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=120, help='hidden state size')
parser.add_argument('--epoch', type=int, default=15, help='the number of epochs to train for') # changed default 30 for 15
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--dynamic', type=bool, default=False)
parser.add_argument('--dot', type=float, default=0.1, help='dot product scaling factor for attention')
parser.add_argument('--l_p', type=float, default=1, help='parameter for attention regularization (default 1)') # новое добавление
parser.add_argument('--last_k', type=int, default=3, help='number of last elements to consider in attention')  # новое добавление
parser.add_argument('--use_attn_conv', action='store_true', help='whether to use attention convolution')  # новое добавление

opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(
        open('D:/python_practice/AM-GNN/datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

    print(f"Размер выборки для тренировки: {len(train_data)}")

    train_data = Data(train_data, shuffle=True, opt=opt)
    test_data = Data(test_data, shuffle=False, opt=opt)
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'diginetica_users':
        n_node = 57070
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraphWithMultiLevelAttention(opt, n_node, max(train_data.len_max, test_data.len_max)))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print(f'epoch:  {epoch}/{opt.epoch-1}')
        precision_at_k_mean, mrr_mean = train_test(model, train_data, test_data)
        flag = 0
        if precision_at_k_mean >= best_result[0]:
            best_result[0] = precision_at_k_mean
            best_epoch[0] = epoch
            flag = 1
        if mrr_mean >= best_result[1]:
            best_result[1] = mrr_mean
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print(f'\tPrecision@{K}:\t{best_result[0]:.4f}\tMMR@{K}:\t{best_result[1]:.4f}\tEpoch:\t{best_epoch[0]},{best_epoch[1]}')
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()