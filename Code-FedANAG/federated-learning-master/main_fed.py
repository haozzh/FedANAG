#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import ssl

import copy
import random
import torch
import numpy as np
from utils.options import args_parser
from utils.seed import setup_seed
from utils.logg import get_logger
from models.Nets import client_model
from utils.dataset import DatasetObject, ShakespeareObjectCrop_noniid
from models.distributed_training_utils import Client, Server
torch.set_printoptions(
    precision=8,
    threshold=1000,
    edgeitems=3,
    linewidth=150, 
    profile=None,
    sci_mode=False  
)
if __name__ == '__main__':

    ssl._create_default_https_context = ssl._create_unverified_context
    # parse args
    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    data_path = 'Folder/'
    data_obj = DatasetObject(dataset=args.dataset, n_client=args.num_users, seed=args.seed, rule=args.iid, rule_arg=args.rule_arg, data_path=data_path)
    #data_obj = ShakespeareObjectCrop_noniid(data_path=data_path, dataset_prefix='dataset_prefix')
    # build model
    if args.model == 'cnn' and args.dataset == 'CIFAR100':
        net_glob = client_model('cifar100_LeNet').to(args.device)
    elif args.model == 'cnn' and args.dataset == 'CIFAR10':
        net_glob = client_model('cifar10_LeNet').to(args.device)
    elif args.model == 'logistic' and args.dataset == 'emnist':
        net_glob = client_model('Linear', [1 * 28 * 28, 10]).to(args.device)
    elif args.model == 'rnn' and args.dataset == 'shakespeare':
        net_glob = client_model('shakes_LSTM').to(args.device)
    else:
        exit('Error: unrecognized model')

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    clients = [Client(model=copy.deepcopy(net_glob).to(args.device), args=args, trn_x=data_obj.clnt_x[i],
                      trn_y=data_obj.clnt_y[i], dataset_name=data_obj.dataset, id_num=i) for i in range(args.num_users)]
    server = Server((net_glob).to(args.device), args)

    logger = get_logger(args.filepath)
    logger.info('start training!')
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')

    for iter in range(args.epochs):

        net_glob.train()

        m = max(int(args.frac * args.num_users), 1)
        participating_clients = random.sample(clients, m)
        for client in participating_clients:
            client.synchronize_with_server(server)
            client.compute_bias(server)
            client.compute_weight_update(server, iter=iter)


        server.aggregate_weight_updates(clients=participating_clients, iter=iter)

        server.computer_weight_update_down_dw(clients=participating_clients, iter=iter)

        results_train, loss_train1 = server.evaluate(data_x=cent_x, data_y= cent_y,
                                                     dataset_name=data_obj.dataset)

        # practical aggregation
        #The accuracy of the test dataset shown in the paper
        results_test, loss_test1 = server.evaluate(data_x=data_obj.tst_x, data_y=data_obj.tst_y,
                                                  dataset_name=data_obj.dataset)

        logger.info('Epoch:[{}]\tlr =\t{:.5f}\tloss=\t{:.5f}\tloss2=\t{:.5f}\t acc_train=\t{:.5f}\tacc_test=\t{:.5f}'.
                    format(iter, args.lr, loss_train1, loss_test1, results_train, results_test))


        args.lr = args.lr * (args.lr_decay)
    logger.info('finish training!')






