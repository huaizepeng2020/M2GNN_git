import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import dgl
import torch
import multiprocessing
import pickle
import os

import random
from time import time
from collections import defaultdict
import warnings
import scipy, gc

warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)

def load_data_both(model_args):
    global args
    args = model_args

    output_path = args.data_path + '/DianPing' + '/graph_tag_1002.pkl'
    with open(output_path, 'rb') as f:
        graph_dp = pickle.load(f)
        graph_dp_tag = pickle.load(f)
    print('load graph')
    del graph_dp
    gc.collect()

    output_path = args.data_path + '/DianPing' + '/train_test_1002.pkl'
    with open(output_path, 'rb') as f:
        train_cf = pickle.load(f)
        test_cf = pickle.load(f)
        len_item = pickle.load(f)
        train_user_set = pickle.load(f)
        test_user_set = pickle.load(f)
    print('load train_test')

    # print all parameters
    global n_users, n_items
    n_users = graph_dp_tag.num_nodes(ntype='user')
    n_items = graph_dp_tag.num_nodes(ntype='review')
    n_item4rs = len_item
    n_tag = graph_dp_tag.num_nodes(ntype='tag')

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_items4rs': int(n_item4rs),
        'n_tag': int(n_tag)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph_dp_tag

def load_data_both_amazon(model_args):
    global args
    args = model_args

    directory = args.data_path + args.dataset + '/'

    # load graph
    output_path = args.data_path + '/DianPing' + '/graph_tag_book2movie2.pkl'
    with open(output_path, 'rb') as f:
        graph_dp = pickle.load(f)
        graph_dp_tag = pickle.load(f)
    print('load graph')
    # del graph_dp
    gc.collect()

    output_path = args.data_path + '/DianPing' + '/train_test_book2movie2.pkl'
    with open(output_path, 'rb') as f:
        train_cf = pickle.load(f)
        test_cf = pickle.load(f)
        len_item = pickle.load(f)
        train_user_set = pickle.load(f)
        test_user_set = pickle.load(f)
    print('load train_test')

    # print all parameters
    global n_users, n_items
    n_users = graph_dp_tag.num_nodes(ntype='user')
    n_items = graph_dp_tag.num_nodes(ntype='review')
    n_item4rs = len_item
    n_tag = graph_dp_tag.num_nodes(ntype='tag')

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_items4rs': int(n_item4rs),
        'n_tag': int(n_tag)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph_dp_tag