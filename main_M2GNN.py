__author__ = "anonymity"

import os
import random
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel
import dgl
import numpy as np

from time import time
from prettytable import PrettyTable
from tqdm import tqdm
import pickle
import gc


from utils.parser import parse_args
from utils.data_loader import load_data_both
from modules.base_model import M2GNN_word
from utils.evaluate import test_multi
from utils.helper import early_stopping

from dgl.dataloading import MultiLayerNeighborSampler


n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict


def load_obj(name, directory):
    with open(directory + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def init_model():
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.port

    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return local_rank, device


if __name__ == '__main__':
    print(os.getpid())
    """fix the random seed"""
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    args.data_load = True

    """init DDP"""
    local_rank, device = init_model()

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph_tag = load_data_both(args)
    args.batch_size = int(args.batch_size / graph_tag.num_edges('interaction')
                          * (graph_tag.num_edges('t2t') + graph_tag.num_edges('interaction')))

    '''training set'''
    train_eids = {'interaction': torch.tensor(range(graph_tag.num_edges(etype='interaction'))),
                  't2t': torch.tensor(range(graph_tag.num_edges(etype='t2t')))}
    num_nei = args.max_len
    sampler_dict00 = {'u_r_t': num_nei, 'u_q_t': num_nei, 'u_p_r_t': num_nei,
                      't2t': num_nei, 'r_h_t': num_nei, 'interaction': 0, 'test': 0}
    sampler = MultiLayerNeighborSampler([sampler_dict00] * 2)
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    dataloader = dgl.dataloading.EdgeDataLoader(
        graph_tag, train_eids, sampler, device=device,
        negative_sampler=neg_sampler,
        exclude='self',
        batch_size=args.batch_size,
        shuffle=True, use_ddp=True, drop_last=False)
    '''-----------------testing set--------------------'''
    train_eids = {'test': torch.tensor(range(graph_tag.num_edges(etype='test')))}
    test_dataloader = dgl.dataloading.EdgeDataLoader(
        graph_tag, train_eids, sampler, device=device,
        exclude='self',
        batch_size=args.test_batch_size,
        shuffle=False, use_ddp=False, drop_last=False)

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_items4rs = n_params['n_items4rs']

    """define model"""
    n_params['epoch_num'] = len(train_cf) // args.batch_size + 1
    n_params['k_word2cf'] = 1
    model = M2GNN_word(n_params, args).to(device)

    model.user_embed_final = model.user_embed_final.to(device)
    model.item_embed_final = model.item_embed_final.to(device)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                    find_unused_parameters=True)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False


    print('--------------------start training------------------------')

    for epoch in range(args.epoch):
        """init DDP training"""
        dataloader.set_epoch(epoch)
        """training"""
        loss, s, cos_loss = 0, 0, 0
        train_s_t = time()


        for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
            model.train()

            batch_loss = model(input_nodes, blocks, pos_pair_graph, neg_pair_graph)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()

        train_e_t = time()

        """Evaluate on only the first GPU"""
        if local_rank == 0:
            if epoch % 5 == 4:
                with torch.no_grad():
                    model = model.eval()
                    for input_nodes, pos_pair_graph, blocks in test_dataloader:
                        model.module.generate(input_nodes, blocks, pos_pair_graph)

                    user_emb = model.module.user_embed_final
                    item_emb = model.module.item_embed_final

                """testing"""
                test_s_t = time()
                ret = test_multi(user_emb, item_emb, user_dict, n_params, model)
                test_e_t = time()

                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg",
                                         "precision",
                                         "hit_ratio"]
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg'],
                     ret['precision'], ret['hit_ratio']]
                )
                print(train_res)

                # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
                cur_best_pre_0, stopping_step, should_stop, best_flag = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                                       stopping_step,
                                                                                       expected_order='acc',
                                                                                       flag_step=args.duration_epoch)
                if should_stop:
                    break

                """save weight"""
                if ret['recall'][0] == cur_best_pre_0 and args.save:
                    torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

            else:
                print('using time %.4f, training loss at epoch %d: %.4f' % (
                    train_e_t - train_s_t, epoch, loss))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
    os.system('ps -ef | grep main_M2GNN | grep -v grep | cut -c 9-15 | xargs kill -9')
