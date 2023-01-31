from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
import dgl
from tqdm import tqdm
import pickle
import gc
from time import time
# from lshash import LSHash
import faiss


seed = 2022
np.random.seed(seed)

cores = multiprocessing.cpu_count() // 6
cores = 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.local_rank)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def load_obj(name, directory):
    with open(directory + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def eval_topK(pred, ture, K):
    n_test_users = len(pred)
    pool = multiprocessing.Pool(3)

    batch = zip(pred, ture, [K] * len(pred))
    batch_result = pool.map(test_one_user0, batch)

    result = {'precision': 0,
              'recall': 0,
              'ndcg': 0,
              'hit_ratio': 0}

    for re in batch_result:
        result['precision'] += re['precision'] / n_test_users
        result['recall'] += re['recall'] / n_test_users
        result['ndcg'] += re['ndcg'] / n_test_users
        result['hit_ratio'] += re['hit_ratio'] / n_test_users

    return result


def eval_topK0(pred, ture, K):
    n_test_users = len(pred)
    pool = multiprocessing.Pool(8)

    batch = zip(pred, ture, [K] * len(pred))
    batch_result = pool.map(test_one_user0, batch)

    result = {'precision': 0,
              'recall': 0,
              'ndcg': 0,
              'hit_ratio': 0}

    for re in batch_result:
        result['precision'] += re['precision']
        result['recall'] += re['recall']
        result['ndcg'] += re['ndcg']
        result['hit_ratio'] += re['hit_ratio']

    pool.close()

    return result


def test_one_user0(x):
    pred_c = x[0]
    true_c = x[1]
    k = x[2]

    r = []
    for i in pred_c:
        if i in true_c:
            r.append(1)
        else:
            r.append(0)

    # precision
    precision_c = np.mean(np.asarray(r[:k]))
    # recall
    recall_c = np.sum(np.asarray(r[:k])) / len(true_c)
    # hit
    if np.sum(np.asarray(r[:k])) > 0:
        hit_c = 1.
    else:
        hit_c = 0.
    # ndcg
    method = 1
    if len(true_c) > k:
        sent_list = [1.0] * k
    else:
        sent_list = [1.0] * len(true_c) + [0.0] * (k - len(true_c))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        ndcg_c = 0.
    else:
        ndcg_c = dcg_at_k(r, k, method) / dcg_max

    return {'recall': recall_c, 'precision': precision_c, 'ndcg': ndcg_c, 'hit_ratio': hit_c}


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def test_multi(user_emb, item_emb, user_dict, n_params, model):
    # lsh = LSHash(8, item_emb.shape[1], num_hashtables=2)
    # lsh.index(item_emb.cpu().numpy())

    # index = faiss.IndexFlatL2(item_emb.shape[1])

    nlist = 100  # 聚类中心的个数
    k = 4
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    gpu_id = 0
    flat_config.device = gpu_id
    item_faiss = item_emb.cpu().numpy().astype('float32')
    index = faiss.GpuIndexFlatIP(res, item_faiss.shape[1], flat_config)
    print('start train faiss')
    index.add(item_faiss)
    print('finish train faiss')

    # index = faiss.IndexLSH(item_emb.shape[1],16)
    # index.add(item_emb.cpu().numpy())

    # result = {'precision': 0.,
    #           'recall': 0.,
    #           'ndcg': 0.,
    #           'hit_ratio': 0.}
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}

    global n_users, n_items
    n_items = n_params['n_items4rs']
    # n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    n_test_users = len(test_user_set)
    n_user_batchs = n_test_users // args.test_batch_size + 1

    all_test_user = list(test_user_set.keys())
    with torch.no_grad():
        # for u_batch_id in tqdm(range(n_user_batchs), desc='testtt'):
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * args.test_batch_size
            end = (u_batch_id + 1) * args.test_batch_size

            user_idx = []
            y_entity_true = []
            for i in all_test_user[start:end]:
                user_idx.append(i)
                y_entity_true.append(test_user_set[i])

            K_max = max(Ks)
            # y_entity_pred = []

            # rating = torch.mm(user_emb[user_idx], item_emb.T)
            rating = torch.mm(user_emb[user_idx].cpu(), item_emb.cpu().T)
            # for user_cc in tqdm(user_idx,desc=):
            #     re = lsh.query(user_emb[user_cc].cpu().numpy(), num_results=K_max)
            #     y_entity_pred.append([iii[0] for iii in re])

            # _, y_entity_pred = index.search(user_emb[user_idx].cpu().numpy(), K_max)
            # y_entity_pred = y_entity_pred.tolist()

            # K_max = max(Ks)
            # y_entity_pred = []

            # rating = torch.mm(user_emb[user_idx].cpu(), item_emb.cpu().T)
            # _, idx_rank = rating.topk(k=K_max, dim=1, largest=True, sorted=True)
            # y_entity_pred = idx_rank.numpy().tolist()

            _, I = index.search(user_emb[user_idx].cpu().numpy(), K_max)
            y_entity_pred = I.tolist()

            for idx, eval_k in enumerate(Ks):
                re_entity = eval_topK0(y_entity_pred, y_entity_true, eval_k)

                result['precision'][idx] += re_entity['precision']
                result['recall'][idx] += re_entity['recall']
                result['ndcg'][idx] += re_entity['ndcg']
                result['hit_ratio'][idx] += re_entity['hit_ratio']

            # re_entity = eval_topK0(y_entity_pred, y_entity_true, 20)
            # result['precision'] += re_entity['precision']
            # result['recall'] += re_entity['recall']
            # result['ndcg'] += re_entity['ndcg']
            # result['hit_ratio'] += re_entity['hit_ratio']

    result['precision'] = result['precision'] / n_test_users
    result['recall'] = result['recall'] / n_test_users
    result['ndcg'] = result['ndcg'] / n_test_users
    result['hit_ratio'] = result['hit_ratio'] / n_test_users

    return result


def get_feed_dict(train_entity_pairs, start, end, train_user_set, n_items):
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
