__author__ = "anonymity"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean
import dgl
import math


class M2GNN_word(nn.Module):
    def __init__(self, data_config, args_config):
        super(M2GNN_word, self).__init__()

        self.n_users = data_config['n_users']
        self.n_reviews = data_config['n_items']
        self.n_items = data_config['n_items4rs']
        self.n_tags = data_config['n_tag']
        self.K_word2cf = data_config['k_word2cf']

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.dim = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.iteration = args_config.iteration
        self.max_K = args_config.max_K
        self.max_len = args_config.max_len
        self.gamma = args_config.gamma

        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        gain = 1.414

        self.all_embed = initializer(torch.empty(int(self.n_users + self.n_reviews + self.n_tags), self.dim), gain=gain)

        # self.v_embeddings = nn.Embedding(self.n_tags, self.dim, sparse=True)
        self.v_embeddings = nn.Embedding(self.n_tags, self.dim)
        self.v_embeddings.weight.data.uniform_(-0, 0)

        self.user_embed_final = torch.zeros(size=(self.n_users, self.dim))
        self.item_embed_final = torch.zeros(size=(self.n_items, self.dim))

    def _init_model(self):
        return M2GNN_c(dim=self.dim,
                       n_hops=self.context_hops,
                       n_users=self.n_users,
                       n_reviews=self.n_reviews,
                       n_items=self.n_items,
                       n_tags=self.n_tags,
                       iteration=self.iteration,
                       max_K=self.max_K,
                       max_len=self.max_len,
                       gamma=self.gamma,
                       ratedrop=self.mess_dropout_rate)

    def forward(self, input_nodes, blocks, pos_pair_graph, neg_pair_graph):
        user_gcn_emb, item_gcn_emb = self.gcn(blocks, input_nodes, self.all_embed, True)

        tag_embed = self.all_embed[self.n_users + self.n_reviews:, :]

        # u_e = neg_pair_graph.ndata[dgl.NID]['user'][pos_pair_graph.edges(etype='interaction')[0]]
        # pos_e = neg_pair_graph.ndata[dgl.NID]['review'][pos_pair_graph.edges(etype='interaction')[1]]
        # neg_e = neg_pair_graph.ndata[dgl.NID]['review'][neg_pair_graph.edges(etype='interaction')[1]]

        u_e = user_gcn_emb[pos_pair_graph.edges(etype='interaction')[0]]
        pos_e = item_gcn_emb[pos_pair_graph.edges(etype='interaction')[1], :]
        neg_e = item_gcn_emb[neg_pair_graph.edges(etype='interaction')[1], :]

        center_word = pos_pair_graph.edges(etype='t2t')[0]
        pos_word = pos_pair_graph.edges(etype='t2t')[1]
        neg_word = neg_pair_graph.edges(etype='t2t')[1]

        tag_idx = pos_pair_graph.ndata[dgl.NID]['tag']
        emb_u = tag_embed[tag_idx[center_word]]
        emb_v = self.v_embeddings(tag_idx[pos_word])
        neg_emb_v = self.v_embeddings(tag_idx[neg_word])

        return self.create_bpr_loss(u_e, pos_e, neg_e) + self.K_word2cf * self.create_word_loss(emb_u, emb_v, neg_emb_v)

    def generate(self, input_nodes, blocks, pos_pair_graph):
        user_gcn_emb, item_gcn_emb = self.gcn(blocks, input_nodes, self.all_embed, False)

        u_e = pos_pair_graph.ndata[dgl.NID]['user']
        pos_e = pos_pair_graph.ndata[dgl.NID]['review']

        self.user_embed_final[u_e] = user_gcn_emb
        self.item_embed_final[pos_e] = item_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2

        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss

    def create_word_loss(self, emb_u, emb_v, neg_emb_v):
        batch_size = emb_u.shape[0]

        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)

        neg_score = torch.mul(emb_u, neg_emb_v).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)

        loss_word = -1 * (torch.mean(score) + torch.mean(neg_score))

        regularizer = (torch.norm(emb_u) ** 2
                       + torch.norm(emb_v) ** 2
                       + torch.norm(neg_emb_v) ** 2) / 2

        emb_loss = self.decay * regularizer / batch_size
        return loss_word + emb_loss


class M2GNN_word_amazon(nn.Module):
    def __init__(self, data_config, args_config):
        super(M2GNN_word_amazon, self).__init__()

        self.n_users = data_config['n_users']
        self.n_reviews = data_config['n_items']
        self.n_items = data_config['n_items4rs']
        self.n_tags = data_config['n_tag']
        self.K_word2cf = data_config['k_word2cf']

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.dim = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.iteration = args_config.iteration
        self.max_K = args_config.max_K
        self.max_len = args_config.max_len

        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        gain = 1.414

        self.all_embed = initializer(torch.empty(int(self.n_users + self.n_reviews + self.n_tags), self.dim), gain=gain)

        # self.v_embeddings = nn.Embedding(self.n_tags, self.dim, sparse=True)
        self.v_embeddings = nn.Embedding(self.n_tags, self.dim)
        self.v_embeddings.weight.data.uniform_(-0, 0)

        self.user_embed_final = torch.zeros(size=(self.n_users, self.dim))
        self.item_embed_final = torch.zeros(size=(self.n_items, self.dim))

    def _init_model(self):
        return M2GNN_c_amazon(dim=self.dim,
                              n_hops=self.context_hops,
                              n_users=self.n_users,
                              n_reviews=self.n_reviews,
                              n_items=self.n_items,
                              n_tags=self.n_tags,
                              iteration=self.iteration,
                              max_K=self.max_K,
                              max_len=self.max_len,
                              ratedrop=self.mess_dropout_rate)

    def forward(self, input_nodes, blocks, pos_pair_graph, neg_pair_graph):
        user_gcn_emb, item_gcn_emb = self.gcn(blocks, input_nodes, self.all_embed, True)

        tag_embed = self.all_embed[self.n_users + self.n_reviews:, :]

        # u_e = neg_pair_graph.ndata[dgl.NID]['user'][pos_pair_graph.edges(etype='interaction')[0]]
        # pos_e = neg_pair_graph.ndata[dgl.NID]['review'][pos_pair_graph.edges(etype='interaction')[1]]
        # neg_e = neg_pair_graph.ndata[dgl.NID]['review'][neg_pair_graph.edges(etype='interaction')[1]]

        u_e = user_gcn_emb[pos_pair_graph.edges(etype='interaction')[0]]
        pos_e = item_gcn_emb[pos_pair_graph.edges(etype='interaction')[1], :]
        neg_e = item_gcn_emb[neg_pair_graph.edges(etype='interaction')[1], :]

        center_word = pos_pair_graph.edges(etype='t2t')[0]
        pos_word = pos_pair_graph.edges(etype='t2t')[1]
        neg_word = neg_pair_graph.edges(etype='t2t')[1]

        tag_idx = pos_pair_graph.ndata[dgl.NID]['tag']
        emb_u = tag_embed[tag_idx[center_word]]
        emb_v = self.v_embeddings(tag_idx[pos_word])
        neg_emb_v = self.v_embeddings(tag_idx[neg_word])

        return self.create_bpr_loss(u_e, pos_e, neg_e) + self.K_word2cf * self.create_word_loss(emb_u, emb_v, neg_emb_v)

    def generate(self, input_nodes, blocks, pos_pair_graph):
        user_gcn_emb, item_gcn_emb = self.gcn(blocks, input_nodes, self.all_embed, False)

        u_e = pos_pair_graph.ndata[dgl.NID]['user']
        pos_e = pos_pair_graph.ndata[dgl.NID]['review']

        self.user_embed_final[u_e] = user_gcn_emb
        self.item_embed_final[pos_e] = item_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2

        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss

    def create_word_loss(self, emb_u, emb_v, neg_emb_v):
        batch_size = emb_u.shape[0]

        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)

        neg_score = torch.mul(emb_u, neg_emb_v).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)

        loss_word = -1 * (torch.mean(score) + torch.mean(neg_score))

        regularizer = (torch.norm(emb_u) ** 2
                       + torch.norm(emb_v) ** 2
                       + torch.norm(neg_emb_v) ** 2) / 2

        emb_loss = self.decay * regularizer / batch_size
        return loss_word + emb_loss


class M2GNN_c(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_reviews, n_items, n_tags, iteration, max_K, max_len, gamma, ratedrop):
        super(M2GNN_c, self).__init__()

        self.dim = dim
        self.n_hops = n_hops
        self.n_users = n_users
        self.n_reviews = n_reviews
        self.n_items = n_items
        self.n_tags = n_tags
        self.ratedrop = ratedrop
        self.gamma = gamma

        self.iteration = iteration
        self.max_K = max_K
        self.max_len = max_len
        self.input_units = dim
        self.input_units1 = int(dim * 4)
        self.hidden = self.input_units1 // 2
        self.output_units = dim
        # step-1 multi-interest extraction
        self.B_matrix = nn.init.normal_(torch.empty(1, max_K, max_len), mean=0, std=1)
        self.B_matrix.requires_grad = False
        self.S_matrix = nn.init.normal_(torch.empty(self.input_units, self.output_units), mean=0, std=1)
        self.S_matrix = nn.Parameter(self.S_matrix)
        # step-2 important interest extraction
        self.M1_matrix = nn.init.normal_(torch.empty(self.input_units, self.output_units), mean=0, std=1)
        self.M1_matrix = nn.Parameter(self.M1_matrix)

        self.M2_matrix = nn.init.normal_(torch.empty(self.output_units, 1), mean=0, std=1)
        self.M2_matrix = nn.Parameter(self.M2_matrix)

        self.MLP = nn.Sequential(
            nn.Linear(self.input_units1, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )

        self.convs = nn.ModuleList()

        # self.tri_embed_u = nn.ModuleList()
        #
        # self.tri_embed = KGA00(n_users=n_users, n_entities=n_entities, n_items=n_items,
        #                        n_relations=n_relations, dim=dim, dim_flag1=self.dim_flag1)

        for j in range(n_hops):
            self.convs.append(
                M2GNN_one(n_users=n_users, n_reviews=n_reviews, n_items=n_items, n_tags=n_tags, n_hops=n_hops, dim=dim,
                          iteration=self.iteration, max_K=self.max_K, max_len=self.max_len, gamma=self.gamma))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

    def forward(self, blocks, input_nodes, all_embed, dropout):
        # all_embed = torch.nn.functional.normalize(all_embed, dim=1)

        num_review = blocks[-1].num_dst_nodes('review')
        num_user = blocks[-1].num_dst_nodes('user')

        user = input_nodes['user']
        tag = input_nodes['tag']
        review = input_nodes['review']

        user_embed = all_embed[:self.n_users, :][user, :]
        review_embed = all_embed[self.n_users:self.n_users + self.n_reviews, :][review, :]
        tag_embed = all_embed[self.n_users + self.n_reviews:, :][tag, :]

        if dropout:
            user_embed = self.dropout(user_embed)
            review_embed = self.dropout(review_embed)
            tag_embed = self.dropout(tag_embed)

        # if dropout:
        #     random_indices = np.random.choice(graph_UIS.edges()[0].shape[0],
        #                                       size=int(graph_UIS.edges()[0].shape[0] * self.ratedrop),
        #                                       replace=False)
        #     graph_UIS = dgl.edge_subgraph(graph_UIS, random_indices, preserve_nodes=True)

        """cal edge embedding"""
        user_embed_res = user_embed[:num_user, :]
        review_embed_res = review_embed[:num_review, :]

        for j in range(self.n_hops):
            user_embed, review_embed, tag_embed = self.convs[j](j, blocks[j], user_embed, review_embed, tag_embed,
                                                                self.B_matrix, self.S_matrix, self.M1_matrix,
                                                                self.M2_matrix, self.MLP)
            # user_embed = torch.nn.functional.normalize(user_embed, dim=1)
            # review_embed = torch.nn.functional.normalize(review_embed, dim=1)
            # tag_embed = torch.nn.functional.normalize(tag_embed, dim=1)

            if dropout:
                user_embed_res = self.dropout(user_embed_res)
                review_embed_res = self.dropout(review_embed_res)
                tag_embed = self.dropout(tag_embed)
            user_embed_res = torch.add(user_embed_res, user_embed[:num_user, :])
            review_embed_res = torch.add(review_embed_res, review_embed[:num_review, :])

        # user_embed_res = torch.nn.functional.normalize(user_embed_res, dim=1)
        # review_embed_res = torch.nn.functional.normalize(review_embed_res, dim=1)

        return user_embed_res, review_embed_res

    def cal_deg(self, edges):
        atr = torch.exp(torch.sum(torch.mul(self.W_R[edges.data['type']], self.atr_att), dim=1))
        return {'atr': atr}


class M2GNN_c_amazon(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_reviews, n_items, n_tags, iteration, max_K, max_len, gamma, ratedrop):
        super(M2GNN_c_amazon, self).__init__()

        self.dim = dim
        self.n_hops = n_hops
        self.n_users = n_users
        self.n_reviews = n_reviews
        self.n_items = n_items
        self.n_tags = n_tags
        self.ratedrop = ratedrop
        self.gamma = gamma

        self.iteration = iteration
        self.max_K = max_K
        self.max_len = max_len
        self.input_units = dim
        self.input_units1 = int(dim * 4)
        self.hidden = self.input_units1 // 2
        self.output_units = dim
        # step-1 multi-interest extraction
        self.B_matrix = nn.init.normal_(torch.empty(1, max_K, max_len), mean=0, std=1)
        self.B_matrix.requires_grad = False
        self.S_matrix = nn.init.normal_(torch.empty(self.input_units, self.output_units), mean=0, std=1)
        self.S_matrix = nn.Parameter(self.S_matrix)
        # step-2 important interest extraction
        self.M1_matrix = nn.init.normal_(torch.empty(self.input_units, self.output_units), mean=0, std=1)
        self.M1_matrix = nn.Parameter(self.M1_matrix)

        self.M2_matrix = nn.init.normal_(torch.empty(self.output_units, 1), mean=0, std=1)
        self.M2_matrix = nn.Parameter(self.M2_matrix)

        self.MLP = nn.Sequential(
            nn.Linear(self.input_units1, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )

        self.convs = nn.ModuleList()

        # self.tri_embed_u = nn.ModuleList()
        #
        # self.tri_embed = KGA00(n_users=n_users, n_entities=n_entities, n_items=n_items,
        #                        n_relations=n_relations, dim=dim, dim_flag1=self.dim_flag1)

        for j in range(n_hops):
            self.convs.append(
                M2GNN_one_amazon(n_users=n_users, n_reviews=n_reviews, n_items=n_items, n_tags=n_tags, n_hops=n_hops,
                                 dim=dim,
                                 iteration=self.iteration, max_K=self.max_K, max_len=self.max_len, gamma=self.gamma))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

    def forward(self, blocks, input_nodes, all_embed, dropout):
        # all_embed = torch.nn.functional.normalize(all_embed, dim=1)

        num_review = blocks[-1].num_dst_nodes('review')
        num_user = blocks[-1].num_dst_nodes('user')

        user = input_nodes['user']
        tag = input_nodes['tag']
        review = input_nodes['review']

        user_embed = all_embed[:self.n_users, :][user, :]
        review_embed = all_embed[self.n_users:self.n_users + self.n_reviews, :][review, :]
        tag_embed = all_embed[self.n_users + self.n_reviews:, :][tag, :]

        if dropout:
            user_embed = self.dropout(user_embed)
            review_embed = self.dropout(review_embed)
            tag_embed = self.dropout(tag_embed)

        # if dropout:
        #     random_indices = np.random.choice(graph_UIS.edges()[0].shape[0],
        #                                       size=int(graph_UIS.edges()[0].shape[0] * self.ratedrop),
        #                                       replace=False)
        #     graph_UIS = dgl.edge_subgraph(graph_UIS, random_indices, preserve_nodes=True)

        """cal edge embedding"""
        user_embed_res = user_embed[:num_user, :]
        review_embed_res = review_embed[:num_review, :]

        for j in range(self.n_hops):
            user_embed, review_embed, tag_embed = self.convs[j](j, blocks[j], user_embed, review_embed, tag_embed,
                                                                self.B_matrix, self.S_matrix, self.M1_matrix,
                                                                self.M2_matrix, self.MLP)
            # user_embed = torch.nn.functional.normalize(user_embed, dim=1)
            # review_embed = torch.nn.functional.normalize(review_embed, dim=1)
            # tag_embed = torch.nn.functional.normalize(tag_embed, dim=1)

            if dropout:
                user_embed_res = self.dropout(user_embed_res)
                review_embed_res = self.dropout(review_embed_res)
                tag_embed = self.dropout(tag_embed)
            user_embed_res = torch.add(user_embed_res, user_embed[:num_user, :])
            review_embed_res = torch.add(review_embed_res, review_embed[:num_review, :])

        # user_embed_res = torch.nn.functional.normalize(user_embed_res, dim=1)
        # review_embed_res = torch.nn.functional.normalize(review_embed_res, dim=1)

        return user_embed_res, review_embed_res

    def cal_deg(self, edges):
        atr = torch.exp(torch.sum(torch.mul(self.W_R[edges.data['type']], self.atr_att), dim=1))
        return {'atr': atr}


class M2GNN_one(nn.Module):

    def __init__(self, n_users, n_reviews, n_items, n_tags, n_hops, dim, iteration, max_K, max_len, gamma):
        super(M2GNN_one, self).__init__()
        self.n_users = n_users
        self.n_reviews = n_reviews
        self.n_items = n_items
        self.n_tags = n_tags
        self.n_hops = n_hops
        self.dim = dim
        self.iteration = iteration
        self.max_K = max_K
        self.max_len = max_len
        self.gamma = gamma

    def cal_attribute1(self, edges):
        edge_emb = edges.src['node'] * self.W_r

        return {'emb': edge_emb}

    def cal_attribute2(self, edges):
        att = edges.data['att'] / edges.dst['nodeatt']
        return {'att1': att}

    def e_mul_e(self, edges):
        att = edges.data['att1'].unsqueeze(1) * edges.data['emb']
        return {'nei': att}

    def sequence_mask(self, engths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask.type(dtype)
        return mask

    def squash(self, inputs):
        vec_squared_norm = torch.sum(torch.square(inputs), dim=1, keepdim=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + 1e-9)
        vec_squashed = scalar_factor * inputs  # element-wise
        return vec_squashed

    def multi_interest(self, nodes):
        if self.flag == 0:
            B_matrix_c = self.B_matrix
            low_capsule = nodes.mailbox['send_{}'.format(self.r)]

            B, num_nei, embed_size = low_capsule.size()

            mask = torch.zeros(size=(B, self.max_K, self.max_len))
            mask[:, :, :num_nei] = 1
            mask = mask.bool()
            mask = mask.to(low_capsule.device)

            low_capsule = torch.nn.functional.pad(low_capsule, [0, 0, 0, self.max_len - num_nei, 0, 0])

            # print(low_capsule.size())

            for i in range(self.iteration):
                ## mask: B * max_K * max_len
                ## W: B * max_K * max_len
                ## low_capsule_new: B * max_len * hidden_units
                ## high_capsule: B * max_K * hidden_units

                pad = torch.ones_like(mask, dtype=torch.float32) * (-2 ** 16 + 1)
                pad = pad.to(low_capsule.device)
                B_tile = B_matrix_c.repeat(B, 1, 1)
                # print(B_tile.size())
                B_mask = torch.where(mask, B_tile, pad)
                # print(B_mask.size())
                W = nn.functional.softmax(B_mask, dim=-1)
                # print(W.size())
                # W = W[:, :, :num_nei]
                # low_capsule_new = torch.einsum('ijk,lo->ilk', (low_capsule, self.S_matrix))
                low_capsule_new = torch.einsum('ijk,ko->ijo', (low_capsule, self.S_matrix))
                # print(low_capsule_new.size())
                high_capsule_tmp = torch.matmul(W, low_capsule_new)
                high_capsule = self.squash(high_capsule_tmp)
                B_delta = torch.sum(
                    torch.matmul(high_capsule, torch.transpose(low_capsule_new, dim0=1, dim1=2)),
                    dim=0, keepdim=True)
                B_matrix_c += B_delta

            # re = {'node_{}'.format(self.r): torch.mean(high_capsule, dim=1)}
            re = {'node_{}'.format(self.r): high_capsule}
        elif self.flag == 1:
            re = {'node_{}'.format(self.r): torch.mean(nodes.mailbox['send_{}'.format(self.r)], dim=1)}
        elif self.flag == 2:
            re = {'node_{}'.format(self.r): torch.mean(nodes.mailbox['send_{}'.format(self.r)], dim=1)}
        return re

    def multi_interest0(self, nodes):
        if self.flag == 0:
            # B_matrix_c = self.B_matrix
            low_capsule = nodes.mailbox['send_{}'.format(self.r)]

            B, num_nei, embed_size = low_capsule.size()

            # max(1, min(K, log2(|history|)))
            K_c = max(1, min(self.max_K, int(math.log2(num_nei))))
            mask = torch.zeros(size=(B, K_c, self.max_len))

            if K_c == 1:
                B_matrix_c = self.B_matrix[:, 0, :].unsqueeze(1)
            else:
                B_matrix_c = self.B_matrix[:, :K_c, :]

            mask[:, :, :num_nei] = 1
            mask = mask.bool()
            mask = mask.to(low_capsule.device)

            low_capsule = torch.nn.functional.pad(low_capsule, [0, 0, 0, self.max_len - num_nei, 0, 0])

            # print(low_capsule.size())

            for i in range(self.iteration):
                ## mask: B * max_K * max_len
                ## W: B * max_K * max_len
                ## low_capsule_new: B * max_len * hidden_units
                ## high_capsule: B * max_K * hidden_units

                pad = torch.ones_like(mask, dtype=torch.float32) * (-2 ** 16 + 1)
                pad = pad.to(low_capsule.device)
                B_tile = B_matrix_c.repeat(B, 1, 1)
                # print(B_tile.size())
                B_mask = torch.where(mask, B_tile, pad)
                # print(B_mask.size())
                W = nn.functional.softmax(B_mask, dim=-1)
                # print(W.size())
                # W = W[:, :, :num_nei]
                # low_capsule_new = torch.einsum('ijk,lo->ilk', (low_capsule, self.S_matrix))
                low_capsule_new = torch.einsum('ijk,ko->ijo', (low_capsule, self.S_matrix))
                # print(low_capsule_new.size())
                high_capsule_tmp = torch.matmul(W, low_capsule_new)
                high_capsule = self.squash(high_capsule_tmp)
                B_delta = torch.sum(
                    torch.matmul(high_capsule, torch.transpose(low_capsule_new, dim0=1, dim1=2)),
                    dim=0, keepdim=True)
                B_matrix_c += B_delta

            # re = {'node_{}'.format(self.r): torch.mean(high_capsule, dim=1)}
            if 2 > 1:
                high_capsule = torch.nn.functional.pad(high_capsule, [0, 0, 0, self.max_K - K_c, 0, 0])
            re = {'node_{}'.format(self.r): high_capsule}
        elif self.flag == 1:
            re = {'node_{}'.format(self.r): torch.mean(nodes.mailbox['send_{}'.format(self.r)], dim=1)}
        elif self.flag == 2:
            re = {'node_{}'.format(self.r): torch.mean(nodes.mailbox['send_{}'.format(self.r)], dim=1)}

            # low_capsule = nodes.mailbox['send_{}'.format(self.r)]
            # re = {
            #     'node_{}'.format(self.r): torch.mean(torch.einsum('ijk,ko->ijo', (low_capsule, self.S_matrix)), dim=1)}
        return re

    def forward(self, idx_layer, graph, user_embed_res, review_embed_res, tag_embed_res, B_matrix, S_matrix, M1, M2,
                MLP):
        self.B_matrix = B_matrix.to(user_embed_res.device)
        self.S_matrix = S_matrix
        # print(graph)
        # graph = dgl.block_to_graph(graph)
        graph = graph.local_var()
        graph.nodes['user'].data['node'] = user_embed_res
        graph.nodes['review'].data['node'] = review_embed_res
        graph.nodes['tag'].data['node'] = tag_embed_res

        idx_list = ['u_r_t', 'u_q_t', 'u_p_r_t', 'r_h_t', 't2t']
        # update_edges = {}
        # for i in idx_list:
        #     update_edges[i] = (dgl.function.copy_src('node', 'send'), dgl.function.mean('send', 'node_0'))
        # graph.multi_update_all(update_edges, 'sum')
        for idx, i in enumerate(idx_list):
            if graph.num_edges(i) > 0:
                self.r = i

                if idx == 0 or idx == 1 or idx == 2:
                    self.flag = 0
                elif idx == 3:
                    self.flag = 2
                elif idx == 4:
                    self.flag = 1
                graph[i].update_all(
                    dgl.function.copy_u('node', 'send_{}'.format(i)),
                    self.multi_interest0, etype=i)
                # graph[i].update_all(
                #     dgl.function.copy_u('node', 'send_{}'.format(i)),
                #     dgl.function.mean('send_{}'.format(i), 'node_{}'.format(self.r)), etype=i)
        # gamma = 8
        if idx_layer < self.n_hops - 1:
            # user_dst = graph.srcdata['node']['user'][:graph.num_dst_nodes('user'), :]  # [B,num_K,dim]
            user_embed_res_urt = graph.dstdata['node_u_r_t']['user']  # [B,num_K,dim]
            user_embed_res_uqt = graph.dstdata['node_u_q_t']['user']
            user_embed_res_uprt = graph.dstdata['node_u_p_r_t']['user']

            # # self-attention pooling
            user_embed_res_all = torch.cat([user_embed_res_urt, user_embed_res_uqt, user_embed_res_uprt], dim=1)
            user_embed_res_all0 = F.tanh(torch.matmul(user_embed_res_all, M1))
            user_embed_res_all1 = torch.matmul(user_embed_res_all0, M2).squeeze(2)
            user_embed_res_all1 = torch.pow(user_embed_res_all1, self.gamma)
            att = nn.functional.softmax(user_embed_res_all1, dim=1).unsqueeze(1)
            user_embed_res = torch.matmul(att, user_embed_res_all).squeeze(1)

            review_embed_res = graph.dstdata['node_r_h_t']['review']

            tag_embed_res = graph.dstdata['node_t2t']['tag']
        else:
            # user_dst = graph.srcdata['node']['user'][:graph.num_dst_nodes('user'), :]
            user_embed_res_urt = graph.dstdata['node_u_r_t']['user']
            user_embed_res_uqt = graph.dstdata['node_u_q_t']['user']
            user_embed_res_uprt = graph.dstdata['node_u_p_r_t']['user']

            # # self-attention pooling
            user_embed_res_all = torch.cat([user_embed_res_urt, user_embed_res_uqt, user_embed_res_uprt], dim=1)
            user_embed_res_all0 = F.tanh(torch.matmul(user_embed_res_all, M1))
            user_embed_res_all1 = torch.matmul(user_embed_res_all0, M2).squeeze(2)
            user_embed_res_all1 = torch.pow(user_embed_res_all1, self.gamma)
            att = nn.functional.softmax(user_embed_res_all1, dim=1).unsqueeze(1)
            user_embed_res = torch.matmul(att, user_embed_res_all).squeeze(1)

            review_embed_res = graph.dstdata['node_r_h_t']['review']

        return user_embed_res, review_embed_res, tag_embed_res


class M2GNN_one_amazon(nn.Module):

    def __init__(self, n_users, n_reviews, n_items, n_tags, n_hops, dim, iteration, max_K, max_len):
        super(M2GNN_one_amazon, self).__init__()
        self.n_users = n_users
        self.n_reviews = n_reviews
        self.n_items = n_items
        self.n_tags = n_tags
        self.n_hops = n_hops
        self.dim = dim
        self.iteration = iteration
        self.max_K = max_K
        self.max_len = max_len

    def cal_attribute1(self, edges):
        edge_emb = edges.src['node'] * self.W_r

        return {'emb': edge_emb}

    def cal_attribute2(self, edges):
        att = edges.data['att'] / edges.dst['nodeatt']
        return {'att1': att}

    def e_mul_e(self, edges):
        att = edges.data['att1'].unsqueeze(1) * edges.data['emb']
        return {'nei': att}

    def sequence_mask(self, engths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask.type(dtype)
        return mask

    def squash(self, inputs):
        vec_squared_norm = torch.sum(torch.square(inputs), dim=1, keepdim=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + 1e-9)
        vec_squashed = scalar_factor * inputs  # element-wise
        return vec_squashed

    def multi_interest(self, nodes):
        if self.flag == 0:
            B_matrix_c = self.B_matrix
            low_capsule = nodes.mailbox['send_{}'.format(self.r)]

            B, num_nei, embed_size = low_capsule.size()

            mask = torch.zeros(size=(B, self.max_K, self.max_len))
            mask[:, :, :num_nei] = 1
            mask = mask.bool()
            mask = mask.to(low_capsule.device)

            low_capsule = torch.nn.functional.pad(low_capsule, [0, 0, 0, self.max_len - num_nei, 0, 0])

            # print(low_capsule.size())

            for i in range(self.iteration):
                ## mask: B * max_K * max_len
                ## W: B * max_K * max_len
                ## low_capsule_new: B * max_len * hidden_units
                ## high_capsule: B * max_K * hidden_units

                pad = torch.ones_like(mask, dtype=torch.float32) * (-2 ** 16 + 1)
                pad = pad.to(low_capsule.device)
                B_tile = B_matrix_c.repeat(B, 1, 1)
                # print(B_tile.size())
                B_mask = torch.where(mask, B_tile, pad)
                # print(B_mask.size())
                W = nn.functional.softmax(B_mask, dim=-1)
                # print(W.size())
                # W = W[:, :, :num_nei]
                # low_capsule_new = torch.einsum('ijk,lo->ilk', (low_capsule, self.S_matrix))
                low_capsule_new = torch.einsum('ijk,ko->ijo', (low_capsule, self.S_matrix))
                # print(low_capsule_new.size())
                high_capsule_tmp = torch.matmul(W, low_capsule_new)
                high_capsule = self.squash(high_capsule_tmp)
                B_delta = torch.sum(
                    torch.matmul(high_capsule, torch.transpose(low_capsule_new, dim0=1, dim1=2)),
                    dim=0, keepdim=True)
                B_matrix_c += B_delta

            # re = {'node_{}'.format(self.r): torch.mean(high_capsule, dim=1)}
            re = {'node_{}'.format(self.r): high_capsule}
        elif self.flag == 1:
            re = {'node_{}'.format(self.r): torch.mean(nodes.mailbox['send_{}'.format(self.r)], dim=1)}
        elif self.flag == 2:
            re = {'node_{}'.format(self.r): torch.mean(nodes.mailbox['send_{}'.format(self.r)], dim=1)}

            # low_capsule = nodes.mailbox['send_{}'.format(self.r)]
            # re = {
            #     'node_{}'.format(self.r): torch.mean(torch.einsum('ijk,ko->ijo', (low_capsule, self.S_matrix)), dim=1)}
        return re

    def multi_interest0(self, nodes):
        if self.flag == 0:
            # B_matrix_c = self.B_matrix
            low_capsule = nodes.mailbox['send_{}'.format(self.r)]

            B, num_nei, embed_size = low_capsule.size()

            # max(1, min(K, log2(|history|)))
            K_c = max(1, min(self.max_K, int(math.log2(num_nei))))
            mask = torch.zeros(size=(B, K_c, self.max_len))

            if K_c == 1:
                B_matrix_c = self.B_matrix[:, 0, :].unsqueeze(1)
            else:
                B_matrix_c = self.B_matrix[:, :K_c, :]

            mask[:, :, :num_nei] = 1
            mask = mask.bool()
            mask = mask.to(low_capsule.device)

            low_capsule = torch.nn.functional.pad(low_capsule, [0, 0, 0, self.max_len - num_nei, 0, 0])

            # print(low_capsule.size())

            for i in range(self.iteration):
                ## mask: B * max_K * max_len
                ## W: B * max_K * max_len
                ## low_capsule_new: B * max_len * hidden_units
                ## high_capsule: B * max_K * hidden_units

                pad = torch.ones_like(mask, dtype=torch.float32) * (-2 ** 16 + 1)
                pad = pad.to(low_capsule.device)
                B_tile = B_matrix_c.repeat(B, 1, 1)
                # print(B_tile.size())
                B_mask = torch.where(mask, B_tile, pad)
                # print(B_mask.size())
                W = nn.functional.softmax(B_mask, dim=-1)
                # print(W.size())
                # W = W[:, :, :num_nei]
                # low_capsule_new = torch.einsum('ijk,lo->ilk', (low_capsule, self.S_matrix))
                low_capsule_new = torch.einsum('ijk,ko->ijo', (low_capsule, self.S_matrix))
                # print(low_capsule_new.size())
                high_capsule_tmp = torch.matmul(W, low_capsule_new)
                high_capsule = self.squash(high_capsule_tmp)
                B_delta = torch.sum(
                    torch.matmul(high_capsule, torch.transpose(low_capsule_new, dim0=1, dim1=2)),
                    dim=0, keepdim=True)
                B_matrix_c += B_delta

            # re = {'node_{}'.format(self.r): torch.mean(high_capsule, dim=1)}
            if 2 > 1:
                high_capsule = torch.nn.functional.pad(high_capsule, [0, 0, 0, self.max_K - K_c, 0, 0])
            re = {'node_{}'.format(self.r): high_capsule}
        elif self.flag == 1:
            re = {'node_{}'.format(self.r): torch.mean(nodes.mailbox['send_{}'.format(self.r)], dim=1)}
        elif self.flag == 2:
            re = {'node_{}'.format(self.r): torch.mean(nodes.mailbox['send_{}'.format(self.r)], dim=1)}

            # low_capsule = nodes.mailbox['send_{}'.format(self.r)]
            # re = {
            #     'node_{}'.format(self.r): torch.mean(torch.einsum('ijk,ko->ijo', (low_capsule, self.S_matrix)), dim=1)}
        return re

    def forward(self, idx_layer, graph, user_embed_res, review_embed_res, tag_embed_res, B_matrix, S_matrix, M1, M2,
                MLP):
        self.B_matrix = B_matrix.to(user_embed_res.device)
        self.S_matrix = S_matrix
        # print(graph)
        # graph = dgl.block_to_graph(graph)
        graph = graph.local_var()
        graph.nodes['user'].data['node'] = user_embed_res
        graph.nodes['review'].data['node'] = review_embed_res
        graph.nodes['tag'].data['node'] = tag_embed_res

        idx_list = ['u_i_t_t', 'u_i_s_t', 'r_h_t', 't2t']
        # update_edges = {}
        # for i in idx_list:
        #     update_edges[i] = (dgl.function.copy_src('node', 'send'), dgl.function.mean('send', 'node_0'))
        # graph.multi_update_all(update_edges, 'sum')
        for idx, i in enumerate(idx_list):
            if graph.num_edges(i) > 0:
                self.r = i

                if idx == 0 or idx == 1:
                    self.flag = 0
                elif idx == 2:
                    self.flag = 2
                elif idx == 3:
                    self.flag = 1
                graph[i].update_all(
                    dgl.function.copy_u('node', 'send_{}'.format(i)),
                    self.multi_interest0, etype=i)
                # graph[i].update_all(
                #     dgl.function.copy_u('node', 'send_{}'.format(i)),
                #     dgl.function.mean('send_{}'.format(i), 'node_{}'.format(self.r)), etype=i)
        gamma = 1
        if idx_layer < self.n_hops - 1:
            # user_dst = graph.srcdata['node']['user'][:graph.num_dst_nodes('user'), :]  # [B,num_K,dim]
            user_embed_res_urt = graph.dstdata['node_u_i_t_t']['user']  # [B,num_K,dim]
            user_embed_res_uqt = graph.dstdata['node_u_i_s_t']['user']

            # # self-attention pooling
            user_embed_res_all = torch.cat([user_embed_res_urt, user_embed_res_uqt], dim=1)
            user_embed_res_all0 = F.tanh(torch.matmul(user_embed_res_all, M1))
            user_embed_res_all1 = torch.matmul(user_embed_res_all0, M2).squeeze(2)
            user_embed_res_all1 = torch.pow(user_embed_res_all1, gamma)
            att = nn.functional.softmax(user_embed_res_all1, dim=1).unsqueeze(1)
            user_embed_res = torch.matmul(att, user_embed_res_all).squeeze(1)

            review_embed_res = graph.dstdata['node_r_h_t']['review']
            # review_embed_res = torch.mean(graph.dstdata['node_r_h_t']['review'], dim=1)

            tag_embed_res = graph.dstdata['node_t2t']['tag']
        else:
            # user_dst = graph.srcdata['node']['user'][:graph.num_dst_nodes('user'), :]  # [B,num_K,dim]
            user_embed_res_urt = graph.dstdata['node_u_i_t_t']['user']  # [B,num_K,dim]
            user_embed_res_uqt = graph.dstdata['node_u_i_s_t']['user']

            # # self-attention pooling
            user_embed_res_all = torch.cat([user_embed_res_urt, user_embed_res_uqt], dim=1)
            user_embed_res_all0 = F.tanh(torch.matmul(user_embed_res_all, M1))
            user_embed_res_all1 = torch.matmul(user_embed_res_all0, M2).squeeze(2)
            user_embed_res_all1 = torch.pow(user_embed_res_all1, gamma)
            att = nn.functional.softmax(user_embed_res_all1, dim=1).unsqueeze(1)
            user_embed_res = torch.matmul(att, user_embed_res_all).squeeze(1)

            review_embed_res = graph.dstdata['node_r_h_t']['review']
            # review_embed_res = torch.mean(graph.dstdata['node_r_h_t']['review'], dim=1)

        return user_embed_res, review_embed_res, tag_embed_res
