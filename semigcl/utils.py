import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
from sklearn import metrics
from .models import *
from scipy.sparse import csc_matrix
from scipy.linalg import inv
from .layers import aggregator_lookup
import torch
import torch.nn.functional as F
import os
import pickle

    
def top_k_preds(y_true, y_pred):
    # import ipdb; ipdb.set_trace()
    top_k_list = np.array(np.sum(y_true, 1), np.int32)
    predictions = []
    for i in range(y_true.shape[0]):
        pred_i = np.zeros(y_true.shape[1])
        pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
        predictions.append(np.reshape(pred_i, (1, -1)))
    predictions = np.concatenate(predictions, axis=0)
    top_k_array = np.array(predictions, np.int64)

    return top_k_array


def cal_f1_score(y_true, y_pred):
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

    return micro_f1, macro_f1


def batch_generator(nodes, batch_size, shuffle=True):
    num = nodes.shape[0]
    chunk = num // batch_size
    while True:
        if chunk * batch_size + batch_size > num:
            chunk = 0
            if shuffle:
                idx = np.random.permutation(num)
            else:
                idx = np.arange(num)
        b_nodes = nodes[idx[chunk*batch_size:(chunk+1)*batch_size]]
        chunk += 1

        yield b_nodes

def eval_iterate(nodes, batch_size, shuffle=False):
    idx = np.arange(nodes.shape[0])
    if shuffle:
        idx = np.random.permutation(idx)
    n_chunk = idx.shape[0] // batch_size + 1
    for chunk_id, chunk in enumerate(np.array_split(idx, n_chunk)):
        b_nodes = nodes[chunk]

        yield b_nodes


def do_iter(emb_model, cly_model, adj, adj_val, feature, labels, diff_idx, diff_val, idx, cal_f1=False):
    embs_1 = emb_model(idx, adj, adj_val, feature)
    embs_2 = emb_model(idx, diff_idx, diff_val, feature)
    embs = torch.cat([embs_1, embs_2], dim=1)
    preds = cly_model(embs)
    labels_idx = torch.argmax(labels[idx], dim=1)
    cly_loss = F.cross_entropy(preds, labels_idx)
    if not cal_f1:
        return embs, cly_loss
    else:
        targets = labels[idx].cpu().numpy()
        preds = top_k_preds(targets, preds.detach().cpu().numpy())
        return embs, cly_loss, preds, targets


def evaluate(emb_model, cly_model, adj, adj_val, feature, labels, diff_idx, diff_val, idx, batch_size, mode='val'):
    assert mode in ['val', 'test']
    embs, preds, targets = [], [], []
    cly_loss = 0
    for b_nodes in eval_iterate(idx, batch_size):
        embs_per_batch, cly_loss_per_batch, preds_per_batch, targets_per_batch = do_iter(emb_model, cly_model, adj, adj_val, feature, labels,
                                                                                         diff_idx, diff_val, b_nodes, cal_f1=True)
        # import ipdb; ipdb.set_trace()
        embs.append(embs_per_batch.detach().cpu().numpy())
        preds.append(preds_per_batch)
        targets.append(targets_per_batch)
        cly_loss += cly_loss_per_batch.item()

    cly_loss /= len(preds)
    embs_whole = np.vstack(embs)
    targets_whole = np.vstack(targets)
    # import ipdb; ipdb.set_trace()
    micro_f1, macro_f1 = cal_f1_score(targets_whole, np.vstack(preds))
    target_indices = np.argmax(targets_whole, axis=1)
    pred_indices = np.argmax(np.vstack(preds), axis=1)
    acc = np.mean(target_indices == pred_indices)
    return cly_loss, micro_f1, macro_f1, acc, embs_whole, targets_whole


def get_split(labels, shot, target_flag, seed):
    idx_tot = np.arange(labels.shape[0])
    num_class = labels.shape[1]
    np.random.seed(seed)
    np.random.shuffle(idx_tot)
    if target_flag: # target domain 
        # import ipdb; ipdb.set_trace()
        labels = np.argmax(labels, axis=1)
        idx_train = []
        for class_idx in range(num_class):
            node_idx = np.where(labels == class_idx)[0]
            node_idx = np.random.choice(node_idx, shot, replace=False)
            idx_train.append(node_idx)
        idx_train = np.concatenate(idx_train, axis=0) # (25,) , num_of_class * shot
        np.random.shuffle(idx_train)
        idx_val = np.array([])
        idx_test = np.array(list(set(list(idx_tot)) - set(list(idx_train))))
        np.random.shuffle(idx_test)
    else: # soure domain
        partition = [0.7, 0.1, 0.2]
        num_train, num_val = int(labels.shape[0] * partition[0]), int(labels.shape[0] * partition[1])
        idx_train, idx_val, idx_test = idx_tot[:num_train], idx_tot[num_train:num_train+num_val], \
                                       idx_tot[num_train+num_val:]

    return idx_train, idx_val, idx_test, idx_tot


def make_adjacency(G, max_degree, seed):
    # import ipdb; ipdb.set_trace()
    all_nodes = np.sort(np.array(G.nodes()))
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes, max_degree)) + (n_nodes - 1)).astype(int)
    neibs_num = np.array([])
    np.random.seed(seed)
    for node in all_nodes:
        neibs = np.array([i for i in G.neighbors(node)])
        neibs_num = np.append(neibs_num, len(neibs))
        if len(neibs) == 0:
            neibs = np.array(node).repeat(max_degree)
        elif len(neibs) < max_degree:
            neibs = np.random.choice(neibs, max_degree, replace=True)
        else:
            neibs = np.random.choice(neibs, max_degree, replace=False)
        adj[node, :] = neibs

    return adj


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def pre_social_net(adj, features, labels):
    features = csc_matrix(features.astype(np.uint8))
    labels = labels.astype(np.int32)

    return adj, features, labels


def load_data(file_path="./Datasets", dataset='acmv9.mat', device='cpu', shot=None, target_flag=False, seed=123,
              is_blog=False, alpha_ppr=0.2, diff_k=128):
    # load raw data
    
    # data_path = file_path + '/' + dataset
    data_path = './dataset/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network'] # 邻接矩阵，（5463， 5463）
    features = data_mat['attrb'] # 特征矩阵，（5463， 6775）
    labels = data_mat['group'] # lable 矩阵，（5463， 5）
    # import ipdb; ipdb.set_trace()
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features) # 行归一化邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # 对称化, 保留adj中更大的部分
    # import ipdb; ipdb.set_trace()
 
    
    name = dataset.split('.')[0]
    ppr_file = f'ppr_results/{name}_alpha_{alpha_ppr}_diffk_{diff_k}.pkl'
    if os.path.exists(ppr_file):
        with open(ppr_file, 'rb') as f:
            diff_val, diff_idx = pickle.load(f)
    else:
        diff_val, diff_idx = compute_ppr(adj, alpha=alpha_ppr, diff_k=diff_k) # (5463, 20), (5463, 20), 20=diff_k
        with open(ppr_file, 'wb') as f:
            pickle.dump((diff_val, diff_idx), f)
 
    adj_dense = np.array(adj.todense()) 
    edges = np.vstack(np.where(adj_dense)).T # (number of edges, 2)
    Graph = nx.from_edgelist(edges)
    adj = make_adjacency(Graph, 32, seed) # 对每个node进行最大 32 度数的采样或选择, (5463, 32)
    adj_val = np.ones(shape=adj.shape, dtype=diff_val.dtype) # (5463, 32), 全1
    if shot is not None:
        target_flag = True
    idx_train, idx_val, idx_test, idx_tot = get_split(labels, shot, target_flag, seed)
    features = torch.FloatTensor(features)
   
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    adj_val = torch.from_numpy(adj_val).float()
    diff_idx = torch.from_numpy(diff_idx)
    diff_val = torch.from_numpy(diff_val).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), adj_val.to(device), diff_idx.to(device), diff_val.to(device), features.to(device), labels.to(device), \
           idx_train.to(device), idx_val.to(device), idx_test.to(device), idx_tot.to(device)


def compute_ppr(adj, alpha=0.2, diff_k=128, self_loop=True):
    a = adj.todense().A
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.sum(a, axis=1)
    dinv_sqrt = np.power(d, -0.5)
    dinv_sqrt[np.isinf(dinv_sqrt)] = 0.
    dinv = np.diag(dinv_sqrt)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    diff = alpha * inv(np.eye(a.shape[0]) - (1 - alpha) * at)  # （2), P
    diff_val, diff_idx = get_top_k_matrix_row_norm(diff, k=diff_k)

    return diff_val, diff_idx


def get_top_k_matrix_row_norm(A, k=128):
    A = A.T
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    diff_idx = A.argsort(axis=0)[num_nodes - k:]
    diff_val = A[A.argsort(axis=0)[num_nodes - k:], row_idx]
    norm = diff_val.sum(axis=0)
    norm[norm <= 0] = 1
    diff_val = diff_val/norm

    return diff_val.T, diff_idx.T

def adentropy(F1, feat, lamda=1.0):
    out_t1 = F1(feat, reverse=True, lamda=lamda)
    out_t1 = F.softmax(out_t1, dim=-1)
    loss_adent = torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))

    return loss_adent