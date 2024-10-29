import numpy as np
import contextlib
import torch
# from udagcn.dual_gnn.dataset.DomainData import DomainData
from semigcl.utils import load_data
from semigcl.model import SemiGCL
from semigcl.utils import batch_generator

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        
        
def get_source_dataset(model, source, source_ratio=1.0, use_source_score=0, score_path=None, seed=200):
    '''
    return: 
        source.source_mask
    '''
    
    adj_s, adj_val_s, diff_idx_s, diff_val_s, feature_s, label_s, idx_train_s, _, _, idx_tot_s = load_data(dataset=f"{source}.mat", 
                            device="cpu", 
                            seed=seed,
                            alpha_ppr=0.1,
                            diff_k=20
                        )
    source_data = {}
    source_data["adj_s"] = adj_s
    source_data["adj_val_s"] = adj_val_s
    source_data["diff_idx_s"] = diff_idx_s
    source_data["diff_val_s"] = diff_val_s
    source_data["feature_s"] = feature_s
    source_data["label_s"] = label_s
    source_data["idx_train_s"] = idx_train_s

    idx_tot_s_tmp = idx_tot_s # 为了在train 上evaluate
    if not use_source_score:
        source_train_num = int(len(idx_tot_s) * source_ratio)
        idx_tot_s = idx_tot_s[:source_train_num] 
        source_data["idx_tot_s"] = idx_tot_s
        return source_data
    
    else:
        source_data["idx_tot_s"] = idx_tot_s
        source_data = get_source_idx(model, source, source_data, 
                                                    score_path, 
                                                    source_ratio)
    return source_data
    

def get_structure_score(query, protoemb):
    distances = torch.cdist(query.unsqueeze(1), protoemb.unsqueeze(0)).squeeze(1) # (N, C)
    # 每行取最小值作为每个查询样本的得分
    scores = torch.min(distances, dim=1).values
    return scores
    
        
def get_target_dataset(model, target, target_shots=None, seed=200):
    '''
    return: 
        target.train_mask: 指代用于train的数据
        target.test_mask: 指代用于test的数据
    '''

    adj_t, adj_val_t, diff_idx_t, diff_val_t, feature_t, label_t, idx_train_t, idx_val_t, idx_test_t, idx_tot_t = load_data(dataset=f"{target}.mat", 
                            device="cpu", 
                            shot=target_shots,
                            seed=seed,
                            alpha_ppr=0.1,
                            diff_k=20
                        )
    num_class = label_t.shape[1]
    target_data = {}
    target_data["adj_t"] = adj_t
    target_data["adj_val_t"] = adj_val_t
    target_data["diff_idx_t"] = diff_idx_t
    target_data["diff_val_t"] = diff_val_t
    target_data["feature_t"] = feature_t
    target_data["label_t"] = label_t
    target_data["idx_train_t"] = idx_train_t
    target_data["idx_val_t"] = idx_val_t
    target_data["idx_test_t"] = idx_test_t
    target_data["idx_tot_t"] = idx_tot_t
        
    assert (len(idx_train_t)== num_class * target_shots)
    return target_data
        
    
    
def load_model(num_features, num_class, args):
    model = SemiGCL(num_features, num_class, args)
    return model
    
    
def create_optim(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer
      
        
def get_source_idx(model, dataname, data, score_path, select_ratio=1.0):

    score = torch.load(score_path)  # score: [num_data, num_class * few_shots]
            
    labels = data['label_s'].argmax(1)
    _, origin = torch.unique(data['label_s'].argmax(1), return_counts=True)
    selected_indices = []

    score = score.reshape(-1, 5, 5).sum(2).cpu()  #(8935, 5) ,将每个datapoint对应不同class的few-shots分别求和作为class score
 
    score_desend = np.argsort(-score.max(1)[0], kind='stable') # descending
    cnt_per_class = [0, 0, 0, 0, 0]
    bound_per_class = [int(c * select_ratio) for c in origin]
    # import ipdb; ipdb.set_trace()
    for idx in score_desend:
        if cnt_per_class[labels[idx]] < bound_per_class[labels[idx]]:
            selected_indices.append(idx)
            cnt_per_class[labels[idx]] += 1
        else:
            continue
    
    # print("origin select: ", f'{len(selected_indices)}/{score.shape[0]}={len(selected_indices)/score.shape[0]}')
    selected_indices = list(set(selected_indices))
    # print("After delete duplicate: ", f'{len(selected_indices)}/{score.shape[0]}={len(selected_indices)/score.shape[0]}')

    data['idx_train_s'] =  data["idx_tot_s"]
    data["idx_tot_s"] = torch.LongTensor(selected_indices)
    
    labels = data['label_s'][selected_indices].argmax(1)
    
    # unique_values, counts = torch.unique(labels, return_counts=True)
    # for ori, cnt in zip(origin, counts):
    #     print("class ratio", cnt/ori)

    return data
       
 
def calculate_gradient_penalty(critic, x_src, x_tgt):
    x = torch.cat([x_src, x_tgt], dim=0).requires_grad_(True)
    x_out = critic(x)
    grad_out = torch.ones(x_out.shape, requires_grad=False).to(x_out.device)

    # Get gradient w.r.t. x
    grad = torch.autograd.grad(outputs=x_out,
                               inputs=x,
                               grad_outputs=grad_out,
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True,)[0]
    grad = grad.view(grad.shape[0], -1)
    grad_penalty = torch.mean((grad.norm(2, dim=1) - 1) ** 2)
    return grad_penalty

   
def normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

@contextlib.contextmanager
def disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)
    
    
###### 

def prune(dist, beta=0.1):
    # Calculate minimum and maximum scores along the last two dimensions
    # min_score = torch.min(dist, dim=[1, 2], keepdim=True).values
    # max_score = torch.max(dist, dim=[1, 2], keepdim=True).values
    min_score = torch.min(torch.min(dist, dim=2, keepdim=True).values, dim=1, keepdim=True).values
    max_score = torch.max(torch.max(dist, dim=2, keepdim=True).values, dim=1, keepdim=True).values
    # Calculate threshold
    threshold = min_score + beta * (max_score - min_score)
    
    # Subtract the threshold and apply ReLU to remove negative values
    res = dist - threshold
    return torch.relu(res)

def FGW_distance(Cs, Ct, C, beta=0.5, iteration=5, OT_iteration=20):
    # Get T and Cst using GW_alg in PyTorch
    
    T, Cst = GW_alg(Cs, Ct, beta=beta, iteration=iteration, OT_iteration=OT_iteration)
    # Calculate the Gromov-Wasserstein distance
    GW_distance = torch.einsum('bij,bij->b', Cst, T)  # Trace of Cst @ T
    
    # Calculate the Wasserstein distance
    W_distance = torch.einsum('bij,bij->b', C, T)  # Trace of C @ T
    
    return GW_distance, W_distance

def IPOT_alg(C, beta=1, t_steps=10, k_steps=1):
    b, n, m = C.shape
    sigma = torch.ones(b, m, 1, device=C.device) / m  # [b, m, 1]
    T = torch.ones(b, n, m, device=C.device)
    A = torch.exp(-C / beta)  # [b, n, m]
    
    for t in range(t_steps):
        Q = A * T  # [b, n, m]
        for k in range(k_steps):
            delta = 1 / (n * torch.matmul(Q, sigma))  # [b, n, 1]
            sigma = 1 / (m * torch.matmul(Q.transpose(1, 2), delta))  # [b, m, 1]
        
        T = delta * Q * sigma.transpose(1, 2)  # [b, n, m]

    return T

def GW_alg(Cs, Ct, beta=0.5, iteration=5, OT_iteration=20):
    bs, _, n = Cs.shape
    _, _, m = Ct.shape

    one_m = torch.ones(bs, m, 1, device=Ct.device) / m
    one_n = torch.ones(bs, n, 1, device=Cs.device) / n
    p = torch.ones(bs, m, 1, device=Ct.device) / m
    q = torch.ones(bs, n, 1, device=Cs.device) / n
    
    # Calculate Cst using matrix multiplication
    Cst = torch.matmul(torch.matmul(Cs**2, q), one_m.transpose(1, 2)) + \
          torch.matmul(one_n, torch.matmul(p.transpose(1,2), (Ct**2).transpose(1, 2)))
    
    # Initialize gamma as the outer product of q and p
    gamma = torch.matmul(q, p.transpose(1, 2))

    for i in range(iteration):
        tmp1 = torch.matmul(Cs, gamma)
        C_gamma = Cst - 2 * torch.matmul(tmp1, Ct.transpose(1, 2))

        # Apply IPOT algorithm for transport plan update
        gamma = IPOT_alg(C_gamma, beta=beta, t_steps=OT_iteration)

    Cgamma = Cst - 2 * torch.matmul(torch.matmul(Cs, gamma), Ct.transpose(1, 2))
    
    return gamma, Cgamma

def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    # device = torch.device("cuda" if adj.is_cuda else "cpu")
    # device = torch.device(device if adj.is_cuda else "cpu")
    device = adj.device
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx

def get_subgraph(adj_list, target_node):
    N, max_neighbors = adj_list.shape
    one_hop_neighbors = set(adj_list[target_node].tolist()) - {target_node}
    two_hop_neighbors = set()
    for node in one_hop_neighbors:
        neighbors = set(adj_list[node].tolist()) - {node} - one_hop_neighbors - {target_node}
        two_hop_neighbors.update(neighbors)
    subgraph_nodes = one_hop_neighbors | two_hop_neighbors | {target_node}
    # print(subgraph_nodes)
    n = len(subgraph_nodes)
    node2idx = {node: i for i, node in enumerate(subgraph_nodes)}
    idx2node = {i: node for i, node in enumerate(subgraph_nodes)}
    adj_matrix = torch.zeros((n, n), dtype=torch.float32)
    for node in subgraph_nodes:
        cur_idx = node2idx[node]  # 当前节点的索引
        for neighbor in adj_list[node]:
            neighbor = neighbor.item()
            if neighbor in node2idx:
                ngh_idx = node2idx[neighbor]
                adj_matrix[cur_idx, ngh_idx] = 1  # 添加邻接关系
                adj_matrix[ngh_idx, cur_idx] = 1  # 对称化
    return adj_matrix, node2idx, idx2node

def get_adjlist(adj_matrix, idx2node, max_degree=32):
    m = len(adj_matrix)
    update_adjlst = dict()
    for idx in range(m):
        neibs = torch.nonzero(adj_matrix[idx]).squeeze(1).cpu().numpy()
        if len(neibs) == 0:
            neibs = np.array(idx).repeat(max_degree)
        elif len(neibs) < max_degree:
            neibs = np.random.choice(neibs, max_degree, replace=True)
        else:
            neibs = np.random.choice(neibs, max_degree, replace=False)
        update_adjlst[idx2node[idx]] = list(idx2node[i] for i in neibs)
    return update_adjlst

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

