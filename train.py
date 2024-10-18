import wandb
from sklearn import metrics
import torch
import torch.nn as nn
import copy
from sklearn.metrics import classification_report
import json
from semigcl.utils import batch_generator, adentropy
import numpy as np
import torch.nn.functional as F
from semigcl.utils import eval_iterate
from utils import calculate_gradient_penalty, normalize, disable_tracking_bn_stats
from tqdm import tqdm
from torch.nn.functional import normalize
from sklearn.neighbors import KDTree

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

def save_checkpoint(model, optimizer, epoch, checkpoint_path, loss):
    torch.save({
        'epoch': epoch,
        'lr': optimizer.param_groups[0]['lr'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{checkpoint_path}/{epoch}.pth')
    with open(f'{checkpoint_path}/checkpoint.log', 'a') as f:
        log = {
            'epoch': epoch, 
            'lr': float(optimizer.param_groups[0]['lr']),
            'loss': float(loss),
        }
        json.dump(log, f)
        f.write('\n')
    
def predict(name, model, data, cache_name, mask=None, return_emb=False, info=""):
    if name == "SemiGCL":
        if cache_name == "source":
            adj, adj_val, feature, label, diff_idx, diff_val, idx = (
            data["adj_s"], data["adj_val_s"],data["feature_s"],
            data["label_s"], data["diff_idx_s"], data["diff_val_s"],
            data["idx_tot_s"]
        )
            if len(idx) == 0:
                idx = torch.arange(feature.shape[0])
        else:
            adj, adj_val, feature, label, diff_idx, diff_val, idx = (
            data["adj_t"], data["adj_val_t"],data["feature_t"],
            data["label_t"], data["diff_idx_t"], data["diff_val_t"],
            data["idx_test_t"]
        )
        preds, targets = [], []
        # import ipdb; ipdb.set_trace()
        embs = []
        for b_nodes in eval_iterate(idx, 256):
            emb, cly_loss, preds_per_batch, targets_per_batch = model(adj, adj_val, feature, label, diff_idx,
                                                    diff_val, b_nodes, cal_f1=True)
            preds.append(preds_per_batch)
            targets.append(targets_per_batch)
            embs.append(emb)
        
        preds_whole = np.vstack(preds)
        targets_whole = np.vstack(targets)
        #####
        if return_emb and cache_name == "target":
            # import ipdb; ipdb.set_trace()
            embs_whole = torch.cat(embs, dim=0).detach().cpu().numpy()
            y = targets_whole.argmax(1)
            np.save(f'analysis/embs_whole_{info}.npy', embs_whole)
            np.save(f'analysis/y_{info}.npy', y)
        #####
        # acc, micro_f1, macro_f1 = evaluate(preds_whole, targets_whole)
        if return_emb:
            return preds_whole, targets_whole, embs_whole
        else:
            return preds_whole, targets_whole, None

def evaluate(preds, labels):
    if type(preds) == np.ndarray:
        micro_f1 = metrics.f1_score(labels, preds, average='micro' )
        macro_f1 = metrics.f1_score(labels, preds, average='macro')
        target_indices = np.argmax(labels, axis=1)
        pred_indices = np.argmax(preds, axis=1)
        accuracy = np.mean(target_indices == pred_indices)
    else:
        corrects = preds.eq(labels)
        accuracy = corrects.float().mean()
        micro_f1 = metrics.f1_score(labels.cpu(), preds.cpu(), average='micro')
        macro_f1 = metrics.f1_score(labels.cpu(), preds.cpu(), average='macro')
    # import ipdb; ipdb.set_trace()
    return accuracy, micro_f1, macro_f1

def test(name, model, data, cache_name, mask=None, cls_report=False, return_emb=False, info=""):
    # for model in models:
    #     model.eval()
    model.eval()
    if name == "SemiGCL":
        preds, labels, embs = predict(name, model, data, cache_name, mask, return_emb=return_emb, info=info)
    else:
        preds, labels = predict(name, model, data, cache_name, mask)
    if cls_report:
        if type(preds) == np.ndarray:
            report = classification_report(labels, preds, output_dict=True)
        else:
            report = classification_report(labels.cpu(), preds.cpu(), output_dict=True)
        return report
    else:
        accuracy, micro_f1, macro_f1 = evaluate(preds, labels)
        return accuracy, micro_f1, macro_f1

def mme_loss(model, target_graph, lamda=0.5):
    out = model(target_graph, reverse=True)[target_graph.test_mask]
    out = F.softmax(out, dim=1)
    return lamda * torch.mean(torch.sum(out * (torch.log(out + 1e-10)), dim=1))
    

def get_ssl_logits(model, z, zn):
    g = torch.sigmoid(torch.mean(z, dim=0, keepdim=True))
    pos_scores, neg_scores = model.disc(g, z, zn)
    lbl_1 = torch.ones(pos_scores.size(0)).to(z.device)
    lbl_2 = torch.zeros(neg_scores.size(0)).to(z.device)
    loss = F.binary_cross_entropy_with_logits(pos_scores, lbl_1) + F.binary_cross_entropy_with_logits(neg_scores, lbl_2)
    return loss

def get_number_of_params(model):
    
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params

def prepare_optimizer_state(model, optim_state, model_name):
    
    names = [i for i, (n, p) in enumerate(model.named_parameters()) if p.requires_grad and n.split('.')[0] in [
                    'lin']]
    avg = torch.cat([optim_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optim_state[n]["exp_avg_sq"].view(-1)
                       for n in names])
    return avg, avg_sq
    
def get_gradient(model, data, mask, optimizer):
    model.train()
    
    grads = []
    loss_func = nn.CrossEntropyLoss()
    
    optim_state = optimizer.state_dict()['state']
    avg, avg_sq = prepare_optimizer_state(model, optim_state, "GCN")
    
    
    cly_model = model.lin
   
    num_of_parameters = get_number_of_params(cly_model)
    print(num_of_parameters)
    select_idx = torch.nonzero(data[mask]).squeeze().tolist()
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08
     # (8935, 5)
    logits = model(data)
    # import ipdb; ipdb.set_trace()
    for idx in tqdm(select_idx):
        model.zero_grad()
        
        loss = loss_func(logits[idx], data.y[idx])
        loss.backward(retain_graph=True)
        vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in cly_model.named_parameters() if p.grad is not None])
        
        num_nan = torch.isnan(vectorized_grads).sum()
        assert num_nan <= 0
        grads.append(vectorized_grads)
    # import ipdb; ipdb.set_trace()
    grads = torch.stack(grads, dim=0)  
    grads = normalize(grads, dim=1)
    return grads

feat_source_warmup = []
labels_source_warmup = []

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # import ipdb; ipdb.set_trace()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        
    return res


def train_epoch(model, optimizer, source_data, target_data, epoch, epochs, args):
    model.train()

    if args.model == "SemiGCL":
        adj_s, adj_val_s, diff_idx_s, diff_val_s, feature_s, label_s, idx_train_s, idx_tot_s = (
            source_data["adj_s"], source_data["adj_val_s"], source_data["diff_idx_s"],
            source_data["diff_val_s"], source_data["feature_s"], source_data["label_s"],
            source_data["idx_train_s"], source_data["idx_tot_s"]
        )


        adj_t, adj_val_t, diff_idx_t, diff_val_t, feature_t, label_t, idx_train_t, idx_val_t, idx_test_t, idx_tot_t = (
            target_data["adj_t"], target_data["adj_val_t"], target_data["diff_idx_t"], 
            target_data["diff_val_t"], target_data["feature_t"], target_data["label_t"], 
            target_data["idx_train_t"], target_data["idx_val_t"], target_data["idx_test_t"], 
            target_data["idx_tot_t"]
        )
        src_cs = None
        if args.soft:
            
            # 1. 对每个target node 计算不确定性，不稳定性
            # args.xi, args.eps, args.ip
            h = []
            # import ipdb; ipdb.set_trace()
            with torch.no_grad():
                model.eval()
                for b_nodes in eval_iterate(idx_tot_t, 256):
                    h_per_batch, _,  = model(adj_t, adj_val_t, feature_t, label_t, diff_idx_t,
                                                            diff_val_t, b_nodes)
                    h.append(h_per_batch)
                    # targets.append(targets_per_batch)
            ### uncertainty 
            h = torch.cat(h) #(N_t, D)
            h = h.detach()
            h_tgt = copy.deepcopy(h).detach().cpu().numpy()
            tree = KDTree(h_tgt)
            preds = model.cly_model(h) # logit, p
            preds = F.softmax(preds, dim=1)
            preds = preds.detach()
            # uncertainty
            uncertain_tgt = -1 * torch.sum(preds * torch.log(preds + 1e-10), dim=1).detach() # (N_t, )
            
            
            # prepare random unit tensor
            delta_h = torch.rand(h.shape).sub(0.5).to(h.device)
            delta_h  = normalize(delta_h)
            
            with disable_tracking_bn_stats(model):
                for i in range(args.ip):
                    delta_h.requires_grad_()
                    preds_hat = model.cly_model(h + args.xi * delta_h)
                    logp_hat = F.log_softmax(preds_hat, dim=1)
                    kl_div = F.kl_div(logp_hat, preds, reduction='batchmean')
                    kl_div.backward()
                    delta_h = normalize(delta_h.grad)
                    model.zero_grad()
                    
                #calulate uncertainty
                adv_h = delta_h * args.eps
                preds_hat = model.cly_model(h + adv_h)
                logp_hat = F.log_softmax(preds_hat, dim=1)
                unstable_tgt = F.kl_div(logp_hat, preds, reduction='none').sum(1).detach() # (N_t, 1)
            unstable_tgt = torch.zeros(h_tgt.shape[0]).to(h.device)
            # 2. 对每个source node
            # 验证  uncertainty_tgt, unstable_tgt 取值在 [0， 1]范围内
            # sorted_inx_s = torch.sort(idx_tot_s).values.cpu().numpy()
            # src_idx_map = dict()
            # for i in range(len(sorted_inx_s)):
            #     src_idx_map[sorted_inx_s[i]] = i
            # with torch.no_grad():
            #     h = []
            #     for b_nodes in eval_iterate(torch.sort(idx_tot_s).values, 256):
            #         h_per_batch, _,  = model(adj_s, adj_val_s, feature_s, label_s, diff_idx_s,
            #                                                 diff_val_s, b_nodes)
            #         h.append(h_per_batch)
            
            # h_src = torch.cat(h).detach().cpu().numpy()
                
        src_feat = torch.zeros(adj_s.shape[0], 128).to(args.device)
        
        num_batch = int(max(feature_s.shape[0]/(args.batch_size/2), idx_test_t.shape[0]/(args.batch_size/2)))
        # import ipdb; ipdb.set_trace()
        s_batches = batch_generator(idx_tot_s, int(args.batch_size/2))
        s_batches2 = batch_generator(idx_train_s, int(args.batch_size/2))
        t_batches = batch_generator(idx_test_t, int(args.batch_size/2))
        model.train()
        p = float(epoch) / args.epochs
        grl_lambda = min(2. / (1. + np.exp(-10. * p)) - 1, 0.1) 
        for iter in range(num_batch):
            b_nodes_s = next(s_batches) 
            b_nodes_2 = next(s_batches2)
            b_nodes_t = next(t_batches)
            if len(b_nodes_s) == 0: ##### TODO, for source ratio
                cly_loss_s = 0.0
            else:
                if args.soft:
                    # score_lst = []
                    # for i in b_nodes_s:
                    #     score_lst.append(src_cs[src_idx_map[i.item()]])
                    # score = torch.stack(score_lst)
                    
                    ###
                    source_features, _ = model(adj_s, adj_val_s, feature_s, label_s, diff_idx_s,
                                                    diff_val_s, idx=b_nodes_s)
                    
                    src_feat[b_nodes_s] = source_features
                    dist, ind = tree.query(source_features.detach().cpu().numpy(), k=args.K) # (N_s, K), (N_s, K)
                    ind = torch.from_numpy(ind) # （N_s, K)
                    # dist = torch.from_numpy(dist).to(b_nodes_s.device) # （N_s, K)
                    # import ipdb; ipdb.set_trace()
                    ### 
                    src_one_hop_neighbors = adj_s[b_nodes_s]
                    src_subgraph = torch.cat((b_nodes_s.unsqueeze(1), src_one_hop_neighbors), dim=1)
                    src_subgraph_feat = src_feat[src_subgraph] # (N_s, neighbors + 1, hiden)
                    src_subgraph_feat = src_subgraph_feat.unsqueeze(1).repeat(1, args.K, 1, 1) # (N_s, K, neighbors + 1, hidden)
                    
                    
                    import ipdb; ipdb.set_trace()
                    tgt_one_hop_neighbors = adj_t[ind] # (N_s, K, neighbors)
                    tgt_subgraph = torch.cat((ind.unsqueeze(-1).to(args.device), tgt_one_hop_neighbors), dim=-1) # (N_s, K, neighbors + 1)
                    tgt_subgraph_feat = h[tgt_subgraph] # (N_s, K, neighbors + 1, hidden)
                    
                    src_subgraph_feat = tgt_subgraph_feat.unsqueeze(1).repeat(1, args.K, 1, 1) # (N_s, K, neighbors + 1, hidden)
                    
                    cosine_cost = 1 - torch.einsum(
                        'aijk,aijk->aij', src_subgraph_feat, tgt_subgraph_feat) # (N_s, K, neighbors)
                    
                    # C_s = 1 - torch.einsum('aijk,aijk->')
                    
                    # alpha1 = torch.sigmoid(model.alpha1)
                    # alpha2 = torch.sigmoid(model.alpha2)
                    # alpha3 = torch.sigmoid(model.alpha3)
                    # phi1 = 1  / (1 + torch.exp((dist - alpha1))) # (N_s, K)
                    # phi2 = 1 / (1 + torch.exp(torch.sigmoid(-uncertain_tgt[ind]) +  alpha2)) # (N_s, K)
                    # phi3 = 1 / (1 + torch.exp(torch.sigmoid(- unstable_tgt[ind]) +  alpha3)) # (N_t, K)
                    phi1 = 1  / (1 + torch.exp(-(dist - 0.4))) # (N_s, K)
                    phi2 = 1 / (1 + torch.exp(torch.sigmoid(uncertain_tgt[ind]) -  0.6)) # (N_s, K)
                    phi3 = 1 / (1 + torch.exp(torch.sigmoid(unstable_tgt[ind]) -  0.6)) # (N_t, K)
                    
                    phi = torch.max(
                        torch.tensor(0.0),  
                        phi1 + torch.min(torch.tensor(1.0), phi2 + phi3) - 2.0 + 1.0 
                    )
                    # score  = 0.5 * (1 + (1 + 1/(1 + alpha1) + alpha2 + alpha3) * torch.mean(phi, dim=1) ) 
                    score  = 0.5 * (1 + torch.mean(phi, dim=1))
                    # print(phi1[0][:3].detach().cpu().numpy(), phi2[0][:3].detach().cpu().numpy(), phi3[0][:3].detach().cpu().numpy(), phi[0][:3].detach().cpu().numpy(), score[:3].detach().cpu().numpy())
                    # print(alpha1.item(), alpha2.item(), alpha3.item(), score[:3].detach().cpu().numpy())
                else:
                    score = None
                source_features, cly_loss_s = model(adj_s, adj_val_s, feature_s, label_s, diff_idx_s,
                                                    diff_val_s, idx=b_nodes_s, src_cs=score) # (bs, output_dim), scalar, task_loss

            ### TODO, b_nodes_t 似乎应该从idx_tot_t
            target_features, _ = model(adj_t, adj_val_t, feature_t, label_t, diff_idx_t,
                                                    diff_val_t, idx=b_nodes_t)
            
            if idx_train_t.shape[0] == 0 or args.warmup: # few-hot target domain task loss
                cly_loss_t = 0.0
            else:
                feats_train_t, cly_loss_t = model(adj_t, adj_val_t, feature_t, label_t, diff_idx_t,
                                                    diff_val_t, idx=idx_train_t)
            total_cly_loss = cly_loss_s + cly_loss_t
            device = args.device
            ssl_loss = torch.zeros(1).to(device)
            ssl_loss_s = torch.zeros(1).to(device)
            ssl_loss_t = torch.zeros(1).to(device)
            domain_loss = torch.zeros(1).to(device)
            if args.cal_ssl:
                model.ssl_model.train()
                shuf_idx_s = np.arange(label_s.shape[0])
                np.random.shuffle(shuf_idx_s)
                shuf_feat_s = feature_s[shuf_idx_s, :]
                shuf_idx_t = np.arange(label_t.shape[0])
                np.random.shuffle(shuf_idx_t)
                shuf_feat_t = feature_t[shuf_idx_t, :]

                if len(b_nodes_2) == 0: ##### TODO, for source ratio
                    ssl_loss_s = 0.0 
                else:
                    h_s_1 = model.emb_model(b_nodes_2, adj_s, adj_val_s, feature_s)
                    h_s_2 = model.emb_model(b_nodes_2, diff_idx_s, diff_val_s, feature_s)
                    h_s_3 = model.emb_model(b_nodes_2, adj_s, adj_val_s, shuf_feat_s)
                    h_s_4 = model.emb_model(b_nodes_2, diff_idx_s, diff_val_s, shuf_feat_s)
                    logits_s = model.ssl_model(h_s_1, h_s_2, h_s_3, h_s_4)
                    labels_ssl_s = torch.cat([torch.ones(h_s_1.shape[0] * 2), torch.zeros(h_s_1.shape[0] * 2)]).unsqueeze(0).to(device)
                    ssl_loss_s = F.binary_cross_entropy_with_logits(logits_s, labels_ssl_s)
                
                if args.warmup:
                    ssl_loss = args.ssl_param * ssl_loss_s
                    domain_loss = 0.0
                else: 
                    b_nodes_t_plus = torch.cat((b_nodes_t, idx_train_t), dim=0)
                    h_t_1 = model.emb_model(b_nodes_t_plus, adj_t, adj_val_t, feature_t)
                    h_t_2 = model.emb_model(b_nodes_t_plus, diff_idx_t, diff_val_t, feature_t)
                    h_t_3 = model.emb_model(b_nodes_t_plus, adj_t, adj_val_t, shuf_feat_t)
                    h_t_4 = model.emb_model(b_nodes_t_plus, diff_idx_t, diff_val_t, shuf_feat_t)
                    logits_t = model.ssl_model(h_t_1, h_t_2, h_t_3, h_t_4)
                    labels_ssl_t = torch.cat([torch.ones(h_t_1.shape[0] * 2), torch.zeros(h_t_1.shape[0] * 2)]).unsqueeze(0).to(device)
                    ssl_loss_t = F.binary_cross_entropy_with_logits(logits_t, labels_ssl_t)
                    
                    ssl_loss = args.ssl_param * (ssl_loss_s + ssl_loss_t)
                    
                    domain_loss = args.mme_param * adentropy(model.cly_model, target_features, grl_lambda)
                    
     
            loss = total_cly_loss + ssl_loss + domain_loss
            
            
            if args.wandb:
                wandb.log({
                    "train/cly_loss": total_cly_loss,
                    "train/cly_loss_s": cly_loss_s,
                    "train/cly_loss_t": cly_loss_t,
                    "train/ssl_loss": ssl_loss,
                    "train/ssl_loos_s": ssl_loss_s,
                    "train/ssl_loos_t": ssl_loss_s,
                    "train/domain_loss": domain_loss
                    })

            optimizer.zero_grad()
            # print(epoch)
            # import ipdb; ipdb.set_trace()
            loss.backward()
            optimizer.step()

            if args.save_checkpoint and epoch % args.save_step == 0:
                save_checkpoint(model, optimizer, epoch, args.checkpoint_path, loss)

def train(model, optimizer, source_data, target_data, epochs, args, logger):
    model = model.to(args.device)
    if args.model == "SemiGCL":
        for key, val in source_data.items():
            source_data[key] = val.to(args.device)
        for key, val in target_data.items():
            target_data[key] = val.to(args.device)
    else:
        source_data = source_data.to(args.device)
        target_data = target_data.to(args.device)
    
    if args.wandb:
        wandb.init(project="GraphGDA", name=args.wandb_info)

    model.train()
  
    best_source_acc = 0.0
    best_target_micro_f1 = 0.0
    best_target_macro_f1 = 0.0 
    best_target_acc = 0.0
    best_epoch = 0.0
    best_model = model

    for epoch in range(1, epochs + 1):
        # import ipdb; ipdb.set_trace()
        # if args.model == "UDAGCN":
        train_epoch(model, optimizer, source_data, target_data, epoch, epochs, args)
        # import ipdb; ipdb.set_trace()
        s_accuracy, s_micro_f1, s_macro_f1 = test(args.model, model, source_data, "source")
        if args.model == "SemiGCL":
            # if epoch % 10 == 0:
            #     return_emb = True
            # else:
            return_emb = False
            t_accuracy, t_micro_f1, t_macro_f1 = test(args.model, model, target_data, "target", return_emb=return_emb, info=f"epoch_{epoch}")
        
        logger.info("epoch {:03d} | source acc {:.4f} | source micro-F1 {:.4f} | source macro-F1 {:.4f}".
            format(epoch, s_accuracy, s_micro_f1, s_macro_f1))
        logger.info("\t| target acc {:.4f} | target micro-F1 {:.4f} | target macro-F1 {:.4f}".
            format(t_accuracy, t_micro_f1, t_macro_f1))
        if t_accuracy > best_target_acc:
            best_target_acc = t_accuracy
            best_source_acc = s_accuracy
            best_target_macro_f1 = t_macro_f1
            best_target_micro_f1 = t_micro_f1
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            
        if args.wandb:
            wandb.log({
                "Source/Acc": s_accuracy,
                "Source/Micro-F1": s_micro_f1,
                "Source/Macro-F1": s_macro_f1
            })
            wandb.log({
                "Target/Acc": t_accuracy,
                "Target/Micro-F1": t_micro_f1,
                "Target/Macro-F1": t_macro_f1
            })
            
            wandb.log({
                "Best/Acc": best_target_acc,
                "Best/Micro-F1": best_target_micro_f1,
                "Best/Macro-F1": best_target_macro_f1
            })

            
            
    logger.info("=============================================================")
    line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}, best_target_micro_f1: {}, best_target_macro_f1: {}"\
        .format(id, best_epoch, best_source_acc, best_target_acc, best_target_micro_f1, best_target_macro_f1)
    logger.info(line)
    
    if args.model == "SemiGCL":
        cls_report = test(args.model, best_model, target_data, "target", cls_report=True)
    
    for label, metrics in cls_report.items():
        logger.info(f"Label: {label}, Metrics: {metrics}")
        
    