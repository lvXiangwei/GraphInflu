import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from torch.autograd import Function
from .models import GraphSAGE, Predictor
from .layers import aggregator_lookup
from .mvgrl import MVGRL
from .utils import top_k_preds

class SemiGCL(nn.Module):
    def __init__(self, num_features, num_class, args):
        super(SemiGCL, self).__init__()
        n_samples = args.n_samples.split(',')
        output_dims = args.output_dims.split(',')
        self.emb_model = GraphSAGE(**{
            "aggregator_class": aggregator_lookup[args.aggregator_class], # diffusion aggregator
            "input_dim": num_features,
            "layer_specs": [
                {
                    "n_sample": int(n_samples[0]),
                    "output_dim": int(output_dims[0]),
                    "activation": F.relu,
                },
                {
                    "n_sample": int(n_samples[-1]),
                    "output_dim": int(output_dims[-1]),
                    "activation": F.relu,
                }
            ],
            # "device": device
        })
        self.cly_model = Predictor(num_class=num_class, inc=2*int(output_dims[-1]), temp=args.T)
        # self.avg_group_centroid = None
        
        ### alpha1, alpha2, alpha3
        self.alpha1 = nn.Parameter(torch.tensor(0.0))  
        self.alpha2 = nn.Parameter(torch.tensor(0.0))  
        self.alpha3 = nn.Parameter(torch.tensor(0.0))
        
        if args.cal_ssl:
            self.ssl_model = MVGRL(int(output_dims[-1]))

    def get_features(self, adj, adj_val, diff_idx, diff_val, feature, idx):
        embs1 = self.emb_model(idx, adj, adj_val, feature)
        embs2 = self.emb_model(idx, diff_idx, diff_val, feature)
        embs = torch.cat([embs1, embs2], dim=1)
        return embs
    
    
    def forward(self, adj, adj_val, feature, labels, diff_idx, diff_val, idx, cal_f1=False, src_cs = None):
        # import ipdb; ipdb.set_trace()
        embs_1 = self.emb_model(idx, adj, adj_val, feature)
        embs_2 = self.emb_model(idx, diff_idx, diff_val, feature)
        embs = torch.cat([embs_1, embs_2], dim=1)
        preds = self.cly_model(embs)
        labels_idx = torch.argmax(labels[idx], dim=1)
        if src_cs is None:
            cly_loss = F.cross_entropy(preds, labels_idx)
        else:
            # import ipdb; ipdb.set_trace() 
            labels = torch.scatter(torch.zeros_like(preds), 1, labels_idx.unsqueeze(1), 1).to(labels_idx.device)
            # cly_loss = - ((src_cs[idx].unsqueeze(-1) * (labels * F.log_softmax(preds, dim=1))).sum(1)).mean()
            cly_loss = - ((src_cs.unsqueeze(-1) * (labels * F.log_softmax(preds, dim=1))).sum(1)).mean()
            
        if not cal_f1:
            return embs, cly_loss
        else:
            targets = labels[idx].cpu().numpy()
            # import ipdb; ipdb.set_trace()
            preds = top_k_preds(targets, preds.detach().cpu().numpy())
            return embs, cly_loss, preds, targets