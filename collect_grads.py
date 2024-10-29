import torch
import torch.nn as nn
from transformers import set_seed
from utils import get_source_dataset, get_target_dataset, load_model, create_optim
from parse_args import parse_args
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from tqdm import tqdm
from torch.nn.functional import normalize
import torch.nn.functional as F
import os
import numpy as np


def get_number_of_params(model):
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params

def prepare_optimizer_state(model, optimizer_state, model_name):
    '''
        根据warmup后的Adam optimizer得到相应的m和v
    '''
    if model_name == "UDAGCN":
        # import ipdb; ipdb.set_trace()
        names = [i for i, (n, p) in enumerate(model.named_parameters()) if p.requires_grad and n.split('.')[0] in [
                    'encoder', 'cls_model', 'att_model']]
    else:
        # import ipdb; ipdb.set_trace()
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        names = list(range(len(names)))
    # print(names)
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in names])
    avg = avg.cuda()
    avg_sq = avg_sq.cuda()
    # assert avg_sq.min() >= 0
    return avg, avg_sq

def obtain_gradients_with_adam(name, model, data, avg, avg_sq, idx, origin='source'):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    if name == "SemiGCL":
        if origin == "source":
            adj_s, adj_val_s, diff_idx_s, diff_val_s, feature_s, label_s = (
            data["adj_s"], data["adj_val_s"], data["diff_idx_s"],
            data["diff_val_s"], data["feature_s"], data["label_s"],
            )
            _, cly_loss_s = model(adj_s, adj_val_s, feature_s, label_s, diff_idx_s, diff_val_s, idx=idx)
        
            if args.cal_ssl:
                model.ssl_model.train()
                shuf_idx_s = np.arange(label_s.shape[0])
                np.random.shuffle(shuf_idx_s)
                shuf_feat_s = feature_s[shuf_idx_s, :]
                # shuf_idx_t = np.arange(label_t.shape[0])
                # np.random.shuffle(shuf_idx_t)
                # shuf_feat_t = feature_t[shuf_idx_t, :]

                if len(idx) == 0: ##### TODO, for source ratio
                    ssl_loss_s = 0.0 
                else:
                    h_s_1 = model.emb_model(idx, adj_s, adj_val_s, feature_s)
                    h_s_2 = model.emb_model(idx, diff_idx_s, diff_val_s, feature_s)
                    h_s_3 = model.emb_model(idx, adj_s, adj_val_s, shuf_feat_s)
                    h_s_4 = model.emb_model(idx, diff_idx_s, diff_val_s, shuf_feat_s)
                    logits_s = model.ssl_model(h_s_1, h_s_2, h_s_3, h_s_4)
                    labels_ssl_s = torch.cat([torch.ones(h_s_1.shape[0] * 2), torch.zeros(h_s_1.shape[0] * 2)]).unsqueeze(0).to(adj_s.device)
                    ssl_loss_s = F.binary_cross_entropy_with_logits(logits_s, labels_ssl_s)


            loss = ssl_loss_s + cly_loss_s 

        elif origin == "target":
            adj_t, adj_val_t, diff_idx_t, diff_val_t, feature_t, label_t, idx_train_t, idx_val_t, idx_test_t, idx_tot_t = (
            data["adj_t"], data["adj_val_t"], data["diff_idx_t"], 
            data["diff_val_t"], data["feature_t"], data["label_t"], 
            data["idx_train_t"], data["idx_val_t"], data["idx_test_t"], 
            data["idx_tot_t"]
        )
            _, cly_loss_t = model(adj_t, adj_val_t, feature_t, label_t, diff_idx_t,
                                                    diff_val_t, idx=idx)
            
            model.ssl_model.train()
            shuf_idx_t = np.arange(label_t.shape[0])
            np.random.shuffle(shuf_idx_t)
            shuf_feat_t = feature_t[shuf_idx_t, :]
            
            if args.cal_ssl:
                b_nodes_t_plus = torch.cat((idx, idx_train_t), dim=0)
                h_t_1 = model.emb_model(b_nodes_t_plus, adj_t, adj_val_t, feature_t)
                h_t_2 = model.emb_model(b_nodes_t_plus, diff_idx_t, diff_val_t, feature_t)
                h_t_3 = model.emb_model(b_nodes_t_plus, adj_t, adj_val_t, shuf_feat_t)
                h_t_4 = model.emb_model(b_nodes_t_plus, diff_idx_t, diff_val_t, shuf_feat_t)
                logits_t = model.ssl_model(h_t_1, h_t_2, h_t_3, h_t_4)
                labels_ssl_t = torch.cat([torch.ones(h_t_1.shape[0] * 2), torch.zeros(h_t_1.shape[0] * 2)]).unsqueeze(0).to(adj_t.device)
                ssl_loss_t = F.binary_cross_entropy_with_logits(logits_t, labels_ssl_t)

            loss = ssl_loss_t + cly_loss_t         
        loss.backward()
    elif name == "UDAGCN":
        
        if origin == "source":
            encoded_source = model.encode(data, "source")
            source_logits = model.cls_model(encoded_source)
            loss = model.loss_func(source_logits[idx], data.y[idx])
        elif origin == "target":
            encoded_target = model.encode(data, "target")
            target_logits = model.cls_model(encoded_target)
            loss = model.loss_func(target_logits[idx], data.y[idx])
        loss.backward()
    else:
        loss_func = nn.CrossEntropyLoss()
        logits = model(data)
        loss = loss_func(logits[idx], data.y[idx])
        # print(logits[idx], graph.y[idx], loss)
        loss.backward()

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
    # num_nan = torch.isnan(vectorized_grads).sum()
    # print("Number of NaN values:", num_nan)
    # import ipdb; ipdb.set_trace()
    assert len(vectorized_grads) == len(avg_sq), print(len(vectorized_grads), len(avg_sq))
    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * (vectorized_grads ** 2)
    # print(avg_sq.min())
    vectorized_grads = updated_avg / (torch.sqrt(updated_avg_sq) + eps)
    num_nan = torch.isnan(vectorized_grads).sum()
    assert num_nan <= 0
    return vectorized_grads

def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    # import ipdb; ipdb.set_trace()
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")

def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def main(args):
    set_seed(args.seed)
    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors
    project_interval = 16  # project every 16 batches
    save_interval = 160  # save every 160 batches
    proj_dim = args.proj_dim
    
    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(
                current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            print(
                f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
            projected_grads[dim] = []
    
    # get data
    if args.grad_source == 1:
        data = get_source_dataset(args.model, 
                                args.source,
                                args.source_ratio,
                                )
        datafrom = "source" # gradients
        if args.model == "SemiGCL":
            num_features = data['feature_s'].shape[1]
            num_class = data['label_s'].shape[1]
            select_idx =  data['idx_tot_s'].squeeze().tolist()
        else:
            select_idx = torch.nonzero(data.source_mask).squeeze().tolist()
        dataset_name = args.source # save name
    elif args.grad_target == 1:
        data = get_target_dataset(args.model, 
                                args.target,
                                args.target_shots,
                                args.target_sample_seed)
        datafrom = "target" # gradients
        if args.model == "SemiGCL":
            num_features = data['feature_t'].shape[1]
            num_class = data['label_t'].shape[1]
            # import ipdb; ipdb.set_trace()
            select_idx =  data['idx_train_t'].squeeze().tolist()
        else:
            select_idx = torch.nonzero(data.train_mask).squeeze().tolist()
        dataset_name = args.target # save name
        
    if args.model == "SemiGCL":
        for key, val in data.items():
            data[key] = val.cuda()
    else:
        data = data.cuda()
    # load checkpoints
    checkpoint_state = torch.load(f'{args.checkpoint_path}/{args.checkpoint_epoch}.pth')
    
    # load model and optim
    if args.model == "SemiGCL":
        model = load_model(num_features, num_class, args)
    else:
        num_features, num_class = data.x.shape[1], data.y.max() + 1
        model = load_model(num_features, num_class, args)
    model.load_state_dict(checkpoint_state['model_state_dict'])
    model = model.cuda()
    number_of_params = get_number_of_params(model)
    adam_optim_state = checkpoint_state['optimizer_state_dict']['state']
    
    # projectors
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    projectors = []
    
    projector = get_trak_projector(device)
    for dim in proj_dim: # proj_dim = [8192,]
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        projectors.append(proj)
    
    output_dirs = {}
    
    desp=args.source[0].capitalize() + '2' + args.target[0].capitalize()
    for dim in proj_dim:        
        output_dir_per_dim = os.path.join(f'grads/{desp}/{dataset_name}_epoch{args.checkpoint_epoch}', f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)
    
    # projected_gradients
    full_grads = []
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients
    
    count = 0
    
    m, v = prepare_optimizer_state(model, adam_optim_state, args.model)
    
    model.train()
    # import ipdb; ipdb.set_trace()
    for idx in tqdm(select_idx):
        count += 1
        if args.model == "SemiGCL":
            vectorized_grads = obtain_gradients_with_adam(args.model, model, data, m, v, torch.LongTensor([idx]).cuda(), datafrom)
        elif args.model == "UDAGCN":
            vectorized_grads = obtain_gradients_with_adam(args.model, model, data, m, v, idx, datafrom)
        else:
            vectorized_grads = obtain_gradients_with_adam(args.model, model, data, m, v, idx)
        full_grads.append(vectorized_grads)
        model.zero_grad()
        
        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = []
        
        if count % save_interval == 0:
            _save(projected_grads, output_dirs)
    
    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []
    
    for dim in proj_dim:
        _save(projected_grads, output_dirs)
          
    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    