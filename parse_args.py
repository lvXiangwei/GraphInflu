import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # origin
    parser.add_argument("--source", type=str, default='acmv9')
    parser.add_argument("--target", type=str, default='dblpv7')
    parser.add_argument("--model", type=str, default='SemiGCL')
    parser.add_argument("--seed", type=int,default=200)
    parser.add_argument("--encoder_dim", type=int, default=16)


    ### SemiGCL
    parser.add_argument('--aggregator_class', type=str, default='diffusion')
    parser.add_argument('--n_samples', type=str, default='20,20')
    parser.add_argument('--output_dims', type=str, default='1024,64')
    parser.add_argument('--T', type=float, default=20.0)
    parser.add_argument('--cal_ssl', type=int, default=1)
    parser.add_argument('--ssl_param', type=float, default=0.1)
    parser.add_argument('--mme_param', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=0)

    ##### TODO
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--use_target", type=int, default=0) 
    parser.add_argument("--soft", type=int, default=0)
    #### GCN
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[1024, 64])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    #less
    ### step1 
    parser.add_argument("--checkpoint_path", type=str, default='') 
    parser.add_argument("--save_step", type=int, default=50) 
    
    ### step2
    parser.add_argument("--checkpoint_epoch", type=int, default=15) 
    parser.add_argument("--proj_dim", type=int, nargs='+', default=[4096,]) 
    
    ### step3
    parser.add_argument("--grad_source", type=int, default=0)
    parser.add_argument("--grad_target", type=int, default=0)
    
    ### step5
    parser.add_argument("--use_source_score", type=int, default=0)
    parser.add_argument("--score_path", type=str, default="select-{}/{}/{}_influence_score.pt")
    parser.add_argument("--save_checkpoint", type=int, default=1)
    
    # source data
    parser.add_argument('--source_ratio', type=float, default=1.0)
    parser.add_argument('--source_sample_seed', type=int, default=200)
    
    # target data
    parser.add_argument('--target_shots', type=int, default=0)
    parser.add_argument('--target_sample_seed', type=int, default=200)
    
    # wandb settings
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--wandb_info", type=str, default="")

    ######
    parser.add_argument('--K', type=int, default=5)
    args = parser.parse_args()

    return args

