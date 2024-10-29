import argparse
import os

import torch

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--source_gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--source', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target', type=str,
                       nargs='+', help="The name of the target tasks")
argparser.add_argument('--target_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the target gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')


args = argparser.parse_args()

# N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_influence_score(source_info: torch.Tensor, target_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        source_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        target_info (torch.Tensor): target info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        source_info, target_info.transpose(0, 1))
    return influence_scores


# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# calculate the influence score for each target task
for target_file in args.target:
    for source_file in args.source:
        influence_score = 0
        for i, ckpt in enumerate(args.ckpts):
            target_path = args.target_gradient_path.format(
            target_file, ckpt)
            
            # target_path = args.target_gradient_path.format(
            #     ckpt, target_file)
            if os.path.isdir(target_path):
                target_path = os.path.join(target_path, "all_orig.pt")
            target_info = torch.load(target_path)
            # import ipdb; ipdb.set_trace()
            if not torch.is_tensor(target_info):
                target_info = torch.tensor(target_info)
            target_info = target_info.to(device).float()
            gradient_path = args.source_gradient_path.format(source_file, ckpt)
            # gradient_path = args.gradient_path.format(ckpt, train_file_name)
            if os.path.isdir(gradient_path):
                gradient_path = os.path.join(gradient_path, "all_orig.pt")
            source_info = torch.load(gradient_path)

            if not torch.is_tensor(source_info):
                source_info = torch.tensor(source_info)
            source_info = source_info.to(device).float()

            influence_score += args.checkpoint_weights[i] * \
                calculate_influence_score(
                    source_info=source_info, target_info=target_info)

        output_dir = os.path.join(args.output_path, target_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(
            args.output_path, target_file, f"{source_file}_influence_score.pt")
        torch.save(influence_score, output_file)
        print("Saved influence score to {}".format(output_file))
