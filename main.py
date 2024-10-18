import logging
import sys

from parse_args import parse_args
from transformers import set_seed
from utils import get_source_dataset, get_target_dataset, load_model, create_optim, get_source_idx
from train import train

def main(args):
    # Setup logging
    logger = logging.getLogger(args.wandb_info)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler = logging.FileHandler(f"logs/{args.wandb_info}.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    
    logger.info(f"Training parameters {args}")
    set_seed(args.seed)

    # 1. Load source dataset (source_ratio， use_source_score are used to select source data)
    source_data = get_source_dataset(args.model, 
                                         args.source,
                                         args.source_ratio, # 
                                         args.use_source_score,
                                         args.score_path,
                                         args.source_sample_seed)

    

    # 2. Load target dataset (target_shots, number of shots for each class)
    target_data = get_target_dataset(args.model, 
                                         args.target,
                                         args.target_shots,
                                         args.target_sample_seed)
    
    

                
    # 3. Load modes
    num_features = source_data['feature_s'].shape[1]
    num_class = target_data['label_t'].shape[1]
    model = load_model(num_features, num_class, args)
    optimizer = create_optim(model, args)
    train(model, optimizer, source_data, target_data, args.epochs, args, logger)
        
    
if __name__ == "__main__":
    args = parse_args()
    main(args)