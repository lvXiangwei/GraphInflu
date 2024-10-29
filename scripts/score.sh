# step 1
seed=200
source="dblpv7"
target="acmv9"
desp="D2A"

python -u main.py \
    --lr 3e-3\
    --source $source \
    --target  $target\
    --seed 200\
    --wandb 0 \
    --wandb_info $desp-warmup\
    --target_shots 5 \
    --batch_size 256\
    --epochs 30 \
    --device cuda:0\
    --warmup 1\
    --checkpoint_path checkpoints/$desp \
    --save_step 5

# step 2, get source, target gradient store
for epoch in 5 10 15 20; do
    python collect_grads.py \
        --lr 3e-3\
        --source $source \
        --target  $target\
        --seed $seed\
        --wandb 0 \
        --wandb_info $desp-grad_src\
        --source_ratio 1.0 \
        --target_shots 5 \
        --batch_size 256\
        --epochs 30 \
        --grad_source 1 \
        --device cuda:1 \
        --checkpoint_path checkpoints/$desp \
        --checkpoint_epoch $epoch 
done

for epoch in 5 10 15 20; do
    python collect_grads.py \
        --lr 3e-3\
        --source $source \
        --target  $target\
        --seed $seed\
        --wandb 1 \
        --wandb_info $desp-grad_tgt\
        --target_shots 5 \
        --batch_size 256\
        --epochs 30 \
        --grad_target 1\
        --checkpoint_path checkpoints/$desp\
        --checkpoint_epoch $epoch
done

# step 3, get contribution score
python match.py \
    --source_gradient_path grads/$desp/{}_epoch{}/dim4096 \
    --source $source\
    --ckpts 5 10 15 20\
    --checkpoint_weights 3e-3 3e-3 3e-3 3e-3\
    --target $target\
    --target_gradient_path grads/$desp/{}_epoch{}/dim4096\
    --output_path score