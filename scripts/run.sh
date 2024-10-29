########  baseline
# python -u main.py \
#     --lr 3e-3\
#     --source "dblpv7" \
#     --target  "acmv9"\
#     --seed 200\
#     --wandb 0 \
#     --wandb_info D2A-baseline\
#     --target_shots 5 \
#     --batch_size 256\
#     --device cuda:6\
#     --epochs 30 \
#     --source_ratio 0.3 \


######## GraphInflu
python -u main.py \
    --lr 3e-3\
    --source "dblpv7" \
    --target  "acmv9"\
    --seed 0\
    --wandb 0 \
    --wandb_info D2A-GraphInflu_seed0\
    --target_shots 5 \
    --batch_size 256\
    --device cuda:6\
    --epochs 30 \
    --source_ratio 0.3 \
    --use_source_score 1 \
    --score_path score/acmv9/dblpv7_influence_score.pt \
    --soft 1

