# GraphInflu

### Run Baseline
```bash
python -u main.py \
    --lr 3e-3\
    --source "dblpv7" \
    --target  "acmv9"\
    --seed 200\
    --wandb 0 \
    --wandb_info D2A-baseline\
    --target_shots 5 \
    --batch_size 256\
    --device cuda:1\
    --epochs 30 \
    --source_ratio 0.3 \
```

### Run GraphInflu

```bash
python -u main.py \
    --lr 3e-3\
    --source "dblpv7" \
    --target  "acmv9"\
    --seed 200\
    --wandb 0 \
    --wandb_info D2A-GraphInflu\
    --source_ratio 1.0 \
    --select_ratio 0.3\
    --target_shots 5 \
    --batch_size 256\
    --device cuda:1\
    --epochs 30 \
    --use_source_score 1 \
    --score_path score/acmv9/dblpv7_influence_score.pt \
```

