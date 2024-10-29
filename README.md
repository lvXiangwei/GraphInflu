# GraphInflu

### Baseline
#### source ratio: 0.3(random)
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

#### source ratio: 1.0
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
    --source_ratio 1.0 \
```

### Run GraphInflu
#### source ratio: 0.3 (based on contribution score)
```bash
python -u main.py \
    --lr 3e-3\
    --source "dblpv7" \
    --target  "acmv9"\
    --seed 200\
    --wandb 0 \
    --wandb_info D2A-GraphInflu\
    --target_shots 5 \
    --batch_size 256\
    --device cuda:1\
    --epochs 30 \
    --source_ratio 0.3 \
    --use_source_score 1 \
    --score_path score/acmv9/dblpv7_influence_score.pt \
    --soft 1 \
```
---
##### Extra: Calculate custom contribution score
```bash
bash scripts/score.sh
```
