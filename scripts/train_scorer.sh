#!/usr/bin/env bash
set -euo pipefail

python train_scorer.py \
  --metadata data/scorer/metadata.jsonl \
  --model-name-or-path /scratch/zt1/project/kaiqing-prj/user/mbezick/ladir/models/Llama-3.1-8B \
  --output-dir checkpoints/latent_scorer \
  --batch-size 64 \
  --epochs 3 \
  --lr 1e-4 \
  --bf16
