#!/bin/bash

accelerate launch train.py \
    --outdir=training-runs \
    --train_dir=./data/cifar10/train \
    --val_dir=./data/cifar10/test \
    --batch_size=128 \
    --cond=1 \
    --num_steps=200000 \
    --model_channels=128 \
    --channel_mult=1,2,2,2 \
    --num_res_blocks=2 \
    --attn_resolutions=16,8 \
    --dropout_rate=0.1 \
    --lr=2e-4 \
    --lr_warmup=0 \
    --schedule_name=cosine \
    --timesteps=1000 \
    --ce_weight=0.001 \
    --label_smooth=0.2 \
    --eval_interval=100 \
    --seed=1 \
    --log_interval=10 \
    --save_interval=5000 \
    --resume_from=training-runs/00013-run