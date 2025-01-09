#!/bin/bash

accelerate launch train.py \
    --outdir=training-runs \
    --train_dir=./data/bloodmnist_224/train \
    --val_dir=./data/bloodmnist_224/val \
    --batch_size=128 \
    --cond=1 \
    --num_steps=100000 \
    --accum_steps=4 \
    --model_channels=256 \
    --channel_mult=1,2,3,4 \
    --num_res_blocks=3 \
    --attn_resolutions=16,8 \
    --dropout_rate=0.0 \
    --lr=1e-4 \
    --schedule_name=linear \
    --timesteps=1000 \
    --target=epsilon \
    --ce_weight=0.001 \
    --seed=1 \
    --log_interval=10 \
    --eval_interval=1000 \
    --save_interval=5000 \
    --train_on_latents=1 \
