#!/bin/bash

dataset_dir="./data/bloodmnist_224"

accelerate launch train_egc.py \
    --outdir=egc-runs \
    --train_dir=${dataset_dir}/train \
    --val_dir=${dataset_dir}/val \
    --test_dir=${dataset_dir}/test \
    --batch_size=128 \
    --cond=1 \
    --num_steps=100000 \
    --accum_steps=8 \
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
    --train_on_latents=1
