#!/bin/bash

seeds=(1)
# datasets=("bloodmnist_256" "dermamnist_256" "organcmnist_256" "organsmnist_256")
datasets=("bloodmnist_256")

for seed in "${seeds[@]}"; do
    for dataset_name in "${datasets[@]}"; do
        dataset_dir="./data/${dataset_name}"
        accelerate launch train_egc.py \
            --outdir=egc-runs/${dataset_name}/baseline \
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
            --seed=$seed \
            --log_interval=10 \
            --eval_interval=1000 \
            --save_interval=5000 \
            --exp_type=baseline \
            --train_on_latents=1
    done
done