#!/bin/bash

seeds=(1 2 3 4 5)
datasets=("bloodmnist_256" "dermamnist_256" "organcmnist_256" "organsmnist_256")

# NOTE: Only difference from EGC paper was the LR used, instead of 1e-4, we used 5e-5
# This was done to prevent the model from overfitting too quickly and diverging

for seed in "${seeds[@]}"; do
    for dataset_name in "${datasets[@]}"; do
        dataset_dir="./data/${dataset_name}"
        accelerate launch train_egc.py \
            --outdir=egc-runs/${dataset_name}/least_conf \
            --train_dir=${dataset_dir}/train \
            --val_dir=${dataset_dir}/val \
            --test_dir=${dataset_dir}/test \
            --batch_size=128 \
            --cond=1 \
            --num_steps=5000 \
            --accum_steps=4 \
            --model_channels=256 \
            --channel_mult=1,2,3,4 \
            --num_res_blocks=3 \
            --attn_resolutions=16,8 \
            --dropout_rate=0.0 \
            --lr=5e-5 \
            --schedule_name=linear \
            --timesteps=1000 \
            --target=epsilon \
            --ce_weight=0.001 \
            --seed=${seed} \
            --log_interval=10 \
            --eval_interval=100 \
            --save_interval=0 \
            --exp_type=active \
            --strategy=lc \
            --train_on_latents=1
    done
done
