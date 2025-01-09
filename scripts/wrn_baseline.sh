#!/bin/bash

seeds=(1 2 3 4 5)

for seed in "${seeds[@]}"; do
    accelerate launch train_wrn.py \
        --outdir=wrn-runs/seed_$seed \
        --train_dir=./data/cifar10/train \
        --val_dir=./data/cifar10/test \
        --test_dir=./data/cifar10/test \
        --batch_size=128 \
        --cond=1 \
        --num_epochs=200 \
        --accum_steps=1 \
        --decay_epochs=60,120,160 \
        --decay_rate=0.2 \
        --depth=28 \
        --width_factor=10 \
        --dropout_rate=0.3 \
        --lr=0.1 \
        --seed=$seed \
        --eval_interval=5 \
        --exp_type=baseline \
        --num_samples=0.1 \
        --calibrate=0
done


