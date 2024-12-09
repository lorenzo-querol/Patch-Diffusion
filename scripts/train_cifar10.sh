#!/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L gpu=2
#PJM -L elapse=6:00:00
#PJM -j

# source ~/.bashrc
# conda activate edm

accelerate launch train.py \
    --outdir=training-runs \
    --train_dir=./data/cifar10/train \
    --val_dir=./data/cifar10/test \
    --batch_size=128 \
    --cond=1 \
    --num_epochs=500 \
    --model_channels=64 \
    --channel_mult=1,2,2 \
    --num_res_blocks=3 \
    --attn_resolutions=16,8 \
    --dropout_rate=0 \
    --lr=1e-4 \
    --lr_warmup=0 \
    --schedule_name=cosine \
    --timesteps=1000 \
    --ce_weight=0.001 \
    --label_smooth=0.2 \
    --eval_interval=5 \
    --seed=1 \
    --save_interval=10