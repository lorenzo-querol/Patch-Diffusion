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
    --num_steps=200000 \
    --model_channels=128 \
    --channel_mult=1,2,2,2 \
    --num_res_blocks=3 \
    --attn_resolutions=16,8 \
    --dropout_rate=0.3 \
    --lr=1e-4 \
    --lr_warmup=0 \
    --schedule_name=cosine \
    --timesteps=1000 \
    --target=epsilon \
    --ce_weight=0.001 \
    --label_smooth=0 \
    --seed=1 \
    --log_interval=10 \
    --eval_interval=1000 \
    --save_interval=5000
