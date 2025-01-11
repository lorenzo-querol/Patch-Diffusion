#!/bin/bash

python test_wrn.py \
    --outdir=for_keep/wrn_lc_v2/00000-run \
    --test_dir=./data/cifar10/test \
    --batch_size=128 \
    --cond=1 \
    --seed=1

python test_wrn.py \
    --outdir=for_keep/wrn_lc_v2/00001-run \
    --test_dir=./data/cifar10/test \
    --batch_size=128 \
    --cond=1 \
    --seed=1

python test_wrn.py \
    --outdir=for_keep/wrn_lc_v2/00002-run \
    --test_dir=./data/cifar10/test \
    --batch_size=128 \
    --cond=1 \
    --seed=1

python test_wrn.py \
    --outdir=for_keep/wrn_lc_v2/00003-run \
    --test_dir=./data/cifar10/test \
    --batch_size=128 \
    --cond=1 \
    --seed=1

python test_wrn.py \
    --outdir=for_keep/wrn_lc_v2/00004-run \
    --test_dir=./data/cifar10/test \
    --batch_size=128 \
    --cond=1 \
    --seed=1