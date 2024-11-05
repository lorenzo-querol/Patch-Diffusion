#!/bin/bash

# Default values
OUTDIR="training-runs"
DATA="../data/cifar10"
COND=true
ARCH="adm"
PRECOND="pedm"
DURATION=200
BATCH=128
CBASE=128
CRES="1,2,2"
LR=0.001
EMA=0.5
DROPOUT=0.13
AUGMENT=0.12
XFLIP=false
IMPLICIT_MLP=false
FP16=false
LS=1
BENCH=true
CACHE=true
WORKERS=1
TICK=50
SNAP=50
DUMP=500
SEED=42
TRANSFER=""
RESUME=""
DRY_RUN=false
REAL_P=0.5
TRAIN_ON_LATENTS=false
DESC="custom"

# Run the training script
torchrun --standalone --nproc_per_node=1 train.py \
    --outdir $OUTDIR \
    --data $DATA \
    --cond $COND \
    --arch $ARCH \
    --precond $PRECOND \
    --duration $DURATION \
    --batch $BATCH \
    --cbase $CBASE \
    --cres $CRES \
    --lr $LR \
    --ema $EMA \
    --dropout $DROPOUT \
    --augment $AUGMENT \
    --xflip $XFLIP \
    --implicit_mlp $IMPLICIT_MLP \
    --fp16 $FP16 \
    --ls $LS \
    --bench $BENCH \
    --cache $CACHE \
    --workers $WORKERS \
    --tick $TICK \
    --snap $SNAP \
    --dump $DUMP \
    --seed $SEED \
    --real_p $REAL_P \
    --desc $DESC \
    $(if [ "$DRY_RUN" = true ]; then echo "--dry-run"; fi)