# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/


import json
import os
import re
import warnings

import click
import torch
from torch_utils import distributed as dist


import dnnlib
from training import training_loop

warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")  # False warning printed by PyTorch 1.12.

# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


@click.command()

# Patch options
@click.option("--real_p", help="Full size image ratio", metavar="INT", type=click.FloatRange(min=0, max=1), default=0.5, show_default=True)
@click.option("--train_on_latents", help="Training on latent embeddings", metavar="BOOL", type=bool, default=False, show_default=True)

# Main options.
@click.option("--outdir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--train_dir", help="Path to the train dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--val_dir", help="Path to the valid dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--cond", help="Train class-conditional model", metavar="BOOL", type=bool, default=False, show_default=True)

# Hyperparameters.
@click.option("--duration", help="Training duration", metavar="MIMG", type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option("--batch_size", help="Total batch size", metavar="INT", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--batch-gpu", help="Limit batch size per GPU", metavar="INT", type=click.IntRange(min=1))
@click.option("--channel_mult", help="Channel multiplier  [default: varies]", metavar="LIST", type=parse_int_list)
@click.option("--model_channels", help="Channels per resolution  [default: varies]", metavar="INT", type=int)
@click.option("--attn_resolutions", help="Resolutions to use attention layers", metavar="LIST", type=parse_int_list)
@click.option("--dropout_rate", help="Dropout rate", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option("--ema", help="EMA half-life", metavar="MIMG", type=click.FloatRange(min=0), default=0.5, show_default=True)

# Diffusion-related.
@click.option("--schedule_name", help="Diffusion schedule", metavar="str", type=click.Choice(["linear", "cosine"]), show_default=True)
@click.option("--timesteps", help="Number of diffusion timesteps", metavar="INT", type=click.IntRange(min=1), default=1000, show_default=True)

# Classification-related.
@click.option("--ce_weight", help="Cross-entropy loss weight", metavar="FLOAT", type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option("--eval_interval", help="How often to evaluate the model on the test dataset", metavar="TICKS", type=click.IntRange(min=1), default=10, show_default=True)

# Performance-related.
@click.option("--bench", help="Enable cuDNN benchmarking", metavar="BOOL", type=bool, default=True, show_default=True)

# I/O-related.
@click.option("--tick", help="How often to print progress", metavar="KIMG", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--snap", help="How often to save snapshots", metavar="TICKS", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--dump", help="How often to dump state", metavar="TICKS", type=click.IntRange(min=1), default=100, show_default=True)
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int, default=1)
@click.option("--transfer", help="Transfer learning from network pickle", metavar="PKL|URL", type=str)
@click.option("--resume", help="Resume from previous training state", metavar="PT", type=str)
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    c = dnnlib.EasyDict()

    def get_dataset_kwargs(path):
        kwargs = dnnlib.EasyDict()
        kwargs.class_name = "training.dataset.ImageFolderDataset"
        kwargs.use_labels = opts.cond
        kwargs.path = path
        return kwargs

    # Dataset/loader options
    c.dataset_kwargs = get_dataset_kwargs(opts.train_dir)
    c.val_dataset_kwargs = get_dataset_kwargs(opts.val_dir)
    c.dataloader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=4)

    c.network_kwargs = dnnlib.EasyDict(
        class_name="training.networks.EBMUNet",
        model_channels=opts.model_channels,
        channel_mult=opts.channel_mult,
        attn_resolutions=opts.attn_resolutions,
        dropout_rate=opts.dropout_rate,
    )
    c.diffusion_kwargs = dnnlib.EasyDict(
        class_name="training.diffusion.GaussianDiffusion",
        schedule_name=opts.schedule_name,
        timesteps=opts.timesteps,
    )
    c.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", lr=opts.lr)

    c.real_p = opts.real_p
    c.train_on_latents = opts.train_on_latents

    # Training options
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch_size, batch_gpu=opts.batch_gpu)
    c.update(cudnn_benchmark=opts.bench)

    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed
    c.seed = opts.seed

    # Transfer learning and resume
    if opts.transfer:
        if opts.resume:
            raise click.ClickException("--transfer and --resume cannot be specified at the same time")
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume:
        match = re.fullmatch(r"training-state-(\d+).pt", os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException("--resume must point to training-state-*.pt from a previous training run")
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f"network-snapshot-{match.group(1)}.pkl")
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f"{cur_run_id:05d}-run")
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0("Training options:")
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f"Output directory:        {c.run_dir}")
    dist.print0(f"Dataset path:            {c.dataset_kwargs.path}")
    dist.print0(f"Validation path:         {c.val_dataset_kwargs.path}")
    dist.print0(f"Batch size:              {c.batch_size}")
    dist.print0(f"Diffusion schedule:      {c.diffusion_kwargs.schedule_name}")
    dist.print0(f"Timesteps:               {c.diffusion_kwargs.timesteps}")
    dist.print0(f"Number of GPUs:          {dist.get_world_size()}")
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0("Dry run; exiting.")
        return

    # Create output directory.
    dist.print0("Creating output directory...")
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, "log.txt"), file_mode="a", should_flush=True)

    # Train.
    training_loop.fit(**c)


if __name__ == "__main__":
    main()
