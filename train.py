import json
import os
import re
import warnings

import click
from accelerate import Accelerator
import torch_utils.distributed as dist

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
@click.option("--train_on_latents", help="Training on latent embeddings", metavar="BOOL", type=bool, default=False, show_default=True)

# Main options.
@click.option("--outdir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--train_dir", help="Path to the train dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--val_dir", help="Path to the valid dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--cond", help="Train class-conditional model", metavar="BOOL", type=bool, default=False, show_default=True)

# Hyperparameters.
@click.option("--duration", help="Training duration", metavar="MIMG", type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option("--batch_size", help="Total batch size", metavar="INT", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--channel_mult", help="Channel multiplier  [default: varies]", metavar="LIST", type=parse_int_list)
@click.option("--model_channels", help="Channels per resolution  [default: varies]", metavar="INT", type=int)
@click.option("--num_blocks", help="Number of residual blocks", metavar="INT", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--attn_resolutions", help="Resolutions to use attention layers", metavar="LIST", type=parse_int_list)
@click.option("--dropout_rate", help="Dropout rate", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)

# Diffusion-related.
@click.option("--schedule_name", help="Diffusion schedule", metavar="str", type=click.Choice(["linear", "cosine"]), show_default=True)
@click.option("--timesteps", help="Number of diffusion timesteps", metavar="INT", type=click.IntRange(min=1), default=1000, show_default=True)

# Classification-related.
@click.option("--ce_weight", help="Cross-entropy loss weight", metavar="FLOAT", type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option("--label_smooth", help="Label smoothing", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option("--eval_interval", help="How often to evaluate the model on the test dataset", metavar="TICKS", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--resume_from", help="Resume from a previous checkpoint", metavar="DIR", type=str, default=None)

# I/O-related.
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int, default=1)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    accelerator = Accelerator()
    print_fn = accelerator.print
    trainer_kwargs = dnnlib.EasyDict()

    # Dataset/loader options
    trainer_kwargs.dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", use_labels=opts.cond, path=opts.train_dir)
    trainer_kwargs.val_dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", use_labels=opts.cond, path=opts.val_dir)
    trainer_kwargs.network_kwargs = dnnlib.EasyDict(
        class_name="training.networks.EBMUNet",
        model_channels=opts.model_channels,
        num_blocks=opts.num_blocks,
        attn_resolutions=opts.attn_resolutions,
        dropout_rate=opts.dropout_rate,
        channel_mult=opts.channel_mult,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=True,
        context_dim=512,
        use_spatial_transformer=True,
        transformer_depth=1,
        pool="sattn",
    )
    trainer_kwargs.diffusion_kwargs = dnnlib.EasyDict(
        class_name="training.diffusion.GaussianDiffusion",
        schedule_name=opts.schedule_name,
        timesteps=opts.timesteps,
    )
    trainer_kwargs.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", lr=opts.lr)
    trainer_kwargs.batch_size = opts.batch_size
    trainer_kwargs.ce_weight = opts.ce_weight
    trainer_kwargs.label_smooth = opts.label_smooth
    trainer_kwargs.train_on_latents = opts.train_on_latents
    trainer_kwargs.seed = opts.seed
    trainer_kwargs.resume_from = opts.resume_from

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(opts.outdir):
        prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    trainer_kwargs.run_dir = os.path.join(opts.outdir, f"{cur_run_id:05d}-run")
    assert not os.path.exists(trainer_kwargs.run_dir)

    # Print options.
    print_fn()
    print_fn("Training options:")
    print_fn(json.dumps(trainer_kwargs, indent=2))
    print_fn()
    print_fn(f"Output directory:        {trainer_kwargs.run_dir}")
    print_fn(f"Dataset path:            {trainer_kwargs.dataset_kwargs.path}")
    print_fn(f"Validation path:         {trainer_kwargs.val_dataset_kwargs.path}")
    print_fn(f"Batch size:              {trainer_kwargs.batch_size}")
    print_fn(f"Diffusion schedule:      {trainer_kwargs.diffusion_kwargs.schedule_name}")
    print_fn(f"Timesteps:               {trainer_kwargs.diffusion_kwargs.timesteps}")
    print_fn(f"Number of GPUs:          {accelerator.num_processes}")
    print_fn()

    # Create output directory.
    print_fn("Creating output directory...")

    if accelerator.is_main_process:
        os.makedirs(trainer_kwargs.run_dir, exist_ok=True)
        with open(os.path.join(trainer_kwargs.run_dir, "training_options.json"), "wt") as f:
            json.dump(trainer_kwargs, f, indent=2)

        dnnlib.util.Logger(file_name=os.path.join(trainer_kwargs.run_dir, "log.txt"), file_mode="a", should_flush=True)

    trainer = training_loop.Trainer(**trainer_kwargs)
    trainer.train()
    accelerator.end_training()


if __name__ == "__main__":
    main()
