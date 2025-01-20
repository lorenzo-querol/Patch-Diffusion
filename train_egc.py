import json

import click
from accelerate import Accelerator

import dnnlib
from exp_utils import create_output_directory, generate_run_id, parse_int_list
from training import trainer_egc
from training.trainer_active import EGCActiveLearningTrainer

import warnings

warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")  # False warning printed by PyTorch 1.12.


@click.command()

# Main
@click.option("--outdir", help="Directory to save the results", metavar="DIR", type=str, required=True)
@click.option("--train_dir", help="Path to the training dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--val_dir", help="Path to the validation dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--test_dir", help="Path to the test dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--batch_size", help="Total batch size", metavar="INT", type=click.IntRange(min=1), default=128, show_default=True)
@click.option("--cond", help="Train class-conditional model", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--num_steps", help="Number of training steps", metavar="INT", type=click.IntRange(min=1), default=100000, show_default=True)
@click.option("--accum_steps", help="Number of steps to accumulate gradients over", metavar="INT", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--resume_from", help="Resume from a previous checkpoint", metavar="DIR", type=str, default=None)

# Active learning
@click.option("--num_samples", help="Number of samples to query", metavar="FLOAT", type=float, default=0.1, show_default=True)
@click.option("--strategy", help="Active learning strategy", metavar="str", type=click.Choice(["random", "lc", "sm", "entropy"]), default="random", show_default=True)

# Hyperparameters
@click.option("--model_channels", help="Number of channels per resolution", metavar="INT", type=int)
@click.option("--channel_mult", help="Channel multiplier", metavar="LIST", type=parse_int_list)
@click.option("--num_res_blocks", help="Number of residual blocks", metavar="INT", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--attn_resolutions", help="Resolutions to use attention layers", metavar="LIST", type=parse_int_list)
@click.option("--dropout_rate", help="Dropout rate", metavar="FLOAT", type=click.FloatRange(min=0, max=1), show_default=True)
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=1e-4, show_default=True)
@click.option("--ce_weight", help="Cross-entropy loss weight", metavar="FLOAT", type=click.FloatRange(min=0), default=0.001, show_default=True)
@click.option("--train_on_latents", help="Train on latent embeddings", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--real_p", help="Probability of patches", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=0.5, show_default=True)

# Diffusion
@click.option("--schedule_name", help="Diffusion schedule", metavar="STR", type=click.Choice(["linear", "cosine"]), show_default=True)
@click.option("--timesteps", help="Number of diffusion timesteps", metavar="INT", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--target", help="Target value for diffusion", metavar="STR", type=click.Choice(["epsilon", "x_0", "v"]), default="epsilon", show_default=True)

# I/O
@click.option("--seed", help="Random seed", metavar="INT", type=int, default=1)
@click.option("--eval_interval", help="Interval to evaluate the model on the test dataset", metavar="TICKS", type=click.IntRange(min=1), default=100, show_default=True)
@click.option("--log_interval", help="Interval to log the training metrics", metavar="TICKS", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--save_interval", help="Interval to save the model", metavar="TICKS", type=click.IntRange(min=0), default=5000, show_default=True)
@click.option("--exp_type", help="Experiment type", metavar="STR", type=click.Choice(["baseline", "active"]), default="baseline", show_default=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    accelerator = Accelerator()
    print_fn = accelerator.print
    trainer_kwargs = dnnlib.EasyDict()

    # Dataset options
    trainer_kwargs.dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", use_labels=opts.cond, path=opts.train_dir)
    trainer_kwargs.val_dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", use_labels=opts.cond, path=opts.val_dir)
    trainer_kwargs.test_dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", use_labels=opts.cond, path=opts.test_dir)

    # Network options
    trainer_kwargs.network_kwargs = dnnlib.EasyDict(
        class_name="training.networks.EBMUNet",
        model_channels=opts.model_channels,
        num_res_blocks=opts.num_res_blocks,
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
    trainer_kwargs.diffusion_kwargs = dnnlib.EasyDict(class_name="training.diffusion.GaussianDiffusionTrainer", target=opts.target, schedule_name=opts.schedule_name, timesteps=opts.timesteps)
    trainer_kwargs.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.AdamW", lr=opts.lr, weight_decay=0.0)
    trainer_kwargs.target = opts.target
    trainer_kwargs.ce_weight = opts.ce_weight
    trainer_kwargs.train_on_latents = opts.train_on_latents

    # Training options
    trainer_kwargs.num_steps = opts.num_steps
    trainer_kwargs.accum_steps = opts.accum_steps
    trainer_kwargs.batch_size = opts.batch_size
    trainer_kwargs.seed = opts.seed
    trainer_kwargs.resume_from = opts.resume_from
    trainer_kwargs.run_dir = generate_run_id(opts)

    print_fn()
    print_fn("Training options:")
    print_fn(json.dumps(trainer_kwargs, indent=2))
    print_fn()
    print_fn(f"Output directory:        {trainer_kwargs.run_dir}")
    print_fn(f"Dataset path:            {trainer_kwargs.dataset_kwargs.path}")
    print_fn(f"Validation path:         {trainer_kwargs.val_dataset_kwargs.path}")
    print_fn(f"Test path:               {trainer_kwargs.test_dataset_kwargs.path}")
    print_fn(f"Batch size:              {trainer_kwargs.batch_size}")
    print_fn(f"Diffusion schedule:      {trainer_kwargs.diffusion_kwargs.schedule_name}")
    print_fn(f"Timesteps:               {trainer_kwargs.diffusion_kwargs.timesteps}")
    print_fn(f"Target:                  {trainer_kwargs.target}")
    print_fn(f"Training steps:          {trainer_kwargs.num_steps}")
    print_fn(f"Resume from:             {trainer_kwargs.resume_from}")
    print_fn(f"Random seed:             {trainer_kwargs.seed}")
    print_fn(f"Accumulation steps:      {trainer_kwargs.accum_steps}")
    print_fn(f"Number of GPUs:          {accelerator.num_processes}")
    print_fn()

    print_fn("Creating output directory...")
    if accelerator.is_main_process:
        create_output_directory(trainer_kwargs)

    match opts.exp_type:
        case "active":
            trainer = EGCActiveLearningTrainer(num_samples=opts.num_samples, strategy=opts.strategy, **trainer_kwargs)
            trainer.run_active_learning_loop(log_interval=opts.log_interval, save_interval=opts.save_interval, eval_interval=opts.eval_interval)
        case "baseline":
            trainer = trainer_egc.Trainer(**trainer_kwargs)
            trainer.train(log_interval=opts.log_interval, save_interval=opts.save_interval, eval_interval=opts.eval_interval)
        case _:
            raise NotImplementedError(f"Experiment type {opts.exp_type} not implemented.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
