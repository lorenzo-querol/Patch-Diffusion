import json

import click
from accelerate import Accelerator

import dnnlib
from exp_utils import create_output_directory, generate_run_id, parse_int_list
from training import trainer_wrn
from training.trainer_active import WRNActiveLearningTrainer


@click.command()

# Main
@click.option("--outdir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--train_dir", help="Path to the train dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--val_dir", help="Path to the valid dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--test_dir", help="Path to the test dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--batch_size", help="Total batch size", metavar="INT", type=click.IntRange(min=1), default=128, show_default=True)
@click.option("--cond", help="Train class-conditional model", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--num_epochs", help="Number of training steps", metavar="INT", type=click.IntRange(min=1), default=200, show_default=True)
@click.option("--accum_steps", help="Number of steps to accumulate gradients over", metavar="INT", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--decay_epochs", help="Epochs to decay learning rate over", metavar="INT", type=parse_int_list, default=[60, 120, 160], show_default=True)
@click.option("--decay_rate", help="Learning rate decay factor", metavar="FLOAT", type=click.FloatRange(min=0), default=0.2, show_default=True)
@click.option("--train_on_latents", help="Training on latent embeddings", metavar="BOOL", type=bool, default=False, show_default=True)

# Active learning
@click.option("--num_samples", help="Number of samples to query", metavar="FLOAT", type=float, default=0.1, show_default=True)
@click.option("--calibrate", help="Apply temperature scaling", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--strategy", help="Active learning strategy", metavar="str", type=click.Choice(["random", "lc", "sm", "entropy"]), default="random", show_default=True)

# Hyperparameters
@click.option("--depth", help="Network depth (should be 6n+4)", metavar="INT", type=click.IntRange(min=1), default=28, show_default=True)
@click.option("--width_factor", help="Width factor k", metavar="INT", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--dropout_rate", help="Dropout rate", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=0.3, show_default=True)
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=0.1, show_default=True)

# I/O
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int, default=1)
@click.option("--eval_interval", help="How often to evaluate the model on the test dataset", metavar="TICKS", type=click.IntRange(min=0), default=100, show_default=True)
@click.option("--resume_from", help="Resume from a previous checkpoint", metavar="DIR", type=str, default=None)
@click.option("--exp_type", help="Type of experiment", metavar="str", type=click.Choice(["active", "baseline"]), default="baseline", show_default=True)
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
        class_name="training.wrn.WideResNet",
        depth=opts.depth,
        width_factor=opts.width_factor,
        dropout_rate=opts.dropout_rate,
        use_bn=False,
    )
    trainer_kwargs.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.AdamW", lr=opts.lr, weight_decay=0.0)
    trainer_kwargs.train_on_latents = opts.train_on_latents

    # Training options
    trainer_kwargs.num_epochs = opts.num_epochs
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
    print_fn(f"Epochs:                  {trainer_kwargs.num_epochs}")
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
            trainer = WRNActiveLearningTrainer(num_samples=opts.num_samples, calibrate=opts.calibrate, strategy=opts.strategy, **trainer_kwargs)
            trainer.run_active_learning_loop(eval_interval=opts.eval_interval)
        case "baseline":
            trainer = trainer_wrn.Trainer(**trainer_kwargs)
            trainer.train(eval_interval=opts.eval_interval)
        case _:
            raise NotImplementedError(f"Experiment type {opts.exp_type} not implemented.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
