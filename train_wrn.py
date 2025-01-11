import json
import os
import re
import warnings

import click
from accelerate import Accelerator

import dnnlib
from training import trainer_wrn
from training.trainer_active import WRNActiveLearningTrainer

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

# Main options.
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

# Active learning-related.
@click.option("--num_samples", help="Number of samples to query", metavar="FLOAT", type=float, default=0.1, show_default=True)
@click.option("--calibrate", help="Apply temperature scaling", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--strategy", help="Active learning strategy", metavar="str", type=click.Choice(["random", "lc"]), default="lc", show_default=True)

# Hyperparameters.
@click.option("--depth", help="Network depth (should be 6n+4)", metavar="INT", type=click.IntRange(min=1), default=28, show_default=True)
@click.option("--width_factor", help="Width factor k", metavar="INT", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--dropout_rate", help="Dropout rate", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=0.3, show_default=True)
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=0.1, show_default=True)

# Classification-related.
@click.option("--eval_interval", help="How often to evaluate the model on the test dataset", metavar="TICKS", type=click.IntRange(min=0), default=100, show_default=True)

# I/O-related.
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int, default=1)
@click.option("--resume_from", help="Resume from a previous checkpoint", metavar="DIR", type=str, default=None)
@click.option("--exp_type", help="Type of experiment", metavar="str", type=click.Choice(["active", "baseline"]), default="baseline", show_default=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    accelerator = Accelerator()
    print_fn = accelerator.print
    trainer_kwargs = dnnlib.EasyDict()

    # Dataset/loader options
    trainer_kwargs.dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        use_labels=opts.cond,
        path=opts.train_dir,
    )
    trainer_kwargs.val_dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        use_labels=opts.cond,
        path=opts.val_dir,
    )
    trainer_kwargs.test_dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        use_labels=opts.cond,
        path=opts.test_dir,
    )
    trainer_kwargs.network_kwargs = dnnlib.EasyDict(
        class_name="training.wrn.WideResNet",
        depth=opts.depth,
        width_factor=opts.width_factor,
        dropout_rate=opts.dropout_rate,
        use_bn=True,
    )
    trainer_kwargs.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.SGD", lr=opts.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    trainer_kwargs.num_epochs = opts.num_epochs
    trainer_kwargs.accum_steps = opts.accum_steps
    trainer_kwargs.batch_size = opts.batch_size
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
    print_fn(f"Test path:               {trainer_kwargs.test_dataset_kwargs.path}")
    print_fn(f"Batch size:              {trainer_kwargs.batch_size}")
    print_fn(f"Epochs:                  {trainer_kwargs.num_epochs}")
    print_fn(f"Resume from:             {trainer_kwargs.resume_from}")
    print_fn(f"Random seed:             {trainer_kwargs.seed}")
    print_fn(f"Accumulation steps:      {trainer_kwargs.accum_steps}")
    print_fn(f"Number of GPUs:          {accelerator.num_processes}")
    print_fn()

    # Create output directory.
    print_fn("Creating output directory...")

    if accelerator.is_main_process:
        os.makedirs(trainer_kwargs.run_dir, exist_ok=True)
        with open(os.path.join(trainer_kwargs.run_dir, "training_options.json"), "wt") as f:
            json.dump(trainer_kwargs, f, indent=2)

        dnnlib.util.Logger(
            file_name=os.path.join(trainer_kwargs.run_dir, "log.txt"),
            file_mode="a",
            should_flush=True,
        )

    if opts.exp_type == "active":
        trainer = WRNActiveLearningTrainer(
            num_samples=opts.num_samples,
            calibrate=opts.calibrate,
            strategy=opts.strategy,
            **trainer_kwargs,
        )
        trainer.run_active_learning(eval_interval=opts.eval_interval)
    elif opts.exp_type == "baseline":
        trainer = trainer_wrn.Trainer(**trainer_kwargs)
        trainer.train(eval_interval=opts.eval_interval)


if __name__ == "__main__":
    main()
