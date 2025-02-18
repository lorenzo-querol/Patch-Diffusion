import click
from accelerate import Accelerator

import dnnlib
from exp_utils import create_output_directory, generate_run_id
from training.datamodule import WRNDataModule
from training.resnet import BasicBlock, Bottleneck
from training.trainer_active import ActiveLearningTrainer
from training.trainer_wrn import WRNTrainer


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
@click.option("--warmup_steps", help="Number of warmup steps", metavar="INT", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--train_on_latents", help="Training on late nt embeddings", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--model", help="Model to use", metavar="str", type=click.Choice(["resnet50", "resnet18", "wrn"]), default="wrn", show_default=True)
# Active learning
@click.option("--num_samples", help="Number of samples to query", metavar="FLOAT", type=float, default=0.1, show_default=True)
@click.option("--strategy", help="Active learning strategy", metavar="str", type=click.Choice(["random", "lc", "sm", "entropy"]), default="random", show_default=True)
# Hyperparameters
@click.option("--depth", help="Network depth (should be 6n+4)", metavar="INT", type=click.IntRange(min=1), default=28, show_default=True)
@click.option("--width_factor", help="Width factor k", metavar="INT", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--dropout_rate", help="Dropout rate", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=0.3, show_default=True)
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=0.1, show_default=True)
@click.option("--optimizer", help="Optimizer", metavar="str", type=click.Choice(["adam", "sgd"]), default="adam", show_default=True)
@click.option("--use_bn", help="Use batch normalization", metavar="BOOL", type=bool, default=True, show_default=True)
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
    datamodule_kwargs = dnnlib.EasyDict()
    active_learning_kwargs = dnnlib.EasyDict()

    # Dataset options
    datamodule_kwargs.dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        use_labels=opts.cond,
        path=opts.train_dir,
    )
    datamodule_kwargs.val_dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        use_labels=opts.cond,
        path=opts.val_dir,
    )
    datamodule_kwargs.test_dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        use_labels=opts.cond,
        path=opts.test_dir,
    )
    datamodule_kwargs.batch_size = opts.batch_size

    # Active learning options
    active_learning_kwargs.num_samples = opts.num_samples
    active_learning_kwargs.strategy = opts.strategy

    # Training options
    if opts.model == "resnet50":
        trainer_kwargs.network_kwargs = dnnlib.EasyDict(
            class_name="training.resnet.ResNet",
            block="Bottleneck",
            num_blocks=[3, 4, 6, 3],
        )
    elif opts.model == "resnet18":
        trainer_kwargs.network_kwargs = dnnlib.EasyDict(
            class_name="training.resnet.ResNet",
            block="BasicBlock",
            num_blocks=[2, 2, 2, 2],
        )
    elif opts.model == "wrn":
        trainer_kwargs.network_kwargs = dnnlib.EasyDict(
            class_name="training.wrn.WideResNet",
            depth=opts.depth,
            width_factor=opts.width_factor,
            dropout_rate=opts.dropout_rate,
            use_bn=opts.use_bn,
        )

    if opts.optimizer == "adam":
        trainer_kwargs.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.AdamW", lr=opts.lr, weight_decay=0.0)
    elif opts.optimizer == "sgd":
        trainer_kwargs.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.SGD", lr=opts.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {opts.optimizer}")

    trainer_kwargs.num_epochs = opts.num_epochs
    trainer_kwargs.train_on_latents = opts.train_on_latents
    trainer_kwargs.warmup_steps = opts.warmup_steps
    trainer_kwargs.seed = opts.seed
    trainer_kwargs.resume_from = opts.resume_from
    trainer_kwargs.run_dir = generate_run_id(opts)

    # DataModule options
    print_fn("DataModule options:")
    print_fn(f"Dataset path:            {datamodule_kwargs.dataset_kwargs.path}")
    print_fn(f"Validation path:         {datamodule_kwargs.val_dataset_kwargs.path}")
    print_fn(f"Test path:               {datamodule_kwargs.test_dataset_kwargs.path}")
    print_fn(f"Batch size:              {datamodule_kwargs.batch_size}")
    print_fn()

    # Active learning options
    print_fn("Active learning options:")
    print_fn(f"Number of samples:       {active_learning_kwargs.num_samples}")
    print_fn(f"Strategy:                {active_learning_kwargs.strategy}")
    print_fn()

    # Trainer options
    print_fn("Training options:")
    print_fn(f"Output directory:        {trainer_kwargs.run_dir}")
    print_fn(f"Epochs:                  {trainer_kwargs.num_epochs}")
    print_fn(f"Resume from:             {trainer_kwargs.resume_from}")
    print_fn(f"Random seed:             {trainer_kwargs.seed}")
    print_fn(f"Number of GPUs:          {accelerator.num_processes}")
    print_fn()

    # Network options
    print_fn("Network options:")
    print_fn(f"Model:                   {trainer_kwargs.network_kwargs.class_name}")
    print_fn(f"Optimizer:               {trainer_kwargs.optimizer_kwargs.class_name}")
    print_fn()

    print_fn("Creating output directory...")
    if accelerator.is_main_process:
        create_output_directory(trainer_kwargs, datamodule_kwargs, active_learning_kwargs)

    datamodule = WRNDataModule(**datamodule_kwargs)

    match opts.exp_type:
        case "active":
            trainer = ActiveLearningTrainer(datamodule=datamodule, **active_learning_kwargs)
            trainer.run_loop("wrn", opts.eval_interval, **trainer_kwargs)
        case "baseline":
            trainer = WRNTrainer(**trainer_kwargs, datamodule=datamodule)
            trainer.fit(eval_interval=opts.eval_interval)
        case _:
            raise NotImplementedError(f"Experiment type {opts.exp_type} not implemented.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
