import json
import os

import click

import dnnlib
from training.tester_wrn import Tester


@click.command()
@click.option("--outdir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--test_dir", help="Path to the train dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--batch_size", help="Total batch size", metavar="INT", type=click.IntRange(min=1), default=128, show_default=True)
@click.option("--cond", help="Train class-conditional model", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int, default=1)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    tester_kwargs = dnnlib.EasyDict()

    tester_kwargs.test_dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        use_labels=opts.cond,
        path=opts.test_dir,
    )

    with open(os.path.join(opts.outdir, "network_kwargs.json"), "r") as f:
        network_kwargs = json.load(f)

    tester_kwargs.network_kwargs = network_kwargs
    tester_kwargs.batch_size = opts.batch_size
    tester_kwargs.seed = opts.seed
    tester_kwargs.outdir = opts.outdir

    print()
    print("Testing options:")
    print(json.dumps(tester_kwargs, indent=2))
    print()
    print(f"Output directory:        {tester_kwargs.outdir}")
    print(f"Test Dataset path:       {tester_kwargs.test_dataset_kwargs.path}")
    print(f"Batch size:              {tester_kwargs.batch_size}")
    print(f"Random seed:             {tester_kwargs.seed}")
    print()

    ckpt_list = [os.path.join(root, file) for root, _, files in os.walk(tester_kwargs.outdir) for file in files if file.endswith(".pt")]
    tester = Tester(**tester_kwargs)
    tester.test(ckpt_list)


if __name__ == "__main__":
    main()
