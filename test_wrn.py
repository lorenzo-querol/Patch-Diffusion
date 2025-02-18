import json
import os

import click

import dnnlib
from training.testers import WRNTester, EGCTester


@click.command()
@click.option("--model_type", help="Type of the model", metavar="STR", type=str, required=True)
@click.option("--outdir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--data_dir", help="Path to the train dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--ckpt_dir", help="Path to checkpoints", metavar="ZIP|DIR", type=str, required=True)
@click.option("--ckpt_type", help="Type of checkpoint to test", metavar="STR", type=str, default="final")
@click.option("--train_on_latents", help="Train on latents", type=bool, default=False)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    tester_kwargs = dnnlib.EasyDict()

    tester_kwargs.test_dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        use_labels=True,
        path=opts.data_dir,
    )

    with open(os.path.join(opts.ckpt_dir, "network_kwargs.json"), "r") as f:
        network_kwargs = json.load(f)

    tester_kwargs.network_kwargs = network_kwargs
    tester_kwargs.outdir = opts.outdir
    tester_kwargs.ckpt_dir = opts.ckpt_dir
    tester_kwargs.ckpt_type = opts.ckpt_type
    tester_kwargs.train_on_latents = opts.train_on_latents

    print()
    print("Testing options:")
    print(json.dumps(tester_kwargs, indent=2))
    print()
    print(f"Output directory:        {tester_kwargs.outdir}")
    print(f"Test path:               {tester_kwargs.test_dataset_kwargs.path}")
    print(f"Checkpoint directory:    {tester_kwargs.ckpt_dir}")
    print(f"Checkpoint type:         {opts.ckpt_type}")
    print(f"Train on latents:        {tester_kwargs.train_on_latents}")
    print()

    ckpt_list = [os.path.join(root, file) for root, _, files in os.walk(tester_kwargs.ckpt_dir) for file in files if file.endswith(".pt")]
    ckpt_list = [ckpt for ckpt in ckpt_list if opts.ckpt_type in ckpt]

    match opts.model_type:
        case "wrn":
            tester = WRNTester(**tester_kwargs)
            tester.test(ckpt_list)
        case "egc":
            tester = EGCTester(**tester_kwargs)
            tester.test(ckpt_list)
        case _:
            raise ValueError(f"Unknown model type: {opts.model_type}")


if __name__ == "__main__":
    main()
