import re
import torch
import dnnlib
import click


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
@click.option("--out_dir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--train_dir", help="Path to the train dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--model_dir", help="Path to the model directory", metavar="DIR", type=str, required=True)
@click.option("--cond", help="Train class-conditional model", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--num_samples", help="Number of samples", metavar="INT", type=int, default=1)
@click.option("--classifier_scale", help="Classifier scale", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True)

# Hyperparameters.
@click.option("--duration", help="Training duration", metavar="MIMG", type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option("--batch_size", help="Total batch size", metavar="INT", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--channel_mult", help="Channel multiplier  [default: varies]", metavar="LIST", type=parse_int_list)
@click.option("--model_channels", help="Channels per resolution  [default: varies]", metavar="INT", type=int)
@click.option("--num_res_blocks", help="Number of residual blocks", metavar="INT", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--attn_resolutions", help="Resolutions to use attention layers", metavar="LIST", type=parse_int_list)
@click.option("--dropout_rate", help="Dropout rate", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)

# Diffusion-related.
@click.option("--schedule_name", help="Diffusion schedule", metavar="str", type=click.Choice(["linear", "cosine"]), show_default=True)
@click.option("--timesteps", help="Number of diffusion timesteps", metavar="INT", type=click.IntRange(min=1), default=1000, show_default=True)

# I/O-related.
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int, default=1)
def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts = dnnlib.EasyDict(kwargs)
    sampler_kwargs = dnnlib.EasyDict()

    sampler_kwargs.dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", use_labels=opts.cond, path=opts.train_dir)
    sampler_kwargs.network_kwargs = dnnlib.EasyDict(
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
    sampler_kwargs.diffusion_kwargs = dnnlib.EasyDict(
        class_name="training.diffusion.GaussianDiffusion",
        schedule_name=opts.schedule_name,
        timesteps=opts.timesteps,
    )

    dataset_obj = dnnlib.util.construct_class_by_name(**sampler_kwargs.dataset_kwargs)
    img_resolution, img_channels, label_dim = (
        dataset_obj.resolution,
        dataset_obj.num_channels,
        dataset_obj.label_dim,
    )
    del dataset_obj

    interface_kwargs = dnnlib.EasyDict(
        img_resolution=img_resolution,
        in_channels=img_channels,
        out_channels=label_dim,
    )
    sampler_kwargs.network_kwargs.update(interface_kwargs)

    sampler_kwargs.batch_size = opts.batch_size

    model = dnnlib.util.construct_class_by_name(**sampler_kwargs.network_kwargs)
    diffusion = dnnlib.util.construct_class_by_name(**sampler_kwargs.diffusion_kwargs)

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = model(x_in, t, cls_mode=True)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = torch.autograd.grad(selected.sum(), x_in)[0] * opts.classifier_scale
            return grad

    all_images = []
    while len(all_images) * opts.batch_size < opts.num_samples:
        model_kwargs = {}

        if opts.cond:
            class_labels = torch.randint(low=0, high=label_dim, size=(opts.batch_size,), device=device)
            model_kwargs["class_labels"] = class_labels

        sample = diffusion.p_sample_loop(
            model,
            (opts.batch_size, img_channels, img_resolution, img_resolution),
            model_kwargs=model_kwargs,
            cond_fn=cond_fn if opts.cond else None,
        )


if __name__ == "__main__":
    main()
