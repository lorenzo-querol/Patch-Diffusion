import copy
import json
import os
import pickle
import time

import numpy as np
import torch
from diffusers import AutoencoderKL
from calibration.ece import ECE
from training.resample import UniformSampler
from .patch import get_patches
from .utils import evaluate, initialize, set_requires_grad
from torch.utils.data import DataLoader
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc, training_stats
from torchvision import transforms


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def get_patch_size_list(img_resolution, real_p, train_on_latents):
    is_smaller_than_64 = img_resolution == 32

    if is_smaller_than_64:
        batch_mul_dict = {32: 1, 16: 4}  # Simplified multipliers for 32x32
        if real_p < 1.0:
            p_list = np.array([(1 - real_p), real_p])
            patch_list = np.array([16, 32])  # Only two patch sizes for 32x32
            batch_mul_avg = np.sum(p_list * np.array([4, 1]))
        else:
            p_list = np.array([0, 1.0])
            patch_list = np.array([16, 32])
            batch_mul_avg = 1
    else:
        batch_mul_dict = {512: 1, 256: 2, 128: 4, 64: 16, 32: 32, 16: 64}
        if train_on_latents:
            p_list = np.array([(1 - real_p), real_p])
            patch_list = np.array([img_resolution // 2, img_resolution])
            batch_mul_avg = np.sum(p_list * np.array([2, 1]))
        else:
            p_list = np.array([(1 - real_p) * 2 / 5, (1 - real_p) * 3 / 5, real_p])
            patch_list = np.array([img_resolution // 4, img_resolution // 2, img_resolution])
            batch_mul_avg = np.sum(np.array(p_list) * np.array([4, 2, 1]))  # 2

    return p_list, patch_list, batch_mul_dict, batch_mul_avg


def encode_images_to_latents(img_vae, images, latent_scale_factor: float, train_on_latents: bool):
    """Encode images to latents using the given VAE. Return latents if train_on_latents is True, otherwise the input images.
    Args:
        img_vae: VAE model.
        images: Input images.
        latent_scale_factor (float): Scaling factor for latents.
        train_on_latents (bool): Whether to train on latents.
    Returns:
        Latents if train_on_latents is True, otherwise the input images.
    """
    if train_on_latents:
        assert img_vae is not None, "img_vae must be provided when train_on_latents is True."
        with torch.no_grad():
            images = img_vae.encode(images)["latent_dist"].sample()
            images = latent_scale_factor * images
    return images


def get_batch_data(dataset_iterator, device, batch_mul=1):
    """Get a batch of data from the dataset iterator.

    :param dataset_iterator: The dataset iterator.
    :param device: The device to move the data to.
    :param batch_mul: The number of batches to get from the iterator. Default is 1.
    """
    images, labels = [], []

    for _ in range(batch_mul):
        images_, labels_ = next(dataset_iterator)
        images.append(images_), labels.append(labels_)
    images, labels = torch.cat(images).to(device), torch.cat(labels).to(device)
    return images, labels


def fit(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    val_dataset_kwargs={},  # Options for validation set.
    dataloader_kwargs={},  # Options for dataloader.
    network_kwargs={},  # Options for model
    diffusion_kwargs={},  # Options for diffusion
    optimizer_kwargs={},  # Options for optimizer.
    ce_weight=1.0,  # Weight of the classification loss.
    label_smooth=0.0,  # Label smoothing factor.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    real_p=0.5,
    train_on_latents=False,
    device=torch.device("cuda"),
):
    # Initialize.
    start_time = time.time()
    initialize(seed, cudnn_benchmark)

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total

    num_accumulation_rounds = batch_gpu_total // batch_gpu

    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    img_resolution, img_channels = dataset_obj.resolution, dataset_obj.num_channels
    del dataset_obj

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * img_channels, std=[0.5] * img_channels),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * img_channels, std=[0.5] * img_channels),
        ]
    )

    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs, transform=val_transform)
    dataloader = cycle(DataLoader(dataset=dataset_obj, batch_size=batch_gpu, drop_last=True, **dataloader_kwargs))

    cls_dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs, transform=train_transform)
    cls_dataloader = cycle(DataLoader(dataset=cls_dataset_obj, batch_size=batch_gpu, drop_last=True, **dataloader_kwargs))

    val_dataset_obj = dnnlib.util.construct_class_by_name(**val_dataset_kwargs, transform=val_transform)
    val_dataloader = DataLoader(dataset=val_dataset_obj, batch_size=batch_gpu, **dataloader_kwargs)

    # if train_on_latents:
    #     img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    #     img_vae.eval()
    #     set_requires_grad(img_vae, False)
    #     latent_scale_factor = 0.18215
    #     img_resolution, img_channels = dataset_obj.resolution // 8, 4
    # else:
    #     img_vae = None

    # Construct network
    dist.print0("Constructing network...")
    net_input_channels = img_channels + 2
    interface_kwargs = dict(
        img_resolution=img_resolution,
        in_channels=net_input_channels,
        out_channels=dataset_obj.label_dim,
        label_dim=dataset_obj.label_dim,
    )
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    diffusion = dnnlib.util.construct_class_by_name(**diffusion_kwargs)  # subclass of training.diffusion.GaussianDiffusion

    # Setup optimizer
    dist.print0("Setting up optimizer...")
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)

    # Distribute
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=ema, require_all=False)
        del data  # conserve memory

    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(src_module=data["net"], dst_module=net, require_all=True)
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    p_list, patch_list, batch_mul_dict, batch_mul_avg = get_patch_size_list(img_resolution, real_p, train_on_latents)

    # Classification-related
    criterion = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smooth)
    ece_criterion = ECE()

    # Schedule sampler
    schedule_sampler = UniformSampler(diffusion)

    while True:
        optimizer.zero_grad(set_to_none=True)

        L = 0

        if ce_weight > 0:
            # Return full-size images and positions
            cls_images, cls_labels = get_batch_data(cls_dataloader, device)
            patches, x_pos = get_patches(cls_images, img_resolution)
            x_in = torch.cat([patches, x_pos], dim=1)

            timesteps, weights = schedule_sampler.sample(x_in.shape[0], device)
            sqrt_alphas_cumprod = torch.from_numpy(diffusion.sqrt_alphas_cumprod).to(device)[timesteps].float()

            logits = ddp(x_in, timesteps, class_labels=cls_labels, cls_mode=True)

            cls_labels = cls_labels.argmax(dim=1)
            ce_loss = criterion(logits, cls_labels)
            cls_loss = ce_weight * (ce_loss * sqrt_alphas_cumprod).mean()
            cls_acc = (logits.argmax(dim=1) == cls_labels).float().mean()

            # ECE computation
            probs = torch.nn.functional.softmax(logits, dim=1)
            cls_ece = ece_criterion.measure(probs.cpu().detach().numpy(), cls_labels.cpu().detach().numpy())

            training_stats.report("train/cls_acc", cls_acc)
            training_stats.report("train/cls_loss", cls_loss)
            training_stats.report("train/cls_ece", cls_ece)

            L += cls_loss.sum().mul(loss_scaling / batch_gpu_total)

        # Return patch-size images and positions
        patch_size = int(np.random.choice(patch_list, p=p_list))
        batch_mul = batch_mul_dict[patch_size] // batch_mul_dict[img_resolution]
        images, _ = get_batch_data(dataloader, device, batch_mul=batch_mul)
        # images = encode_images_to_latents(train_on_latents, img_vae, latent_scale_factor, images)
        patches, x_pos = get_patches(images, patch_size)
        x_in = torch.cat([patches, x_pos], dim=1)

        timesteps, weights = schedule_sampler.sample(x_in.shape[0], device)
        mse_loss = diffusion.training_losses(net=ddp, x_start=x_in, t=timesteps)
        mse_loss = (mse_loss * weights).mean()

        training_stats.report("train/loss", mse_loss)

        L += mse_loss.sum().mul(loss_scaling / batch_gpu_total / batch_mul)

        L.backward()

        # Update weights
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        optimizer.step()

        # Update EMA
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)

        ema_beta = 0.5 ** (batch_size * batch_mul_avg / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += int(batch_size * batch_mul_avg)
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Evaluate on validation set
        if ce_weight > 0 and cur_tick % 5 == 0:
            val_loss, val_acc, val_ece = evaluate(ddp, val_dataloader, device)

            training_stats.report("val/cls_loss", val_loss)
            training_stats.report("val/cls_acc", val_acc)
            training_stats.report("val/cls_ece", val_ece)

            fields = []
            fields += [f"val_loss {training_stats.report0('val/cls_loss', val_loss):<5.5f}"]
            fields += [f"val_acc {training_stats.report0('val/cls_acc', val_acc):<5.5f}"]
            fields += [f"val_ece {training_stats.report0('val/cls_ece', val_ece):<5.5f}"]
            dist.print0("\t".join(fields))
            dist.print0("")

        # Print status line, accumulating the same information in training_stats
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"loss {mse_loss.mean().item():<5.5f}"]

        if ce_weight > 0:
            fields += [f"cls_loss {cls_loss.mean().item():<5.5f}"]
            fields += [f"cls_acc {cls_acc.item():<5.5f}"]
            fields += [f"cls_ece {cls_ece:<5.5f}"]

        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        dist.print0("\n".join(fields))
        dist.print0("")

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl"), "wb") as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f"training-state-{cur_nimg//1000:06d}.pt"))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + "\n")
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()

        if done:
            break

    # Done.
    dist.print0()
    dist.print0("Exiting...")


# ----------------------------------------------------------------------------
