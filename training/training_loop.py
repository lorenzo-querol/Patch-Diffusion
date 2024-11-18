# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import copy
import json
import os
import pickle
import time

import numpy as np
import torch
from diffusers import AutoencoderKL

import dnnlib
from calibration.ece import ECELoss
from torch_utils import distributed as dist
from torch_utils import misc, training_stats


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def eval_cls(net, val_dataloader, loss_fn, device):
    losses, accs, eces = [], [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_dataloader):
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)

            logits, ce_loss = loss_fn(
                net=net,
                images=images,
                patch_size=images.shape[-1],
                labels=labels,
                cls_mode=True,
                eval_mode=True,
            )

            acc = (logits.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
            ece = ECELoss()(logits, labels.argmax(dim=1))

            losses.append(ce_loss.mean().item())
            accs.append(acc.item())
            eces.append(ece.item())

    return np.mean(losses), np.mean(accs), np.mean(eces)


def initialize(seed: int, cudnn_benchmark: bool):
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def training_loop(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    val_dataset_kwargs={},  # Options for valid set. (NEW!!)
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
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
    progressive=False,
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
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    val_dataset_obj = dnnlib.util.construct_class_by_name(**val_dataset_kwargs)  # subclass of training.dataset.Dataset
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset_obj, batch_size=batch_gpu, **data_loader_kwargs)

    img_resolution, img_channels = dataset_obj.resolution, dataset_obj.num_channels

    if train_on_latents:
        img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        img_vae.eval()
        set_requires_grad(img_vae, False)
        latent_scale_factor = 0.18215
        img_resolution, img_channels = dataset_obj.resolution // 8, 4
    else:
        img_vae = None

    # Construct network.
    dist.print0("Constructing network...")
    net_input_channels = img_channels + 2
    interface_kwargs = dict(
        img_resolution=img_resolution,
        img_channels=net_input_channels,
        out_channels=dataset_obj.label_dim,
        label_dim=dataset_obj.label_dim,
    )

    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            x_pos = torch.zeros([batch_gpu, 2, net.img_resolution, net.img_resolution], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, x_pos, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0("Setting up optimizer...")
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss
    ece_fn = ECELoss()
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)  # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None  # training.augment.AugmentPipe
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

    # Train.
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

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

    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            patch_size = int(np.random.choice(patch_list, p=p_list))
            batch_mul = batch_mul_dict[patch_size] // batch_mul_dict[img_resolution]
            images, labels = [], []

            for _ in range(batch_mul):
                images_, labels_ = next(dataset_iterator)
                images.append(images_), labels.append(labels_)

            images, labels = torch.cat(images, dim=0), torch.cat(labels, dim=0)
            images = images.to(device).to(torch.float32) / 127.5 - 1

            if train_on_latents:
                with torch.no_grad():
                    images = img_vae.encode(images)["latent_dist"].sample()
                    images = latent_scale_factor * images

            labels = labels.to(device)

            with misc.ddp_sync(ddp, sync=False):
                ce_loss, cls_acc, cls_ece = loss_fn(net=ddp, images=images, patch_size=patch_size, labels=labels, augment_pipe=augment_pipe, cls_mode=True)
                CE_WEIGHT = 1.0
                cls_loss = CE_WEIGHT * ce_loss.mean()

                training_stats.report("train/cls_acc", cls_acc)
                training_stats.report("train/cls_loss", cls_loss.mean())
                training_stats.report("train/cls_ece", cls_ece)

                cls_loss.sum().mul(loss_scaling / batch_gpu_total / batch_mul).backward()

            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                loss = loss_fn(net=ddp, images=images, patch_size=patch_size, labels=labels, augment_pipe=augment_pipe)
                training_stats.report("train/sm_loss", loss)

                loss.sum().mul(loss_scaling / batch_gpu_total / batch_mul).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)

        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        optimizer.step()

        # Update EMA.
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

        # Evaluate validation set.
        if cur_tick % 5 == 0:
            ddp.eval()
            val_loss, val_acc, val_ece = eval_cls(net, val_dataloader, loss_fn, device)
            ddp.train()

            fields = []
            fields += [f"val_loss {training_stats.report0('val/cls_loss', val_loss):<9.5f}"]
            fields += [f"val_acc {training_stats.report0('val/cls_acc', val_acc):<5.5f}"]
            fields += [f"val_ece {training_stats.report0('val/cls_ece', val_ece):<5.5f}"]
            dist.print0(" ".join(fields))

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"sm_loss {loss.mean().item():<9.5f}"]
        fields += [f"cls_loss {cls_loss.mean().item():<9.5f}"]
        fields += [f"cls_acc {cls_acc.item():<5.5f}"]
        fields += [f"cls_ece {cls_ece.item():<5.5f}"]

        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
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
