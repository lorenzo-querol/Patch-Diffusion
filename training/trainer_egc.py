import copy
import json
import math
import os

import numpy as np
import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from diffusers import AutoencoderKL
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dnnlib

from .ece import ECELoss
from .patch import get_batch_data, get_patches
from .utils import cycle

accelerator = Accelerator()


class Trainer:
    """Trainer for EGC."""

    def __init__(
        self,
        run_dir: str,
        dataset_kwargs,
        val_dataset_kwargs,
        test_dataset_kwargs,
        network_kwargs,
        diffusion_kwargs,
        optimizer_kwargs,
        num_steps: int = 100000,
        accum_steps: int = 1,
        batch_size: int = 128,
        ce_weight: float = 0.001,
        real_p: float = 0.5,
        target: str = "epsilon",
        train_on_latents: bool = False,
        seed: int = 1,
        resume_from: str = None,
    ):
        """
        Args:
            run_dir (`str`): Output directory.
            dataset_kwargs (`dict`): Training dataset options.
            val_dataset_kwargs (`dict`): Validation dataset options.
            test_dataset_kwargs (`dict`): Test dataset options.
            network_kwargs (`dict`): Model options.
            diffusion_kwargs (`dict`): Diffusion options.
            optimizer_kwargs (`dict`): Optimizer options.
            num_steps (`int`): Number of training steps.
            accum_steps (`int`): Accumulate gradients over multiple steps.
            batch_size (`int`): Batch size.
            ce_weight (`float`): Weight of the classification loss.
            real_p (`float`): Probability of full-sized or patch images.
            target (`str`): Target for the diffusion model.
            train_on_latents (`bool`): Train on latent representations.
            seed (`int`): Seed for reproducibility.
            resume_from (`str`): Checkpoint to resume from.
        """
        self.run_dir = run_dir

        self.dataset_kwargs = dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.test_dataset_kwargs = test_dataset_kwargs
        self.network_kwargs = network_kwargs
        self.diffusion_kwargs = diffusion_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.num_steps = num_steps
        self.accum_steps = accum_steps
        self.batch_size = batch_size
        self.ce_weight = ce_weight
        self.real_p = real_p
        self.target = target
        self.train_on_latents = train_on_latents
        self.seed = seed
        self.resume_from = resume_from

        self.ece = ECELoss(n_bins=10)

        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(dispatch_batches=True, split_batches=False),
            log_with="wandb",
            gradient_accumulation_steps=self.accum_steps,
        )
        self.device = self.accelerator.device
        self.print_fn = self.accelerator.print

        self.per_device_batch_size = self._calculate_per_device_batch_size()
        self.dataloader_kwargs = dict(
            batch_size=self.per_device_batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.img_vae = None
        self.latent_scale_factor = 0.18215

        self._init_trainer()

    def _calculate_per_device_batch_size(self):
        """Calculate the batch size per device."""
        world_size = self.accelerator.num_processes
        per_device_batch_size = self.batch_size // (world_size * self.accelerator.gradient_accumulation_steps)
        assert per_device_batch_size * world_size * self.accelerator.gradient_accumulation_steps == self.batch_size, "Batch size must be divisible by num_processes * gradient_accumulation_steps."
        return per_device_batch_size

    def _init_trainer(self):
        self._init_env()
        self._prepare_dataloaders()
        self._build_network_and_diffusion()
        self._prepare_patch_info()
        self._load()

        # NOTE: We need to initialize the diffusion model after loading the checkpoint if it exists
        self.diffusion = dnnlib.util.construct_class_by_name(**self.diffusion_kwargs, model=self.accelerator.unwrap_model(self.net))
        self.diffusion = self.accelerator.prepare(self.diffusion)

    def _init_env(self):
        """Sets seeds and other environment variables."""
        set_seed(self.seed)
        torch.backends.cudnn.benchmark = True

    def _prepare_dataloaders(self):
        """Prepare datasets and dataloaders."""
        self.print_fn("Loading datasets...")

        dataset_obj = dnnlib.util.construct_class_by_name(**self.dataset_kwargs)
        self.img_resolution, self.img_channels, self.label_dim = (
            dataset_obj.resolution,
            dataset_obj.num_channels,
            dataset_obj.label_dim,
        )
        del dataset_obj

        augment_transform = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(self.img_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * self.img_channels, std=[0.5] * self.img_channels),
            ]
        )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * self.img_channels, std=[0.5] * self.img_channels),
            ]
        )

        self.train_dataset = dnnlib.util.construct_class_by_name(**self.dataset_kwargs, transform=transform)
        self.cls_dataset = dnnlib.util.construct_class_by_name(**self.dataset_kwargs, transform=augment_transform)
        self.val_dataset = dnnlib.util.construct_class_by_name(**self.val_dataset_kwargs, transform=transform)
        self.test_dataset = dnnlib.util.construct_class_by_name(**self.test_dataset_kwargs, transform=transform)

        self.train_dataloader = DataLoader(self.train_dataset, **self.dataloader_kwargs)
        self.cls_dataloader = DataLoader(self.cls_dataset, **self.dataloader_kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

        self.train_dataloader, self.cls_dataloader, self.val_dataloader, self.test_dataloader = self.accelerator.prepare(
            self.train_dataloader,
            self.cls_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        )

        self.train_dataloader, self.cls_dataloader = cycle(self.train_dataloader), cycle(self.cls_dataloader)

    def _set_requires_grad(self, model: torch.nn.Module, requires_grad: bool):
        """
        Set requires_grad for all parameters in the model.

        Args:
            model (torch.nn.Module): The model to set `requires_grad` for.
            requires_grad (bool): Whether to set `requires_grad` to true or false.
        """
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _prepare_patch_info(self):
        """Prepare patch information for Patch Diffusion."""
        real_p = self.real_p
        img_resolution = self.img_resolution
        is_32 = self.img_resolution == 32

        if is_32:
            batch_mul_dict = {32: 1, 16: 4}  # Simplified multipliers for 32x32
            if self.real_p < 1.0:
                p_list = np.array([(1 - real_p), real_p])
                patch_list = np.array([16, 32])  # Only two patch sizes for 32x32
            else:
                p_list = np.array([0, 1.0])
                patch_list = np.array([16, 32])
        else:
            """Default options for Patch Diffusion"""
            batch_mul_dict = {512: 1, 256: 2, 128: 4, 64: 16, 32: 32, 16: 64}

            if self.train_on_latents:
                p_list = np.array([(1 - real_p), real_p])
                patch_list = np.array([img_resolution // 2, img_resolution])
            else:
                p_list = np.array([(1 - real_p) * 2 / 5, (1 - real_p) * 3 / 5, real_p])
                patch_list = np.array([img_resolution // 4, img_resolution // 2, img_resolution])

        self.p_list = p_list
        self.patch_list = patch_list
        self.batch_mul_dict = batch_mul_dict

    def _build_network_and_diffusion(self):
        """Setup network and diffusion."""
        self.print_fn("Setting up network and diffusion...")

        if self.train_on_latents:
            self.img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.img_vae.eval()
            self._set_requires_grad(self.img_vae, False)
            self.img_resolution, self.img_channels = self.img_resolution // 8, 4

        attention_ds = []
        for res in self.network_kwargs["attn_resolutions"]:
            attention_ds.append(self.img_resolution // int(res))

        self.network_kwargs.update(
            {
                "attn_resolutions": tuple(attention_ds),
                "img_resolution": self.img_resolution,
                "in_channels": self.img_channels + 2,
                "out_channels": self.label_dim,
                "label_dim": self.label_dim,
            }
        )
        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs)

        sampler_kwargs = copy.deepcopy(self.diffusion_kwargs)
        sampler_kwargs.update({"class_name": "training.diffusion.DDIMSampler"})

        if self.accelerator.is_main_process:
            with open(os.path.join(self.run_dir, "network_kwargs.json"), "w") as f:
                json.dump(self.network_kwargs, f)

            with open(os.path.join(self.run_dir, "sampler_kwargs.json"), "w") as f:
                json.dump(sampler_kwargs, f)

        # Setup EMA
        self.print_fn("Setting up EMA...")
        if self.accelerator.is_main_process:
            self.ema = EMA(self.net, beta=0.9999, update_every=1, power=3 / 4)
            self.sampler = dnnlib.util.construct_class_by_name(**sampler_kwargs, model=self.ema.ema_model)

        # Setup the optimizer
        self.print_fn("Setting up optimizer...")
        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)

        # Prepare for distributed training
        self.net, self.optimizer = self.accelerator.prepare(self.net, self.optimizer)

        # NOTE (BUG): Accelerate's prepare resets the model to NOT require gradients, so we need to set it back to true!
        self._set_requires_grad(self.net, True)

    @accelerator.on_main_process
    def _update_ema(self):
        """Update the EMA model."""
        self.ema.to(self.device)
        self.ema.update()

    @accelerator.on_main_process
    def _sample_images(self, num_images=64):
        """
        Sample images from the EMA model and save them.

        Args:
            num_images (`int`, optional, defaults to `64`): Number of images to sample.
        """
        self.ema.ema_model.eval()

        x_start, y_start, resolution, image_size = (0, 0, self.img_resolution, self.img_resolution)
        x_pos = torch.arange(x_start, x_start + image_size).view(1, -1).repeat(image_size, 1)
        y_pos = torch.arange(y_start, y_start + image_size).view(-1, 1).repeat(1, image_size)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.0
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.0

        pos = torch.stack([x_pos, y_pos], dim=0).to(self.device)
        pos = pos.unsqueeze(0).repeat(num_images, 1, 1, 1)
        shape = (num_images, self.img_channels, self.img_resolution, self.img_resolution)
        x_0 = torch.randn(shape, device=self.device)
        class_labels = torch.randint(0, self.label_dim, (num_images,), device=self.device)

        samples = self.sampler(x_0, pos, class_labels, steps=10)

        if self.train_on_latents:
            samples = self._decode_latents(samples)

        image_grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(num_images)), normalize=True, scale_each=True)
        fname = os.path.join(self.run_dir, f"sample-{self.cur_step}.png")
        torchvision.utils.save_image(image_grid, fname)

    def _decode_latents(self, latents: torch.Tensor):
        """
        Decode the given latents to images.

        Args:
            latents (`torch.Tensor`): The latents to decode.

        Returns:
            The decoded images.
        """
        latents = 1 / self.latent_scale_factor * latents
        images = self.img_vae.decode(latents.float()).sample

        return images

    def _encode_latents(self, images: torch.Tensor):
        """
        Encode the given images to compressed latent space.

        Args:
            images (`torch.Tensor`): The images to encode.

        Returns:
            The encoded latents.
        """
        with torch.no_grad():
            images = self.img_vae.encode(images)["latent_dist"].sample()
            latents = self.latent_scale_factor * images

        return latents

    def train(self, log_interval: int, eval_interval: int, save_interval: int):
        """
        Main training loop.

        Args:
            log_interval (`int`): When to log the metrics.
            eval_interval (`int`): When to evaluate the model.
            save_interval (`int`): When to save the model.
        """

        self.print_fn(f"Training for {self.num_steps - self.cur_step} steps...")
        self.print_fn("")

        self.accelerator.init_trackers(project_name="EGC")

        self.cur_step = 0
        for step in range(self.num_steps):
            self.cur_step = step

            metrics = self._training_step()

            if self.cur_step % log_interval == 0:
                self.print_fn("")
                self.print_fn(f"Step {self.cur_step}/{self.num_steps}")
                self._report_metrics(metrics)

            if eval_interval > 0 and self.cur_step % eval_interval == 0:
                metrics = self.evaluate(self.net, self.val_dataloader)
                self._report_metrics(metrics)

            if save_interval > 0 and self.cur_step % save_interval == 0:
                self._save(f"model-{self.cur_step}")
                self._sample_images()

        self.cur_step = self.num_steps
        metrics = self.evaluate(self.net, self.val_dataloader)
        self._report_metrics(metrics)
        self._save("model-final")
        self._sample_images()

    def _training_step(self):
        """Perform a single training step."""
        metrics = {}

        self.optimizer.zero_grad(set_to_none=True)

        if self.ce_weight > 0:
            cls_images, cls_labels = next(self.cls_dataloader)

            if self.train_on_latents:
                cls_images = self._encode_latents(cls_images)

            cls_images, cls_labels = get_patches(cls_images, self.img_resolution), torch.cat([cls_labels, cls_labels]).argmax(dim=1)

            with self.accelerator.no_sync(self.net):
                logits, ce_loss, weighted_ce_loss = self.diffusion(cls_images, cls_labels, cls_mode=True)
                acc = (logits.argmax(dim=1) == cls_labels).float().mean()
                ece = self.ece(logits, cls_labels)

                metrics["cls_loss"] = ce_loss
                metrics["cls_acc"] = acc
                metrics["cls_ece"] = ece

                self.accelerator.backward(self.ce_weight * weighted_ce_loss)

        patch_size = int(np.random.choice(self.patch_list, p=self.p_list))
        batch_mul = self.batch_mul_dict[patch_size] // self.batch_mul_dict[self.img_resolution]

        images, labels = get_batch_data(self.train_dataloader, batch_mul)
        images, labels = images.to(self.device), labels.to(self.device)

        if self.train_on_latents:
            images = self._encode_latents(images)

        images, labels = get_patches(images, patch_size), labels.argmax(dim=1)

        mse_loss = self.diffusion(images, labels)
        metrics["mse_loss"] = mse_loss

        self.accelerator.backward(mse_loss / batch_mul)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)

        self.accelerator.wait_for_everyone()
        self.optimizer.step()

        grad_norm, param_norm = self._compute_norms()
        metrics["grad_norm"] = grad_norm
        metrics["param_norm"] = param_norm
        metrics["lr"] = torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device)

        self._update_ema()

        return metrics

    @torch.no_grad()
    def evaluate(self, net: torch.nn.Module, dataloader: DataLoader):
        """
        Evaluate the model on the given dataloader.

        Args:
            net (`torch.nn.Module`): The model to evaluate.
            dataloader (`DataLoader`): A validation dataloader.

        Returns:
            The computed metrics.
        """
        dataloader = self.accelerator.prepare(dataloader)
        metrics = {"val_cls_loss": [], "val_cls_acc": [], "val_cls_ece": []}

        net.eval()
        for images, labels in dataloader:
            labels = labels.argmax(dim=1)

            if self.train_on_latents:
                images = self._encode_latents(images)

            images = get_patches(images, self.img_resolution)
            clean_timesteps = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)

            logits = net(images, clean_timesteps, cls_mode=True)

            metrics["val_cls_loss"].append(torch.nn.functional.cross_entropy(logits, labels))
            metrics["val_cls_acc"].append((logits.argmax(dim=1) == labels).float().mean())
            metrics["val_cls_ece"].append(self.ece(logits, labels))

        net.train()

        metrics = {k: torch.stack(v).mean() for k, v in metrics.items()}

        return metrics

    @torch.no_grad()
    def get_probs(self, net: torch.nn.Module, dataloader: DataLoader):
        """
        Get the computed probabilities.

        Args:
            net (`torch.nn.Module`): The model to use for computing probabilities.
            dataloader (`DataLoader`): The dataloader to use for computing probabilities.

        Returns:
            The computed probabilities.
        """
        dataloader = self.accelerator.prepare(dataloader)
        probs = []

        net.eval()

        with tqdm(total=len(dataloader), desc="Computing Probabilities", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for images, _ in dataloader:
                pbar.update(1)

                if self.train_on_latents:
                    images = self._encode_latents(images)

                images = get_patches(images, self.img_resolution)
                clean_timesteps = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)

                logits = net(images, clean_timesteps, cls_mode=True)
                prob = logits.softmax(dim=1)
                probs.append(prob)

            pbar.close()

        net.train()

        return torch.cat(probs, dim=0)

    def _compute_norms(self):
        """Compute the gradient and parameter norms."""
        grad_norm = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm(2) ** 2
        grad_norm = grad_norm**0.5

        param_norm = 0.0
        for p in self.net.parameters():
            param_norm += p.norm(2) ** 2
        param_norm = param_norm**0.5

        return grad_norm, param_norm

    def _report_metrics(self, metrics: dict):
        """
        Log the metrics and print them.

        Args:
            metrics (`dict`): The metrics to log and print.
        """
        global_metrics = {}
        for name, value in metrics.items():
            gathered_values = self.accelerator.gather_for_metrics(value)
            if self.accelerator.is_main_process:
                global_avg = gathered_values.mean().item()
                global_metrics[name] = global_avg
                self.print_fn(f"{name} = {global_avg:.6f}")

        self.accelerator.log(global_metrics, step=self.cur_step)

    @accelerator.on_main_process
    def _save(self, filename: str):
        """
        Save a checkpoint file.

        Args:
            filename (`str`): The name of the file to save.
        """
        data = {
            "step": self.cur_step,
            "net": self.accelerator.unwrap_model(self.net).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
        }
        self.print_fn(f"Saving checkpoint to {filename}...")
        torch.save(data, os.path.join(self.run_dir, f"{filename}.pt"))

    def _load(self):
        """
        Load the latest checkpoint specified by `self.resume_from`.\n

        If `self.resume_from` is a directory, then the latest checkpoint is loaded from that directory.\n
        If `self.resume_from` is a file, then the checkpoint is loaded from that file.\n
        If `self.resume_from` is `None`, then no checkpoint is loaded.
        """
        if self.resume_from is None:
            return

        ckpt = self.resume_from
        if self.resume_from.endswith(".pt"):
            pt_files = [f for f in os.listdir(self.resume_from) if f.endswith(".pt")]
            ckpt = os.path.join(self.resume_from, pt_files[-1])

        self.print_fn(f"Resuming from {ckpt}...")
        data = torch.load(ckpt, weights_only=True)

        self.cur_step = data["step"]
        self.accelerator.unwrap_model(self.net).load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
            self.ema.to(self.device)
