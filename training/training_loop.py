import copy
import math
import os

import numpy as np
import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dnnlib
from calibration.ece import ECE
from training.resample import UniformSampler

from .ece import ECELoss
from .patch import get_patches
from .utils import cycle


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
    :return: A tuple of images and labels.
    """
    images, labels = [], []

    for _ in range(batch_mul):
        images_, labels_ = next(dataset_iterator)
        images.append(images_), labels.append(labels_)
    images, labels = torch.cat(images).to(device), torch.cat(labels).to(device)
    return images, labels


class Trainer:
    def __init__(
        self,
        run_dir="./training-runs",  # Output directory
        dataset_kwargs={},  # Training dataset options
        val_dataset_kwargs={},  # Validation dataset options
        network_kwargs={},  # Model options
        diffusion_kwargs={},  # Diffusion options
        optimizer_kwargs={},  # Optimizer options
        scheduler_kwargs={},  # Scheduler options
        num_steps=100,  # Number of training steps
        accum_steps=1,  # Accumulate gradients over multiple steps
        lr_warmup=0,  # Number of warmup steps for the learning rate
        batch_size=128,  # Batch size
        ce_weight=0.001,  # Weight of the classification loss
        label_smooth=0.2,  # Label smoothing factor
        real_p=0.5,  # Probability of real images
        target="epsilon",  # Target for the diffusion model
        train_on_latents=False,  # Train on latent representations
        seed=1,  # Seed for reproducibility
        resume_from=None,  # Checkpoint to resume from
    ):
        self.run_dir = run_dir
        self.dataset_kwargs = dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.network_kwargs = network_kwargs
        self.diffusion_kwargs = diffusion_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs

        self.num_steps = num_steps
        self.accum_steps = accum_steps
        self.lr_warmup = lr_warmup
        self.ce_weight = ce_weight
        self.label_smooth = label_smooth
        self.real_p = real_p
        self.target = target
        self.train_on_latents = train_on_latents
        self.seed = seed
        self.resume_from = resume_from
        self.batch_size = batch_size
        self.cur_step = 0

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=self.label_smooth)
        self.ece_criterion = ECELoss(n_bins=10)

        self.accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=self.accum_steps)
        self.device = self.accelerator.device
        self.print_fn = self.accelerator.print
        self.per_device_batch_size = self._calculate_per_device_batch_size()
        self._init_trainer()

    def _calculate_per_device_batch_size(self):
        """Calculate the batch size per device."""
        world_size = self.accelerator.num_processes
        per_device_batch_size = self.batch_size // (world_size * self.accelerator.gradient_accumulation_steps)
        assert (
            per_device_batch_size * world_size * self.accelerator.gradient_accumulation_steps == self.batch_size
        ), "Batch size must be divisible by num_processes * gradient_accumulation_steps."
        return per_device_batch_size

    def _init_trainer(self):
        """Initialize the Trainer: seeds, datasets, and network."""
        self._init_env()
        self._prepare_dataloaders()
        self._prepare_patch_info()
        self._build_network_and_diffusion()

        if self.resume_from is not None:
            self._load()

    def _init_env(self):
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
        dataloader_kwargs = dict(
            batch_size=self.per_device_batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.train_dataloader = DataLoader(self.train_dataset, **dataloader_kwargs)
        self.cls_dataloader = DataLoader(self.cls_dataset, **dataloader_kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

        self.train_dataloader, self.cls_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader,
            self.cls_dataloader,
            self.val_dataloader,
        )

        self.train_dataloader, self.cls_dataloader = (
            cycle(self.train_dataloader),
            cycle(self.cls_dataloader),
        )

    def _set_requires_grad(self, model, requires_grad):
        """Set requires_grad for all parameters in the model"""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _prepare_patch_info(self):
        real_p = self.real_p
        img_resolution = self.img_resolution
        is_smaller_than_64 = self.img_resolution == 32

        if is_smaller_than_64:
            batch_mul_dict = {32: 1, 16: 4}  # Simplified multipliers for 32x32
            if self.real_p < 1.0:
                p_list = np.array([(1 - real_p), real_p])
                patch_list = np.array([16, 32])  # Only two patch sizes for 32x32
                batch_mul_avg = np.sum(p_list * np.array([4, 1]))
            else:
                p_list = np.array([0, 1.0])
                patch_list = np.array([16, 32])
                batch_mul_avg = 1
        else:
            batch_mul_dict = {512: 1, 256: 2, 128: 4, 64: 16, 32: 32, 16: 64}
            if self.train_on_latents:
                p_list = np.array([(1 - real_p), real_p])
                patch_list = np.array([img_resolution // 2, img_resolution])
                batch_mul_avg = np.sum(p_list * np.array([2, 1]))
            else:
                p_list = np.array([(1 - real_p) * 2 / 5, (1 - real_p) * 3 / 5, real_p])
                patch_list = np.array([img_resolution // 4, img_resolution // 2, img_resolution])
                batch_mul_avg = np.sum(np.array(p_list) * np.array([4, 2, 1]))  # 2

        self.p_list = p_list
        self.patch_list = patch_list
        self.batch_mul_dict = batch_mul_dict
        self.batch_mul_avg = batch_mul_avg

    def _build_network_and_diffusion(self):
        """Setup network and diffusion"""
        self.print_fn("Constructing network and diffusion...")

        attention_ds = []
        for res in self.network_kwargs["attn_resolutions"]:
            attention_ds.append(self.img_resolution // int(res))

        self.network_kwargs.update({"attn_resolutions": tuple(attention_ds)})

        self.net = dnnlib.util.construct_class_by_name(
            **self.network_kwargs,
            img_resolution=self.img_resolution,
            in_channels=self.img_channels + 2,
            out_channels=self.label_dim,
            label_dim=self.label_dim,
        )
        self.diffusion = dnnlib.util.construct_class_by_name(**self.diffusion_kwargs, model=self.net)
        sampler_kwargs = copy.deepcopy(self.diffusion_kwargs)
        sampler_kwargs.update({"class_name": "training.diffusion.DDIMSampler"})

        # ---------------------------------------------------------------------
        """ Setup EMA """
        self.print_fn("Setting up EMA...")
        if self.accelerator.is_main_process:
            self.ema = EMA(self.net)
            self.sampler = dnnlib.util.construct_class_by_name(**sampler_kwargs, model=self.ema)

        # ---------------------------------------------------------------------
        """ Setup the optimizer """
        self.print_fn("Setting up optimizer...")

        def lambda_lr_warmup(step):
            if self.lr_warmup == 0:
                return 1.0

            return min(1.0, step / self.lr_warmup)

        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)
        self.scheduler = dnnlib.util.construct_class_by_name(
            optimizer=self.optimizer,
            lr_lambda=lambda_lr_warmup,
            **self.scheduler_kwargs,
        )

        # ---------------------------------------------------------------------
        """ Prepare for distributed training """
        self.net, self.diffusion, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.net,
            self.diffusion,
            self.optimizer,
            self.scheduler,
        )

        # BUG: Acclerate's prepare resets the model to NOT require gradients, so we need to set it back to True!!
        self._set_requires_grad(self.net, True)

    def _update_ema(self):
        if not self.accelerator.is_main_process:
            return

        self.ema.to(self.device)
        self.ema.update()

    def _sample_images(self, num_images=64):
        if not self.accelerator.is_main_process:
            return

        self.ema.eval()

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

        samples = self.sampler(x_0, pos, class_labels, steps=10, guidance_scale=3.0)
        image_grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(num_images)), normalize=True, scale_each=True)
        torchvision.utils.save_image(image_grid, os.path.join(self.run_dir, f"sample-{self.cur_step}.png"))

    def train(self, log_interval, eval_interval, save_interval):
        """Main training loop.

        :param log_interval: When to log the metrics.
        :param eval_interval: When to evaluate the model.
        :param save_interval: When to save the model.
        """

        self.print_fn(f"Training for {self.num_steps - self.cur_step} steps...")
        self.print_fn("")

        self.accelerator.init_trackers(project_name="EGC")

        for step in range(self.cur_step, self.num_steps):
            self.cur_step = step

            metrics = self._training_step()

            if self.cur_step % log_interval == 0:
                self.print_fn("")
                self.print_fn(f"Step {self.cur_step}/{self.num_steps}")
                self._report_metrics(metrics)

            if self.ce_weight > 0 and step % eval_interval == 0:
                metrics = self._evaluate()
                self._report_metrics(metrics)

            if self.cur_step % save_interval == 0:
                self.print_fn("Saving model...")
                self._save()
                self._sample_images()

    def _training_step(self):
        metrics = {}

        self.optimizer.zero_grad(set_to_none=True)

        if self.ce_weight > 0:
            cls_images, cls_labels = next(self.cls_dataloader)
            cls_images, cls_labels = get_patches(cls_images, self.img_resolution), torch.cat([cls_labels, cls_labels]).argmax(dim=1)

            with self.accelerator.no_sync(self.net):
                logits, ce_loss, weighted_ce_loss = self.diffusion(cls_images, cls_labels, cls_mode=True)
                acc = (logits.argmax(dim=1) == cls_labels).float().mean()
                ece = self.ece_criterion(logits, cls_labels)

                metrics["cls_loss"] = ce_loss
                metrics["cls_acc"] = acc
                metrics["cls_ece"] = ece

                self.accelerator.backward(self.ce_weight * weighted_ce_loss)

        patch_size = int(np.random.choice(self.patch_list, p=self.p_list))
        batch_mul = self.batch_mul_dict[patch_size] // self.batch_mul_dict[self.img_resolution]
        images, labels = get_batch_data(self.train_dataloader, self.device, batch_mul)
        images, labels = get_patches(images, patch_size), labels.argmax(dim=1)

        mse_loss = self.diffusion(images, labels)
        metrics["mse_loss"] = mse_loss

        self.accelerator.backward(mse_loss / batch_mul)
        self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)

        self.accelerator.wait_for_everyone()
        self.optimizer.step()
        self.scheduler.step()

        grad_norm, param_norm = self._compute_norms()

        metrics["grad_norm"] = grad_norm
        metrics["param_norm"] = param_norm

        self._update_ema()

        return metrics

    def _evaluate(self):
        metrics = {"val_cls_loss": [], "val_cls_acc": [], "val_cls_ece": []}

        self.net.eval()
        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images, labels = images, labels.argmax(dim=1)
                images = get_patches(images, self.img_resolution)

                clean_timesteps = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)
                logits = self.net(images, clean_timesteps, cls_mode=True)

                metrics["val_cls_loss"].append(torch.nn.functional.cross_entropy(logits, labels))
                metrics["val_cls_acc"].append((logits.argmax(dim=1) == labels).float().mean())
                metrics["val_cls_ece"].append(self.ece_criterion(logits, labels))
        self.net.train()

        return {k: torch.stack(v).mean() for k, v in metrics.items()}

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
        """Log the metrics and print them.

        Args:
            metrics (dict): _description_
        """
        global_metrics = {}
        for name, value in metrics.items():
            gathered_values = self.accelerator.gather_for_metrics(value)
            if self.accelerator.is_main_process:
                global_avg = gathered_values.mean().item()
                global_metrics[name] = global_avg
                self.print_fn(f"{name} = {global_avg:.6f}")

        self.accelerator.log(global_metrics, step=self.cur_step)

    def _save(self):
        if not self.accelerator.is_main_process:
            return

        data = {
            "step": self.cur_step,
            "net": self.accelerator.unwrap_model(self.net).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
        }

        torch.save(data, os.path.join(self.run_dir, f"model-{self.cur_step:06d}.pt"))

    def _load(self):
        """Either load the latest checkpoint from the directory or load the checkpoint from the file

        :params resume_from: The directory or the file to resume from. If it is a directory, the latest
        checkpoint will be loaded. Else, the specified file will be loaded.
        """
        if not self.resume_from.endswith(".pt"):
            pt_files = [f for f in os.listdir(self.resume_from) if f.endswith(".pt")]
            ckpt = os.path.join(self.resume_from, pt_files[-1])
        else:
            ckpt = self.resume_from

        self.print_fn(f"Resuming from {ckpt}...")

        data = torch.load(ckpt, map_location=self.device, weights_only=True)

        self.cur_step = data["step"]
        self.accelerator.unwrap_model(self.net).load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.scheduler.load_state_dict(data["scheduler"])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
