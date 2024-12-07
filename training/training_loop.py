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
        num_epochs=100,  # Number of training steps
        lr_warmup=0,  # Number of warmup steps for the learning rate
        batch_size=128,  # Batch size
        ce_weight=0.001,  # Weight of the classification loss
        label_smooth=0.2,  # Label smoothing factor
        real_p=0.5,  # Probability of real images
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

        self.num_epochs = num_epochs
        self.lr_warmup = lr_warmup
        self.ce_weight = ce_weight
        self.label_smooth = label_smooth
        self.real_p = real_p
        self.train_on_latents = train_on_latents
        self.seed = seed
        self.resume_from = resume_from
        self.batch_size = batch_size
        self.cur_epoch = 0

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=self.label_smooth)
        self.ece_criterion = ECELoss(n_bins=10)

        self.accelerator = Accelerator(split_batches=True, log_with="wandb", gradient_accumulation_steps=2)
        self.device = self.accelerator.device
        self.print_fn = self.accelerator.print
        self._init_trainer()

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
                transforms.RandomResizedCrop(self.img_resolution),
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
            batch_size=self.batch_size,
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

        self.cls_dataloader = cycle(self.cls_dataloader)

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

        self.network_kwargs.update({"attn_resolutions": attention_ds})

        self.net = dnnlib.util.construct_class_by_name(
            **self.network_kwargs,
            img_resolution=self.img_resolution,
            in_channels=self.img_channels,
            out_channels=self.label_dim,
            label_dim=self.label_dim,
        )
        self.diffusion = dnnlib.util.construct_class_by_name(**self.diffusion_kwargs)
        self.schedule_sampler = UniformSampler(self.diffusion)

        # ---------------------------------------------------------------------
        """ Setup EMA """
        self.print_fn("Setting up EMA...")
        if self.accelerator.is_main_process:
            self.ema = EMA(self.net)

        # ---------------------------------------------------------------------
        """ Setup the optimizer """
        self.print_fn("Setting up optimizer...")

        def lambda_lr_warmup(step):
            if self.lr_warmup == 0:
                return 1.0

            return min(1.0, step / self.lr_warmup)

        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)
        self.scheduler = dnnlib.util.construct_class_by_name(optimizer=self.optimizer, lr_lambda=lambda_lr_warmup, **self.scheduler_kwargs)

        # ---------------------------------------------------------------------
        """ Prepare for distributed training """
        self.net, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.net,
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

    def _sample_images(self, num_images=25):
        if not self.accelerator.is_main_process:
            return

        img_size, img_channels = self.img_resolution, self.img_channels
        samples = torch.randn((num_images, img_channels, img_size, img_size), device=self.device)

        self.ema.eval()
        for t in reversed(range(0, self.diffusion.num_timesteps)):
            timesteps = torch.full((num_images,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                pred_noise = self.ema(samples, timesteps)
                samples = self.diffusion.p_sample(pred_noise, samples, timesteps)

        image_grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(num_images)), normalize=True, scale_each=True)
        torchvision.utils.save_image(image_grid, os.path.join(self.run_dir, f"sample-{self.cur_epoch}.png"))

    def train(self, eval_interval=5, save_interval=10):
        """Main training loop.

        :param num_epochs: Total number of training epochs.
        :param log_interval: How often to print metrics. Default is 1 epoch.
        :param eval_interval: When to evaluate the model. Default is 1 epoch.
        :param save_interval: When to save the model. Default is 1 epoch.
        """

        self.print_fn(f"Training for {self.num_epochs - self.cur_epoch} epochs...")
        self.print_fn("")

        self.accelerator.init_trackers(project_name="EGC")

        for epoch in range(self.cur_epoch, self.num_epochs):
            self.cur_epoch = epoch

            metrics = self._training_epoch()
            self._report_metrics(metrics)

            if self.ce_weight > 0 and epoch % eval_interval == 0:
                metrics = self._evaluate()
                self._report_metrics(metrics)

            if epoch % save_interval == 0:
                self._save()
                self._sample_images()

    def _training_epoch(self):
        self.net.train()
        metrics = {}
        metric_sums = {}

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.cur_epoch}", disable=not self.accelerator.is_main_process)

        for images, labels in pbar:

            self.optimizer.zero_grad()

            if self.ce_weight > 0:
                cls_images, cls_labels = next(self.cls_dataloader)

                timesteps = torch.zeros(cls_images.shape[0], dtype=torch.long, device=self.device)
                sqrt_alphas_cumprod = torch.from_numpy(self.diffusion.sqrt_alphas_cumprod).to(self.device)[timesteps].float()

                with self.accelerator.no_sync(self.net):
                    logits = self.net(cls_images, timesteps, cls_mode=True)

                    cls_labels = cls_labels.argmax(dim=1)
                    ce_loss = (self.criterion(logits, cls_labels) * sqrt_alphas_cumprod).mean()
                    acc = (logits.argmax(dim=1) == cls_labels).float().mean()
                    ece = self.ece_criterion(logits, cls_labels)

                    metrics["cls_loss"] = ce_loss
                    metrics["cls_acc"] = acc
                    metrics["cls_ece"] = ece

                    self.accelerator.backward(self.ce_weight * ce_loss)

            timesteps, weights = self.schedule_sampler.sample(images.shape[0], self.device)
            mask = torch.rand(labels.shape[0], device=self.device) < 0.1
            labels[mask] = self.label_dim

            with self.accelerator.accumulate(self.net):
                mse_loss = self.diffusion.training_losses(
                    net=self.net,
                    x_start=images,
                    t=timesteps,
                    model_kwargs={"class_labels": labels.argmax(dim=1)},
                )

                mse_loss = (mse_loss * weights).mean()
                metrics["mse_loss"] = mse_loss

                self.accelerator.backward(mse_loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

            grad_norm, param_norm = self._compute_norms()
            metrics["grad_norm"] = grad_norm
            metrics["param_norm"] = param_norm
            metrics["lr"] = torch.tensor(self.scheduler.get_last_lr()[0], device=self.device)

            for key, value in metrics.items():
                if key not in metric_sums:
                    metric_sums[key] = 0.0
                metric_sums[key] += value.item() if torch.is_tensor(value) else value

            self._update_ema()

        epoch_metrics = {k: torch.tensor(v / len(self.train_dataloader), device=self.device) for k, v in metric_sums.items()}

        return epoch_metrics

    def _evaluate(self):
        self.net.eval()
        metrics = {"val_cls_loss": [], "val_cls_acc": [], "val_cls_ece": []}

        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images, labels = images.to(self.device), labels.to(self.device).argmax(dim=1)

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
        """
        Gathers and reports metrics across all processes using Accelerate.

        Args:
            metrics (dict): Dictionary of metrics with names as keys and tensors as values.
                            Example: {"loss": tensor_loss, "accuracy": tensor_acc}.
        Returns:
            dict: A dictionary with global (averaged) metric values.
        """
        global_metrics = {}
        for name, value in metrics.items():
            gathered_values = self.accelerator.gather_for_metrics(value)
            if self.accelerator.is_main_process:
                global_avg = gathered_values.mean().item()
                global_metrics[name] = global_avg
                self.print_fn(f"{name} = {global_avg:.6f}")

        self.accelerator.log(global_metrics, step=self.cur_epoch)

    def _save(self):
        if not self.accelerator.is_main_process:
            return

        data = {
            "epoch": self.cur_epoch,
            "net": self.accelerator.unwrap_model(self.net).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
        }

        torch.save(data, os.path.join(self.run_dir, f"model-{self.cur_epoch:06d}.pt"))

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

        self.cur_epoch = data["epoch"]
        self.accelerator.unwrap_model(self.net).load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.scheduler.load_state_dict(data["scheduler"])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
