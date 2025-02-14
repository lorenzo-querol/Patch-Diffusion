import os
from typing import OrderedDict

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed, broadcast

from training.ece import ECELoss

accelerator = Accelerator()


class BaseTrainer:
    def __init__(
        self,
        datamodule,
        run_dir: str,
        network_kwargs,
        optimizer_kwargs,
        accum_steps: int = 1,
        seed: int = 1,
        resume_from: str = None,
        train_on_latents: bool = False,
        active_learning: bool = False,
        al_mul: int = 0,
    ):
        """Common initialization parameters shared by all trainers.

        Args:
            datamodule (`DataModule`): DataModule instance.
            run_dir (`str`): Output directory.
            network_kwargs (`dict`): Model options.
            optimizer_kwargs (`dict`): Optimizer options.
            accum_steps (`int`): Number of steps to accumulate gradients over. Defaults to 1.
            seed (`int`): Random seed. Defaults to 1.
            resume_from (`str`): Path to resume checkpoint from. Defaults to None.
            train_on_latents (`bool`): Whether to train on VAE latents. Defaults to False.
            active_learning (`bool`): Whether trainer is used for active learning. Defaults to False.
            al_mul (`int`): Current active learning iteration. Defaults to 0.
        """
        self.datamodule = datamodule
        self.run_dir = run_dir
        self.network_kwargs = network_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.accum_steps = accum_steps
        self.seed = seed
        self.resume_from = resume_from
        self.train_on_latents = train_on_latents
        self.active_learning = active_learning
        self.al_mul = al_mul

        self.ece = ECELoss(n_bins=10)

        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(dispatch_batches=True, split_batches=False),
            gradient_accumulation_steps=self.accum_steps,
            log_with="wandb",
        )
        self.accelerator.init_trackers(project_name="EGC", init_kwargs={"wandb": {"name": "_".join(self.run_dir.split("/"))}})

        self.device = self.accelerator.device
        self.print_fn = self.accelerator.print

        self.img_vae = None
        self.latent_scale_factor = 0.18215

    def _init_env(self):
        """Sets seeds and other environment variables."""
        set_seed(self.seed)
        torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def _encode_latents(self, images: torch.Tensor):
        """Encode the given images to compressed latent space.

        Args:
            images (`torch.Tensor`): The images to encode.

        Returns:
            The encoded latents.
        """
        images = self.img_vae.encode(images)["latent_dist"].sample()
        latents = self.latent_scale_factor * images

        return latents

    def _set_requires_grad(self, model: torch.nn.Module, requires_grad: bool):
        """Set requires_grad for all parameters in the model.

        Args:
            model (torch.nn.Module): The model to set `requires_grad` for.
            requires_grad (bool): Whether to set `requires_grad` to true or false.
        """
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _update_ema(self):
        """Update the EMA model."""
        self.ema.to(self.device)
        self.ema.update()

    @accelerator.on_main_process
    def _save_checkpoint(self, filename: str, additional_data: dict = None):
        """Save a checkpoint file.

        Args:
            filename (`str`): The name of the file to save.
            additional_data (`dict`): Additional data to save in the checkpoint.
        """
        data = {
            "net": self.accelerator.unwrap_model(self.net).state_dict(),
            "ema": self.accelerator.unwrap_model(self.ema).state_dict(),
        }

        if additional_data:
            data.update(additional_data)

        if hasattr(self, "temperature"):
            data.update({"temperature": self.temperature})

        self.print_fn(f"Saving checkpoint to {filename}...")
        torch.save(data, os.path.join(self.run_dir, f"{filename}.pt"))

    def _load_checkpoint(self, filename: str):
        """Load a checkpoint file.

        Args:
            filename (`str`): The name of the file to load.
        """
        self.print_fn(f"Loading checkpoint from {filename}...")
        checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
        self.accelerator.unwrap_model(self.net).load_state_dict(checkpoint["net"])
        self.accelerator.unwrap_model(self.ema).load_state_dict(checkpoint["ema"])

        if "temperature" in checkpoint:
            self.temperature = checkpoint["temperature"]

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
