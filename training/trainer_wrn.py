import json
import math
import os
from typing import OrderedDict

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch_uncertainty.post_processing import TemperatureScaler
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

import dnnlib
from training.ece import ECELoss


class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.total += value * count
        self.count += count

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


class Trainer:
    def __init__(
        self,
        run_dir="./training-runs",  # Output directory
        dataset_kwargs={},  # Training dataset options
        val_dataset_kwargs={},  # Validation dataset options
        test_dataset_kwargs={},  # Test dataset options
        network_kwargs={},  # Model options
        optimizer_kwargs={},  # Optimizer options
        decay_epochs=[60, 120, 160],  # Learning rate milestones
        decay_rate=0.2,  # Learning rate decay rate
        num_epochs=200,  # Number of training steps
        accum_steps=1,  # Accumulate gradients over multiple steps
        batch_size=128,  # Batch size
        seed=1,  # Seed for reproducibility
        resume_from=None,  # Checkpoint to resume from
        calibrate=False,  # Calibrate the model
    ):
        self.run_dir = run_dir
        self.dataset_kwargs = dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.test_dataset_kwargs = test_dataset_kwargs
        self.network_kwargs = network_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.num_epochs = num_epochs
        self.accum_steps = accum_steps
        self.seed = seed
        self.resume_from = resume_from
        self.batch_size = batch_size
        self.calibrate = calibrate

        self.ece = ECELoss(n_bins=10)

        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(dispatch_batches=True, split_batches=False),
            gradient_accumulation_steps=self.accum_steps,
        )
        self.device = self.accelerator.device
        self.print_fn = self.accelerator.print
        self.per_device_batch_size = self._calculate_per_device_batch_size()
        self._init_trainer()

    def _calculate_per_device_batch_size(self):
        """Calculate the batch size per device."""
        world_size = self.accelerator.num_processes
        per_device_batch_size = self.batch_size // (world_size * self.accelerator.gradient_accumulation_steps)
        assert per_device_batch_size * world_size * self.accelerator.gradient_accumulation_steps == self.batch_size, "Batch size must be divisible by num_processes * gradient_accumulation_steps."
        return per_device_batch_size

    def _init_trainer(self):
        """Initialize the Trainer: seeds, datasets, and network."""
        self._init_env()
        self._prepare_dataloaders()
        self._build_network()

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

        self.cls_dataset = dnnlib.util.construct_class_by_name(**self.dataset_kwargs, transform=augment_transform)
        self.val_dataset = dnnlib.util.construct_class_by_name(**self.val_dataset_kwargs, transform=transform)
        self.test_dataset = dnnlib.util.construct_class_by_name(**self.test_dataset_kwargs, transform=transform)
        self.dataloader_kwargs = dict(
            batch_size=self.per_device_batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.cls_dataloader = DataLoader(self.cls_dataset, **self.dataloader_kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

        self.cls_dataloader, self.val_dataloader, self.test_dataloader = self.accelerator.prepare(self.cls_dataloader, self.val_dataloader, self.test_dataloader)

    def _set_requires_grad(self, model, requires_grad):
        """Set requires_grad for all parameters in the model"""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _build_network(self):
        """Setup network"""
        self.print_fn("Constructing network...")

        self.network_kwargs = {**self.network_kwargs, "num_classes": self.label_dim}
        self.net = dnnlib.util.construct_class_by_name(**self.network_kwargs)

        if self.accelerator.is_local_main_process:
            with open(os.path.join(self.run_dir, "network_kwargs.json"), "w") as f:
                json.dump(self.network_kwargs, f, indent=4)

        if self.accelerator.is_main_process:
            self.ema = EMA(self.net)
            self.ema = self.ema.to(self.device)

        # ---------------------------------------------------------------------
        """ Setup the optimizer """
        self.print_fn("Setting up optimizer...")
        self.optimizer = dnnlib.util.construct_class_by_name(params=self.net.parameters(), **self.optimizer_kwargs)

        # ---------------------------------------------------------------------
        """ Prepare for distributed training """
        self.net, self.optimizer = self.accelerator.prepare(self.net, self.optimizer)

        summary(
            self.net,
            input_size=(self.per_device_batch_size, self.img_channels, self.img_resolution, self.img_resolution),
            col_names=("input_size", "output_size", "num_params", "trainable"),
            device="cuda",
        )

    def _update_lr(self):
        lr_min = 1e-6
        lr_max = self.optimizer.param_groups[0]["lr"]
        lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * (self.cur_epoch / self.num_epochs))) / 2
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, eval_interval: int):
        """Main training loop.

        :param log_interval: When to log the metrics.
        :param eval_interval: When to evaluate the model.
        :param save_interval: When to save the model.
        """

        self.accelerator.init_trackers(project_name="EGC")

        for epoch in range(self.num_epochs):
            self.cur_epoch = epoch
            self._train_one_epoch()
            self._update_lr()

            if eval_interval > 0 and self.cur_epoch % eval_interval == 0:
                model = self.net

                if self.calibrate:
                    self.print_fn("Calibrating the model...")
                    model = TemperatureScaler(model=self.base_trainer.net, device=self.base_trainer.device)
                    model.fit(calibration_set=self.base_trainer.val_dataset)

                self.evaluate(model, self.test_dataloader)

        self.cur_epoch = self.num_epochs
        self.evaluate(self.net, self.test_dataloader)
        self._save("model-final")

    def _gather(self, metrics: OrderedDict):
        """Gather the metrics across all processes."""
        global_metrics = self.accelerator.gather_for_metrics(metrics)
        if self.accelerator.is_main_process:
            global_metrics = {k: v.mean().item() for k, v in global_metrics.items()}
        return global_metrics

    def _update_ema(self):
        if not self.accelerator.is_main_process:
            return

        self.ema.update()

    def _train_one_epoch(self):
        loss_meter = Meter()
        acc_meter = Meter()
        ece_meter = Meter()

        self.net.train()

        with tqdm(total=len(self.cls_dataloader), desc=f"Epoch {self.cur_epoch}", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for x, y in self.cls_dataloader:
                pbar.update(1)

                y = y.argmax(dim=1)

                logits = self.net(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()

                loss_meter.update(loss.mean(), x.size(0))
                acc_meter.update(acc, x.size(0))
                ece_meter.update(self.ece(logits, y))

                self._update_ema()

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Log metrics
            metrics = {
                "cls_loss": loss_meter.compute().clone().detach(),
                "cls_acc": acc_meter.compute().clone().detach(),
                "cls_ece": ece_meter.compute().clone().detach(),
                "lr": torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device),
            }
            metrics = self._gather(metrics)
            pbar.set_postfix(metrics)
            pbar.close()

        self.accelerator.log(metrics, step=self.cur_epoch)

    @torch.no_grad()
    def evaluate(self, net: torch.nn.Module, dataloader: DataLoader):
        dataloader = self.accelerator.prepare(dataloader)
        loss_meter = Meter()
        acc_meter = Meter()
        ece_meter = Meter()

        net.eval()

        with tqdm(total=len(dataloader), desc="Validation", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for x, y in dataloader:
                pbar.update(1)

                y = y.argmax(dim=1)

                logits = net(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()

                # Update global meters
                loss_meter.update(loss.mean(), x.size(0))
                acc_meter.update(acc, x.size(0))
                ece_meter.update(self.ece(logits, y))

            # Log metrics
            metrics = {
                "val_cls_loss": loss_meter.compute().clone().detach(),
                "val_cls_acc": acc_meter.compute().clone().detach(),
                "val_cls_ece": ece_meter.compute().clone().detach(),
            }
            metrics = self._gather(metrics)
            pbar.set_postfix(metrics)
            pbar.close()

        net.train()

        self.accelerator.log(metrics, step=self.cur_epoch)

    @torch.no_grad()
    def get_probs(self, net: torch.nn.Module, dataloader: DataLoader):
        dataloader = self.accelerator.prepare(dataloader)
        probs = []

        net.eval()

        with tqdm(total=len(dataloader), desc="Getting Probabilities", disable=not self.accelerator.is_local_main_process, dynamic_ncols=True) as pbar:
            for x, _ in dataloader:
                pbar.update(1)
                logits = net(x)
                probs.append(logits.softmax(dim=1))
            pbar.close()

        net.train()

        probs = torch.cat(probs, dim=0)
        probs = torch.max(probs, dim=1).values

        return probs

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

    def _save(self, filename: str):
        if not self.accelerator.is_main_process:
            return

        data = {
            "epoch": self.cur_epoch,
            "net": self.accelerator.unwrap_model(self.net).state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(data, os.path.join(self.run_dir, f"{filename}.pt"))

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

        data = torch.load(ckpt, weights_only=True)

        self.cur_epoch = data["epoch"]
        self.accelerator.unwrap_model(self.net).load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])
