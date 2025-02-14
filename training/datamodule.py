import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import dnnlib

from .utils import cycle

accelerator = Accelerator()


class DataModule:
    def __init__(self, dataset_kwargs, val_dataset_kwargs, test_dataset_kwargs, batch_size):
        self.dataset_kwargs = dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.test_dataset_kwargs = test_dataset_kwargs
        self.cls_dataloader_kwargs = dict(batch_size=batch_size, pin_memory=True, num_workers=4)

    def _prepare_data(self):
        """Prepare datasets."""
        accelerator.print("Preparing data...")

        dataset_obj = dnnlib.util.construct_class_by_name(**self.dataset_kwargs)
        self.img_resolution, self.img_channels, self.label_dim = (
            dataset_obj.resolution,
            dataset_obj.num_channels,
            dataset_obj.label_dim,
        )
        del dataset_obj

        multiplier = 3 if self.img_channels == 1 and self.img_resolution == 256 else 1

        cls_transform = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(self.img_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 and self.img_resolution == 256 else x),
                transforms.Normalize(mean=[0.5] * multiplier, std=[0.5] * multiplier),
            ]
        )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 and self.img_resolution == 256 else x),
                transforms.Normalize(mean=[0.5] * multiplier, std=[0.5] * multiplier),
            ]
        )

        self.train_dataset = dnnlib.util.construct_class_by_name(**self.dataset_kwargs, transform=transform)
        self.cls_dataset = dnnlib.util.construct_class_by_name(**self.dataset_kwargs, transform=cls_transform)
        self.val_dataset = dnnlib.util.construct_class_by_name(**self.val_dataset_kwargs, transform=transform)
        self.test_dataset = dnnlib.util.construct_class_by_name(**self.test_dataset_kwargs, transform=transform)

    def _prepare_dataloaders(self):
        """Prepare dataloaders using Accelerate."""
        cls_dataloader = DataLoader(self.cls_dataset, **self.cls_dataloader_kwargs)
        val_dataloader = DataLoader(self.val_dataset, batch_size=128, pin_memory=True, num_workers=4)
        test_dataloader = DataLoader(self.test_dataset, batch_size=128, pin_memory=True, num_workers=4)

        self.cls_dataloader, self.val_dataloader, self.test_dataloader = accelerator.prepare(
            cls_dataloader,
            val_dataloader,
            test_dataloader,
        )

    def update_dataloaders(self, labeled_indices: np.ndarray):
        """Update dataloaders with the new labeled indices.

        Args:
            labeled_indices (np.ndarray): The indices of the labeled samples.
        """
        pass


class EGCDataModule(DataModule):
    def __init__(self, dataset_kwargs, val_dataset_kwargs, test_dataset_kwargs, batch_size, accum_steps):
        super().__init__(dataset_kwargs, val_dataset_kwargs, test_dataset_kwargs, batch_size)
        self._prepare_data()

        world_size = accelerator.num_processes
        per_device_batch_size = batch_size // (world_size * accum_steps)
        assert per_device_batch_size * world_size * accum_steps == batch_size, "Batch size must be divisible by num_processes * gradient_accumulation_steps."
        self.cls_dataloader_kwargs.update({"batch_size": per_device_batch_size})

        self._prepare_dataloaders()

    def _prepare_dataloaders(self):
        """Prepare dataloaders using Accelerate."""
        train_dataloader = DataLoader(self.train_dataset, **self.cls_dataloader_kwargs)
        cls_dataloader = DataLoader(self.cls_dataset, **self.cls_dataloader_kwargs)
        val_dataloader = DataLoader(self.val_dataset, batch_size=128, pin_memory=True, num_workers=4)
        test_dataloader = DataLoader(self.test_dataset, batch_size=128, pin_memory=True, num_workers=4)

        self.train_dataloader, self.cls_dataloader, self.val_dataloader, self.test_dataloader = accelerator.prepare(
            train_dataloader,
            cls_dataloader,
            val_dataloader,
            test_dataloader,
        )
        self.train_dataloader, self.cls_dataloader = cycle(self.train_dataloader), cycle(self.cls_dataloader)

    def update_dataloaders(self, labeled_indices: np.ndarray):
        dataset = Subset(self.cls_dataset, labeled_indices)
        dataloader = DataLoader(dataset, **self.cls_dataloader_kwargs)
        self.cls_dataloader = accelerator.prepare(dataloader)
        self.cls_dataloader = cycle(self.cls_dataloader)  # NOTE: EGC is trained in a semi-supervised manner


class WRNDataModule(DataModule):
    def __init__(self, dataset_kwargs, val_dataset_kwargs, test_dataset_kwargs, batch_size):
        super().__init__(dataset_kwargs, val_dataset_kwargs, test_dataset_kwargs, batch_size)
        self._prepare_data()
        self._prepare_dataloaders()

    def _prepare_dataloaders(self):
        """Prepare dataloaders using Accelerate."""
        cls_dataloader = DataLoader(self.cls_dataset, **self.cls_dataloader_kwargs)
        val_dataloader = DataLoader(self.val_dataset, batch_size=128, pin_memory=True, num_workers=4)
        test_dataloader = DataLoader(self.test_dataset, batch_size=128, pin_memory=True, num_workers=4)

        self.cls_dataloader, self.val_dataloader, self.test_dataloader = accelerator.prepare(
            cls_dataloader,
            val_dataloader,
            test_dataloader,
        )

    def update_dataloaders(self, labeled_indices: np.ndarray):
        dataset = Subset(self.cls_dataset, labeled_indices)
        dataloader = DataLoader(dataset, **self.cls_dataloader_kwargs)
        self.cls_dataloader = accelerator.prepare(dataloader)
