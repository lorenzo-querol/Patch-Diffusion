import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset

from training.datamodule import DataModule
from training.trainer_egc import EGCTrainer
from training.trainer_wrn import WRNTrainer

accelerator = Accelerator()


class ActiveLearningTrainer:
    """Trainer for active learning."""

    def __init__(self, datamodule: DataModule, num_samples: float, strategy="random"):
        """
        Args:
            datamodule (`DataModule`): DataModule instance.
            num_samples (`float`): Number of samples to query at each iteration.
            strategy (`str`, optional, defaults to `random`): Active learning strategy to use. Options are `random`, `lc`, `sm`, and `entropy`.
        """
        self.datamodule = datamodule
        self.num_samples = int(num_samples * len(datamodule.cls_dataset))
        self.strategy = strategy
        self.dataloader_kwargs = {"batch_size": 128, "num_workers": 4, "pin_memory": True, "drop_last": False, "shuffle": False}

        np.random.seed(42)
        all_indices = np.array(list(range(len(datamodule.cls_dataset))))
        self.labeled_indices = np.random.choice(all_indices, size=self.num_samples, replace=False)
        self.unlabeled_indices = np.setdiff1d(all_indices, self.labeled_indices)
        self.datamodule.update_dataloaders(self.labeled_indices)

    def random_query(self):
        """Query samples randomly.

        Returns:
            query_indices (np.ndarray): The indices of the samples to query.
        """
        np.random.seed(42)
        query_size = min(self.num_samples, len(self.unlabeled_indices))
        query_indices = np.random.choice(self.unlabeled_indices, size=query_size, replace=False)

        return query_indices

    def least_confidence_query(self, trainer, net: torch.nn.Module):
        """Query samples using least confidence strategy.

        Args:
            net (torch.nn.Module): The model to use for querying.

        Returns:
            query_indices (np.ndarray): The indices of the samples to query.
        """
        unlabeled_dataset = Subset(self.datamodule.cls_dataset, self.unlabeled_indices)
        dataloader = DataLoader(unlabeled_dataset, **self.dataloader_kwargs)

        probs = trainer.get_probs(net, dataloader)
        lc_scores = 1 - torch.max(probs, dim=1).values
        lc_scores = lc_scores[: len(self.unlabeled_indices)]

        query_size = min(self.num_samples, len(self.unlabeled_indices))
        sorted_indices = torch.argsort(lc_scores, descending=True)[:query_size].cpu().numpy()
        query_indices = self.unlabeled_indices[sorted_indices]

        return query_indices

    def _active_learning_step(self, trainer, net: torch.nn.Module):
        """Perform one active learning iteration.

        Args:
            net (torch.nn.Module): The model to use for querying.
        """
        match self.strategy:
            case "lc":
                query_indices = self.least_confidence_query(trainer, net)
            case "random":
                query_indices = self.random_query()
            case _:
                raise NotImplementedError(f"Active learning strategy {self.strategy} not implemented.")

        self.labeled_indices = np.concatenate([self.labeled_indices, query_indices])
        self.unlabeled_indices = np.setdiff1d(self.unlabeled_indices, query_indices)
        self.datamodule.update_dataloaders(self.labeled_indices)
        self.cur_al_iteration += 1

    def run_loop(self, model_type: str, *fit_args, **trainer_kwargs):
        """Run active learning loop.

        Args:
            model_type (`str`): Model type to use. Options are `egc` and `wrn`.
            *fit_args: Arguments to pass to the `train` method.
            **trainer_kwargs: Keyword arguments to pass to the trainer.
        """

        self.cur_al_iteration = 1

        # 1. Instantiate trainer
        match model_type:
            case "egc":
                trainer = EGCTrainer(datamodule=self.datamodule, active_learning=True, **trainer_kwargs)
            case "wrn":
                trainer = WRNTrainer(datamodule=self.datamodule, active_learning=True, **trainer_kwargs)
            case _:
                raise NotImplementedError(f"Trainer for {model_type} not implemented.")

        while True:
            accelerator.print(f"\nActive learning iteration: {self.cur_al_iteration}")
            self._log_distribution(trainer)

            # 2. Fit model
            trainer.fit(*fit_args)

            # Save models
            filename = f"al_iter_{self.cur_al_iteration}-final"
            trainer._save_checkpoint(filename)
            if hasattr(trainer, "_sample_images"):
                trainer._sample_images(filename)

            # Check if the unlabeled set is exhausted
            accelerator.wait_for_everyone()
            if len(self.unlabeled_indices) == 0:
                accelerator.print("Unlabeled set is exhausted, stopping active learning...")
                break

            # 3. Active learning step
            self._active_learning_step(trainer, trainer.ema)
            trainer.datamodule = self.datamodule
            trainer.al_mul += 1

    @accelerator.on_main_process
    def _log_distribution(self, trainer):
        """Log class distribution of labeled data."""

        labeled_dataset = Subset(self.datamodule.cls_dataset, self.labeled_indices)
        dataloader = DataLoader(labeled_dataset, **self.dataloader_kwargs)
        distribution = torch.zeros(self.datamodule.label_dim, dtype=torch.long)

        for _, targets in dataloader:
            targets = targets.argmax(dim=1)
            distribution += torch.bincount(targets.to(torch.long), minlength=self.datamodule.label_dim)

        distribution = distribution.cpu().numpy().tolist()
        print(f"Labeled: {len(self.labeled_indices)}, Unlabeled: {len(self.unlabeled_indices)}")
        print(f"Class distribution: {distribution}")

        with open(f"{trainer.run_dir}/class_distribution.csv", "a") as f:
            f.write(",".join(map(str, distribution)) + "\n")
