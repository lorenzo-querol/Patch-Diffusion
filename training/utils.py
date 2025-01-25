from torch.utils.data import DataLoader
import torch
import math


def exists(val):
    return val is not None


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


def cycle(dataloader: DataLoader):
    while True:
        for data in dataloader:
            yield data


def linear_beta_schedule(timesteps: int):
    """Linear schedule proposed in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

    Args:
        timesteps (int): The number of timesteps.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s=0.008):
    """Cosine schedule proposed in [Improved Denoising Diffusion Probabilistic Models](https://openreview.net/forum?id=-NEXDKk8gZ).

    Args:
        timesteps (int): The number of timesteps.
        s (float,optional): The scaling factor. Defaults to 0.008.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


def get_beta_schedule(schedule_name: str, timesteps: int):
    """Get the beta schedule based on the schedule name.

    Args:
        schedule_name (str): The name of the schedule. Can be `linear` or `cosine`.
        timesteps (int): The number of timesteps.
    """
    match schedule_name:
        case "linear":
            return linear_beta_schedule(timesteps)
        case "cosine":
            return cosine_beta_schedule(timesteps)
        case _:
            raise NotImplementedError(f"Unknown schedule name: {schedule_name}")


def extract(v, i, shape):
    """Get the `i`-th number in `v`, and the shape of v is mostly `(T, )`. The shape of `i` is mostly `(batch_size, )` equal to `[v[index] for index in i]`"""
    out = torch.gather(v.to(i.device), index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))

    return out
