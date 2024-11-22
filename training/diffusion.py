# Description: Gaussian diffusion utility for the diffusion models.

import numpy as np
import torch
import math


def linear_beta_schedule(timesteps):
    """Linear schedule.

    Proposed in https://arxiv.org/abs/2006.11239
    """

    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule.

    Proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """

    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps, dtype=np.float64) / timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def get_beta_schedule(schedule_name, timesteps):
    """Get the beta schedule based on the schedule name.

    Args:
        schedule_name: Name of the schedule ('linear' or 'cosine')
        timesteps: Number of time steps in the forward process
    """
    if schedule_name == "linear":
        return linear_beta_schedule(timesteps)
    elif schedule_name == "cosine":
        return cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown schedule name: {schedule_name}")


class GaussianDiffusion:
    """Gaussian diffusion utility.

    Args:
        schedule_name: Name of the schedule ('linear' or 'cosine')
        timesteps: Number of time steps in the forward process
    """

    def __init__(self, schedule_name, timesteps=1000, clip_min=-1.0, clip_max=1.0):
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define variance schedule
        self.betas = betas = get_beta_schedule(schedule_name, timesteps)
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

    def _extract(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        timesteps = timesteps.to(torch.long)

        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def q_mean_variance(self, x_start, t):
        """Extracts the mean, and the variance at current timestep.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """
        x_start_shape = torch.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start_shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        x_start_shape = x_start.shape
        return self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = torch.shape(x_t)
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion posterior `q(x_{t-1} | x_t, x_0)`.

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """

        x_t_shape = torch.shape(x_t)
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t_shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)

        if clip_denoised:
            x_recon = torch.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(pred_noise, x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.random.normal(shape=x.shape, dtype=x.dtype)

        # No noise when t == 0
        nonzero_mask = torch.reshape(1 - torch.cast(torch.equal(t, 0), torch.float32), [torch.shape(x)[0], 1, 1, 1])
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def training_losses(self, net, x_start, t, noise=None, net_kwargs={}):
        """Compute the training loss for the diffusion model.

        Args:
            net: Diffusion model
            x_start: the [N x C x ...] tensor of inputs.
            t: A batch of timesteps
            noise: If specified, the specific Gaussian noise to try to remove.
        """

        def mean_flat(tensor):
            """
            Take the mean over all non-batch dimensions.
            """
            return tensor.mean(dim=list(range(1, len(tensor.shape))))

        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        score = net(x_t, t, **net_kwargs)
        loss = mean_flat((noise[:, :3] - score) ** 2)

        return loss
