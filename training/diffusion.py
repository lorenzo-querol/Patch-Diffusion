# Description: Gaussian diffusion utility for the diffusion models.

import numpy as np
import torch
import math

import torch.nn.functional as F
from tqdm import tqdm


def exists(x):
    return x is not None


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

    def mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def get_v(self, x, noise, t):
        return self._extract(self.sqrt_alphas_cumprod, t, x.shape) * noise - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x

    def training_losses(self, net, x_start, t, target="eps", model_kwargs={}):
        """Compute the training loss for the diffusion model.

        :param net: Diffusion model.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: A batch of timesteps.
        :param target: the target to predict. One of "eps", "x_start", or "v".
        :param model_kwargs: additional keyword arguments to pass to the model.
        :return loss: the training loss.
        """
        assert target in ["eps", "x_start", "v"], f"Unknown target: {target}"

        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        target_map = {"eps": noise, "x_start": x_start, "v": self.get_v(x_start, noise, t)}
        target = target_map[target]

        output = net(x_t, t, **model_kwargs)
        loss = F.mse_loss(output, target[:, :3])

        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = x_t.shape
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """

        x_t_shape = x_t.shape
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t_shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(pred_noise, x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def sample_images_ddim(self, net, shape, pos, labels, device, eta=0.0, timesteps=50):
        # Start from pure noise
        x = torch.randn(shape).to(device)

        # Select timesteps for DDIM (evenly spaced)
        skip = self.num_timesteps // timesteps
        seq = list(range(0, self.num_timesteps, skip))

        for i in tqdm(range(len(seq)), desc="DDIM sampling", leave=False):
            t = seq[len(seq) - 1 - i]  # Current timestep
            t_next = seq[len(seq) - 2 - i] if i < len(seq) - 1 else 0  # Next timestep

            t_batch = torch.full((shape[0],), t, device=device)
            t_next_batch = torch.full((shape[0],), t_next, device=device)

            with torch.no_grad():
                # Model predicts x_start directly
                pred_x_start = net(torch.cat((x, pos), dim=1), t_batch, class_labels=labels)

                # Get alpha values for current and next timestep
                alpha_t = self._extract(self.sqrt_alphas_cumprod, t_batch, x.shape)
                alpha_next = self._extract(self.sqrt_alphas_cumprod, t_next_batch, x.shape)

                # Get sigma values
                sigma_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)

                # Calculate predicted noise
                pred_noise = (x - alpha_t * pred_x_start) / sigma_t

                # DDIM formula
                sigma_t_next = eta * torch.sqrt((1 - alpha_next**2) / (1 - alpha_t**2) * (1 - alpha_t**2 / alpha_next**2))

                # No noise when t_next == 0 or eta == 0
                noise = 0 if t_next == 0 or eta == 0 else torch.randn_like(x)

                # DDIM update step
                x = alpha_next * pred_x_start + torch.sqrt(1 - alpha_next**2 - sigma_t_next**2) * pred_noise + sigma_t_next * noise

        return x
