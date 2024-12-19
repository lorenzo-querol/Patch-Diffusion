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
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule.

    Proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


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


def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v.to(i.device), index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


class GaussianDiffusionTrainer(torch.nn.Module):
    def __init__(self, model, target="epsilon", schedule_name="linear", timesteps=1000):
        super().__init__()
        self.model = model
        self.T = timesteps
        self.target = target

        # generate T steps of beta
        self.register_buffer("beta_t", get_beta_schedule(schedule_name, timesteps))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def get_v(self, x, noise, t):
        return extract(self.signal_rate, t, x.shape) * noise - extract(self.noise_rate, t, x.shape) * x

    def sample_q(self, x_0, t, noise):
        x_t = extract(self.signal_rate, t, x_0.shape) * x_0 + extract(self.noise_rate, t, x_0.shape) * noise
        return x_t

    def forward(self, x_0, y=None, cls_mode=False):
        if cls_mode:
            inputs = torch.cat([x_0, x_0])

            noise_t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
            clean_t = torch.zeros(size=(x_0.shape[0],), device=x_0.device)
            t = torch.cat([noise_t, clean_t]).int()

            loss_weights = self.signal_rate[t]

            logits = self.model(inputs, t, cls_mode=cls_mode)
            ce_loss = F.cross_entropy(logits, y, reduction="none")

            return logits, ce_loss, (loss_weights * ce_loss).mean()
        else:
            # NOTE: This is a Classifier-Free Guidance (CFG) technique.
            # Set some labels to a negative/null class (i.e., does not exist)
            # mask = torch.rand(y.shape[0], device=y.device) < 0.1
            # y[mask] = self.model.label_dim

            # get a random training step $t \sim Uniform({1, ..., T})$
            t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

            epsilon = torch.randn_like(x_0)
            target_map = {"epsilon": epsilon, "x_0": x_0, "v": self.get_v(x_0, epsilon, t)}
            target = target_map[self.target]

            x_t = self.sample_q(x_0, t, epsilon)
            output = self.model(x_t, t, class_labels=y)
            loss = F.mse_loss(output, target[:, :3])

            return loss


class DDIMSampler(torch.nn.Module):
    def __init__(self, model, target="epsilon", schedule_name="linear", timesteps=1000):
        super().__init__()
        self.model = model
        self.T = timesteps
        self.target = target

        # generate T steps of beta
        self.register_buffer("beta_t", get_beta_schedule(schedule_name, timesteps))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

        self.register_buffer("signal_rate", torch.sqrt(self.alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - self.alpha_t_bar))

    @torch.no_grad()
    def sample_one_step(self, x_t, pos, class_labels, time_step: int, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict noise using model
        epsilon_theta_t = self.model(torch.cat([x_t, pos], dim=1), t, class_labels=class_labels)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
            torch.sqrt(alpha_t_prev / alpha_t) * x_t + (torch.sqrt(1 - alpha_t_prev - sigma_t**2) - torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t + sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def sample_one_step_v(self, x_t, pos, class_labels, time_step: int, prev_time_step: int, eta: float, clip_denoised: bool = True, clip_value: int = 3):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape).clip(min=0)

        sigma_t = extract(1 - self.alpha_t_bar, t, x_t.shape)
        sigma_t_prev = extract(1 - self.alpha_t_bar, prev_t, x_t.shape).clip(min=0)

        v = self.model(torch.cat([x_t, pos], dim=1), t, class_labels=class_labels)

        pred = x_t * alpha_t - v * sigma_t

        if clip_denoised:
            pred = pred.clip(-clip_value, clip_value)

        epsilon_t = (x_t - alpha_t * pred) / sigma_t

        if clip_denoised:
            epsilon_t = epsilon_t.clip(-clip_value, clip_value)

        ddim_sigma = eta * (sigma_t_prev**2 / sigma_t**2).sqrt() * (1 - alpha_t**2 / alpha_t_prev**2).sqrt()
        adjusted_sigma = (sigma_t_prev**2 - ddim_sigma**2).sqrt()
        pred = pred * alpha_t_prev + epsilon_t * adjusted_sigma

        if eta:
            pred += torch.randn_like(pred) * ddim_sigma

        return pred

    @torch.no_grad()
    def forward(self, x_t, pos, class_labels, steps: int = 1, method="linear", eta=0.0, only_return_x_0: bool = True, interval: int = 1):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures, according to `step`.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int64)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]
        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:

                if self.target == "v":
                    x_t = self.sample_one_step_v(x_t, pos, class_labels, time_steps[i], time_steps_prev[i], eta)
                elif self.target == "epsilon":
                    x_t = self.sample_one_step(x_t, pos, class_labels, time_steps[i], time_steps_prev[i], eta)
                else:
                    raise NotImplementedError(f"target {self.target} is not implemented!")

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(x_t)

                sampling_steps.set_postfix(ordered_dict={"Step": i + 1, "Sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]

        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]


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
        assert schedule_name in ["linear", "cosine"], f"Unknown schedule name: {schedule_name}, choose from ['linear', 'cosine']"
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

        # Log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
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

    def training_losses(self, net, x_start, t, target="eps", model_kwargs={}):
        """Compute the training loss for the diffusion model.

        :param net: Diffusion model.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: A batch of timesteps.
        :param target: the target to predict. One of "eps", "x_start", or "v".
        :param model_kwargs: additional keyword arguments to pass to the model.
        :return loss: the training loss.
        """
        assert target in ["eps", "x_start", "v"], f"Unknown target: {target}, choose from ['eps', 'x_start', 'v']"

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

    def sample_images_ddim_v(self, net, shape, pos, labels, device, eta=0.0, timesteps=50):
        # Start from pure noise
        x = torch.randn(shape).to(device)

        # Select timesteps for DDIM
        skip = self.num_timesteps // timesteps
        seq = list(range(0, self.num_timesteps, skip))

        for i in range(len(seq) - 1):
            t = seq[len(seq) - 1 - i]
            t_next = seq[len(seq) - 2 - i] if i < len(seq) - 1 else 0

            t_batch = torch.full((shape[0],), t, device=device)
            t_next_batch = torch.full((shape[0],), t_next, device=device)

            with torch.no_grad():
                # Model predicts v
                pred_v = net(torch.cat((x, pos), dim=1), t_batch, class_labels=labels)

                # Get alphas and sigmas for current and next timestep
                alpha_t = self._extract(self.sqrt_alphas_cumprod, t_batch, x.shape)
                alpha_next = self._extract(self.sqrt_alphas_cumprod, t_next_batch, x.shape)
                sigma_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)

                # Convert v to x_0 and epsilon predictions
                pred_eps = (x - alpha_t * pred_v) / sigma_t
                pred_x0 = (pred_v * sigma_t - x) / (-alpha_t)

                # DDIM formula
                sigma_t_next = eta * torch.sqrt((1 - alpha_next**2) / (1 - alpha_t**2) * (1 - alpha_t**2 / alpha_next**2))

                # No noise when t_next == 0 or eta == 0
                noise = 0 if t_next == 0 or eta == 0 else torch.randn_like(x)

                # DDIM update step
                x = alpha_next * pred_x0 + torch.sqrt(1 - alpha_next**2 - sigma_t_next**2) * pred_eps + sigma_t_next * noise

        return x
