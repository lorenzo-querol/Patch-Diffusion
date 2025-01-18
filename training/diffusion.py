import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.utils import get_beta_schedule, extract


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

        # NOTE: This is a Classifier-Free Guidance (CFG) technique. Set some labels to a negative/null class (i.e., does not exist)
        mask = torch.rand(y.shape[0], device=y.device) < 0.1
        y[mask] = self.model.label_dim

        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        epsilon = torch.randn_like(x_0)
        target_map = {"epsilon": epsilon, "x_0": x_0, "v": self.get_v(x_0, epsilon, t)}
        target = target_map[self.target]

        x_t = self.sample_q(x_0, t, epsilon)
        output = self.model(x_t, t, class_labels=y)
        num_channels = output.shape[1]
        loss = F.mse_loss(output, target[:, :num_channels])

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
    def sample_one_step(self, x_t, pos, class_labels, time_step: int, prev_time_step: int, eta: float, guidance_scale: float = 1.0):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict noise using model
        epsilon_theta_t = self.model(torch.cat([x_t, pos], dim=1), t, class_labels=class_labels)

        # Classifier-Free Guidance
        if guidance_scale > 1.0:
            uncond_labels = torch.ones_like(class_labels, device=x_t.device, dtype=torch.long)
            epsilon_uncond = self.model(torch.cat([x_t, pos], dim=1), t, class_labels=uncond_labels)
            epsilon_theta_t = epsilon_uncond + guidance_scale * (epsilon_theta_t - epsilon_uncond)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
            torch.sqrt(alpha_t_prev / alpha_t) * x_t + (torch.sqrt(1 - alpha_t_prev - sigma_t**2) - torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t + sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def sample_one_step_x_0(self, x_t, pos, class_labels, time_step: int, prev_time_step: int, eta: float, guidance_scale: float = 1.0):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # Get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # Predict x_0 using the model
        x_0 = self.model(torch.cat([x_t, pos], dim=1), t, class_labels=class_labels)

        # Classifier-Free Guidance
        if guidance_scale > 1.0:
            uncond_labels = torch.ones_like(class_labels, device=x_t.device, dtype=torch.long)
            x_0_uncond = self.model(torch.cat([x_t, pos], dim=1), t, class_labels=uncond_labels)
            x_0 = x_0_uncond + guidance_scale * (x_0 - x_0_uncond)

        # Compute x_{t-1} using the predicted x_0
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = torch.sqrt(alpha_t_prev) * x_0 + torch.sqrt(1 - alpha_t_prev - sigma_t**2) * ((x_t - torch.sqrt(alpha_t) * x_0) / torch.sqrt(1 - alpha_t)) + sigma_t * epsilon_t

        return x_t_minus_one

    @torch.no_grad()
    def sample_one_step_v(self, x_t, pos, class_labels, time_step: int, prev_time_step: int, eta: float, guidance_scale: float = 1.0, clip_denoised: bool = True, clip_value: int = 3):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape).clip(min=0)

        sigma_t = extract(1 - self.alpha_t_bar, t, x_t.shape)
        sigma_t_prev = extract(1 - self.alpha_t_bar, prev_t, x_t.shape).clip(min=0)

        v = self.model(torch.cat([x_t, pos], dim=1), t, class_labels=class_labels)

        # Classifier-Free Guidance
        if guidance_scale > 1.0:
            uncond_labels = torch.ones_like(class_labels, device=x_t.device, dtype=torch.long)
            v_uncond = self.model(torch.cat([x_t, pos], dim=1), t, class_labels=uncond_labels)
            v = v_uncond + guidance_scale * (v - v_uncond)

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
    def forward(self, x_t, pos, class_labels, steps: int = 1, method="linear", eta=0.0, guidance_scale: float = 1.0, only_return_x_0: bool = True, interval: int = 1):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            guidance_scale: Scale for classifier-free guidance.
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
        with tqdm(reversed(range(0, steps)), desc="Sampling", colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:

                if self.target == "epsilon":
                    x_t = self.sample_one_step(x_t, pos, class_labels, time_steps[i], time_steps_prev[i], eta, guidance_scale)
                elif self.target == "x_0":
                    x_t = self.sample_one_step_x_0(x_t, pos, class_labels, time_steps[i], time_steps_prev[i], eta, guidance_scale)
                elif self.target == "v":
                    x_t = self.sample_one_step_v(x_t, pos, class_labels, time_steps[i], time_steps_prev[i], eta, guidance_scale)
                else:
                    raise NotImplementedError(f"target {self.target} is not implemented!")

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(x_t)

                sampling_steps.set_postfix(ordered_dict={"Step": i + 1, "Sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]

        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]
