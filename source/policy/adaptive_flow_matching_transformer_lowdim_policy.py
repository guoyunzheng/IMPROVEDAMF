from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, reduce

from source.model.common.normalizer import LinearNormalizer
from source.model.diffusion.transformer_for_adaptive_flow import (
    TransformerForAdaptiveFlow,
)
from source.policy.base_lowdim_policy import BaseLowdimPolicy


class AdaptiveFlowMatchingTransformerLowdimPolicy(BaseLowdimPolicy):
    """Flow matching with independent per-action flow times and noise."""

    def __init__(
            self,
            action_dim,
            obs_dim,
            horizon=20,
            n_action_steps=1,
            n_obs_steps=10,
            num_inference_steps=10,
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.2,
            causal_attn=True,
            time_cond="AdaLNZero",
            time_dist=("lognorm", -0.4, 1.0),
            monotone_schedule_ratio=0.5,
            t_start=0.5,
            t_end=1.0,
            p_curve=1.5,
            refresh_every=None,
            **kwargs):
        super().__init__()
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not (0 < t_start <= 1 and 0 < t_end <= 1):
            raise ValueError("t_start and t_end must lie in (0, 1]")
        if n_action_steps < 1 or n_action_steps > horizon:
            raise ValueError("n_action_steps must be in [1, horizon]")
        if n_obs_steps < 2 or horizon < n_obs_steps:
            raise ValueError("Adaptive Mean Flow-style scheduling requires 2 <= n_obs_steps <= horizon")
        if not 0 <= monotone_schedule_ratio <= 1:
            raise ValueError("monotone_schedule_ratio must lie in [0, 1]")

        self.model = TransformerForAdaptiveFlow(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            time_cond=time_cond,
            cond_dim=obs_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            n_cond_layers=n_cond_layers,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.time_dist = tuple(time_dist)
        self.monotone_schedule_ratio = monotone_schedule_ratio
        self.t_start = t_start
        self.t_end = t_end
        self.p_curve = p_curve
        self.refresh_every = refresh_every
        self.kwargs = kwargs

        self.prev_action_pred = None
        self.steps_since_refresh = 0

    def _sample_training_times(self, batch_size, device, dtype):
        n_independent = int(batch_size * (1 - self.monotone_schedule_ratio))
        n_monotone = batch_size - n_independent

        shape = (n_independent, self.horizon)
        if self.time_dist[0] == "uniform":
            independent = torch.rand(shape, device=device, dtype=dtype)
        elif self.time_dist[0] == "lognorm":
            mu, sigma = self.time_dist[-2:]
            normal = torch.randn(shape, device=device, dtype=dtype) * sigma + mu
            independent = torch.sigmoid(normal)
        else:
            raise ValueError(f"Unsupported time distribution {self.time_dist[0]}")

        # Match Adaptive Mean Flow's randomized monotone training schedules exactly.
        p = torch.rand((n_monotone, 1), device=device, dtype=dtype) * 1.4 + 0.1
        t_start = torch.rand((n_monotone, 1), device=device, dtype=dtype) * 0.5 + 0.2
        monotone = t_start.expand(-1, self.horizon).clone()
        progress = torch.arange(
            self.horizon - self.n_obs_steps + 2, device=device, dtype=dtype
        ) / (self.horizon - self.n_obs_steps + 1)
        curve = t_start * (1.0 / t_start) ** (progress.unsqueeze(0) ** p)
        monotone[:, self.n_obs_steps - 2:] = curve

        return torch.cat((independent, monotone), dim=0)

    def _get_inference_schedule(self, batch_size, device, dtype):
        """Construct the same exponential monotone schedule used by Adaptive Mean Flow."""
        T, To = self.horizon, self.n_obs_steps
        schedule = torch.full((batch_size, T), self.t_start,
                              device=device, dtype=dtype)
        progress = torch.arange(T - To + 2, device=device, dtype=dtype) / (T - To + 1)
        curve = self.t_start * (self.t_end / self.t_start) ** (progress ** self.p_curve)
        schedule[:, To - 2:] = curve
        return schedule

    def _shift_action_buffer(self):
        shift = self.n_action_steps
        shifted = torch.roll(self.prev_action_pred, shifts=-shift, dims=1)
        shifted[:, -shift:] = self.prev_action_pred[:, [-1]]
        return shifted

    def conditional_sample(self, trajectory, initial_t, cond):
        initial_t = initial_t.to(device=trajectory.device, dtype=trajectory.dtype)
        for step in range(self.num_inference_steps):
            fraction = step / self.num_inference_steps
            next_fraction = (step + 1) / self.num_inference_steps
            current_t = initial_t * (1 - fraction)
            next_t = initial_t * (1 - next_fraction)
            velocity = self.model(trajectory, current_t, cond)
            trajectory = trajectory - rearrange(
                current_t - next_t, "b t -> b t 1") * velocity
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]):
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action conditioning is not supported")
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        if Do != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {Do}")
        cond = nobs[:, :self.n_obs_steps]
        shape = (B, self.horizon, self.action_dim)

        refresh = (
            self.prev_action_pred is None
            or (self.refresh_every is not None
                and self.steps_since_refresh >= self.refresh_every)
        )
        noise = torch.randn(shape, device=self.device, dtype=self.dtype)
        if refresh:
            trajectory = noise
            initial_t = torch.ones((B, self.horizon), device=self.device, dtype=self.dtype)
            self.steps_since_refresh = 0
        else:
            initial_t = self._get_inference_schedule(
                B, device=self.device, dtype=self.dtype)
            shifted = self._shift_action_buffer()
            trajectory = (
                (1 - initial_t.unsqueeze(-1)) * shifted
                + initial_t.unsqueeze(-1) * noise
            )

        naction_pred = self.conditional_sample(trajectory, initial_t, cond)
        self.prev_action_pred = naction_pred.detach()
        self.steps_since_refresh += 1
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

    def compute_loss(self, batch):
        if "valid_mask" in batch:
            raise NotImplementedError("valid_mask is not supported")
        nbatch = self.normalizer.normalize(batch)
        action, obs = nbatch["action"], nbatch["obs"]
        cond = obs[:, :self.n_obs_steps]
        B = action.shape[0]
        noise = torch.randn_like(action)
        t = self._sample_training_times(B, action.device, action.dtype)
        t_view = t.unsqueeze(-1)
        noisy_action = (1 - t_view) * action + t_view * noise
        target_velocity = noise - action
        pred_velocity = self.model(noisy_action, t, cond)
        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        return reduce(loss, "b ... -> b", "mean").mean()

    def reset_history(self):
        self.prev_action_pred = None
        self.steps_since_refresh = 0

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]):
        groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        return torch.optim.AdamW(groups, lr=learning_rate, betas=tuple(betas))
