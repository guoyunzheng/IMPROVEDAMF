from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from source.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from source.policy.base_lowdim_policy import BaseLowdimPolicy
from source.model.mean_flow.transformer_for_mean_flow import TransformerForMeanFlow
from source.model.diffusion.mask_generator import LowdimMaskGenerator
from functools import partial

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (w.detach() * loss).mean()


class MeanFlowTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            action_dim,
            obs_dim,
            # task params
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=1,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_dist=['lognorm', -0.4, 1.0],
            flow_ratio=0.50,
            flow_matching_inference=False,
            use_warm_start=False,
            warm_start_t=0.5,
            refresh_every=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # create diffusion model
        input_dim = action_dim
        output_dim = input_dim
        cond_dim = obs_dim

        model = TransformerForMeanFlow(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            n_cond_layers=n_cond_layers
        )

        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.time_dist = time_dist
        self.flow_ratio = flow_ratio
        self.flow_matching_inference = flow_matching_inference
        self.use_warm_start = use_warm_start
        self.warm_start_t = warm_start_t
        self.refresh_every = refresh_every
        self.prev_action_pred = None
        self.steps_since_refresh = 0
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps


    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None, generator=None,
            trajectory=None,
            start_t=1.0,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model

        if trajectory is None:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator)

        B = trajectory.shape[0]

        # set step values
        t_vals = torch.linspace(
            float(start_t), 0.0, self.num_inference_steps + 1,
            device=trajectory.device,
            dtype=trajectory.dtype)
        for i in range(self.num_inference_steps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            t = t_vals[[i]].expand(B)
            r = t_vals[[i+1]].expand(B)
            t_ = rearrange(t, 'B -> B 1 1').detach().clone()
            r_ = rearrange(r, 'B -> B 1 1').detach().clone()

            # 2. predict model output
            if self.flow_matching_inference:
                v = model(trajectory, t, t, cond)
            else:
                v = model(trajectory, t, r, cond)
            trajectory = trajectory - (t_ - r_) * v


        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def _shift_previous_prediction(self, prev_action_pred: torch.Tensor) -> torch.Tensor:
        shift = self.n_action_steps
        if shift <= 0:
            return prev_action_pred.detach().clone()

        init_traj = torch.roll(prev_action_pred, shifts=-shift, dims=1)
        fill_value = prev_action_pred[:, -1:, :]
        n_fill = min(shift, init_traj.shape[1])
        init_traj[:, -n_fill:, :] = fill_value
        return init_traj.detach()

    def _prev_prediction_matches(self, batch_size, action_dim, device, dtype):
        return (
            self.prev_action_pred is not None
            and self.prev_action_pred.shape[0] == batch_size
            and self.prev_action_pred.shape[1] == self.horizon
            and self.prev_action_pred.shape[2] == action_dim
            and self.prev_action_pred.device == device
            and self.prev_action_pred.dtype == dtype
        )

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet

        value = next(iter(obs_dict.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        nobs_features = self.normalizer['obs'].normalize(obs_dict['obs'])

        # reshape back to B, To, Do
        cond = nobs_features[:,:To,:]
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        trajectory = None
        start_t = 1.0
        if self.use_warm_start:
            should_refresh = (
                self.refresh_every is not None
                and self.steps_since_refresh >= self.refresh_every
            )
            if should_refresh or (not self._prev_prediction_matches(B, Da, device, dtype)):
                self.prev_action_pred = None
                self.steps_since_refresh = 0

            if self.prev_action_pred is not None:
                start_t = self.warm_start_t
                init_traj = self._shift_previous_prediction(self.prev_action_pred)
                noise = torch.randn_like(cond_data)
                trajectory = start_t * noise + (1.0 - start_t) * init_traj

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            trajectory=trajectory,
            start_t=start_t,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]

        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        self.prev_action_pred = naction_pred.detach()
        self.steps_since_refresh += 1

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def reset_history(self):
        self.prev_action_pred = None
        self.steps_since_refresh = 0

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        nactions = nbatch['action']
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        e = torch.randn_like(nactions)
        t, r = self.sample_t_r(batch_size, nactions.device)
        t_ = rearrange(t, 'B -> B 1 1').detach().clone()
        r_ = rearrange(r, 'B -> B 1 1').detach().clone()

        z = (1-t_)*nactions + t_*e
        v = e - nactions

        # reshape back to B, T, Do
        cond = obs[:,:To,:]

        # Approximated marginal velocity for jvp input
        #v_pred = self.model(z, t, t, cond=cond)

        model_partial = partial(self.model, cond=cond)
        if self.flow_ratio == 1.0:
            # With r=t the target is v; no Mean Flow derivative is needed.
            u = model_partial(z, t, r)
            u_tgt = v
        else:
            jvp_args = (
                lambda z,t,r: model_partial(z, t, r),
                (z, t, r),
                (v, torch.ones_like(t), torch.zeros_like(r))
            )
            u, dudt = torch.autograd.functional.jvp(*jvp_args, create_graph=True)
            u_tgt = v - (t_ - r_)*dudt
        error = u - u_tgt.detach()


        loss = adaptive_l2_loss(error)
        return loss
