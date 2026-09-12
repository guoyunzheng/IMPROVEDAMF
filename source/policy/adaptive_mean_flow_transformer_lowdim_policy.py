from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from source.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from source.policy.base_lowdim_policy import BaseLowdimPolicy
from source.model.mean_flow.transformer_for_adaptive_mean_flow import TransformerForAdaptiveMeanFlow
from source.model.diffusion.mask_generator import LowdimMaskGenerator

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


class AdaptiveMeanFlowTransformerLowdimPolicy(BaseLowdimPolicy):
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
            time_cond="cat",
            time_dist=['lognorm', -0.4, 1.0],
            monotone_schedule_ratio = 0.5,
            t_start = 0.5,
            t_end = 0.01,
            p_curve = 1.0,
            refresh_every=None,
            flow_ratio=0.50,
            flow_matching_inference=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # create diffusion model
        input_dim = action_dim
        output_dim = input_dim
        cond_dim = obs_dim

        model = TransformerForAdaptiveMeanFlow(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            time_cond=time_cond,
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
        self.monotone_schedule_ratio = monotone_schedule_ratio
        self.flow_ratio = flow_ratio
        self.flow_matching_inference = flow_matching_inference
        self.t_start = t_start
        self.t_end = t_end
        self.p_curve = p_curve
        self.refresh_every = refresh_every
        self.steps_since_refresh = 0
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps

        self.prev_action_pred = None

    def model_call(self, trajectory, t, r, cond):
        return self.model(trajectory, t, r, cond)


    def sample_t_r(self, batch_size, device):

        T = self.horizon
        To = self.n_obs_steps

        n_regular = int(batch_size*(1-self.monotone_schedule_ratio))
        n_mon = batch_size - n_regular

        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(n_regular, T, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(n_regular, T, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_regular = np.maximum(samples[:, :, 0], samples[:, :, 1])
        r_regular = np.minimum(samples[:, :, 0], samples[:, :, 1])

        # Sample monotone schedule values
        p_min = 0.1
        p_max = 1.5
        ts_min = 0.2
        ts_max = 0.7
        te = 1.0

        p_vals = (np.random.rand(n_mon)*(p_max - p_min) + p_min)[:,None].astype(np.float32)
        ts_vals = (np.random.rand(n_mon)*(ts_max-ts_min) + ts_min)[:,None].astype(np.float32)
        t_trajs = (np.ones((n_mon,T))*ts_vals).astype(np.float32)
        t_curve = (np.arange(T-To+2)/(T-To+1))[None,:].repeat(n_mon,0).astype(np.float32)
        t_curve = ts_vals*(te/ts_vals)**(t_curve**p_vals)
        t_trajs[:,To-2:] = t_curve

        # Sample r values to use with monotone t values. Sample from lognorm with mean close to zero
        mu_r_traj = -1.5
        sigma_r_traj = 1.0
        normal_r_traj = (np.random.randn(n_mon,1).repeat(T,1).astype(np.float32) * sigma_r_traj + mu_r_traj)
        r_trajs = (1 / (1 + np.exp(-normal_r_traj)))*ts_vals

        t_np = np.concatenate((t_regular, t_trajs), axis=0)
        r_np = np.concatenate((r_regular, r_trajs), axis=0)

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    # ========= inference  ============
    def conditional_sample(self, trajectory, t_traj,
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        # trajectory = torch.randn(
        #     size=condition_data.shape,
        #     dtype=condition_data.dtype,
        #     device=condition_data.device,
        #     generator=generator)

        B = trajectory.shape[0]

        # set step values
        #t_vals = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=trajectory.device)
        r_traj = torch.zeros_like(t_traj)
        for i in range(self.num_inference_steps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # t = t_vals[[i]].expand(B)
            # r = t_vals[[i+1]].expand(B)
            t_ = rearrange(t_traj, 'B T -> B T 1').detach().clone()
            r_ = rearrange(r_traj, 'B T -> B T 1').detach().clone()

            # 2. predict model output
            if self.flow_matching_inference:
                v = self.model_call(trajectory, t_traj, t_traj, cond)
            else:
                v = self.model_call(trajectory, t_traj, r_traj, cond)
            trajectory = trajectory - (t_ - r_) * v


        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory


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

        t_s = self.t_start
        t_e = self.t_end
        p = self.p_curve

        # Create initial t trajectory
        t_traj = torch.ones((B,T))*t_s
        t_curve = torch.arange(T-To+2)/(T-To+1)
        t_curve = t_s*(t_e/t_s)**(t_curve**p)
        t_traj[:,To-2:] = t_curve
        t_traj = t_traj.to(device=device, dtype=dtype)

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

        # Create initial noisy action trajectory
        if self.prev_action_pred is None:
            trajectory = torch.randn_like(cond_data)
            t_traj = torch.ones_like(t_traj)

        elif (self.refresh_every is not None) and (self.steps_since_refresh >= self.refresh_every):
            trajectory = torch.randn_like(cond_data)
            t_traj = torch.ones_like(t_traj)
            self.steps_since_refresh = 0

        else:
            init_traj = torch.roll(self.prev_action_pred, -1, 1)
            init_traj[:,-1] = init_traj[:,-2]
            noise = torch.randn_like(cond_data)
            t_traj_ = t_traj.unsqueeze(-1).repeat(1,1,Da)
            #t_traj_[:,:,-1] = 1.0
            trajectory = t_traj_*noise + (1-t_traj_)*init_traj


        # run sampling
        nsample = self.conditional_sample(
            trajectory,
            t_traj,
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]

        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        self.prev_action_pred = naction_pred
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
        t_ = rearrange(t, 'B T -> B T 1').detach().clone()
        r_ = rearrange(r, 'B T -> B T 1').detach().clone()

        z = (1-t_)*nactions + t_*e
        v = e - nactions

        # reshape back to B, T, Do
        cond = obs[:,:To,:]

        # Approximated marginal velocity for jvp input
        #v_pred = self.model(z, t, t, cond=cond)

        jvp_args = (
            lambda z,t,r: self.model_call(z, t, r, cond),
            (z, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r))
        )
        u, dudt = torch.autograd.functional.jvp(*jvp_args, create_graph=True)
        u_tgt = v - (t_ - r_)*dudt
        error = u - u_tgt.detach()


        loss = adaptive_l2_loss(error)
        return loss

# Testing
