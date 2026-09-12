from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from source.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from source.policy.base_image_policy import BaseImagePolicy
from source.model.mean_flow.transformer_for_mean_flow import TransformerForMeanFlow
from source.model.diffusion.mask_generator import LowdimMaskGenerator
from source.model.vision.model_getter import get_resnet
from source.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from source.common.pytorch_util import dict_apply
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


class MeanFlowTransformerImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            # task params
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=1,
            # vision pretraining params
            freeze_obs_encoder: bool = False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_dist=['lognorm', -0.4, 1.0],
            flow_ratio=0.75,
            flow_matching_inference=False,
            use_ptp=False,
            ptp_n_sample=32,
            ptp_only_past = False,
            # parameters passed to step
            **kwargs):
        super().__init__(
            obs_encoder=obs_encoder,
            n_obs_steps=n_obs_steps,
            freeze_obs_encoder=freeze_obs_encoder
        )

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim
        output_dim = input_dim
        cond_dim = obs_feature_dim

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

        self.obs_encoder = obs_encoder
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
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.time_dist = time_dist
        self.flow_ratio = flow_ratio
        self.flow_matching_inference = flow_matching_inference
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.shape_meta = shape_meta

        self.use_ptp = use_ptp
        self.ptp_n_sample = ptp_n_sample
        self.ptp_only_past = ptp_only_past
        self.prev_action_pred: torch.Tensor = None

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
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        B = trajectory.shape[0]

        # set step values
        t_vals = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=trajectory.device)
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


    def predict_single_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        nobs_features = self.process_obs(obs_dict)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(B, To, -1)
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def ptp_predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = self.device
        dtype = self.dtype
        N = self.ptp_n_sample
        B = next(iter(obs_dict.values())).shape[0]
        h_past_overlap = self.n_obs_steps - 1
        h_future_overlap = self.horizon - h_past_overlap - self.n_action_steps
        w_past = torch.ones(h_past_overlap)
        if self.ptp_only_past:
            w_future = torch.zeros(h_future_overlap)
        else:
            f_range = torch.arange(h_future_overlap,0,-1)
            c_range = f_range/(h_future_overlap+1)
            w_future = c_range*(torch.exp(c_range)-1)/(torch.exp(torch.Tensor([1.0]))-1)
        w = torch.cat([w_past, w_future], dim=0).unsqueeze(1).to(device=device, dtype=dtype)

        expanded_obs_dict = dict_apply(obs_dict, lambda x: x.repeat_interleave(N, dim=0))
        preds = self.predict_single_action(expanded_obs_dict)
        preds = dict_apply(preds, lambda x: x.reshape(B, N, *x.shape[1:]))
        diffs = preds['action_pred'][:,:,:-self.n_action_steps,:] - self.prev_action_pred[:,self.n_action_steps:,:].unsqueeze(1)
        diffs = torch.pow(diffs, 2)
        weighted_dists = torch.sum(diffs*w, dim=(-1,-2))
        min_idx = torch.argmin(weighted_dists, dim=1)
        best_preds = dict_apply(preds, lambda x: x[torch.arange(B), min_idx, ...])
        self.prev_action_pred = best_preds['action_pred'].detach()
        result = {
            'action': best_preds['action'],
            'action_pred': best_preds['action_pred']
        }
        return result

    def predict_action(self, obs_dict):
        if self.use_ptp:
            if self.prev_action_pred is None:
                out = self.predict_single_action(obs_dict)
                self.prev_action_pred = out['action_pred'].detach()
                return out
            else:
                return self.ptp_predict_action(obs_dict)
        else:
            return self.predict_single_action(obs_dict)

    def reset_history(self):
        self.prev_action_pred = None

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            obs_encoder_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
                "params": self.obs_encoder.parameters(),
                "weight_decay": obs_encoder_weight_decay
            })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nactions = self.normalizer['action'].normalize(batch['action'])
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

        # reshape B, T, ... to B*T
        nobs_features = self.process_obs(batch['obs'])
        # reshape back to B, T, Do
        cond = nobs_features.reshape(batch_size, To, -1)

        model_partial = partial(self.model, cond=cond)
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
