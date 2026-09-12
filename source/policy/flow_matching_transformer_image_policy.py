import numpy as np
from typing import Dict, Tuple
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from functools import partial
import math
from torch.autograd.functional import vjp

from source.model.common.normalizer import LinearNormalizer
from source.policy.base_image_policy import BaseImagePolicy
from source.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from source.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from source.model.diffusion.mask_generator import LowdimMaskGenerator

class FlowMatchingTransformerImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            num_inference_steps=10,
            time_dist=['lognorm', -0.4, 1.0],
            pred_action_steps_only=False,
            use_rtc=False,
            rtc_beta=5.0,
            # vision pretraining params
            freeze_obs_encoder: bool = False,
            # parameters passed to step
            **kwargs):
        super().__init__(
            obs_encoder=obs_encoder,
            n_obs_steps=n_obs_steps,
            freeze_obs_encoder=freeze_obs_encoder
        )
        if pred_action_steps_only:
            assert obs_as_cond

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
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
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
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
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.time_dist = time_dist
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps

        self.prev_action_preds = None # Used by some sampling schemes
        self.use_rtc = use_rtc
        self.rtc_beta = rtc_beta


    def sample_t(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid


        t = torch.tensor(samples, device=device)
        return t

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

        # set step values
        t_vals = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=trajectory.device)
        dt = t_vals[0] - t_vals[1]
        for i in range(self.num_inference_steps):
            t = t_vals[i]
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = trajectory - model_output * dt

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    # ========= rtc inference  ============

    def guided_step_vjp(self,
        A, o, t, net,
        A_prev, W,
        beta, r_t,
        dt
    ):
        """
        A:      (..., H, Da) current action chunk at time t
        A_prev: (..., H, Da) previous chunk (right-padded to H)
        W:      broadcastable to A (e.g. (..., H, 1) or (..., H, Da))
        t:      scalar float tensor in [0,1], here decreasing from 1->0
        """


        # Define function whose output is y, and aux is v (so we reuse it)
        def denoise_map(A_):
            v_ = net(A_, t, o)                 # ONE forward pass
            y_ = A_ - t * v_           # f(A') in your pseudocode
            return y_, v_                           # (out, aux)

        # IMPORTANT: A should be a "fresh" leaf for this step
        A_leaf = A.detach().requires_grad_(True)

        # vjp with aux: returns output, vjp_fn, aux
        y, vjp_fn, v = torch.func.vjp(denoise_map, A_leaf, has_aux=True)

        # weighted error term: e = (A_prev - y) * diag(W)
        e = (A_prev - y) * W

        # VJP: g = J^T e
        (g,) = vjp_fn(e)

        # coefficient used in your update (plug in your exact expression)
        # min(beta, (1-t) / (t * r_t^2)) is what you hinted earlier
        coeff = torch.minimum(
            torch.as_tensor(beta, device=A.device, dtype=A.dtype),
            (1.0 - t) / (t * (r_t**2))
        )

        # Euler step with dt < 0 since t decreases
        A_next = A_leaf - dt * (v + coeff * g)

        # Return detached state for next iteration (inference-time solver)
        return A_next.detach(), y.detach(), v.detach(), g.detach()

    def rtc_conditional_sample(self,
            condition_data, condition_mask,
            cond=None, inference_delay=0, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        Ta = self.n_action_steps
        T = self.horizon
        beta=self.rtc_beta

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        cached_action_preds = torch.roll(self.prev_action_preds, shifts=-Ta, dims=1)
        cached_action_preds[:,-Ta:,:] = cached_action_preds[:,[-Ta-1],:]

        #Calculate weight matrix W
        W = torch.zeros_like(trajectory)
        for i in range(T):
            if i < inference_delay:
                W[:,i,:self.action_dim] = 1.0
            elif i < T - Ta:
                ci =  (T - Ta - i) / (T - Ta - inference_delay + 1)
                wi = ci*(math.exp(ci)-1)/(math.exp(1)-1)
                W[:,i,:self.action_dim] = wi
            else:
                W[:,i,:self.action_dim] = 0.0

        # set step values
        t_vals = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=trajectory.device)
        dt = t_vals[0] - t_vals[1]
        for i in range(self.num_inference_steps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            t = t_vals[i]
            r_t = t**2/((1-t)**2 + t**2)

            trajectory,_,_,_ = self.guided_step_vjp(
                A=trajectory,
                o=cond,
                t=t_vals[i],
                net=model,
                A_prev=cached_action_preds,
                W=W,
                beta=beta,
                r_t=r_t,
                dt=dt
            )

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory


    def reset_history(self):
        self.prev_action_preds = None


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], inference_delay: int = 0) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'past_action' not in obs_dict # not implemented yet
        value = next(iter(obs_dict.values()))
        B = value.shape[0]
        To = self.n_obs_steps
        Do = self.obs_feature_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            nobs_features = self.process_obs(obs_dict)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            nobs_features = self.process_obs(obs_dict)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        if self.use_rtc and not (self.prev_action_preds is None):
            nsample = self.rtc_conditional_sample(
                condition_data=cond_data,
                condition_mask=cond_mask,
                cond=cond,
                inference_delay=inference_delay,
                **self.kwargs)
        else:
            nsample = self.conditional_sample(
                cond_data,
                cond_mask,
                cond=cond,
                **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        self.prev_action_preds = naction_pred.detach().clone()
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

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
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            nobs_features = self.process_obs(batch['obs'])
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            nobs_features = self.process_obs(batch['obs'])
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        t = self.sample_t(bsz, device=trajectory.device)
        t_ = rearrange(t, 'B -> B 1 1').detach().clone()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        z = (1-t_)*trajectory + t_*noise
        v = noise - trajectory

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        z[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual
        pred = self.model(z, t, cond)


        loss = F.mse_loss(pred, v, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


# Testing
