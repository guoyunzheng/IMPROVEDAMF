from typing import Dict
from einops import rearrange, reduce
import torch.nn.functional as F
import math
import torch

from source.model.common.normalizer import LinearNormalizer
from source.model.diffusion.mask_generator import LowdimMaskGenerator
from source.model.streaming_diffusion.schedulers import DDPMStreamingDiffusionScheduler
from source.model.streaming_diffusion.transformer_for_streaming_diffusion import TransformerForStreamingDiffusion
from source.policy.base_lowdim_policy import BaseLowdimPolicy


class StreamingDiffusionTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(
        self,
        noise_scheduler: DDPMStreamingDiffusionScheduler,
        obs_dim,
        action_dim,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        n_layer=8,
        n_cond_layers=0,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.2,
        causal_attn=True,
        time_cond="cat",
        temporally_constant_weight=0.0,
        temporally_increasing_weight=0.0,
        temporally_random_weights=0.0,
        chunk_wise_weight=1.0,
        buffer_init="zero",
        **kwargs,
    ):
        BaseLowdimPolicy.__init__(self)
        if not obs_as_global_cond:
            raise NotImplementedError("StreamingDiffusionTransformerLowdimPolicy only supports obs_as_global_cond=True.")

        self.model = TransformerForStreamingDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            time_cond=time_cond,
            cond_dim=obs_dim,
            n_layer=n_layer,
            n_cond_layers=n_cond_layers,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
        )
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        if horizon < n_obs_steps - 1 + n_action_steps:
            raise ValueError("Horizon must be longer than (To-1) + Ta.")
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.action_buffer = None
        self.buffer_diff_steps = None

        self.temporally_constant_weight = temporally_constant_weight
        self.temporally_increasing_weight = temporally_increasing_weight
        self.temporally_random_weights = temporally_random_weights
        self.chunk_wise_weight = chunk_wise_weight

        assert math.isclose(
            self.temporally_constant_weight
            + self.temporally_increasing_weight
            + self.temporally_random_weights
            + self.chunk_wise_weight,
            1,
            rel_tol=1e-2,
        )

        self.buffer_init = buffer_init
        assert self.buffer_init in ["zero", "constant", "denoise"]
        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))

    def _get_obs_cond(self, obs):
        return obs[:, : self.n_obs_steps, :]

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            learning_rate: float,
            betas
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=tuple(betas))


    def predict_action(self, obs_dict):
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        assert Do == self.obs_dim

        device = self.device
        dtype = self.dtype
        local_cond = None
        global_cond = self._get_obs_cond(nobs)
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if (
            self.action_buffer is not None
            and (
                self.action_buffer.shape[0] != B
                or self.action_buffer.shape[-1] != Da
                or self.action_buffer.device != device
                or self.action_buffer.dtype != dtype
            )
        ):
            self.reset_history()

        if self.action_buffer is None:
            if self.buffer_init == "zero":
                self.initialize_buffer_as_zero(shape, generator=None)
            elif self.buffer_init == "constant":
                self.initialize_buffer_as_constant(shape, cond_data, generator=None)
            elif self.buffer_init == "denoise":
                self.initialize_buffer(
                    shape,
                    cond_data,
                    cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    generator=None,
                    **self.kwargs,
                )
            else:
                raise ValueError(f"Unsupported buffer initialization {self.buffer_init}")

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        nactions = nbatch["action"]
        obs = nbatch["obs"]
        trajectory = nactions
        cond_data = trajectory
        global_cond = self._get_obs_cond(obs)
        local_cond = None

        condition_mask = self.mask_generator(trajectory.shape)

        B = trajectory.shape[0]
        Tp = trajectory.shape[1]
        To = self.n_obs_steps
        Ta = self.n_action_steps

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        diff_steps = self.get_diff_steps_training(B, Tp, To, Ta)

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, diff_steps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory,
            diff_steps,
            cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1)
        return loss.mean()

    def reset_history(self):
        self.action_buffer = None
        self.buffer_diff_steps = None

    def push_buffer(self, new_value, new_sigma_indices):
        """
        Adds the new value to the end of the action buffer like a FIFO queue.
        Also add the corresponding values to buffer_sigma_indices.
        Args:
            new_value: (B, N, Da)
        """
        self.action_buffer = torch.cat([self.action_buffer, new_value], dim=1)
        self.buffer_diff_steps = torch.cat(
            [self.buffer_diff_steps, new_sigma_indices], dim=1
        )

    @torch.no_grad()
    def initialize_buffer(
        self,
        shape,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        """
        Args:
            shape: The shape of the action buffer, for obs_as_global_cond, it is (B, T, Da), for inpainting (B, T, Da+Do)
            condition_data: The conditioning data, shape corresponds to shape
            condition_mask: The mask for the conditioning data, shape corresponds to shape
        """
        # if first step, initialize action buffer as pure noise with std. dev.
        # sigma_max, and do T_a number of steps

        B, Tp, _ = shape
        Ta = self.n_action_steps
        To = self.n_obs_steps
        N = self.num_inference_steps
        scheduler = self.noise_scheduler
        if self.action_buffer is None:
            trajectory = torch.randn(
                size=shape, dtype=self.dtype, device=self.device, generator=generator
            )

            # set step values
            scheduler.set_timesteps(self.num_inference_steps)

            # 1. Denoise buffer
            for t in scheduler.timesteps.to(device=self.device, dtype=torch.long):
                # 1. apply conditioning
                trajectory[condition_mask] = condition_data[condition_mask]

                # 2. predict model output
                model_output = self.model(
                    trajectory,
                    t.repeat(B, Tp),
                    local_cond=local_cond,
                    global_cond=global_cond,
                )

                # 3. compute previous image: x_t -> x_t-1
                trajectory = scheduler.step(
                    model_output,
                    t.repeat(B, Tp),
                    trajectory,
                    generator=generator,
                    **kwargs,
                ).prev_sample

            self.action_buffer = trajectory

            # 2. Find the diffusion steps (B, T) for each element in the buffer

            # Calculate chunk indices i
            num_complete_chunks = math.floor((Tp - (To - 1)) / Ta)
            incomplete_index = Ta * num_complete_chunks + (To - 1)
            indices = torch.arange(0, Tp, device=self.device)
            chunk_indices = torch.div((indices - (To - 1)), Ta, rounding_mode="floor")
            chunk_indices = torch.where(indices < To, 0, chunk_indices)
            chunk_indices = torch.where(
                indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
            )
            chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

            # Calculate corresponding diffusion levels
            diff_steps = (
                torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1
            ).long()
            self.buffer_diff_steps = diff_steps

            # 3. Noise action buffer corresponding to the sigma_indices
            # Note: we don't noise the observation steps + 1st chunk
            buffer_to_be_noised = self.action_buffer[:, Ta + (To - 1) :]
            diff_steps = self.buffer_diff_steps[:, Ta + (To - 1) :]
            noise = torch.randn(
                size=buffer_to_be_noised.shape, dtype=self.dtype, device=self.device
            )
            self.action_buffer[:, Ta + (To - 1) :] = scheduler.add_noise(
                buffer_to_be_noised, noise, diff_steps
            )

            # Apply conditioning
            self.action_buffer[condition_mask] = condition_data[condition_mask]

    def initialize_buffer_as_zero(self, shape, generator=None):
        N = self.num_inference_steps
        B, Tp, _ = shape
        Ta = self.n_action_steps
        To = self.n_obs_steps

        action_buffer = torch.zeros(shape, dtype=self.dtype, device=self.device)
        scheduler = self.noise_scheduler

        # Find diffusion steps
        # Calculate chunk indices i
        num_complete_chunks = math.floor((Tp - (To - 1)) / Ta)
        incomplete_index = Ta * num_complete_chunks + (To - 1)
        indices = torch.arange(0, Tp, device=self.device)
        chunk_indices = torch.div((indices - (To - 1)), Ta, rounding_mode="floor")
        chunk_indices = torch.where(indices < To, 0, chunk_indices)
        chunk_indices = torch.where(
            indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
        )
        chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

        diff_steps = (
            torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1
        ).long()
        self.buffer_diff_steps = diff_steps

        # Add noise
        noise = torch.randn(
            size=action_buffer.shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        self.action_buffer = scheduler.add_noise(action_buffer, noise, diff_steps)

    def initialize_buffer_as_constant(self, shape, cond_data, generator=None):
        N = self.num_inference_steps
        B, Tp, _ = shape
        Ta = self.n_action_steps
        To = self.n_obs_steps
        Da = self.action_dim

        first_agent_pos = cond_data[:, 0, :Da]

        constant_action = first_agent_pos.unsqueeze(1).repeat(1, Tp, 1)
        action_buffer = torch.zeros(shape, dtype=self.dtype, device=self.device)
        action_buffer[:, :, :Da] = constant_action
        scheduler = self.noise_scheduler

        # Find diffusion steps
        # Calculate chunk indices i
        num_complete_chunks = math.floor((Tp - (To - 1)) / Ta)
        incomplete_index = Ta * num_complete_chunks + (To - 1)
        indices = torch.arange(0, Tp, device=self.device)
        chunk_indices = torch.div((indices - (To - 1)), Ta, rounding_mode="floor")
        chunk_indices = torch.where(indices < To, 0, chunk_indices)
        chunk_indices = torch.where(
            indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
        )
        chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

        diff_steps = (
            torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1
        ).long()
        self.buffer_diff_steps = diff_steps

        # Add noise
        noise = torch.randn(
            size=action_buffer.shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        self.action_buffer = scheduler.add_noise(action_buffer, noise, diff_steps)

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        """
        Sample from the diffusion model conditioned on condition_data. Unlike EDM, we just do one
        condition_data: (B, T, C) The conditioning data
        Returns:
            action_pred: (B, T, Da) The predicted action including the observation steps
        """
        Tp = condition_data.shape[1]
        Ta = self.n_action_steps
        To = self.n_obs_steps
        N = self.num_inference_steps

        model = self.model
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(N)

        # We need to denoise the first T_o+T_a steps, i.e. push their sigma to 0
        while torch.max(self.buffer_diff_steps[:, (To - 1) + Ta - 1]) != -1:
            # 1. apply conditioning
            self.action_buffer[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                self.action_buffer,
                self.buffer_diff_steps,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            # 3. compute previous image: x_t -> x_t-1
            self.action_buffer = scheduler.step(
                model_output,
                self.buffer_diff_steps,
                self.action_buffer,
                generator=generator,
                **kwargs,
            ).prev_sample

            # 4. update the diffusion step for the buffer
            self.buffer_diff_steps = torch.clamp(self.buffer_diff_steps - 1, min=-1)

        # Finally, make sure conditioning is enforced
        self.action_buffer[condition_mask] = condition_data[condition_mask]

        # Return whole buffer as prediction, we slice later
        action_pred = self.action_buffer  # (B, T, Da) or (B, T, Da+Do)

        # Remove the first T_a steps from the buffer,
        # as they are not needed for the next iteration
        self.action_buffer = self.action_buffer[:, Ta:]
        self.buffer_diff_steps = self.buffer_diff_steps[:, Ta:]

        # Remove excess steps (not part of complete chunks) that are partly denoised
        if (Tp - (To - 1)) % Ta != 0:
            self.action_buffer = self.action_buffer[:, : -((Tp - (To - 1)) % Ta)]
            self.buffer_diff_steps = self.buffer_diff_steps[
                :, : -((Tp - (To - 1)) % Ta)
            ]

        # Add the actions with sigma_max to noise to the end
        B = condition_data.shape[0]
        num_new_actions = Ta + ((Tp - (To - 1)) % Ta)
        new_noise = torch.randn(
            size=(B, num_new_actions, self.action_buffer.shape[-1]),
            dtype=self.dtype,
            device=self.device,
        )
        new_sigma_indices = torch.ones(
            size=(B, num_new_actions), dtype=torch.long, device=self.device
        ) * (N - 1)
        self.push_buffer(new_noise, new_sigma_indices)

        # Noise the first T_o-1 steps with same noise as for index T_o-1
        next_chunk_diff_step = self.buffer_diff_steps[:, To - 1]  # (B, 1)
        next_chunk_diff_step = next_chunk_diff_step.unsqueeze(1).repeat(1, To - 1)
        self.buffer_diff_steps[:, : To - 1] = next_chunk_diff_step

        noise = torch.randn(
            size=self.action_buffer[:, : To - 1].shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.noise_scheduler.add_noise(
            self.action_buffer[:, : To - 1], noise, next_chunk_diff_step
        )

        return action_pred

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_diff_steps_training(self, B, Tp, To, Ta):
        """
        Args:
            B: batch size
            T: number of timesteps
            To: number of observation steps
            Ta: number of action steps
        Returns:
            diff_steps: (T,) or (B, T) The indices of the noise injection regime for each timestep
        """
        N = self.noise_scheduler.config.num_train_timesteps

        # Sample the noise injection regime for this batch from a categoriecal distribution
        # over the different noise injection regimes.
        probabilities = torch.tensor(
            [
                self.temporally_constant_weight,
                self.temporally_increasing_weight,
                self.temporally_random_weights,
                self.chunk_wise_weight,
            ]
        )

        # Sample from the categorical distribution based on the defined probabilities
        noise_regime = torch.multinomial(
            probabilities, num_samples=1, replacement=True
        ).item()

        # Select the noise injection regime based on the sampled value
        if noise_regime == 0:  # Constant scheme
            diff_steps = (
                torch.randint(0, N, (B, 1), device=self.device).long().repeat(1, Tp)
            )
        elif noise_regime == 1:  # Linearly increasing scheme
            diff_steps = (
                torch.linspace(0, N - 1, Tp, device=self.device, dtype=self.dtype)
                .long()
                .repeat(B, 1)
            )
        elif noise_regime == 2:  # Random scheme
            diff_steps = timesteps = torch.randint(
                0, N, (B, Tp), device=self.device
            ).long()
        elif noise_regime == 3:  # Chunk-wise scheme

            num_complete_chunks = math.floor((Tp - (To - 1)) / Ta)
            incomplete_index = Ta * num_complete_chunks + (To - 1)

            # Calculate chunk indices i
            indices = torch.arange(0, Tp, device=self.device)
            chunk_indices = torch.div((indices - (To - 1)), Ta, rounding_mode="floor")
            chunk_indices = torch.where(indices < To, 0, chunk_indices)
            chunk_indices = torch.where(
                indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
            )
            chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

            # 1. Sample j from [0, upper_limit]
            upper_limit = math.floor(N / num_complete_chunks)
            j = torch.randint(
                0, upper_limit, (B, 1), device=self.device
            )  # Don't include N/h, as that corresponds to a clean first chunk.
            j = j.repeat(1, Tp)  # (B,T)

            # 2. Evaluate k(i,j)=round((N-1) * ((i+1)/h)) - j
            diff_steps = (
                torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1 - j
            ).long()

        return diff_steps
