from typing import Optional

import torch
import torch.nn as nn

from source.model.mean_flow.transformer_for_adaptive_mean_flow import TransformerForAdaptiveMeanFlow


class TransformerForStreamingDiffusion(TransformerForAdaptiveMeanFlow):
    """
    Adaptive Mean Flow-style transformer backbone adapted to Streaming Diffusion's single per-token diffusion
    timestep. This keeps the observation conditioning and transformer hyperparams
    from TransformerForAdaptiveMeanFlow, but changes time conditioning from (t, r) to t.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            time_cond: str = "cat",
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool = False,
            n_cond_layers: int = 0
        ) -> None:
        super().__init__(
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
            n_cond_layers=n_cond_layers,
        )

        if time_cond == "cat":
            self.input_emb = nn.Linear(input_dim + 1, n_emb)
            self._init_weights(self.input_emb)
        elif time_cond == "add":
            self.t_emb = nn.Linear(1, n_emb)
            if hasattr(self, "r_emb"):
                del self.r_emb
            self._init_weights(self.t_emb)
        elif time_cond == "AdaLNZero":
            # The DiT helper takes two scalars. Streaming Diffusion only has one diffusion
            # time, so forward passes a zero-valued second channel.
            pass
        else:
            raise AssertionError(f"Unsupported time conditioning method {time_cond}")

    def forward(
            self,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            cond: Optional[torch.Tensor] = None,
            **kwargs):
        """
        sample: (B, T, input_dim)
        timestep: (B, T), (B,), scalar tensor, or int diffusion step
        cond: (B, To, cond_dim)
        """
        if cond is None:
            cond = kwargs.get("global_cond", None)
        if cond is None:
            raise ValueError("TransformerForStreamingDiffusion requires observation conditioning via cond or global_cond.")

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=sample.device)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)

        B, T = sample.shape[:2]
        if timestep.ndim == 0:
            timestep = timestep.reshape(1, 1).expand(B, T)
        elif timestep.ndim == 1:
            if timestep.shape[0] == B:
                timestep = timestep[:, None].expand(B, T)
            elif timestep.shape[0] == T:
                timestep = timestep[None].expand(B, T)
            else:
                raise ValueError(
                    f"Expected timestep shape ({B},) or ({T},), got {tuple(timestep.shape)}")
        elif timestep.ndim == 2:
            if timestep.shape != (B, T):
                raise ValueError(
                    f"Expected timestep shape ({B}, {T}), got {tuple(timestep.shape)}")
        else:
            raise ValueError(f"Expected timestep with 0, 1, or 2 dims, got {tuple(timestep.shape)}")

        time = timestep.unsqueeze(-1)

        if self.time_cond == "cat":
            sample = torch.concatenate((sample, time), dim=-1)
            input_emb = self.input_emb(sample)
        elif self.time_cond == "add":
            input_emb = self.input_emb(sample) + self.t_emb(time)
        elif self.time_cond == "AdaLNZero":
            input_emb = self.input_emb(sample)
        else:
            raise AssertionError(f"Unsupported time conditioning method {self.time_cond}")

        cond_obs_emb = self.cond_obs_emb(cond)
        tc = cond_obs_emb.shape[1]
        cond_pos = self.cond_pos_emb[:, :tc, :]
        memory = self.encoder(self.drop(cond_obs_emb + cond_pos))

        t = input_emb.shape[1]
        pos = self.pos_emb[:, :t, :]
        x = self.drop(input_emb + pos)
        if self.time_cond == "AdaLNZero":
            zero_time = torch.zeros_like(time)
            x = self.decoder(
                tgt=x,
                memory=memory,
                time_t=time,
                time_r=zero_time,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
            )
        else:
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
            )

        x = self.ln_f(x)
        x = self.head(x)
        return x
