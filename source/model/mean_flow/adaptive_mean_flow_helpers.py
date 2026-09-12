import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Pair-time embedding (time_t, time_r) per token
# ----------------------------

class FourierPairTimeEmbedding(nn.Module):
    """
    Per-token (time_t, time_r) -> embedding vector.
    Uses fixed Fourier features on [t_s, t_e, dt] then an MLP.
    """
    def __init__(self, emb_dim: int, n_frequencies: int = 32, mlp_hidden: int = 128, add_raw_dt: bool = True):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.emb_dim = emb_dim
        self.add_raw_dt = add_raw_dt

        freqs = 2.0 ** torch.arange(n_frequencies, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

        # 3 scalars -> each gets (sin, cos) * K => 3 * 2K
        base_in_dim = 3 * 2 * n_frequencies
        in_dim = base_in_dim + (1 if add_raw_dt else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, emb_dim),
        )

        # Start near "no conditioning"
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, time_t: torch.Tensor, time_r: torch.Tensor) -> torch.Tensor:
        """
        time_t, time_r: (B, T) (or (B,) broadcastable)
        returns: (B, T, emb_dim)
        """
        if time_t.dim() == 1:
            time_t = time_t[:, None]
        if time_r.dim() == 1:
            time_r = time_r[:, None]
        if time_t.dim() != 2 or time_r.dim() != 2:
            raise ValueError(f"Expected (B,T) tensors; got {tuple(time_t.shape)}, {tuple(time_r.shape)}")

        dt = time_r - time_t

        def fourier_feats(x: torch.Tensor) -> torch.Tensor:
            angles = math.pi * x.unsqueeze(-1) * self.freqs.view(1, 1, -1)  # (B,T,K)
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,T,2K)

        fs = fourier_feats(time_t)
        fe = fourier_feats(time_r)
        fdt = fourier_feats(dt)

        feats = torch.cat([fs, fe, fdt], dim=-1)  # (B,T, 3*2K)
        if self.add_raw_dt:
            feats = torch.cat([feats, dt.unsqueeze(-1)], dim=-1)  # (B,T, 3*2K + 1)

        return self.mlp(feats)  # (B,T,emb_dim)


class TimeFiLM(nn.Module):
    """Per-token FiLM parameters (gamma, beta) in R^d_model each."""
    def __init__(self, time_emb_dim: int, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.net(time_emb)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


class TimeGate(nn.Module):
    """Per-token residual gate in (0,1) (either scalar or vector)."""
    def __init__(self, time_emb_dim: int, d_model: int, hidden: int = 128, scalar_gate: bool = False, init_bias: float = 4.0):
        super().__init__()
        out_dim = 1 if scalar_gate else d_model
        self.scalar_gate = scalar_gate
        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)  # sigmoid(init_bias) ~ 1

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(time_emb))  # (B,T,1 or d)


# ----------------------------
# Time-conditioned Transformer Decoder
# ----------------------------

@dataclass
class TimeCondConfig:
    mode: Literal["film", "gated_residual"] = "gated_residual"
    film_on: Literal["self_attn", "cross_attn", "ffn", "all"] = "self_attn"
    scalar_gate: bool = False
    time_emb_dim: int = 128
    time_fourier_freqs: int = 32
    time_mlp_hidden: int = 128
    add_raw_dt: bool = True


class TimeConditionedTransformerDecoderLayer(nn.Module):
    """
    Pre-norm (norm_first=True) TransformerDecoderLayer variant with per-token
    (time_t, time_r) conditioning for the *tgt tokens*.

    - batch_first=True
    - you pass tgt_mask (causal/custom) externally, same as your current setup
    - supports memory/cross-attn via `memory` and memory masks/padding masks
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        cfg: Optional[TimeCondConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.cfg = cfg or TimeCondConfig()

        # Attention blocks
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Dropouts
        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # Norms (pre-norm)
        self.norm_self = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_cross = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Time embedding for tgt tokens
        self.time_embed = FourierPairTimeEmbedding(
            emb_dim=self.cfg.time_emb_dim,
            n_frequencies=self.cfg.time_fourier_freqs,
            mlp_hidden=self.cfg.time_mlp_hidden,
            add_raw_dt=self.cfg.add_raw_dt,
        )

        if self.cfg.mode == "film":
            self.film = TimeFiLM(self.cfg.time_emb_dim, d_model, hidden=self.cfg.time_mlp_hidden)
        elif self.cfg.mode == "gated_residual":
            self.gate_self = TimeGate(self.cfg.time_emb_dim, d_model, hidden=self.cfg.time_mlp_hidden, scalar_gate=self.cfg.scalar_gate)
            self.gate_cross = TimeGate(self.cfg.time_emb_dim, d_model, hidden=self.cfg.time_mlp_hidden, scalar_gate=self.cfg.scalar_gate)
            self.gate_ffn = TimeGate(self.cfg.time_emb_dim, d_model, hidden=self.cfg.time_mlp_hidden, scalar_gate=self.cfg.scalar_gate)
        else:
            raise ValueError(f"Unknown cfg.mode: {self.cfg.mode}")

        # Activation
        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        elif activation == "silu":
            self.act = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    @staticmethod
    def _film(x_norm: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # AdaLN-style: LN(x) * (1 + gamma) + beta
        return x_norm * (1.0 + gamma) + beta

    def forward(
        self,
        tgt: torch.Tensor,                          # (B, T_tgt, D)
        memory: torch.Tensor,                       # (B, T_mem, D)
        time_t: torch.Tensor,                      # (B, T_tgt) or (B,)
        time_r: torch.Tensor,                        # (B, T_tgt) or (B,)
        tgt_mask: Optional[torch.Tensor] = None,     # passed through to self-attn
        memory_mask: Optional[torch.Tensor] = None,  # passed through to cross-attn
        tgt_key_padding_mask: Optional[torch.Tensor] = None,     # (B, T_tgt) bool
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T_mem) bool
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        B, Tt, D = tgt.shape
        assert D == self.d_model
        assert memory.shape[0] == B and memory.shape[2] == D

        # Broadcast times if you pass (B,)
        if time_t.dim() == 1:
            time_t = time_t[:, None].expand(B, Tt)
        if time_r.dim() == 1:
            time_r = time_r[:, None].expand(B, Tt)
        assert time_t.shape == (B, Tt) and time_r.shape == (B, Tt)

        time_emb = self.time_embed(time_t, time_r)  # (B, Tt, time_emb_dim)

        x = tgt

        # ----- Self-attention (pre-norm) -----
        x_norm = self.norm_self(x)
        if self.cfg.mode == "film" and self.cfg.film_on in ("self_attn", "all"):
            gamma, beta = self.film(time_emb)
            x_qkv = self._film(x_norm, gamma, beta)
        else:
            x_qkv = x_norm

        self_out, _ = self.self_attn(
            x_qkv, x_qkv, x_qkv,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
            is_causal=tgt_is_causal,
        )
        self_out = self.dropout_self(self_out)

        if self.cfg.mode == "gated_residual":
            g = self.gate_self(time_emb)  # (B,Tt,1 or D)
            self_out = self_out * g

        x = x + self_out

        # ----- Cross-attention (pre-norm) -----
        x_norm = self.norm_cross(x)
        if self.cfg.mode == "film" and self.cfg.film_on in ("cross_attn", "all"):
            gamma, beta = self.film(time_emb)
            x_q = self._film(x_norm, gamma, beta)
        else:
            x_q = x_norm

        cross_out, _ = self.cross_attn(
            x_q, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
            is_causal=memory_is_causal,
        )
        cross_out = self.dropout_cross(cross_out)

        if self.cfg.mode == "gated_residual":
            g = self.gate_cross(time_emb)
            cross_out = cross_out * g

        x = x + cross_out

        # ----- FFN (pre-norm) -----
        x_norm = self.norm_ffn(x)
        if self.cfg.mode == "film" and self.cfg.film_on in ("ffn", "all"):
            gamma, beta = self.film(time_emb)
            x_ff = self._film(x_norm, gamma, beta)
        else:
            x_ff = x_norm

        ff = self.linear2(self.dropout_ffn(self.act(self.linear1(x_ff))))
        ff = self.dropout_ffn(ff)

        if self.cfg.mode == "gated_residual":
            g = self.gate_ffn(time_emb)
            ff = ff * g

        x = x + ff
        return x


class TimeConditionedTransformerDecoder(nn.Module):
    """
    Drop-in-ish replacement for nn.TransformerDecoder, but:
      - requires per-token (time_t, time_r) for tgt tokens
      - mask handling stays identical (you pass in computed masks)
      - batch_first + norm_first semantics baked in
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        cfg: Optional[TimeCondConfig] = None,
        final_norm: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TimeConditionedTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                cfg=cfg,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if final_norm else None

    def forward(
        self,
        tgt: torch.Tensor,                          # (B, T_tgt, D)
        memory: torch.Tensor,                       # (B, T_mem, D)
        time_t: torch.Tensor,                      # (B, T_tgt) or (B,)
        time_r: torch.Tensor,                        # (B, T_tgt) or (B,)
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(
                x, memory,
                time_t=time_t,
                time_r=time_r,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
        #assert False, f"{x}"
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x


## DIT https://apxml.com/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/practice-building-dit-block

class AdaLNModulation(nn.Module):
    """
    Calculates shift, scale, and gate parameters from conditioning embeddings.
    Initializes the final linear layer's weights and biases to zero.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        # Initialize the linear layer with zeros for the "Zero" aspect
        self.linear = nn.Linear(embedding_dim, 6 * hidden_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c):
        # c shape: [batch_size, embedding_dim]
        mod_params = self.linear(self.silu(c))
        # mod_params shape: [batch_size, 6 * hidden_dim]
        # Split into 6 parts for shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        # Each part has shape [batch_size, hidden_dim]
        # Add unsqueeze(1) to make them broadcastable with token dim: [batch_size, 1, hidden_dim]
        return mod_params.chunk(6, dim=-1)

class DiTBlock(nn.Module):
    """
    A standard DiT block with adaLN-Zero modulation.
    """
    def __init__(self, hidden_dim: int, embedding_dim: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        # Normalization layers (using LayerNorm without elementwise affine,
        # as adaLN provides the affine parameters)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Modulation network to generate adaLN parameters
        self.modulation = AdaLNModulation(embedding_dim, hidden_dim)

        # Attention layer
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)

        # MLP layer
        mlp_hidden_dim = dim_feedforward
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(), # Or nn.SiLU()
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, x,c_att, c_inj, attn_mask = None, memory_mask = None):
        # x shape: [batch_size, num_patches, hidden_dim]
        # c shape: [batch_size, embedding_dim]

        # Calculate modulation parameters (shift, scale, gate for MSA and MLP)
        # Each param shape: [batch_size, 1, hidden_dim] after unsqueeze

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            p for p in self.modulation(c_inj)
        ]

        # --- Apply adaLN-Zero before MHSA ---
        x_norm1 = self.norm1(x)
        # Modulate: Scale, Shift
        x_modulated1 = x_norm1 * (1 + scale_msa) + shift_msa
        # Apply attention
        attn_output, _ = self.self_attn(x_modulated1, x_modulated1, x_modulated1, attn_mask=attn_mask)
        attn_output = self.dropout1(attn_output)
        # Apply gating and add residual connection
        x = x + gate_msa * attn_output

        # --- Apply adaLN-Zero before MLP ---
        x_norm2 = self.norm2(x)

        # Apply cross attention
        cross_attn_output,_ = self.cross_attn(x_norm2, c_att, c_att, attn_mask=memory_mask)
        x = x + self.dropout2(cross_attn_output)
        x_norm3 = self.norm3(x)

        # Modulate: Scale, Shift
        x_modulated2 = x_norm3 * (1 + scale_mlp) + shift_mlp
        # Apply MLP
        mlp_output = self.dropout3(self.mlp(x_modulated2))
        # Apply gating and add residual connection
        x = x + gate_mlp * mlp_output

        # Output shape: [batch_size, num_patches, hidden_dim]
        return x


class DiT(nn.Module):
    """
    Drop-in-ish replacement for nn.TransformerDecoder, but:
      - requires per-token (time_t, time_r) for tgt tokens
      - mask handling stays identical (you pass in computed masks)
      - batch_first + norm_first semantics baked in
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dim_c_inj: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DiTBlock(
                hidden_dim=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                embedding_dim = dim_c_inj,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.time_emb = nn.Linear(2,dim_c_inj)

    def forward(
        self,
        tgt: torch.Tensor,                          # (B, T_tgt, D)
        memory: torch.Tensor,                       # (B, T_mem, D)
        time_t: torch.Tensor,                      # (B, T_tgt) or (B,)
        time_r: torch.Tensor,                        # (B, T_tgt) or (B,)
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = tgt

        time = torch.cat((time_t, time_r), dim=-1)
        time_emb = self.time_emb(time)

        for layer in self.layers:
            x = layer(
                x, memory, time_emb,
                attn_mask=tgt_mask,
                memory_mask=memory_mask
            )

        return x

# ----------------------------
# Example usage
# ----------------------------
