from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND


def get_nn_feats(
    x: torch.Tensor,
    y: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Nearest-neighbour feature matching from the StreamV2V thesis (§3.4.2, Eq 3.2).

    For each token in ``x`` find the most cosine-similar token in ``y``.  If the best
    match exceeds ``threshold`` the cached token replaces the current one; otherwise
    the current token is kept as-is (novel regions are not injected).

    Args:
        x: Current frame features  ``[B, N, C]``.
        y: Cached feature bank     ``[B, M, C]`` — M = cache_maxframes × N.
        threshold: Cosine-similarity gate (0–1).  Higher = more conservative injection.

    Returns:
        Tensor ``[B, N, C]`` — the feature-fused output.
    """
    x_norm = F.normalize(x, dim=-1)  # [B, N, C]
    y_norm = F.normalize(y, dim=-1)  # [B, M, C]
    cos = torch.bmm(x_norm, y_norm.transpose(1, 2))  # [B, N, M]
    max_cos, idx = cos.max(dim=-1)  # both [B, N]
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, y.size(-1))  # [B, N, C]
    nn_feats = torch.gather(y, 1, idx_exp)
    gate = (max_cos >= threshold).unsqueeze(-1)  # [B, N, 1]
    return torch.where(gate, nn_feats, x)


class CachedSTAttnProcessor2_0:
    r"""Self-attention processor with K/V caching (Extended Attention, EA) and optional
    Feature Injection (FI).

    EA — concatenates stored K/V from the feature bank into each self-attention
    forward pass (thesis §3.4.1, Eq 3.1).  This is the existing functionality.

    FI — after computing the self-attention output, blends it with the nearest-neighbour-
    matched cached output (thesis §3.4.2, Eq 3.2):
        h' = (1 - fi_strength) * h + fi_strength * get_nn_feats(h, O_fb, thr)
    Applied only when ``fi_eligible=True``.

    FI inputs are provided via instance attributes set by the export wrapper before each
    forward pass (same pattern as ``set_ipadapter_scale`` in unet_ipadapter_export.py):

        proc._fi_cache      : Tensor(maxframes, B, N, C) or None
        proc._fi_strength   : Tensor([1], fp32) or None
        proc._fi_threshold  : Tensor([1], fp32) or None

    FI output (raw pre-injection block output) is stored in ``proc._fi_cache_out`` after
    each self-attn forward.  The ONNX export wrapper includes it in the traced function's
    return value so TRT sees it as an engine output.  During TRT inference, unet_engine.py
    reads it from the engine output bindings directly.

    The return signature is unchanged: ``(hidden_states, kvo_cache)`` — the kvo_patch
    and all downstream diffusers forward functions require no modification.
    """

    def __init__(self, fi_eligible: bool = False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.fi_eligible = fi_eligible

        # FI input attributes — set by the export wrapper before each forward pass.
        # None means FI is disabled / cache not yet warm.
        self._fi_cache: Optional[torch.Tensor] = None  # (maxframes, B, N, C)
        self._fi_strength: Optional[torch.Tensor] = None  # [1] fp32
        self._fi_threshold: Optional[torch.Tensor] = None  # [1] fp32

        # FI output attribute — written after each self-attn forward.
        # Included in the ONNX export wrapper's return so it becomes a graph output.
        self._fi_cache_out: Optional[torch.Tensor] = None  # (1, B, N, C)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        kvo_cache: Optional[torch.FloatTensor] = None,
    ) -> tuple:
        """Forward — returns ``(hidden_states, kvo_cache)``; same as the stock processor."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        is_selfattn = False
        if encoder_hidden_states is None:
            is_selfattn = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if kvo_cache is not None:
            cached_key = kvo_cache[0]
            cached_value = kvo_cache[1]
        else:
            cached_key, cached_value = None, None

        if is_selfattn:
            # Save current K/V before extending for K/V cache output
            curr_key = key.clone()
            curr_value = value.clone()

            if cached_key is not None:
                # EA: extend self-attention to include banked keys and values
                # cached_key shape: (maxframes, batch, seq, hidden)
                # reshape to: (batch, maxframes*seq, hidden) for attention concat
                cached_key_reshaped = cached_key.transpose(0, 1).contiguous().flatten(1, 2)
                cached_value_reshaped = cached_value.transpose(0, 1).contiguous().flatten(1, 2)
                key = torch.cat([curr_key, cached_key_reshaped], dim=1)
                value = torch.cat([curr_value, cached_value_reshaped], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # --- K/V cache output ---
        kvo_cache_out = None
        if is_selfattn:
            kvo_cache_out = torch.stack([curr_key.unsqueeze(0), curr_value.unsqueeze(0)], dim=0)

        # --- Feature Injection (FF) ---
        # Operates on 3D [B, N, C] after to_out + residual + rescale (thesis §3.4.2).
        # attn1 hidden_states are always 3D here — the input_ndim==4 branch rearranges
        # to 2D sequences and early-returns the same shape, not 4D.
        #
        # The raw (pre-injection) output is stored in self._fi_cache_out so the ONNX
        # export wrapper can include it in the traced function return value.  The
        # export wrapper also sets self._fi_cache / _fi_strength / _fi_threshold before
        # each forward pass (same pattern as set_ipadapter_scale).
        if is_selfattn and self.fi_eligible:
            # Capture raw output before any injection — shape [1, B, N, C] to match
            # the fio_cache_out binding shape convention.
            self._fi_cache_out = hidden_states.unsqueeze(0)

            fi_cache = self._fi_cache
            fi_strength = self._fi_strength
            fi_threshold = self._fi_threshold

            if fi_cache is not None and fi_strength is not None and fi_threshold is not None:
                # Keep fi_strength / fi_threshold as tensors (no .item()) so ONNX tracing
                # records them as live graph inputs rather than baking them as Python constants.
                # strength=0 yields the algebraic identity (1-0)*h + 0*blended = h — no branch
                # needed; the FI-off fast path is a no-op blend, not a skip.
                strength = fi_strength.to(hidden_states.dtype)
                threshold = fi_threshold.to(hidden_states.dtype)
                # Reshape bank: (maxframes, B, N, C) → (B, maxframes*N, C)
                bank = fi_cache.transpose(0, 1).reshape(batch_size, -1, hidden_states.shape[-1])
                blended = get_nn_feats(hidden_states, bank, threshold)
                hidden_states = (1.0 - strength) * hidden_states + strength * blended
        else:
            self._fi_cache_out = None

        return hidden_states, kvo_cache_out
