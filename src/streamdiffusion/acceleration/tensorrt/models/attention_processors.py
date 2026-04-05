from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND


class CachedSTAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # Per-layer pre-allocated buffers (lazy-init on first call — shape is model-dependent)
        self._curr_key_buf: Optional[torch.Tensor] = None
        self._curr_value_buf: Optional[torch.Tensor] = None
        self._kv_out_buf: Optional[torch.Tensor] = None  # shape: (2, 1, B, seq, inner_dim)
        # When False (default): ONNX-safe .clone() path — used during torch.onnx.export() tracing.
        # When True: zero-alloc .copy_() path — set after ONNX export for non-TRT runtime inference.
        # NOTE: aten::copy has no ONNX symbolic and cannot be traced; never set True before export.
        self._use_prealloc: bool = False

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        kvo_cache: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
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
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
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
            if self._use_prealloc:
                # Zero-alloc path: .copy_() into pre-allocated buffers eliminates 2 mallocs per layer.
                # NOT ONNX-traceable — only active after export (aten::copy has no ONNX symbolic).
                if self._curr_key_buf is None or self._curr_key_buf.shape != key.shape:
                    self._curr_key_buf = torch.empty_like(key)
                    self._curr_value_buf = torch.empty_like(value)
                    self._kv_out_buf = torch.empty((2, 1, *key.shape), dtype=key.dtype, device=key.device)
                self._curr_key_buf.copy_(key)
                self._curr_value_buf.copy_(value)
                curr_key = self._curr_key_buf
                curr_value = self._curr_value_buf
            else:
                # ONNX-safe path: .clone() exports cleanly to aten::clone (has ONNX symbolic).
                curr_key = key.clone()
                curr_value = value.clone()

            if cached_key is not None:
                cached_key_reshaped = cached_key.transpose(0, 1).contiguous().flatten(1, 2)
                cached_value_reshaped = cached_value.transpose(0, 1).contiguous().flatten(1, 2)
                key = torch.cat([curr_key, cached_key_reshaped], dim=1)
                value = torch.cat([curr_value, cached_value_reshaped], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
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

        if is_selfattn:
            if self._use_prealloc:
                # In-place fill pre-allocated output buffer: eliminates torch.stack malloc per layer.
                self._kv_out_buf[0, 0].copy_(curr_key)
                self._kv_out_buf[1, 0].copy_(curr_value)
                kvo_cache = self._kv_out_buf
            else:
                # ONNX-safe fallback: torch.stack is exportable.
                kvo_cache = torch.stack([curr_key.unsqueeze(0), curr_value.unsqueeze(0)])

        return hidden_states, kvo_cache
