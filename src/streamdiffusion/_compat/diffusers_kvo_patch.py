"""Re-applies varshith15/diffusers@3e3b72f kvo_cache patch onto upstream diffusers.

Source fork: https://github.com/varshith15/diffusers @ 3e3b72f557e91546894340edabc845e894f00922
Target: diffusers >= 0.38.0 (upstream HuggingFace)

The patch threads an optional KV-cache through the UNet2DConditionModel forward pass so
that StreamDiffusion's TRT cached-attention pipeline (unet_unified_export.py) works with
vanilla upstream diffusers. When kvo_cache=None (the default) behaviour is identical to
upstream diffusers — no performance or correctness impact for non-cached paths.

Called automatically at ``import streamdiffusion`` via _compat/__init__.py.
"""

from __future__ import annotations

import inspect
import logging


logger = logging.getLogger(__name__)

_PATCHED = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply() -> None:
    """Apply kvo_cache patch. Idempotent — safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return
    if _is_patched():
        _PATCHED = True
        return

    _patch_attn_processor()
    _patch_attention_forward()
    _patch_basic_transformer_block()
    _patch_transformer2d()
    _patch_mid_block()
    _patch_down_block()
    _patch_up_block()
    _patch_unet2d()

    _PATCHED = True
    logger.debug("diffusers_kvo_patch: kvo_cache patch applied")


def _is_patched() -> bool:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

    return "kvo_cache" in inspect.signature(UNet2DConditionModel.forward).parameters


# ---------------------------------------------------------------------------
# Individual patches
# ---------------------------------------------------------------------------


def _patch_attn_processor() -> None:
    """AttnProcessor2_0.__call__: accept kvo_cache kwarg, return (hidden_states, kvo_cache)."""
    from diffusers.models.attention_processor import AttnProcessor2_0

    _orig = AttnProcessor2_0.__call__

    def _call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        kvo_cache=None,
        *args,
        **kwargs,
    ):
        result = _orig(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )
        return result, kvo_cache

    AttnProcessor2_0.__call__ = _call


def _patch_attention_forward() -> None:
    """Attention.forward: accept kvo_cache, route to processor only for self-attn."""
    from diffusers.models.attention_processor import Attention
    from diffusers.utils import logging as _dlog

    _dlogger = _dlog.get_logger("diffusers.models.attention_processor")

    def _forward(
        self, hidden_states, encoder_hidden_states=None, attention_mask=None, kvo_cache=None, **cross_attention_kwargs
    ):
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k in cross_attention_kwargs if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if unused_kwargs:
            _dlogger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by "
                f"{self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        if encoder_hidden_states is None:
            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                kvo_cache=kvo_cache,
                **cross_attention_kwargs,
            )
        else:
            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

    Attention.forward = _forward


def _patch_basic_transformer_block() -> None:
    """BasicTransformerBlock.forward: thread kvo_cache through attn1, handle attn2 tuple return."""
    from diffusers.models.attention import BasicTransformerBlock

    def _forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        added_cond_kwargs=None,
        kvo_cache=None,
    ):
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output, kvo_cache_out = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            kvo_cache=kvo_cache,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            hidden_states = attn_output + hidden_states

        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            from diffusers.models.attention import _chunked_feed_forward

            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states, kvo_cache_out

    BasicTransformerBlock.forward = _forward


def _patch_transformer2d() -> None:
    """Transformer2DModel.forward: thread kvo_cache through transformer_blocks."""
    import torch
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.transformers.transformer_2d import Transformer2DModel

    def _forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        added_cond_kwargs=None,
        class_labels=None,
        cross_attention_kwargs=None,
        attention_mask=None,
        encoder_attention_mask=None,
        kvo_cache=None,
        return_dict=True,
    ):
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.is_input_continuous:
            batch_size, _, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
                hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
            )

        kvo_cache_out = []
        for idx, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                )
            else:
                block_cache_in = kvo_cache[idx] if kvo_cache else None
                hidden_states, block_cache_out = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    kvo_cache=block_cache_in,
                )
                if block_cache_out is not None:
                    kvo_cache_out.append(block_cache_out)

        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(
                hidden_states=hidden_states,
                residual=residual,
                batch_size=batch_size,
                height=height,
                width=width,
                inner_dim=inner_dim,
            )
        elif self.is_input_vectorized:
            output = self._get_output_for_vectorized_inputs(hidden_states)
        elif self.is_input_patches:
            output = self._get_output_for_patched_inputs(
                hidden_states=hidden_states,
                timestep=timestep,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep,
                height=height,
                width=width,
            )

        if not return_dict:
            return (output, kvo_cache_out)

        return Transformer2DModelOutput(sample=output)

    Transformer2DModel.forward = _forward


def _patch_mid_block() -> None:
    """UNetMidBlock2DCrossAttn.forward: thread kvo_cache through attention loop."""
    import torch
    from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn

    def _forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        encoder_attention_mask=None,
        kvo_cache=None,
    ):
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        hidden_states = self.resnets[0](hidden_states, temb)
        kvo_cache_out = []
        for idx, (attn, resnet) in enumerate(zip(self.attentions, self.resnets[1:])):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                block_cache_in = kvo_cache[idx] if kvo_cache else None
                hidden_states, block_cache_out = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    kvo_cache=block_cache_in,
                    return_dict=False,
                )
                hidden_states = resnet(hidden_states, temb)
                if block_cache_out is not None:
                    kvo_cache_out.append(block_cache_out)

        return hidden_states, kvo_cache_out

    UNetMidBlock2DCrossAttn.forward = _forward


def _patch_down_block() -> None:
    """CrossAttnDownBlock2D.forward: thread kvo_cache, return (hidden, output_states, kvo_cache_out)."""
    import torch
    from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D

    def _forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        encoder_attention_mask=None,
        additional_residuals=None,
        kvo_cache=None,
    ):
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        output_states = ()
        blocks = list(zip(self.resnets, self.attentions))
        kvo_cache_out = []

        for i, (resnet, attn) in enumerate(blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                block_cache_in = kvo_cache[i] if kvo_cache else None
                hidden_states, block_cache_out = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    kvo_cache=block_cache_in,
                    return_dict=False,
                )
                if block_cache_out is not None:
                    kvo_cache_out.append(block_cache_out)

            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states, kvo_cache_out

    CrossAttnDownBlock2D.forward = _forward


def _patch_up_block() -> None:
    """CrossAttnUpBlock2D.forward: thread kvo_cache, return (hidden, kvo_cache_out)."""
    from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D

    try:
        from diffusers.models.unets.unet_2d_blocks import apply_freeu
    except ImportError:
        apply_freeu = None
    import torch

    def _forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
        upsample_size=None,
        attention_mask=None,
        encoder_attention_mask=None,
        kvo_cache=None,
    ):
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        is_freeu_enabled = (
            apply_freeu is not None
            and getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        kvo_cache_out = []
        for idx, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                block_cache_in = kvo_cache[idx] if kvo_cache else None
                hidden_states, block_cache_out = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    kvo_cache=block_cache_in,
                    return_dict=False,
                )
                if block_cache_out is not None:
                    kvo_cache_out.append(block_cache_out)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states, kvo_cache_out

    CrossAttnUpBlock2D.forward = _forward


def _patch_unet2d() -> None:
    """UNet2DConditionModel.forward: add kvo_cache param, wire through all blocks."""
    import torch
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
    from diffusers.utils import deprecate

    def _forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels=None,
        timestep_cond=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        down_intrablock_additional_residuals=None,
        encoder_attention_mask=None,
        kvo_cache=None,
        return_dict=True,
    ):
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        sample = self.conv_in(sample)

        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = down_intrablock_additional_residuals is not None
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated "
                "and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used "
                "for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        cache_idx = 0
        kvo_cache_out = []

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                block_cache_in = kvo_cache[cache_idx] if kvo_cache else None
                sample, res_samples, block_cache_out = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    kvo_cache=block_cache_in,
                    **additional_residuals,
                )
                cache_idx += 1
                if block_cache_out is not None:
                    kvo_cache_out.append(block_cache_out)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                block_cache_in = kvo_cache[cache_idx] if kvo_cache else None
                sample, block_cache_out = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    kvo_cache=block_cache_in,
                )
                if block_cache_out is not None:
                    kvo_cache_out.append(block_cache_out)
                cache_idx += 1
            else:
                sample = self.mid_block(sample, emb)

            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                block_cache_in = kvo_cache[cache_idx] if kvo_cache else None
                sample, block_cache_out = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    kvo_cache=block_cache_in,
                )
                cache_idx += 1
                if block_cache_out is not None:
                    kvo_cache_out.append(block_cache_out)
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample, kvo_cache_out)

        return UNet2DConditionOutput(sample=sample)

    UNet2DConditionModel.forward = _forward
