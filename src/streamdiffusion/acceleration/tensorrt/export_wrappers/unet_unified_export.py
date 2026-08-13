from typing import Dict, List, Optional, Tuple

import torch
from diffusers import UNet2DConditionModel
from peft.tuners.tuners_utils import BaseTunerLayer

from streamdiffusion._patches.diffusers_kvo_patch import apply as _apply_kvo_patch

_apply_kvo_patch()  # ensure kvo_cache patch is present even if diffusers was imported first

from ..models.utils import convert_list_to_structure
from .unet_controlnet_export import create_controlnet_wrapper
from .unet_ipadapter_export import create_ipadapter_wrapper


def _collect_fi_processors(unet: UNet2DConditionModel) -> List:
    """Walk the UNet in kvo-cache walk order (down→mid→up) and return all
    ``CachedSTAttnProcessor2_0`` instances that have ``fi_eligible=True``.

    Walk order MUST match ``get_kvo_cache_info`` in models/utils.py so that the
    returned list is index-aligned with ``fi_layer_indices`` from
    ``get_fi_eligible_mask``.  Down-blocks always contribute nothing (they are
    never FI-eligible), but we still walk them to keep the global layer counter
    consistent with the kvo walk.
    """
    from ..models.attention_processors import CachedSTAttnProcessor2_0

    procs: List = []

    for block in unet.down_blocks:
        if hasattr(block, "attentions") and block.attentions is not None:
            for attn_block in block.attentions:
                for transformer in attn_block.transformer_blocks:
                    p = transformer.attn1.processor
                    if isinstance(p, CachedSTAttnProcessor2_0) and p.fi_eligible:
                        procs.append(p)

    if hasattr(unet.mid_block, "attentions") and unet.mid_block.attentions is not None:
        for attn_block in unet.mid_block.attentions:
            for transformer in attn_block.transformer_blocks:
                p = transformer.attn1.processor
                if isinstance(p, CachedSTAttnProcessor2_0) and p.fi_eligible:
                    procs.append(p)

    for block in unet.up_blocks:
        if hasattr(block, "attentions") and block.attentions is not None:
            for attn_block in block.attentions:
                for transformer in attn_block.transformer_blocks:
                    p = transformer.attn1.processor
                    if isinstance(p, CachedSTAttnProcessor2_0) and p.fi_eligible:
                        procs.append(p)

    return procs


def _collect_lora_layers(
    unet: torch.nn.Module, lora_order: List[Tuple[str, str]]
) -> Dict[str, List[Tuple[BaseTunerLayer, float]]]:
    """Walk ``unet.named_modules()`` and, for every adapter name in ``lora_order``,
    capture every ``BaseTunerLayer`` carrying that adapter along with its base PEFT
    scaling (``lora_alpha / r``, established once by ``pipe.set_adapters(..., 1.0)``
    at load time — see wrapper.py's live-adapter-activation block). ``_set_lora_scale``
    multiplies this base value by the live ``lora_scale`` tensor every forward pass.

    Fails loudly (``RuntimeError``) rather than shipping an inert slider: every touched
    layer must be vanilla LoRA (no DoRA/other ``lora_variant``), unmerged, and not
    adapter-disabled — the three states under which PEFT's ``scaling`` dict multiply is
    bypassed entirely (``peft/tuners/lora/layer.py``'s forward()/merge()/unmerge()).
    """
    adapter_names = {adapter_name for _path, adapter_name in lora_order}
    layers: Dict[str, List[Tuple[BaseTunerLayer, float]]] = {name: [] for name in adapter_names}

    for module_path, module in unet.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        for adapter_name in adapter_names:
            if adapter_name not in module.scaling:
                continue
            if adapter_name in module.lora_variant:
                raise RuntimeError(
                    f"LoRA adapter '{adapter_name}' uses a non-vanilla PEFT variant "
                    f"(e.g. DoRA) on module '{module_path}'. The runtime lora_scale "
                    f"tensor only patches the vanilla scaling path "
                    f"('result + lora_B(lora_A(x)) * scaling' in peft's layer.py) — a "
                    f"variant adapter would silently ignore it and ship an inert slider. "
                    f"Refusing to export rather than do that."
                )
            if module.merged:
                raise RuntimeError(
                    f"LoRA adapter '{adapter_name}' is merged into module '{module_path}'. "
                    f"Dynamic lora_scale requires every adapter to stay live/unfused."
                )
            if module.disable_adapters:
                raise RuntimeError(
                    f"LoRA adapter '{adapter_name}' is disabled on module '{module_path}'. "
                    f"Dynamic lora_scale requires every adapter to stay active."
                )
            layers[adapter_name].append((module, module.scaling[adapter_name]))

    for adapter_name in adapter_names:
        if not layers[adapter_name]:
            raise RuntimeError(
                f"LoRA adapter '{adapter_name}' was requested via lora_order but no PEFT "
                f"BaseTunerLayer in the UNet carries it — nothing to scale."
            )

    return layers


class UnifiedExportWrapper(torch.nn.Module):
    """
    Unified wrapper that composes wrappers for conditioning modules.

    Positional args in ``forward`` (after the three base inputs) follow this order:

        ipadapter_scale  (optional, only when use_ipadapter=True; stripped before routing)
        lora_scale       (optional, [N] fp32, only when num_loras>0; stripped before routing)
        kvo_cache_in_0 … kvo_cache_in_N     (kvo_cache_count tensors)
        fio_cache_in_<idx0> … fio_cache_in_<idxM>   (fi_layer_count tensors, FI only)
        fi_strength                          (scalar [1] fp32, FI only)
        fi_threshold                         (scalar [1] fp32, FI only)

    Outputs (ONNX flat order after PyTorch flattening):
        latent
        kvo_cache_out_0 … kvo_cache_out_N   (same N)
        fio_cache_out_<idx0> … fio_cache_out_<idxM>  (same M, FI only)
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        use_controlnet: bool = False,
        use_ipadapter: bool = False,
        control_input_names: Optional[List[str]] = None,
        num_tokens: int = 4,
        kvo_cache_structure: Optional[List[int]] = None,
        fi_layer_count: int = 0,
        lora_order: Optional[List[Tuple[str, str]]] = None,
        **kwargs,
    ):
        super().__init__()
        kvo_cache_structure = kvo_cache_structure or []
        self.use_controlnet = use_controlnet
        self.use_ipadapter = use_ipadapter
        self.controlnet_wrapper = None
        self.ipadapter_wrapper = None
        self.unet = unet
        self.kvo_cache_structure = kvo_cache_structure
        self.fi_layer_count = fi_layer_count
        self._lora_order: List[Tuple[str, str]] = lora_order or []

        # Precompute kvo cache count for arg-splitting in _basic_unet_forward
        self._kvo_cache_count: int = sum(sum(block) for block in kvo_cache_structure)

        # Apply IPAdapter first (installs processors into UNet)
        if use_ipadapter:
            ipadapter_kwargs = {k: v for k, v in kwargs.items() if k in ["install_processors"]}
            if "install_processors" not in ipadapter_kwargs:
                ipadapter_kwargs["install_processors"] = True

            self.ipadapter_wrapper = create_ipadapter_wrapper(unet, num_tokens=num_tokens, **ipadapter_kwargs)
            self.unet = self.ipadapter_wrapper.unet

        # Apply ControlNet second (wraps whatever UNet we have)
        if use_controlnet and control_input_names:
            controlnet_kwargs = {k: v for k, v in kwargs.items() if k in ["num_controlnets", "conditioning_scales"]}

            self.controlnet_wrapper = create_controlnet_wrapper(
                self.unet, control_input_names, kvo_cache_structure, **controlnet_kwargs
            )

        # Collect every PEFT BaseTunerLayer touched by a loaded LoRA adapter, capturing
        # its base scaling (lora_alpha/r) and asserting the vanilla/unmerged/enabled
        # invariants dynamic scaling depends on. Raises RuntimeError immediately on any
        # violation — see _collect_lora_layers' docstring.
        self._lora_layers: Dict[str, List[Tuple[BaseTunerLayer, float]]] = (
            _collect_lora_layers(self.unet, self._lora_order) if self._lora_order else {}
        )

        # Best-effort collection at construction time — may return [] if processors are
        # not yet installed (e.g. when wrapper.py constructs UnifiedExportWrapper before
        # the if-use_cached_attn block installs CachedSTAttnProcessor2_0).
        # Call refresh_fi_procs() after processor installation for the authoritative list.
        self._fi_procs: List = _collect_fi_processors(self.unet) if fi_layer_count > 0 else []

    def refresh_fi_procs(self) -> None:
        """Re-collect FI-eligible processors from the UNet.

        Call this AFTER installing ``CachedSTAttnProcessor2_0`` on attn1 layers
        (i.e. after the ``if use_cached_attn:`` block in wrapper.py).  Construction-
        time collection runs before processors are installed, so ``self._fi_procs``
        is empty at that point.  This method is the authoritative collection.

        Raises ``RuntimeError`` if ``fi_layer_count > 0`` but the collected count
        does not match — fail-fast replaces the cryptic ONNX output-count error.
        """
        if self.fi_layer_count == 0:
            return
        self._fi_procs = _collect_fi_processors(self.unet)
        if len(self._fi_procs) != self.fi_layer_count:
            raise RuntimeError(
                f"refresh_fi_procs: expected {self.fi_layer_count} FI-eligible processors "
                f"but found {len(self._fi_procs)}.  Check that CachedSTAttnProcessor2_0 "
                f"was installed on all fi_eligible attn1 layers before this call."
            )

    def _set_fi_cache(
        self,
        fi_args: tuple,
        fi_strength: Optional[torch.Tensor],
        fi_threshold: Optional[torch.Tensor],
    ) -> None:
        """Assign FI cache inputs to eligible processors before the UNet forward.

        Mirrors the ``set_ipadapter_scale`` pattern from unet_ipadapter_export.py:
        attributes are written on each processor object so that the ONNX tracer
        sees them as graph inputs flowing into the FI blend op.
        """
        for proc, fi_slice in zip(self._fi_procs, fi_args):
            proc._fi_cache = fi_slice
            proc._fi_strength = fi_strength
            proc._fi_threshold = fi_threshold

    def _set_lora_scale(self, lora_scale: torch.Tensor) -> None:
        """Write the live per-adapter weight into every PEFT layer's ``scaling`` dict
        before the UNet forward.

        Mirrors ``set_ipadapter_scale`` (unet_ipadapter_export.py) and ``_set_fi_cache``
        above: the traced tensor is stashed directly on the consuming object (here,
        PEFT's ``BaseTunerLayer.scaling`` dict, normally a plain float) so the ONNX
        tracer follows the data edge without threading the argument through every
        transformer block. ``base_scale`` is ``lora_alpha/r``, captured once at adapter-
        activation time in wrapper.py — multiplying it back in here reproduces
        ``fuse_lora(lora_scale=w)``'s exact math, just live instead of baked in.

        NOTE: this assigns a tensor into a dict PEFT's own code treats as floats
        (``LoraLayer.scaling``). Safe for the vanilla forward() multiply this class
        guards for (see ``_collect_lora_layers``), but PEFT helpers such as
        ``scale_layer``/``unscale_layer``/``scale_lora_layers`` are not called anywhere
        in this codebase and would misbehave against a tensor value if that changes.
        """
        for i, (_path, adapter_name) in enumerate(self._lora_order):
            weight = lora_scale[i]
            for module, base_scale in self._lora_layers[adapter_name]:
                module.scaling[adapter_name] = base_scale * weight

    def _basic_unet_forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        """Basic UNet forward that passes through all parameters to handle any model type.

        Positional *args layout (after stripping ipadapter_scale in forward()):
            args[:kvo_count]                        → kvo_cache tensors
            args[kvo_count:kvo_count+fi_count]      → fi_cache tensors (FI only)
            args[kvo_count+fi_count]                → fi_strength scalar (FI only)
            args[kvo_count+fi_count+1]              → fi_threshold scalar (FI only)
        """
        kvo_count = self._kvo_cache_count
        fi_count = self.fi_layer_count

        # Split positional args into kvo_cache, fi_cache, and FI scalars
        kvo_args = args[:kvo_count]
        fi_args: tuple = ()
        fi_strength: Optional[torch.Tensor] = None
        fi_threshold: Optional[torch.Tensor] = None

        if fi_count > 0:
            fi_args = args[kvo_count : kvo_count + fi_count]
            fi_scalar_base = kvo_count + fi_count
            if len(args) > fi_scalar_base:
                fi_strength = args[fi_scalar_base]
            if len(args) > fi_scalar_base + 1:
                fi_threshold = args[fi_scalar_base + 1]

        formatted_kvo_cache: List = []
        if kvo_args:
            formatted_kvo_cache = convert_list_to_structure(kvo_args, self.kvo_cache_structure)

        # Assign FI inputs to processors BEFORE the UNet call so ONNX tracing
        # includes the dependency edge from fio_cache_in tensors into the FI ops.
        if fi_args and self._fi_procs:
            self._set_fi_cache(fi_args, fi_strength, fi_threshold)

        # Auto-generate SDXL conditioning if missing and UNet requires it
        if "added_cond_kwargs" not in kwargs or kwargs.get("added_cond_kwargs") is None:
            base_unet = self.unet
            if hasattr(base_unet, "config") and getattr(base_unet.config, "addition_embed_type", None) == "text_time":
                batch_size = sample.shape[0]
                kwargs["added_cond_kwargs"] = {
                    "text_embeds": torch.zeros(batch_size, 1280, device=sample.device, dtype=sample.dtype),
                    "time_ids": torch.zeros(batch_size, 6, device=sample.device, dtype=sample.dtype),
                }

        unet_kwargs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
            "kvo_cache": formatted_kvo_cache,
            **kwargs,  # Pass through all additional parameters (SDXL, future model types, etc.)
        }
        res = self.unet(**unet_kwargs)

        if not kvo_args:
            # No cache at all — return latent only (same as before)
            return res[0]

        # Collect FI output tensors written by each processor's __call__ and append
        # them to the return tuple.  ONNX flattens (latent, kvo_nested, fi_0, fi_1, …)
        # depth-first, producing the flat output ordering that matches get_output_names:
        #   [latent, kvo_cache_out_0…N-1, fio_cache_out_<idx0>…<idxM-1>]
        if fi_count > 0 and self._fi_procs:
            fi_cache_outs = tuple(proc._fi_cache_out for proc in self._fi_procs if proc._fi_cache_out is not None)
            return (res[0], res[1]) + fi_cache_outs

        return res

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass that handles any UNet parameters via **kwargs passthrough"""
        # Handle IP-Adapter runtime scale vector as a positional argument placed before control tensors
        if self.use_ipadapter and self.ipadapter_wrapper is not None:
            # ipadapter_scale is appended as the first extra positional input after the 3 base inputs
            if len(args) == 0:
                import logging

                logging.getLogger(__name__).error(
                    "UnifiedExportWrapper: ipadapter_scale missing; required when use_ipadapter=True"
                )
                raise RuntimeError("UnifiedExportWrapper: ipadapter_scale tensor is required when use_ipadapter=True")
            ipadapter_scale = args[0]
            if not isinstance(ipadapter_scale, torch.Tensor):
                import logging

                logging.getLogger(__name__).error(
                    f"UnifiedExportWrapper: ipadapter_scale wrong type: {type(ipadapter_scale)}"
                )
                raise TypeError("ipadapter_scale must be a torch.Tensor")
            try:
                import logging

                logging.getLogger(__name__).debug(
                    f"UnifiedExportWrapper: ipadapter_scale shape={tuple(ipadapter_scale.shape)}, dtype={ipadapter_scale.dtype}"
                )
            except Exception:
                pass
            # assign per-layer scale tensors into processors
            self.ipadapter_wrapper.set_ipadapter_scale(ipadapter_scale)
            # remove it from control args before passing to controlnet wrapper
            args = args[1:]

        if self._lora_order:
            # lora_scale is appended right after ipadapter_scale (see class docstring)
            if len(args) == 0:
                raise RuntimeError("UnifiedExportWrapper: lora_scale tensor is required when LoRA adapters are loaded")
            lora_scale = args[0]
            if not isinstance(lora_scale, torch.Tensor):
                raise TypeError(f"lora_scale must be a torch.Tensor, got {type(lora_scale)}")
            self._set_lora_scale(lora_scale)
            args = args[1:]

        if self.controlnet_wrapper:
            # ControlNet wrapper handles the UNet call with all parameters.
            # When FI is active the tail of *args is:
            #   [...controls, ...kvo_in, ...fio_in, fi_strength, fi_threshold]
            # The CN wrapper only understands (controls, kvo_in), so we strip the
            # FI tail here, wire it into processors via _set_fi_cache, then
            # collect and append the FI outputs exactly as _basic_unet_forward does.
            fi_count = self.fi_layer_count
            if fi_count > 0 and self._fi_procs:
                fi_tail = fi_count + 2  # fio tensors + fi_strength + fi_threshold
                fi_args = args[-fi_tail:-2]
                fi_strength_t = args[-2]
                fi_threshold_t = args[-1]
                cn_args = args[:-fi_tail]
                self._set_fi_cache(fi_args, fi_strength_t, fi_threshold_t)
                res = self.controlnet_wrapper(sample, timestep, encoder_hidden_states, *cn_args, **kwargs)
                fi_cache_outs = tuple(proc._fi_cache_out for proc in self._fi_procs if proc._fi_cache_out is not None)
                return (res[0], res[1]) + fi_cache_outs
            else:
                return self.controlnet_wrapper(sample, timestep, encoder_hidden_states, *args, **kwargs)
        else:
            # Basic UNet call with all parameters passed through
            return self._basic_unet_forward(sample, timestep, encoder_hidden_states, *args, **kwargs)
