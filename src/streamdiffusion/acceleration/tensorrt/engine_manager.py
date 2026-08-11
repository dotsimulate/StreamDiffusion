import hashlib
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, cast

logger = logging.getLogger(__name__)


class EngineType(Enum):
    """Engine types supported by the TensorRT engine manager."""

    UNET = "unet"
    VAE_ENCODER = "vae_encoder"
    VAE_DECODER = "vae_decoder"
    CONTROLNET = "controlnet"
    SAFETY_CHECKER = "safety_checker"


class EngineManager:
    """
    Universal TensorRT engine manager using factory pattern.

    Consolidates all engine management logic into a single class:
    - Path generation (moves create_prefix from wrapper.py)
    - Compilation (moves compile_* calls from wrapper.py)
    - Loading (returns appropriate engine objects)
    """

    def __init__(self, engine_dir: str):
        """Initialize with engine directory."""
        self.engine_dir = Path(engine_dir)
        self.engine_dir.mkdir(parents=True, exist_ok=True)

        # Import the existing compile functions from tensorrt/__init__.py
        from streamdiffusion.acceleration.tensorrt import (
            compile_controlnet,
            compile_safety_checker,
            compile_unet,
            compile_vae_decoder,
            compile_vae_encoder,
        )
        from streamdiffusion.acceleration.tensorrt.runtime_engines.controlnet_engine import ControlNetModelEngine
        from streamdiffusion.acceleration.tensorrt.runtime_engines.unet_engine import UNet2DConditionModelEngine

        # TODO: add function to get use_cuda_graph from kwargs
        # Engine configurations - maps each type to its compile function and loader
        self._configs = {
            EngineType.UNET: {
                "filename": "unet.engine",
                "compile_fn": compile_unet,
                "loader": lambda path, cuda_stream, **kwargs: UNet2DConditionModelEngine(
                    str(path), cuda_stream, use_cuda_graph=True
                ),
            },
            EngineType.VAE_ENCODER: {
                "filename": "vae_encoder.engine",
                "compile_fn": compile_vae_encoder,
                "loader": lambda path, cuda_stream, **kwargs: str(path),  # Return path for AutoencoderKLEngine
            },
            EngineType.VAE_DECODER: {
                "filename": "vae_decoder.engine",
                "compile_fn": compile_vae_decoder,
                "loader": lambda path, cuda_stream, **kwargs: str(path),  # Return path for AutoencoderKLEngine
            },
            EngineType.CONTROLNET: {
                "filename": "cnet.engine",
                "compile_fn": compile_controlnet,
                "loader": lambda path, cuda_stream, **kwargs: ControlNetModelEngine(
                    str(path),
                    cuda_stream,
                    use_cuda_graph=kwargs.get("use_cuda_graph", False),
                    model_type=kwargs.get("model_type", "sd15"),
                ),
            },
            EngineType.SAFETY_CHECKER: {
                "filename": "safety_checker.engine",
                "compile_fn": compile_safety_checker,
                "loader": lambda path, cuda_stream, **kwargs: str(path),
            },
        }

    def _lora_signature(self, lora_dict: Dict[str, float]) -> str:
        """Create a short, stable signature for a set of LoRAs.

        Uses sorted basenames and weights, hashed to a short hex to avoid
        long/invalid paths while keeping cache keys stable across runs.
        """
        # Build canonical string of basename:weight pairs
        parts = []
        for path, weight in sorted(lora_dict.items(), key=lambda x: str(x[0])):
            base = Path(str(path)).name  # basename only
            parts.append(f"{base}:{weight}")
        canon = "|".join(parts)
        h = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:10]
        return f"{len(lora_dict)}-{h}"

    def _calibration_image_signature(self, path: Optional[str]) -> Optional[str]:
        """Short content hash of `fp8_calibration_style_image` (fp8-round-9.1)
        — a single file, or every recognized-extension file in a directory,
        sorted by filename to match wrapper.py's
        `_load_fp8_calibration_style_images` loading order.

        Hashes file *contents*, not the path string, so editing an image in
        place (same filename, new bytes) or adding/removing a file correctly
        forks the cache — a path-only signature would miss both. Returns None
        when nothing is configured or nothing hashable was found, so the
        `--ci<hash>` suffix is skipped entirely rather than forked on an
        empty signature.
        """
        if not path:
            return None
        from .fp8_quantize import _list_calibration_images

        files = _list_calibration_images(path)
        if not files:
            return None
        h = hashlib.sha1()
        for f in files:
            try:
                h.update(f.read_bytes())
            except OSError:
                continue
        return h.hexdigest()[:6]

    def _fp8_recipe_tag(
        self,
        fp8_mha_qdq: bool,
        fp8_scale_headroom: float,
        fp8_exclude_attention: bool,
        fp8_exclude_ipadapter: bool = False,
    ) -> str:
        """Build the FP8 recipe cache-key tag. Called from both the "canonical" prefix
        (feeds the UNet directory-name hash) and the final human-readable UNet prefix —
        the two call sites must stay byte-identical, which is why this is factored out
        rather than duplicated (fp8-round-8 lesson: engine_manager.py:194 and :284 drifted
        out of sync once already when the tag was inlined at both sites).

        v3 -> v4 (fp8-round-8): the underlying scale-conversion math changed (see
        fp8_quantize.py::_rescale_fp8_qdq_scales), so every existing v3 engine is stale
        under v4's semantics even with every recipe flag at its default — the base tag
        itself must fork, not just gain a suffix.
        """
        tag = "--fp8v4"
        if fp8_mha_qdq:
            tag += "-mhaq"
        if fp8_exclude_attention:
            tag += "-noattn"
        if fp8_exclude_ipadapter:
            tag += "-noip"
        if fp8_scale_headroom != 1.0:
            tag += f"-hr{fp8_scale_headroom:g}"
        return tag

    def _trt_cc_tag(self) -> str:
        """TRT version + compute-capability cache-key token, shared by every engine
        type so a TRT upgrade or GPU change auto-invalidates every branch's cache, not
        just the standard UNet/VAE branch (fp8-round-12: previously only that branch
        called this, so ControlNet engines silently never invalidated on a TRT/GPU
        change). Fails silently to "" if tensorrt/torch aren't importable yet (e.g.
        during a partial install) -- same semantics as the inline block this replaces.
        Factored out for the same anti-drift reason as _fp8_recipe_tag above: two
        hand-written copies of a cache-key token is how the fp8-round-8 prefix-vs-hash
        drift bug happened."""
        try:
            import tensorrt as _trt
            import torch as _torch

            _cc = _torch.cuda.get_device_capability(0)
            return f"--trt{_trt.__version__}--cc{_cc[0]}{_cc[1]}"
        except Exception:
            return ""

    def get_engine_path(
        self,
        engine_type: EngineType,
        model_id_or_path: str,
        max_batch_size: int,
        min_batch_size: int,
        mode: str,
        use_tiny_vae: bool,
        lora_dict: Optional[Dict[str, float]] = None,
        ipadapter_scale: Optional[float] = None,
        ipadapter_tokens: Optional[int] = None,
        controlnet_model_id: Optional[str] = None,
        is_faceid: Optional[bool] = None,
        use_cached_attn: bool = False,
        use_feature_injection: bool = False,
        use_controlnet: bool = False,
        fp8: bool = False,
        fp8_mha_qdq: bool = False,
        fp8_scale_headroom: float = 1.0,
        fp8_exclude_attention: bool = False,
        fp8_exclude_ipadapter: bool = False,
        fp8_calib_t_index_len: Optional[int] = None,
        fp8_calib_num_steps: Optional[int] = None,
        fp8_calib_scheduler_type: Optional[str] = None,
        fp8_calibration_style_image: Optional[str] = None,
        resolution: Optional[tuple] = None,
        builder_optimization_level: Optional[int] = None,
        build_static_batch: Optional[bool] = None,
        static_batch_size: Optional[int] = None,
        pin_cache_frames: bool = False,
        cache_maxframes: Optional[int] = None,
    ) -> Path:
        """
        Generate engine path using wrapper.py's current logic.

        Moves and consolidates create_prefix() function from wrapper.py lines 995-1014.
        Special handling for ControlNet engines which use model_id-based directories.
        """
        # self._configs' inner dicts also hold compile_fn/loader callables under other
        # keys, so the type checker infers "filename" as str | Callable | Callable across
        # the whole dict shape rather than narrowing per-key. The "filename" value is
        # always a plain str at runtime (see __init__); cast pins that down for the
        # `self.engine_dir / prefix / filename` join below.
        filename = cast(str, self._configs[engine_type]["filename"])
        optlvl_suffix = f"--optlvl{builder_optimization_level}" if builder_optimization_level is not None else ""

        if engine_type == EngineType.CONTROLNET:
            # ControlNet engines use special model_id-based directory structure
            if controlnet_model_id is None:
                raise ValueError("get_engine_path: controlnet_model_id required for CONTROLNET engines")

            # Convert model_id to directory name format (replace "/" with "_")
            model_dir_name = controlnet_model_id.replace("/", "_")

            if resolution is not None:
                prefix = f"controlnet_{model_dir_name}--min_batch-{min_batch_size}--max_batch-{max_batch_size}--res-{resolution[0]}x{resolution[1]}"
            else:
                prefix = f"controlnet_{model_dir_name}--min_batch-{min_batch_size}--max_batch-{max_batch_size}--dyn-256-1024"
            # fp8-round-12: ControlNet is always built with build_static_batch=True
            # (_get_default_controlnet_build_options), at opt_batch_size baked to the
            # caller's trt_unet_batch_size -- but that value previously never reached
            # this path, only the capacity range (min/max_batch above, fixed wrapper
            # constructor defaults) did. Two different t-index counts collided on the
            # same directory and the second failed set_input_shape on the first frame.
            # Mirrors the standard branch's --sbatch/--batch-/_trt_cc_tag guards below
            # (see :368-376 and _trt_cc_tag) so both branches read the same way.
            if build_static_batch is not None:
                prefix += f"--sbatch{int(build_static_batch)}"
            if build_static_batch and static_batch_size is not None:
                prefix += f"--batch-{static_batch_size}"
            prefix += self._trt_cc_tag()
            fp8_suffix = "--fp8" if fp8 else ""
            engine_path = self.engine_dir / (prefix + optlvl_suffix + fp8_suffix) / filename
        else:
            # Standard engines use the unified prefix format
            # Extract base name (from wrapper.py lines 1002-1003)
            maybe_path = Path(model_id_or_path)
            base_name = maybe_path.stem if maybe_path.exists() else model_id_or_path

            # Create prefix (from wrapper.py lines 1005-1013)
            prefix = f"{base_name}--tiny_vae-{use_tiny_vae}--min_batch-{min_batch_size}--max_batch-{max_batch_size}"

            # IP-Adapter differentiation: add type and (optionally) tokens
            # Keep scale out of identity for runtime control, but include a type flag to separate caches
            if is_faceid:
                prefix += "--fid"
            if ipadapter_tokens is not None:
                prefix += f"--tokens{ipadapter_tokens}"

            # Fused Loras - use concise hashed signature to avoid long/invalid paths.
            # Only UNet engines bake LoRA weights; VAE and other standard engines are
            # LoRA-agnostic, so scoping the suffix to UNET prevents redundant VAE rebuilds
            # every time the LoRA dict changes.
            if engine_type == EngineType.UNET and lora_dict is not None and len(lora_dict) > 0:
                prefix += f"--lora-{self._lora_signature(lora_dict)}"

            if engine_type == EngineType.UNET:
                prefix += f"--use_cached_attn-{use_cached_attn}"
                # FI suffix MUST come right after cached_attn so stale engines
                # (built without FI bindings) are never loaded when FI is enabled.
                prefix += f"--fi-{use_feature_injection}"
                if use_controlnet:
                    prefix += "--controlnet"
                if fp8:
                    # Quantization-recipe changes are NOT otherwise part of the cache
                    # key, so experimental recipe flags must fork the tag here. The
                    # base tag must stay byte-identical when every recipe flag is off
                    # — bumping it would orphan the deployed production engines.
                    prefix += self._fp8_recipe_tag(
                        fp8_mha_qdq, fp8_scale_headroom, fp8_exclude_attention, fp8_exclude_ipadapter
                    )
                    # Band-based calibration schedule (fp8-round-5-handoff Step 2f):
                    # calib_data.npz is cached inside the engine dir and short-circuited
                    # on existence AND explicitly preserved by artifact cleanup, so a
                    # config-agnostic tag would silently keep serving pre-fix calibration
                    # data forever. "calv1" forks away from every engine built before this
                    # schedule existed; len(t_index_list)/num_inference_steps/scheduler_type
                    # fork again only when the *derived band layout* would actually change.
                    # Deliberately excludes t_index_list's literal values — a live /t_list
                    # change must not force a rebuild.
                    # "calv2" (fp8-round-5-handoff Step 4): forks again away from calv1
                    # engines, whose calib_data.npz was captured against the raw diffusers
                    # UNet -- 91.2% of quantized activations (kvo_cache_in_*, fio_cache_in_*,
                    # fi_strength/threshold, ipadapter_scale) calibrated on zeros/ones
                    # (Defect B) plus the band-0 t=999 spurious index (Defect A). Deliberately
                    # still excludes the scalar deployment values themselves (fi_strength,
                    # ipadapter_scale, ...) for the same "keep runtime scale out of identity"
                    # reason as above -- a live scale tweak must not force a rebuild.
                    # "calv3" (fp8-round-6.1, then fp8-round-7): forks again away from calv2
                    # engines. Round 6.1 fixed forward-only spill dropping budget in the
                    # terminal band (observed: 7/8 distinct timesteps for t_index_list=
                    # [7,16,25]) with a round-robin top-up. Round 7 replaced the fixed-decade
                    # band grid itself with neighbour-midpoint bands derived from the
                    # configured values, fixing two more defects the decade grid had: a short
                    # t_index_list collapsed most of the budget into the noisiest indices
                    # (single-entry configs put 7/8 rows in index 1..9), and band k was
                    # assigned to t_index_list[k] positionally with no guarantee the value
                    # actually landed in it. No engine ever shipped under the intermediate
                    # top-up-only form of calv3, so the label is reused rather than bumped
                    # again — band_width is no longer a parameter (bands derive from the
                    # values, not a fixed width), so it is dropped from the tag.
                    # "calv4" (fp8-round-9): forks again away from calv3 engines, whose
                    # calib_data.npz calibrated the IP-Adapter cross-attention branch on
                    # identically-zero input on BOTH axes independently — encoder_hidden_states
                    # was zero-padded (bias-free to_k_ip/to_v_ip -> zero activations) AND
                    # ipadapter_scale was baked in at its then-current deployment value (0.0 at
                    # capture time), zeroing the merge term a second, independent way. Round 9
                    # feeds real IP-Adapter projection tokens (or their measured zeros-surrogate)
                    # instead of zero-padding, and calibrates the scale at its runtime bound
                    # (max(config, 1.0)) instead of the deployment snapshot. calv3 engines are
                    # not invalidated by anything else changing here — fp8v4 stays, the rescale
                    # math (_rescale_fp8_qdq_scales) is unchanged.
                    # "calv5" (fp8-round-9.1): forks again away from calv4 engines. Round 9's
                    # real-token path only fired for `is_faceid and not is_plus`; every other
                    # adapter flavour, including the shipped `type: regular` config, still fell
                    # through to the zero-pad reconciliation calv4 was meant to fix (confirmed
                    # byte-identical fp8_uncalibrated_scales=420 across a calv4 rebuild). Round
                    # 9.1 (_resolve_fp8_ipadapter_calibration_tokens, wrapper.py) dispatches on
                    # image_proj_model.proj's actual shape (nn.Sequential vs. bare nn.Linear)
                    # instead of adapter flavour, so regular adapters now get a real zeros-
                    # surrogate too. calv4 engines are not invalidated by anything else changing
                    # here — fp8v4 stays, the rescale math is unchanged.
                    # "calv6" (fp8-round-10): forks again away from calv5 engines, whose
                    # calib_data.npz was captured with no height/width passed to pipe(), so
                    # diffusers silently fell back to pipe.unet.config.sample_size (512 for
                    # SDXL-Turbo) regardless of the engine's actual build resolution. Confirmed
                    # via two completed builds (h92807e04ef45 @ 512, hb290e7269028 @ 1024):
                    # both captured identical (8, 4, 64, 64) `sample`/512-valued `time_ids`,
                    # which is the actual evidence (`fp8_scale_realized_peak` also matched
                    # bit-for-bit between them, but that is NOT corroborating: it samples
                    # only static-weight QuantizeLinear nodes (_rescale_fp8_qdq_scales,
                    # fp8_quantize.py) and is invariant to calibration content by
                    # construction). Real-vs-zero-padded seq fraction on kvo_cache_in_*/
                    # fio_cache_in_* scales as (512/R)² — 100% at 512, 25% at 1024, 11% at 1536
                    # — so every build above 512 was calibrating on a mostly zero-padded cache.
                    # capture_calibration_data now takes image_height/image_width and
                    # builder.py passes opt_image_height/opt_image_width (mirroring the
                    # ControlNet branch, which already did this). 512 builds are unaffected in
                    # substance — the fix makes explicit what the fallback already chose — but
                    # the tag still forks so every resolution re-captures at its real shape
                    # rather than reusing a stale, wrongly-shaped calv5 calib_data.npz.
                    # "calv7" (fp8-round-11): forks again away from calv6 engines.
                    # _select_calibration_calls took bucket[round_idx] with round_idx
                    # starting at 0, so every distinct timestep's selected row came from
                    # the SAME early pipe() call -- all 8 calibration rows were drawn from
                    # prompt batch 0, and the other ~31 of 32 captured batches contributed
                    # zero signal to any amax, despite calib_data.npz showing correct
                    # per-timestep stratification (confirmed on-disk: the calv6
                    # h92807e04ef45/haebb243bc74e/hb290e7269028 captures all show 8 rows,
                    # 8 distinct timesteps -- distinctness of timestep was checked, but
                    # never the source call index behind it). Selection is now staggered
                    # per ordered-timestep position so the 8 selected rows span 8 distinct
                    # prompt batches instead of 1 (see _select_calibration_calls,
                    # fp8_quantize.py); same row count, same memory footprint. The
                    # recorder cap feeding that selection was also widened (byte-budgeted,
                    # SDTD_FP8_CALIB_RECORD_BYTES, on top of the existing call-count cap).
                    # CORRECTION (fp8-round-14, see below): that widening did not actually
                    # work -- SDTD_FP8_CALIB_RECORD_BYTES (4 GiB) trips before the widened
                    # call-count cap (64) at ~124 MiB/call on a real SDXL-Turbo UNet, so
                    # only ~33 of the 64 calls needed for the full 8-batch stagger were ever
                    # recorded. calv6 engines are not invalidated by anything else changing
                    # here -- fp8v4 stays, the rescale math is unchanged.
                    # "calv8" (fp8-round-14): forks again away from calv7 engines, fixing
                    # the byte-cap-defeats-call-cap defect the calv7 paragraph above
                    # describes. Measured on disk (hashing per-n_itr K chunks of
                    # kvo_cache_in_0): calv6 (pre-stagger-fix) has 8 distinct K sources;
                    # both calv7 engines (h081fefde15ae @ 512x512, hfb5ce64fea8d @ 640x384)
                    # have only 4 -- the second half of every stagger (calls 36/45/54/63)
                    # was silently dropped, and _pool_for_layer's cyclic reuse of the
                    # surviving 4 sources (not a zero-fill -- see fp8_quantize.py's
                    # _pool_missed_calls warning) halved kvo/fio calibration diversity in
                    # every calv7 build at both resolutions. capture_calibration_data now
                    # predicts, before the capture loop runs, exactly which ~8 calls
                    # _select_calibration_calls will select (_predict_selected_calls,
                    # fp8_quantize.py) and gates the K/V/FI recorder hooks on membership in
                    # that set instead of a call-count prefix -- so the recorder admits
                    # exactly the calls that matter and peak recorder memory drops from
                    # ~4 GiB to ~1 GiB, comfortably under the byte cap. calv7 engines are
                    # not invalidated by anything else changing here -- fp8v4 stays, the
                    # rescale math is unchanged.
                    if fp8_calib_t_index_len is not None and fp8_calib_num_steps is not None:
                        prefix += (
                            f"--calv8-n{fp8_calib_t_index_len}"
                            f"-s{fp8_calib_num_steps}"
                            f"-{fp8_calib_scheduler_type or 'na'}"
                        )
                    # fp8-round-9.1: the calibration image set is part of the recipe —
                    # swapping which image(s) feed the real-token path must fork the
                    # cache the same way every other fp8_* flag does. Content-hashed
                    # (see _calibration_image_signature) rather than path-hashed so an
                    # edited file at the same path also forks. Not folded into
                    # _fp8_recipe_tag: like calv5, this only needs to live in the
                    # hashed `canonical` string below, not in the short human-readable
                    # directory name.
                    _ci_sig = self._calibration_image_signature(fp8_calibration_style_image)
                    if _ci_sig is not None:
                        prefix += f"--ci{_ci_sig}"
                # Encode the actual batch-profile policy so that a static-batch engine
                # and a dynamic-batch engine never share the same directory.
                # The capacity range (min_batch / max_batch above) is the same for both,
                # so without this suffix a stale dynamic engine is silently reused after
                # the static-batch switch — and TRT emits "l2tc doesn't take effect"
                # because the loaded engine has a symbolic batch dim.
                if build_static_batch is not None:
                    prefix += f"--sbatch{int(build_static_batch)}"
                # A static-batch engine only accepts the exact batch it was built
                # with (= steps x frame_buffer x cfg factor), so that value must be
                # part of the cache key: without it a 1-step config resolves to an
                # engine frozen at batch 2 and fails set_input_shape on the first
                # frame. min_batch/max_batch above are only the capacity range.
                if build_static_batch and static_batch_size is not None:
                    prefix += f"--batch-{static_batch_size}"
                # pin_cache_frames bakes cache_maxframes into the engine (min==opt==max on
                # the KVO/FI cache-frames axis) so TRT l2tc can engage — the resulting engine
                # only accepts that exact frame count and is not interchangeable with an
                # unpinned engine, so the value must be part of the cache key.
                if pin_cache_frames and cache_maxframes is not None:
                    prefix += f"--cachef{cache_maxframes}"

            prefix += optlvl_suffix

            prefix += f"--mode-{mode}"

            # Embed TRT version + compute capability so upgrading TRT invalidates
            # stale engines automatically. Old engine dirs are orphaned (not deleted),
            # keeping them available for rollback.
            prefix += self._trt_cc_tag()

            if resolution is not None:
                prefix += f"--res-{resolution[0]}x{resolution[1]}"

            if engine_type == EngineType.UNET:
                # The UNet prefix above encodes every build flag verbatim and can exceed
                # ~170 chars. Combined with a real-world engine_dir root, that pushes
                # unet.engine.onnx / .opt.onnx past Windows' 260-char MAX_PATH — the
                # directory itself fits (mkdir succeeds) but torch.onnx.export's
                # open(path, "wb") then fails with a bare FileNotFoundError. Compact the
                # directory name to a short hash of the full prefix (same idea as
                # _lora_signature) so every distinct config still maps to a distinct,
                # stable directory, just without spelling out every flag on disk.
                canonical = prefix
                short_hash = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
                fp8_tag = (
                    self._fp8_recipe_tag(fp8_mha_qdq, fp8_scale_headroom, fp8_exclude_attention, fp8_exclude_ipadapter)
                    if fp8
                    else ""
                )
                res_tag = f"--res-{resolution[0]}x{resolution[1]}" if resolution is not None else ""
                prefix = f"{base_name}{fp8_tag}--h{short_hash}{res_tag}"
                logger.debug(f"EngineManager: UNet cache key h{short_hash} = {canonical}")

            engine_path = self.engine_dir / prefix / filename

        # Belt-and-suspenders: warn (rather than silently mkdir-then-fail-on-write) if
        # the longest artifact derived from this path would still exceed Windows'
        # 260-char MAX_PATH — e.g. an unusually deep user-configured engine_dir.
        # ".opt.onnx" is the longest suffix appended to engine_path (see builder.py).
        # Covers BOTH branches (fp8-round-12): ControlNet has no hash compaction to
        # fall back on, and its prefix just grew three tokens longer, so it needs this
        # warning at least as much as the standard branch does.
        longest_derived = len(str(engine_path)) + len(".opt.onnx")
        if longest_derived >= 250:
            logger.warning(
                f"EngineManager: engine path is {longest_derived} chars once .opt.onnx is "
                f"appended (Windows MAX_PATH=260). Move engine_dir closer to the drive root "
                f"if TensorRT export fails with FileNotFoundError: {engine_path}"
            )

        return engine_path

    def _get_embedding_dim_for_model_type(self, model_type: str) -> int:
        """Get embedding dimension based on model type."""
        if model_type.lower() in ["sdxl"]:
            return 2048
        elif model_type.lower() in ["sd21", "sd2.1"]:
            return 1024
        else:  # sd15 and others
            return 768

    def _execute_compilation(
        self, compile_fn, engine_path: Path, model, model_config, batch_size: int, kwargs: Dict
    ) -> None:
        """Execute compilation with common pattern to eliminate duplication."""
        compile_fn(
            model,
            model_config,
            str(engine_path) + ".onnx",
            str(engine_path) + ".opt.onnx",
            str(engine_path),
            opt_batch_size=batch_size,
            engine_build_options=kwargs.get("engine_build_options", {}),
        )

    def _prepare_controlnet_models(self, kwargs: Dict):
        """Prepare ControlNet models for compilation."""
        import torch

        from streamdiffusion.acceleration.tensorrt.models.controlnet_models import create_controlnet_model

        model_type = kwargs.get("model_type", "sd15")
        max_batch_size = kwargs["max_batch_size"]
        min_batch_size = kwargs["min_batch_size"]
        embedding_dim = self._get_embedding_dim_for_model_type(model_type)

        # Create ControlNet model configuration
        controlnet_model = create_controlnet_model(
            model_type=model_type,
            unet=kwargs.get("unet"),
            model_path=kwargs.get("model_path", ""),
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            conditioning_channels=kwargs.get("conditioning_channels", 3),
        )

        # Prepare ControlNet model for compilation
        pytorch_model = kwargs["model"].to(dtype=torch.float16)

        return pytorch_model, controlnet_model

    def _get_default_controlnet_build_options(
        self,
        opt_image_height: int = 704,
        opt_image_width: int = 704,
        build_dynamic_shape: bool = False,
        builder_optimization_level: Optional[int] = None,
        fp8: bool = False,
    ) -> Dict:
        """Get default engine build options for ControlNet engines."""
        opts = {
            "opt_image_height": opt_image_height,
            "opt_image_width": opt_image_width,
            "build_dynamic_shape": build_dynamic_shape,
            "build_static_batch": True,
        }
        if build_dynamic_shape:
            # Match BaseModel/UNet's 256 floor (was 384) so [256, 384) resolutions
            # don't hard-fail with ControlNet active — see get_input_profile in
            # controlnet_models.py, which now derives its own floor from the same
            # BaseModel.min_image_shape instead of a separate hardcoded literal.
            opts["min_image_resolution"] = 256
            opts["max_image_resolution"] = 1024
        if builder_optimization_level is not None:
            opts["builder_optimization_level"] = builder_optimization_level
        if fp8:
            opts["fp8"] = True
            opts["fp8_allow_fp16_fallback"] = True
            opts["onnx_opset"] = 19
        return opts

    def compile_and_load_engine(
        self, engine_type: EngineType, engine_path: Path, load_engine: bool = True, **kwargs
    ) -> Any:
        """
        Universal compile and load logic for all engine types.

        Moves compilation blocks from wrapper.py lines 1200-1252, 1254-1283, 1285-1313.
        """
        if not engine_path.exists():
            # Get the appropriate compile function for this engine type
            config = self._configs[engine_type]
            compile_fn = config["compile_fn"]

            # Ensure parent directory exists
            engine_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle engine-specific compilation requirements
            if engine_type == EngineType.VAE_DECODER:
                # VAE decoder requires modifying forward method during compilation
                stream_vae = kwargs["stream_vae"]
                stream_vae.forward = stream_vae.decode
                try:
                    self._execute_compilation(
                        compile_fn, engine_path, kwargs["model"], kwargs["model_config"], kwargs["batch_size"], kwargs
                    )
                finally:
                    # Always clean up the forward attribute
                    delattr(stream_vae, "forward")
            elif engine_type == EngineType.CONTROLNET:
                # ControlNet requires special model creation and compilation
                model, model_config = self._prepare_controlnet_models(kwargs)
                self._execute_compilation(compile_fn, engine_path, model, model_config, kwargs["batch_size"], kwargs)
            else:
                # Standard compilation for UNet and VAE encoder
                self._execute_compilation(
                    compile_fn, engine_path, kwargs["model"], kwargs["model_config"], kwargs["batch_size"], kwargs
                )
        else:
            logger.info("EngineManager: engine_path already exists, skipping compile")

        if load_engine:
            return self.load_engine(engine_type, engine_path, **kwargs)
        else:
            logger.info("EngineManager: load_engine is False, skipping load engine")
            return None

    def load_engine(self, engine_type: EngineType, engine_path: Path, **kwargs: Dict) -> Any:
        """Load engine with type-specific handling."""
        config = self._configs[engine_type]
        loader = config["loader"]

        if engine_type == EngineType.UNET:
            # UNet engine needs special handling for metadata and error recovery
            loaded_engine = loader(engine_path, kwargs.get("cuda_stream"))
            self._set_unet_metadata(loaded_engine, kwargs)
            return loaded_engine
        elif engine_type == EngineType.CONTROLNET:
            # ControlNet engine needs model_type parameter
            return loader(
                engine_path,
                kwargs.get("cuda_stream"),
                model_type=kwargs.get("model_type", "sd15"),
                use_cuda_graph=kwargs.get("use_cuda_graph", False),
            )
        else:
            return loader(engine_path, kwargs.get("cuda_stream"))

    def _set_unet_metadata(self, loaded_engine, kwargs: Dict) -> None:
        """Set metadata on UNet engine for runtime use."""
        loaded_engine.use_control = kwargs.get("use_controlnet_trt", False)
        loaded_engine.use_ipadapter = kwargs.get("use_ipadapter_trt", False)

        if kwargs.get("use_controlnet_trt", False):
            loaded_engine.unet_arch = kwargs.get("unet_arch", {})

        if kwargs.get("use_ipadapter_trt", False):
            loaded_engine.ipadapter_arch = kwargs.get("unet_arch", {})
            # number of IP-attention layers for runtime vector sizing
            if "num_ip_layers" in kwargs and kwargs["num_ip_layers"] is not None:
                loaded_engine.num_ip_layers = kwargs["num_ip_layers"]

    def get_or_load_controlnet_engine(
        self,
        model_id: str,
        pytorch_model: Any,
        load_engine=True,
        model_type: str = "sd15",
        batch_size: int = 1,
        min_batch_size: int = 1,
        max_batch_size: int = 4,
        cuda_stream=None,
        use_cuda_graph: bool = False,
        unet=None,
        model_path: str = "",
        conditioning_channels: int = 3,
        opt_image_height: int = 704,
        opt_image_width: int = 704,
        builder_optimization_level: Optional[int] = None,
        fp8: bool = False,
    ) -> Any:
        """
        Get or load ControlNet engine, providing unified interface for ControlNet management.

        Replaces ControlNetEnginePool.get_or_load_engine functionality.
        """
        # fp8-round-12: build the options dict ONCE and feed both the path and the
        # compile step from it. Previously get_engine_path() never saw
        # build_static_batch/static_batch_size at all -- they weren't cache-key
        # inputs for ControlNet -- and _get_default_controlnet_build_options() was
        # called a second, independent time down in compile_and_load_engine's
        # engine_build_options kwarg. Two independently-constructed option dicts is
        # exactly how the prefix-vs-hash cache key drifted apart in fp8-round-8; a
        # single source here prevents a repeat.
        build_options = self._get_default_controlnet_build_options(
            opt_image_height=opt_image_height,
            opt_image_width=opt_image_width,
            builder_optimization_level=builder_optimization_level,
            fp8=fp8,
        )

        # Generate engine path using ControlNet-specific logic
        engine_path = self.get_engine_path(
            EngineType.CONTROLNET,
            model_id_or_path="",  # Not used for ControlNet
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            mode="",  # Not used for ControlNet
            use_tiny_vae=False,  # Not used for ControlNet
            controlnet_model_id=model_id,
            resolution=(opt_image_height, opt_image_width),
            builder_optimization_level=builder_optimization_level,
            build_static_batch=build_options["build_static_batch"],
            static_batch_size=batch_size,
            fp8=fp8,
        )

        # Compile and load ControlNet engine
        return self.compile_and_load_engine(
            EngineType.CONTROLNET,
            engine_path,
            load_engine=load_engine,
            model=pytorch_model,
            model_type=model_type,
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            cuda_stream=cuda_stream,
            use_cuda_graph=use_cuda_graph,
            unet=unet,
            model_path=model_path,
            conditioning_channels=conditioning_channels,
            engine_build_options=build_options,
        )
