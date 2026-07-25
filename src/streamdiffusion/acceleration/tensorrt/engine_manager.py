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
            fp8_suffix = "--fp8" if fp8 else ""
            return self.engine_dir / (prefix + optlvl_suffix + fp8_suffix) / filename
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
                    prefix += "--fp8v3"
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
            # keeping them available for rollback. Fails silently if tensorrt isn't
            # installed yet (e.g. during a partial install).
            try:
                import tensorrt as _trt
                import torch as _torch

                _cc = _torch.cuda.get_device_capability(0)
                prefix += f"--trt{_trt.__version__}--cc{_cc[0]}{_cc[1]}"
            except Exception:
                pass

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
                fp8_tag = "--fp8v3" if fp8 else ""
                res_tag = f"--res-{resolution[0]}x{resolution[1]}" if resolution is not None else ""
                prefix = f"{base_name}{fp8_tag}--h{short_hash}{res_tag}"
                logger.debug(f"EngineManager: UNet cache key h{short_hash} = {canonical}")

            engine_path = self.engine_dir / prefix / filename

            # Belt-and-suspenders: warn (rather than silently mkdir-then-fail-on-write)
            # if the longest artifact derived from this path would still exceed Windows'
            # 260-char MAX_PATH — e.g. an unusually deep user-configured engine_dir.
            # ".opt.onnx" is the longest suffix appended to engine_path (see builder.py).
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
            engine_build_options=self._get_default_controlnet_build_options(
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                builder_optimization_level=builder_optimization_level,
                fp8=fp8,
            ),
        )
