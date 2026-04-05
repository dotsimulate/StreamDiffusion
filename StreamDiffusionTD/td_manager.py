"""
TouchDesigner StreamDiffusion Manager

Core bridge between TouchDesigner and the LivePeer StreamDiffusion fork.
Handles configuration, streaming loop, and parameter updates.
"""

import logging
import os
import platform
import sys
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

# Logger will be configured by td_main.py based on debug_mode
logger = logging.getLogger("TouchDesignerManager")

# Add StreamDiffusion to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from streamdiffusion.config import create_wrapper_from_config, load_config

from streamdiffusion import StreamDiffusionWrapper


class TouchDesignerManager:
    """
    Main manager class that bridges TouchDesigner with StreamDiffusion fork.

    Key differences from your old version:
    1. Uses new fork's config system and unified parameter updates
    2. Maintains same SharedMemory/Syphon interface patterns
    3. Leverages new fork's caching and performance improvements
    """

    def __init__(
        self,
        config: Union[str, Dict[str, Any]],
        input_mem_name: str,
        output_mem_name: str,
        debug_mode: bool = False,
        osc_reporter=None,
    ):
        self.input_mem_name = input_mem_name
        self.output_mem_name = output_mem_name
        self.osc_reporter = osc_reporter  # Lightweight telemetry reporter
        self.osc_handler = None  # Parameter handler (set later)
        self.debug_mode = debug_mode

        # Handle both config dict (new) and config path (legacy compatibility)
        if isinstance(config, dict):
            if debug_mode:
                print("Using pre-merged configuration dictionary")
            self.config = config
            self.config_path = None
        else:
            # Legacy: Load configuration using new fork's config system
            if debug_mode:
                print(f"Loading configuration from: {config}")
            self.config = load_config(config)
            self.config_path = config

        # Extract TD-specific settings
        self.td_settings = self.config.get("td_settings", {})

        # Platform detection (same as your current version)
        self.is_macos = platform.system() == "Darwin"
        self.stream_method = "syphon" if self.is_macos else "shared_mem"

        # Track which seed indices should be randomized every frame
        self._randomize_seed_indices = []

        # Initialize StreamDiffusion wrapper using new fork
        logger.debug("Creating StreamDiffusion wrapper...")
        self.wrapper: StreamDiffusionWrapper = create_wrapper_from_config(self.config)

        # Memory interfaces (will be initialized in start_streaming)
        self.input_memory: Optional[shared_memory.SharedMemory] = None
        self.output_memory: Optional[shared_memory.SharedMemory] = None
        self.control_memory: Optional[shared_memory.SharedMemory] = None
        self.control_processed_memory: Optional[shared_memory.SharedMemory] = (
            None  # For pre-processed ControlNet output
        )
        self.ipadapter_memory: Optional[shared_memory.SharedMemory] = None
        self.syphon_handler = None

        # Streaming state
        self.streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.paused = False
        self.process_frame = False
        self.frame_acknowledged = False

        # State tracking for logging (only log changes)
        self._last_paused_state = False
        self._frames_processed_in_pause = 0

        # ControlNet and IPAdapter state
        self.ipadapter_update_requested = False
        self.control_mem_name = self.input_mem_name + "-cn"
        self.control_processed_mem_name = (
            self.input_mem_name + "-cn-processed"
        )  # For pre-processed ControlNet output
        self.ipadapter_mem_name = self.input_mem_name + "-ip"

        # Track live IPAdapter scale from OSC updates
        self._current_ipadapter_scale = self.config.get("ipadapters", [{}])[0].get(
            "scale", 1.0
        )

        # Performance tracking
        self.frame_count = 0
        self.total_frame_count = 0  # Total frames processed (for OSC)
        self.start_time = time.time()
        self.fps_smoothing = 0.9  # For exponential moving average
        self.current_fps = 0.0          # Output FPS: includes cached/skipped frames
        self.inference_fps = 0.0        # Inference FPS: only frames that ran GPU inference
        self.last_frame_output_time = 0.0   # Wall-clock time of last frame output (any frame)
        self.last_inference_time = 0.0      # Wall-clock time of last real inference frame

        # OSC notification flags
        self._sent_processed_cn_name = False

        # Async GPU→CPU output transfer (eliminates full GPU sync on every output frame)
        self._pinned_output_buf: Optional[torch.Tensor] = None
        self._output_copy_event: Optional[torch.cuda.Event] = None

        # Mode tracking (img2img or txt2img)
        self.mode = self.config.get("mode", "img2img")
        logger.info(f"Initialized in {self.mode} mode")

        logger.info("TouchDesigner Manager initialized successfully")

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update StreamDiffusion parameters using new fork's unified system.

        This replaces the scattered parameter updates in your old version
        with a single, atomic update call that handles caching efficiently.
        """
        try:
            # Track IPAdapter scale changes from OSC
            if "ipadapter_config" in params and "scale" in params["ipadapter_config"]:
                self._current_ipadapter_scale = params["ipadapter_config"]["scale"]
                # Removed noisy log: logger.info(f"Updated IPAdapter scale to: {self._current_ipadapter_scale}")

            # Filter out invalid parameters that wrapper doesn't accept
            valid_params = [
                "num_inference_steps",
                "guidance_scale",
                "delta",
                "t_index_list",
                "seed",
                "prompt_list",
                "negative_prompt",
                "prompt_interpolation_method",
                "normalize_prompt_weights",
                "seed_list",
                "seed_interpolation_method",
                "normalize_seed_weights",
                "controlnet_config",
                "ipadapter_config",
                "image_preprocessing_config",
                "image_postprocessing_config",
                "latent_preprocessing_config",
                "latent_postprocessing_config",
                "use_safety_checker",
                "safety_checker_threshold",
                "cache_maxframes",
                "cache_interval",
            ]

            filtered_params = {k: v for k, v in params.items() if k in valid_params}

            # Enforce guidance_scale > 1.0 for cfg_type "full" or "initialize"
            # (pipeline.py requires this for negative prompt embeddings)
            if "guidance_scale" in filtered_params:
                cfg_type = getattr(self.wrapper.stream, "cfg_type", None)
                if (
                    cfg_type in ["full", "initialize"]
                    and filtered_params["guidance_scale"] <= 1.0
                ):
                    filtered_params["guidance_scale"] = 1.2

            # Track which seed indices have -1 for continuous per-frame randomization
            if "seed_list" in filtered_params:
                import random

                # Clear and rebuild tracking based on this OSC update
                self._randomize_seed_indices = []
                new_seed_list = []
                for idx, (seed, weight) in enumerate(filtered_params["seed_list"]):
                    if seed == -1:
                        # Mark this index for continuous randomization
                        self._randomize_seed_indices.append(idx)
                        # Generate random seed for this initial update
                        seed = random.randint(0, 2**32 - 1)
                    new_seed_list.append((seed, weight))
                filtered_params["seed_list"] = new_seed_list

            # Use the new fork's unified parameter update system
            self.wrapper.update_stream_params(**filtered_params)
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")

    def start_streaming(self) -> None:
        """Initialize memory interfaces and start the streaming loop"""
        if self.streaming:
            logger.warning("Already streaming!")
            return

        try:
            # Send starting state via reporter (not handler - that's for parameters)
            if self.osc_reporter:
                self.osc_reporter.set_state("local_starting")

            self._initialize_memory_interfaces()
            self.streaming = True

            # Start streaming in separate thread (non-blocking)
            self.stream_thread = threading.Thread(
                target=self._streaming_loop, daemon=True
            )
            self.stream_thread.start()

            # Print styled startup message
            print("\n\033[32mStream active\033[0m\n")

            logger.info(f"Streaming started - Method: {self.stream_method}")

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            raise

    def stop_streaming(self) -> None:
        """Stop streaming and cleanup resources"""
        if not self.streaming:
            return

        logger.info("Stopping streaming...")
        self.streaming = False

        # Send offline state via reporter
        if self.osc_reporter:
            self.osc_reporter.set_state("local_offline")

        # Wait for streaming thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)

        # Cleanup memory interfaces
        self._cleanup_memory_interfaces()

        logger.info("Streaming stopped")

    def _initialize_memory_interfaces(self) -> None:
        """Initialize platform-specific memory interfaces"""
        width = self.config["width"]
        height = self.config["height"]

        if self.is_macos:
            # Initialize Syphon (using existing syphon_utils.py)
            try:
                from syphon_utils import SyphonUtils

                self.syphon_handler = SyphonUtils(
                    sender_name=self.output_mem_name,
                    input_name=self.input_mem_name,
                    control_name=None,
                    width=width,
                    height=height,
                    debug=False,
                )
                self.syphon_handler.start()
                logger.debug(
                    f"Syphon initialized - Input: {self.input_mem_name}, Output: {self.output_mem_name}"
                )

            except Exception as e:
                logger.error(f"Failed to initialize Syphon: {e}")
                raise
        else:
            # Initialize SharedMemory (same pattern as your current version)
            try:
                # Input memory (from TouchDesigner)
                self.input_memory = shared_memory.SharedMemory(name=self.input_mem_name)
                logger.debug(f"Connected to input SharedMemory: {self.input_mem_name}")

                # Output memory (to TouchDesigner) - try to connect first, create if not exists
                frame_size = width * height * 3  # RGB
                try:
                    # Try to connect to existing memory first (like main_sdtd.py)
                    self.output_memory = shared_memory.SharedMemory(
                        name=self.output_mem_name
                    )
                    logger.debug(
                        f"Connected to existing output SharedMemory: {self.output_mem_name}"
                    )
                except FileNotFoundError:
                    # Create new if doesn't exist
                    self.output_memory = shared_memory.SharedMemory(
                        name=self.output_mem_name, create=True, size=frame_size
                    )
                    logger.debug(
                        f"Created new output SharedMemory: {self.output_mem_name}"
                    )

                # ControlNet memory (per-frame updates)
                try:
                    self.control_memory = shared_memory.SharedMemory(
                        name=self.control_mem_name
                    )
                    logger.debug(
                        f"Connected to ControlNet SharedMemory: {self.control_mem_name}"
                    )
                except FileNotFoundError:
                    logger.debug(
                        f"ControlNet SharedMemory not found: {self.control_mem_name} (will create if needed)"
                    )
                    self.control_memory = None

                # ControlNet processed output memory (to send pre-processed image back to TD)
                # Create output memory for processed ControlNet with -cn-processed suffix
                control_processed_mem_name = self.output_mem_name + "-cn-processed"
                try:
                    # Try to connect to existing memory first
                    self.control_processed_memory = shared_memory.SharedMemory(
                        name=control_processed_mem_name
                    )
                    logger.debug(
                        f"Connected to existing ControlNet processed output SharedMemory: {control_processed_mem_name}"
                    )
                except FileNotFoundError:
                    # Create new if doesn't exist
                    self.control_processed_memory = shared_memory.SharedMemory(
                        name=control_processed_mem_name,
                        create=True,
                        size=frame_size,  # Same size as main output
                    )
                    logger.debug(
                        f"Created new ControlNet processed output SharedMemory: {control_processed_mem_name}"
                    )

                # IPAdapter memory (OSC-triggered updates)
                try:
                    self.ipadapter_memory = shared_memory.SharedMemory(
                        name=self.ipadapter_mem_name
                    )
                    logger.debug(
                        f"Connected to IPAdapter SharedMemory: {self.ipadapter_mem_name}"
                    )
                except FileNotFoundError:
                    logger.debug(
                        f"IPAdapter SharedMemory not found: {self.ipadapter_mem_name} (will create if needed)"
                    )
                    self.ipadapter_memory = None

            except Exception as e:
                logger.error(f"Failed to initialize SharedMemory: {e}")
                raise

    def _cleanup_memory_interfaces(self) -> None:
        """Cleanup memory interfaces"""
        try:
            if self.syphon_handler:
                self.syphon_handler.stop()
                self.syphon_handler = None

            if self.input_memory:
                self.input_memory.close()
                self.input_memory = None

            if self.output_memory:
                self.output_memory.close()
                self.output_memory.unlink()  # Delete the shared memory
                self.output_memory = None

            if self.control_memory:
                self.control_memory.close()
                self.control_memory = None

            if self.control_processed_memory:
                self.control_processed_memory.close()
                self.control_processed_memory.unlink()  # Delete the shared memory
                self.control_processed_memory = None

            if self.ipadapter_memory:
                self.ipadapter_memory.close()
                self.ipadapter_memory = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _streaming_loop(self) -> None:
        """
        Main streaming loop - processes frames as fast as possible.

        Key insight: The wrapper.img2img() call runs as fast as it can.
        You don't manually control the speed - the wrapper handles timing internally.
        """
        logger.info("Starting streaming loop...")

        frame_time_accumulator = 0.0
        fps_report_interval = 1.0  # Report FPS every second

        while self.streaming:
            try:
                # CRITICAL: Check pause state FIRST before any processing
                if self.paused:
                    if not self.process_frame:
                        time.sleep(
                            0.001
                        )  # Brief sleep if paused and no frame requested
                        continue
                    else:
                        # Reset acknowledgment flag when starting frame processing
                        self.frame_acknowledged = False
                        self._frames_processed_in_pause += 1

                        # Log useful info every 10th frame
                        if self._frames_processed_in_pause % 10 == 0:
                            logger.info(
                                f"PAUSE: Frame {self._frames_processed_in_pause} | FPS: {self.current_fps:.1f}"
                            )

                        # CRITICAL: Reset process_frame IMMEDIATELY after confirming we're processing
                        # This must happen INSIDE the pause block, not outside
                        self.process_frame = False
                else:
                    # In continuous mode, we don't need to reset process_frame
                    pass

                loop_start = time.time()

                # Process through StreamDiffusion - mode determines which method to call
                if self.mode == "txt2img":
                    # txt2img mode: generate from prepared noise (respects seed)
                    # Use init_noise instead of random noise to respect seed parameter
                    stream = self.wrapper.stream
                    batch_size = 1

                    # Check if seed is -1 (randomize every frame)
                    # Get current seed from config (updated via OSC)
                    current_seed = self.config.get("seed", -1)

                    if current_seed == -1:
                        # Generate fresh random noise every frame (random seed)
                        adjusted_noise = torch.randn(
                            (batch_size, 4, stream.latent_height, stream.latent_width),
                            device=stream.device,
                            dtype=stream.dtype,
                        )
                    else:
                        # Generate noise from seed (consistent image every frame)
                        generator = torch.Generator(device=stream.device)
                        generator.manual_seed(current_seed)
                        adjusted_noise = torch.randn(
                            (batch_size, 4, stream.latent_height, stream.latent_width),
                            device=stream.device,
                            dtype=stream.dtype,
                            generator=generator,
                        )

                    # Apply latent preprocessing hooks before prediction
                    adjusted_noise = stream._apply_latent_preprocessing_hooks(
                        adjusted_noise
                    )

                    # Call predict_x0_batch with prepared noise
                    x_0_pred_out = stream.predict_x0_batch(adjusted_noise)

                    # Apply latent postprocessing hooks
                    x_0_pred_out = stream._apply_latent_postprocessing_hooks(
                        x_0_pred_out
                    )

                    # Store for latent feedback processors
                    stream.prev_latent_result = x_0_pred_out.detach().clone()

                    # Decode to image
                    x_output = stream.decode_image(x_0_pred_out).detach().clone()

                    # Apply image postprocessing hooks
                    x_output = stream._apply_image_postprocessing_hooks(x_output)

                    # Postprocess to desired output format
                    output_image = self.wrapper.postprocess_image(
                        x_output, output_type=self.wrapper.output_type
                    )
                else:
                    # img2img mode: get input frame and process
                    input_image = self._get_input_frame()
                    if input_image is None:
                        time.sleep(0.001)  # Brief sleep if no input
                        continue

                    # Convert input for pipeline
                    if input_image.dtype == np.uint8:
                        input_image = input_image.astype(np.float32) / 255.0

                    # Process ControlNet frame (per-frame if enabled)
                    self._process_controlnet_frame()

                    # Process IPAdapter frame (only on OSC request)
                    self._process_ipadapter_frame()

                    # Randomize tracked seed indices every frame (indices marked as -1 via OSC)
                    if self._randomize_seed_indices:
                        import random

                        # Make a copy of current seed_list and randomize tracked indices
                        current_seed_list = (
                            self.wrapper.stream._param_updater._current_seed_list.copy()
                        )
                        for idx in self._randomize_seed_indices:
                            _, weight = current_seed_list[idx]
                            current_seed_list[idx] = (
                                random.randint(0, 2**32 - 1),
                                weight,
                            )
                        # Apply the randomized seed_list to the stream
                        self.wrapper.update_stream_params(seed_list=current_seed_list)

                    # Transform input image
                    output_image = self.wrapper.img2img(input_image)

                # Send output frame to TouchDesigner
                self._send_output_frame(output_image)

                # Output FPS: wall-clock rate of all frames sent to TD (includes cached skips)
                frame_output_time = time.time()
                if self.last_frame_output_time > 0:
                    frame_interval = frame_output_time - self.last_frame_output_time
                    instantaneous_fps = (
                        1.0 / frame_interval if frame_interval > 0 else 0.0
                    )
                    self.current_fps = (
                        self.current_fps * self.fps_smoothing
                        + instantaneous_fps * (1 - self.fps_smoothing)
                    )
                self.last_frame_output_time = frame_output_time

                # Inference FPS: only frames that actually ran GPU inference (similar filter skips excluded)
                was_skipped = getattr(self.wrapper.stream, 'last_frame_was_skipped', False)
                if not was_skipped:
                    if self.last_inference_time > 0:
                        inf_interval = frame_output_time - self.last_inference_time
                        instantaneous_inf_fps = (
                            1.0 / inf_interval if inf_interval > 0 else 0.0
                        )
                        self.inference_fps = (
                            self.inference_fps * self.fps_smoothing
                            + instantaneous_inf_fps * (1 - self.fps_smoothing)
                        )
                    self.last_inference_time = frame_output_time

                # Update frame counters
                self.frame_count += 1
                self.total_frame_count += 1

                # Update performance metrics for display
                loop_end = time.time()
                loop_time = loop_end - loop_start
                frame_time_accumulator += loop_time

                # Send OSC messages EVERY FRAME via reporter (like main_sdtd.py - critical for TD connection!)
                if self.osc_reporter:
                    self.osc_reporter.send_frame_count(self.total_frame_count)
                    # Only send frame_ready in continuous mode, NOT in pause mode
                    if not self.paused:
                        self.osc_reporter.send_frame_ready(self.total_frame_count)

                    # Send connection state update (transition to streaming after 30 frames)
                    if self.total_frame_count == 30:
                        self.osc_reporter.set_state("local_streaming")

                    # Send processed ControlNet memory name (only once at startup or when needed)
                    if (
                        not self._sent_processed_cn_name
                        and self.control_processed_memory
                    ):
                        processed_cn_name = self.output_mem_name + "-cn-processed"
                        self.osc_reporter.send_controlnet_processed_name(
                            processed_cn_name
                        )
                        self._sent_processed_cn_name = True

                # Report FPS periodically using the correct frame output FPS (ALWAYS show)
                if frame_time_accumulator >= fps_report_interval:
                    # Calculate uptime
                    uptime_seconds = int(time.time() - self.start_time)
                    uptime_mins = uptime_seconds // 60
                    uptime_secs = uptime_seconds % 60
                    uptime_str = f"{uptime_mins:02d}:{uptime_secs:02d}"

                    # Clear the line properly and write new status with color
                    status_line = f"\033[38;5;208mStreaming | FPS: {self.inference_fps:.1f} (out: {self.current_fps:.1f}) | Uptime: {uptime_str}\033[0m"
                    print(f"\r{' ' * 80}\r{status_line}", end="", flush=True)

                    # Reset counters
                    frame_time_accumulator = 0.0
                    self.frame_count = 0

                # Send FPS EVERY FRAME via reporter — use inference FPS (excludes similar-filter skips)
                if self.osc_reporter:
                    reported_fps = self.inference_fps if self.inference_fps > 0 else self.current_fps
                    if reported_fps > 0:
                        self.osc_reporter.send_fps(reported_fps)

                # Handle frame acknowledgment for pause mode synchronization
                if self.paused:
                    # Send frame completion signal to TouchDesigner
                    if self.osc_handler:
                        self.osc_handler.send_message(
                            "/frame_ready", self.total_frame_count
                        )

                    # Wait for TouchDesigner to acknowledge it has processed the frame
                    acknowledgment_timeout = time.time() + 5.0  # 5 second timeout
                    while (
                        not self.frame_acknowledged
                        and time.time() < acknowledgment_timeout
                    ):
                        time.sleep(0.001)  # Small sleep to prevent busy waiting

                    if not self.frame_acknowledged:
                        logger.warning(
                            f"TIMEOUT: No /frame_ack after 5s (frame {self._frames_processed_in_pause})"
                        )

                    # Reset for next frame
                    self.frame_acknowledged = False

            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                # Continue streaming unless explicitly stopped
                if self.streaming:
                    time.sleep(0.1)  # Brief pause before retrying
                else:
                    break

        logger.info("Streaming loop ended")

    def _get_input_frame(self) -> Optional[np.ndarray]:
        """Get input frame from TouchDesigner (platform-specific)"""
        try:
            if self.is_macos and self.syphon_handler:
                # Get frame from Syphon
                frame = self.syphon_handler.capture_input_frame()
                return frame

            elif self.input_memory:
                # Get frame from SharedMemory
                width = self.config["width"]
                height = self.config["height"]

                # Create numpy array view of shared memory
                frame = np.ndarray(
                    (height, width, 3), dtype=np.uint8, buffer=self.input_memory.buf
                )

                return frame.copy()  # Copy to avoid memory issues

            return None

        except Exception as e:
            logger.error(f"Error getting input frame: {e}")
            return None

    def _send_output_frame(
        self, output_image: Union[Image.Image, np.ndarray]
    ) -> None:
        """Send output frame to TouchDesigner (platform-specific)"""
        try:
            # Convert output to numpy array if needed
            if isinstance(output_image, Image.Image):
                frame_np = np.array(output_image)
            elif isinstance(output_image, torch.Tensor):
                if output_image.is_cuda:
                    # Async GPU→CPU via pinned memory: eliminates full stream sync (saves 1-3ms/frame)
                    if (
                        self._pinned_output_buf is None
                        or self._pinned_output_buf.shape != output_image.shape
                        or self._pinned_output_buf.dtype != output_image.dtype
                    ):
                        self._pinned_output_buf = torch.empty(
                            output_image.shape, dtype=output_image.dtype, pin_memory=True
                        )
                        self._output_copy_event = torch.cuda.Event()
                    self._pinned_output_buf.copy_(output_image, non_blocking=True)
                    self._output_copy_event.record()
                    self._output_copy_event.synchronize()  # wait only for this transfer
                    frame_np = self._pinned_output_buf.numpy()
                else:
                    frame_np = output_image.numpy()
                if frame_np.shape[0] == 3:  # CHW -> HWC
                    frame_np = np.transpose(frame_np, (1, 2, 0))
            else:
                frame_np = output_image

            # Ensure proper scaling like main_sdtd.py (line 916)
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            elif frame_np.dtype != np.uint8:
                frame_np = frame_np.astype(np.uint8)

            if self.is_macos and self.syphon_handler:
                # Send via Syphon
                self.syphon_handler.send_frame(frame_np)

            elif self.output_memory:
                # Send via SharedMemory
                output_array = np.ndarray(
                    frame_np.shape, dtype=np.uint8, buffer=self.output_memory.buf
                )
                np.copyto(output_array, frame_np)

                # Diagnostic: confirm first frame write
                if self.total_frame_count == 0:
                    print(
                        f"[SharedMem] First frame written: shape={frame_np.shape}, "
                        f"dtype={frame_np.dtype}, mem_name={self.output_mem_name}, "
                        f"buf_size={self.output_memory.size}"
                    )

        except Exception as e:
            logger.error(f"Error sending output frame: {e}")
            import traceback; traceback.print_exc()  # TEMP diagnostic

    def _send_processed_controlnet_frame(
        self, processed_tensor: Optional[torch.Tensor]
    ) -> None:
        """Send processed ControlNet frame to TouchDesigner via shared memory"""
        if not self.control_processed_memory or processed_tensor is None:
            return

        try:
            # Convert tensor to numpy array (similar to _send_output_frame)
            if isinstance(processed_tensor, torch.Tensor):
                frame_np = processed_tensor.cpu().numpy()

                # Handle tensor format: CHW -> HWC
                if frame_np.ndim == 4 and frame_np.shape[0] == 1:
                    frame_np = frame_np.squeeze(0)  # Remove batch dimension
                if frame_np.ndim == 3 and frame_np.shape[0] == 3:
                    frame_np = np.transpose(frame_np, (1, 2, 0))  # CHW -> HWC
            else:
                frame_np = processed_tensor

            # Ensure proper scaling (0-1 range to 0-255)
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            elif frame_np.dtype != np.uint8:
                frame_np = frame_np.astype(np.uint8)

            # Send via SharedMemory
            if self.control_processed_memory:
                output_array = np.ndarray(
                    frame_np.shape,
                    dtype=np.uint8,
                    buffer=self.control_processed_memory.buf,
                )
                np.copyto(output_array, frame_np)

        except Exception as e:
            logger.error(f"Error sending processed ControlNet frame: {e}")

    def pause_streaming(self) -> None:
        """Pause streaming (frame processing continues only on process_frame requests)"""
        if not self.paused:  # Only log state changes
            self.paused = True
            self._frames_processed_in_pause = 0  # Reset counter

            logger.info("STREAMING PAUSED - waiting for /process_frame commands")

    def resume_streaming(self) -> None:
        """Resume streaming (continuous frame processing)"""
        if self.paused:  # Only log state changes
            self.paused = False
            logger.info(
                f"STREAMING RESUMED - processed {self._frames_processed_in_pause} frames while paused"
            )

    def process_single_frame(self) -> None:
        """Process a single frame when paused"""
        if self.paused:
            self.process_frame = True
            # Only log occasionally to reduce spam
        else:
            logger.warning("process_single_frame called but streaming is not paused")

    def acknowledge_frame(self) -> None:
        """Acknowledge frame processing completion (called by TouchDesigner)"""
        self.frame_acknowledged = True

    def _process_controlnet_frame(self) -> None:
        """Process ControlNet frame data (per-frame updates)"""
        if not self.control_memory or not self.config.get("use_controlnet", False):
            return

        try:
            width = self.config["width"]
            height = self.config["height"]

            # Read ControlNet frame from shared memory
            control_frame = np.ndarray(
                (height, width, 3), dtype=np.uint8, buffer=self.control_memory.buf
            )

            # Convert to float [0,1] and pass to wrapper
            if control_frame.dtype == np.uint8:
                control_frame = control_frame.astype(np.float32) / 255.0

            # Update ControlNet image in wrapper (index 0 for first ControlNet)
            self.wrapper.update_control_image(0, control_frame)

            # IMPORTANT: After processing, extract the pre-processed image and send it back to TD
            # The processed image is now available in the controlnet module
            try:
                if (
                    hasattr(self.wrapper, "stream")
                    and hasattr(self.wrapper.stream, "_controlnet_module")
                    and self.wrapper.stream._controlnet_module is not None
                ):
                    controlnet_module = self.wrapper.stream._controlnet_module

                    # Check if we have processed images available
                    if (
                        hasattr(controlnet_module, "controlnet_images")
                        and len(controlnet_module.controlnet_images) > 0
                        and controlnet_module.controlnet_images[0] is not None
                    ):
                        processed_tensor = controlnet_module.controlnet_images[0]

                        # Send the processed ControlNet image back to TouchDesigner
                        self._send_processed_controlnet_frame(processed_tensor)

            except Exception as processed_error:
                logger.debug(
                    f"Could not extract processed ControlNet image: {processed_error}"
                )

        except Exception as e:
            logger.error(f"Error processing ControlNet frame: {e}")

    def _process_ipadapter_frame(self) -> None:
        """Process IPAdapter frame data (OSC-triggered updates only)"""
        if not self.config.get("use_ipadapter", False):
            return
        if not self.ipadapter_memory:
            logger.debug(
                f"IPAdapter SharedMemory not connected: {self.ipadapter_mem_name}"
            )
            return

        if not self.ipadapter_update_requested:
            return

        try:
            width = self.config["width"]
            height = self.config["height"]

            # Read IPAdapter frame from shared memory
            ipadapter_frame = np.ndarray(
                (height, width, 3), dtype=np.uint8, buffer=self.ipadapter_memory.buf
            )

            # Convert to float [0,1] and pass to wrapper
            if ipadapter_frame.dtype == np.uint8:
                ipadapter_frame = ipadapter_frame.astype(np.float32) / 255.0

            # Update IPAdapter config FIRST (like img2img example)
            # Use live scale value updated by OSC, not static config
            current_scale = self._current_ipadapter_scale
            logger.info(f"Using live IPAdapter scale: {current_scale}")
            self.wrapper.update_stream_params(ipadapter_config={"scale": current_scale})

            # THEN update the style image (following img2img pattern)
            self.wrapper.update_style_image(ipadapter_frame)

            # CRITICAL FIX: Trigger prompt re-blending to apply the new IPAdapter embeddings
            # The embedding hooks only run during _apply_prompt_blending, so we must trigger it
            # after updating the IPAdapter image to concatenate the new embeddings with prompts
            if (
                hasattr(self.wrapper.stream._param_updater, "_current_prompt_list")
                and self.wrapper.stream._param_updater._current_prompt_list
            ):
                # Get current interpolation method or use default
                interpolation_method = getattr(
                    self.wrapper.stream._param_updater,
                    "_last_prompt_interpolation_method",
                    "slerp",
                )
                # Re-apply prompt blending to trigger embedding hooks
                self.wrapper.stream._param_updater._apply_prompt_blending(
                    interpolation_method
                )
                logger.info(
                    "Re-applied prompt blending to incorporate new IPAdapter embeddings"
                )

            # Debug: Check if embeddings were cached
            cached_embeddings = (
                self.wrapper.stream._param_updater.get_cached_embeddings(
                    "ipadapter_main"
                )
            )
            if cached_embeddings is not None:
                logger.info(
                    f"IPAdapter config+image updated! Scale: {current_scale}, Embeddings: {cached_embeddings[0].shape}"
                )
            else:
                logger.error("IPAdapter embeddings NOT cached!")

            # Reset the update flag
            self.ipadapter_update_requested = False
            logger.info("IPAdapter config and image updated from SharedMemory")

        except Exception as e:
            logger.error(f"Error processing IPAdapter frame: {e}")

    def request_ipadapter_update(self) -> None:
        """Request IPAdapter image update on next frame (called via OSC)"""
        self.ipadapter_update_requested = True
        logger.info("IPAdapter update requested")

    def set_mode(self, mode: str) -> None:
        """
        Switch between txt2img and img2img modes.

        Args:
            mode: Either "txt2img" or "img2img"
        """
        if mode not in ["txt2img", "img2img"]:
            logger.error(f"Invalid mode: {mode}. Must be 'txt2img' or 'img2img'")
            return

        if mode == self.mode:
            logger.info(f"Already in {mode} mode")
            return

        logger.info(f"Switching from {self.mode} to {mode} mode")
        self.mode = mode

        # txt2img mode requirements
        if mode == "txt2img":
            logger.warning(
                "txt2img mode requires cfg_type='none' - verify your config!"
            )

    def get_stream_state(self) -> Dict[str, Any]:
        """Get current streaming state and parameters"""
        return {
            "streaming": self.streaming,
            "paused": self.paused,
            "fps": self.current_fps,
            "frame_count": self.frame_count,
            "stream_method": self.stream_method,
            "wrapper_state": self.wrapper.get_stream_state()
            if hasattr(self.wrapper, "get_stream_state")
            else {},
            "controlnet_connected": self.control_memory is not None,
            "controlnet_processed_connected": self.control_processed_memory is not None,
            "ipadapter_connected": self.ipadapter_memory is not None,
        }
