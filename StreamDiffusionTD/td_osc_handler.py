"""
OSC Parameter Handler for TouchDesigner StreamDiffusion

Maps OSC addresses to the new fork's unified parameter system.
Maintains compatibility with existing TouchDesigner OSC patterns.
"""

import json
import logging
import threading
import time
from typing import Any, Dict

from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# Logger will be configured by td_main.py based on debug_mode
logger = logging.getLogger("OSCHandler")


class OSCParameterHandler:
    """
    Handles OSC communication with TouchDesigner and maps parameters
    to the new StreamDiffusion fork's unified parameter system.
    """

    def __init__(
        self,
        manager,
        main_app=None,
        listen_port: int = 8247,
        transmit_port: int = 8248,
        transmit_ip: str = "127.0.0.1",
        debug_mode: bool = False,
    ):
        self.manager = manager
        self.main_app = main_app  # Reference to main application for shutdown
        self.listen_port = listen_port
        self.transmit_port = transmit_port
        self.transmit_ip = transmit_ip
        self.debug_mode = debug_mode

        # OSC communication
        self.server = None
        self.client = udp_client.SimpleUDPClient(transmit_ip, transmit_port)
        self.server_thread = None
        self.running = False

        # Parameter batching for efficiency
        self.parameter_batch: Dict[str, Any] = {}
        self.batch_lock = threading.Lock()
        self.last_batch_time = time.time()
        self.batch_interval = 0.016  # ~60Hz parameter updates

        # OSC dispatcher setup
        self.dispatcher = Dispatcher()
        self._setup_osc_handlers()

        logger.info(
            f"OSC Handler initialized - Listen: {listen_port}, Transmit: {transmit_port}"
        )

    def start(self) -> None:
        """Start OSC server in separate thread"""
        if self.running:
            return

        self.running = True

        try:
            self.server = BlockingOSCUDPServer(
                (self.transmit_ip, self.listen_port), self.dispatcher
            )
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()

            # Start parameter batch processing
            batch_thread = threading.Thread(
                target=self._parameter_batch_loop, daemon=True
            )
            batch_thread.start()

            logger.info(f"OSC server started on {self.transmit_ip}:{self.listen_port}")

        except Exception as e:
            logger.error(f"Failed to start OSC server: {e}")
            self.running = False
            raise

    def stop(self) -> None:
        """Stop OSC server"""
        if not self.running:
            return

        logger.info("Stopping OSC server...")
        self.running = False

        if self.server:
            self.server.shutdown()

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)

        logger.info("OSC server stopped")

    def send_message(self, address: str, value: Any) -> None:
        """Send OSC message to TouchDesigner"""
        try:
            self.client.send_message(address, value)
        except Exception as e:
            logger.error(f"Error sending OSC message {address}: {e}")

    def _server_loop(self) -> None:
        """OSC server loop"""
        try:
            self.server.serve_forever()
        except Exception as e:
            if self.running:
                logger.error(f"OSC server error: {e}")

    def _parameter_batch_loop(self) -> None:
        """
        Process parameter updates in batches for efficiency.
        This prevents overwhelming the StreamDiffusion wrapper with rapid updates.
        """
        while self.running:
            try:
                current_time = time.time()

                if current_time - self.last_batch_time >= self.batch_interval:
                    with self.batch_lock:
                        if self.parameter_batch:
                            # Send batched parameters to manager
                            self.manager.update_parameters(self.parameter_batch.copy())
                            self.parameter_batch.clear()
                            self.last_batch_time = current_time

                time.sleep(0.001)  # Small sleep to prevent busy waiting

            except Exception as e:
                logger.error(f"Error in parameter batch loop: {e}")
                time.sleep(0.1)

    def _queue_parameter_update(self, param_name: str, value: Any) -> None:
        """Queue a parameter for batched update"""
        with self.batch_lock:
            self.parameter_batch[param_name] = value

    def _setup_osc_handlers(self) -> None:
        """Setup OSC message handlers mapping to new fork's parameter system"""

        # === Core Generation Parameters ===
        self.dispatcher.map(
            "/guidance_scale",
            lambda addr, *args: self._queue_parameter_update(
                "guidance_scale", float(args[0])
            ),
        )

        self.dispatcher.map(
            "/delta",
            lambda addr, *args: self._queue_parameter_update("delta", float(args[0])),
        )

        self.dispatcher.map(
            "/num_inference_steps",
            lambda addr, *args: self._queue_parameter_update(
                "num_inference_steps", int(args[0])
            ),
        )

        self.dispatcher.map(
            "/seed",
            lambda addr, *args: self._queue_parameter_update("seed", int(args[0])),
        )

        # === T-Index List ===
        self.dispatcher.map(
            "/t_list",
            lambda addr, *args: self._queue_parameter_update(
                "t_index_list", list(args)
            ),
        )

        # === Prompt Handling (NEW: supports both single and blending) ===
        self.dispatcher.map("/prompt", self._handle_single_prompt)
        self.dispatcher.map(
            "/negative_prompt",
            lambda addr, *args: self._queue_parameter_update(
                "negative_prompt", str(args[0])
            ),
        )

        # NEW: Prompt blending support
        self.dispatcher.map("/prompt_list", self._handle_prompt_list)
        self.dispatcher.map(
            "/prompt_interpolation_method",
            lambda addr, *args: self._queue_parameter_update(
                "prompt_interpolation_method", str(args[0])
            ),
        )

        # === Seed Blending (NEW) ===
        self.dispatcher.map("/seed_list", self._handle_seed_list)
        self.dispatcher.map(
            "/seed_interpolation_method",
            lambda addr, *args: self._queue_parameter_update(
                "seed_interpolation_method", str(args[0])
            ),
        )

        # === ControlNet Support (NEW: Multi-ControlNet) ===
        self.dispatcher.map("/controlnets", self._handle_controlnet_config)

        # Only keep use_controlnet for enabling/disabling (sets conditioning scale to 0)
        self.dispatcher.map("/use_controlnet", self._handle_controlnet_enable)

        # === IPAdapter Support (NEW) ===
        self.dispatcher.map("/ipadapter_config", self._handle_ipadapter_config)
        self.dispatcher.map(
            "/ipadapter_scale",
            lambda addr, *args: self._queue_parameter_update(
                "ipadapter_config", {"scale": float(args[0])}
            ),
        )
        self.dispatcher.map("/ipadapter_enable", self._handle_ipadapter_enable)
        self.dispatcher.map("/ipadapter_update", self._handle_ipadapter_update)

        # === Cached Attention / StreamV2V (live-updateable) ===
        self.dispatcher.map("/cached_attention", self._handle_cached_attention_config)

        # === Mode Switching (txt2img/img2img) ===
        self.dispatcher.map("/sdmode", self._handle_mode_switch)

        # === Latent Preprocessing (Latent Feedback) ===
        self.dispatcher.map(
            "/latent_feedback_strength", self._handle_latent_feedback_strength
        )

        # === FX Dynamic Parameters (Latent-domain processors) ===
        # Generic handler for all FX processor parameters
        self.dispatcher.map("/fx/*", self._handle_fx_parameter)

        # === System Commands ===
        self.dispatcher.map("/start_streaming", self._handle_start_streaming)
        self.dispatcher.map("/stop_streaming", self._handle_stop_streaming)
        self.dispatcher.map(
            "/stop", self._handle_stop_application
        )  # Main application stop (like main_sdtd.py)
        self.dispatcher.map("/pause", self._handle_pause_streaming)
        self.dispatcher.map("/play", self._handle_resume_streaming)
        self.dispatcher.map("/process_frame", self._handle_process_frame)
        self.dispatcher.map("/frame_ack", self._handle_frame_acknowledgment)
        self.dispatcher.map("/get_status", self._handle_get_status)

        # === Heartbeat removed - now handled by OSCReporter in td_main.py ===
        # This handler now focuses solely on parameter updates

    # === Handler Methods ===

    def _handle_single_prompt(self, address, *args):
        """Handle single prompt (maintains compatibility)"""
        prompt_text = str(args[0])
        # Convert single prompt to prompt_list format for unified handling
        prompt_list = [(prompt_text, 1.0)]
        self._queue_parameter_update("prompt_list", prompt_list)

    def _handle_prompt_list(self, address, *args):
        """Handle prompt list for blending - expects JSON string"""
        try:
            prompt_data = json.loads(args[0])
            # Convert to list of tuples: [(text, weight), ...]
            prompt_list = [(item[0], float(item[1])) for item in prompt_data]
            self._queue_parameter_update("prompt_list", prompt_list)
        except Exception as e:
            logger.error(f"Error parsing prompt_list: {e}")

    def _handle_seed_list(self, address, *args):
        """Handle seed list for blending - expects JSON string"""
        try:
            seed_data = json.loads(args[0])
            # Convert to list of tuples: [(seed, weight), ...]
            seed_list = [(int(item[0]), float(item[1])) for item in seed_data]
            self._queue_parameter_update("seed_list", seed_list)
        except Exception as e:
            logger.error(f"Error parsing seed_list: {e}")

    def _handle_controlnet_config(self, address, *args):
        """Handle full ControlNet configuration array"""
        try:
            controlnet_data = json.loads(args[0])
            self._queue_parameter_update("controlnet_config", controlnet_data)
        except Exception as e:
            logger.error(f"Error parsing controlnet config: {e}")

    def _handle_controlnet_enable(self, address, *args):
        """Enable/disable ControlNet by setting conditioning scale to 0"""
        enabled = bool(args[0])
        if enabled:
            # Enable: Let extension handle proper config via /controlnets
            logger.info("ControlNet enabled via OSC")
        else:
            # Disable: Set conditioning scale to 0 for all ControlNets
            controlnet_config = [{"enabled": False, "conditioning_scale": 0.0}]
            self._queue_parameter_update("controlnet_config", controlnet_config)
            logger.info("ControlNet disabled via OSC (conditioning scale set to 0)")

    def _handle_ipadapter_config(self, address, *args):
        """Handle IPAdapter configuration"""
        try:
            ipadapter_data = json.loads(args[0])
            self._queue_parameter_update("ipadapter_config", ipadapter_data)
        except Exception as e:
            logger.error(f"Error parsing ipadapter config: {e}")

    def _handle_ipadapter_enable(self, address, *args):
        """Handle IPAdapter enable/disable"""
        enabled = bool(args[0])
        self._queue_parameter_update("use_ipadapter", enabled)
        logger.info(f"IPAdapter {'enabled' if enabled else 'disabled'}")

    def _handle_ipadapter_update(self, address, *args):
        """Handle IPAdapter update request (triggers image loading from SharedMemory)"""
        logger.info(f"OSC: Received {address}")
        try:
            if self.manager:
                self.manager.request_ipadapter_update()
            else:
                logger.warning("No manager reference - cannot trigger IPAdapter update")
        except Exception as e:
            logger.error(f"Error handling IPAdapter update: {e}")

    def _handle_cached_attention_config(self, address, *args):
        """Handle Cached Attention (StreamV2V) configuration - live updateable"""
        try:
            config = json.loads(args[0])
            # Queue both cache params for batched update
            if "cache_maxframes" in config:
                self._queue_parameter_update(
                    "cache_maxframes", int(config["cache_maxframes"])
                )
            if "cache_interval" in config:
                self._queue_parameter_update(
                    "cache_interval", int(config["cache_interval"])
                )
        except Exception as e:
            logger.error(f"Error parsing cached_attention config: {e}")

    def _handle_mode_switch(self, address, *args):
        """Handle mode switching between txt2img and img2img"""
        mode = str(args[0])
        logger.info(f"OSC: Received /sdmode = {mode}")
        if mode in ["txt2img", "img2img"]:
            try:
                if self.manager:
                    self.manager.set_mode(mode)
                    self.send_message("/mode_status", mode)
                else:
                    logger.warning("No manager reference - cannot switch mode")
            except Exception as e:
                logger.error(f"Error switching mode: {e}")
        else:
            logger.warning(f"Invalid mode: {mode} (must be 'txt2img' or 'img2img')")

    def _handle_latent_feedback_strength(self, address, *args):
        """Handle latent feedback strength updates for temporal consistency"""
        feedback_strength = float(args[0])

        try:
            if hasattr(self.manager, "wrapper") and self.manager.wrapper:
                stream = self.manager.wrapper.stream

                if hasattr(stream, "_latent_preprocessing_module"):
                    module = stream._latent_preprocessing_module
                    for processor in module.processors:
                        if processor.__class__.__name__ == "LatentFeedbackPreprocessor":
                            processor.feedback_strength = feedback_strength
                            return

                logger.warning("LatentFeedbackPreprocessor not found")
        except Exception as e:
            logger.error(f"Error updating latent feedback strength: {e}")

    def _handle_fx_parameter(self, address, *args):
        """
        DYNAMIC handler for FX processor parameters.

        Format: /fx/{processor_type}/{param_name}
        Example:
            /fx/latent_transform/zoom 1.05
            /fx/feedback_transform/feedback_strength 0.8
            /fx/any_new_processor/any_param value

        Works with ANY processor in ANY module - no hardcoding needed!
        Automatically discovers processors across all preprocessing/postprocessing modules:
        - _image_preprocessing_module
        - _latent_preprocessing_module
        - _latent_postprocessing_module
        - _image_postprocessing_module

        Just ensure StreamDiffusionExt.py generates matching OSC addresses!
        """
        # Removed noisy log: logger.debug(f"FX OSC received: {address} = {args}")

        try:
            # Parse address: /fx/processor_type/param_name
            parts = address.split("/")[2:]  # Skip empty string and 'fx'
            if len(parts) != 2:
                logger.warning(
                    f"Invalid FX parameter address: {address} (expected /fx/processor/param)"
                )
                return

            processor_type = parts[0]  # e.g., 'feedback_transform', 'latent_transform'
            param_name = parts[1]  # e.g., 'zoom', 'feedback_strength'
            param_value = args[0]

            # Removed noisy log: logger.debug(f"Parsed FX: processor={processor_type}, param={param_name}, value={param_value}")

            if not (hasattr(self.manager, "wrapper") and self.manager.wrapper):
                logger.warning("Manager or wrapper not available")
                return

            stream = self.manager.wrapper.stream
            # Removed noisy log: logger.debug(f"Scanning all modules for processor type '{processor_type}'...")

            # DYNAMIC: Search all preprocessing/postprocessing modules
            module_names = [
                "_image_preprocessing_module",
                "_latent_preprocessing_module",
                "_latent_postprocessing_module",
                "_image_postprocessing_module",
            ]

            # Convert processor_type to expected class name pattern
            # Examples:
            #   'feedback_transform' -> 'FeedbackTransformPreprocessor'
            #   'latent_transform' -> 'LatentTransformPreprocessor'
            #   'latent_feedback' -> 'LatentFeedbackPreprocessor'
            #   'custom_effect' -> 'CustomEffectPreprocessor' or 'CustomEffectPostprocessor'

            # Try both Preprocessor and Postprocessor suffixes
            processor_type_title = "".join(
                word.capitalize() for word in processor_type.split("_")
            )
            possible_class_names = [
                f"{processor_type_title}Preprocessor",
                f"{processor_type_title}Postprocessor",
                processor_type_title,  # In case class name exactly matches (no suffix)
            ]

            # Removed noisy log: logger.debug(f"Looking for class names: {possible_class_names}")

            # Search all modules
            for module_name in module_names:
                if not hasattr(stream, module_name):
                    continue

                module = getattr(stream, module_name)
                if not hasattr(module, "processors"):
                    continue

                # Removed noisy log: logger.debug(f"Checking {module_name} ({len(module.processors)} processors)")

                # Search processors in this module
                for processor in module.processors:
                    class_name = processor.__class__.__name__

                    # Check if this processor matches our target
                    if class_name in possible_class_names:
                        # Removed noisy log: logger.debug(f"Found matching processor: {class_name} in {module_name}")

                        # Verify attribute exists
                        if hasattr(processor, param_name):
                            setattr(processor, param_name, param_value)
                            # Successfully updated parameter (noisy logs removed)
                            return
                        else:
                            logger.warning(
                                f"{class_name} has no attribute '{param_name}' (available: {dir(processor)})"
                            )
                            return

            # Not found in any module
            logger.warning(
                f"Processor '{processor_type}' (looking for {possible_class_names}) not found in any module"
            )
            logger.debug(
                f"Available modules: {[name for name in module_names if hasattr(stream, name)]}"
            )

        except Exception as e:
            logger.error(f"Error updating FX parameter {address}: {e}", exc_info=True)

    def _handle_start_streaming(self, address, *args):
        """Handle start streaming command"""
        try:
            self.manager.start_streaming()
            self.send_message("/streaming_status", "started")
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            self.send_message("/streaming_status", f"error: {e}")

    def _handle_stop_streaming(self, address, *args):
        """Handle stop streaming command"""
        try:
            self.manager.stop_streaming()
            self.send_message("/streaming_status", "stopped")
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")

    def _handle_stop_application(self, address, *args):
        """Handle stop application command (shuts down entire app like main_sdtd.py)"""
        logger.info(f"OSC: Received {address}")
        try:
            if self.main_app:
                self.main_app.request_shutdown()
                self.send_message("/application_status", "stopping")
            else:
                logger.warning("No main app reference - cannot stop application")
        except Exception as e:
            logger.error(f"Error stopping application: {e}")

    def _handle_pause_streaming(self, address, *args):
        """Handle pause streaming command"""
        logger.info(f"OSC: Received {address}")
        try:
            self.manager.pause_streaming()
            self.send_message("/streaming_status", "paused")
        except Exception as e:
            logger.error(f"Error pausing streaming: {e}")

    def _handle_resume_streaming(self, address, *args):
        """Handle resume streaming command"""
        logger.info(f"OSC: Received {address}")
        try:
            self.manager.resume_streaming()
            self.send_message("/streaming_status", "resumed")
        except Exception as e:
            logger.error(f"Error resuming streaming: {e}")

    def _handle_process_frame(self, address, *args):
        """Handle process single frame command (when paused)"""
        # Don't log every process_frame - too noisy
        try:
            self.manager.process_single_frame()
        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    def _handle_frame_acknowledgment(self, address, *args):
        """Handle frame acknowledgment from TouchDesigner"""
        # Don't log frame acks - too frequent
        try:
            self.manager.acknowledge_frame()
        except Exception as e:
            logger.error(f"Error handling frame acknowledgment: {e}")

    def _handle_get_status(self, address, *args):
        """Handle status request"""
        try:
            status = self.manager.get_stream_state()
            self.send_message("/status", json.dumps(status))
        except Exception as e:
            logger.error(f"Error getting status: {e}")
