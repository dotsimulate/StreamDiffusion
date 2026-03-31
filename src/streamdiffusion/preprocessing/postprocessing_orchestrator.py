import logging
from typing import Any, Dict, List, Optional

import torch

from .base_orchestrator import BaseOrchestrator


logger = logging.getLogger(__name__)


class PostprocessingOrchestrator(BaseOrchestrator[torch.Tensor, torch.Tensor]):
    """
    Orchestrates postprocessing with parallelization and pipelining.

    Handles super-resolution, enhancement, style transfer, and other postprocessing operations
    that are applied to generated images after diffusion.
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_workers: int = 4,
        pipeline_ref: Optional[Any] = None,
    ):
        # Postprocessing: 50ms timeout for quality-critical operations like upscaling
        super().__init__(device, dtype, max_workers, timeout_ms=20.0, pipeline_ref=pipeline_ref)

        # Postprocessing-specific state
        # Cache uses data_ptr + shape for O(1) identity check instead of torch.equal (O(N))
        self._last_input_ptr = None
        self._last_input_shape = None
        self._last_processed_result = None
        self._current_input_tensor = None  # For BaseOrchestrator fallback logic

    def process_pipelined(
        self, input_tensor: torch.Tensor, postprocessors: List[Any], *args, **kwargs
    ) -> torch.Tensor:
        """
        Process input with intelligent pipelining.

        Overrides base method to store current input tensor for fallback logic.
        """
        # Store current input for fallback logic
        self._current_input_tensor = input_tensor

        # RACE CONDITION FIX: Check if there are actually enabled processors
        # Filter to only enabled processors (same logic as _get_ordered_processors)
        enabled_processors = [p for p in postprocessors if getattr(p, "enabled", True)] if postprocessors else []

        if not enabled_processors:
            return input_tensor

        # Call parent implementation
        return super().process_pipelined(input_tensor, postprocessors, *args, **kwargs)

    def _should_use_sync_processing(self, *args, **kwargs) -> bool:
        """
        Determine if synchronous processing should be used instead of pipelined.

        Checks if any postprocessor has requires_sync_processing=True attribute.
        Postprocessors that affect temporal coherence (e.g. when their output is
        stored as prev_image_result and read by feedback loops) need synchronous
        processing to avoid frame-lag oscillation.

        Returns:
            True if any postprocessor requires sync processing, False otherwise
        """
        if len(args) < 1:
            return False
        processors = args[0]
        if not processors:
            return False
        for proc in processors:
            if proc is not None and getattr(proc, 'requires_sync_processing', False):
                return True
        return False

    def process_sync(self, input_tensor: torch.Tensor, postprocessors: List[Any], *args, **kwargs) -> torch.Tensor:
        """
        Process tensor through postprocessors synchronously.

        Args:
            input_tensor: Input tensor to postprocess (typically from diffusion output)
            postprocessors: List of postprocessor instances
            *args, **kwargs: Additional arguments for postprocessors

        Returns:
            Postprocessed tensor
        """
        if not postprocessors:
            return input_tensor

        # Use same stream context as background processing for consistency
        original_stream = self._set_background_stream_context()
        try:
            # Sequential application of postprocessors
            current_tensor = input_tensor
            for postprocessor in postprocessors:
                if postprocessor is not None:
                    current_tensor = self._apply_single_postprocessor(current_tensor, postprocessor)

            return current_tensor
        finally:
            self._restore_stream_context(original_stream)

    def _process_frame_background(
        self, input_tensor: torch.Tensor, postprocessors: List[Any], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Process a frame in the background thread.

        Implementation of BaseOrchestrator._process_frame_background for postprocessing.

        Returns:
            Dictionary containing processing results and status
        """
        try:
            # Set CUDA stream for background processing
            original_stream = self._set_background_stream_context()

            if not postprocessors:
                return {"result": input_tensor, "status": "success"}

            # Check for cache hit using data_ptr + shape — O(1) vs torch.equal's O(N).
            # TRT engines reuse the same output buffer each frame, so data_ptr identity
            # reliably detects whether the input is the same buffer as last frame.
            cache_hit = (
                self._last_input_ptr is not None
                and self._last_processed_result is not None
                and input_tensor.data_ptr() == self._last_input_ptr
                and input_tensor.shape == self._last_input_shape
            )

            if cache_hit:
                return {
                    "result": self._last_processed_result,
                    "status": "success",
                    "cache_hit": True,
                }

            # Update cache — store ptr + shape, no tensor clone needed
            self._last_input_ptr = input_tensor.data_ptr()
            self._last_input_shape = input_tensor.shape

            # Process postprocessors in parallel if multiple, sequential if single
            if len(postprocessors) > 1:
                result = self._process_postprocessors_parallel(input_tensor, postprocessors)
            else:
                result = self._apply_single_postprocessor(input_tensor, postprocessors[0])

            # Cache the processed result for future cache hits
            self._last_processed_result = result

            return {"result": result, "status": "success"}

        except Exception as e:
            logger.error(f"PostprocessingOrchestrator: Background processing failed: {e}")
            return {
                "result": input_tensor,  # Return original on error
                "error": str(e),
                "status": "error",
            }
        finally:
            # Restore original CUDA stream
            self._restore_stream_context(original_stream)

    def _process_postprocessors_parallel(self, input_tensor: torch.Tensor, postprocessors: List[Any]) -> torch.Tensor:
        """
        Process multiple postprocessors in parallel.

        Note: This applies postprocessors sequentially for now, but could be extended
        to support parallel processing for independent postprocessors in the future.

        Args:
            input_tensor: Input tensor to process
            postprocessors: List of postprocessor instances

        Returns:
            Processed tensor
        """
        # For now, apply sequentially since most postprocessors are dependent
        # Future enhancement: Detect independent postprocessors and run in parallel
        current_tensor = input_tensor
        for postprocessor in postprocessors:
            if postprocessor is not None:
                current_tensor = self._apply_single_postprocessor(current_tensor, postprocessor)

        return current_tensor

    def _apply_single_postprocessor(self, input_tensor: torch.Tensor, postprocessor: Any) -> torch.Tensor:
        """
        Apply a single postprocessor to the input tensor.

        Handles normalization conversion between VAE output range [-1,1] and
        processor input range [0,1], then converts back to VAE range.

        Args:
            input_tensor: Input tensor from VAE (range [-1,1])
            postprocessor: Postprocessor instance

        Returns:
            Processed tensor in VAE range [-1,1]
        """
        try:
            # Ensure tensor is on correct device and dtype
            processed_tensor = input_tensor.to(device=self.device, dtype=self.dtype)

            logger.debug(
                "_apply_single_postprocessor: Converting tensor from VAE range [-1,1] to processor range [0,1]"
            )
            processor_input = (processed_tensor / 2.0 + 0.5).clamp(0, 1)

            # Apply postprocessor
            if hasattr(postprocessor, "process_tensor"):
                # Prefer tensor processing if available
                processor_output = postprocessor.process_tensor(processor_input)
            elif hasattr(postprocessor, "process"):
                # Fallback to general process method
                processor_output = postprocessor.process(processor_input)
            elif callable(postprocessor):
                # Treat as callable
                processor_output = postprocessor(processor_input)
            else:
                logger.warning(f"PostprocessingOrchestrator: Unknown postprocessor type: {type(postprocessor)}")
                return processed_tensor

            # Ensure result is a tensor
            if isinstance(processor_output, torch.Tensor):
                # CRITICAL: Convert back from processor output range [0,1] to VAE input range [-1,1]
                logger.debug(
                    "_apply_single_postprocessor: Converting result from processor range [0,1] back to VAE range [-1,1]"
                )
                result = (processor_output - 0.5) * 2.0  # Convert [0,1] -> [-1,1]

                return result.to(device=self.device, dtype=self.dtype)
            else:
                logger.warning(
                    f"PostprocessingOrchestrator: Postprocessor returned non-tensor: {type(processor_output)}"
                )
                return processed_tensor

        except Exception as e:
            logger.error(f"PostprocessingOrchestrator: Postprocessor failed: {e}")
            return input_tensor  # Return original on error

    def clear_cache(self) -> None:
        """Clear postprocessing cache"""
        self._last_input_ptr = None
        self._last_input_shape = None
        self._last_processed_result = None
