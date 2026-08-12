import concurrent.futures
import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

import torch

logger = logging.getLogger(__name__)

# Type variables for generic orchestrator
T = TypeVar("T")  # Input type (e.g., ControlImage for preprocessing)
R = TypeVar("R")  # Result type (e.g., List[torch.Tensor] for preprocessing)


class BaseOrchestrator(Generic[T, R], ABC):
    """
    Generic base orchestrator for parallelized and pipelined processing.

    Handles thread pool management, pipeline state, and inter-frame pipelining
    while leaving domain-specific processing logic to subclasses.

    Type Parameters:
        T: Input type for processing operations
        R: Result type returned from processing operations
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_workers: int = 4,
        timeout_ms: float = 10.0,
        pipeline_ref: Optional[Any] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.timeout_ms = timeout_ms
        self.pipeline_ref = pipeline_ref
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Pipeline state for pipelined processing
        self._next_frame_future = None
        self._next_frame_result = None

        # CUDA stream for background processing to avoid GPU contention
        self._background_stream = None
        self._bg_pre_event: Optional[torch.cuda.Event] = None  # caller stream -> background stream barrier
        self._bg_post_event: Optional[torch.cuda.Event] = None  # background stream -> consumer stream barrier
        device_str = str(device)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            self._background_stream = torch.cuda.Stream()
            self._bg_pre_event = torch.cuda.Event()
            self._bg_post_event = torch.cuda.Event()

    def cleanup(self) -> None:
        """Cleanup thread pool and CUDA stream resources"""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)

        # Cleanup CUDA stream if it exists
        if hasattr(self, "_background_stream") and self._background_stream is not None:
            # Synchronize the stream before cleanup
            torch.cuda.synchronize()
            self._background_stream = None

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup()
        except Exception:
            pass

    @abstractmethod
    def _should_use_sync_processing(self, *args, **kwargs) -> bool:
        """
        Determine if synchronous processing should be used instead of pipelined.

        Subclasses implement domain-specific logic (e.g., feedback preprocessor detection).

        Returns:
            True if sync processing should be used, False for pipelined processing
        """
        pass

    @abstractmethod
    def _process_frame_background(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Process a frame in the background thread.

        Subclasses implement their specific processing logic here.

        Returns:
            Dictionary containing processing results and status
        """
        pass

    def process_pipelined(self, input_data: T, *args, **kwargs) -> R:
        """
        Process input with intelligent pipelining.

        Automatically falls back to sync processing when required by domain logic,
        otherwise uses pipelined processing for performance.

        Args:
            input_data: Input data to process
            *args, **kwargs: Additional arguments passed to processing methods

        Returns:
            Processing results
        """
        # Check if sync processing is required (domain-specific logic)
        if self._should_use_sync_processing(*args, **kwargs):
            return self.process_sync(input_data, *args, **kwargs)

        # Use pipelined processing
        # Wait for previous frame processing; non-blocking with short timeout
        self._wait_for_previous_processing()

        # Start next frame processing in background
        self._start_next_frame_processing(input_data, *args, **kwargs)

        # Apply current frame processing results if available; otherwise signal no update
        return self._apply_current_frame_processing(*args, **kwargs)

    @abstractmethod
    def process_sync(self, input_data: T, *args, **kwargs) -> R:
        """
        Process input synchronously.

        Subclasses implement their specific synchronous processing logic.

        Args:
            input_data: Input data to process
            *args, **kwargs: Additional arguments passed to processing methods

        Returns:
            Processing results
        """
        pass

    def _start_next_frame_processing(self, input_data: T, *args, **kwargs) -> None:
        """Start processing for next frame in background thread"""
        # Pre-exec barrier: order the background stream's work after this (caller)
        # thread's queued work before handing off to the executor. Must happen here,
        # not inside _process_frame_background -- a worker thread has no notion of
        # "the caller's current stream" (see _arm_background_handoff).
        self._arm_background_handoff()
        # Submit background processing
        self._next_frame_future = self._executor.submit(self._process_frame_background, input_data, *args, **kwargs)

    def _wait_for_previous_processing(self) -> None:
        """Wait for previous frame processing with configurable timeout"""
        if hasattr(self, "_next_frame_future") and self._next_frame_future is not None:
            try:
                # Use configurable timeout based on orchestrator type
                self._next_frame_result = self._next_frame_future.result(timeout=self.timeout_ms / 1000.0)
                # Post-exec barrier: make this (consumer) thread's stream wait for the
                # background stream's work to land, and record_stream() the results.
                self._consume_background_handoff(self._next_frame_result)
            except concurrent.futures.TimeoutError:
                # Non-blocking: skip applying results this frame
                self._next_frame_result = None
            except Exception as e:
                logger.error(f"BaseOrchestrator: Processing error: {e}")
                self._next_frame_result = None
        else:
            self._next_frame_result = None

    def _apply_current_frame_processing(self, processors=None, *args, **kwargs) -> R:
        """
        Apply processing results from previous iteration.

        Default implementation provides common fallback logic for tensor-to-tensor orchestrators.
        Subclasses can override this method for specialized behavior.

        Args:
            processors: List of processors/postprocessors to apply (parameter name varies by subclass)
            *args, **kwargs: Additional arguments

        Returns:
            Processing results, or processed current input if no results available
        """
        if not hasattr(self, "_next_frame_result") or self._next_frame_result is None:
            # First frame or no background results - process current input synchronously
            if hasattr(self, "_current_input_tensor") and self._current_input_tensor is not None:
                if processors:
                    return self.process_sync(self._current_input_tensor, processors)
                else:
                    return self._current_input_tensor

            # If we don't have current input stored, we have an issue
            class_name = self.__class__.__name__
            logger.error(f"{class_name}: No background results and no current input tensor available")
            raise RuntimeError(f"{class_name}: No processing results available")

        result = self._next_frame_result
        if result["status"] != "success":
            class_name = self.__class__.__name__
            logger.warning(f"{class_name}: Background processing failed: {result.get('error', 'Unknown error')}")
            # Process current input synchronously on error
            if hasattr(self, "_current_input_tensor") and self._current_input_tensor is not None:
                if processors:
                    return self.process_sync(self._current_input_tensor, processors)
                else:
                    return self._current_input_tensor
            raise RuntimeError(f"{class_name}: Background processing failed and no fallback available")

        return result["result"]

    # ------------------------------------------------------------------
    # Event-based cross-stream handoff for background processing.
    #
    # torch.cuda.Stream() is non-blocking by default, so there is no implicit
    # legacy-stream synchronization between it and whatever stream the caller/
    # consumer is on -- an explicit event barrier is required on both ends of the
    # handoff. Mirrors the producer/consumer event protocol already proven in
    # preprocessing/processors/trt_base.py's TensorRTEngine.infer():
    #   producer: pre_event.record(); dedicated_stream.wait_event(pre_event); ... work ...
    #             post_event.record(dedicated_stream)
    #   consumer: current_stream().wait_event(post_event); tensor.record_stream(current_stream())
    #
    # Split across two call sites because _process_frame_background runs in a
    # ThreadPoolExecutor worker thread: torch.cuda.current_stream() is thread-local,
    # so "the caller's current stream" can only be read/recorded on the caller
    # thread itself, before dispatch -- not from inside the worker. See each
    # method's docstring for exactly which thread must call it.
    # ------------------------------------------------------------------

    def _arm_background_handoff(self) -> None:
        """
        Pre-exec barrier: record an event on the CALLING thread's current stream,
        then make `_background_stream` wait on it before running anything new.

        Must run on the thread whose queued work the background stream needs to
        wait behind. For the async pipeline path that's `_start_next_frame_processing`,
        called on the caller thread before dispatch to the executor. For a
        same-thread caller (e.g. `process_sync`) there is no thread hop, so it may
        call this immediately before doing the GPU work itself.
        """
        if self._background_stream is not None:
            self._bg_pre_event.record()
            self._background_stream.wait_event(self._bg_pre_event)

    def _background_stream_scope(self):
        """
        Scope `torch.cuda.current_stream()` (thread-local) to `_background_stream`
        for GPU work about to run on the calling thread. Records no barrier --
        callers that need the pre-exec barrier call `_arm_background_handoff()`
        themselves first (see that method for why it can't just happen here).

        Returns a context manager (`torch.cuda.stream(...)`), or a no-op context
        manager if there is no background stream (CPU device).
        """
        if self._background_stream is None:
            return contextlib.nullcontext()
        return torch.cuda.stream(self._background_stream)

    def _finish_background_handoff(self) -> None:
        """
        Post-exec barrier, producer side: record the completion event on
        `_background_stream`. Must be the last GPU-related action taken for this
        frame's background work, before the consumer touches results -- called
        from `_process_frame_background`'s `finally` block (still on the worker
        thread) or, for a same-thread caller, right after its GPU work.
        """
        if self._background_stream is not None:
            self._bg_post_event.record(self._background_stream)

    def _consume_background_handoff(self, result: Any) -> Any:
        """
        Post-exec barrier, consumer side: make the CONSUMING thread's current
        stream wait on `_background_stream`'s completion event, then
        `record_stream()` every CUDA tensor found in `result` (recursing through
        list/tuple/dict containers) so PyTorch's caching allocator can't reclaim a
        buffer the background stream might still be writing.

        Safe to call even when `_finish_background_handoff()` was never reached
        this frame (e.g. an exception path bailed out early): waiting on a
        never-recorded `torch.cuda.Event` is a documented no-op -- "If
        cudaEventRecord() has not been called on event, this call acts as if the
        record has already completed."

        Returns `result` unchanged, for call-site chaining.
        """
        if self._background_stream is None:
            return result
        current = torch.cuda.current_stream()
        current.wait_event(self._bg_post_event)
        self._record_stream_on_tensors(result, current)
        return result

    @staticmethod
    def _record_stream_on_tensors(obj: Any, stream: "torch.cuda.Stream") -> None:
        """Recurse through list/tuple/dict containers, calling `tensor.record_stream(stream)`
        on every CUDA tensor found."""
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                obj.record_stream(stream)
        elif isinstance(obj, dict):
            for v in obj.values():
                BaseOrchestrator._record_stream_on_tensors(v, stream)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                BaseOrchestrator._record_stream_on_tensors(v, stream)
