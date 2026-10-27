"""Shared classification helpers for HuggingFace Hub download failures.

Both `wrapper.py._load_model` and `controlnet_module.py._load_pytorch_controlnet_model`
call into `diffusers`/`huggingface_hub` loaders that can fail for two very different
reasons that look identical at the top level:

1. The model id/path is genuinely wrong (typo, private repo, missing file).
2. The download itself failed (timeout, connection reset, or a HuggingFace Xet
   transfer-layer failure) - and `diffusers` rewrites that into a generic
   `EnvironmentError`/`OSError` that reads exactly like case 1 (see
   `diffusers/utils/hub_utils.py::_get_model_file`, `except EnvironmentError`).

A real-world example of case 2: a 729 MB ControlNet weight download stalls at
0 bytes for 33 minutes, then `hf_xet`'s Rust download layer raises
`OSError: I/O error: I/O error: error decoding response body`. That `OSError`
*is* an `EnvironmentError` (they're the same type in Python 3), so it is caught
by the same handler as "this model id doesn't exist" and reported as: "make
sure '<repo>' is the correct path to a directory containing a file named
<weights_name>". The real cause survives only in `__cause__`.

This module gives both call sites a single, chain-aware way to detect that
case and react to it (retry with Xet disabled, or at least report the real
cause with an actionable hint) instead of independently re-deriving it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import requests
from huggingface_hub import constants as _hf_constants
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError

# Substrings seen in the wrapped OSError's own message when Xet fails. Kept as a
# fallback for cases where the traceback frames aren't available (e.g. the
# exception was reconstructed/pickled across a process boundary).
_XET_MESSAGE_MARKERS = (
    "error decoding response body",
    "cas-server",
    "xethub",
)

# Function names that only appear in the hf_xet download path's own traceback
# frames (huggingface_hub/file_download.py calling into the hf_xet package).
_XET_FRAME_FUNCTION_NAMES = frozenset({"xet_get", "download_files"})

NETWORK_ERROR_TYPES = (requests.exceptions.RequestException, HfHubHTTPError, LocalEntryNotFoundError)

# Shared hint appended when a download/network failure is about to be reported.
# NOTE: the substring "huggingface download failure" is asserted verbatim by
# tests/unit/test_load_model_network_error_reporting.py - do not reword it away.
NETWORK_HINT = (
    " Hint: this looks like a HuggingFace download failure (timeout/connection/Xet transfer), not an "
    "invalid model id. The download resumes from where it stopped, so re-launching is usually enough. "
    "If this keeps happening, consider `pip install hf_xet` for large-file transfers, set "
    "HF_HUB_DISABLE_XET=1 to fall back to plain HTTPS downloads, and raise HF_HUB_DOWNLOAD_TIMEOUT if "
    "this keeps happening on a slow connection."
)


# Appended when a model load dies on host memory / Windows commit limit rather than the network.
PAGEFILE_HINT = (
    " Hint: Windows ran out of virtual memory (RAM + paging file) while loading the model weights. "
    "Set the paging file to 'System managed size' (or at least 32 GB) under System > Advanced > "
    "Performance > Virtual memory, close other memory-heavy apps, then re-launch."
)

# Messages from loading a model folder whose files are missing or don't match each other: diffusers'
# `load_sub_model` falling back to the root for a missing component folder, and torch's state_dict
# load when a config.json disagrees with its weights (e.g. a truncated text_encoder_2/config.json
# makes CLIPTextConfig fall back to hidden_size=512 -> "[77, 1280] vs [77, 512]").
_CORRUPT_CACHE_MARKERS = (
    "size mismatch for",
    "error(s) in loading state_dict",
)


def is_corrupt_cache_error(exc: BaseException) -> bool:
    """Chain-aware check for "the model files on disk are incomplete or inconsistent".

    huggingface_hub reuses a file that is already in the snapshot without checking it, so a bad
    cached file fails the same way on every launch, online or not, until it is deleted.
    """
    for ancestor in iter_exception_chain(exc):
        message = str(ancestor).lower()
        if any(marker in message for marker in _CORRUPT_CACHE_MARKERS):
            return True
        if isinstance(ancestor, OSError) and "no file named" in message and "found in directory" in message:
            return True
    return False


def cache_corrupt_hint(model_id_or_path: str) -> str:
    """Hint for `is_corrupt_cache_error`, naming the HF cache folder to delete for a bare repo id."""
    if os.path.exists(model_id_or_path) or model_id_or_path.endswith((".safetensors", ".ckpt")):
        return (
            f" Hint: the model files at '{model_id_or_path}' look incomplete or don't match each other "
            "(a missing component folder or config.json, or a config that doesn't fit its weights). "
            "Re-download or re-copy the model, then re-launch."
        )
    repo_dir = os.path.join(_hf_constants.HF_HUB_CACHE, "models--" + model_id_or_path.replace("/", "--"))
    return (
        f" Hint: the cached download of '{model_id_or_path}' looks incomplete or corrupt (a missing "
        "component folder or config.json, or a config that doesn't fit its weights). Cached files are "
        f"reused without being checked, so this repeats on every launch. Delete '{repo_dir}', make sure "
        "HF_HUB_OFFLINE is not set to 1, and re-launch while online to download it again."
    )


def iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield `exc`, then each `__cause__`/`__context__` ancestor, deepest last.

    Guards against reference cycles (shouldn't normally happen, but a cycle
    here would otherwise hang forever).
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _traceback_mentions_xet(exc: BaseException) -> bool:
    """Inspect exc's own traceback frames for hf_xet's download call stack."""
    tb = exc.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        if code.co_name in _XET_FRAME_FUNCTION_NAMES:
            return True
        filename = code.co_filename.replace("\\", "/")
        if "/hf_xet/" in filename or filename.endswith("hf_xet.py"):
            return True
        tb = tb.tb_next
    return False


def is_xet_download_error(exc: BaseException) -> bool:
    """Detect a HuggingFace Xet transfer-layer failure hiding inside `exc`'s chain.

    The exception that reaches call sites is usually the *rewritten* diffusers
    `EnvironmentError` ("can't find the model..."), not the original `OSError`
    from `hf_xet`. Matching on the top-level message alone would miss it, so
    this walks `__cause__`/`__context__` and checks each ancestor's own
    traceback frames and message.
    """
    for ancestor in iter_exception_chain(exc):
        if not isinstance(ancestor, OSError):
            continue
        if _traceback_mentions_xet(ancestor):
            return True
        message = str(ancestor).lower()
        if any(marker in message for marker in _XET_MESSAGE_MARKERS):
            return True
    return False


def is_network_error(exc: BaseException) -> bool:
    """Chain-aware check for "this is a download/connection failure, not a bad model id"."""
    for ancestor in iter_exception_chain(exc):
        if isinstance(ancestor, NETWORK_ERROR_TYPES):
            return True
    return is_xet_download_error(exc)


def is_missing_hub_file(exc: BaseException) -> bool:
    """True when the chain holds a server-side "file not in this repo" (HTTP 404) error.

    `EntryNotFoundError` subclasses `HfHubHTTPError`, so `is_network_error` also matches it, but
    the hub answered fine: the file simply does not exist (e.g. a repo with no fp16 variant).
    `LocalEntryNotFoundError` is excluded: it means "offline and not cached", a real network failure.
    """
    return any(
        isinstance(a, EntryNotFoundError) and not isinstance(a, LocalEntryNotFoundError)
        for a in iter_exception_chain(exc)
    )


def is_pagefile_error(exc: BaseException) -> bool:
    """Chain-aware check for host-memory exhaustion while loading weights.

    On Windows safetensors maps the file copy-on-write, so the whole file counts against the
    commit limit; a small paging file fails with `OSError ... (os error 1455)` and diffusers'
    text-mode fallback then re-raises a bare `MemoryError` (empty `str()`).
    """
    for ancestor in iter_exception_chain(exc):
        if isinstance(ancestor, MemoryError):
            return True
        if isinstance(ancestor, OSError):
            if getattr(ancestor, "winerror", None) == 1455:
                return True
            message = str(ancestor).lower()
            if "os error 1455" in message or "paging file is too small" in message:
                return True
    return False


@contextmanager
def xet_disabled():
    """Temporarily force `huggingface_hub` to skip the Xet transfer layer.

    `huggingface_hub.constants.HF_HUB_DISABLE_XET` is evaluated from
    `os.environ` exactly once, at import time (see `constants.py`), so setting
    the environment variable alone has no effect on an already-imported
    process. Both call sites that matter (`is_xet_available()` and
    `file_download.py`'s `xet_get`/`http_get` branch) read the *module
    attribute* at call time, so patch that directly. The environment variable
    is set too, for the benefit of any late imports or subprocesses.
    """
    original_attr = _hf_constants.HF_HUB_DISABLE_XET
    original_env = os.environ.get("HF_HUB_DISABLE_XET")
    _hf_constants.HF_HUB_DISABLE_XET = True
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        yield
    finally:
        _hf_constants.HF_HUB_DISABLE_XET = original_attr
        if original_env is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = original_env
