"""Global pytest configuration, tier hooks, and fixture scaffolding.

Follows the harden-test-suite architecture:
- Auto-marking by directory structure (unit, integration, slow, gpu, quality)
- Global state reset autouse fixture to eliminate ordering leaks
- Isolated test storage helpers
"""

import sys
from pathlib import Path
from typing import Generator

import pytest


def _requires_cuda(item: pytest.Item) -> bool:
    """True if the item carries a skipif whose reason says it needs CUDA."""
    return any("cuda" in str(m.kwargs.get("reason", "")).lower() for m in item.iter_markers("skipif"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark tests based on filesystem directory and naming conventions."""
    for item in items:
        path_str = str(item.fspath).replace("\\", "/")
        if "tests/unit/" in path_str:
            item.add_marker(pytest.mark.unit)
            # CUDA-only unit tests (skipped on CPU hosts) are still GPU tests: let `-m gpu`
            # and `-m "not gpu"` select them.
            if _requires_cuda(item):
                item.add_marker(pytest.mark.gpu)
        elif "tests/gpu/" in path_str:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.gpu)
        elif "tests/quality/" in path_str:
            item.add_marker(pytest.mark.quality)


@pytest.fixture(autouse=True)
def reset_global_state() -> Generator[None, None, None]:
    """Reset module-level singletons, GPU cache, and temporary states between tests."""
    yield
    # Only touch CUDA if the test already imported torch; never import it just for cleanup.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def isolated_storage(tmp_path: Path) -> Path:
    """Provide a guaranteed temporary directory isolated from production storage."""
    storage = tmp_path / "streamdiffusion_test_data"
    storage.mkdir(parents=True, exist_ok=True)
    return storage
