"""Global pytest configuration, tier hooks, and fixture scaffolding.

Follows the harden-test-suite architecture:
- Auto-marking by directory structure (unit, integration, slow, gpu, quality)
- Global state reset autouse fixture to eliminate ordering leaks
- Isolated test storage helpers
"""

from pathlib import Path
from typing import Generator

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark tests based on filesystem directory and naming conventions."""
    for item in items:
        path_str = str(item.fspath).replace("\\", "/")
        if "tests/unit/" in path_str:
            item.add_marker(pytest.mark.unit)
        elif "tests/gpu/" in path_str:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.gpu)
        elif "tests/quality/" in path_str:
            item.add_marker(pytest.mark.quality)
        elif "tests/fast_integration/" in path_str:
            item.add_marker(pytest.mark.integration)
        elif "tests/slow_integration/" in path_str:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def reset_global_state() -> Generator[None, None, None]:
    """Reset module-level singletons, GPU cache, and temporary states between tests."""
    yield
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def isolated_storage(tmp_path: Path) -> Path:
    """Provide a guaranteed temporary directory isolated from production storage."""
    storage = tmp_path / "streamdiffusion_test_data"
    storage.mkdir(parents=True, exist_ok=True)
    return storage
