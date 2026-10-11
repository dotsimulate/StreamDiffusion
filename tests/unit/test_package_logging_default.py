"""Importing streamdiffusion keeps INFO logs visible without touching the root logger.

pipeline.py's import-time basicConfig(INFO) was removed so host apps own root logging; this
guards the replacement (a package-logger level) in clean interpreters.
"""

import subprocess
import sys

import pytest


def _run(snippet: str) -> str:
    result = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_plain_import_enables_package_info_and_leaves_root_alone():
    out = _run(
        "import logging, streamdiffusion;"
        "print(logging.getLogger('streamdiffusion').getEffectiveLevel(), logging.getLogger().level)"
    )
    assert out == f"{20} {30}"


@pytest.mark.slow
def test_host_configured_root_is_respected():
    out = _run(
        "import logging; logging.getLogger().setLevel(logging.DEBUG); import streamdiffusion;"
        "print(logging.getLogger('streamdiffusion').level)"
    )
    assert out == "0"  # NOTSET: inherits the host's DEBUG
