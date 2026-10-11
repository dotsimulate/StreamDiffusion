"""Importing the pipeline must not configure the root logger.

pipeline.py used to call logging.basicConfig(INFO) at import time, which pre-empted the host
application's (TouchDesigner fork's) own logging setup and leaked third-party DEBUG/INFO noise.
"""

import ast
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[2] / "src" / "streamdiffusion" / "pipeline.py"


def test_pipeline_does_not_call_basicconfig():
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "basicConfig"
    ]
    assert calls == [], f"pipeline.py calls basicConfig at line(s) {calls}"
