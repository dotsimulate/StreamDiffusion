"""
Standalone TensorRT installation script for StreamDiffusionTD
This is a self-contained version that doesn't rely on the streamdiffusion package imports

Version pins aligned with sd_installer/tensorrt.py and src/streamdiffusion/tools/install-tensorrt.py
"""

import platform
import subprocess
import sys
from typing import Optional

# Canonical version pins — keep in sync with sd_installer/tensorrt.py
TENSORRT_PINS = {
    "cu12": {
        "cudnn": "nvidia-cudnn-cu12==9.7.1.26",
        "tensorrt": "tensorrt==10.12.0.36",
    },
    "cu11": {
        "cudnn": "nvidia-cudnn-cu11==8.9.7.29",
        "tensorrt": "tensorrt==9.0.1.post11.dev4",
    },
    "polygraphy": "polygraphy==0.49.26",
    "onnx_graphsurgeon": "onnx-graphsurgeon==0.5.8",
    "pywin32": "pywin32==311",
    "triton_windows": "triton-windows==3.4.0.post21",
}


def run_pip(command: str):
    """Run pip command with proper error handling"""
    return subprocess.check_call([sys.executable, "-m", "pip"] + command.split())


def is_installed(package_name: str) -> bool:
    """Check if a package is installed"""
    try:
        __import__(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False


def version(package_name: str) -> Optional[str]:
    """Get version of installed package"""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def get_cuda_version_from_torch() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None

    cuda_version = torch.version.cuda
    if cuda_version:
        # Return full version like "12.8" for better detection
        major_minor = ".".join(cuda_version.split(".")[:2])
        return major_minor
    return None


def install(cu: Optional[str] = None):
    if cu is None:
        cu = get_cuda_version_from_torch()

    if cu is None:
        print("Could not detect CUDA version. Please specify manually.")
        return

    print(f"Detected CUDA version: {cu}")
    print("Installing TensorRT requirements...")

    # Determine CUDA major version for package selection
    cuda_major = cu.split(".")[0] if cu else "12"
    cuda_version_float = float(cu) if cu else 12.0

    # Uninstall old TensorRT versions (anything below 10.8)
    if is_installed("tensorrt"):
        current_version_str = version("tensorrt")
        if current_version_str:
            try:
                from packaging.version import Version
                needs_uninstall = Version(current_version_str) < Version("10.8.0")
            except ImportError:
                # packaging not available - compare by major version
                try:
                    major = int(current_version_str.split(".")[0])
                    needs_uninstall = major < 10
                except (ValueError, IndexError):
                    needs_uninstall = False
            if needs_uninstall:
                print("Uninstalling old TensorRT version...")
                run_pip("uninstall -y tensorrt")

    if cuda_major == "12":
        pins = TENSORRT_PINS["cu12"]
        if cuda_version_float >= 12.8:
            print("Installing TensorRT 10.12+ for CUDA 12.8+ (Blackwell GPU support)...")
        else:
            print("Installing TensorRT for CUDA 12.x...")

        cudnn_name = pins["cudnn"]
        tensorrt_pkg = pins["tensorrt"]

        print(f"Installing cuDNN: {cudnn_name}")
        run_pip(f"install {cudnn_name} --no-cache-dir")

        print(f"Installing TensorRT for CUDA {cu}: {tensorrt_pkg}")
        run_pip(f"install --extra-index-url https://pypi.nvidia.com {tensorrt_pkg} --no-cache-dir")

    elif cuda_major == "11":
        pins = TENSORRT_PINS["cu11"]
        print("Installing TensorRT for CUDA 11.x...")

        cudnn_name = pins["cudnn"]
        tensorrt_pkg = pins["tensorrt"]

        print(f"Installing cuDNN: {cudnn_name}")
        run_pip(f"install {cudnn_name} --no-cache-dir")

        print(f"Installing TensorRT for CUDA {cu}: {tensorrt_pkg}")
        run_pip(
            f"install --pre --extra-index-url https://pypi.nvidia.com {tensorrt_pkg} --no-cache-dir"
        )
    else:
        print(f"Unsupported CUDA version: {cu}")
        print("Supported versions: CUDA 11.x, 12.x")
        return

    # Install additional TensorRT tools (pinned versions)
    if not is_installed("polygraphy"):
        print("Installing polygraphy...")
        run_pip(
            f"install {TENSORRT_PINS['polygraphy']} --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir"
        )
    if not is_installed("onnx_graphsurgeon"):
        print("Installing onnx-graphsurgeon...")
        run_pip(
            f"install {TENSORRT_PINS['onnx_graphsurgeon']} --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir"
        )
    if platform.system() == "Windows" and not is_installed("pywin32"):
        print("Installing pywin32...")
        run_pip(f"install {TENSORRT_PINS['pywin32']} --no-cache-dir")
    if platform.system() == "Windows" and not is_installed("triton"):
        print("Installing triton-windows...")
        run_pip(f"install {TENSORRT_PINS['triton_windows']} --no-cache-dir")

    print("TensorRT installation completed successfully!")


if __name__ == "__main__":
    install()
