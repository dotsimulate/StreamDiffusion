import os
import re
import sys

from setuptools import find_packages, setup

# Copied from pip_utils.py to avoid import
def _check_torch_installed():
    try:
        import torch
        import torchvision
    except Exception:
        msg = (
            "Missing required pre-installed packages: torch, torchvision\n"
            "On Linux/Windows: pip install --index-url https://download.pytorch.org/whl/cu12x torch torchvision\n"
            "On Mac: pip install torch torchvision  (MPS acceleration is used automatically)"
        )
        raise RuntimeError(msg)

    is_mac = sys.platform == "darwin"
    has_cuda = bool(torch.version.cuda)
    has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

    if not has_cuda and not has_mps and not is_mac:
        raise RuntimeError(
            "Detected CPU-only PyTorch on a non-Mac platform. "
            "Install CUDA-enabled torch/torchvision before installing this package."
        )


def is_mac():
    return sys.platform == "darwin"


def get_cuda_constraint():
    if is_mac():
        return None  # cuda-python not used on Mac

    cuda_version = os.environ.get("STREAMDIFFUSION_CUDA_VERSION") or \
                    os.environ.get("CUDA_VERSION")

    if not cuda_version:
        try:
            import torch
            cuda_version = torch.version.cuda
        except Exception:
            # might not be available during wheel build, so we have to ignore
            pass

    if not cuda_version:
        return ">=11,<13"

    parts = cuda_version.split(".")
    if len(parts) < 2:
        raise RuntimeError(f"Invalid CUDA version: {cuda_version}")
    return f"~={parts[0]}.{parts[1]}"


if any(cmd in sys.argv for cmd in ("install", "develop")):
    _check_torch_installed()

_cuda_constraint = get_cuda_constraint()
_deps = [
    *([] if _cuda_constraint is None else [f"cuda-python{_cuda_constraint}"]),
    "xformers==0.0.30",
    "diffusers @ git+https://github.com/varshith15/diffusers.git@3e3b72f557e91546894340edabc845e894f00922",
    "transformers==4.56.0",
    "accelerate==1.13.0",
    "huggingface_hub==0.35.0",
    "Pillow>=12.1.1",  # CVE-2026-25990: out-of-bounds write in PSD loading
    "fire==0.7.1",
    "omegaconf==2.3.0",
    "onnx==1.18.0",  # onnx-graphsurgeon 0.5.8 requires onnx.helper.float32_to_bfloat16 (removed in onnx 1.19+)
    "onnxruntime==1.24.3",
    "onnxruntime-gpu==1.24.3",
    "polygraphy==0.49.26",
    "protobuf>=4.25.8,<5",  # mediapipe 0.10.21 requires protobuf 4.x; 4.25.8 fixes CVE-2025-4565; CVE-2026-0994 (JSON DoS) accepted risk for local pipeline
    "colored==2.3.1",
    "pywin32==311;sys_platform == 'win32'",
    "onnx-graphsurgeon==0.5.8",
    "controlnet-aux==0.0.10",
    "diffusers-ipadapter @ git+https://github.com/livepeer/Diffusers_IPAdapter.git@405f87da42932e30bd55ee8dca3ce502d7834a99",
    "mediapipe==0.10.21",
    "insightface==0.7.3",
    # We can't really pin torch version as it depends on CUDA, but we check if it's pre-installed above
    "torch",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ @]+)(?:[!=<>~ @].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["xformers"] = deps_list("xformers")
extras["torch"] = deps_list("torch", "accelerate")
_tensorrt_pkgs = ["protobuf", "onnx", "onnxruntime", "onnxruntime-gpu", "colored", "polygraphy", "onnx-graphsurgeon"]
if not is_mac():
    _tensorrt_pkgs.insert(0, "cuda-python")
extras["tensorrt"] = deps_list(*_tensorrt_pkgs)
extras["controlnet"] = deps_list("onnx-graphsurgeon", "controlnet-aux")
extras["ipadapter"] = deps_list("diffusers-ipadapter", "mediapipe", "insightface")

extras["dev"] = extras["xformers"] + extras["torch"] + extras["tensorrt"] + extras["controlnet"]

install_requires = [
    deps["fire"],
    deps["omegaconf"],
    deps["diffusers"],
    deps["transformers"],
    deps["accelerate"],
    deps["huggingface_hub"],
    deps["Pillow"],
]


setup(
    name="streamdiffusion",
    version="0.1.1",
    description="real-time interactive image generation pipeline",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning diffusion pytorch stable diffusion audioldm streamdiffusion real-time",
    license="Apache 2.0 License",
    author="Aki, kizamimi, ddPn08, Verb, ramune, teftef6220, Tonimono, Chenfeng Xu, Ararat with the help of all our contributors (https://github.com/cumulo-autumn/StreamDiffusion/graphs/contributors)",
    author_email="cumulokyoukai@gmail.com",
    url="https://github.com/cumulo-autumn/StreamDiffusion",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"streamdiffusion": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=list(install_requires),
    extras_require=extras,
)
