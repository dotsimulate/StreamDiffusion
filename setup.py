import os
import re
import sys

from setuptools import find_packages, setup


# Copied from pip_utils.py to avoid import
def _check_torch_installed():
    try:
        import torch
    except Exception:
        msg = (
            "Missing required pre-installed packages: torch, torchvision\n"
            "Install the PyTorch CUDA wheels from the appropriate index first, e.g.:\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu12x torch torchvision\n"
            "Replace the index URL and versions to match your CUDA runtime."
        )
        raise RuntimeError(msg)

    if not torch.version.cuda:
        raise RuntimeError(
            "Detected CPU-only PyTorch. Install CUDA-enabled torch/vision/audio before installing this package."
        )


def get_cuda_constraint():
    cuda_version = os.environ.get("STREAMDIFFUSION_CUDA_VERSION") or os.environ.get("CUDA_VERSION")

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

_deps = [
    f"cuda-python{get_cuda_constraint()}",
    "xformers==0.0.30",
    "diffusers @ git+https://github.com/varshith15/diffusers.git@3e3b72f557e91546894340edabc845e894f00922",
    "transformers==4.56.0",
    "accelerate==1.13.0",
    "huggingface_hub==0.35.0",
    "Pillow>=12.2.0",  # CVE-2026-25990: out-of-bounds write in PSD loading; 12.2.0 verified
    "fire==0.7.1",
    "omegaconf==2.3.0",
    "onnx==1.19.1",  # IR 11; modelopt needs FLOAT4E2M1 (added in 1.18); onnx-gs 0.6.1 no longer needs float32_to_bfloat16
    "onnxruntime-gpu==1.24.4",  # TRT EP, supports IR 11; never co-install CPU onnxruntime — shared files conflict
    "onnxoptimizer==0.4.2",
    "onnxslim==0.1.91",
    "onnxscript==0.6.2",
    "polygraphy==0.49.26",
    "protobuf>=4.25.8,<5",  # mediapipe 0.10.21 requires protobuf 4.x; 4.25.8 fixes CVE-2025-4565; CVE-2026-0994 (JSON DoS) accepted risk for local pipeline
    "colored==2.3.2",
    "pywin32==311;sys_platform == 'win32'",
    "onnx-graphsurgeon==0.6.1",
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
extras["tensorrt"] = deps_list(
    "protobuf",
    "cuda-python",
    "onnx",
    "onnxruntime-gpu",
    "onnxoptimizer",
    "onnxslim",
    "onnxscript",
    "colored",
    "polygraphy",
    "onnx-graphsurgeon",
)
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
