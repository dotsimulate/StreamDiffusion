import platform
from typing import Literal, Optional

import fire
from packaging.version import Version

from ..pip_utils import get_cuda_major, is_installed, run_pip, version


def install(cu: Optional[Literal["11", "12"]] = get_cuda_major()):
    if cu not in ("11", "12"):
        raise RuntimeError("CUDA major version not detected. Pass --cu 11 or --cu 12 explicitly.")

    print("Installing TensorRT requirements...")

    min_trt_version = Version("10.16.0") if cu == "12" else Version("9.0.0")
    trt_version = version("tensorrt")
    if trt_version and trt_version < min_trt_version:
        run_pip("uninstall -y tensorrt")

    cudnn_package, trt_package = (
        ("nvidia-cudnn-cu12==9.7.1.26", "tensorrt==10.16.1.11")
        if cu == "12"
        else ("nvidia-cudnn-cu11==8.9.7.29", "tensorrt==9.0.1.post11.dev4")
    )
    if not is_installed(trt_package):
        run_pip(f"install {cudnn_package} --no-cache-dir")
        run_pip(f"install --extra-index-url https://pypi.nvidia.com {trt_package} --no-cache-dir")

    if not is_installed("polygraphy"):
        run_pip("install polygraphy==0.49.26 --extra-index-url https://pypi.ngc.nvidia.com")
    if not is_installed("onnx_graphsurgeon"):
        run_pip("install onnx-graphsurgeon==0.6.1 --extra-index-url https://pypi.ngc.nvidia.com")
    if platform.system() == "Windows" and not is_installed("pywin32"):
        run_pip("install pywin32==311")
    if platform.system() == "Windows" and not is_installed("triton"):
        run_pip("install triton-windows==3.4.0.post21")

    # ONNX stack aligned with FLUX for TRT 10.16:
    #   - onnx 1.19.1 (IR 11); modelopt's FLOAT4E2M1 support landed in 1.18 and stays in 1.19
    #   - onnx-gs 0.6.1 no longer needs float32_to_bfloat16 (previously forced onnx==1.18)
    #   - onnxruntime-gpu 1.24.4 supports IR 11; never co-install CPU onnxruntime (shared files conflict)
    #   - onnxoptimizer/onnxslim/onnxscript pair with the onnxoptimizer.optimize_from_path pipeline
    run_pip(
        "install onnx==1.19.1 onnxruntime-gpu==1.24.4 onnxoptimizer==0.4.2 onnxslim==0.1.91 onnxscript==0.6.2 --no-cache-dir"
    )

    # FP8 quantization dependencies (CUDA 12 only). Pin modelopt==0.43.0 and skip its [onnx] extra:
    # an unbounded modelopt floats to 0.45.0 (force-upgrades onnx to 1.21.0 -> breaks FP8 quant), and
    # the [onnx] extra downgrades onnxruntime-gpu off our onnx==1.19.1 / onnxruntime-gpu==1.24.4 pins
    # (installed just above). Enumerate the extra's remaining deps; onnxslim/onnxscript (above),
    # onnx-graphsurgeon/polygraphy (above) are already satisfied.
    if cu == "12":
        run_pip(
            "install nvidia-modelopt==0.43.0 "
            "cppimport lief ml_dtypes onnxconverter-common~=1.16.0 "
            "cupy-cuda12x==13.6.0 numpy==1.26.4 --no-cache-dir"
        )


if __name__ == "__main__":
    fire.Fire(install)
