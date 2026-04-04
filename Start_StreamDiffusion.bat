@echo off
cd /d %~dp0

:: ─── CUDA / PyTorch Performance Tuning ───
:: Prevents memory fragmentation from per-frame torch.cat allocations
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
:: Defers CUDA module loading until first use (~1-5s faster startup)
set CUDA_MODULE_LOADING=LAZY
:: Enables cuDNN v8 graph API for better conv kernel selection (VAE, preprocessors)
set TORCH_CUDNN_V8_API_ENABLED=1
:: Ensures async kernel launches (default=0, but explicit protects against debug leftovers)
set CUDA_LAUNCH_BLOCKING=0
:: Caches compiled Triton kernels to disk (eliminates 30-60s JIT warmup on restart)
set TORCHINDUCTOR_FX_GRAPH_CACHE=1

if exist venv (
    call venv\Scripts\activate.bat
    venv\Scripts\python.exe streamdiffusionTD\td_main.py
) else (
    call .venv\Scripts\activate.bat
    .venv\Scripts\python.exe streamdiffusionTD\td_main.py
)
pause
