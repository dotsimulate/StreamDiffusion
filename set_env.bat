@echo off
:: StreamDiffusionTD Runtime Environment Variables
:: Called automatically by Start_StreamDiffusion.bat if this file exists.
:: Edit values here to tune GPU memory and CUDA behavior.

:: Reduce CUDA memory fragmentation (required for large models at 512x512+)
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

:: Lazy CUDA module loading — speeds up startup, reduces VRAM footprint
set CUDA_MODULE_LOADING=LAZY

:: L2 cache persistence (Ampere+ only, compute 8.0+)
:: Set to "0" to disable. Default: "1" (enabled, 64 MB reserved)
set SDTD_L2_PERSIST=1
set SDTD_L2_PERSIST_MB=64

:: HuggingFace offline mode — set to "1" to use cached models only (no downloads)
:: set HF_HUB_OFFLINE=1
:: set TRANSFORMERS_OFFLINE=1

:: Uncomment to override CUDA version detected by setup.py (e.g., for CI)
:: set STREAMDIFFUSION_CUDA_VERSION=12.8
