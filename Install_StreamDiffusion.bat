@echo off
echo ========================================
echo  StreamDiffusionTD v0.3.1 Installation
echo  Daydream Fork with StreamV2V
echo ========================================

cd /d "D:\Users\alexk\FORKNI\STREAM_DIFFUSION\STREAM_DIFFUSION_LIVEPEER\StreamDiffusion"
cd StreamDiffusion-installer

py -3.11 -m sd_installer --base-folder "D:\Users\alexk\FORKNI\STREAM_DIFFUSION\STREAM_DIFFUSION_LIVEPEER\StreamDiffusion" install --cuda cu128 --no-cache

pause
