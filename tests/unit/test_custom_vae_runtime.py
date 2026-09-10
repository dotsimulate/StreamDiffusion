"""Real local VAE loading and precision checks without downloads or engine builds."""
import pytest
import torch
from diffusers import AutoencoderKL

from streamdiffusion.pipeline import StreamDiffusion
from streamdiffusion.wrapper import _load_custom_vae, _resolve_vae


def tiny_kl():
    return AutoencoderKL(
        block_out_channels=(8, 8, 8, 8),
        down_block_types=('DownEncoderBlock2D',) * 4,
        up_block_types=('UpDecoderBlock2D',) * 4,
        norm_num_groups=4, layers_per_block=1, latent_channels=4, force_upcast=True,
    )


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_local_subfolder_and_base_vae_roundtrip(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    model = tiny_kl()
    model.save_pretrained(tmp_path / 'vae')
    loaded = _load_custom_vae(str(tmp_path), device=torch.device(device), dtype=torch.float16)
    assert isinstance(loaded, AutoencoderKL) and loaded.dtype == torch.float32
    selected, name, _ = _resolve_vae(
        None, False, False, torch.device(device), torch.float16, base_vae=loaded,
    )
    assert selected is loaded and name == 'AutoencoderKL'
    stream = object.__new__(StreamDiffusion)
    stream.vae, stream.device, stream.dtype = selected, device, torch.float16
    stream.generator = torch.Generator(device=device).manual_seed(3)
    stream.init_noise = torch.zeros(1, 4, 4, 4, device=device, dtype=torch.float16)
    stream.add_noise = lambda latent, noise, index: latent
    with torch.inference_mode():
        latent = stream.encode_image(torch.zeros(1, 3, 32, 32, device=device, dtype=torch.float16))
        image = stream.decode_image(latent)
    assert latent.dtype == torch.float16 and latent.shape == (1, 4, 4, 4)
    assert image.dtype == torch.float32 and image.shape == (1, 3, 32, 32)
    assert torch.isfinite(image).all()


def test_local_checkpoint_preserves_every_weight(tmp_path):
    model = tiny_kl()
    model.save_pretrained(tmp_path)
    loaded = _load_custom_vae(
        str(tmp_path / 'diffusion_pytorch_model.safetensors'),
        device=torch.device('cpu'), dtype=torch.float16,
    )
    assert isinstance(loaded, AutoencoderKL) and loaded.dtype == torch.float32
    expected, actual = model.state_dict(), loaded.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
