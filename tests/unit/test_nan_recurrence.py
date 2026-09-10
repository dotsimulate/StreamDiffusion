"""Recurrent-buffer recovery through production pipeline and ControlNet methods."""
import pytest
import torch
from streamdiffusion.pipeline import StreamDiffusion
from test_nan_guard_latch_recovery import _make_stream, _NanInjectingUnet, _zero_input
from test_cn_cache_decay import _RampingCN, _make_module_with_controlnets, _make_ctx

@pytest.mark.parametrize('bucketed', [True, False])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_kv_fi_cache_ingress_and_recovery(bucketed, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA required')
    s = object.__new__(StreamDiffusion)
    s.frame_idx, s.cache_interval, s.cache_maxframes = 0, 1, 2
    bank = torch.zeros(1, 2, 2, 1, 4, 8, device=device)
    s.kvo_cache = [bank[0]]
    s.fio_cache = [torch.zeros(2, 1, 4, 8, device=device)]
    s._kvo_buckets = [bank] if bucketed else None
    s._kvo_outputs_by_bucket = [[0]] if bucketed else None
    for poison in [False, True, False, False]:
        kv = torch.full((2, 1, 1, 4, 8), float('nan') if poison else 2., device=device)
        fi = torch.full((1, 1, 4, 8), float('inf') if poison else 3., device=device)
        s.update_kvo_cache([kv], [fi])
        assert torch.isfinite(s.kvo_cache[0]).all()
        assert torch.isfinite(s.fio_cache[0]).all()
    assert (s.kvo_cache[0] == 2).all() and (s.fio_cache[0] == 3).all()



def test_recurrent_latents_recover_independent_of_model_pred():
    s = _make_stream([22, 36], _NanInjectingUnet(nan_on_call=-1))
    s.stock_noise.fill_(float('nan'))
    s.x_t_latent_buffer.fill_(float('inf'))
    for _ in range(3):
        out = s.predict_x0_batch(_zero_input(s))
        assert torch.isfinite(out).all()
        assert torch.isfinite(s.stock_noise).all()
        assert torch.isfinite(s.x_t_latent_buffer).all()



def test_controlnet_ema_recovers_poisoned_history():
    module = _make_module_with_controlnets(_RampingCN())
    module.set_cn_cache_decay(0.5)
    hook = module.build_unet_hook()
    hook(_make_ctx())
    for tensor in module._cn_ema_down + [module._cn_ema_mid]: tensor.fill_(float('nan'))
    for _ in range(3):
        result = hook(_make_ctx())
        assert all(torch.isfinite(t).all() for t in result.down_block_additional_residuals)
        assert torch.isfinite(result.mid_block_additional_residual).all()
    assert result.mid_block_additional_residual.abs().sum() > 0



@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_busy_cuda_guard_preserves_transient_verdict():
    from streamdiffusion.utils.nan_guard import NanGuard
    g = NanGuard('busy test')
    clean = torch.ones(32768, device='cuda')
    g.sanitize_(clean)
    torch.cuda.synchronize()
    poison = torch.full_like(clean, float('nan'))
    seen = [g.sanitize_(poison)]
    for _ in range(8): seen.append(g.sanitize_(clean))
    # Readbacks can be several calls behind a busy GPU, but cannot be lost.
    for _ in range(3):
        torch.cuda.synchronize()
        seen.append(g.sanitize_(clean))
    assert any(seen) or g._warned
    assert torch.isfinite(poison).all()
