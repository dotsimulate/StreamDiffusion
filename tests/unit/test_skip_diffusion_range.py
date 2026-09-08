"""
Regression coverage for `_process_skip_diffusion`'s image-preprocessing-hook range.

Image preprocessing hooks ("image_pre") are a documented [-1,1]-in/[-1,1]-out contract
(custom_processors/sdtd_fx/README.md: "Receives [-1, 1] tensors. Convert to [0, 1]
internally, convert back to [-1, 1] before returning."). StreamDiffusion.__call__'s main
txt2img/img2img path honours this: image_processor.preprocess(..., do_normalize=True)
yields [-1,1], which is fed straight into _apply_image_preprocessing_hooks and then
straight into encode_image with no conversion at either boundary (pipeline.py:1504-1512).

_process_skip_diffusion previously disagreed with itself: the PIL/str branch called
_denormalize_on_gpu before the hook (handing it [0,1]) while the torch.Tensor branch
passed its input straight through (handing it [-1,1] as-is) — so the *same logical input*
produced different values at the hook depending on whether it arrived as a PIL.Image or a
pre-normalized tensor. A second bug compounded it: the code then always ran
_normalize_on_gpu on the hook's output before the postprocessing hooks, which is only
correct if the hook received [0,1] — for the (correct) tensor-branch case this silently
double-transformed an already-[-1,1] result.

Both tests are CPU-only and model-free, using the object.__new__ shell pattern from
test_safety_checker.py / test_wrapper_exception_hygiene.py — no diffusers model is loaded.
"""

import torch

from streamdiffusion.wrapper import StreamDiffusionWrapper


def _make_wrapper(recorded_pre_hook_input, recorded_post_hook_input):
    w = object.__new__(StreamDiffusionWrapper)
    w.mode = "img2img"
    w.device = "cpu"
    w.dtype = torch.float32
    w.output_type = "pt"

    class _StreamStub:
        def _apply_image_preprocessing_hooks(self, x):
            recorded_pre_hook_input.append(x.clone())
            return x  # identity: whatever range comes in, comes back out

        def _apply_image_postprocessing_hooks(self, x):
            recorded_post_hook_input.append(x.clone())
            return x

    w.stream = _StreamStub()
    w._apply_safety_checker = lambda t: t
    w.postprocess_image = lambda t, output_type=None: t
    return w


def test_pil_and_tensor_inputs_reach_preprocessing_hook_with_same_range():
    """A PIL-derived [-1,1] tensor and an equivalent pre-normalized tensor input
    must hit _apply_image_preprocessing_hooks with the SAME values."""
    known = torch.tensor([[[[-1.0, -0.5, 0.0, 0.5, 1.0]]]], dtype=torch.float32)

    pre_pil, post_pil = [], []
    w_pil = _make_wrapper(pre_pil, post_pil)
    w_pil.preprocess_image = lambda image: known.clone()
    w_pil._process_skip_diffusion(image="fake_path.png")

    pre_tensor, post_tensor = [], []
    w_tensor = _make_wrapper(pre_tensor, post_tensor)
    w_tensor._process_skip_diffusion(image=known.clone())

    assert torch.allclose(pre_pil[0], known)
    assert torch.allclose(pre_tensor[0], known)
    assert torch.allclose(pre_pil[0], pre_tensor[0])


def test_hook_output_reaches_postprocessing_hook_unmodified():
    """The [-1,1] hook contract means _process_skip_diffusion must NOT rescale the
    preprocessing hook's output before handing it to the postprocessing hooks."""
    known = torch.tensor([[[[-1.0, -0.5, 0.0, 0.5, 1.0]]]], dtype=torch.float32)

    pre, post = [], []
    w = _make_wrapper(pre, post)
    w._process_skip_diffusion(image=known.clone())

    # Hook is identity, so whatever it received is what it returned. If a stray
    # normalize/denormalize crept back in, this would no longer match `known`.
    assert torch.allclose(post[0], known)
