# FP8 IP-Adapter calibration images

`td_config.yaml`'s `fp8_calibration_style_image` points at this directory. At build time
`wrapper.py`'s `_resolve_fp8_calibration_dir` redirects it to a subfolder keyed on the live
`ipadapters[0].type`, not on the folder you see it land in day to day:

- `general/` — `type: regular` or `type: plus`. Both encode through CLIP, so they want
  identical inputs — any subject works.
- `faces/` — `type: faceid` (or FaceID-Plus/v2). Encoded through InsightFace/ArcFace, which
  raises on any image with no detectable face — **real photographic faces only**.

If the resolved subfolder is missing or has no recognized image in it, the flat directory here
is used as a fallback (a warning names the missing folder), and any image the adapter rejects
at encode time is skipped individually — see `_encode_fp8_calibration_images` in `wrapper.py`.

Recognized extensions: `.png .jpg .jpeg .bmp .webp` (`fp8_quantize._CALIBRATION_IMAGE_EXTENSIONS`).
This file isn't one of them, so it's neither loaded nor hashed into the `--ci<hash>` cache tag.
Directory listing is one level deep only — nested subfolders are invisible to the loader/hasher.

This repo intentionally ships no sample images here — drop your own `general/` and `faces/`
calibration references locally; both subfolders are gitignored by default (`*.png` etc.) so
your images stay local-only.
