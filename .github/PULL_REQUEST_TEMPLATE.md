## Summary

<!-- Brief description of changes -->

## Type of Change

- [ ] Bug fix
- [ ] New feature / enhancement
- [ ] Performance improvement
- [ ] CUDA / TensorRT optimization
- [ ] TouchDesigner integration
- [ ] Documentation
- [ ] CI/CD / tooling

## Related Issues

<!-- Closes #issue_number -->

## Testing

<!-- How was this tested? Note GPU requirements explicitly. -->

- [ ] Tested locally
- [ ] CPU-only compatible (no GPU required to test)
- [ ] Requires GPU testing (specify hardware below)

**Test environment** (if GPU required):
- GPU: <!-- e.g. RTX 3090, A100 -->
- CUDA version: <!-- e.g. 11.8 -->
- TensorRT version: <!-- e.g. 8.6 -->
- OS: <!-- e.g. Ubuntu 22.04, Windows 11 -->

## CUDA / TensorRT Impact

- [ ] No CUDA/TensorRT changes
- [ ] Modified CUDA kernels or memory management
- [ ] TensorRT engine building or loading changes
- [ ] Requires specific GPU architecture (specify: <!-- e.g. sm_86 -->)
- [ ] Changes engine serialization format (breaking for existing .engine files)

## TouchDesigner Impact

- [ ] No TD changes
- [ ] Modified TD Python extensions
- [ ] Modified OSC/parameter interface
- [ ] Requires TD version update (specify: <!-- e.g. 2023.11760+ -->)

## Checklist

- [ ] Code follows project style (ruff format, line-length 119)
- [ ] Self-review completed
- [ ] No local-only files included (CLAUDE.md, MEMORY.md, .claude/)
