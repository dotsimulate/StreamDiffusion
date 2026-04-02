# Contributing to StreamDiffusion Livepeer

Thank you for your interest in contributing to this project!

## Development Setup

### Prerequisites

- NVIDIA GPU with CUDA 11.8+ support
- Python 3.10+
- NVIDIA CUDA Toolkit 11.8 or 12.x
- (Optional) TensorRT 8.6+ for acceleration

### Installation

```bash
# Clone the repository
git clone https://github.com/forkni/StreamDiffusion-Livepeer.git
cd StreamDiffusion-Livepeer

# Install PyTorch with CUDA (adjust version for your CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install the package
pip install -e ".[xformers]"

# Install TensorRT support (optional)
pip install -e ".[tensorrt]"
```

### Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

Or use the git hooks script:

```bash
../Scripts/git/install_hooks.sh
```

## Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

**Key style rules:**
- Line length: 119 characters (see `pyproject.toml`)
- Double quotes for strings
- Import sorting enforced (isort-compatible)

## Branch Strategy

- `main` — stable, production-ready
- `development` — active development, feature branches merge here
- Feature branches: `feature/your-feature-name`
- Bug fixes: `fix/issue-description`

## Pull Request Process

1. Create a branch from `development`
2. Make your changes
3. Run lint: `ruff check . && ruff format --check .`
4. Commit using conventional format (see below)
5. Push and open a PR against `development`
6. Wait for Claude Code AI review

### Commit Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add SDXL ControlNet support
fix: resolve TensorRT engine loading on Windows
perf: optimize CUDA memory allocation for batch processing
docs: update installation guide for CUDA 12
chore: update diffusers dependency to 0.21
```

Prefixes: `feat`, `fix`, `perf`, `docs`, `chore`, `test`, `refactor`, `style`

### Git Automation Scripts

Enhanced commit workflow with validation:

```bash
# Standard commit (validates local-only files, checks lint)
../Scripts/git/commit_enhanced.sh "feat: your feature"

# Skip lint check (faster)
../Scripts/git/commit_enhanced.sh --skip-md-lint "chore: quick fix"

# Use pre-staged files only
git add specific_file.py
../Scripts/git/commit_enhanced.sh --staged-only "fix: specific fix"
```

## CUDA Development Guidelines

- Always include CUDA error checking after kernel launches
- Use `torch.cuda.synchronize()` before timing measurements
- Document GPU memory requirements in docstrings
- Test on multiple GPU architectures when possible (sm_75, sm_86, sm_89)

## TensorRT Guidelines

- Handle TensorRT version compatibility explicitly
- Document minimum TensorRT version requirements
- Avoid engine format changes without migration path

## Local-Only Files

The following files are local-only and must NOT be committed:

- `CLAUDE.md` — AI assistant context
- `MEMORY.md` — Session memory
- `.claude/` — Claude Code configuration

These are blocked by the pre-commit hook and `.gitignore`.

## Getting Help

- Open an issue with the `help wanted` label
- Tag `@claude` in a PR or issue for AI-assisted help

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
