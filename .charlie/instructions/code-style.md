# StreamDiffusion Code Style

Charlie reads CLAUDE.md automatically for project context. These are additional rules.

## Rules

- [R1] Follow existing patterns in the codebase — check surrounding code before suggesting changes
- [R2] Ensure ruff lint and format checks pass: `ruff check . && ruff format --check .` (line-length 119)
- [R3] CUDA kernels and device operations must include error checking — never ignore return codes
- [R4] TensorRT engine building/loading code must handle version compatibility explicitly
- [R5] Use type hints for all new public functions and class methods
- [R6] TouchDesigner extension methods must follow the TD callback pattern (onXxx naming)
- [R7] Do not commit CLAUDE.md, MEMORY.md, or .claude/ — these are local-only files
