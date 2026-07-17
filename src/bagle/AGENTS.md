# Project Instructions

## Shell and Python environment

- Use the **`astro`** conda environment — not system Python.
- Prefer **`zsh -lic '...'`** so your full `~/.zshenv` runs (including `mamba activate astro`, `PARALLAX_CACHE_DIR`, and your `PYTHONPATH` entries). Non-interactive shells skip most of `~/.zshenv` when `PS1` is empty.
- Set **`PYTHONPATH=src`** for this repo when running BAGLE/tests.

Example:

```bash
zsh -lic 'cd /path/to/bagle && PYTHONPATH=src python -m pytest tests/test_jax_physics.py -q'
```

## Project Structure

The `bagle` package source code is in the `src/` directory, with many
subpackages. Each subpackage typically contains:

- `__init__.py` - Public API exports
- `tests/` directory with pytest tests (repo-level tests live in top-level `tests/`)

## Code Style

### Style Rules

- **Line length**: 88 characters (not PEP8's 79)
- **Indentation**: 4 spaces (no tabs)
- **Import convention**: `import numpy as np` (enforced)
- **Docstrings**: NumPy style

## Documentation

### Docstring Format

Use NumPy docstring style.

### Adding a New Function

1. Add comprehensive tests in `tests/`
2. Add docstring with examples

## Architecture

- **`bagle.model`**: NumPy reference physics (`origin/main`); used by fitter, fake_data, plots.
- **`bagle.model_jax`**: JAX-first implementation; forward methods call `jax_physics` kernels (and family helpers). Fitter likelihoods use `Param.get_params_for_jax` + `bagle.jax.likelihood`.
- Follow the repository pattern
