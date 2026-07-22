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
2. Add docstring with examples.
3. Comment small chunks of code.

## Architecture

- **`bagle.model`**: NumPy reference physics (`origin/main`); used by fitter, fake_data, plots.
- **`bagle.model_jax`**: JAX-first implementation; forward methods call
  `jax_physics` kernels. JAX fitter likelihoods call explicit Param-mixin
  methods via `bagle.model_fitter_jax.build_explicit_jax_loglik_fn`
  (also exposed as `jax_physics.build_jax_loglik_fn`).
- **`bagle.model_fitter_jax`**: MultiNest, PyMC, and NumPyro solvers.
  `MicrolensSolverNumPyro` supports `sampler='nuts'` (NumPyro NUTS) or
  `sampler='jaxns'` (direct JAXNS). JAXNS extras: `pip install 'bagle[jaxns]'`
  (needs `jaxns` + `tensorflow-probability`). On Python 3.14, jaxns 2.6.x may
  fail to import due to a typing bug.
- Follow the repository pattern
