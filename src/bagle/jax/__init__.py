"""JAX differentiable likelihoods for BAGLE fitters."""

from bagle.jax.likelihood import build_jax_loglik_fn, supports_jax_loglik_for_fitter

# Back-compat alias
supports_jax_loglik = supports_jax_loglik_for_fitter

__all__ = [
    "supports_jax_loglik",
    "supports_jax_loglik_for_fitter",
    "build_jax_loglik_fn",
]
