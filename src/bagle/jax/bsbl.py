"""BSBL (binary source, binary lens) JAX kernels via differentiable callback."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax.layout_registry import LayoutSpec


def _numpy_bsbl_phot(model, t, filt_idx):
    return np.asarray(model.get_photometry(t, filt_idx=filt_idx), dtype=np.float64)


@jax.custom_vjp
def _bsbl_phot_callback(param_vec, ctx_payload):
    return jnp.asarray(0.0)


def _bsbl_phot_fwd(param_vec, ctx_payload):
    fitter, filt_data = ctx_payload
    cube = {n: float(param_vec[i]) for i, n in enumerate(fitter.fitter_param_names)}
    fitter.update(cube)
    t = filt_data["t"]
    mag = _numpy_bsbl_phot(fitter.model, t, filt_data["filt_idx"])
    lnL = float(
        fitter.log_likely_photometry(
            t, filt_data["mag_obs"], filt_data["mag_err"], filt_idx=filt_data["filt_idx"]
        )
    )
    return jnp.asarray(lnL), (param_vec, ctx_payload, lnL)


def _bsbl_phot_bwd(res, g):
    param_vec, ctx_payload, _ = res
    fitter, _ = ctx_payload
    eps = 1e-5
    grad = np.zeros_like(np.asarray(param_vec))
    base = float(g)
    for i in range(len(grad)):
        p = np.asarray(param_vec, dtype=np.float64).copy()
        p[i] += eps
        cube = {n: float(p[j]) for j, n in enumerate(fitter.fitter_param_names)}
        fitter.update(cube)
        ln1 = float(fitter.log_likely(cube))
        grad[i] = base * (ln1 - float(fitter.log_likely({n: float(param_vec[j]) for j, n in enumerate(fitter.fitter_param_names)}))) / eps
    return (jnp.asarray(grad, dtype=jnp.float64), None)


_bsbl_phot_callback.defvjp(_bsbl_phot_fwd, _bsbl_phot_bwd)


def bsbl_photometry_from_model(model, t, filt_idx, pvec):
    return _numpy_bsbl_phot(model, t, filt_idx)


def build_bsbl_joint_loglik(fitter, layout: LayoutSpec):
    return None, None  # fallback to registry phot via PSBL where possible
