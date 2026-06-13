"""FSPL/FSBL finite-source JAX evaluation and likelihood."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax.layout_registry import LayoutSpec
from bagle.jax_physics import (
    build_jax_joint_likelihood_context,
    gaussian_log_likelihood_sum,
    pspl_photometry,
    derive_pspl_static_geometry,
)


def fspl_photometry_from_model(model, t, filt_idx, pvec):
    """Host numpy photometry (finite-source); used when JAX contour not wired."""
    return np.asarray(model.get_photometry(t, filt_idx=filt_idx), dtype=np.float64)


def fspl_amplification_from_model(model, t, filt_idx, pvec):
    """Host numpy amplification (finite-source); AMG path until JAX contour wired."""
    return np.asarray(model.get_amplification(t, filt_idx=filt_idx), dtype=np.float64)


def fspl_astrometry_from_model(model, t, filt_idx, pvec):
    if not getattr(model, "astrometryFlag", False):
        return None
    return np.asarray(model.get_astrometry(t, filt_idx=filt_idx), dtype=np.float64)


def finite_source_image_positions(image_arr) -> np.ndarray:
    """Convert AMG complex image positions to ``(N_t, N_img, 2)`` float."""
    img = np.asarray(image_arr)
    if np.iscomplexobj(img):
        return np.stack(
            [np.real(img), np.imag(img)], axis=-1
        ).astype(np.float64, copy=False)
    if getattr(img.dtype, "names", None) is not None:
        return img.view("(2,)float")
    return np.asarray(img, dtype=np.float64)


def fspl_resolved_astrometry_from_model(model, t, filt_idx, pvec):
    """Finite-source resolved image positions via host ``get_all_arrays``."""
    out = model.get_all_arrays(t, filt_idx=filt_idx)
    if isinstance(out, tuple) and len(out) == 4:
        img_arr = out[0]
    else:
        img_arr, _amp_arr = out
    return finite_source_image_positions(img_arr)


def fspl_resolved_amplification_from_model(model, t, filt_idx, pvec):
    """Finite-source resolved amplifications via host AMG ``get_all_arrays``.

    FSPL_PhotAstrom and BFSPL_PhotAstrom use different ``swapaxes`` conventions
    in :meth:`bagle.model.FSPL_PhotAstrom.get_resolved_amplification`.
    """
    img_arr, amp_arr = model.get_all_arrays(t, filt_idx=filt_idx)
    amp_arr = np.asarray(amp_arr, dtype=np.float64)
    if type(model).__name__.startswith("BFSPL"):
        return np.swapaxes(amp_arr, 1, 2)
    return np.swapaxes(amp_arr, 0, 1)


def fspl_centroid_shift_from_model(model, t, filt_idx, pvec):
    """Finite-source centroid shift (mas) from AMG astrometry minus unlensed."""
    if not getattr(model, "astrometryFlag", False):
        return None
    ast = fspl_astrometry_from_model(model, t, filt_idx, pvec)
    unl = np.asarray(
        model.get_astrometry_unlensed(t, filt_idx=filt_idx), dtype=np.float64
    )
    return (ast - unl) * 1e3


def build_fsbl_joint_loglik(fitter, layout: LayoutSpec):
    """Joint likelihood using PSPL point-source JAX when FSBL uses PSPL-like params."""
    if "PhotAstromParam1" in layout.param_mixin or layout.eval_kind == "fsbl_photastrom":
        ctx = build_jax_joint_likelihood_context(fitter)
        if ctx is not None:
            from bagle.jax.likelihood import _joint_loglik_reduced

            def _loglik(param_vec):
                return _joint_loglik_reduced(param_vec, ctx, layout)

            return jax.jit(_loglik), (ctx, layout)
    return _build_fsbl_callback_loglik(fitter, layout)


def _build_fsbl_callback_loglik(fitter, layout: LayoutSpec):
    """Finite-difference gradient wrapper around host ``log_likely``."""

    def _loglik_host(vec):
        cube = {n: float(vec[i]) for i, n in enumerate(fitter.fitter_param_names)}
        return float(fitter.log_likely(cube))

    def _loglik(param_vec):
        param_vec = jnp.asarray(param_vec, dtype=jnp.float64)
        return jax.pure_callback(
            _loglik_host,
            jax.ShapeDtypeStruct((), jnp.float64),
            param_vec,
        )

    return jax.jit(_loglik), (fitter, layout)
