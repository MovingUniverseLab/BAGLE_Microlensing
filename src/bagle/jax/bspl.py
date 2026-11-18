"""BSPL (binary source, point lens) JAX kernels."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax_physics import (
    _fitter_weight,
    _param_index,
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    mag2flux_jax,
    flux2mag_jax,
    precompute_parallax_vectors,
    pspl_amplification_from_u,
    pspl_resolved_amplification_from_u,
    einstein_source_position,
    derive_pspl_static_geometry,
)


def _filt_scalar(model, name: str, filt_idx: int) -> float:
    val = getattr(model, name)
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"empty parameter {name!r}")
    return float(arr[filt_idx]) if arr.size > filt_idx else float(arr[0])


def _bspl_u_dual(t, t0_pri, t0_sec, tE, u0_pri, u0_sec, thetaE_hat, pvec, piE_E, piE_N):
    u1 = einstein_source_position(
        t, t0_pri, tE, u0_pri, thetaE_hat, parallax_vectors=pvec, piE_E=piE_E, piE_N=piE_N
    )
    u2 = einstein_source_position(
        t, t0_sec, tE, u0_sec, thetaE_hat, parallax_vectors=pvec, piE_E=piE_E, piE_N=piE_N
    )
    return u1, u2


def bspl_photometry_jax(
    t,
    t0_pri,
    t0_sec,
    tE,
    u0_pri,
    u0_sec,
    thetaE_hat,
    mag_pri,
    mag_sec,
    b_sff,
    pvec=None,
    piE_E=None,
    piE_N=None,
):
    u1, u2 = _bspl_u_dual(t, t0_pri, t0_sec, tE, u0_pri, u0_sec, thetaE_hat, pvec, piE_E, piE_N)
    A1 = pspl_amplification_from_u(u1)
    A2 = pspl_amplification_from_u(u2)
    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    flux = f1 * A1 + f2 * A2 + (f1 + f2) * (1.0 - b_sff) / b_sff
    return flux2mag_jax(flux)


def _bspl_times(model):
    """
    Primary and secondary reference epochs for photometry.

    Option A uses a single epoch (primary CA); ``get_t0_sec`` remains a
    derived diagnostic for the secondary's own closest approach.

    Parameters
    ----------
    model : object
        BSPL-like model with ``t0_pri``.

    Returns
    -------
    t0_pri, t0_sec : float
        Both equal to ``t0_pri`` under Option A.
    """
    t0_pri = float(model.t0_pri)
    return t0_pri, t0_pri


def _bspl_u0_pair(model, filt_idx: int):
    """
    Primary and secondary ``u0`` vectors at ``t0_pri``.

    Parameters
    ----------
    model : object
        BSPL-like model.
    filt_idx : int
        Unused; kept for call-site compatibility.

    Returns
    -------
    u0_pri, u0_sec : jnp.ndarray, shape = (2,)
        Lens-relative positions at ``t0_pri`` (Einstein radii).
    thetaE_hat : jnp.ndarray, shape = (2,)
        Unit vector along relative proper motion.
    """
    thetaE_hat = jnp.asarray(model.thetaE_hat, dtype=jnp.float64).reshape(2)

    # Prefer snapshot vectors already stored on the model (Option A).
    if hasattr(model, "u0_pri") and hasattr(model, "u0_sec"):
        u0_pri = jnp.asarray(model.u0_pri, dtype=jnp.float64).reshape(2)
        u0_sec = jnp.asarray(model.u0_sec, dtype=jnp.float64).reshape(2)
        return u0_pri, u0_sec, thetaE_hat

    # Reconstruct the full static offset from sep / angle.
    u0_pri_j, thetaE_hat, _ = derive_pspl_static_geometry(
        float(getattr(model, "u0_amp_pri", model.u0_amp)),
        float(model.piE[0]),
        float(model.piE[1]),
    )
    # Phot-only: sep is already in Einstein radii. PhotAstrom: mas.
    if hasattr(model, "thetaE_amp"):
        sep = float(model.sep) / float(model.thetaE_amp)
    else:
        sep = float(model.sep)
    alpha = float(getattr(model, "phi_rho1_rad", getattr(model, "alpha_rad", 0.0)))
    sep_vec = sep * np.array([np.sin(alpha), np.cos(alpha)])
    u0_sec_j = jnp.asarray(u0_pri_j, dtype=jnp.float64) + jnp.asarray(
        sep_vec, dtype=jnp.float64)

    return u0_pri_j, u0_sec_j, thetaE_hat


def bspl_photometry_from_model(model, t, filt_idx, pvec):
    u0_pri, u0_sec, thetaE_hat = _bspl_u0_pair(model, filt_idx)
    t0_pri, t0_sec = _bspl_times(model)
    mag = bspl_photometry_jax(
        jnp.asarray(t, dtype=jnp.float64),
        t0_pri,
        t0_sec,
        float(model.tE),
        u0_pri,
        u0_sec,
        thetaE_hat,
        _filt_scalar(model, "mag_src_pri", filt_idx),
        _filt_scalar(model, "mag_src_sec", filt_idx),
        _filt_scalar(model, "b_sff", filt_idx),
        pvec=pvec,
        piE_E=float(model.piE[0]),
        piE_N=float(model.piE[1]),
    )
    return np.asarray(mag, dtype=np.float64)


def bspl_amplification_from_model(model, t, filt_idx, pvec):
    u0_pri, u0_sec, thetaE_hat = _bspl_u0_pair(model, filt_idx)
    t0_pri, t0_sec = _bspl_times(model)
    t_j = jnp.asarray(t, dtype=jnp.float64)
    u1, u2 = _bspl_u_dual(
        t_j,
        t0_pri,
        t0_sec,
        float(model.tE),
        u0_pri,
        u0_sec,
        thetaE_hat,
        pvec,
        float(model.piE[0]),
        float(model.piE[1]),
    )
    a1 = pspl_amplification_from_u(u1)
    a2 = pspl_amplification_from_u(u2)
    f1 = mag2flux_jax(_filt_scalar(model, "mag_src_pri", filt_idx))
    f2 = mag2flux_jax(_filt_scalar(model, "mag_src_sec", filt_idx))
    amp = (f1 * a1 + f2 * a2) / (f1 + f2)
    return np.asarray(amp, dtype=np.float64)


def bspl_resolved_amplification_from_model(model, t, filt_idx, pvec):
    """Dual-source ± image amplifications, shape ``(N_t, 2 sources, 2 images)``.

    Uses host ``get_u`` so static and orbital BSPL layouts match NumPy reference.
    """
    u_vec = np.asarray(model.get_u(t, filt_idx=filt_idx), dtype=np.float64)
    u1 = jnp.asarray(u_vec[:, 0, :], dtype=jnp.float64)
    u2 = jnp.asarray(u_vec[:, 1, :], dtype=jnp.float64)
    a1_plus, a1_minus = pspl_resolved_amplification_from_u(u1)
    a2_plus, a2_minus = pspl_resolved_amplification_from_u(u2)
    a_src1 = jnp.stack([a1_plus, a1_minus], axis=1)
    a_src2 = jnp.stack([a2_plus, a2_minus], axis=1)
    amp = jnp.stack([a_src1, a_src2], axis=1)
    return np.asarray(amp, dtype=np.float64)


def bspl_source_astrometry_unlensed_from_model(model, t, filt_idx, pvec):
    """Flux-weighted unlensed source centroid in Einstein radii (BSPL phot-only).

    Uses host ``get_u`` so static and orbital layouts match NumPy reference.
    """
    u_vec = np.asarray(model.get_u(t, filt_idx=filt_idx), dtype=np.float64)
    u1 = jnp.asarray(u_vec[:, 0, :], dtype=jnp.float64)
    u2 = jnp.asarray(u_vec[:, 1, :], dtype=jnp.float64)
    f1 = mag2flux_jax(_filt_scalar(model, "mag_src_pri", filt_idx))
    f2 = mag2flux_jax(_filt_scalar(model, "mag_src_sec", filt_idx))
    u = (u1 * f1 + u2 * f2) / (f1 + f2)
    return np.asarray(u, dtype=np.float64)


def bspl_photastrom_source_astrometry_unlensed_from_model(
    model, t, filt_idx, pvec
):
    """Flux-weighted unlensed source centroid in arcsec (BSPL PhotAstrom).

    Uses host ``get_resolved_source_astrometry_unlensed`` for orbital motion.
    """
    xS_both = np.asarray(
        model.get_resolved_source_astrometry_unlensed(t, filt_idx=filt_idx),
        dtype=np.float64,
    )
    x1 = jnp.asarray(xS_both[:, 0, :], dtype=jnp.float64)
    x2 = jnp.asarray(xS_both[:, 1, :], dtype=jnp.float64)
    f1 = mag2flux_jax(_filt_scalar(model, "mag_src_pri", filt_idx))
    f2 = mag2flux_jax(_filt_scalar(model, "mag_src_sec", filt_idx))
    xS = (x1 * f1 + x2 * f2) / (f1 + f2)
    return np.asarray(xS, dtype=np.float64)


# BSPL PhotAstrom traced astrometry / joint loglik: see jax_swap_plan Step 7.
