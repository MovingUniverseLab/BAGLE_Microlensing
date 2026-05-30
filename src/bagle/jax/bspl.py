"""BSPL (binary source, point lens) JAX kernels."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax.layout_registry import LayoutSpec
from bagle.jax_physics import (
    _fitter_weight,
    _param_index,
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    mag2flux_jax,
    flux2mag_jax,
    precompute_parallax_vectors,
    pspl_amplification_from_u,
    einstein_source_position,
    derive_pspl_static_geometry,
    build_jax_joint_likelihood_context,
)


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


def bspl_photometry_from_model(model, t, filt_idx, pvec):
    u0_pri, thetaE_hat, _ = derive_pspl_static_geometry(
        float(model.u0_amp_pri[filt_idx]),
        float(model.piE[0]),
        float(model.piE[1]),
    )
    u0_sec, _, _ = derive_pspl_static_geometry(
        float(model.u0_amp_sec[filt_idx]),
        float(model.piE[0]),
        float(model.piE[1]),
    )
    mag = bspl_photometry_jax(
        jnp.asarray(t, dtype=jnp.float64),
        float(model.t0_pri),
        float(model.t0_sec),
        float(model.tE),
        u0_pri,
        u0_sec,
        thetaE_hat,
        float(model.mag_src_pri[filt_idx]),
        float(model.mag_src_sec[filt_idx]),
        float(model.b_sff[filt_idx]),
        pvec=pvec,
        piE_E=float(model.piE[0]),
        piE_N=float(model.piE[1]),
    )
    return np.asarray(mag, dtype=np.float64)


def bspl_astrometry_from_model(model, t, filt_idx, pvec):
    return None  # use numpy path until full BSPL astrometry port


def build_bspl_joint_loglik(fitter, layout: LayoutSpec):
    ctx = build_jax_joint_likelihood_context(fitter)
    if ctx is None:
        return None, None

    def _loglik(param_vec):
        param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
        base = param_vec[jnp.array(ctx.base_indices, dtype=jnp.int32)]
        # BSPL PhotAstromParam1: t0_pri, u0_amp_pri, t0_sec, u0_amp_sec, tE, piE_E, piE_N, ...
        names = layout.base_fitter_names
        p = {names[i]: base[i] for i in range(len(names))}
        u0_pri, thetaE_hat, _ = derive_pspl_static_geometry(p["u0_amp_pri"], p["piE_E"], p["piE_N"])
        u0_sec, _, _ = derive_pspl_static_geometry(p["u0_amp_sec"], p["piE_E"], p["piE_N"])
        lnL = 0.0
        for block in ctx.filters:
            b_sff = param_vec[block.ast.idx_b_sff if block.ast else block.phot.idx_b_sff]
            if block.phot is not None:
                mag_pri = param_vec[block.phot.idx_mag_src]
                # secondary mag index follows primary in fitter names
                idx_sec = block.phot.idx_mag_src + 1
                mag_sec = param_vec[idx_sec]
                pvec = (
                    None
                    if block.phot.parallax_vectors is None
                    else jnp.asarray(block.phot.parallax_vectors, dtype=jnp.float64)
                )
                mag_model = bspl_photometry_jax(
                    jnp.asarray(block.phot.t, dtype=jnp.float64),
                    p["t0_pri"],
                    p["t0_sec"],
                    p["tE"],
                    u0_pri,
                    u0_sec,
                    thetaE_hat,
                    mag_pri,
                    mag_sec,
                    b_sff,
                    pvec=pvec,
                    piE_E=p["piE_E"],
                    piE_N=p["piE_N"],
                )
                lnL = lnL + block.phot.weight * gaussian_log_likelihood_sum(
                    mag_model, block.phot.mag_obs, block.phot.mag_err
                )
        return lnL

    return jax.jit(_loglik), (ctx, layout)
