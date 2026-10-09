"""PSBL PhotAstrom JAX photometry + astrometry."""
from __future__ import annotations

import jax.numpy as jnp

from bagle.jax.geometry import derive_psbl_photastrom_param1, unpack_base_params
from bagle.jax_physics import (
    _assemble_class_vector,
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    psbl_photometry,
    pspl_astrometry_param1,
)


def joint_loglik_psbl_param1(param_vec, ctx, param_cls):
    """PSBL joint log-likelihood with one origin per filter.

    Parameters
    ----------
    param_vec : array_like
        Fitted cube.
    ctx : JaxJointLikelihoodContext
        Per-filter class slots and data blocks.
    param_cls : type
        Parameter class whose ``fitter_param_names`` match the
        class vector.

    Returns
    -------
    lnL : jax.numpy.ndarray
        Photometric plus astrometric log-likelihood.
    """
    param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
    lnL = 0.0
    for block in ctx.filters:
        # This filter's origin. Shared names are the same in every
        # class vector.
        base = _assemble_class_vector(param_vec, block.class_slots)
        p = unpack_base_params(param_cls.fitter_param_names, base)
        geom = derive_psbl_photastrom_param1(
            p["mLp"],
            p["mLs"],
            p["t0"],
            p["xS0_E"],
            p["xS0_N"],
            p["beta"],
            p["muL_E"],
            p["muL_N"],
            p["muS_E"],
            p["muS_N"],
            p["dL"],
            p["dS"],
            p["sep"],
            p["alpha"],
        )
        (
            u0,
            thetaE_hat,
            tE,
            piE_E,
            piE_N,
            xS0,
            xL0,
            muS,
            muL,
            thetaE_amp,
            piS,
            piL,
            m1,
            m2,
            xL1,
            xL2,
            _mLp,
            _mLs,
        ) = geom
        b_sff = param_vec[
            block.ast.idx_b_sff if block.ast else block.phot.idx_b_sff
        ]
        if block.ast is not None:
            pvec_ast = None
            if block.ast.parallax_vectors is not None:
                pvec_ast = jnp.asarray(
                    block.ast.parallax_vectors, dtype=jnp.float64
                )
            pos = pspl_astrometry_param1(
                jnp.asarray(block.ast.t, dtype=jnp.float64),
                p["t0"],
                xS0,
                xL0,
                muS,
                muL,
                thetaE_amp,
                b_sff,
                parallax_vectors=pvec_ast,
                piS=piS,
                piL=piL,
            )
            lnL = lnL + block.ast.weight * gaussian_astrometry_log_likelihood_sum(
                pos,
                block.ast.x_obs,
                block.ast.y_obs,
                block.ast.x_err,
                block.ast.y_err,
            )
        if block.phot is not None:
            mag_src = param_vec[block.phot.idx_mag_src]
            pvec_phot = None
            if block.phot.parallax_vectors is not None:
                pvec_phot = jnp.asarray(
                    block.phot.parallax_vectors, dtype=jnp.float64
                )
            mag_model = psbl_photometry(
                jnp.asarray(block.phot.t, dtype=jnp.float64),
                p["t0"],
                tE,
                u0,
                thetaE_hat,
                xL1,
                xL2,
                m1,
                m2,
                mag_src,
                b_sff=b_sff,
                parallax_vectors=pvec_phot,
                piE_E=piE_E,
                piE_N=piE_N,
            )
            lnL = lnL + block.phot.weight * gaussian_log_likelihood_sum(
                mag_model, block.phot.mag_obs, block.phot.mag_err
            )
    return lnL


def psbl_astrometry_from_model(model, t, filt_idx, pvec):
    """PSBL centroid for one filter.

    Parameters
    ----------
    model : object
        Model instance. ``xS0`` and ``xL0`` are ``(n_filters, 2)``.
    t : array_like
        Observation times (MJD).
    filt_idx : int
        0-based filter.
    pvec : array_like or None
        Parallax table for this filter's observer.

    Returns
    -------
    pos : numpy.ndarray
        East/North centroid, shape ``(n_times, 2)``.
    """
    import numpy as np

    # One East/North pair. The kernel reshape expects shape (2,).
    pos = pspl_astrometry_param1(
        jnp.asarray(t, dtype=jnp.float64),
        float(model.t0),
        jnp.asarray(model.xS0[filt_idx], dtype=jnp.float64),
        jnp.asarray(model.xL0[filt_idx], dtype=jnp.float64),
        jnp.asarray(model.muS, dtype=jnp.float64),
        jnp.asarray(model.muL, dtype=jnp.float64),
        float(model.thetaE_amp),
        float(model.b_sff[filt_idx]),
        parallax_vectors=pvec,
        piS=float(model.piS),
        piL=float(model.piL),
    )
    return np.asarray(pos, dtype=np.float64)
