"""
Traced linear / accelerated orbital PSBL likelihoods.

Checks ``PSBL_PhotAstrom_*Orbs_Param1`` ``jax_log_likely_*`` against the host
photometry and astrometry, that the orbit parameters enter the likelihood,
and that gradients are finite.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp

from bagle import jax_physics
from bagle import model_jax as mj


def test_traced_linorbs_nopar_matches_host():
    """noPar LinOrbs traced photometry and astrometry match the host model."""
    model_class = mj.PSBL_PhotAstrom_noPar_LinOrbs_Param1
    t = np.linspace(56985.0, 57015.0, 40)

    # Caustic-crossing event with an orbiting secondary and luminous lenses.
    host = model_class(
        mLp=0.5, mLs=0.05, t0=57000.0,
        xS0_E=[0.001], xS0_N=[-0.002], beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0,
        sep=0.75, alpha=70.0,
        b_sff=[0.7], mag_src=[19.0], dmag_Lp_Ls=[1.5],
    )
    mag_h = host.get_photometry(t)
    pos_h = host.get_astrometry(t)

    params = dict(
        mLp=0.5, mLs=0.05, t0=57000.0, xS0_E=0.001, xS0_N=-0.002, beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0, sep=0.75, alpha=70.0,
        b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5,
    )
    vec = jnp.asarray([params[n] for n in model_class.fitter_param_names])

    # Tiny errors: chi^2 at the host values measures the model difference.
    err = np.full(t.size, 1e-8)

    def phot_lnL(v, e):
        return model_class.jax_log_likely_photometry(
            v, t, mag_h, e, 0.7, 19.0)

    def ast_lnL(v, e):
        return model_class.jax_log_likely_astrometry(
            v, t, pos_h[:, 0], pos_h[:, 1], e, e,
            b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5)

    # Photometry: RMS magnitude difference from the chi^2.
    lnl_max = jax_physics.gaussian_log_likelihood_sum(mag_h, mag_h, err)
    rms_mag = 1e-8 * np.sqrt(
        np.maximum(-2.0 * (phot_lnL(vec, err) - lnl_max), 0.0) / t.size)
    assert rms_mag < 1e-6

    # Astrometry: RMS per-axis position difference (mas).
    lnl_max = jax_physics.gaussian_astrometry_log_likelihood_sum(
        pos_h, pos_h[:, 0], pos_h[:, 1], err, err)
    rms_mas = 1e3 * 1e-8 * np.sqrt(
        np.maximum(-2.0 * (ast_lnL(vec, err) - lnl_max), 0.0) / (2 * t.size))
    assert rms_mas < 1e-6

    # Gradients at realistic errors are finite, and the orbit enters them.
    orbit_idx = np.array([
        model_class.fitter_param_names.index('delta_muL_sec_E'),
        model_class.fitter_param_names.index('delta_muL_sec_N'),
    ])
    for lnL_fn in (phot_lnL, ast_lnL):
        grad = jax.grad(lnL_fn)(vec, err * 1e5)
        assert np.all(np.isfinite(grad))
        assert np.all(grad[orbit_idx] != 0.0)

    return None


def test_traced_linorbs_par_matches_host():
    """Par LinOrbs traced photometry and astrometry match the host model."""
    model_class = mj.PSBL_PhotAstrom_Par_LinOrbs_Param1
    t = np.linspace(56985.0, 57015.0, 40)
    raL, decL = 268.0, -29.0

    host = model_class(
        mLp=0.5, mLs=0.05, t0=57000.0,
        xS0_E=[0.001], xS0_N=[-0.002], beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0,
        sep=0.75, alpha=70.0,
        b_sff=[0.7], mag_src=[19.0], dmag_Lp_Ls=[1.5],
        raL=raL, decL=decL,
    )
    mag_h = host.get_photometry(t)
    pos_h = host.get_astrometry(t)
    pvec = jax_physics.precompute_parallax_vectors(
        raL, decL, t, obs_location='earth')

    params = dict(
        mLp=0.5, mLs=0.05, t0=57000.0, xS0_E=0.001, xS0_N=-0.002, beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0, sep=0.75, alpha=70.0,
        b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5,
    )
    vec = jnp.asarray([params[n] for n in model_class.fitter_param_names])
    err = np.full(t.size, 1e-8)

    def phot_lnL(v, e):
        return model_class.jax_log_likely_photometry(
            v, t, mag_h, e, 0.7, 19.0, parallax_vectors=pvec)

    def ast_lnL(v, e):
        return model_class.jax_log_likely_astrometry(
            v, t, pos_h[:, 0], pos_h[:, 1], e, e,
            b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5, parallax_vectors=pvec)

    lnl_max = jax_physics.gaussian_log_likelihood_sum(mag_h, mag_h, err)
    rms_mag = 1e-8 * np.sqrt(
        np.maximum(-2.0 * (phot_lnL(vec, err) - lnl_max), 0.0) / t.size)
    assert rms_mag < 1e-6

    lnl_max = jax_physics.gaussian_astrometry_log_likelihood_sum(
        pos_h, pos_h[:, 0], pos_h[:, 1], err, err)
    rms_mas = 1e3 * 1e-8 * np.sqrt(
        np.maximum(-2.0 * (ast_lnL(vec, err) - lnl_max), 0.0) / (2 * t.size))
    assert rms_mas < 1e-6

    orbit_idx = np.array([
        model_class.fitter_param_names.index('delta_muL_sec_E'),
        model_class.fitter_param_names.index('delta_muL_sec_N'),
    ])
    for lnL_fn in (phot_lnL, ast_lnL):
        grad = jax.grad(lnL_fn)(vec, err * 1e5)
        assert np.all(np.isfinite(grad))
        assert np.all(grad[orbit_idx] != 0.0)

    return None


def test_traced_accorbs_nopar_matches_host():
    """noPar AccOrbs traced photometry and astrometry match the host model."""
    model_class = mj.PSBL_PhotAstrom_noPar_AccOrbs_Param1
    t = np.linspace(56985.0, 57015.0, 40)

    host = model_class(
        mLp=0.5, mLs=0.05, t0=57000.0,
        xS0_E=[0.001], xS0_N=[-0.002], beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0,
        accLsec_E=2.0, accLsec_N=1.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0,
        sep=0.75, alpha=70.0,
        b_sff=[0.7], mag_src=[19.0], dmag_Lp_Ls=[1.5],
    )
    mag_h = host.get_photometry(t)
    pos_h = host.get_astrometry(t)

    params = dict(
        mLp=0.5, mLs=0.05, t0=57000.0, xS0_E=0.001, xS0_N=-0.002, beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0,
        accLsec_E=2.0, accLsec_N=1.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0, sep=0.75, alpha=70.0,
        b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5,
    )
    vec = jnp.asarray([params[n] for n in model_class.fitter_param_names])
    err = np.full(t.size, 1e-8)

    def phot_lnL(v, e):
        return model_class.jax_log_likely_photometry(
            v, t, mag_h, e, 0.7, 19.0)

    def ast_lnL(v, e):
        return model_class.jax_log_likely_astrometry(
            v, t, pos_h[:, 0], pos_h[:, 1], e, e,
            b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5)

    lnl_max = jax_physics.gaussian_log_likelihood_sum(mag_h, mag_h, err)
    rms_mag = 1e-8 * np.sqrt(
        np.maximum(-2.0 * (phot_lnL(vec, err) - lnl_max), 0.0) / t.size)
    assert rms_mag < 1e-6

    lnl_max = jax_physics.gaussian_astrometry_log_likelihood_sum(
        pos_h, pos_h[:, 0], pos_h[:, 1], err, err)
    rms_mas = 1e3 * 1e-8 * np.sqrt(
        np.maximum(-2.0 * (ast_lnL(vec, err) - lnl_max), 0.0) / (2 * t.size))
    assert rms_mas < 1e-6

    orbit_idx = np.array([
        model_class.fitter_param_names.index('delta_muL_sec_E'),
        model_class.fitter_param_names.index('delta_muL_sec_N'),
        model_class.fitter_param_names.index('accLsec_E'),
        model_class.fitter_param_names.index('accLsec_N'),
    ])
    for lnL_fn in (phot_lnL, ast_lnL):
        grad = jax.grad(lnL_fn)(vec, err * 1e5)
        assert np.all(np.isfinite(grad))
        assert np.all(grad[orbit_idx] != 0.0)

    return None


def test_traced_accorbs_par_matches_host():
    """Par AccOrbs traced photometry and astrometry match the host model."""
    model_class = mj.PSBL_PhotAstrom_Par_AccOrbs_Param1
    t = np.linspace(56985.0, 57015.0, 40)
    raL, decL = 268.0, -29.0

    host = model_class(
        mLp=0.5, mLs=0.05, t0=57000.0,
        xS0_E=[0.001], xS0_N=[-0.002], beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0,
        accLsec_E=2.0, accLsec_N=1.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0,
        sep=0.75, alpha=70.0,
        b_sff=[0.7], mag_src=[19.0], dmag_Lp_Ls=[1.5],
        raL=raL, decL=decL,
    )
    mag_h = host.get_photometry(t)
    pos_h = host.get_astrometry(t)
    pvec = jax_physics.precompute_parallax_vectors(
        raL, decL, t, obs_location='earth')

    params = dict(
        mLp=0.5, mLs=0.05, t0=57000.0, xS0_E=0.001, xS0_N=-0.002, beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0,
        accLsec_E=2.0, accLsec_N=1.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0, sep=0.75, alpha=70.0,
        b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5,
    )
    vec = jnp.asarray([params[n] for n in model_class.fitter_param_names])
    err = np.full(t.size, 1e-8)

    def phot_lnL(v, e):
        return model_class.jax_log_likely_photometry(
            v, t, mag_h, e, 0.7, 19.0, parallax_vectors=pvec)

    def ast_lnL(v, e):
        return model_class.jax_log_likely_astrometry(
            v, t, pos_h[:, 0], pos_h[:, 1], e, e,
            b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5, parallax_vectors=pvec)

    lnl_max = jax_physics.gaussian_log_likelihood_sum(mag_h, mag_h, err)
    rms_mag = 1e-8 * np.sqrt(
        np.maximum(-2.0 * (phot_lnL(vec, err) - lnl_max), 0.0) / t.size)
    assert rms_mag < 1e-6

    lnl_max = jax_physics.gaussian_astrometry_log_likelihood_sum(
        pos_h, pos_h[:, 0], pos_h[:, 1], err, err)
    rms_mas = 1e3 * 1e-8 * np.sqrt(
        np.maximum(-2.0 * (ast_lnL(vec, err) - lnl_max), 0.0) / (2 * t.size))
    assert rms_mas < 1e-6

    orbit_idx = np.array([
        model_class.fitter_param_names.index('delta_muL_sec_E'),
        model_class.fitter_param_names.index('delta_muL_sec_N'),
        model_class.fitter_param_names.index('accLsec_E'),
        model_class.fitter_param_names.index('accLsec_N'),
    ])
    for lnL_fn in (phot_lnL, ast_lnL):
        grad = jax.grad(lnL_fn)(vec, err * 1e5)
        assert np.all(np.isfinite(grad))
        assert np.all(grad[orbit_idx] != 0.0)

    return None


def test_static_offsets_match_static_kernel():
    """Zero orbital motion reproduces the static PSBL traced likelihood."""
    orbit_class = mj.PSBL_PhotAstrom_noPar_LinOrbs_Param1
    static_class = mj.PSBL_PhotAstrom_noPar_Param1
    t = np.linspace(56985.0, 57015.0, 40)

    host = orbit_class(
        mLp=0.5, mLs=0.05, t0=57000.0,
        xS0_E=[0.001], xS0_N=[-0.002], beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=0.0, delta_muL_sec_N=0.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0,
        sep=0.75, alpha=70.0,
        b_sff=[0.7], mag_src=[19.0], dmag_Lp_Ls=[1.5],
    )
    mag_h = host.get_photometry(t)

    orbit_params = dict(
        mLp=0.5, mLs=0.05, t0=57000.0, xS0_E=0.001, xS0_N=-0.002, beta=0.04,
        muL_E=1.0, muL_N=3.0, delta_muL_sec_E=0.0, delta_muL_sec_N=0.0,
        muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0, sep=0.75, alpha=70.0,
        b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5,
    )
    orbit_vec = jnp.asarray(
        [orbit_params[n] for n in orbit_class.fitter_param_names])
    static_vec = jnp.asarray(
        [orbit_params[n] for n in static_class.fitter_param_names])

    # After W1 the orbit models place the primary at +offset like the static
    # kernel, so zero orbital motion matches without an alpha flip.
    err = jnp.full(t.size, 1e-3)
    lnL_orbit = orbit_class.jax_log_likely_photometry(
        orbit_vec, t, mag_h, err, 0.7, 19.0)
    lnL_static = static_class.jax_log_likely_photometry(
        static_vec, t, mag_h, err, 0.7, 19.0)
    assert abs(lnL_orbit - lnL_static) < 1e-6 * abs(lnL_static)

    return None
