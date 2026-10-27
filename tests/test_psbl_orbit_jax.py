"""
Traced linear / accelerated orbital PSBL likelihoods.

Checks ``PSBL_PhotAstrom_{Lin,Acc}Orbs_Param1`` ``jax_log_likely_*`` against
the host photometry and astrometry, that the orbit parameters enter the
likelihood, and that gradients are finite.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp

from bagle import jax_physics
from bagle import model_jax as mj

# Caustic-crossing event with an orbiting secondary and luminous lenses.
KW = dict(mLp=0.5, mLs=0.05, t0=57000.0, xS0_E=0.001, xS0_N=-0.002,
          beta=0.04, muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5,
          delta_muL_sec_N=-1.0, muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0,
          sep=0.75, alpha=70.0, b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5,
          raL=268.0, decL=-29.0)
FILT = ('xS0_E', 'xS0_N', 'b_sff', 'mag_src', 'dmag_Lp_Ls')

# Tiny data errors: the chi^2 of the traced model at the host values
# measures the model difference.
ERR = 1e-8


def _setup(orb, par):
    """
    Host model, packed vector and data for one orbital PSBL class.

    Parameters
    ----------
    orb : str
        ``'LinOrbs'`` or ``'AccOrbs'``.
    par : bool
        Use the parallax class (and parallax vectors).

    Returns
    -------
    cls : type
        Composed model class.
    vec : jnp.ndarray, shape (N_params,)
        Filter-local packed parameters.
    t : ndarray, shape (N_times,)
        Times (MJD).
    mag_h : ndarray, shape (N_times,)
        Host magnitudes.
    pos_h : ndarray, shape (N_times, 2)
        Host centroid (arcsec).
    pvec : ndarray or None
        Parallax vectors for ``t``.
    """
    kw = dict(KW)
    if orb == 'AccOrbs':
        kw.update(accLsec_E=2.0, accLsec_N=1.0)
    if not par:
        kw.pop('raL')
        kw.pop('decL')
    cls = getattr(mj, 'PSBL_PhotAstrom_{}_{}_Param1'.format(
        'Par' if par else 'noPar', orb))
    t = np.linspace(56985.0, 57015.0, 40)

    # Host model (single filter).
    host = cls(**{k: ([v] if k in FILT else v) for k, v in kw.items()})
    mag_h = np.asarray(host.get_photometry(t))
    pos_h = np.asarray(host.get_astrometry(t))

    # Filter-local packed vector and optional parallax vectors.
    vec = jnp.asarray([kw[n] for n in cls.fitter_param_names])
    pvec = None
    if par:
        pvec = jax_physics.precompute_parallax_vectors(
            kw['raL'], kw['decL'], t, obs_location='earth')

    return cls, vec, t, mag_h, pos_h, pvec


@pytest.mark.parametrize("orb,par", [("LinOrbs", False), ("LinOrbs", True),
                                     ("AccOrbs", False), ("AccOrbs", True)])
def test_traced_orbit_matches_host(orb, par):
    """Orbital PSBL traced photometry and astrometry equal the host model."""
    cls, vec, t, mag_h, pos_h, pvec = _setup(orb, par)
    tj = jnp.asarray(t)

    # Photometry: RMS magnitude difference from the chi^2.
    phot_fn = lambda v, e: cls.jax_log_likely_photometry(
        v, tj, jnp.asarray(mag_h), jnp.asarray(e), 0.7, 19.0,
        parallax_vectors=pvec)
    err = np.full(t.size, ERR)
    lnl_max = float(jax_physics.gaussian_log_likelihood_sum(
        jnp.asarray(mag_h), mag_h, err))
    rms_mag = ERR * np.sqrt(max(-2.0 * (float(phot_fn(vec, err)) - lnl_max),
                                0.0) / t.size)
    assert rms_mag < 1e-6

    # Astrometry: RMS per-axis position difference (mas).
    ast_fn = lambda v, e: cls.jax_log_likely_astrometry(
        v, tj, jnp.asarray(pos_h[:, 0]), jnp.asarray(pos_h[:, 1]),
        jnp.asarray(e), jnp.asarray(e), b_sff=0.7, mag_src=19.0,
        dmag_Lp_Ls=1.5, parallax_vectors=pvec)
    lnl_max = float(jax_physics.gaussian_astrometry_log_likelihood_sum(
        jnp.asarray(pos_h), pos_h[:, 0], pos_h[:, 1], err, err))
    rms_mas = 1e3 * ERR * np.sqrt(
        max(-2.0 * (float(ast_fn(vec, err)) - lnl_max), 0.0) / (2 * t.size))
    assert rms_mas < 1e-6

    # Gradients at realistic errors are finite, and the orbit enters them.
    names = list(cls.fitter_param_names)
    orbit_idx = [names.index(n) for n in names
                 if n.startswith(('delta_muL_sec', 'accLsec'))]
    for fn in (phot_fn, ast_fn):
        grad = np.asarray(jax.grad(fn)(vec, err * 1e5))
        assert np.all(np.isfinite(grad))
        assert np.all(grad[orbit_idx] != 0.0)

    return None


def test_static_offsets_match_static_kernel():
    """Zero orbital motion reproduces the static PSBL traced likelihood."""
    cls, vec, t, mag_h, pos_h, pvec = _setup('LinOrbs', False)
    names = list(cls.fitter_param_names)

    # Static vector: the orbit class with delta_muL_sec = 0.
    vec0 = vec.at[names.index('delta_muL_sec_E')].set(0.0)
    vec0 = vec0.at[names.index('delta_muL_sec_N')].set(0.0)
    static = mj.PSBL_PhotAstrom_noPar_Param1
    vec_s = jnp.asarray([vec0[names.index(n)]
                         for n in static.fitter_param_names])

    # Same data, same errors: the orbit and static models agree once the
    # orbit is switched off (up to the orbit models' primary at -offset).
    err = jnp.full(t.size, 1e-3)
    tj = jnp.asarray(t)
    lo = float(cls.jax_log_likely_photometry(vec0, tj, jnp.asarray(mag_h),
                                             err, 0.7, 19.0))
    vec_s = vec_s.at[static.fitter_param_names.index('alpha')].add(180.0)
    ls = float(static.jax_log_likely_photometry(vec_s, tj, jnp.asarray(mag_h),
                                                err, 0.7, 19.0))
    assert abs(lo - ls) < 1e-6 * abs(ls)

    return None
