"""
Traced Keplerian (EllOrbs) PSTL PhotAstrom likelihoods.

Checks ``PSTL_PhotAstrom_{noPar,Par}_EllOrbs_Param{1,2}`` ``jax_log_likely_*``
against the host photometry and astrometry, that the orbit parameters enter
the likelihood with finite gradients, and that the fitter builds the same
log-likelihood on the host and JAX paths.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp

from bagle import jax_physics
from bagle import model_jax as mj
from bagle.model_fitter_jax import MicrolensSolver, build_explicit_jax_loglik_fn

FILT = ('xS0_E', 'xS0_N', 'b_sff', 'mag_src', 'dmag_Lp_Ls', 'dmag_Lp_Lt')

# Orbit elements shared by both parameterizations.
ORB = dict(omega_pri=30.0, omega_sec=60.0, big_omega_sec=20.0,
           big_omega_ter=50.0, i_12=40.0, e_12=0.2, tp_12=56950.0, a_12=3.0,
           i_23=30.0, e_23=0.1, tp_23=56980.0, a_23=2.0)

# Luminous lenses so dmag_Lp_Ls / dmag_Lp_Lt enter the centroid.
FLUX = dict(xS0_E=0.0, xS0_N=0.01, b_sff=0.7, mag_src=18.0,
            dmag_Lp_Ls=2.0, dmag_Lp_Lt=1.0)

KW = {
    1: dict(mLp=10.0, mLs=5.0, mLt=2.0, t0_com=57000.0, beta_com=0.4,
            muL_E=0.0, muL_N=0.0, muS_E=4.0, muS_N=-3.0, dL=2000.0,
            dS=4000.0, **ORB, **FLUX),
    2: dict(t0=57000.0, u0_amp=0.1, tE=100.0, thetaE=5.0, piS=0.25,
            piE_E=0.05, piE_N=0.03, muS_E=4.0, muS_N=-3.0, q_12=0.5,
            q_13=0.2, **ORB, **FLUX),
}

# Tiny data errors: the chi^2 at the host values measures the model gap.
ERR = 1e-8

ORBIT_NAMES = ('omega_pri', 'omega_sec', 'big_omega_sec', 'big_omega_ter',
               'i_12', 'e_12', 'tp_12', 'a_12', 'i_23', 'e_23', 'tp_23',
               'a_23')


def _setup(pnum, par):
    """
    Host model, packed vector and data for one Keplerian PSTL class.

    Parameters
    ----------
    pnum : int
        Parameterization (1 or 2).
    par : bool
        Use the parallax class (and parallax vectors).

    Returns
    -------
    cls : type
        Composed model class.
    kw : dict
        Scalar constructor kwargs (filter values unwrapped).
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
    kw = dict(KW[pnum])
    if par:
        kw.update(raL=268.0, decL=-29.0)
    tag = 'Par' if par else 'noPar'
    cls = getattr(mj, f'PSTL_PhotAstrom_{tag}_EllOrbs_Param{pnum}')
    t = np.linspace(56900.0, 57100.0, 30)

    # Host model (single filter).
    host = cls(**{k: ([v] if k in FILT else v) for k, v in kw.items()},
               root_tol=1e-8)
    mag_h = np.asarray(host.get_photometry(t), dtype=float)
    pos_h = np.asarray(host.get_astrometry(t), dtype=float)

    # Filter-local packed vector and optional parallax vectors.
    vec = jnp.asarray([kw[n] for n in cls.fitter_param_names])
    pvec = None
    if par:
        pvec = jax_physics.precompute_parallax_vectors(
            kw['raL'], kw['decL'], t, obs_location='earth')

    return cls, kw, vec, t, mag_h, pos_h, pvec


def _phot_fn(cls, kw, t, mag_h, pvec):
    """
    Traced photometry log-likelihood as a function of (vec, err).

    Parameters
    ----------
    cls : type
        Composed model class.
    kw : dict
        Scalar constructor kwargs.
    t, mag_h : ndarray, shape (N_times,)
        Times and host magnitudes (the data).
    pvec : ndarray or None
        Parallax vectors.

    Returns
    -------
    fn : callable
        ``fn(vec, err) -> lnL``.
    """
    def fn(vec, err):
        return cls.jax_log_likely_photometry(
            vec, jnp.asarray(t), jnp.asarray(mag_h), jnp.asarray(err),
            kw['b_sff'], kw['mag_src'], parallax_vectors=pvec)
    return fn


def _ast_fn(cls, kw, t, pos_h, pvec):
    """
    Traced astrometry log-likelihood as a function of (vec, err).

    Parameters
    ----------
    cls : type
        Composed model class.
    kw : dict
        Scalar constructor kwargs.
    t : ndarray, shape (N_times,)
        Times (MJD).
    pos_h : ndarray, shape (N_times, 2)
        Host centroid (the data, arcsec).
    pvec : ndarray or None
        Parallax vectors.

    Returns
    -------
    fn : callable
        ``fn(vec, err) -> lnL``.
    """
    def fn(vec, err):
        return cls.jax_log_likely_astrometry(
            vec, jnp.asarray(t), jnp.asarray(pos_h[:, 0]),
            jnp.asarray(pos_h[:, 1]), jnp.asarray(err), jnp.asarray(err),
            b_sff=kw['b_sff'], mag_src=kw['mag_src'],
            dmag_Lp_Ls=kw['dmag_Lp_Ls'], dmag_Lp_Lt=kw['dmag_Lp_Lt'],
            parallax_vectors=pvec)
    return fn


@pytest.mark.parametrize("pnum,par", [(1, False), (1, True),
                                      (2, False), (2, True)])
def test_traced_kepler_pstl_matches_host(pnum, par):
    """Keplerian PSTL traced photometry and astrometry equal the host."""
    cls, kw, vec, t, mag_h, pos_h, pvec = _setup(pnum, par)
    err = np.full(t.size, ERR)

    # Photometry: RMS magnitude difference from the chi^2.
    phot_fn = _phot_fn(cls, kw, t, mag_h, pvec)
    lnl_max = float(jax_physics.gaussian_log_likelihood_sum(
        jnp.asarray(mag_h), mag_h, err))
    chi2 = max(-2.0 * (float(phot_fn(vec, err)) - lnl_max), 0.0)
    rms_mag = ERR * np.sqrt(chi2 / t.size)
    assert rms_mag < 1e-8

    # Astrometry: RMS per-axis position difference (mas).
    ast_fn = _ast_fn(cls, kw, t, pos_h, pvec)
    lnl_max = float(jax_physics.gaussian_astrometry_log_likelihood_sum(
        jnp.asarray(pos_h), pos_h[:, 0], pos_h[:, 1], err, err))
    chi2 = max(-2.0 * (float(ast_fn(vec, err)) - lnl_max), 0.0)
    rms_mas = 1e3 * ERR * np.sqrt(chi2 / (2 * t.size))
    assert rms_mas < 1e-8

    return None


@pytest.mark.parametrize("pnum", [1, 2])
def test_traced_kepler_pstl_gradients(pnum):
    """Gradients are finite and every orbit element enters the likelihood."""
    cls, kw, vec, t, mag_h, pos_h, pvec = _setup(pnum, False)
    names = list(cls.fitter_param_names)
    orbit_idx = [names.index(n) for n in ORBIT_NAMES]

    # Realistic errors so the likelihood is not dominated by round-off.
    err_phot = np.full(t.size, 0.01)
    err_ast = np.full(t.size, 1e-4)

    # Perturb off the truth so the gradient is not identically zero.
    vec_off = vec * 1.001
    for fn, err in ((_phot_fn(cls, kw, t, mag_h, pvec), err_phot),
                    (_ast_fn(cls, kw, t, pos_h, pvec), err_ast)):
        grad = np.asarray(jax.grad(fn)(vec_off, err))
        assert np.all(np.isfinite(grad))

    # Combined gradient: each orbit element moves photometry or astrometry.
    def total(v):
        """Photometry plus astrometry log-likelihood."""
        lnl = _phot_fn(cls, kw, t, mag_h, pvec)(v, err_phot)
        lnl = lnl + _ast_fn(cls, kw, t, pos_h, pvec)(v, err_ast)
        return lnl

    grad = np.asarray(jax.grad(total)(vec_off))
    assert np.all(grad[orbit_idx] != 0.0)
    return None


def test_kepler_pstl_fitter_host_vs_jax(tmp_path):
    """MicrolensSolver host and JAX log-likelihoods agree for EllOrbs."""
    cls, kw, vec, t, mag_h, pos_h, pvec = _setup(1, False)
    data = {
        't_phot1': t,
        'mag1': mag_h,
        'mag_err1': np.full(t.shape, 0.01),
        't_ast1': t,
        'xpos1': pos_h[:, 0],
        'ypos1': pos_h[:, 1],
        'xpos_err1': np.full(t.shape, 1e-4),
        'ypos_err1': np.full(t.shape, 1e-4),
        'phot_data': ['I'],
        'ast_data': ['I'],
    }
    fitter = MicrolensSolver(
        data, mj.PSTL_PhotAstrom_noPar_EllOrbs_Param1,
        outputfiles_basename=str(tmp_path / 'ellorbs_'),
        n_live_points=20, max_iter=1, dump_callback=None, verbose=False)

    # Fitter names carry a 1-based filter suffix on filter parameters.
    cube = {}
    for name in fitter.fitter_param_names:
        base = name[:-1] if name[:-1] in FILT else name
        cube[name] = kw[base]

    fn, _ctx = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None

    # Truth, and a point with a moved tertiary orbit.
    for shift in (0.0, 15.0):
        trial = dict(cube)
        trial['big_omega_ter'] = cube['big_omega_ter'] + shift
        lnL_host = float(fitter.log_likely(trial))
        lnL_jax = float(fitter.evaluate_loglik_jax(trial))
        np.testing.assert_allclose(lnL_jax, lnL_host, rtol=1e-8, atol=1e-6)
    return None
