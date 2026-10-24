"""
Traced FSBL contour integrator (:mod:`bagle.jax.fsbl_contour`) tests.

Checks the JAX port against the host :mod:`bagle.fsbl_contour` integrator
(same limb grid and refined limb), against VBMicrolensing BinaryMag2 on
caustic-crossing tracks, gradients against finite differences, and the
traced FSBL Param-class likelihoods against host photometry.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp

from bagle import fsbl_contour as fc
from bagle.jax import fsbl_contour as jfc

TOL_PHOT = 1.0e-4
TOL_AST = 1.0e-4

# Caustic-crossing and cusp tracks (s, q, rho, y), shared with
# tests/test_fsbl_vs_vbm.py.
CAUSTIC_CASES = [
    (1.0, 0.1, 0.01, 0.05),
    (1.0, 0.1, 0.05, 0.05),
    (0.8, 0.5, 0.02, 0.05),
    (1.0, 0.1, 0.005, 0.0),
]


def _stacked_cases(cases, xs):
    """
    Concatenate several source tracks into one per-row geometry set.

    Parameters
    ----------
    cases : list of tuple
        ``(s, q, rho, y)`` per track.
    xs : ndarray, shape (N_x,)
        Source x positions along each track (centre-of-mass frame).

    Returns
    -------
    zeta, rho, z1, z2, m1, m2 : ndarray, shape (N_cases * N_x,)
        Source centres, radii, lens positions and masses per row.
    """
    geo = [fc.lens_positions_cm(s, q) for s, q, _, _ in cases]
    rep = lambda v: np.repeat(np.asarray(v), xs.size)

    zeta = np.concatenate([xs + 1j * c[3] for c in cases])
    rho = rep([c[2] for c in cases])
    z1, z2 = (rep([g[i] for g in geo]) for i in (0, 1))
    m1, m2 = (rep([g[i] for g in geo]) for i in (2, 3))

    return zeta, rho, z1, z2, m1, m2


def test_same_grid_matches_host():
    """On an identical uniform limb, JAX and host integrals agree."""
    s, q, rho = 1.0, 0.1, 0.005
    z1, z2, m1, m2 = fc.lens_positions_cm(s, q)
    n = 256
    theta = 2.0 * np.pi * np.arange(n) / n

    for zc in (-0.06 + 0.0j, 0.3 + 0.05j, -1.0 + 0.3j):
        # Host integral on the uniform grid.
        z, p, dz, wp, jac, _ = fc._limb_sample_images(
            zc, rho, theta, z1, z2, m1, m2, 1e-10)
        a_h, mx_h, my_h, _ = fc._integrate_epoch(theta, z, p, dz, wp, jac,
                                                 rho)

        # JAX integral with no refinement passes (same grid).
        mom = np.asarray(jfc.fsbl_disk_moments(
            jnp.asarray([zc]), rho, z1, z2, m1, m2, n_init=n, n_pass=0,
            n_add=1))[0]
        assert abs(mom[0] - a_h) <= 1e-10 * abs(a_h)
        assert abs(mom[1] / mom[0] - mx_h / a_h) <= 1e-9
        assert abs(mom[2] / mom[0] - my_h / a_h) <= 1e-9

    return None


def test_uniform_vs_vbm():
    """Default schedule vs BinaryMag2 on caustic tracks (<= 1e-4)."""
    VBM = pytest.importorskip("VBMicrolensing")
    vbm = VBM.VBMicrolensing()
    vbm.Tol, vbm.RelTol = 1e-7, 1e-8
    vbm.astrometry = True

    xs = np.linspace(-0.6, 0.6, 41)
    zeta, rho, z1, z2, m1, m2 = _stacked_cases(CAUSTIC_CASES, xs)
    mag, cen = jfc.fsbl_magnification_centroid(
        jnp.asarray(zeta), jnp.asarray(rho), jnp.asarray(z1), jnp.asarray(z2),
        jnp.asarray(m1), jnp.asarray(m2))
    mag, cen = np.asarray(mag), np.asarray(cen)

    # VBM uses the same centre-of-mass frame.
    A_v = np.empty(zeta.size)
    c_v = np.empty((zeta.size, 2))
    params = [c for c in CAUSTIC_CASES for _ in xs]
    for i, ((s, q, r, _), w) in enumerate(zip(params, zeta)):
        A_v[i] = vbm.BinaryMag2(s, q, float(w.real), float(w.imag), r)
        c_v[i] = (vbm.astrox1, vbm.astrox2)

    frac = np.abs(mag - A_v) / A_v
    dcen = np.linalg.norm(cen - c_v, axis=1)
    assert frac.max() <= TOL_PHOT
    assert dcen.max() <= TOL_AST

    return None


def test_limb_darkened_vs_host():
    """Nested-disk LD matches the host LD integrator off and on caustics."""
    s, q, rho, gamma = 1.0, 0.1, 0.01, 0.4
    z1, z2, m1, m2 = fc.lens_positions_cm(s, q)
    zeta = np.array([-1.0 + 0.3j, -0.3 + 0.05j, 0.0 + 0.05j, 0.33 + 0.05j])

    img_h, amp_h = fc.ld_images_amps(zeta, rho, gamma, s, q, tol=1e-5,
                                     max_pts=2048)
    mag, cen = jfc.fsbl_magnification_centroid(
        jnp.asarray(zeta), rho, z1, z2, m1, m2, gamma=gamma)

    frac = np.abs(np.asarray(mag) - amp_h) / amp_h
    dcen = np.linalg.norm(np.asarray(cen) - img_h, axis=1)
    assert frac.max() <= TOL_PHOT
    assert dcen.max() <= TOL_AST

    # Gamma = 0 reduces to the uniform source.
    mag0, _ = jfc.fsbl_magnification_centroid(
        jnp.asarray(zeta), rho, z1, z2, m1, m2, gamma=0.0)
    mag_u, _ = jfc.fsbl_magnification_centroid(
        jnp.asarray(zeta), rho, z1, z2, m1, m2)
    assert np.max(np.abs(np.asarray(mag0 - mag_u)) / np.asarray(mag_u)) < 1e-6

    return None


@pytest.mark.parametrize("x,y,rel_tol", [
    (-1.0, 0.3, 1e-5),   # off caustic
    (0.05, 0.05, 1e-5),  # source inside the caustic
    (0.33, 0.05, 2e-2),  # limb crossing a fold (chord discretisation)
])
def test_gradients_vs_finite_differences(x, y, rel_tol):
    """Gradients in source position, rho, s and q match central differences."""

    def mag_of(p):
        # p = [x, y, rho, s, q]; lens geometry in the centre-of-mass frame.
        xx, yy, rho, s, q = p
        m1 = 1.0 / (1.0 + q)
        m2 = q / (1.0 + q)
        z1 = -s * q / (1.0 + q) + 0j
        z2 = s / (1.0 + q) + 0j
        mag, _ = jfc.fsbl_magnification_centroid(
            jnp.atleast_1d(xx + 1j * yy), rho, z1, z2, m1, m2)
        return mag[0]

    p0 = jnp.asarray([x, y, 0.01, 1.0, 0.1])
    grad = np.asarray(jax.grad(mag_of)(p0))
    assert np.all(np.isfinite(grad))

    for k in range(5):
        h = 1e-6 if k == 2 else 1e-5
        e = jnp.zeros(5).at[k].set(h)
        fd = float(mag_of(p0 + e) - mag_of(p0 - e)) / (2.0 * h)
        assert abs(grad[k] - fd) <= rel_tol * max(abs(fd), 1e-3)

    return None


def _traced_rms_dmag(cls, kw, t):
    """
    RMS magnitude difference between traced and host FSBL photometry.

    Parameters
    ----------
    cls : type
        FSBL composite model class with ``jax_log_likely_photometry``.
    kw : dict
        Constructor keywords (one photometric filter).
    t : ndarray
        Observation times (MJD).

    Returns
    -------
    rms : float
        RMS of (traced - host) model magnitudes.
    grad_ok : bool
        Whether the traced log-likelihood gradient is finite.
    """
    mod = cls(**kw)
    mag_host = np.asarray(mod.get_photometry(t))

    # Filter-local class vector in fitter_param_names order.
    vec = []
    for name in cls.fitter_param_names:
        v = np.atleast_1d(kw[name])[0] if name in kw else getattr(mod, name)
        vec.append(float(np.asarray(v).reshape(-1)[0]))
    vec = jnp.asarray(vec)

    # With mag_obs = host model and unit errors, -2 lnL - N log(2 pi) is the
    # summed squared magnitude difference.
    err = np.ones_like(t)
    lnl_fn = lambda v: cls.jax_log_likely_photometry(
        v, t, mag_host, err, kw['b_sff'][0], kw['mag_src'][0])
    lnl = float(lnl_fn(vec))
    chi2 = -2.0 * lnl - t.size * np.log(2.0 * np.pi)
    grad = np.asarray(jax.grad(lnl_fn)(vec))

    return np.sqrt(max(chi2, 0.0) / t.size), bool(np.all(np.isfinite(grad)))


@pytest.mark.parametrize("limb", [False, True])
def test_traced_likelihood_matches_host(limb):
    """FSBL(_Limb)_Phot_noPar_Param1 traced photometry equals host photometry."""
    from bagle import model_jax as mj

    t = np.linspace(56990.0, 57010.0, 24)
    kw = dict(t0=57000.0, u0_amp=0.05, tE=30.0, piE_E=0.05, piE_N=0.02,
              q=0.1, sep=1.0, phi=30.0, radiusS=0.01, b_sff=[0.8],
              mag_src=[19.0])
    if limb:
        cls = mj.FSBL_Limb_Phot_noPar_Param1
        kw['gamma'] = [0.5]
    else:
        cls = mj.FSBL_Phot_noPar_Param1

    rms, grad_ok = _traced_rms_dmag(cls, kw, t)
    assert rms < 1e-5
    assert grad_ok

    return None


def test_traced_method_ownership():
    """No FSBL class inherits a point-source PSBL traced likelihood."""
    import inspect

    from bagle import model_jax as mj

    for name, cls in vars(mj).items():
        if not inspect.isclass(cls) or not name.startswith(("FSBL", "BFSBL",
                                                             "BFSPL")):
            continue
        for meth in ("jax_log_likely_photometry", "jax_log_likely_astrometry"):
            if getattr(cls, meth, None) is None:
                continue
            owner = next(k for k in cls.__mro__ if meth in k.__dict__)
            assert owner.__name__.startswith("FSBL"), (name, meth, owner)

    # Orbital FSBL has no traced likelihood (host fallback).
    assert mj.FSBL_PhotAstrom_noPar_LinOrbs_Param1.jax_log_likely_photometry \
        is None
    assert mj.FSBL_PhotAstrom_noPar_AccOrbs_Param1.jax_log_likely_astrometry \
        is None

    return None
