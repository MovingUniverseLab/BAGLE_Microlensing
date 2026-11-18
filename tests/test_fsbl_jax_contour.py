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


def test_ld_radius_nodes_match_host():
    """JAX caustic-tangency radial nodes equal the host nodes."""
    s, q, rho = 1.0, 0.1, 0.01
    z1, z2, m1, m2 = fc.lens_positions_cm(s, q)

    # Off caustic, fold crossing, cusp approach and inside the caustic.
    zeta = np.array([-1.0 + 0.3j, -0.05 + 0.05j, -0.03 + 0.05j, -0.07 + 0.0j,
                     0.0 + 0.0j])
    rho_arr = np.full(zeta.size, rho)

    # Host nodes from one static caustic.
    cz, cd, nx = fc.caustic_samples(np.atleast_1d(z1), np.atleast_1d(z2),
                                    np.atleast_1d(m1), np.atleast_1d(m2))
    r_tan = fc.caustic_tangency_radii(zeta, cz, cd, nx)
    s_h, w_h = fc.ld_break_nodes(r_tan, rho_arr)

    # JAX nodes for the same epochs.
    s_j, w_j = jfc.ld_radius_nodes(jnp.asarray(zeta), jnp.asarray(rho_arr),
                                   z1, z2, m1, m2)

    assert np.allclose(np.asarray(s_j), s_h, atol=1e-8)
    assert np.allclose(np.asarray(w_j), w_h, atol=1e-8)

    # Each epoch's quadrature integrates s^2 over [0, 1] exactly (= 1/3).
    assert np.allclose(np.sum(w_h * s_h ** 2, axis=1), 1.0 / 3.0, atol=1e-6)

    return None


@pytest.mark.parametrize("y,xs,tol", [
    (0.05, np.linspace(-0.07, -0.01, 13), 1e-5),  # fold crossings
    # On-axis cusp passage; floor set by the per-disk contour budget.
    (0.0, np.linspace(-0.09, -0.05, 11), 3e-5),
])
def test_ld_breakpoints_converged(y, xs, tol):
    """Default LD radial nodes match a dense breakpoint rule (Gamma = 0.8)."""
    s, q, rho, gamma = 1.0, 0.1, 0.01, 0.8
    z1, z2, m1, m2 = fc.lens_positions_cm(s, q)
    zeta = jnp.asarray(xs + 1j * y)

    # Defaults (n_ld = 8, n_brk = 2) vs 16 nodes on each of 5 intervals.
    mag, cen = jfc.fsbl_magnification_centroid(zeta, rho, z1, z2, m1, m2,
                                               gamma=gamma)
    mag_d, cen_d = jfc.fsbl_magnification_centroid(
        zeta, rho, z1, z2, m1, m2, gamma=gamma, n_ld=16, n_brk=4)

    frac = np.abs(np.asarray(mag / mag_d) - 1.0)
    dcen = np.linalg.norm(np.asarray(cen - cen_d), axis=1)
    assert frac.max() < tol
    assert dcen.max() < tol

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


@pytest.mark.parametrize("limb,par", [(False, False), (False, True),
                                      (True, False)])
def test_traced_astrometry_matches_host(limb, par):
    """FSBL(_Limb)_PhotAstrom Param1 traced astrometry equals host astrometry."""
    from bagle import jax_physics
    from bagle import model_jax as mj

    # Caustic-crossing event (peak A ~ 10) with blended, luminous lenses.
    t = np.linspace(56985.0, 57015.0, 20 if limb else 40)
    kw = dict(mLp=0.5, mLs=0.05, t0=57000.0, radiusS=8e-6, xS0_E=0.001,
              xS0_N=-0.002, beta=0.04, muL_E=1.0, muL_N=3.0, muS_E=-2.0,
              muS_N=-1.0, dL=4000.0, dS=8000.0, sep=0.75, alpha=70.0,
              b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5, raL=268.0,
              decL=-29.0)
    if limb:
        kw['gamma'] = 0.6
    name = 'FSBL_{}PhotAstrom_{}_Param1'.format('Limb_' if limb else '',
                                               'Par' if par else 'noPar')
    cls = getattr(mj, name)

    # Host model (single filter) and its sky positions (arcsec).
    host_kw = {k: ([v] if k in ('xS0_E', 'xS0_N', 'b_sff', 'mag_src',
                                'dmag_Lp_Ls', 'gamma') else v)
               for k, v in kw.items()}
    pos_h = np.asarray(cls(**host_kw).get_astrometry(t))

    # Filter-local packed vector and optional parallax vectors.
    vec = jnp.asarray([kw[n] for n in cls.fitter_param_names])
    pvec = None
    if par:
        pvec = jax_physics.precompute_parallax_vectors(
            kw['raL'], kw['decL'], t, obs_location='earth')

    # With data at the host positions and tiny errors, the chi^2 of the
    # traced model measures the position difference directly.
    err = np.full(t.size, 1e-8)
    lnl_fn = lambda v: cls.jax_log_likely_astrometry(
        v, jnp.asarray(t), jnp.asarray(pos_h[:, 0]), jnp.asarray(pos_h[:, 1]),
        jnp.asarray(err), jnp.asarray(err), b_sff=0.7, mag_src=19.0,
        dmag_Lp_Ls=1.5, parallax_vectors=pvec)
    lnl = float(lnl_fn(vec))
    lnl_max = float(jax_physics.gaussian_astrometry_log_likelihood_sum(
        jnp.asarray(pos_h), pos_h[:, 0], pos_h[:, 1], err, err))
    rms_mas = 1e3 * 1e-8 * np.sqrt(max(-2.0 * (lnl - lnl_max), 0.0)
                                   / (2 * t.size))
    assert rms_mas < 1e-5

    # Gradient of a realistic (1 mas error) likelihood is finite.
    grad = jax.grad(lambda v: cls.jax_log_likely_astrometry(
        v, jnp.asarray(t), jnp.asarray(pos_h[:, 0]) + 3e-5,
        jnp.asarray(pos_h[:, 1]), jnp.asarray(err * 1e5),
        jnp.asarray(err * 1e5), b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5,
        parallax_vectors=pvec))(vec)
    assert np.all(np.isfinite(np.asarray(grad)))

    return None


_ORBIT_KW = dict(mLp=0.5, mLs=0.05, t0=57000.0, radiusS=8e-6, xS0_E=0.001,
                 xS0_N=-0.002, beta=0.04, muL_E=1.0, muL_N=3.0,
                 delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0, muS_E=-2.0,
                 muS_N=-1.0, dL=4000.0, dS=8000.0, sep=0.75, alpha=70.0,
                 b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5, raL=268.0,
                 decL=-29.0)
_FILT_KEYS = ('xS0_E', 'xS0_N', 'b_sff', 'mag_src', 'dmag_Lp_Ls', 'gamma')


@pytest.mark.parametrize("orb", ["LinOrbs", "AccOrbs"])
def test_orbit_host_frame_matches_psbl(orb):
    """Host orbital FSBL with a tiny source equals point-source PSBL."""
    from bagle import model_jax as mj

    # Long baseline so the secondary moves by a sizeable fraction of thetaE.
    t = np.linspace(56970.0, 57030.0, 13)
    kw = {k: ([v] if k in _FILT_KEYS else v) for k, v in _ORBIT_KW.items()
          if k not in ('radiusS', 'raL', 'decL')}
    if orb == 'AccOrbs':
        kw.update(accLsec_E=2.0, accLsec_N=1.0)
    ps = getattr(mj, f'PSBL_PhotAstrom_noPar_{orb}_Param1')(**kw)
    fs = getattr(mj, f'FSBL_PhotAstrom_noPar_{orb}_Param1')(radiusS=1e-8,
                                                            **kw)

    # Photometry and unresolved astrometry agree in the point-source limit.
    dmag = np.asarray(fs.get_photometry(t)) - np.asarray(ps.get_photometry(t))
    dpos = np.asarray(fs.get_astrometry(t)) - np.asarray(ps.get_astrometry(t))
    assert np.max(np.abs(dmag)) < 1e-5
    assert np.max(np.abs(dpos)) * 1e3 < 1e-5

    return None


@pytest.mark.parametrize("orb,par,limb", [("LinOrbs", False, False),
                                          ("LinOrbs", True, False),
                                          ("AccOrbs", False, False),
                                          ("AccOrbs", True, False),
                                          ("LinOrbs", False, True)])
def test_traced_orbit_matches_host(orb, par, limb):
    """Orbital FSBL(_Limb) Param1 traced photometry/astrometry equal host."""
    from bagle import jax_physics
    from bagle import model_jax as mj

    # Caustic-crossing event with an orbiting secondary and luminous lenses.
    t = np.linspace(56985.0, 57015.0, 16 if limb else 32)
    kw = dict(_ORBIT_KW)
    if orb == 'AccOrbs':
        kw.update(accLsec_E=2.0, accLsec_N=1.0)
    if limb:
        kw['gamma'] = 0.6
    name = 'FSBL_{}PhotAstrom_{}_{}_Param1'.format(
        'Limb_' if limb else '', 'Par' if par else 'noPar', orb)
    cls = getattr(mj, name)
    if not par:
        kw.pop('raL')
        kw.pop('decL')

    # Host model (single filter): magnitudes and sky positions (arcsec).
    host = cls(**{k: ([v] if k in _FILT_KEYS else v) for k, v in kw.items()})
    mag_h = np.asarray(host.get_photometry(t))
    pos_h = np.asarray(host.get_astrometry(t))

    # Filter-local packed vector and optional parallax vectors.
    vec = jnp.asarray([kw[n] for n in cls.fitter_param_names])
    pvec = None
    if par:
        pvec = jax_physics.precompute_parallax_vectors(
            kw['raL'], kw['decL'], t, obs_location='earth')

    # With data at the host model and tiny errors, chi^2 measures the RMS
    # model difference directly.
    err = np.full(t.size, 1e-8)
    phot_fn = lambda v, e: cls.jax_log_likely_photometry(
        v, jnp.asarray(t), jnp.asarray(mag_h), jnp.asarray(e), 0.7, 19.0,
        parallax_vectors=pvec)
    lnl_max = float(jax_physics.gaussian_log_likelihood_sum(
        jnp.asarray(mag_h), mag_h, err))
    rms_mag = 1e-8 * np.sqrt(max(-2.0 * (float(phot_fn(vec, err)) - lnl_max),
                                 0.0) / t.size)
    assert rms_mag < 1e-5

    ast_fn = lambda v, e: cls.jax_log_likely_astrometry(
        v, jnp.asarray(t), jnp.asarray(pos_h[:, 0]), jnp.asarray(pos_h[:, 1]),
        jnp.asarray(e), jnp.asarray(e), b_sff=0.7, mag_src=19.0,
        dmag_Lp_Ls=1.5, parallax_vectors=pvec)
    lnl_max = float(jax_physics.gaussian_astrometry_log_likelihood_sum(
        jnp.asarray(pos_h), pos_h[:, 0], pos_h[:, 1], err, err))
    rms_mas = 1e3 * 1e-8 * np.sqrt(
        max(-2.0 * (float(ast_fn(vec, err)) - lnl_max), 0.0) / (2 * t.size))
    assert rms_mas < 1e-5

    # Gradients of realistic likelihoods (incl. the orbit terms) are finite.
    for fn in (phot_fn, ast_fn):
        grad = np.asarray(jax.grad(fn)(vec, err * 1e5))
        assert np.all(np.isfinite(grad))
    idx = cls.fitter_param_names.index('delta_muL_sec_E')
    assert np.asarray(jax.grad(phot_fn)(vec, err * 1e5))[idx] != 0.0

    return None


def test_traced_method_ownership():
    """FSBL-family classes only inherit their own family's traced likelihood."""
    import inspect

    from bagle import model_jax as mj

    for name, cls in vars(mj).items():
        if not inspect.isclass(cls):
            continue
        family = next((f for f in ("BFSBL", "BFSPL", "FSBL")
                       if name.startswith(f)), None)
        if family is None:
            continue
        for meth in ("jax_log_likely_photometry", "jax_log_likely_astrometry"):
            if getattr(cls, meth, None) is None:
                continue
            owner = next(k for k in cls.__mro__ if meth in k.__dict__).__name__
            ok = owner.startswith(family)
            if family == "FSBL":
                ok = ok or owner == "_FSBLOrbitTracedParam1"
            assert ok, (name, meth, owner)

    # Linear / accelerated orbital FSBL use the traced orbit likelihoods.
    for orb in ("LinOrbs", "AccOrbs"):
        cls = getattr(mj, f"FSBL_Limb_PhotAstrom_Par_{orb}_Param1")
        assert issubclass(cls, mj._FSBLOrbitTracedParam1)

    return None
