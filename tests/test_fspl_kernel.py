"""JAX FSPL Bozza kernel tests (plan A-06–A-08, G-01, G-02)."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from bagle import model
from bagle.jax import fspl as fspl_jax
from bagle.model import function_photometry_two


def _lee_magnification_gauss_legendre(u, rho, n_quad=512):
    """
    High-order Gauss–Legendre on the Lee (2009) photometry integrand.

    Parameters
    ----------
    u : float
        Source-lens separation amplitude in Einstein radii.
    rho : float
        Source radius in Einstein radii.
    n_quad : int, optional
        Number of Gauss–Legendre nodes.

    Returns
    -------
    mag : float
        Total finite-source magnification.
    """
    u = float(u)
    rho = float(rho)
    x, w = leggauss(n_quad)

    # Inside the source: integrate theta in [0, pi]; outside [0, arcsin(rho/u)].
    if u <= rho:
        th_max = np.pi
    else:
        th_max = float(np.arcsin(np.minimum(rho / u, 1.0)))

    theta = 0.5 * th_max * x + 0.5 * th_max
    weights = 0.5 * th_max * w
    integrand = np.array(
        [function_photometry_two(th, rho, u) for th in theta]
    )

    return float(np.sum(weights * integrand) / (np.pi * rho ** 2))


def _sample_u_rho_grid():
    """
    (rho, u) pairs covering the A-06 domain, including u < rho.

    Returns
    -------
    pairs : list of (rho, u)
        Source radius and separation amplitudes.
    """
    pairs = []
    for rho in (1e-3, 1e-2, 5e-2, 0.1):
        for u in (0.0, 0.5 * rho, 0.9 * rho, 1.2 * rho, 2.0 * rho, 0.5, 1.0, 2.0, 5.0):
            if u > 5.0:
                continue
            pairs.append((rho, float(u)))

    return pairs


def test_A06_jax_vs_lee_gauss_legendre():
    """A-06: JAX Bozza kernel vs high-order Lee Gauss–Legendre (rel 1e-3)."""
    n_outline = 800
    max_rel = 0.0

    for rho, u_amp in _sample_u_rho_grid():
        mag_lee = _lee_magnification_gauss_legendre(u_amp, rho, n_quad=512)

        # Place the source on the East axis for a scalar |u|.
        u = jnp.array([[u_amp, 0.0]], dtype=jnp.float64)
        mag_jax = np.asarray(
            fspl_jax.fspl_magnification(u, rho, n_outline)
        )[0]

        rel = abs(mag_jax - mag_lee) / max(abs(mag_lee), 1e-30)
        max_rel = max(max_rel, rel)
        assert rel < 1e-3, (
            f"A-06 fail rho={rho} u={u_amp}: jax={mag_jax} lee={mag_lee} "
            f"rel={rel}"
        )

    assert max_rel < 1e-3
    return None


def test_A07_jax_vs_host_amg():
    """A-07: JAX kernel vs host AMG (rel 1e-3 mag, 1e-3 arcsec centroid)."""
    mL = 10.0
    t0 = 57000.0
    beta = 0.2
    dL = 4000.0
    dL_dS = 0.5
    radiusS = 2e-4
    n_outline = 400

    fspl = model.FSPL_PhotAstrom_noPar_Param1(
        mL, t0, beta, dL, dL_dS,
        0.0, 0.0, 0.0, 0.0, 3.0, 0.0,
        radiusS, 1.0, 18.0, n_outline, raL=259.0, decL=-29.0,
    )
    rho = fspl.source_radius_thetaE_units()
    t = np.linspace(t0 - 1.5 * fspl.tE, t0 + 1.5 * fspl.tE, 41)

    # Host AMG reference (ported vectorized path from b1).
    img_h, amp_h = fspl.get_all_arrays_amg(t)
    mag_h = np.sum(amp_h, axis=1)
    if np.any(mag_h < 0.0):
        mag_h = -mag_h
        amp_h = -amp_h

    xL = np.atleast_2d(fspl.get_lens_astrometry(t))
    abs_amp_h = np.abs(amp_h)
    cent_h = (
        np.sum(img_h * abs_amp_h[:, :, None], axis=1)
        / np.sum(abs_amp_h, axis=1)[:, None]
    )

    # JAX kernel in Einstein radii, then convert centroid to arcsec on sky.
    u = fspl.get_u(t)
    mag_j, cent_ein = fspl_jax.fspl_magnification_centroid(
        jnp.asarray(u), rho, n_outline
    )
    mag_j = np.asarray(mag_j)
    cent_j = xL + np.asarray(cent_ein) * (fspl.thetaE_amp * 1e-3)

    rel_mag = np.max(np.abs(mag_j - mag_h) / np.maximum(np.abs(mag_h), 1e-30))
    max_cent = np.max(np.abs(cent_j - cent_h))

    assert rel_mag < 1e-3, f"A-07 mag rel={rel_mag}"
    assert max_cent < 1e-3, f"A-07 centroid max abs={max_cent} arcsec"

    return None


def test_A08_tiny_rho_recovers_pspl():
    """A-08: tiny rho recovers analytic PSPL magnification."""
    n_outline = 200
    u_list = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 5.0])

    for rho in (1e-8, 1e-9):
        for u_amp in u_list:
            u = jnp.array([[u_amp, 0.0]])
            mag_fs = float(fspl_jax.fspl_magnification(u, rho, n_outline)[0])
            mag_ps = float(fspl_jax.fspl_point_source_limit_mag(u)[0])
            rel = abs(mag_fs - mag_ps) / max(abs(mag_ps), 1e-30)
            assert rel < 1e-6, (
                f"A-08 strict rho={rho} u={u_amp}: rel={rel}"
            )

    # Milder absolute check for rho <= 1e-3 with u > 10 rho. Keep u
    # away from the Einstein-ring peak so the true FS correction is
    # already below the 1e-4 floor (u = 0.05 still has |dA| ~ 1e-3).
    rho = 1e-3
    for u_amp in u_list:
        if u_amp <= 10.0 * rho or u_amp < 0.2:
            continue
        u = jnp.array([[u_amp, 0.0]])
        mag_fs = float(fspl_jax.fspl_magnification(u, rho, n_outline)[0])
        mag_ps = float(fspl_jax.fspl_point_source_limit_mag(u)[0])
        assert abs(mag_fs - mag_ps) < 1e-4, (
            f"A-08 abs rho={rho} u={u_amp}: |dA|={abs(mag_fs - mag_ps)}"
        )

    return None


def test_G01_grad_u_rho_finite_nonzero_vs_fd():
    """G-01: jax.grad of mag in u and rho is finite, nonzero, matches FD."""
    n_outline = 200
    rho0 = 0.05
    u0 = 0.3

    def mag_u(u_x):
        """Magnification vs East separation at fixed rho."""
        return fspl_jax.fspl_magnification(
            jnp.array([[u_x, 0.0]]), rho0, n_outline
        )[0]

    def mag_rho(rho):
        """Magnification vs rho at fixed u."""
        return fspl_jax.fspl_magnification(
            jnp.array([[u0, 0.0]]), rho, n_outline
        )[0]

    g_u = float(jax.grad(mag_u)(u0))
    g_rho = float(jax.grad(mag_rho)(rho0))

    assert np.isfinite(g_u) and np.isfinite(g_rho)
    assert g_u != 0.0 and g_rho != 0.0

    # Central finite differences on the same JAX kernel.
    eps_u = 1e-6
    eps_rho = 1e-6
    fd_u = (float(mag_u(u0 + eps_u)) - float(mag_u(u0 - eps_u))) / (
        2.0 * eps_u
    )
    fd_rho = (
        float(mag_rho(rho0 + eps_rho)) - float(mag_rho(rho0 - eps_rho))
    ) / (2.0 * eps_rho)

    rel_u = abs(g_u - fd_u) / max(abs(fd_u), 1e-30)
    rel_rho = abs(g_rho - fd_rho) / max(abs(fd_rho), 1e-30)
    assert rel_u < 1e-4, f"G-01 dA/du rel={rel_u}"
    assert rel_rho < 1e-4, f"G-01 dA/drho rel={rel_rho}"

    return None


def test_G02_grad_at_u0_finite():
    """G-02: gradient of magnification at u = 0 is finite (may be ~0)."""
    n_outline = 200
    rho = 0.05

    def mag_u(u_x):
        """Magnification vs East separation through the lens."""
        return fspl_jax.fspl_magnification(
            jnp.array([[u_x, 0.0]]), rho, n_outline
        )[0]

    g0 = float(jax.grad(mag_u)(0.0))
    assert np.isfinite(g0)

    return None


def test_A25_gamma0_matches_uniform_fspl():
    """A-25: ``gamma = 0`` limb darkening matches uniform FSPL (rel 1e-3)."""
    t0 = 57000.0
    kw = dict(
        t0=t0, u0_amp=0.01, tE=25.0, piE_E=0.05, piE_N=0.02,
        radiusS=0.03, b_sff=[1.0], mag_src=[18.0], n_outline=200,
    )
    t = np.linspace(t0 - 40.0, t0 + 40.0, 61)

    uni = model.FSPL_Phot_noPar_Param1(**kw)
    limb = model.FSPL_Limb_Phot_noPar_Param1(gamma=[0.0], **kw)

    mag_u = np.asarray(uni.get_photometry(t), dtype=np.float64)
    mag_l = np.asarray(limb.get_photometry(t), dtype=np.float64)
    rel = np.abs(mag_l - mag_u) / np.maximum(np.abs(mag_u), 1e-12)
    assert np.max(rel) < 1e-3
    return None
