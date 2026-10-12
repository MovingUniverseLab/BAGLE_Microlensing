"""JAX FSBL static-limb kernel tests (plan A-08, A-11, A-12, G-03)."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from bagle import model_jax as mj
from bagle.jax import fsbl as fsbl_jax


def _wide_fsbl_setup(radiusS=2e-4, n_outline=120, beta=2.0):
    """
    Build a wide, non-caustic PhotAstrom FSBL and matching PSBL.

    Parameters
    ----------
    radiusS : float
        Source radius in arcsec.
    n_outline : int
        Limb sample count.
    beta : float
        Impact parameter in mas.

    Returns
    -------
    fsbl, psbl : model instances
    """
    mLp, mLs = 0.5, 0.5
    t0 = 57000.0
    dL, dS = 4000.0, 8000.0
    xS0_E, xS0_N = [0.0], [0.0]
    sep, alpha = 1.0, 45.0
    b_sff, mag_src, dmag = [1.0], [18.0], [20.0]

    fsbl = mj.FSBL_PhotAstrom_noPar_Param1(
        mLp, mLs, t0, radiusS, xS0_E, xS0_N, beta,
        0.0, 0.0, 5.0, 0.0, dL, dS, sep, alpha,
        b_sff, mag_src, dmag, n_outline=n_outline, root_tol=1e-8,
    )
    psbl = mj.PSBL_PhotAstrom_noPar_Param1(
        mLp, mLs, t0, xS0_E, xS0_N, beta,
        0.0, 0.0, 5.0, 0.0, dL, dS, sep, alpha,
        b_sff, mag_src, dmag,
    )

    return fsbl, psbl


def _kernel_amps_from_host_geometry(fsbl, t, n_outline=None):
    """
    Evaluate the d1 JAX kernel on host geometry for times ``t``.

    Parameters
    ----------
    fsbl : FSBL model
        Host model providing ``get_u`` and lens astrometry.
    t : array_like
        MJD times.
    n_outline : int or None
        Override limb count; default ``fsbl.n_outline``.

    Returns
    -------
    amp : ndarray, shape (len(t),)
    """
    if n_outline is None:
        n_outline = int(fsbl.n_outline)

    u_vec = np.asarray(fsbl.get_u(t))
    rho = float(fsbl.source_radius_thetaE_units())
    lens_res = np.asarray(fsbl.get_resolved_lens_astrometry(t))
    lens1 = lens_res[0] / fsbl.thetaE_amp * 1e3
    lens2 = lens_res[1] / fsbl.thetaE_amp * 1e3
    w = u_vec[:, 0] + 1j * u_vec[:, 1]
    z1_abs = lens1[:, 0] + 1j * lens1[:, 1]
    z2_abs = lens2[:, 0] + 1j * lens2[:, 1]
    z_cm = 0.5 * (z1_abs + z2_abs)
    m1 = fsbl.mLp / fsbl.mL
    m2 = fsbl.mLs / fsbl.mL

    return np.asarray(
        fsbl_jax.fsbl_magnification(
            w, z1_abs - z_cm, z2_abs - z_cm, m1, m2, rho, n_outline,
            root_tol=1e-8,
        )
    )


def test_A08_tiny_rho_recovers_psbl():
    """A-08 / A-11: tiny rho recovers PSBL (rtol/atol 1e-6)."""
    fsbl, psbl = _wide_fsbl_setup(radiusS=1e-8, n_outline=200)
    t = np.linspace(fsbl.t0 - fsbl.tE, fsbl.t0 + fsbl.tE, 21)
    A_ps = np.asarray(psbl.get_amplification(t))
    A_jax = _kernel_amps_from_host_geometry(fsbl, t, n_outline=200)

    np.testing.assert_allclose(A_jax, A_ps, rtol=1e-6, atol=1e-6)

    return None


def test_A12_finite_source_vs_host_adaptive():
    """A-12 style: d1 kernel vs host adaptive contour (rel 1e-3)."""
    # Smaller rho than the plan's huge 5e-3 arcsec on a 10+10 Msun lens;
    # still a clear FS effect on a wide non-caustic track.
    fsbl, _ = _wide_fsbl_setup(radiusS=2e-4, n_outline=200, beta=2.0)
    t = np.linspace(fsbl.t0 - 1.2 * fsbl.tE, fsbl.t0 + 1.2 * fsbl.tE, 21)
    A_host = np.asarray(fsbl.get_amplification(t))
    A_jax = _kernel_amps_from_host_geometry(fsbl, t, n_outline=200)
    rel = np.max(np.abs(A_jax - A_host) / np.maximum(A_host, 1e-30))

    assert rel < 1e-3, f"A-12 fail max rel={rel}"

    return None


def test_G03_grad_non_caustic():
    """G-03: jacfwd of magnification in rho along a non-caustic track."""
    fsbl, _ = _wide_fsbl_setup(radiusS=2e-4, n_outline=80, beta=2.0)
    t0 = float(fsbl.t0)
    # Single epoch near peak for a cheap gradient check.
    t = np.array([t0])
    u_vec = np.asarray(fsbl.get_u(t))
    w0 = jnp.asarray(u_vec[:, 0] + 1j * u_vec[:, 1])
    lens_res = np.asarray(fsbl.get_resolved_lens_astrometry(t))
    lens1 = lens_res[0] / fsbl.thetaE_amp * 1e3
    lens2 = lens_res[1] / fsbl.thetaE_amp * 1e3
    z1 = jnp.asarray(
        (lens1[:, 0] + 1j * lens1[:, 1])
        - 0.5 * ((lens1[:, 0] + 1j * lens1[:, 1]) + (lens2[:, 0] + 1j * lens2[:, 1]))
    )
    z2 = -z1
    m1 = m2 = 0.5
    rho0 = float(fsbl.source_radius_thetaE_units())
    n_outline = 80

    def mag_of_rho(rho):
        return fsbl_jax.magnification_one_static(
            w0[0], z1[0], z2[0], m1, m2, rho, n_outline
        )

    g = float(jax.grad(mag_of_rho)(rho0))

    # Central difference on the same kernel.
    eps = 1e-6 * max(abs(rho0), 1e-3)
    fd = float(
        (mag_of_rho(rho0 + eps) - mag_of_rho(rho0 - eps)) / (2.0 * eps)
    )
    rel = abs(g - fd) / max(abs(fd), 1e-30)

    assert np.isfinite(g)
    assert rel < 1e-4, f"G-03 fail grad={g} fd={fd} rel={rel}"

    return None


def test_fsbl_phot_param1_loglik_finite():
    """Smoke: traced FSBL_PhotParam1 log-likelihood is finite."""
    t0 = 57000.0
    # Nonzero piE so hats are well-defined.
    vec = jnp.array(
        [t0, 0.5, 50.0, 0.1, 0.05, 0.2, 1.0, 30.0, 0.02, 1.0, 19.0],
        dtype=jnp.float64,
    )
    t = jnp.linspace(t0 - 40.0, t0 + 40.0, 16)
    mag_obs = jnp.ones_like(t) * 19.0
    mag_err = jnp.ones_like(t) * 0.01
    lnL = mj.FSBL_PhotParam1.jax_log_likely_photometry(
        vec, t, mag_obs, mag_err, 1.0, 19.0, n_outline=60
    )

    assert np.isfinite(float(lnL))

    return None


@pytest.mark.skip(reason="FSBL d2 adaptive contour not implemented yet")
def test_A14_d1_vs_d2_non_caustic():
    """A-14: d1 static limb and d2 adaptive agree on a wide trajectory."""
    fsbl, _ = _wide_fsbl_setup(radiusS=2e-4, n_outline=120, beta=2.0)
    t = np.linspace(fsbl.t0 - fsbl.tE, fsbl.t0 + fsbl.tE, 11)
    A_d1 = _kernel_amps_from_host_geometry(fsbl, t, n_outline=120)

    u_vec = np.asarray(fsbl.get_u(t))
    rho = float(fsbl.source_radius_thetaE_units())
    lens_res = np.asarray(fsbl.get_resolved_lens_astrometry(t))
    lens1 = lens_res[0] / fsbl.thetaE_amp * 1e3
    lens2 = lens_res[1] / fsbl.thetaE_amp * 1e3
    w = u_vec[:, 0] + 1j * u_vec[:, 1]
    z1_abs = lens1[:, 0] + 1j * lens1[:, 1]
    z2_abs = lens2[:, 0] + 1j * lens2[:, 1]
    z_cm = 0.5 * (z1_abs + z2_abs)
    A_d2 = np.asarray(
        fsbl_jax.fsbl_magnification_adaptive(
            w, z1_abs - z_cm, z2_abs - z_cm, 0.5, 0.5, rho
        )
    )
    rel = np.max(np.abs(A_d1 - A_d2) / np.maximum(A_d1, 1e-30))

    assert rel < 1e-4

    return None
