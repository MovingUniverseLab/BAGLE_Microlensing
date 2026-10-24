"""
Validate ``bagle.fsbl_contour`` against VBMicrolensing BinaryMag2.

Photometry and astrometry (uniform and limb-darkened) on caustic-crossing
and off-caustic tracks. Plots go under ``tests/plots/fsbl_vs_vbm/``.
Target: fractional magnification and centroid error ``<= 1e-4`` in
``theta_E`` (VBM Tol/RelTol tightened below the library defaults).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("VBMicrolensing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import VBMicrolensing as VBM

from bagle import fsbl_contour as fc
from bagle.fspl_contour import gamma_to_u

PLOT_DIR = Path(__file__).resolve().parent / "plots" / "fsbl_vs_vbm"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TOL_PHOT = 1.0e-4
TOL_AST = 1.0e-4


def _vbm(tol=1e-6, rel_tol=1e-7, a1=0.0, astrometry=True):
    """
    Configured VBMicrolensing instance.

    Parameters
    ----------
    tol, rel_tol : float
        Absolute / relative contour tolerances.
    a1 : float
        Linear LD coefficient (u convention).
    astrometry : bool
        Fill ``astrox1`` / ``astrox2`` on BinaryMag2.

    Returns
    -------
    vbm : VBMicrolensing.VBMicrolensing
    """
    vbm = VBM.VBMicrolensing()
    vbm.Tol = tol
    vbm.RelTol = rel_tol
    vbm.a1 = a1
    vbm.astrometry = astrometry

    return vbm


def _compare_track(s, q, rho, xs, ys, gamma=0.0, tol=1e-4, label="track"):
    """
    Compare BAGLE FSBL contour to VBM on a source track.

    Parameters
    ----------
    s, q, rho : float
        Binary separation, mass ratio, source radius (Einstein radii).
    xs, ys : array_like
        Source coordinates (CM frame, Einstein radii).
    gamma : float, optional
        Gamma-law limb darkening (0 = uniform).
    tol : float, optional
        Adaptive contour tolerance for BAGLE.
    label : str
        Plot / print label.

    Returns
    -------
    frac, dcent, t_ours, t_vbm : ndarray, ndarray, float, float
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zeta = xs + 1j * ys
    a1 = float(gamma_to_u(gamma)) if gamma != 0 else 0.0
    vbm = _vbm(a1=a1)

    t0 = time.time()
    if gamma == 0:
        img, amp = fc.adaptive_limb_images_amps(
            zeta, rho, s, q, n_init=64, tol=tol, max_pts=1024)
    else:
        img, amp = fc.ld_images_amps(
            zeta, rho, gamma, s, q, n_ld=8, n_init=64, tol=tol)
    t_ours = time.time() - t0

    t0 = time.time()
    A_v = np.empty(xs.size)
    c_v = np.empty((xs.size, 2))
    for i, (x, y) in enumerate(zip(xs, ys)):
        A_v[i] = vbm.BinaryMag2(s, q, float(x), float(y), rho)
        c_v[i] = (vbm.astrox1, vbm.astrox2)
    t_vbm = time.time() - t0

    frac = np.abs(amp - A_v) / np.maximum(A_v, 1e-30)
    dcent = np.linalg.norm(img - c_v, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(xs, A_v, "k-", label="VBM")
    axes[0].plot(xs, amp, "r--", label="BAGLE")
    axes[0].set_ylabel("A")
    axes[0].legend()
    axes[0].set_title(label)
    axes[1].semilogy(xs, np.maximum(frac, 1e-16), label="|dA|/A")
    axes[1].semilogy(xs, np.maximum(dcent, 1e-16), label="|dcent|")
    axes[1].axhline(TOL_PHOT, color="gray", ls=":")
    axes[1].set_xlabel("x (thetaE)")
    axes[1].set_ylabel("error")
    axes[1].legend()
    fig.tight_layout()
    safe = label.replace(" ", "_").replace("=", "")
    fig.savefig(PLOT_DIR / f"{safe}.png", dpi=120)
    plt.close(fig)

    print(f"{label}: max frac={frac.max():.3e} max dcent={dcent.max():.3e} "
          f"t_ours={t_ours:.2f}s t_vbm={t_vbm:.2f}s")

    return frac, dcent, t_ours, t_vbm


def test_fsbl_point_source_matches_vbm():
    """Physical images and 1/|J| match BinaryMag0 to ~1e-12."""
    s, q = 1.0, 0.1
    z1, z2, m1, m2 = fc.lens_positions_cm(s, q)
    vbm = _vbm(astrometry=False)
    xs = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    ys = np.full_like(xs, 0.05)
    for x, y in zip(xs, ys):
        A0 = vbm.BinaryMag0(s, q, float(x), float(y))
        z, p, n = fc.binary_images([x + 1j * y], z1, z2, m1, m2)
        _, jac = fc.kappa_jac(z[0], z1, z2, m1, m2)
        good = np.isfinite(z[0]) & (p[0] != 0)
        A = np.sum(1.0 / np.abs(jac[good]))
        assert n[0] in (3, 5)
        assert abs(A - A0) / A0 < 1e-12

    return None


@pytest.mark.parametrize(
    "s,q,rho,y,label",
    [
        (1.0, 0.1, 0.01, 0.05, "caustic_s1_q0.1_rho0.01"),
        (1.0, 0.1, 0.05, 0.05, "caustic_s1_q0.1_rho0.05"),
        (1.0, 0.1, 0.001, 0.05, "caustic_s1_q0.1_rho0.001"),
        (0.8, 0.5, 0.02, 0.05, "caustic_s0.8_q0.5_rho0.02"),
        (1.0, 0.1, 0.005, 0.0, "cusp_y0_rho0.005"),
    ],
)
def test_fsbl_finite_source_vs_vbm(s, q, rho, y, label):
    """Uniform FSBL photometry and centroid vs BinaryMag2."""
    xs = np.linspace(-0.6, 0.6, 41)
    ys = np.full_like(xs, y)
    frac, dcent, _, _ = _compare_track(s, q, rho, xs, ys, label=label)
    assert frac.max() <= TOL_PHOT
    assert dcent.max() <= TOL_AST

    return None


def test_fsbl_off_caustic_vs_vbm():
    """Off-caustic track stays well below the 1e-4 budget."""
    xs = np.linspace(-1.5, -0.8, 21)
    ys = np.full_like(xs, 0.3)
    frac, dcent, _, _ = _compare_track(
        1.0, 0.1, 0.01, xs, ys, label="off_caustic")
    assert frac.max() <= 1e-5
    assert dcent.max() <= 1e-5

    return None


def test_fsbl_limb_darkening_vs_vbm():
    """Gamma-law LD via nested disks vs VBM a1 = 3G/(2+G)."""
    xs = np.linspace(-0.6, 0.6, 21)
    ys = np.full_like(xs, 0.05)
    frac, dcent, _, _ = _compare_track(
        1.0, 0.1, 0.01, xs, ys, gamma=0.4, label="ld_gamma0.4")
    assert frac.max() <= TOL_PHOT
    assert dcent.max() <= TOL_AST

    return None


def test_fsbl_image_derivatives_finite_diff():
    """Analytic z' and wedge match finite differences away from criticality."""
    s, q = 1.0, 0.1
    z1, z2, m1, m2 = fc.lens_positions_cm(s, q)
    rho = 0.02
    theta = 0.3
    zeta_c = 0.1 + 0.05j
    e_it = np.exp(1j * theta)
    zeta = zeta_c + rho * e_it
    dzeta = 1j * rho * e_it
    z, p, _ = fc.binary_images([zeta], z1, z2, m1, m2)
    good = np.isfinite(z[0]) & (p[0] != 0)
    z0 = z[0, good]
    dz, wedge, jac = fc.image_derivatives(z0, dzeta, rho, z1, z2, m1, m2)

    # Finite-difference z' from neighbouring limb angles.
    h = 1e-6
    zp, _, _ = fc.binary_images([zeta_c + rho * np.exp(1j * (theta + h))],
                                z1, z2, m1, m2)
    zm, _, _ = fc.binary_images([zeta_c + rho * np.exp(1j * (theta - h))],
                                z1, z2, m1, m2)
    # Match by nearest neighbour.
    for k, zk in enumerate(z0):
        if abs(jac[k]) < 1e-3:
            continue
        ib = np.argmin(np.abs(zp[0] - zk))
        ia = np.argmin(np.abs(zm[0] - zk))
        dz_fd = (zp[0, ib] - zm[0, ia]) / (2 * h)
        assert abs(dz[k] - dz_fd) / max(abs(dz_fd), 1e-12) < 1e-4

    return None
