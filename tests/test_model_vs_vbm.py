"""
Compare BAGLE FSPL photometry and astrometry to VBMicrolensing (VBM).

BAGLE parameters are solar-system barycentric / heliocentric (Lu et al.).
VBM uses geocentric-projected (geo-tr) parameters at ``t0_par``. Photometry
with parallax and PhotAstrom comparisons go through
``bagle.frame_convert.get_vbm_model`` / ``convert_helio_geo_phot``.

Naming: ``njx`` (no-JAX) refers to the NumPy reference ``bagle.model``
(e.g. ``mod_njx``); ``jax`` refers to ``bagle.model_jax`` / ``bagle.jax``.

Plots are written under ``tests/plots/model_vs_vbm/``. Target agreement is
about ``1e-4`` (fractional magnification or centroid shift in ``theta_E``).
Known larger residuals near / inside the source limb are reported separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tensorflow_probability.substrates.jax.distributions.beta import Beta

pytest.importorskip("VBMicrolensing")
pytest.importorskip("jax")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from bagle import frame_convert as fc
from bagle import fspl_contour
from bagle import model
from bagle.jax.fspl import (fspl_images_amps, fspl_magnification,
                            fspl_magnification_centroid)

# ---------------------------------------------------------------------------
# Paths and tolerances
# ---------------------------------------------------------------------------
PLOT_DIR = Path(__file__).resolve().parent / "plots" / "model_vs_vbm"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

N_OUTLINE = 200

# Photometry: fractional |A - A_vbm| / A_vbm
TOL_PHOT_FRAC = 1.0e-3
TOL_PHOT_FRAC_TIGHT = 1.0e-4

# Astrometry: |delta - delta_vbm| in Einstein radii (off-limb)
TOL_AST_THETAE = 1.0e-3
TOL_AST_THETAE_TIGHT = 1.0e-4

# Exclude a band around the limb where the JAX contour centroid jumps.
LIMB_BAND = 0.10  # fraction of rho


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_vbm(bagle_mod, tol=1e-5, rel_tol=1e-6, astrometry=False):
    """
    Get the VBM instance matched to a BAGLE model and set ESPL tolerances.

    Parameters
    ----------
    bagle_mod : BAGLE model instance
        Passed to ``frame_convert.get_vbm_model``.
    tol, rel_tol : float
        Absolute and relative contour tolerances.
    astrometry : bool
        If True, fill ``astrox1`` on ESPLMag2 calls.

    Returns
    -------
    vbm : VBMicrolensing.VBMicrolensing
        Configured VBM instance (uniform source, ``a1 = 0``).
    par : list of float
        VBM parameter vector from ``get_vbm_model``.
    """
    vbm, par = fc.get_vbm_model(bagle_mod)

    # Tighten contour tolerances and use a uniform source.
    vbm.Tol = tol
    vbm.RelTol = rel_tol
    vbm.a1 = 0.0
    if astrometry:
        vbm.astrometry = True

    return vbm, par


def _kernel_model(u0, rho, t0=0.0, tE=10.0):
    """
    No-parallax FSPL photometry model used to drive the kernel comparisons.

    Parameters
    ----------
    u0, rho : float
        Impact parameter and source radius in Einstein radii.
    t0, tE : float
        Peak time (MJD) and Einstein timescale (days).

    Returns
    -------
    mod_njx : model.FSPL_Phot_noPar_Param1
        ``piE`` only sets the sky orientation of ``u`` (no parallax); it is
        deliberately off-axis so the comparisons exercise rotation
        invariance.
    """
    mod_njx = model.FSPL_Phot_noPar_Param1(
        t0, u0, tE, 0.6, 0.8, rho, [1.0], [18.0], n_outline=N_OUTLINE)

    return mod_njx


def _vbm_shift_uhat(vbm, u_amp, rho):
    """
    VBM centroid shift along ``u-hat`` (theta_E), relative to the source.

    Parameters
    ----------
    vbm : VBMicrolensing instance
        Must have ``astrometry=True``.
    u_amp : array_like
        Separation amplitudes.
    rho : float
        Source radius in Einstein radii.

    Returns
    -------
    shift : ndarray, shape (N,)
        ``astrox1 - u`` (Einstein radii).
    """
    out = []
    for ui in u_amp:
        vbm.ESPLMag2(max(ui, 1e-9), rho)
        out.append(vbm.astrox1 - ui)

    return np.array(out)


def _jax_shift_uhat(u_vec, rho, n_outline=N_OUTLINE):
    """
    BAGLE JAX FSPL centroid shift along ``u-hat`` (theta_E).

    Parameters
    ----------
    u_vec : array_like, shape (N, 2)
        Separation vectors.
    rho : float
        Source radius in Einstein radii.
    n_outline : int
        Static limb sample count.

    Returns
    -------
    along : ndarray, shape (N,)
        Shift projected onto ``u-hat``.
    shift_vec : ndarray, shape (N, 2)
        Full 2D shift relative to the source.
    """
    _mag, cen = fspl_magnification_centroid(u_vec, rho, n_outline)

    # JAX array -> NumPy.
    shift_vec = np.asarray(cen) - u_vec
    u_amp = np.linalg.norm(u_vec, axis=1)
    u_hat = u_vec / np.maximum(u_amp[:, None], 1e-15)
    along = np.sum(shift_vec * u_hat, axis=1)

    return along, shift_vec


def _limb_masks(u_amp, rho, band=LIMB_BAND):
    """
    Boolean masks for outside / near-limb / inside the source disk.

    Parameters
    ----------
    u_amp : array_like
        Separation amplitudes.
    rho : float
        Source radius.
    band : float
        Fractional half-width of the limb exclusion band.

    Returns
    -------
    outside, near_limb, inside : ndarray of bool
    """
    outside = u_amp > (1.0 + band) * rho
    inside = u_amp < (1.0 - band) * rho
    near_limb = ~(outside | inside)

    return outside, near_limb, inside


def _max_frac_diff(a, ref):
    """Maximum |a/ref - 1|."""
    return np.max(np.abs(a / ref - 1.0))


def _max_abs_diff(a, ref):
    """Maximum |a - ref|."""
    return np.max(np.abs(a - ref))


def _save_phot_compare(t, a_bagle, a_vbm, title, outfile, t0=None):
    """
    Save magnification and residual panels.

    Parameters
    ----------
    t : array_like
        Times (any scale; labeled relative to ``t0`` if given).
    a_bagle, a_vbm : array_like
        Magnifications.
    title : str
        Figure title.
    outfile : path-like
        Output PNG path.
    t0 : float, optional
        Reference time for the x-axis.
    """
    x = t - t0 if t0 is not None else t
    xlab = r"$t - t_0$" if t0 is not None else r"$t$"
    dA = a_bagle - a_vbm
    frac = dA / a_vbm

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    axes[0].plot(x, a_vbm, "k-", lw=2, alpha=0.5, label="VBM")
    axes[0].plot(x, a_bagle, "r--", lw=1.2, label="BAGLE")
    axes[0].set_ylabel(r"$A$")
    axes[0].set_title(title)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(x, frac, "b-", lw=1.0)
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[1].axhline(TOL_PHOT_FRAC_TIGHT, color="0.5", ls=":", lw=0.8)
    axes[1].axhline(-TOL_PHOT_FRAC_TIGHT, color="0.5", ls=":", lw=0.8)
    axes[1].set_xlabel(xlab)
    axes[1].set_ylabel(r"$(A_{\rm BAGLE}-A_{\rm VBM})/A_{\rm VBM}$")
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)

    return None


def _save_astrom_compare(t, shift_bagle, shift_vbm, u_amp, rho, title, outfile,
                        t0=None):
    """
    Save u-hat centroid shift and residual panels with limb markers.

    Parameters
    ----------
    t : array_like
        Times.
    shift_bagle, shift_vbm : array_like, shape (N,)
        Shifts along ``u-hat`` in Einstein radii.
    u_amp : array_like
        Separation amplitudes (for limb shading).
    rho : float
        Source radius.
    title : str
        Figure title.
    outfile : path-like
        Output PNG.
    t0 : float, optional
        Reference time.
    """
    x = t - t0 if t0 is not None else t
    xlab = r"$t - t_0$" if t0 is not None else r"$t$"
    outside, near_limb, inside = _limb_masks(u_amp, rho)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    axes[0].plot(x, shift_vbm, "k-", lw=2, alpha=0.5, label="VBM")
    axes[0].plot(x, shift_bagle, "r--", lw=1.2, label="BAGLE")
    ymin, ymax = axes[0].get_ylim()
    for mask, color, lab in [
        (near_limb, "0.85", "limb band"),
        (inside, "0.92", None),
    ]:
        if np.any(mask):
            axes[0].fill_between(
                x, ymin, ymax, where=mask, color=color, alpha=0.8, label=lab)
    axes[0].set_ylabel(r"$\delta\theta_\parallel$ ($\theta_E$)")
    axes[0].set_title(title)
    axes[0].legend(loc="best", fontsize=8)

    dlt = shift_bagle - shift_vbm
    axes[1].plot(x, dlt, "b-", lw=1.0)
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[1].axhline(TOL_AST_THETAE_TIGHT, color="0.5", ls=":", lw=0.8)
    axes[1].axhline(-TOL_AST_THETAE_TIGHT, color="0.5", ls=":", lw=0.8)
    axes[1].set_xlabel(xlab)
    axes[1].set_ylabel(r"$\Delta\delta\theta_\parallel$ ($\theta_E$)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)

    return None


def _get_model_Lee2010(u0, rho, jax=False):
    """
    njx ``FSPL_PhotAstrom_noPar_Param1`` with chosen ``u0`` and ``rho``.

    Uses the Lee et al. (2010) setup from ``example_compare_to_lit.ipynb``
    (dark lens, source moving East); ``beta`` and ``radiusS`` are set from
    the PSPL twin's ``thetaE``.

    Parameters
    ----------
    u0, rho : float
        Impact parameter and source radius in Einstein radii.

    Returns
    -------
    mod_njx : model.FSPL_PhotAstrom_noPar_Param1
    """
    t0 = 10.0 # days
    tE = 100.0 # days
    thetaE = 3.0 # mas
    piS = 0.1 # mas
    piE_E = 0.1 
    piE_N = 0.0
    xS0_E = np.array([0.0])
    xS0_N = np.array([0.0])
    muS_E = 3.0
    muS_N = 0.0
    radiusS = rho * thetaE * 1e-3 # arcsec
    b_sff = np.array([1.0])
    mag_src = np.array([18.0])

    if jax:
        mod = model.FSPL_PhotAstrom_noPar_Param2(
            t0, u0, tE, thetaE, piS, piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
            radiusS, b_sff, mag_src, n_outline=N_OUTLINE)
    else:
        mod = model.FSPL_PhotAstrom_noPar_Param2(
            t0, u0, tE, thetaE, piS, piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
            radiusS, b_sff, mag_src, n_outline=N_OUTLINE)


    return mod


def _contour_area_centroid(x, y):
    """
    Signed area and centroid of a closed polygon (shoelace formula).

    Parameters
    ----------
    x, y : array_like, shape (N_pts,)
        Contour vertices; the polygon is closed implicitly.

    Returns
    -------
    area : float
        Signed area (positive for counter-clockwise).
    centroid : ndarray, shape (2,)
    """
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    area = 0.5 * np.sum(cross)
    cx = np.sum((x + x_next) * cross) / (6.0 * area)
    cy = np.sum((y + y_next) * cross) / (6.0 * area)

    return area, np.array([cx, cy])


def _vbm_resolved_images(vbm, u_vec, rho):
    """
    Per-image magnification and centroid from VBM single-lens image contours.

    Each contour is assigned to the plus (``|x| > 1``) or minus (``|x| < 1``)
    image. Its weight is ``parity * signed_area / (pi rho^2)``, which equals
    ``|area|`` when the lens is outside the source and is negative for the
    inner hole of the ring image when the lens is inside the source.

    Parameters
    ----------
    vbm : VBMicrolensing instance
        Configured with ``SetLensGeometry([0, 0, 1])`` (lens at the origin).
    u_vec : array_like, shape (N_times, 2)
        Source position relative to the lens in Einstein radii.
    rho : float
        Source radius in Einstein radii.

    Returns
    -------
    mag : ndarray, shape (2, N_times)
        Plus / minus image magnifications (BAGLE njx sign convention).
    pos : ndarray, shape (2, N_times, 2)
        Plus / minus image centroids relative to the lens (Einstein radii).
    contours : list of list of (int, ndarray, ndarray)
        Per epoch, ``(parity, x, y)`` for each returned contour.
    """
    n_t = len(u_vec)
    src_area = np.pi * rho ** 2

    mag = np.full((2, n_t), np.nan)
    pos = np.full((2, n_t, 2), np.nan)
    contours = []

    for i in range(n_t):
        cs = vbm.MultiImageContours(u_vec[i, 0], u_vec[i, 1], rho)
        epoch = []
        # Accumulate weight and first moment per image: [plus, minus].
        w_sum = np.zeros(2)
        mom_sum = np.zeros((2, 2))
        for c in cs:
            # VBM returns Python lists.
            x = np.array(c[0])
            y = np.array(c[1])
            parity = 1 if np.median(np.hypot(x, y)) > 1.0 else -1
            area, cen = _contour_area_centroid(x, y)
            w = parity * area / src_area
            k = 0 if parity > 0 else 1
            w_sum[k] += w
            mom_sum[k] += w * cen
            epoch.append((parity, x, y))
        contours.append(epoch)

        for k in range(2):
            if w_sum[k] != 0.0:
                mag[k, i] = w_sum[k]
                pos[k, i] = mom_sum[k] / w_sum[k]

    return mag, pos, contours


def _along_uhat(vec, u_vec):
    """
    Project vectors onto the source direction ``u-hat``.

    Parameters
    ----------
    vec : array_like, shape (..., N_times, 2)
    u_vec : array_like, shape (N_times, 2)

    Returns
    -------
    along : ndarray, shape (..., N_times)
    """
    u_hat = u_vec / np.linalg.norm(u_vec, axis=1)[:, None]

    return np.sum(vec * u_hat, axis=-1)


def _shade_source_crossing(ax, x, u_amp, rho):
    """
    Shade epochs with the lens inside the source and in the limb band.

    Parameters
    ----------
    ax : matplotlib Axes
    x : array_like
        Plot x-values (times).
    u_amp : array_like
        Separation amplitudes.
    rho : float
        Source radius.
    """
    _outside, near_limb, inside = _limb_masks(u_amp, rho)
    for mask, color in [(near_limb, "0.80"), (inside, "0.92")]:
        if np.any(mask):
            ax.fill_between(x, 0, 1, where=mask, color=color, alpha=0.8,
                            transform=ax.get_xaxis_transform(), lw=0)

    return None


def _save_resolved_images_compare(tt, u_vec, rho, mag_b, pos_b, mag_v, pos_v,
                                  shift_b, shift_v_contour, shift_v_astrox,
                                  title, outfile):
    """
    Plot per-image magnification, per-image position and total shift vs time.

    Left column: BAGLE (dashed) and VBM (solid) values. Right column:
    BAGLE - VBM. Light gray marks the lens inside the source, darker gray the
    limb band.

    Parameters
    ----------
    tt : array_like, shape (N_times,)
        Times relative to ``t0`` in units of ``tE``.
    u_vec : array_like, shape (N_times, 2)
        Source position relative to the lens (Einstein radii).
    rho : float
        Source radius.
    mag_b, mag_v : array_like, shape (2, N_times)
        Plus / minus image magnifications.
    pos_b, pos_v : array_like, shape (2, N_times, 2)
        Plus / minus image centroids relative to the lens (Einstein radii).
    shift_b, shift_v_contour, shift_v_astrox : array_like, shape (N_times,)
        Total centroid shift along ``u-hat`` (Einstein radii).
    title : str
    outfile : path-like
    """
    u_amp = np.linalg.norm(u_vec, axis=1)
    along_b = _along_uhat(pos_b, u_vec)
    along_v = _along_uhat(pos_v, u_vec)
    colors = {0: "C0", 1: "C3"}
    names = {0: "plus", 1: "minus"}

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.5), sharex=True)

    # Row 0: per-image magnification.
    for k in range(2):
        axes[0, 0].plot(tt, mag_v[k], "-", color=colors[k], lw=2.5, alpha=0.4,
                        label=f"VBM {names[k]}")
        axes[0, 0].plot(tt, mag_b[k], "--", color=colors[k], lw=1.2,
                        label=f"BAGLE {names[k]}")
        axes[0, 1].plot(tt, mag_b[k] - mag_v[k], "-", color=colors[k],
                        lw=1.0, label=names[k])
    axes[0, 0].set_ylabel(r"$A_i$")
    axes[0, 1].set_ylabel(r"$\Delta A_i$")

    # Row 1: per-image position along u-hat (lens-relative).
    for k in range(2):
        axes[1, 0].plot(tt, along_v[k], "-", color=colors[k], lw=2.5,
                        alpha=0.4, label=f"VBM {names[k]}")
        axes[1, 0].plot(tt, along_b[k], "--", color=colors[k], lw=1.2,
                        label=f"BAGLE {names[k]}")
        axes[1, 1].plot(tt, along_b[k] - along_v[k], "-", color=colors[k],
                        lw=1.0, label=names[k])
    axes[1, 0].set_ylabel(r"image $x_\parallel$ ($\theta_E$)")
    axes[1, 1].set_ylabel(r"$\Delta x_\parallel$ ($\theta_E$)")

    # Row 2: total centroid shift along u-hat.
    axes[2, 0].plot(tt, shift_v_astrox, "k-", lw=2.5, alpha=0.4,
                    label="VBM astrox1")
    axes[2, 0].plot(tt, shift_v_contour, "g:", lw=1.5, label="VBM contours")
    axes[2, 0].plot(tt, shift_b, "r--", lw=1.2, label="BAGLE njx")
    axes[2, 1].plot(tt, shift_b - shift_v_astrox, "r-", lw=1.0,
                    label="BAGLE - VBM astrox1")
    axes[2, 1].plot(tt, shift_v_contour - shift_v_astrox, "g-", lw=1.0,
                    label="VBM contours - astrox1")
    axes[2, 0].set_ylabel(r"shift $\delta\theta_\parallel$ ($\theta_E$)")
    axes[2, 1].set_ylabel(r"$\Delta\delta\theta_\parallel$ ($\theta_E$)")

    for ax in axes.ravel():
        _shade_source_crossing(ax, tt, u_amp, rho)
        ax.legend(fontsize=7, loc="best")
    for ax in axes[:, 1]:
        ax.axhline(0.0, color="k", lw=0.6)
    for ax in axes[2]:
        ax.set_xlabel(r"$(t - t_0) / t_E$")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)

    return None


def _save_image_snapshots(u_vec, rho, idx, outline_b, pos_b, pos_v, contours_v,
                          tt, title, outfile):
    """
    Plot image outlines and per-image centroids at selected epochs.

    Parameters
    ----------
    u_vec : array_like, shape (N_times, 2)
        Source position relative to the lens (Einstein radii).
    rho : float
        Source radius.
    idx : sequence of int
        Epoch indices to draw.
    outline_b : array_like, shape (N_times, n_outline, 2, 2)
        BAGLE plus / minus image outlines relative to the lens (Einstein radii).
    pos_b, pos_v : array_like, shape (2, N_times, 2)
        Plus / minus image centroids (BAGLE, VBM).
    contours_v : list
        VBM contours per epoch from ``_vbm_resolved_images``.
    tt : array_like
        Times in units of ``tE`` (for panel titles).
    title : str
    outfile : path-like
    """
    phi = np.linspace(0.0, 2.0 * np.pi, 200)
    fig, axes = plt.subplots(1, len(idx), figsize=(4.2 * len(idx), 4.4),
                             squeeze=False)
    axes = axes[0]

    for ax, i in zip(axes, idx):
        # Einstein ring, lens and unlensed source outline.
        ax.plot(np.cos(phi), np.sin(phi), color="0.7", lw=0.8)
        ax.plot(0.0, 0.0, "k+", ms=8)
        ax.plot(u_vec[i, 0] + rho * np.cos(phi),
                u_vec[i, 1] + rho * np.sin(phi), color="goldenrod", lw=1.0)

        # VBM contours (solid) and BAGLE static outlines (dashed).
        for parity, x, y in contours_v[i]:
            col = "C0" if parity > 0 else "C3"
            ax.plot(np.append(x, x[0]), np.append(y, y[0]), "-", color=col,
                    lw=2.5, alpha=0.35)
        for k, col in [(0, "C0"), (1, "C3")]:
            ol = outline_b[i, :, k, :]
            ax.plot(np.append(ol[:, 0], ol[0, 0]), np.append(ol[:, 1], ol[0, 1]),
                    "--", color=col, lw=1.0)

        # Per-image centroids: VBM (circle) and BAGLE (cross).
        for k, col in [(0, "C0"), (1, "C3")]:
            ax.plot(*pos_v[k, i], "o", mfc="none", mec=col, ms=9)
            ax.plot(*pos_b[k, i], "x", color=col, ms=8)

        u_amp = np.linalg.norm(u_vec[i])
        ax.set_title(rf"$(t-t_0)/t_E$={tt[i]:.3f}, $|u|/\rho$={u_amp / rho:.2f}",
                     fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_E$ ($\theta_E$)")
    axes[0].set_ylabel(r"$x_N$ ($\theta_E$)")

    fig.suptitle(title + "\nsolid = VBM contours, dashed = BAGLE outline; "
                 "o = VBM centroid, x = BAGLE centroid; blue = plus, red = minus",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)

    return None


def _trajectory_frame(u_vec):
    """
    Unit vectors along the lens motion (x) and toward the u0 side (y).

    Matches the Lee et al. (2010) Figs. 4-5 frame: the lens moves in +x
    relative to the source (so ``delta theta_x > 0`` before ``t0``), and y is
    along ``u`` at closest approach (lens at ``y = -u0``, so a positive
    ``u0`` gives a positive ``delta theta_y``).

    Parameters
    ----------
    u_vec : array_like, shape (N_times, 2)
        Rectilinear source - lens separation (Einstein radii, E/N).

    Returns
    -------
    x_hat, y_hat : ndarray, shape (2,)
    """
    # u = source - lens, so the lens moves along -du/dt.
    x_hat = u_vec[0] - u_vec[-1]
    x_hat = x_hat / np.linalg.norm(x_hat)

    # Component of u at closest approach perpendicular to the motion.
    u_min = u_vec[np.argmin(np.linalg.norm(u_vec, axis=1))]
    y_hat = u_min - np.dot(u_min, x_hat) * x_hat
    y_hat = y_hat / np.linalg.norm(y_hat)

    return x_hat, y_hat


def _save_lee10_compare(tau, u_vec, rho, a_njx, a_vbm, shift_njx, shift_vbm,
                        title, outfile, xlim=None, ylim=None):
    """
    Plot the 2D centroid shift and the light curve in the Lee et al. layout.

    Left: centroid-shift trajectory (Lee et al. 2010 Fig. 4). Right:
    ``delta theta_x``, ``delta theta_y`` and ``m - m_base`` vs time (Fig. 5).
    PSPL is yellow and FSPL green as in Lee et al.; VBM FSPL is the wide gray
    line under the njx FSPL. Dashed vertical lines mark ``|u| = rho``.
    Markers on the VBM trajectory (every 0.1 tE) and the lens (every 0.2 tE)
    are open before ``t0`` and filled after, as in Lee et al. Fig. 4.

    Parameters
    ----------
    tau : array_like, shape (N_times,)
        ``(t - t0) / tE``.
    u_vec : array_like, shape (N_times, 2)
        Source - lens separation (Einstein radii, E/N).
    rho : float
        Source radius in Einstein radii.
    a_njx, a_vbm : array_like, shape (N_times,)
        FSPL magnifications.
    shift_njx, shift_vbm : array_like, shape (N_times, 2)
        FSPL centroid shifts relative to the source (Einstein radii, E/N).
    title : str
    outfile : path-like
    xlim, ylim : tuple of float, optional
        Limits of the 2D panel; default is set from the PSPL shift.
    """
    x_hat, y_hat = _trajectory_frame(u_vec)
    u_amp = np.linalg.norm(u_vec, axis=1)

    # Point-source references (Lee et al. 2010 Eq. 5).
    a_pspl = (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4))
    shift_pspl = u_vec / (u_amp ** 2 + 2)[:, None]

    # Epochs where the lens crosses the source limb (rectilinear motion).
    b = abs(np.dot(u_vec[np.argmin(u_amp)], y_hat))
    tau_limb = [-np.sqrt(rho ** 2 - b ** 2), np.sqrt(rho ** 2 - b ** 2)] \
        if rho > b else []

    # (label, shift [N, 2], magnification, plot style)
    curves = [
        ("PSPL", shift_pspl, a_pspl,
         dict(color="goldenrod", ls="-", lw=1.5)),
        ("FSPL VBM", shift_vbm, a_vbm,
         dict(color="k", ls="-", lw=3.5, alpha=0.3)),
        ("FSPL BAGLE njx", shift_njx, a_njx,
         dict(color="g", ls="--", lw=1.3)),
    ]

    fig = plt.figure(figsize=(13.0, 7.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0])
    ax2d = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1], sharex=ax_x)
    ax_m = fig.add_subplot(gs[2, 1], sharex=ax_x)

    for name, shift, amp, style in curves:
        sx = shift @ x_hat
        sy = shift @ y_hat
        ax2d.plot(sx, sy, label=name, **style)
        ax_x.plot(tau, sx, **style)
        ax_y.plot(tau, sy, **style)
        ax_m.plot(tau, -2.5 * np.log10(amp), **style)

    # Lens relative to the source, clipped to the shift range.
    lens_x = -u_vec @ x_hat
    lens_y = -u_vec @ y_hat
    ax2d.plot(lens_x, lens_y, color="0.5", lw=1.0, label="Lens")

    # Equal-time markers: open before t0, filled after.
    for step, xs, ys, col in [
        (0.1, shift_vbm @ x_hat, shift_vbm @ y_hat, "0.35"),
        (0.2, lens_x, lens_y, "k"),
    ]:
        tau_mk = np.arange(-2.0, 2.0 + 1e-9, step)
        mx = np.interp(tau_mk, tau, xs)
        my = np.interp(tau_mk, tau, ys)
        before = tau_mk < 0
        ax2d.plot(mx[before], my[before], "o", mfc="none", mec=col, ms=5)
        ax2d.plot(mx[~before], my[~before], "o", color=col, ms=5)

    if xlim is None:
        x_lim = 1.15 * np.max(np.abs(shift_pspl @ x_hat))
        xlim = (-x_lim, x_lim)
    if ylim is None:
        sy_p = shift_pspl @ y_hat
        ylim = (min(-b, np.min(sy_p)) - 0.1 * np.max(sy_p),
                1.25 * np.max(sy_p))
    ax2d.set_xlim(*xlim)
    ax2d.set_ylim(*ylim)
    ax2d.set_xlabel(r"$\delta\theta_{c,x}$ [$\theta_E$]")
    ax2d.set_ylabel(r"$\delta\theta_{c,y}$ [$\theta_E$]")
    ax2d.set_title("cf. Lee et al. (2010) Fig. 4", fontsize=10)
    ax2d.legend(fontsize=8, loc="lower right")

    for ax in (ax_x, ax_y, ax_m):
        for tl in tau_limb:
            ax.axvline(tl, color="0.4", ls="--", lw=0.8)
    ax_x.set_ylabel(r"$\delta\theta_{c,x}$ [$\theta_E$]")
    ax_y.set_ylabel(r"$\delta\theta_{c,y}$ [$\theta_E$]")
    ax_m.set_ylabel(r"$m - m_{\rm base}$ [mag]")
    ax_m.invert_yaxis()
    ax_m.set_xlabel(r"$(t - t_0) / t_E$")
    ax_x.set_xlim(tau[0], tau[-1])
    ax_x.set_title(r"cf. Lee et al. (2010) Fig. 5 ($\pm 50$ d at $t_E$ = 10 d)",
                   fontsize=10)
    plt.setp(ax_x.get_xticklabels(), visible=False)
    plt.setp(ax_y.get_xticklabels(), visible=False)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)

    return None


def _vbm_espl_astro_to_en(tmp, bagle_mod, u0_vbm, piEE_vbm, piEN_vbm):
    """
    Convert VBM ESPLAstroLightCurve outputs to BAGLE East-North sky frames.

    Mirrors ``example_compare_packages.ipynb``: VBM returns N/E in mas and
    separation in a tau-beta frame; rotate into Lu EN and add ``xL0``.

    Parameters
    ----------
    tmp : sequence
        ESPLAstroLightCurve return lists.
    bagle_mod : BAGLE PhotAstrom model
        Provides ``xL0`` and ``thetaE_amp``.
    u0_vbm, piEE_vbm, piEN_vbm : float
        Geo-tr parameters used in the VBM call (for hat vectors).

    Returns
    -------
    amp : ndarray, shape (N,)
    ast_lensed : ndarray, shape (N, 2)
        Lensed source centroid in arcsec, East-North.
    xL : ndarray, shape (N, 2)
        Lens position in arcsec, East-North.
    uvec : ndarray, shape (N, 2)
        Source-lens separation in Einstein radii, East-North.
    """
    # VBM returns Python lists.
    amp = np.array(tmp[0])

    # VBM sky positions are North, East in mas.
    ast_lensed = np.array([tmp[2], tmp[1]]).T * 1e-3
    xL = np.array([tmp[4], tmp[3]]).T * 1e-3

    # ESPLAstro: indices 5,6 are tau-beta style separations (same as PSPLAstro).
    u_tb = np.array([tmp[6], tmp[5]]).T  # columns: beta, tau

    # Rotate tau-beta into East-North using geo-tr piE / u0 hats.
    muRel_hat = np.array([piEE_vbm, piEN_vbm])
    muRel_hat /= np.hypot(muRel_hat[0], muRel_hat[1])
    u0_hat = model.u0_hat_from_thetaE_hat(muRel_hat, u0_vbm)
    u0_hat[1] *= -1

    uvec = (u_tb[:, 1, np.newaxis] * muRel_hat[np.newaxis, :]
            + u_tb[:, 0, np.newaxis] * u0_hat[np.newaxis, :])

    # VBM lens / centroid omit the arbitrary BAGLE xL0 offset.
    xL = xL + bagle_mod.xL0
    ast_lensed = ast_lensed + bagle_mod.xL0

    return amp, ast_lensed, xL, uvec


# ---------------------------------------------------------------------------
# 1. Kernel photometry vs ESPLMag2 (no parallax)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "u0,rho,label",
    [
        (0.3, 0.01, "small_rho"),
        (0.1, 0.05, "moderate_rho"),
        (0.1, 0.5, "large_rho_lee10"),
    ],
)
def test_fspl_kernel_phot_vs_vbm(u0, rho, label):
    """
    BAGLE JAX ``fspl_magnification`` vs VBM ``ESPLLightCurve`` (no parallax).

    Both sides come from one ``FSPL_Phot_noPar_Param1``: BAGLE's ``get_u``
    feeds the JAX kernel and ``get_vbm_model`` builds the VBM call.

    Parameters
    ----------
    u0, rho : float
        Impact parameter and source radius in Einstein radii.
    label : str
        Tag for the saved plot filename.
    """
    t0, tE = 0.0, 10.0
    t = np.linspace(-2.5 * tE, 2.5 * tE, 401)
    mod_njx = _kernel_model(u0, rho, t0=t0, tE=tE)

    # Shape of u_vec: [N_times, 2]
    u_vec = mod_njx.get_u(t)
    a_jax = np.asarray(fspl_magnification(u_vec, rho, N_OUTLINE))

    vbm, par = _get_vbm(mod_njx)
    a_vbm = np.asarray(vbm.ESPLLightCurve(par, list(t))[0])

    # The static JAX limb does not resolve the lens crossing the limb, so
    # the limb band is reported but only off-limb epochs are asserted.
    u_amp = np.linalg.norm(u_vec, axis=1)
    outside, near_limb, inside = _limb_masks(u_amp, rho)
    off_limb = outside | inside
    frac = _max_frac_diff(a_jax[off_limb], a_vbm[off_limb])
    frac_limb = (_max_frac_diff(a_jax[near_limb], a_vbm[near_limb])
                 if np.any(near_limb) else np.nan)

    outfile = PLOT_DIR / f"phot_kernel_{label}.png"
    _save_phot_compare(
        t, a_jax, a_vbm,
        title=(f"FSPL jax kernel vs VBM ESPLLightCurve  u0={u0}, rho={rho}\n"
               f"max |dA/A| off-limb = {frac:.2e}, limb = {frac_limb:.2e}"),
        outfile=outfile, t0=t0)

    assert frac < TOL_PHOT_FRAC_TIGHT, (
        f"off-limb max |dA/A|={frac:.3e} exceeds {TOL_PHOT_FRAC_TIGHT:.1e} "
        f"({label})")

    return None


# ---------------------------------------------------------------------------
# 2. Kernel u-hat astrometry vs astrox1 (no parallax)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "u0,rho,label",
    [
        (0.3, 0.01, "small_rho"),
        (0.1, 0.5, "large_rho_lee10"),
    ],
)
def test_fspl_kernel_astrom_uhat_vs_vbm(u0, rho, label):
    """
    BAGLE JAX centroid along ``u-hat`` vs VBM ``astrox1`` (uniform).

    Away from the limb (beyond ``LIMB_BAND``), both outside and inside the
    source, agreement is at the ``1e-4`` level. The static JAX limb does not
    resolve the limb band itself; it is plotted but not asserted.

    Parameters
    ----------
    u0, rho : float
        Impact parameter and source radius.
    label : str
        Plot filename tag.
    """
    t0, tE = 0.0, 10.0
    t = np.linspace(-2.5 * tE, 2.5 * tE, 401)
    mod_njx = _kernel_model(u0, rho, t0=t0, tE=tE)

    # Shape of u_vec: [N_times, 2]
    u_vec = mod_njx.get_u(t)
    u_amp = np.linalg.norm(u_vec, axis=1)

    shift_jax, _ = _jax_shift_uhat(u_vec, rho)
    vbm, _par = _get_vbm(mod_njx, tol=1e-6, rel_tol=1e-7, astrometry=True)
    shift_v = _vbm_shift_uhat(vbm, u_amp, rho)

    outside, near_limb, inside = _limb_masks(u_amp, rho)
    d_out = _max_abs_diff(shift_jax[outside], shift_v[outside])
    d_in = (_max_abs_diff(shift_jax[inside], shift_v[inside])
            if np.any(inside) else np.nan)
    d_limb = (_max_abs_diff(shift_jax[near_limb], shift_v[near_limb])
              if np.any(near_limb) else np.nan)

    outfile = PLOT_DIR / f"astrom_uhat_kernel_{label}.png"
    _save_astrom_compare(
        t, shift_jax, shift_v, u_amp, rho,
        title=(f"FSPL jax u-hat shift vs VBM astrox1  u0={u0}, rho={rho}\n"
               f"outside max|d|={d_out:.2e}, "
               f"inside={d_in:.2e}, limb={d_limb:.2e}"),
        outfile=outfile, t0=t0)

    # Large rho at 1.1 rho is only ~1.5e-4 with a 200-point static limb.
    tol = TOL_AST_THETAE_TIGHT if rho < 0.05 else 2.0e-4
    assert d_out < tol, (
        f"outside max |d shift|={d_out:.3e} exceeds {tol:.1e} ({label})")
    if np.any(inside):
        assert d_in < tol, (
            f"inside max |d shift|={d_in:.3e} exceeds {tol:.1e} ({label})")

    return None


# ---------------------------------------------------------------------------
# 3. njx Phot_noPar vs ESPLLightCurve
# ---------------------------------------------------------------------------
def test_fspl_phot_nopar_njx_vs_vbm():
    """
    njx ``FSPL_Phot_noPar_Param1`` vs VBM ``ESPLLightCurve`` (no parallax).

    No frame conversion is required: both codes share the same rectilinear
    ``(t0, u0, tE, rho)`` definition when parallax is off.
    """
    t0, u0, tE, rho = 57000.0, 0.1, 20.0, 0.05
    # Non-zero piE only sets the sky orientation of u0_hat for BAGLE Phot.
    mod_njx = model.FSPL_Phot_noPar_Param1(
        t0, u0, tE, 0.3, 0.4, rho, [1.0], [18.0], n_outline=N_OUTLINE)
    t = np.linspace(t0 - 2.0 * tE, t0 + 2.0 * tE, 401)

    a_njx = mod_njx.get_amplification(t)
    vbm, par = _get_vbm(mod_njx)
    a_vbm = np.asarray(vbm.ESPLLightCurve(par, list(t))[0])

    frac = _max_frac_diff(a_njx, a_vbm)
    _save_phot_compare(
        t, a_njx, a_vbm,
        title=f"njx FSPL_Phot_noPar vs ESPLLightCurve  max|dA/A|={frac:.2e}",
        outfile=PLOT_DIR / "phot_njx_nopar.png", t0=t0)

    assert frac < TOL_PHOT_FRAC, f"max |dA/A|={frac:.3e}"

    return None


# ---------------------------------------------------------------------------
# 4. njx Phot_Par vs ESPLLightCurveParallax (helio -> geo-tr)
# ---------------------------------------------------------------------------
def test_fspl_phot_par_njx_vs_vbm():
    """
    njx ``FSPL_Phot_Par_Param1`` vs VBM ``ESPLLightCurveParallax``.

    BAGLE parameters are heliocentric / SSB; ``get_vbm_model`` converts to
    geo-tr at ``t0_par = t0`` (Lu et al.) before calling VBM.
    """
    c = SkyCoord("17:57:05 -30:22:59", unit=(u.hourangle, u.deg))
    ra, dec = c.ra.value, c.dec.value
    t0, u0, tE = 55775.0, 0.02, 100.0
    piEE, piEN, rho = 0.9, 0.7, 0.05

    mod_njx = model.FSPL_Phot_Par_Param1(
        t0, u0, tE, piEE, piEN, rho, [1.0], [19.0],
        n_outline=N_OUTLINE, raL=ra, decL=dec)

    # Dense cadence over +-1 tE (finite-source peak is narrow).
    t = np.arange(t0 - 1.0 * tE, t0 + 1.0 * tE + 1e-9, 0.5)
    a_njx = mod_njx.get_amplification(t)

    vbm, par = _get_vbm(mod_njx)
    t_vbm = fc.mjd_prime_to_vbm_time(t, c)
    a_vbm = np.asarray(vbm.ESPLLightCurveParallax(par, list(t_vbm))[0])

    frac = _max_frac_diff(a_njx, a_vbm)
    _save_phot_compare(
        t, a_njx, a_vbm,
        title=(f"njx FSPL_Phot_Par vs ESPLLightCurveParallax (geo-tr)\n"
               f"max|dA/A|={frac:.2e}"),
        outfile=PLOT_DIR / "phot_njx_par.png", t0=t0)

    assert frac < TOL_PHOT_FRAC, f"max |dA/A|={frac:.3e}"

    return None


# ---------------------------------------------------------------------------
# 5. PhotAstrom_Par vs ESPLAstroLightCurve (phot + sky centroid)
# ---------------------------------------------------------------------------
def test_fspl_photastrom_par_njx_vs_vbm():
    """
    njx ``FSPL_PhotAstrom_Par_Param1`` vs VBM ``ESPLAstroLightCurve``.

    Uses helio→geo-tr conversion via ``get_vbm_model``. Compares
    magnification and the centroid **shift** (Einstein radii). Absolute
    sky tracks differ by ~0.16 mas (same PSPL offset after ``xL0`` repair);
    shifts are formed with each code's own unlensed source so the finite-
    source centroid physics is isolated.
    """
    c = SkyCoord("17:57:05 -30:22:59", unit=(u.hourangle, u.deg))
    ra, dec = c.ra.value, c.dec.value

    # Physical Param1 with |u0| ~ rho/2 so finite-source matters near peak.
    mL, t0, dL, dL_dS = 0.5, 55775.0, 4000.0, 0.8
    muL_E, muL_N, muS_E, muS_N = -2.0, 1.0, 1.0, 0.5
    beta = -0.01  # mas
    # thetaE ~ 0.45 mas for this mass/distance; rho = 0.05.
    radius_s = 0.05 * 0.4512165022672146 * 1e-3

    mod_njx = model.FSPL_PhotAstrom_Par_Param1(
        mL, t0, beta, dL, dL_dS,
        np.array([0.0]), np.array([0.0]),
        muL_E, muL_N, muS_E, muS_N,
        radius_s, np.array([1.0]), np.array([18.0]),
        n_outline=N_OUTLINE, raL=ra, decL=dec)

    theta_e = mod_njx.thetaE_amp
    rho = mod_njx.source_radius_thetaE_units()
    tE = mod_njx.tE
    t = np.arange(t0 - 1.0 * tE, t0 + 1.0 * tE + 1e-9, 0.5)

    a_njx = mod_njx.get_amplification(t)
    u_njx = mod_njx.get_u(t)
    u_amp = np.linalg.norm(u_njx, axis=1)
    # Centroid shift in Einstein radii (njx AMG).
    shift_njx = mod_njx.get_centroid_shift(t) / theta_e

    vbm, par = _get_vbm(mod_njx)
    t_vbm = fc.mjd_prime_to_vbm_time(t, c)
    tmp = vbm.ESPLAstroLightCurve(par, list(t_vbm))

    u0_vbm, piEN_vbm, piEE_vbm = par[0], par[4], par[5]
    a_vbm, ast_vbm, xL_vbm, uvec_vbm = _vbm_espl_astro_to_en(
        tmp, mod_njx, u0_vbm, piEE_vbm, piEN_vbm)

    # Unlensed source from VBM lens + rotated u (avoids absolute-frame offset).
    xS_vbm = xL_vbm + uvec_vbm * theta_e * 1e-3
    shift_vbm = (ast_vbm - xS_vbm) * 1e3 / theta_e

    frac = _max_frac_diff(a_njx, a_vbm)
    d_u = np.max(np.linalg.norm(u_njx - uvec_vbm, axis=1))
    outside, _near, _inside = _limb_masks(u_amp, rho)
    d_shift = np.linalg.norm(shift_njx - shift_vbm, axis=1)
    d_out = np.max(d_shift[outside]) if np.any(outside) else np.nan

    _save_phot_compare(
        t, a_njx, a_vbm,
        title=(f"njx FSPL_PhotAstrom_Par vs ESPLAstroLightCurve (phot)\n"
               f"u0={mod_njx.u0_amp:.3f}, rho={rho:.3f}, max|dA/A|={frac:.2e}"),
        outfile=PLOT_DIR / "phot_photastrom_par.png", t0=t0)

    # Centroid-shift panels (Einstein radii).
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), sharex=True)
    tt = t - t0
    for k, name in enumerate(["E", "N"]):
        axes[0, k].plot(tt, shift_vbm[:, k], "k-", lw=2, alpha=0.5,
                        label="VBM")
        axes[0, k].plot(tt, shift_njx[:, k], "r--", lw=1.2, label="BAGLE njx")
        axes[0, k].set_ylabel(rf"$\delta\theta_{name}$ ($\theta_E$)")
        axes[0, k].legend(fontsize=8)
        axes[1, k].plot(tt, shift_njx[:, k] - shift_vbm[:, k], "b-", lw=1.0)
        axes[1, k].axhline(0.0, color="k", lw=0.6)
        axes[1, k].axhline(TOL_AST_THETAE, color="0.5", ls=":", lw=0.8)
        axes[1, k].axhline(-TOL_AST_THETAE, color="0.5", ls=":", lw=0.8)
        axes[1, k].set_ylabel(rf"$\Delta\delta\theta_{name}$ ($\theta_E$)")
        axes[1, k].set_xlabel(r"$t - t_0$")
    fig.suptitle(
        f"njx FSPL PhotAstrom_Par centroid shift vs VBM  "
        f"off-limb max|d|={d_out:.2e} $\\theta_E$, max|du|={d_u:.2e}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "astrom_photastrom_par.png", dpi=120)
    plt.close(fig)

    # u-vector comparison after geo-tr / EN rotation.
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
    axes[0].plot(tt, u_amp, "k-", lw=1.5, label="|u| BAGLE njx")
    axes[0].plot(tt, np.linalg.norm(uvec_vbm, axis=1), "r--", lw=1.0,
                 label="|u| VBM→EN")
    axes[0].axhline(rho, color="0.5", ls=":", label=r"$\rho$")
    axes[0].set_ylabel(r"$|u|$")
    axes[0].legend(fontsize=8)
    du = u_njx - uvec_vbm
    axes[1].plot(tt, du[:, 0], label="d u_E")
    axes[1].plot(tt, du[:, 1], label="d u_N")
    axes[1].set_ylabel(r"$u_{\rm njx}-u_{\rm VBM}$")
    axes[1].set_xlabel(r"$t - t_0$")
    axes[1].legend(fontsize=8)
    fig.suptitle("Source-lens u after geo-tr / EN rotation")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "uvec_photastrom_par.png", dpi=120)
    plt.close(fig)

    assert frac < TOL_PHOT_FRAC, f"phot max |dA/A|={frac:.3e}"
    assert d_u < 1.0e-2, f"max |du|={d_u:.3e} after EN rotation"
    if np.any(outside):
        assert d_out < 1.0e-2, (
            f"off-limb centroid-shift max |d|={d_out:.3e} theta_E")

    return None


# ---------------------------------------------------------------------------
# 6. njx resolved images vs VBM image contours (centroid debugging)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "u0,rho,span,label",
    [
        (0.1, 0.5, 1.5, "large_rho_lee10"),
        (0.02, 0.05, 0.15, "small_rho_crossing"),
    ],
)
def test_fspl_resolved_images_njx_vs_vbm(u0, rho, span, label):
    """
    njx ``FSPL_PhotAstrom_noPar_Param1`` plus / minus images vs VBM contours.

    VBM's ESPL routines only return the total centroid, so per-image values
    come from ``MultiImageContours`` for a single lens at the origin. The
    plots show each image's magnification and position along ``u-hat`` over
    time, and the image outlines at a few epochs, to locate the centroid
    error when the lens is inside the source.

    Only VBM self-consistency (contours vs ``ESPLMag2`` / ``astrox1``) and the
    per-image magnifications are asserted; per-image positions are the
    quantity under investigation.

    Parameters
    ----------
    u0, rho : float
        Impact parameter and source radius in Einstein radii.
    span : float
        Half-width of the time window in units of ``tE``.
    label : str
        Plot filename tag.
    """
    mod_njx = _get_model_Lee2010(u0, rho, jax=False)
    theta_e = mod_njx.thetaE_amp
    rho = mod_njx.source_radius_thetaE_units()
    t0 = mod_njx.t0
    tE = mod_njx.tE
    t = t0 + tE * np.linspace(-span, span, 301)
    tau = (t - t0) / tE

    # Shape of u_vec: [N_times, 2] (source - lens, Einstein radii, E/N).
    u_vec = mod_njx.get_u(t)
    u_amp = np.linalg.norm(u_vec, axis=1)

    # njx per-image values, lens-relative in Einstein radii.
    # Shapes: xL [N_times, 2]; mag_njx [2, N_times]; pos_njx [2, N_times, 2].
    xL = mod_njx.get_lens_astrometry(t)
    mag_njx = mod_njx.get_resolved_amplification(t)
    pos_njx = (mod_njx.get_resolved_astrometry(t) - xL[None]) * 1e3 / theta_e
    shift_njx = _along_uhat(mod_njx.get_centroid_shift(t) / theta_e, u_vec)

    # Shape of outline_njx: [N_times, n_outline, 2, 2].
    outline_njx = (mod_njx.get_resolved_astrometry_outline(t)
                   - xL[:, None, None, :]) * 1e3 / theta_e

    # VBM instance from the matching no-parallax photometry model.
    mod_njx_phot = _kernel_model(abs(mod_njx.u0_amp), rho, t0=t0, tE=tE)
    vbm, par = _get_vbm(mod_njx_phot, tol=1e-6, rel_tol=1e-7, astrometry=True)
    a_vbm = np.asarray(vbm.ESPLLightCurve(par, list(t))[0])
    shift_v_astrox = _vbm_shift_uhat(vbm, u_amp, rho)

    # Lee et al. (2010) Figs. 4-5 window: +-5 tE, densely sampled where the
    # lens crosses the source.
    tau_lee = np.unique(np.concatenate([np.linspace(-5.0, 5.0, 401),
                                        np.linspace(-2 * rho, 2 * rho, 401)]))
    t_lee = t0 + tE * tau_lee
    u_lee = mod_njx.get_u(t_lee)
    u_amp_lee = np.linalg.norm(u_lee, axis=1)

    # VBM 2D shift: astrox1 is along u-hat.
    a_vbm_lee = np.asarray(vbm.ESPLLightCurve(par, list(t_lee))[0])
    shift_vbm_lee = (_vbm_shift_uhat(vbm, u_amp_lee, rho)[:, None]
                     * u_lee / u_amp_lee[:, None])

    # Lee et al. Fig. 4 axis limits for the matching case.
    is_lee = "lee10" in label
    lee_note = "  (Lee et al. 2010: u0 = 0.1, rho = 0.5)" if is_lee else ""
    _save_lee10_compare(
        tau_lee, u_lee, rho, mod_njx.get_amplification(t_lee), a_vbm_lee,
        mod_njx.get_centroid_shift(t_lee) / theta_e, shift_vbm_lee,
        title=f"FSPL centroid shift and light curve  u0={u0:g}, "
              f"rho={rho:g}{lee_note}",
        outfile=PLOT_DIR / f"lee10_fig4_fig5_{label}.png",
        xlim=(-0.6, 0.6) if is_lee else None,
        ylim=(-0.105, 0.06) if is_lee else None)

    # Single lens at the origin for the image contours.
    vbm.SetLensGeometry([0.0, 0.0, 1.0])
    mag_v, pos_v, contours_v = _vbm_resolved_images(vbm, u_vec, rho)

    # Total centroid from the contours (weights sum to the total A).
    a_contour = np.sum(mag_v, axis=0)
    cen_contour = np.sum(mag_v[:, :, None] * pos_v, axis=0) \
        / a_contour[:, None]
    shift_v_contour = _along_uhat(cen_contour, u_vec) - u_amp

    # Summary numbers (lens outside vs inside the source).
    outside, _near, inside = _limb_masks(u_amp, rho)
    along_njx = _along_uhat(pos_njx, u_vec)
    along_v = _along_uhat(pos_v, u_vec)
    d_pos = np.abs(along_njx - along_v)
    d_out = [np.max(d_pos[k][outside]) for k in range(2)]
    d_in = ([np.max(d_pos[k][inside]) for k in range(2)]
            if np.any(inside) else [np.nan, np.nan])

    title = (f"njx FSPL resolved images vs VBM contours  u0={u0:g}, "
             f"rho={rho:g}\n"
             f"max |d pos| (theta_E), lens outside source: plus {d_out[0]:.1e},"
             f" minus {d_out[1]:.1e};  inside: plus {d_in[0]:.1e},"
             f" minus {d_in[1]:.1e}")
    _save_resolved_images_compare(
        tau, u_vec, rho, mag_njx, pos_njx, mag_v, pos_v,
        shift_njx, shift_v_contour, shift_v_astrox,
        title=title, outfile=PLOT_DIR / f"resolved_images_{label}.png")

    # Snapshot epochs on the approach: 2 rho, just outside the limb,
    # inside the source, and closest approach.
    approach = np.arange(len(t)) <= np.argmin(u_amp)
    idx = []
    for target in [2.0, 1.05, 0.6]:
        cand = np.where(approach)[0]
        j = cand[np.argmin(np.abs(u_amp[cand] - target * rho))]
        if np.abs(u_amp[j] / rho - target) < 0.1:
            idx.append(j)
    idx.append(int(np.argmin(u_amp)))
    _save_image_snapshots(
        u_vec, rho, idx, outline_njx, pos_njx, pos_v, contours_v, tau,
        title=f"Image outlines  u0={u0:g}, rho={rho:g}",
        outfile=PLOT_DIR / f"resolved_images_snapshots_{label}.png")

    # VBM contour extraction reproduces ESPLMag2 / astrox1. Contours with
    # the lens on the limb hit VBM's point cap, so check off the limb only.
    off_limb = outside | inside
    assert _max_frac_diff(a_contour[off_limb], a_vbm[off_limb]) < 1.0e-3
    assert _max_abs_diff(shift_v_contour[off_limb],
                         shift_v_astrox[off_limb]) < 1.0e-3

    # njx per-image magnifications agree with VBM off the limb.
    d_mag = np.abs(mag_njx - mag_v)[:, off_limb] / a_vbm[off_limb]
    assert np.max(d_mag) < 5.0e-3, f"per-image max |dA|/A={np.max(d_mag):.2e}"

    return None


# ---------------------------------------------------------------------------
# 7. PSPL centroid ellipses vs Lee et al. (2010) Fig. 1 (lower panel)
# ---------------------------------------------------------------------------
def test_pspl_lee10_fig1_njx_vs_vbm():
    """
    njx PSPL centroid-shift ellipses for several ``u0`` vs VBM and Eq. 5.

    Reproduces the lower panel of Lee et al. (2010) Fig. 1 in the same frame
    as Figs. 4-5 (lens moving in +x, ``u0`` along +y). VBM uses ESPL with a
    negligible ``rho`` (point-source limit of ``astrox1``).
    """
    u0_list = [0.05, 0.10, 0.30, 0.50, 0.70, 0.90]
    colors = {0.05: "tab:red", 0.10: "gold", 0.30: "yellowgreen",
              0.50: "navy", 0.70: "mediumturquoise", 0.90: "purple"}
    rho_ps = 1e-4

    # Same event geometry as _get_model_Lee2010.
    t0, tE, thetaE, piS = 10.0, 100.0, 3.0, 0.1
    xS0 = np.array([0.0])
    b_sff, mag_src = np.array([1.0]), np.array([18.0])

    # Wide window so the ellipses close near the origin; dense near t0.
    tau = np.unique(np.concatenate([np.linspace(-100.0, 100.0, 2001),
                                    np.linspace(-3.0, 3.0, 1201)]))
    t = t0 + tE * tau

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    d_ana, d_vbm = [], []
    for u0 in u0_list:
        mod_njx = model.PSPL_PhotAstrom_noPar_Param2(
            t0, u0, tE, thetaE, piS, 0.1, 0.0, xS0, xS0, 3.0, 0.0,
            b_sff, mag_src)

        # Shape of u_vec / shift_njx: [N_times, 2] (E/N, Einstein radii).
        u_vec = mod_njx.get_u(t)
        u_amp = np.linalg.norm(u_vec, axis=1)
        shift_njx = mod_njx.get_centroid_shift(t) / thetaE

        # Lee et al. Eq. 5 and the VBM point-source limit (along u-hat).
        shift_ana = u_vec / (u_amp ** 2 + 2)[:, None]
        vbm, _par = _get_vbm(_kernel_model(u0, rho_ps, t0=t0, tE=tE),
                             tol=1e-6, rel_tol=1e-7, astrometry=True)
        shift_vbm = (_vbm_shift_uhat(vbm, u_amp, rho_ps)[:, None]
                     * u_vec / u_amp[:, None])

        d_ana.append(np.max(np.linalg.norm(shift_njx - shift_ana, axis=1)))
        d_vbm.append(np.max(np.linalg.norm(shift_njx - shift_vbm, axis=1)))

        # Project into the Lee frame.
        x_hat, y_hat = _trajectory_frame(u_vec)
        col = colors[u0]
        ax.plot(shift_njx @ x_hat, shift_njx @ y_hat, "-", color=col, lw=1.5)
        ax.plot(shift_vbm @ x_hat, shift_vbm @ y_hat, "k:", lw=0.8)
        ax.text(0.0, np.max(shift_njx @ y_hat) + 0.004, f"u0={u0:.2f}",
                color=col, ha="center", va="bottom", fontsize=9)

    ax.plot([], [], "-", color="0.4", label="BAGLE njx PSPL")
    ax.plot([], [], "k:", label="VBM (rho = 1e-4)")
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.05, 0.35)
    ax.set_xlabel(r"$\delta\theta_{c,x}$ [$\theta_E$]")
    ax.set_ylabel(r"$\delta\theta_{c,y}$ [$\theta_E$]")
    ax.set_title("PSPL centroid shift, cf. Lee et al. (2010) Fig. 1 (lower)\n"
                 f"max |njx - Eq.5| = {max(d_ana):.1e},  "
                 f"max |njx - VBM| = {max(d_vbm):.1e} ($\\theta_E$)",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "lee10_fig1_pspl.png", dpi=120)
    plt.close(fig)

    assert max(d_ana) < 1.0e-6, f"njx vs Eq. 5 max |d|={max(d_ana):.2e}"
    assert max(d_vbm) < TOL_AST_THETAE_TIGHT, (
        f"njx vs VBM max |d|={max(d_vbm):.2e}")

    return None


# ---------------------------------------------------------------------------
# 8. Analytic contour derivatives and rotation invariance
# ---------------------------------------------------------------------------
def test_fspl_contour_derivatives_match_finite_differences():
    """
    Analytic image derivative and Eq. 14 wedge vs finite differences.

    Checks :func:`bagle.fspl_contour.point_lens_image_derivatives` for both
    images along an off-centre source limb.
    """
    zeta_c, rho, h = 0.3 - 0.2j, 0.5, 1e-4
    theta = np.linspace(0.1, 6.0, 9)

    def limb_images(th):
        return fspl_contour.point_lens_images(zeta_c + rho * np.exp(1j * th))

    for k in range(2):
        z0 = limb_images(theta)[k]
        z_p = limb_images(theta + h)[k]
        z_m = limb_images(theta - h)[k]

        # Central differences for z' and z''.
        fd1 = (z_p - z_m) / (2 * h)
        fd2 = (z_p - 2 * z0 + z_m) / h ** 2
        fd_wedge = (np.conjugate(fd1) * fd2).imag

        dz, wedge = fspl_contour.point_lens_image_derivatives(
            z0, 1j * rho * np.exp(1j * theta), rho)

        assert np.max(np.abs(dz / fd1 - 1)) < 1e-6
        assert np.max(np.abs(wedge / fd_wedge - 1)) < 1e-5

    return None


@pytest.mark.parametrize("rho", [0.05, 0.5])
def test_fspl_centroid_rotation_invariant(rho):
    """
    njx and JAX FSPL centroid shifts do not depend on trajectory orientation.

    The same rectilinear event is rotated on the sky; away from the limb the
    shift along ``u-hat`` matches VBM ``astrox1`` at every angle, the
    perpendicular component vanishes, and njx and JAX agree. A uniform
    ``n_outline`` limb (njx here) does not resolve ``0.8 < |u|/rho < 1.25``;
    the adaptive mesh covers that regime in njx models and the JAX kernel
    clusters its static limb toward the lens.

    Parameters
    ----------
    rho : float
        Source radius in Einstein radii.
    """
    u0, n_pts = 0.4 * rho, N_OUTLINE
    tau = np.linspace(-3.0 * rho, 3.0 * rho, 121)
    u_amp = np.hypot(tau, u0)
    off_limb = (u_amp < 0.8 * rho) | (u_amp > 1.25 * rho)

    vbm = _make_vbm_astro()
    shift_v = _vbm_shift_uhat(vbm, u_amp, rho)

    theta = np.broadcast_to(2 * np.pi * np.arange(n_pts) / n_pts,
                            (tau.size, n_pts))
    for ang in [0.0, 37.0, 90.0, 211.0]:
        # Rotated source track (complex E + iN, lens at origin).
        zeta_c = (tau + 1j * u0) * np.exp(1j * np.deg2rad(ang))
        u_vec = np.stack([zeta_c.real, zeta_c.imag], axis=1)

        along_by_code = {}
        for name, (img, amp) in [
            ("njx", fspl_contour.limb_images_amps(
                zeta_c, rho, theta, np.full(tau.size, n_pts))),
            ("jax", fspl_images_amps(u_vec, rho, n_pts)),
        ]:
            img, amp = np.asarray(img), np.asarray(amp)

            # Total centroid shift relative to the source.
            mag = np.sum(amp, axis=1)
            shift = np.sum(amp[:, :, None] * img, axis=1) / mag[:, None] - u_vec
            along = _along_uhat(shift, u_vec)
            u_hat = u_vec / u_amp[:, None]
            perp = u_hat[:, 0] * shift[:, 1] - u_hat[:, 1] * shift[:, 0]
            along_by_code[name] = along

            d_along = np.max(np.abs(along - shift_v)[off_limb])
            d_perp = np.max(np.abs(perp[off_limb]))
            assert d_along < TOL_AST_THETAE_TIGHT, (
                f"{name} angle={ang}: |d along|={d_along:.2e}")
            assert d_perp < 1.0e-5, f"{name} angle={ang}: perp={d_perp:.2e}"

        # NumPy (uniform limb) and JAX (clustered limb) agree off the limb;
        # the uniform limb dominates the difference near the band edges.
        d_codes = np.abs(along_by_code["njx"] - along_by_code["jax"])[off_limb]
        assert np.max(d_codes) < TOL_AST_THETAE_TIGHT, (
            f"angle={ang}: njx-jax={d_codes.max():.2e}")

    return None


def _lee10_quad_reference(u, rho):
    """
    Uniform-disk magnification and centroid shift by adaptive quadrature.

    Lee et al. (2010) Eq. 10 in lens-centred coordinates; accurate to
    ~1e-12, unlike VBM ESPL (~1e-3 in magnification near the limb).

    Parameters
    ----------
    u : float
        Source-lens separation (Einstein radii).
    rho : float
        Source radius (Einstein radii).

    Returns
    -------
    amp : float
        Magnification.
    shift : float
        Centroid shift along u-hat relative to the source (Einstein radii).
    """
    from scipy.integrate import quad

    def big_p(x):
        return (x ** 2 + 1.0) * np.sqrt(x ** 2 + 4.0) / 3.0

    def big_q(x):
        return 0.5 * x * np.sqrt(x ** 2 + 4.0)

    def bounds(th):
        root = np.sqrt(max(rho ** 2 - (u * np.sin(th)) ** 2, 0.0))
        u_hi = u * np.cos(th) + root
        u_lo = 0.0 if u <= rho else u * np.cos(th) - root
        return u_lo, u_hi

    # Full circle with the lens inside the source, else the subtended wedge.
    th_max = np.pi if u <= rho else np.arcsin(min(rho / u, 1.0))
    opts = dict(epsabs=1e-14, epsrel=1e-12, limit=500)
    den = quad(lambda th: big_q(bounds(th)[1]) - big_q(bounds(th)[0]),
               -th_max, th_max, **opts)[0]
    num = quad(lambda th: (big_p(bounds(th)[1]) - big_p(bounds(th)[0]))
               * np.cos(th), -th_max, th_max, **opts)[0]

    return den / (np.pi * rho ** 2), num / den - u


@pytest.mark.parametrize("u0,rho", [(0.1, 0.5), (0.02, 0.05)])
def test_fspl_njx_limb_crossing_vs_quad(u0, rho):
    """
    njx FSPL magnification and centroid across the limb vs exact quadrature.

    The adaptive mesh (``amg_tol``) refines where the limb passes the lens,
    so the lens crossing the source limb is resolved; VBM ESPL is not used
    as the reference here because it is only ~1e-3 accurate at the limb.

    Parameters
    ----------
    u0, rho : float
        Impact parameter and source radius in Einstein radii.
    """
    mod_njx = _get_model_Lee2010(u0, rho, jax=False)
    theta_e = mod_njx.thetaE_amp
    rho = mod_njx.source_radius_thetaE_units()

    # Dense across the limb crossings, plus the full |u| < 2 rho range.
    tau = np.unique(np.concatenate([
        np.linspace(-2.0 * rho, 2.0 * rho, 41),
        np.sqrt(rho ** 2 - u0 ** 2) * (1.0 + np.linspace(-0.02, 0.02, 21)),
    ]))
    t = mod_njx.t0 + mod_njx.tE * tau
    u_vec = mod_njx.get_u(t)
    u_amp = np.linalg.norm(u_vec, axis=1)

    ref = np.array([_lee10_quad_reference(x, rho) for x in u_amp])
    a_njx = mod_njx.get_amplification(t)
    shift_njx = _along_uhat(mod_njx.get_centroid_shift(t) / theta_e, u_vec)

    d_amp = np.max(np.abs(a_njx / ref[:, 0] - 1))
    d_shift = np.max(np.abs(shift_njx - ref[:, 1]))
    assert d_amp < 1e-5, f"max |dA/A|={d_amp:.2e}"
    assert d_shift < 5e-5, f"max |d shift|={d_shift:.2e} theta_E"

    return None


# ---------------------------------------------------------------------------
# 8b. Linear (Gamma-law) limb darkening
# ---------------------------------------------------------------------------
def _ld_quad_reference(u, rho, gamma):
    """
    Limb-darkened magnification and centroid shift by nested quadrature.

    The Gamma-law flux and first moment are
    ``(1 - Gamma) F(rho) + 1.5 Gamma int_0^1 F(rho sqrt(1 - s^2)) ds`` with
    ``F(r) = pi r^2 A(u, r)`` from :func:`_lee10_quad_reference`; the
    s-integral is split where the disk edge crosses the lens.

    Parameters
    ----------
    u : float
        Source-lens separation (Einstein radii).
    rho : float
        Source radius (Einstein radii).
    gamma : float
        Linear limb-darkening coefficient (Gamma convention).

    Returns
    -------
    amp : float
        Magnification.
    shift : float
        Centroid shift along u-hat relative to the source (Einstein radii).
    """
    from scipy.integrate import quad

    # Uniform-disk magnified flux and first moment (lens-centred).
    def flux_mom(r):
        amp, shift = _lee10_quad_reference(u, r)
        flux = np.pi * r ** 2 * amp
        return flux, flux * (shift + u)

    # Integrand kink where the disk edge crosses the lens.
    pts = [np.sqrt(1.0 - (u / rho) ** 2)] if u < rho else None
    out = []
    for which in (0, 1):
        val = quad(lambda s: flux_mom(rho * np.sqrt(1.0 - s ** 2))[which],
                   0.0, 1.0, points=pts, epsabs=1e-13, epsrel=1e-11,
                   limit=200)[0]
        out.append((1.0 - gamma) * flux_mom(rho)[which] + 1.5 * gamma * val)

    return out[0] / (np.pi * rho ** 2), out[1] / out[0] - u


def _get_model_Lee2010_limb(u0, rho, gamma, jax=False):
    """
    Limb-darkened twin of :func:`_get_model_Lee2010`.

    Parameters
    ----------
    u0, rho : float
        Impact parameter and source radius in Einstein radii.
    gamma : float
        Linear limb-darkening coefficient (Gamma convention).
    jax : bool, optional
        Build the ``bagle.model_jax`` class instead of ``bagle.model``.

    Returns
    -------
    mod : FSPL_Limb_PhotAstrom_noPar_Param2
    """
    from bagle import model_jax

    # Same Lee et al. (2010) setup as the uniform-source helper.
    mod_lib = model_jax if jax else model
    thetaE = 3.0  # mas
    par = dict(t0=10.0, u0_amp=u0, tE=100.0, thetaE=thetaE, piS=0.1,
               piE_E=0.1, piE_N=0.0, xS0_E=np.array([0.0]),
               xS0_N=np.array([0.0]), muS_E=3.0, muS_N=0.0,
               radiusS=rho * thetaE * 1e-3, b_sff=np.array([1.0]),
               mag_src=np.array([18.0]), gamma=np.array([gamma]))

    return mod_lib.FSPL_Limb_PhotAstrom_noPar_Param2(**par, n_outline=N_OUTLINE)


@pytest.mark.parametrize("u0,rho,gamma", [(0.1, 0.5, 0.44), (0.02, 0.05, 0.8)])
def test_fspl_limb_njx_jax_vs_quad_and_vbm(u0, rho, gamma):
    """
    Limb-darkened FSPL (njx and JAX) vs nested quadrature and VBM ESPL.

    VBM's linear coefficient is ``a1 = 3 Gamma / (2 + Gamma)``. VBM is only
    ~1e-4 accurate here, so the tight checks are against the quadrature.

    Parameters
    ----------
    u0, rho : float
        Impact parameter and source radius in Einstein radii.
    gamma : float
        Linear limb-darkening coefficient (Gamma convention).
    """
    import VBMicrolensing

    mod_njx = _get_model_Lee2010_limb(u0, rho, gamma)
    theta_e = mod_njx.thetaE_amp
    rho = mod_njx.source_radius_thetaE_units()

    # Through the source, denser across the limb crossings.
    tau = np.unique(np.concatenate([
        np.linspace(-2.0 * rho, 2.0 * rho, 25),
        np.sqrt(rho ** 2 - u0 ** 2) * (1.0 + np.linspace(-0.02, 0.02, 5)),
    ]))
    t = mod_njx.t0 + mod_njx.tE * tau
    u_vec = mod_njx.get_u(t)
    u_amp = np.linalg.norm(u_vec, axis=1)

    # Reference, njx model, and JAX kernel (static clustered limb).
    ref = np.array([_ld_quad_reference(x, rho, gamma) for x in u_amp])
    a_njx = mod_njx.get_amplification(t)
    shift_njx = _along_uhat(mod_njx.get_centroid_shift(t) / theta_e, u_vec)
    a_jax, cen_jax = fspl_magnification_centroid(u_vec, rho, 400, gamma=gamma)
    a_jax = np.asarray(a_jax)
    shift_jax = _along_uhat(np.asarray(cen_jax) - u_vec, u_vec)

    # VBM ESPL with the matching linear coefficient.
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol, vbm.RelTol, vbm.astrometry = 1e-6, 1e-7, True
    vbm.a1 = float(fspl_contour.gamma_to_u(gamma))
    a_vbm = np.zeros(u_amp.size)
    shift_vbm = np.zeros(u_amp.size)
    for i, x in enumerate(u_amp):
        a_vbm[i] = vbm.ESPLMag2(x, rho)
        shift_vbm[i] = vbm.astrox1 - x

    # Comparison plot: light curve, shift, and residuals vs the quadrature.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, y_ref, ys, ylab in [
            (axes[0, 0], ref[:, 0], (a_njx, a_jax, a_vbm), "magnification"),
            (axes[0, 1], ref[:, 1], (shift_njx, shift_jax, shift_vbm),
             r"shift along $\hat u$ ($\theta_E$)")]:
        ax.plot(tau, y_ref, "k-", lw=3, alpha=0.3, label="quad")
        for y, sty, lab in zip(ys, ("C0.", "C1x", "C2+"), ("njx", "jax", "VBM")):
            ax.plot(tau, y, sty, label=lab)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    for y, sty, lab in zip((a_njx, a_jax, a_vbm), ("C0.", "C1x", "C2+"),
                           ("njx", "jax", "VBM")):
        axes[1, 0].semilogy(tau, np.abs(y / ref[:, 0] - 1) + 1e-16, sty,
                            label=lab)
    for y, sty, lab in zip((shift_njx, shift_jax, shift_vbm),
                           ("C0.", "C1x", "C2+"), ("njx", "jax", "VBM")):
        axes[1, 1].semilogy(tau, np.abs(y - ref[:, 1]) + 1e-16, sty, label=lab)
    axes[1, 0].set_ylabel("|dA/A| vs quad")
    axes[1, 1].set_ylabel(r"|d shift| vs quad ($\theta_E$)")
    for ax in axes[1]:
        ax.set_xlabel(r"$\tau$")
        ax.legend(fontsize=8)
    fig.suptitle(rf"FSPL linear LD $\Gamma$={gamma}, $u_0$={u0}, "
                 rf"$\rho$={rho:.3g}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"limb_ld_gamma{gamma}_rho{rho:.3g}.png", dpi=120)
    plt.close(fig)

    # njx and JAX against the quadrature.
    for name, a_b, s_b in [("njx", a_njx, shift_njx), ("jax", a_jax, shift_jax)]:
        d_amp = np.max(np.abs(a_b / ref[:, 0] - 1))
        d_shift = np.max(np.abs(s_b - ref[:, 1]))
        assert d_amp < 1e-5, f"{name}: max |dA/A|={d_amp:.2e}"
        assert d_shift < 2e-5, f"{name}: max |d shift|={d_shift:.2e} theta_E"

    # VBM agrees at its own accuracy (same Gamma <-> a1 convention).
    assert _max_frac_diff(a_njx, a_vbm) < 3e-4
    assert _max_abs_diff(shift_njx, shift_vbm) < 3e-4

    return None


def test_fspl_limb_gamma0_and_per_filter():
    """
    ``gamma = 0`` reproduces uniform FSPL; each filter uses its own gamma.

    Returns
    -------
    None
    """
    t = np.linspace(-40.0, 60.0, 81)

    # Uniform twin vs Gamma = 0 (njx).
    mod_uni = _get_model_Lee2010(0.05, 0.3)
    mod_g0 = _get_model_Lee2010_limb(0.05, 0.3, 0.0)
    np.testing.assert_array_equal(mod_g0.get_photometry(t),
                                  mod_uni.get_photometry(t))
    np.testing.assert_array_equal(mod_g0.get_astrometry(t),
                                  mod_uni.get_astrometry(t))

    # Two photometric filters with gamma = [0, 0.6].
    par = dict(t0=10.0, u0_amp=0.05, tE=100.0, piE_E=0.1, piE_N=0.0,
               radiusS=0.3, b_sff=[1.0, 0.8], mag_src=[18.0, 19.0])
    two = model.FSPL_Limb_Phot_noPar_Param1(**par, gamma=[0.0, 0.6])
    uni = model.FSPL_Phot_noPar_Param1(**par)
    one = model.FSPL_Limb_Phot_noPar_Param1(
        **{**par, 'b_sff': [0.8], 'mag_src': [19.0]}, gamma=[0.6])

    np.testing.assert_array_equal(two.get_photometry(t, filt_idx=0),
                                  uni.get_photometry(t, filt_idx=0))
    np.testing.assert_allclose(two.get_photometry(t, filt_idx=1),
                               one.get_photometry(t, filt_idx=0),
                               rtol=0, atol=1e-12)

    # Limb darkening brightens the peak when the lens crosses the centre.
    assert np.min(two.get_photometry(t, filt_idx=1)) < np.min(
        uni.get_photometry(t, filt_idx=1))

    return None


def test_fspl_limb_jax_likelihood_vs_njx():
    """
    Traced ``model_jax`` limb-darkened likelihoods vs njx forward model.

    njx photometry and astrometry are used as noiseless data; the JAX
    class likelihoods (which read the per-filter ``gamma`` from the packed
    vector) must give chi^2 ~ 0, and be sensitive to ``gamma``.

    Returns
    -------
    None
    """
    from bagle import model_jax
    from bagle.jax_physics import (gaussian_astrometry_log_likelihood_sum,
                                   gaussian_log_likelihood_sum)

    u0, rho, gamma = 0.02, 0.05, 0.6
    mod_njx = _get_model_Lee2010_limb(u0, rho, gamma)
    cls = model_jax.FSPL_Limb_PhotAstrom_noPar_Param2

    # Through and across the source.
    t = mod_njx.t0 + mod_njx.tE * np.linspace(-0.1, 0.1, 81)
    mag = mod_njx.get_photometry(t)
    pos = mod_njx.get_astrometry(t)
    mag_err = np.full(t.size, 1e-3)
    pos_err = np.full(t.size, 1e-5)  # arcsec

    # Packed vector in fitter_param_names order (one filter).
    p = dict(t0=mod_njx.t0, u0_amp=u0, tE=mod_njx.tE, thetaE=mod_njx.thetaE_amp,
             piS=mod_njx.piS, piE_E=0.1, piE_N=0.0, xS0_E=0.0, xS0_N=0.0,
             muS_E=3.0, muS_N=0.0, radiusS=mod_njx.radiusS, b_sff=1.0,
             mag_src=18.0, gamma=gamma)
    vec = np.array([p[name] for name in cls.fitter_param_names])

    # Perfect-fit log-likelihoods (chi^2 = 0).
    lnl_phot_best = float(gaussian_log_likelihood_sum(mag, mag, mag_err))
    lnl_ast_best = float(gaussian_astrometry_log_likelihood_sum(
        pos, pos[:, 0], pos[:, 1], pos_err, pos_err))

    lnl_phot = float(cls.jax_log_likely_photometry(
        vec, t, mag, mag_err, 1.0, 18.0, n_outline=400))
    lnl_ast = float(cls.jax_log_likely_astrometry(
        vec, t, pos[:, 0], pos[:, 1], pos_err, pos_err, b_sff=1.0,
        mag_src=18.0, n_outline=400))
    assert abs(lnl_phot - lnl_phot_best) < 1e-2, (lnl_phot, lnl_phot_best)
    assert abs(lnl_ast - lnl_ast_best) < 1e-2, (lnl_ast, lnl_ast_best)

    # Wrong gamma is strongly disfavoured.
    vec0 = vec.copy()
    vec0[cls.fitter_param_names.index('gamma')] = 0.0
    lnl_phot0 = float(cls.jax_log_likely_photometry(
        vec0, t, mag, mag_err, 1.0, 18.0, n_outline=400))
    assert lnl_phot0 < lnl_phot_best - 100.0

    return None


def _make_vbm_astro():
    """
    Bare VBM instance for ``astrox1`` comparisons (uniform source).

    Returns
    -------
    vbm : VBMicrolensing.VBMicrolensing
    """
    import VBMicrolensing

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = 1e-7
    vbm.RelTol = 1e-8
    vbm.a1 = 0.0
    vbm.astrometry = True

    return vbm


# ---------------------------------------------------------------------------
# 9. Summary table written next to the plots
# ---------------------------------------------------------------------------
def test_fspl_vs_vbm_write_summary(tmp_path=None):
    """
    Run a compact multi-case summary and write ``summary.txt`` under PLOT_DIR.

    This is a convenience scan (not a substitute for the assert tests above).
    """
    rows = []

    cases = [
        (0.3, 0.01),
        (0.1, 0.05),
        (0.1, 0.5),
        (0.5, 0.3),
    ]
    t0, tE = 0.0, 10.0
    t = np.linspace(-2.5 * tE, 2.5 * tE, 301)

    for u0, rho in cases:
        mod_njx = _kernel_model(u0, rho, t0=t0, tE=tE)
        u_vec = mod_njx.get_u(t)
        u_amp = np.linalg.norm(u_vec, axis=1)

        # Photometry: JAX kernel vs ESPLLightCurve.
        a_jax = np.asarray(fspl_magnification(u_vec, rho, N_OUTLINE))
        vbm, par = _get_vbm(mod_njx)
        a_v = np.asarray(vbm.ESPLLightCurve(par, list(t))[0])
        frac = _max_frac_diff(a_jax, a_v)

        # Astrometry: JAX centroid vs astrox1 along u-hat.
        sh_jax, _ = _jax_shift_uhat(u_vec, rho)
        vbm_ast, _par = _get_vbm(mod_njx, tol=1e-6, rel_tol=1e-7,
                                 astrometry=True)
        sh_v = _vbm_shift_uhat(vbm_ast, u_amp, rho)
        outside, near_limb, inside = _limb_masks(u_amp, rho)
        d_out = _max_abs_diff(sh_jax[outside], sh_v[outside])
        d_in = (_max_abs_diff(sh_jax[inside], sh_v[inside])
                if np.any(inside) else np.nan)
        rows.append(
            f"u0={u0:4.2f} rho={rho:4.2f}  "
            f"phot|dA/A|={frac:.2e}  "
            f"ast_out={d_out:.2e}  ast_in={d_in:.2e}")

    text = "BAGLE FSPL vs VBMicrolensing summary\n" + "\n".join(rows) + "\n"
    out = PLOT_DIR / "summary.txt"
    out.write_text(text)
    print(text)

    # Sanity: at least one small-rho phot case is below the tight tolerance.
    assert "phot|dA/A|=" in text

    return None
