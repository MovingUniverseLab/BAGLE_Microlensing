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

pytest.importorskip("VBMicrolensing")
pytest.importorskip("jax")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from bagle import frame_convert as fc
from bagle import model
from bagle.jax.fspl import fspl_magnification, fspl_magnification_centroid

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
        ``piE`` only sets the sky orientation of ``u`` (no parallax).

    Notes
    -----
    ``piE`` points East so the trajectory lies along the JAX kernel's
    first limb-sample axis. Other orientations currently give up to ~10x
    larger centroid residuals vs VBM (static limb sampling is not
    rotation invariant).
    """
    mod_njx = model.FSPL_Phot_noPar_Param1(
        t0, u0, tE, 1.0, 0.0, rho, [1.0], [18.0], n_outline=N_OUTLINE)

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
    for ui in np.atleast_1d(u_amp):
        vbm.ESPLMag2(max(float(ui), 1e-9), float(rho))
        out.append(vbm.astrox1 - ui)

    return np.asarray(out, dtype=float)


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
    u_vec = np.asarray(u_vec, dtype=float)
    _mag, cen = fspl_magnification_centroid(u_vec, rho, n_outline)
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
    u_amp = np.asarray(u_amp, dtype=float)
    outside = u_amp > (1.0 + band) * rho
    inside = u_amp < (1.0 - band) * rho
    near_limb = ~(outside | inside)

    return outside, near_limb, inside


def _max_frac_diff(a, ref):
    """Maximum |a/ref - 1|."""
    a = np.asarray(a, dtype=float)
    ref = np.asarray(ref, dtype=float)

    return float(np.max(np.abs(a / ref - 1.0)))


def _max_abs_diff(a, ref):
    """Maximum |a - ref|."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(ref))))


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
    t = np.asarray(t, dtype=float)
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
    t = np.asarray(t, dtype=float)
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


def _njx_photastrom_nopar(u0, rho):
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
    mL, t0, dL, dL_dS = 10.0, 57000.0, 4000.0, 0.5
    xS0 = np.array([0.0])
    b_sff, mag_src = np.array([1.0]), np.array([18.0])

    # thetaE (mas) from the PSPL twin.
    pspl = model.PSPL_PhotAstrom_noPar_Param1(
        mL, t0, 0.1, dL, dL_dS, xS0, xS0, 0.0, 0.0, 3.0, 0.0, b_sff, mag_src)
    theta_e = pspl.thetaE_amp

    # beta (mas) and radiusS (arcsec) for the requested u0 and rho.
    mod_njx = model.FSPL_PhotAstrom_noPar_Param1(
        mL, t0, -u0 * theta_e, dL, dL_dS, xS0, xS0, 0.0, 0.0, 3.0, 0.0,
        rho * theta_e * 1e-3, b_sff, mag_src, n_outline=N_OUTLINE)

    return mod_njx


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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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
    u_vec = np.asarray(u_vec, dtype=float)
    n_t = len(u_vec)
    src_area = np.pi * rho ** 2

    mag = np.full((2, n_t), np.nan)
    pos = np.full((2, n_t, 2), np.nan)
    contours = []

    for i in range(n_t):
        cs = vbm.MultiImageContours(float(u_vec[i, 0]), float(u_vec[i, 1]),
                                    float(rho))
        epoch = []
        # Accumulate weight and first moment per image: [plus, minus].
        w_sum = np.zeros(2)
        mom_sum = np.zeros((2, 2))
        for c in cs:
            x = np.asarray(c[0], dtype=float)
            y = np.asarray(c[1], dtype=float)
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
    u_vec = np.asarray(u_vec, dtype=float)
    u_hat = u_vec / np.linalg.norm(u_vec, axis=1)[:, None]

    return np.sum(np.asarray(vec) * u_hat, axis=-1)


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
    fig, axes = plt.subplots(1, len(idx), figsize=(4.2 * len(idx), 4.4))

    for ax, i in zip(np.atleast_1d(axes), idx):
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
    np.atleast_1d(axes)[0].set_ylabel(r"$x_N$ ($\theta_E$)")

    fig.suptitle(title + "\nsolid = VBM contours, dashed = BAGLE outline; "
                 "o = VBM centroid, x = BAGLE centroid; blue = plus, red = minus",
                 fontsize=9)
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
    amp = np.asarray(tmp[0], dtype=float)

    # VBM sky positions are North, East in mas.
    ast_lensed = np.array([tmp[2], tmp[1]], dtype=float).T * 1e-3
    xL = np.array([tmp[4], tmp[3]], dtype=float).T * 1e-3

    # ESPLAstro: indices 5,6 are tau-beta style separations (same as PSPLAstro).
    u_tb = np.array([tmp[6], tmp[5]], dtype=float).T  # columns: beta, tau

    # Rotate tau-beta into East-North using geo-tr piE / u0 hats.
    muRel_hat = np.array([piEE_vbm, piEN_vbm], dtype=float)
    muRel_hat /= np.hypot(muRel_hat[0], muRel_hat[1])
    u0_hat = model.u0_hat_from_thetaE_hat(muRel_hat, u0_vbm)
    u0_hat = np.asarray(u0_hat, dtype=float).copy()
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
    u_vec = np.asarray(mod_njx.get_u(t))
    a_jax = np.asarray(fspl_magnification(u_vec, rho, N_OUTLINE))

    vbm, par = _get_vbm(mod_njx)
    a_vbm = np.asarray(vbm.ESPLLightCurve(par, list(t))[0])

    frac = _max_frac_diff(a_jax, a_vbm)
    outfile = PLOT_DIR / f"phot_kernel_{label}.png"
    _save_phot_compare(
        t, a_jax, a_vbm,
        title=(f"FSPL jax kernel vs VBM ESPLLightCurve  u0={u0}, rho={rho}\n"
               f"max |dA/A| = {frac:.2e}"),
        outfile=outfile, t0=t0)

    # Large-rho contour integration is slightly looser than 1e-4.
    tol = TOL_PHOT_FRAC if rho >= 0.3 else TOL_PHOT_FRAC_TIGHT
    assert frac < tol, f"max |dA/A|={frac:.3e} exceeds {tol:.1e} ({label})"

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

    Outside the limb (beyond ``LIMB_BAND``) agreement should be at the
    ``1e-3``--``1e-4`` level. Inside / near the limb residuals are known to
    be large (contour-weight formula); they are plotted but not asserted.

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
    u_vec = np.asarray(mod_njx.get_u(t))
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

    # Small-rho off-limb is ~1e-4; large-rho needs the wider 1e-3 band
    # because the contour jump contaminates out to ~1.1 rho.
    tol = 2.0e-4 if rho < 0.05 else TOL_AST_THETAE
    assert d_out < tol, (
        f"off-limb max |d shift|={d_out:.3e} exceeds {tol:.1e} ({label})")

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

    a_njx = np.asarray(mod_njx.get_amplification(t))
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
    ra, dec = float(c.ra.value), float(c.dec.value)
    t0, u0, tE = 55775.0, 0.02, 100.0
    piEE, piEN, rho = 0.9, 0.7, 0.05

    mod_njx = model.FSPL_Phot_Par_Param1(
        t0, u0, tE, piEE, piEN, rho, [1.0], [19.0],
        n_outline=N_OUTLINE, raL=ra, decL=dec)

    # Dense cadence over +-1 tE (finite-source peak is narrow).
    t = np.arange(t0 - 1.0 * tE, t0 + 1.0 * tE + 1e-9, 0.5)
    a_njx = np.asarray(mod_njx.get_amplification(t))

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
    ra, dec = float(c.ra.value), float(c.dec.value)

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

    theta_e = float(mod_njx.thetaE_amp)
    rho = float(mod_njx.source_radius_thetaE_units())
    tE = float(mod_njx.tE)
    t = np.arange(t0 - 1.0 * tE, t0 + 1.0 * tE + 1e-9, 0.5)

    a_njx = np.asarray(mod_njx.get_amplification(t))
    u_njx = np.asarray(mod_njx.get_u(t))
    u_amp = np.linalg.norm(u_njx, axis=1)
    # Centroid shift in Einstein radii (njx AMG).
    shift_njx = np.asarray(mod_njx.get_centroid_shift(t)) / theta_e

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
    d_u = float(np.max(np.linalg.norm(u_njx - uvec_vbm, axis=1)))
    outside, _near, _inside = _limb_masks(u_amp, rho)
    d_shift = np.linalg.norm(shift_njx - shift_vbm, axis=1)
    d_out = float(np.max(d_shift[outside])) if np.any(outside) else np.nan

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
    mod_njx = _njx_photastrom_nopar(u0, rho)
    theta_e = float(mod_njx.thetaE_amp)
    rho = float(mod_njx.source_radius_thetaE_units())
    t0, tE = float(mod_njx.t0), float(mod_njx.tE)
    t = t0 + tE * np.linspace(-span, span, 301)
    tt = (t - t0) / tE

    # Shape of u_vec: [N_times, 2] (source - lens, Einstein radii, E/N).
    u_vec = np.asarray(mod_njx.get_u(t))
    u_amp = np.linalg.norm(u_vec, axis=1)

    # njx per-image values, lens-relative in Einstein radii.
    # Shapes: xL [N_times, 2]; mag_njx [2, N_times]; pos_njx [2, N_times, 2].
    xL = np.asarray(mod_njx.get_lens_astrometry(t))
    mag_njx = np.asarray(mod_njx.get_resolved_amplification(t))
    pos_njx = (np.asarray(mod_njx.get_resolved_astrometry(t)) - xL[None]) \
        * 1e3 / theta_e
    shift_njx = _along_uhat(
        np.asarray(mod_njx.get_centroid_shift(t)) / theta_e, u_vec)

    # Shape of outline_njx: [N_times, n_outline, 2, 2].
    outline_njx = (np.asarray(mod_njx.get_resolved_astrometry_outline(t))
                   - xL[:, None, None, :]) * 1e3 / theta_e

    # VBM instance from the matching no-parallax photometry model.
    mod_njx_phot = _kernel_model(abs(mod_njx.u0_amp), rho, t0=t0, tE=tE)
    vbm, par = _get_vbm(mod_njx_phot, tol=1e-6, rel_tol=1e-7, astrometry=True)
    a_vbm = np.asarray(vbm.ESPLLightCurve(par, list(t))[0])
    shift_v_astrox = _vbm_shift_uhat(vbm, u_amp, rho)

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
        tt, u_vec, rho, mag_njx, pos_njx, mag_v, pos_v,
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
        u_vec, rho, idx, outline_njx, pos_njx, pos_v, contours_v, tt,
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
# 7. Summary table written next to the plots
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
        u_vec = np.asarray(mod_njx.get_u(t))
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
                if np.any(inside) else float("nan"))
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
