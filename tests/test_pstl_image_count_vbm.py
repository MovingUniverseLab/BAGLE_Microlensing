"""
PSTL image count and positions vs VBMicrolensing.

VBMicrolensing ``MultiImageContours`` traces one closed contour per image
of a tiny source, so the number of contours is the image count. BAGLE's
triple-lens solver must keep the same number of images (4 outside the
caustics, more inside) and put them at the contour centers. Epochs close
to a caustic are skipped: there a finite source's image count changes.
"""
from __future__ import annotations

import numpy as np
import pytest

VBM = pytest.importorskip('VBMicrolensing')

from bagle import model_jax as mj

# Tiny VBM source radius and the caustic exclusion margin (Einstein radii).
RHO = 1e-3
CAUSTIC_MARGIN = 0.02


def _vbm_images(lenses, u):
    """
    Image centers from VBMicrolensing contours, one list per epoch.

    Parameters
    ----------
    lenses : array_like, shape (3, 3)
        East, North, mass for each lens (total-mass Einstein units).
    u : array_like, shape (N_times, 2)
        Source East/North in the same frame.

    Returns
    -------
    centers : list of ndarray
        Per epoch, an array of shape (N_images, 2) of contour centers.
    """
    vbm = VBM.VBMicrolensing()
    vbm.Tol = 1e-4
    vbm.RelTol = 1e-4
    vbm.SetLensGeometry([float(v) for row in lenses for v in row])

    centers = []
    for ue, un in np.asarray(u, dtype=float):
        contours = vbm.MultiImageContours(float(ue), float(un), RHO)
        # Each contour is [list_of_x1, list_of_x2]; its mean is the center.
        centers.append(np.array([[np.mean(c[0]), np.mean(c[1])]
                                 for c in contours]))
    return centers


def _compare(images, u, lenses, caustics):
    """
    Compare BAGLE images with VBM contour centers away from caustics.

    Parameters
    ----------
    images : ndarray, complex, shape (N_times, N_slots)
        BAGLE image positions (NaN for rejected roots), same frame as ``u``.
    u : ndarray, shape (N_times, 2)
        Source positions.
    lenses : ndarray, shape (3, 3)
        VBM lens table in the same frame.
    caustics : ndarray, complex
        Caustic points in the same frame.

    Returns
    -------
    n_bagle : ndarray, shape (N_used,)
        BAGLE image counts at the epochs compared.
    max_offset : float
        Largest BAGLE-to-VBM image distance (Einstein radii).
    """
    # Distance from each source position to the caustics.
    w = u[:, 0] + 1j * u[:, 1]
    d_caus = np.min(np.abs(w[:, None] - caustics.reshape(1, -1)), axis=1)
    use = d_caus > CAUSTIC_MARGIN
    assert use.sum() >= 0.5 * len(w)

    centers = _vbm_images(lenses, u[use])
    n_bagle = []
    max_offset = 0.0
    for img_t, cen in zip(images[use], centers):
        z = img_t[np.isfinite(img_t)]
        n_bagle.append(z.size)

        # Same number of images as VBM.
        assert z.size == len(cen)

        # Every VBM contour center has a BAGLE image on it.
        cz = cen[:, 0] + 1j * cen[:, 1]
        offset = np.max(np.min(np.abs(cz[:, None] - z[None, :]), axis=1))
        max_offset = max(max_offset, float(offset))
    return np.array(n_bagle), max_offset


def test_pstl_phot_image_count_matches_vbm():
    """Phot-only PSTL: image counts and positions match VBM contours."""
    inst = mj.PSTL_Phot_noPar_Param1(
        t0=57000.0, u0_amp=0.05, tE=80.0, piE_E=0.05, piE_N=-0.02,
        q_12=0.5, q_13=0.2, sep12=0.8, sep13=0.6, phi12=40.0, phi13=80.0,
        b_sff=[1.0], mag_src=[18.0], root_tol=1e-8)
    t = np.linspace(56880.0, 57120.0, 161)

    # Phot-only frame: Einstein radii of the total mass (VBM's frame).
    img, _amp = inst.get_all_arrays(t)
    images = np.asarray(img)[:, 0, :]
    u = np.asarray(inst.get_u(t), dtype=float)
    xL = np.asarray(inst.get_resolved_lens_astrometry(t[:1]))[0]
    lenses = np.column_stack([xL, [inst.m1, inst.m2, inst.m3]])
    caustics = np.asarray(inst.get_caustics(t[:1], N_pts=2000))

    n_bagle, max_offset = _compare(images, u, lenses, caustics)
    assert set(n_bagle.tolist()) >= {4}
    assert n_bagle.max() > 4
    assert max_offset < 5 * RHO
    return None


def test_pstl_photastrom_image_count_matches_vbm():
    """PhotAstrom PSTL (example_PSTL setup): counts and positions match."""
    inst = mj.PSTL_PhotAstrom_noPar_Param1(
        mLp=10.0, mLs=5.0, mLt=2.0, t0=57000.0,
        xS0_E=np.array([0.0]), xS0_N=np.array([0.01]), beta=0.4,
        muL_E=0.0, muL_N=0.0, sep12=3.0, sep13=2.0, muS_E=4.0, muS_N=-3.0,
        dL=2000.0, dS=4000.0, alpha12=90.0, alpha13=45.0,
        b_sff=np.array([1.0]), mag_src=np.array([18.0]),
        dmag_Lp_Ls=np.array([20.0]), root_tol=1e-8)
    t = np.linspace(56800.0, 57200.0, 161)

    # VBM renormalizes masses to sum to 1: use the total-mass Einstein
    # radius; BAGLE's thetaE is from mLp + mLs only.
    m_tot = inst.mLp + inst.mLs + inst.mLt
    th_tot = inst.thetaE_amp * 1e-3 * np.sqrt(m_tot / (inst.mLp + inst.mLs))
    x0 = np.asarray(inst.get_lens_origin_astrometry(t), dtype=float)
    xS = np.asarray(inst.get_source_astrometry_unlensed(t), dtype=float)
    xL = np.asarray(inst.get_resolved_lens_astrometry(t), dtype=float)
    u = (xS - x0) / th_tot
    uL = (xL[0] - x0[0]) / th_tot
    lenses = np.column_stack([uL, np.array([inst.mLp, inst.mLs, inst.mLt])
                              / m_tot])

    # BAGLE images (arcsec) into the same origin-centered frame.
    img, _amp = inst.get_all_arrays(t)
    images = (np.asarray(img)[:, 0, :] - (x0[:, 0] + 1j * x0[:, 1])[:, None]) \
        / th_tot

    # Caustics are in absolute pair-Einstein units; move to the VBM frame.
    th_pair = inst.thetaE_amp * 1e-3
    caus = np.asarray(inst.get_caustics(t[:1], N_pts=2000)) * th_pair
    caustics = (caus - (x0[0, 0] + 1j * x0[0, 1])) / th_tot

    n_bagle, max_offset = _compare(images, u, lenses, caustics)
    assert set(n_bagle.tolist()) >= {4, 6}
    assert max_offset < 5 * RHO
    return None
