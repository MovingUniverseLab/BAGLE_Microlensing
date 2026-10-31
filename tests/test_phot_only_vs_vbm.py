"""Compare BAGLE phot-only (and PSTL PhotAstrom) amplification to VBM.

Point-source families use ``SetLensGeometry`` + ``MultiMag0``. Finite-source
single-lens uses ``ESPLMag2``. Finite-source binary/triple uses ``MultiMag2``.
Plots go under ``tests/plots/phot_vs_vbm/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('VBMicrolensing')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import VBMicrolensing as VBM

from bagle import model_jax as mj
from bagle.model import mag2flux


PLOT_DIR = Path(__file__).resolve().parent / 'plots' / 'phot_vs_vbm'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

T = np.linspace(56880.0, 57120.0, 81)

_PHOT_BASE = dict(
    t0=57000.0,
    u0_amp=0.4,
    tE=80.0,
    piE_E=0.05,
    piE_N=-0.02,
    b_sff=[1.0],
    mag_src=[18.0],
)


def _vbm():
    """Configured VBMicrolensing instance for MultiMag comparisons.

    Returns
    -------
    vbm : VBMicrolensing.VBMicrolensing
    """
    vbm = VBM.VBMicrolensing()
    vbm.Tol = 1e-6
    vbm.RelTol = 1e-7
    return vbm


def _multimag0(vbm, lenses, u):
    """Total magnification from ``SetLensGeometry`` + ``MultiMag0``.

    Parameters
    ----------
    vbm : VBMicrolensing.VBMicrolensing
        Configured solver.
    lenses : array_like, shape (N_lenses, 3)
        Columns are East, North, mass (Einstein units).
    u : array_like, shape (N_times, 2)
        Source East/North in the same Einstein frame.

    Returns
    -------
    amp : ndarray, shape (N_times,)
    """
    geom = []
    for east, north, mass in np.asarray(lenses, dtype=float):
        geom.extend([float(east), float(north), float(mass)])
    vbm.SetLensGeometry(geom)

    amp = np.empty(len(u), dtype=float)
    for i, (ue, un) in enumerate(np.asarray(u, dtype=float)):
        val = vbm.MultiMag0(float(ue), float(un))
        amp[i] = float(np.sum(np.atleast_1d(val)))
    return amp


def _multimag2(vbm, lenses, u, rho):
    """Total magnification from ``SetLensGeometry`` + ``MultiMag2``.

    Parameters
    ----------
    vbm : VBMicrolensing.VBMicrolensing
        Configured solver.
    lenses : array_like, shape (N_lenses, 3)
        Columns are East, North, mass (Einstein units).
    u : array_like, shape (N_times, 2)
        Source East/North in the same Einstein frame.
    rho : float
        Source radius in Einstein radii.

    Returns
    -------
    amp : ndarray, shape (N_times,)
    """
    geom = []
    for east, north, mass in np.asarray(lenses, dtype=float):
        geom.extend([float(east), float(north), float(mass)])
    vbm.SetLensGeometry(geom)

    amp = np.empty(len(u), dtype=float)
    for i, (ue, un) in enumerate(np.asarray(u, dtype=float)):
        val = vbm.MultiMag2(float(ue), float(un), float(rho))
        amp[i] = float(np.sum(np.atleast_1d(val)))
    return amp


def _esplmag2(vbm, u, rho):
    """Point-lens finite-source magnification (``ESPLMag2``).

    Parameters
    ----------
    vbm : VBMicrolensing.VBMicrolensing
        Configured solver.
    u : array_like, shape (N_times, 2)
        Source East/North in Einstein radii.
    rho : float
        Source radius in Einstein radii.

    Returns
    -------
    amp : ndarray, shape (N_times,)
    """
    uamp = np.linalg.norm(np.asarray(u, dtype=float), axis=1)
    amp = np.empty(len(uamp), dtype=float)
    for i, ui in enumerate(uamp):
        amp[i] = float(vbm.ESPLMag2(float(ui), float(rho)))
    return amp


def _phot_lenses(inst):
    """Static phot-only lenses in Einstein radii.

    Parameters
    ----------
    inst
        Phot-only BAGLE model.

    Returns
    -------
    lenses : ndarray, shape (N_lenses, 3)
        East, North, mass.
    """
    t0 = np.array([float(inst.t0)])
    if hasattr(inst, 'get_resolved_lens_astrometry'):
        xL = np.asarray(inst.get_resolved_lens_astrometry(t0), dtype=float)
        if xL.ndim == 2:
            xL = xL.reshape(1, -1, 2) if xL.shape[1] != 2 or xL.shape[0] != 1 \
                else xL.reshape(1, 1, 2)
        pos = np.atleast_2d(xL[0])
        if pos.shape[-1] != 2:
            pos = pos.reshape(-1, 2)
    else:
        pos = np.array([[0.0, 0.0]])

    nL = pos.shape[0]
    if nL >= 3:
        mass = np.array([inst.m1, inst.m2, inst.m3], dtype=float)
    elif nL == 2:
        mass = np.array([inst.m1, inst.m2], dtype=float)
    else:
        mass = np.array([1.0], dtype=float)
        pos = np.array([[0.0, 0.0]])
    return np.column_stack([pos, mass])


def _source_tracks(inst, t):
    """Unlensed source tracks in Einstein radii.

    Parameters
    ----------
    inst
        Phot-only BAGLE model.
    t : array_like
        Times (MJD).

    Returns
    -------
    tracks : list of ndarray
        One ``(N_times, 2)`` array per source.
    fluxes : ndarray
        Intrinsic source fluxes, one per track.
    """
    u = np.asarray(inst.get_u(t), dtype=float)
    if u.ndim == 2:
        mag = inst.mag_src[0] if hasattr(inst, 'mag_src') else \
            inst.mag_src_pri[0]
        return [u], np.array([mag2flux(mag)], dtype=float)

    tracks = [u[:, i, :] for i in range(u.shape[1])]
    f1 = mag2flux(inst.mag_src_pri[0])
    f2 = mag2flux(inst.mag_src_sec[0])
    return tracks, np.array([f1, f2], dtype=float)


def _vbm_amp_from_phot(inst, t, rho=None, use_espl=False):
    """VBM amplification for a phot-only BAGLE instance.

    Parameters
    ----------
    inst
        Phot-only BAGLE model.
    t : array_like
        Times (MJD).
    rho : float or sequence, optional
        Source radius (or per-source radii) in Einstein radii.
    use_espl : bool
        Use ``ESPLMag2`` (single point lens) instead of ``MultiMag2``.

    Returns
    -------
    amp : ndarray, shape (N_times,)
    """
    vbm = _vbm()
    lenses = _phot_lenses(inst)
    tracks, fluxes = _source_tracks(inst, t)
    rhos = None
    if rho is not None:
        rhos = np.broadcast_to(np.asarray(rho, dtype=float), (len(tracks),))

    amps = []
    for i, u in enumerate(tracks):
        if rhos is None:
            amps.append(_multimag0(vbm, lenses, u))
        elif use_espl:
            amps.append(_esplmag2(vbm, u, float(rhos[i])))
        else:
            amps.append(_multimag2(vbm, lenses, u, float(rhos[i])))

    amp = np.zeros(len(t), dtype=float)
    fsum = float(np.sum(fluxes))
    for a, f in zip(amps, fluxes):
        amp += a * (f / fsum)
    return amp


def _compare_plot(t, a_bagle, a_vbm, title, outfile):
    """Two-panel light-curve comparison.

    Parameters
    ----------
    t : array_like
        Times (MJD).
    a_bagle, a_vbm : array_like
        Amplifications.
    title : str
        Figure title.
    outfile : pathlib.Path
        Destination PNG.

    Returns
    -------
    frac : ndarray
        ``(A_vbm - A_bagle) / A_bagle``.
    """
    a_bagle = np.asarray(a_bagle, dtype=float)
    a_vbm = np.asarray(a_vbm, dtype=float)
    frac = (a_vbm - a_bagle) / a_bagle

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    axes[0].plot(t - 57000.0, a_bagle, 'k-', lw=1.8, label='BAGLE')
    axes[0].plot(t - 57000.0, a_vbm, 'C3--', lw=1.4, label='VBMicrolensing')
    axes[0].set_ylabel('Amplification')
    axes[0].set_title(title)
    axes[0].legend(frameon=False)
    axes[1].plot(t - 57000.0, frac, 'C0-', lw=1.2)
    axes[1].axhline(0.0, color='0.5', lw=0.8)
    axes[1].set_xlabel(r'$t - t_0$ (days)')
    axes[1].set_ylabel(r'$(A_{\mathrm{VBM}}-A_{\mathrm{BAGLE}})/A$')
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)
    return frac


def test_pspl_phot_vs_vbm():
    """PSPL phot-only matches VBM MultiMag0 at ~1e-12."""
    inst = mj.PSPL_Phot_noPar_Param1(**_PHOT_BASE)
    a_b = np.asarray(inst.get_amplification(T), dtype=float)
    a_v = _vbm_amp_from_phot(inst, T)
    frac = _compare_plot(
        T, a_b, a_v, 'PSPL_Phot vs VBM MultiMag0',
        PLOT_DIR / 'phot_pspl.png')
    np.testing.assert_allclose(a_v, a_b, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(frac)) < 1e-10
    return None


def test_psbl_phot_vs_vbm():
    """PSBL phot-only matches VBM MultiMag0."""
    inst = mj.PSBL_Phot_noPar_Param1(
        **_PHOT_BASE, q=0.5, sep=0.8, phi=40.0, root_tol=1e-4)
    a_b = np.asarray(inst.get_amplification(T), dtype=float)
    a_v = _vbm_amp_from_phot(inst, T)
    frac = _compare_plot(
        T, a_b, a_v, 'PSBL_Phot vs VBM MultiMag0',
        PLOT_DIR / 'phot_psbl.png')
    np.testing.assert_allclose(a_v, a_b, rtol=1e-8, atol=1e-10)
    assert np.max(np.abs(frac)) < 1e-7
    return None


def test_pstl_phot_vs_vbm():
    """PSTL phot-only matches VBM MultiMag0."""
    inst = mj.PSTL_Phot_noPar_Param1(
        **_PHOT_BASE, q_12=0.5, q_13=0.2, sep12=0.8, sep13=0.6,
        phi12=40.0, phi13=80.0, root_tol=1e-4)
    a_b = np.asarray(inst.get_amplification(T), dtype=float)
    a_v = _vbm_amp_from_phot(inst, T)
    frac = _compare_plot(
        T, a_b, a_v, 'PSTL_Phot vs VBM MultiMag0',
        PLOT_DIR / 'phot_pstl.png')
    np.testing.assert_allclose(a_v, a_b, rtol=1e-8, atol=1e-10)
    assert np.max(np.abs(frac)) < 1e-7
    return None


def test_bspl_phot_vs_vbm():
    """BSPL phot-only (flux-weighted) matches two VBM MultiMag0 calls."""
    inst = mj.BSPL_Phot_noPar_Param1(
        t0=_PHOT_BASE['t0'], u0_amp=_PHOT_BASE['u0_amp'],
        tE=_PHOT_BASE['tE'], piE_E=_PHOT_BASE['piE_E'],
        piE_N=_PHOT_BASE['piE_N'], sep=0.3, phi=40.0,
        mag_src_pri=[19.0], mag_src_sec=[20.0], b_sff=[1.0])
    a_b = np.asarray(inst.get_amplification(T), dtype=float)
    a_v = _vbm_amp_from_phot(inst, T)
    frac = _compare_plot(
        T, a_b, a_v, 'BSPL_Phot vs VBM MultiMag0 (2 sources)',
        PLOT_DIR / 'phot_bspl.png')
    np.testing.assert_allclose(a_v, a_b, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(frac)) < 1e-10
    return None


def test_bsbl_phot_vs_vbm():
    """BSBL phot-only matches flux-weighted VBM MultiMag0."""
    inst = mj.BSBL_Phot_noPar_Param1(
        t0=_PHOT_BASE['t0'], u0_amp=_PHOT_BASE['u0_amp'],
        tE=_PHOT_BASE['tE'], piE_E=_PHOT_BASE['piE_E'],
        piE_N=_PHOT_BASE['piE_N'], q=0.4, sep_SL=0.4, sep_S=0.25,
        phi_S=30.0, sep_L=0.8, phi_L=50.0, mag_src_pri=[19.0],
        mag_src_sec=[20.0], b_sff=[1.0], dmag_Lp_Ls=[20.0])
    a_b = np.asarray(inst.get_amplification(T), dtype=float)
    a_v = _vbm_amp_from_phot(inst, T)
    frac = _compare_plot(
        T, a_b, a_v, 'BSBL_Phot vs VBM MultiMag0',
        PLOT_DIR / 'phot_bsbl.png')
    np.testing.assert_allclose(a_v, a_b, rtol=1e-8, atol=1e-10)
    assert np.max(np.abs(frac)) < 1e-7
    return None


def test_fspl_phot_vs_vbm():
    """FSPL phot-only matches VBM ESPLMag2."""
    inst = mj.FSPL_Phot_noPar_Param1(
        **_PHOT_BASE, radiusS=0.05, n_outline=40)
    a_b = np.asarray(inst.get_amplification(T), dtype=float)
    a_v = _vbm_amp_from_phot(inst, T, rho=0.05, use_espl=True)
    frac = _compare_plot(
        T, a_b, a_v, 'FSPL_Phot vs VBM ESPLMag2',
        PLOT_DIR / 'phot_fspl.png')
    np.testing.assert_allclose(a_v, a_b, rtol=2e-3, atol=2e-3)
    assert np.max(np.abs(frac)) < 5e-3
    return None


def test_fsbl_phot_vs_vbm():
    """FSBL phot-only vs VBM MultiMag2 (contour, looser tolerance)."""
    inst = mj.FSBL_Phot_noPar_Param1(
        **_PHOT_BASE, q=0.5, sep=0.8, phi=40.0, radiusS=0.03,
        n_outline=40, root_tol=1e-4)
    a_b = np.asarray(inst.get_amplification(T), dtype=float)
    a_v = _vbm_amp_from_phot(inst, T, rho=0.03)
    frac = _compare_plot(
        T, a_b, a_v, 'FSBL_Phot vs VBM MultiMag2',
        PLOT_DIR / 'phot_fsbl.png')
    # Contour integration; keep the check at the percent level.
    assert np.median(np.abs(frac)) < 2e-2
    return None


def test_bfspl_phot_vs_vbm():
    """BFSPL phot-only vs two VBM ESPLMag2 calls."""
    inst = mj.BFSPL_Phot_noPar_Param1(
        t0=_PHOT_BASE['t0'], u0_amp=_PHOT_BASE['u0_amp'],
        tE=_PHOT_BASE['tE'], piE_E=_PHOT_BASE['piE_E'],
        piE_N=_PHOT_BASE['piE_N'], sep=0.3, phi=40.0,
        mag_src_pri=[19.0], mag_src_sec=[20.0], b_sff=[1.0],
        radiusS_pri=0.04, radiusS_sec=0.03)
    a_b = np.asarray(inst.get_amplification(T), dtype=float)
    a_v = _vbm_amp_from_phot(inst, T, rho=[0.04, 0.03], use_espl=True)
    frac = _compare_plot(
        T, a_b, a_v, 'BFSPL_Phot vs VBM ESPLMag2 (2 sources)',
        PLOT_DIR / 'phot_bfspl.png')
    np.testing.assert_allclose(a_v, a_b, rtol=5e-3, atol=5e-3)
    assert np.max(np.abs(frac)) < 1e-2
    return None


def _pstl_photastrom_kw():
    """Constructor kwargs for PSTL PhotAstrom vs VBM."""
    return dict(
        mLp=10.0, mLs=5.0, mLt=2.0, t0=57000.0,
        xS0_E=np.array([0.0]), xS0_N=np.array([0.01]),
        beta=0.4, muL_E=0.0, muL_N=0.0, sep12=3.0, sep13=2.0,
        muS_E=4.0, muS_N=-3.0, dL=2000.0, dS=4000.0,
        alpha12=90.0, alpha13=45.0, b_sff=np.array([1.0]),
        mag_src=np.array([18.0]), dmag_Lp_Ls=np.array([20.0]),
        root_tol=1e-4,
    )


def test_pstl_photastrom_vs_vbm():
    """PSTL PhotAstrom amplification matches VBM in pair-thetaE units."""
    inst = mj.PSTL_PhotAstrom_noPar_Param1(**_pstl_photastrom_kw())
    t = T
    a_b = np.asarray(inst.get_amplification(t), dtype=float)

    # VBM MultiMag0 uses Einstein radii of the *total* lens mass so
    # the three mass fractions sum to 1. Host thetaE is from mLp+mLs.
    m_tot = inst.mLp + inst.mLs + inst.mLt
    thetaE_tot_as = inst.thetaE_amp * 1e-3 * np.sqrt(m_tot / (inst.mLp + inst.mLs))
    xS = np.asarray(inst.get_source_astrometry_unlensed(t), dtype=float)
    x0 = np.asarray(inst.get_lens_origin_astrometry(t), dtype=float)
    xL = np.asarray(inst.get_resolved_lens_astrometry(t), dtype=float)
    u = (xS - x0) / thetaE_tot_as
    uL = (xL - x0[:, None, :]) / thetaE_tot_as
    m_ein = np.array([inst.mLp, inst.mLs, inst.mLt], dtype=float) / m_tot
    # Static lenses: offsets are constant in the 1-2 origin frame.
    lenses = np.column_stack([uL[len(t) // 2], m_ein])

    a_v = _multimag0(_vbm(), lenses, u)
    frac = _compare_plot(
        t, a_b, a_v, 'PSTL_PhotAstrom vs VBM MultiMag0',
        PLOT_DIR / 'photastrom_pstl.png')
    np.testing.assert_allclose(a_v, a_b, rtol=1e-8, atol=1e-10)
    assert np.max(np.abs(frac)) < 1e-7
    return None
