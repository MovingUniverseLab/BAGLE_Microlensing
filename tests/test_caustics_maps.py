"""Caustic and magnification-map plotting for PSBL / PSTL."""
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bagle import model_jax as mj
from bagle import plot_models


PLOT_DIR = Path(__file__).resolve().parent / 'plots' / 'caustics_maps'
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _pstl_photastrom(mLt=2.0):
    """PSTL PhotAstrom_noPar_Param1 constructor kwargs + instance."""
    kw = dict(
        mLp=10.0, mLs=5.0, mLt=mLt, t0=57000.0,
        xS0_E=np.array([0.0]), xS0_N=np.array([0.01]),
        beta=0.4, muL_E=0.0, muL_N=0.0, sep12=3.0, sep13=2.0,
        muS_E=4.0, muS_N=-3.0, dL=2000.0, dS=4000.0,
        alpha12=90.0, alpha13=45.0, b_sff=np.array([1.0]),
        mag_src=np.array([18.0]), dmag_Lp_Ls=np.array([20.0]),
        root_tol=1e-4)
    return mj.PSTL_PhotAstrom_noPar_Param1(**kw)


def _psbl_photastrom():
    """Matching PSBL PhotAstrom_noPar_Param1 (no tertiary)."""
    return mj.PSBL_PhotAstrom_noPar_Param1(
        mLp=10.0, mLs=5.0, t0=57000.0,
        xS0_E=np.array([0.0]), xS0_N=np.array([0.01]),
        beta=0.4, muL_E=0.0, muL_N=0.0, muS_E=4.0, muS_N=-3.0,
        dL=2000.0, dS=4000.0, sep=3.0, alpha=90.0,
        b_sff=np.array([1.0]), mag_src=np.array([18.0]),
        dmag_Lp_Ls=np.array([20.0]), root_tol=1e-4)


def _pstl_phot():
    """PSTL phot-only leaf."""
    return mj.PSTL_Phot_noPar_Param1(
        t0=57000.0, u0_amp=0.4, tE=80.0, piE_E=0.05, piE_N=-0.02,
        q_12=0.5, q_13=0.2, sep12=0.8, sep13=0.6, phi12=40.0,
        phi13=80.0, b_sff=[1.0], mag_src=[18.0], root_tol=1e-4)


def _psbl_phot():
    """PSBL phot-only leaf."""
    return mj.PSBL_Phot_noPar_Param1(
        t0=57000.0, u0_amp=0.4, tE=80.0, piE_E=0.05, piE_N=-0.02,
        q=0.5, sep=0.8, phi=40.0, b_sff=[1.0], mag_src=[18.0],
        root_tol=1e-4)


def test_pstl_photastrom_caustics_finite():
    """PSTL PhotAstrom critical curves and caustics are finite."""
    pstl = _pstl_photastrom()
    t = np.array([pstl.t0])
    cc = np.asarray(pstl.get_critical_curves(t, N_pts=80))
    caus = np.asarray(pstl.get_caustics(t, N_pts=80))
    assert cc.shape == (1, 80, 6)
    assert caus.shape == cc.shape
    assert np.any(np.isfinite(cc))
    assert np.any(np.isfinite(caus))
    return None


def test_pstl_phot_caustics_finite():
    """PSTL phot-only caustics are finite Einstein-radius points."""
    pstl = _pstl_phot()
    t = np.array([pstl.t0])
    caus = np.asarray(pstl.get_caustics(t, N_pts=80))
    assert caus.shape[0] == 1
    assert caus.shape[-1] == 6
    assert np.any(np.isfinite(caus))
    return None


def test_psbl_phot_caustics_finite():
    """PSBL phot-only no longer requires mLp / thetaE_amp."""
    psbl = _psbl_phot()
    t = np.array([psbl.t0])
    caus = np.asarray(psbl.get_caustics(t, N_pts=80))
    assert caus.shape[-1] == 4
    assert np.any(np.isfinite(caus))
    return None


def test_gen_and_plot_caustics_pstl_and_psbl():
    """``gen_and_plot_caustics_and_critical_curves`` runs for both families."""
    for inst, name in (
            (_psbl_photastrom(), 'psbl_photastrom'),
            (_pstl_photastrom(), 'pstl_photastrom'),
            (_psbl_phot(), 'psbl_phot'),
            (_pstl_phot(), 'pstl_phot')):
        ax = plot_models.gen_and_plot_caustics_and_critical_curves(
            inst, show=False, N_pts=80)
        fig = ax.figure
        fig.savefig(PLOT_DIR / f'caustics_{name}.png', dpi=120)
        plt.close(fig)
        assert ax is not None
    return None


def test_magnification_map_pstl_and_psbl():
    """``get_magnification_map`` returns axes for PSBL and PSTL."""
    for inst, name in (
            (_psbl_photastrom(), 'psbl_photastrom'),
            (_pstl_photastrom(), 'pstl_photastrom'),
            (_pstl_phot(), 'pstl_phot')):
        ax = plot_models.get_magnification_map(
            inst, bins=80, N_pts=60, show=False)
        ax.figure.savefig(PLOT_DIR / f'magmap_{name}.png', dpi=120)
        plt.close(ax.figure)
        assert ax is not None
    return None
