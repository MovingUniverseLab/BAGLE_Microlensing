"""BAGLE binary, triple, finite-source, and Keplerian models vs VBM.

``get_vbm_model`` builds the VBMicrolensing parameter vector.
``vbm_light_curve`` evaluates it. Single-lens PSPL / FSPL stay on the
original light-curve parameter lists.
"""
import numpy as np
import pytest

pytest.importorskip('VBMicrolensing')

from bagle import frame_convert as fc
from bagle import model


def _max_abs(a, b):
    """Peak absolute residual.

    Parameters
    ----------
    a, b : array_like

    Returns
    -------
    err : float
    """
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_pspl_parameter_list_unchanged():
    """Single-lens no-parallax parameters stay ``[log u0, log tE, t0]``."""
    inst = model.PSPL_Phot_noPar_Param1(
        57000.0, 0.4, 80.0, 0.05, -0.02, [1.0], [18.0])
    _vbm, par = fc.get_vbm_model(inst)

    assert len(par) == 3
    assert par[2] == pytest.approx(57000.0)
    return None


def test_pstl_phot_matches_triple_light_curve():
    """PSTL phot-only matches VBM TripleLightCurve."""
    inst = model.PSTL_Phot_noPar_Param1(
        57000.0, 0.4, 80.0, 0.05, -0.02,
        0.5, 0.2, 0.8, 0.6, 40.0, 80.0,
        [1.0], [18.0], root_tol=1e-4)
    t = np.linspace(57000.0 - 120.0, 57000.0 + 120.0, 41)
    amp, _y1, _y2, sep = fc.vbm_light_curve(inst, t)

    assert sep is None
    assert _max_abs(amp, inst.get_amplification(t)) < 1e-6
    return None


def test_pstl_photastrom_matches_triple_light_curve():
    """PSTL PhotAstrom matches after the total-mass Einstein rescaling."""
    inst = model.PSTL_PhotAstrom_noPar_Param1(
        10.0, 5.0, 2.0, 57000.0,
        np.array([0.0]), np.array([0.01]),
        0.4, 0.0, 0.0, 3.0, 2.0,
        4.0, -3.0, 2000.0, 4000.0,
        90.0, 45.0, np.array([1.0]), np.array([18.0]),
        np.array([20.0]), dmag_Lp_Lt=np.array([-20.0]),
        root_tol=1e-4)
    t = np.linspace(56900.0, 57120.0, 31)
    amp, _y1, _y2, sep = fc.vbm_light_curve(inst, t)

    assert sep is None
    assert _max_abs(amp, inst.get_amplification(t)) < 1e-6
    return None


def test_psbl_phot_matches_binary_light_curve():
    """Static PSBL phot-only matches VBM BinaryLightCurve, both u0 signs."""
    t = np.linspace(56900.0, 57100.0, 41)
    for u0 in (0.4, -0.3):
        inst = model.PSBL_Phot_noPar_Param1(
            57000.0, u0, 80.0, 0.05, -0.02, 0.5, 0.8, 40.0,
            [1.0], [18.0], root_tol=1e-4)
        amp, _y1, _y2, sep = fc.vbm_light_curve(inst, t)

        assert sep is None
        assert _max_abs(amp, inst.get_amplification(t)) < 1e-6
    return None


def test_psbl_ellorbs_phot_matches_kepler():
    """Keplerian PSBL phot-only matches BinaryLightCurveKepler."""
    inst = model.PSBL_Phot_noPar_EllOrbs_Param1(
        57100.0, 0.5, 60.0, 0.08, -0.06, 0.5, 1.1, 35.0,
        1.0, 2.0, -1.5, 0.5, 1.4,
        [1.0], [19.0], raL=268.0, decL=-29.0)
    t = np.linspace(inst.t0_com - 40.0, inst.t0_com + 40.0, 31)
    amp, y1, y2, sep = fc.vbm_light_curve(inst, t)
    a_bagle = np.asarray(inst.get_amplification(t))

    assert sep.shape == t.shape
    assert y1.shape == t.shape
    assert y2.shape == t.shape
    assert _max_abs(amp, a_bagle) < 1e-6
    return None


def test_psbl_ellorbs_photastrom_matches_kepler():
    """Keplerian PSBL PhotAstrom matches BinaryLightCurveKepler."""
    inst = model.PSBL_PhotAstrom_noPar_EllOrbs_Param1(
        mLp=3.0, mLs=1.5, t0_com=57100.0,
        xS0_com_E=[0.001], xS0_com_N=[-0.002],
        beta_com=0.4, muL_E=2.0, muL_N=-1.0,
        omega_pri=40.0, big_omega_sec=110.0, i=60.0, e=0.3,
        tp=57100.0 - 900.0, a=2.0,
        muS_E=-3.0, muS_N=-6.0, dL=2400.0, dS=8000.0,
        b_sff=[1.0], mag_src=[19.0], dmag_Lp_Ls=[20.0],
        raL=268.0, decL=-29.0)
    t = np.linspace(inst.t0_com - 40.0, inst.t0_com + 40.0, 21)
    amp, _y1, _y2, sep = fc.vbm_light_curve(inst, t)

    assert sep.shape == t.shape
    frac = np.max(np.abs(amp / inst.get_amplification(t) - 1.0))
    assert frac < 1e-4
    return None


def test_multilens_parallax_is_rejected():
    """Parallax binary and triple models are not silently treated as PSPL."""
    inst = model.PSBL_Phot_Par_Param1(
        57000.0, 0.4, 80.0, 0.05, -0.02, 0.5, 0.8, 40.0,
        [1.0], [18.0], raL=268.0, decL=-29.0, root_tol=1e-4)

    with pytest.raises(NotImplementedError):
        fc.get_vbm_model(inst)
    return None


def test_fsbl_phot_and_ellorbs_use_source_radius():
    """FSBL passes radiusS into BinaryLightCurve and BinaryLightCurveKepler."""
    mj = pytest.importorskip('bagle.model_jax')

    static = mj.FSBL_Phot_noPar_Param1(
        t0=57000.0, u0_amp=0.4, tE=80.0, piE_E=0.05, piE_N=-0.02,
        b_sff=[1.0], mag_src=[18.0], q=0.5, sep=0.8, phi=40.0,
        radiusS=0.03, n_outline=40, root_tol=1e-4)
    _vbm, par = fc.get_vbm_model(static)
    assert len(par) == 7
    assert np.exp(par[4]) == pytest.approx(0.03)

    t = np.linspace(56940.0, 57060.0, 21)
    amp, _y1, _y2, sep = fc.vbm_light_curve(static, t)
    a_bagle = np.asarray(static.get_amplification(t))
    frac = np.abs(amp / a_bagle - 1.0)

    assert sep is None
    assert np.median(frac) < 2e-2

    # Same orbit as a point-source EllOrbs model, with rho = radiusS.
    # Compare VBM to that point-source light curve away from the caustic:
    # BAGLE's own FSBL Keplerian magnification does not match it.
    kep = mj.FSBL_Phot_noPar_EllOrbs_Param1(
        t0=57100.0, u0_amp=0.5, tE=60.0, piE_E=0.08, piE_N=-0.06,
        q=0.5, sep=1.1, phi=35.0, v_para=1.0, v_perp=2.0, v_rad=-1.5,
        r_s=0.5, a_s=1.4, radiusS=0.01, b_sff=[1.0], mag_src=[19.0],
        raL=268.0, decL=-29.0)
    point = mj.PSBL_Phot_noPar_EllOrbs_Param1(
        57100.0, 0.5, 60.0, 0.08, -0.06, 0.5, 1.1, 35.0,
        1.0, 2.0, -1.5, 0.5, 1.4, [1.0], [19.0], raL=268.0, decL=-29.0)
    _vbm, par_k = fc.get_vbm_model(kep)
    assert len(par_k) == 14
    assert np.exp(par_k[4]) == pytest.approx(0.01)

    t_k = np.linspace(kep.t0_com - 40.0, kep.t0_com + 40.0, 21)
    amp_k, _y1, _y2, sep_k = fc.vbm_light_curve(kep, t_k)
    wing = np.abs(t_k - kep.t0_com) > 25.0
    frac_k = np.abs(amp_k[wing] / point.get_amplification(t_k)[wing] - 1.0)

    assert sep_k.shape == t_k.shape
    assert np.max(frac_k) < 1e-3
    return None
