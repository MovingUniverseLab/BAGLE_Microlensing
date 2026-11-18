"""
Reference-frame definitions for Keplerian (EllOrbs / CircOrbs) binary models.

Every orbiting binary-lens class stores:

* ``t0_mid``, ``u0_amp_mid``, ``xL0_mid``: the geometric midpoint of the two
  lenses *at time t0_mid*. At ``t0_mid`` the unlensed source sits at
  ``u0_amp_mid * thetaE`` from the midpoint's position at that instant,
  perpendicular to the relative proper motion. These are lazy (root-solved
  on first access) unless the midpoint pair is the input.
* ``t0_com``, ``u0_amp_com``, ``u0_com``, ``xL0_com``: the lens center of mass,
  which moves on a straight line, so this is a true closest approach.
* ``t0_p``, ``u0_amp_p``: the same instantaneous definition about the primary
  (inputs for Param7, lazy otherwise).
* The internal ``t0`` / ``u0_amp`` / ``xL0`` are the CoM values, except for
  the midpoint-input classes (Param2, Param3), which keep the midpoint pair.

Each test lists its constructor parameters explicitly and checks both
``bagle.model`` and ``bagle.model_jax`` when the class exists in both.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bagle import model, model_jax
from bagle import orbits

T0 = 57100.0
PLOT_DIR = Path(__file__).resolve().parent / "plots" / "binary_orbit_vbm"


def _both_modules(class_name):
    """
    Yield (module, class) for model and model_jax when the class exists.

    Parameters
    ----------
    class_name : str
        Leaf class name.

    Yields
    ------
    mod : module
    cls : type
    """
    for mod in (model, model_jax):
        if hasattr(mod, class_name):
            yield mod, getattr(mod, class_name)
    return None


def positions(m, t):
    """
    Unlensed source and lens reference points at time t (Einstein units).

    Parameters
    ----------
    m : model instance
        PhotAstrom binary-lens model.
    t : float
        Time (MJD).

    Returns
    -------
    src : numpy array, shape = [2]
        Unlensed source position.
    points : dict
        ``'geom_mid'``, ``'COM'``, ``'prim_center'`` positions.
    """
    scale = m.thetaE_amp * 1e-3
    tt = np.array([t])

    # Lens positions, shape [N_times, 2 lenses, 2].
    lens = m.get_resolved_lens_astrometry(tt)[0] / scale
    src = m.get_source_astrometry_unlensed(tt)[0] / scale

    # Mass-weighted CoM.
    w_p = m.mLp / (m.mLp + m.mLs)
    points = {"geom_mid": lens.mean(axis=0),
              "COM": w_p * lens[0] + (1 - w_p) * lens[1],
              "prim_center": lens[0]}

    return src.reshape(2), points


def assert_closest_approach(m, frame, t_ref, u_vec):
    """
    At t_ref the source is u_vec from the frame's point, perpendicular to muRel.

    Parameters
    ----------
    m : model instance
    frame : {'geom_mid', 'COM', 'prim_center'}
    t_ref : float
        Reference time (MJD).
    u_vec : numpy array, shape = [2]
        Expected source - point vector (Einstein units).

    Returns
    -------
    None
    """
    src, points = positions(m, t_ref)
    rel = src - points[frame]

    np.testing.assert_allclose(rel, u_vec, rtol=0, atol=1e-7)
    assert abs(rel @ m.thetaE_hat) < 1e-7
    return None


def assert_psbl_orbit_frames(m, t_name, u_name, t_in, u_in, frame, xS0_E, xS0_N):
    """
    Check geometric, CoM, primary and input-frame closest approaches.

    Parameters
    ----------
    m : model instance
        PSBL / FSBL PhotAstrom Keplerian model.
    t_name, u_name : str
        Names of the input time and impact-parameter attributes.
    t_in, u_in : float
        Expected input values.
    frame : {'geom_mid', 'COM', 'prim_center'}
        Input reference frame.
    xS0_E, xS0_N : float
        Input source East / North at the input time (arcsec).

    Returns
    -------
    None
    """
    # Input values are kept as given.
    assert getattr(m, t_name) == t_in
    assert getattr(m, u_name) == u_in

    # The input xS0 is the source position at the input reference time.
    src = m.get_source_astrometry_unlensed(np.array([t_in]))
    np.testing.assert_allclose(np.ravel(src), [xS0_E, xS0_N], rtol=0, atol=1e-12)

    u0_hat_pos = model.u0_hat_from_thetaE_hat(m.thetaE_hat, 1.0)

    # Geometric midpoint at t0_mid, and the stored lens position there.
    assert_closest_approach(m, "geom_mid", m.t0_mid, m.u0_amp_mid * u0_hat_pos)
    _, points = positions(m, m.t0_mid)
    np.testing.assert_allclose(m.xL0_mid[0] / (m.thetaE_amp * 1e-3),
                               points["geom_mid"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.u0, m.u0_amp * u0_hat_pos, rtol=0, atol=1e-12)

    # CoM closest approach, and the CoM position at t0_com.
    assert_closest_approach(m, "COM", m.t0_com, m.u0_com)
    _, points = positions(m, m.t0_com)
    np.testing.assert_allclose(m.xL0_com[0] / (m.thetaE_amp * 1e-3),
                               points["COM"], rtol=0, atol=1e-7)

    # Internal frame: midpoint pair for midpoint inputs, else the CoM.
    internal = "t0_mid" if frame == "geom_mid" else "t0_com"
    assert m.t0 == getattr(m, internal)
    if frame != "geom_mid":
        np.testing.assert_array_equal(m.xL0, m.xL0_com)

    # Primary pair (input for Param7, lazy otherwise) holds at t0_p.
    assert_closest_approach(m, "prim_center", m.t0_p, m.u0_amp_p * u0_hat_pos)

    # The binary axis is reported at t0 (secondary -> primary).
    _, points = positions(m, m.t0)
    axis = points["prim_center"] - points["geom_mid"]
    np.testing.assert_allclose(m.alpha_rad, np.arctan2(axis[0], axis[1]),
                               rtol=0, atol=1e-9)
    return None


# ----------------------------------------------------------------------
# PSBL / FSBL PhotAstrom Keplerian frames
# ----------------------------------------------------------------------
def test_psbl_ellorbs_param1_com_inputs():
    """PSBL EllOrbs Param1: CoM inputs (t0_com, beta_com, xS0_com)."""
    for mod, cls in _both_modules("PSBL_PhotAstrom_noPar_EllOrbs_Param1"):
        m = cls(
            mLp=3.0, mLs=1.5, t0_com=T0, xS0_com_E=[0.001], xS0_com_N=[-0.002],
            beta_com=0.4, muL_E=2.0, muL_N=-1.0, omega_pri=40.0,
            big_omega_sec=110.0, i=60.0, e=0.3, tp=T0 - 900.0, a=2.0,
            muS_E=-3.0, muS_N=-6.0, dL=2400.0, dS=8000.0, b_sff=[0.8],
            mag_src=[19.0], dmag_Lp_Ls=[20.0], raL=268.0, decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0_com", "beta_com", T0, 0.4, "COM", 0.001, -0.002)
    return None


def test_psbl_ellorbs_param2_midpoint_inputs():
    """PSBL EllOrbs Param2: midpoint inputs (t0, u0_amp, xS0)."""
    for mod, cls in _both_modules("PSBL_PhotAstrom_noPar_EllOrbs_Param2"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, thetaE=3.0, piS=0.125,
            piE_E=0.08, piE_N=-0.06, xS0_E=[0.001], xS0_N=[-0.002],
            muS_E=-2.0, muS_N=1.5, q=0.5, omega_pri=40.0, big_omega_sec=110.0,
            i=60.0, e=0.3, tp=T0 - 900.0, a=2.0, b_sff=[0.8], mag_src=[19.0],
            dmag_Lp_Ls=[20.0], raL=268.0, decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0", "u0_amp", T0, 0.15, "geom_mid", 0.001, -0.002)
    return None


def test_psbl_ellorbs_param3_midpoint_inputs():
    """PSBL EllOrbs Param3: midpoint inputs with log10_thetaE."""
    for mod, cls in _both_modules("PSBL_PhotAstrom_noPar_EllOrbs_Param3"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, log10_thetaE=np.log10(3.0), piS=0.125,
            piE_E=0.08, piE_N=-0.06, xS0_E=[0.001], xS0_N=[-0.002],
            muS_E=-2.0, muS_N=1.5, q=0.5, omega_pri=40.0, big_omega_sec=110.0,
            i=60.0, e=0.3, tp=T0 - 900.0, a=2.0, b_sff=[0.8],
            mag_base=[19.0 + 2.5 * np.log10(0.8)], dmag_Lp_Ls=[20.0],
            raL=268.0, decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0", "u0_amp", T0, 0.15, "geom_mid", 0.001, -0.002)
    return None


def test_psbl_ellorbs_param4_com_inputs():
    """PSBL EllOrbs Param4: CoM inputs (t0_com, u0_amp_com, xS0_com)."""
    for mod, cls in _both_modules("PSBL_PhotAstrom_noPar_EllOrbs_Param4"):
        m = cls(
            t0_com=T0, u0_amp_com=0.15, tE=60.0, thetaE=3.0, piS=0.125,
            piE_E=0.08, piE_N=-0.06, xS0_com_E=[0.001], xS0_com_N=[-0.002],
            muS_E=-2.0, muS_N=1.5, q=0.5, omega_pri=40.0, big_omega_sec=110.0,
            i=60.0, e=0.3, tp=T0 - 900.0, a=2.0, b_sff=[0.8], mag_src=[19.0],
            dmag_Lp_Ls=[20.0], raL=268.0, decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0_com", "u0_amp_com", T0, 0.15, "COM", 0.001, -0.002)
    return None


def test_psbl_ellorbs_param7_primary_inputs():
    """PSBL EllOrbs Param7: primary inputs (t0_p, beta_p, xS0_p)."""
    for mod, cls in _both_modules("PSBL_PhotAstrom_noPar_EllOrbs_Param7"):
        m = cls(
            mLp=3.0, mLs=1.5, t0_p=T0, xS0_p_E=[0.001], xS0_p_N=[-0.002],
            beta_p=0.4, muL_E=2.0, muL_N=-1.0, omega_pri=40.0,
            big_omega_sec=110.0, i=60.0, e=0.3, tp=T0 - 900.0, a=2.0,
            muS_E=-3.0, muS_N=-6.0, dL=2400.0, dS=8000.0, b_sff=[0.8],
            mag_src=[19.0], dmag_Lp_Ls=[20.0], raL=268.0, decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0_p", "beta_p", T0, 0.4, "prim_center", 0.001, -0.002)
    return None


def test_psbl_ellorbs_param8_com_inputs():
    """PSBL EllOrbs Param8: CoM inputs with log10_thetaE."""
    for mod, cls in _both_modules("PSBL_PhotAstrom_noPar_EllOrbs_Param8"):
        m = cls(
            t0_com=T0, u0_amp_com=0.15, tE=60.0, log10_thetaE=np.log10(3.0),
            piS=0.125, piE_E=0.08, piE_N=-0.06, xS0_com_E=[0.001],
            xS0_com_N=[-0.002], omega_pri=40.0, big_omega_sec=110.0, i=60.0,
            e=0.3, tp=T0 - 900.0, a=2.0, muS_E=-2.0, muS_N=1.5, q=0.5,
            b_sff=[0.8], mag_src=[19.0], dmag_Lp_Ls=[20.0], raL=268.0,
            decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0_com", "u0_amp_com", T0, 0.15, "COM", 0.001, -0.002)
    return None


def test_psbl_circorbs_param4_com_inputs():
    """PSBL CircOrbs Param4: CoM inputs, negative u0_amp_com."""
    for mod, cls in _both_modules("PSBL_PhotAstrom_noPar_CircOrbs_Param4"):
        m = cls(
            t0_com=T0, u0_amp_com=-0.15, tE=60.0, thetaE=3.0, piS=0.125,
            piE_E=0.08, piE_N=-0.06, xS0_com_E=[0.001], xS0_com_N=[-0.002],
            muS_E=-2.0, muS_N=1.5, q=0.5, omega_pri=40.0, big_omega_sec=110.0,
            i=60.0, tp=T0 - 900.0, a=2.0, b_sff=[0.8], mag_src=[19.0],
            dmag_Lp_Ls=[20.0], raL=268.0, decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0_com", "u0_amp_com", T0, -0.15, "COM", 0.001, -0.002)
    return None


def test_fsbl_ellorbs_param1_com_inputs():
    """FSBL EllOrbs Param1: CoM inputs (model_jax only)."""
    for mod, cls in _both_modules("FSBL_PhotAstrom_noPar_EllOrbs_Param1"):
        m = cls(
            mLp=3.0, mLs=1.5, t0_com=T0, xS0_com_E=[0.001], xS0_com_N=[-0.002],
            beta_com=-0.4, muL_E=2.0, muL_N=-1.0, omega_pri=40.0,
            big_omega_sec=110.0, i=60.0, e=0.3, tp=T0 - 900.0, a=2.0,
            muS_E=-3.0, muS_N=-6.0, dL=2400.0, dS=8000.0, b_sff=[0.8],
            mag_src=[19.0], dmag_Lp_Ls=[20.0], radiusS=1e-6, raL=268.0,
            decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0_com", "beta_com", T0, -0.4, "COM", 0.001, -0.002)
    return None


def test_fsbl_ellorbs_param4_com_inputs():
    """FSBL EllOrbs Param4: CoM inputs (model_jax only)."""
    for mod, cls in _both_modules("FSBL_PhotAstrom_noPar_EllOrbs_Param4"):
        m = cls(
            t0_com=T0, u0_amp_com=0.15, tE=60.0, thetaE=3.0, piS=0.125,
            piE_E=0.08, piE_N=-0.06, xS0_com_E=[0.001], xS0_com_N=[-0.002],
            muS_E=-2.0, muS_N=1.5, q=0.5, omega_pri=40.0, big_omega_sec=110.0,
            i=60.0, e=0.3, tp=T0 - 900.0, a=2.0, b_sff=[0.8], mag_src=[19.0],
            dmag_Lp_Ls=[20.0], radiusS=1e-6, raL=268.0, decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0_com", "u0_amp_com", T0, 0.15, "COM", 0.001, -0.002)
    return None


def test_fsbl_ellorbs_param7_primary_inputs():
    """FSBL EllOrbs Param7: primary inputs (model_jax only)."""
    for mod, cls in _both_modules("FSBL_PhotAstrom_noPar_EllOrbs_Param7"):
        m = cls(
            mLp=3.0, mLs=1.5, t0_p=T0, xS0_p_E=[0.001], xS0_p_N=[-0.002],
            beta_p=0.4, muL_E=2.0, muL_N=-1.0, omega_pri=40.0,
            big_omega_sec=110.0, i=60.0, e=0.3, tp=T0 - 900.0, a=2.0,
            muS_E=-3.0, muS_N=-6.0, dL=2400.0, dS=8000.0, b_sff=[0.8],
            mag_src=[19.0], dmag_Lp_Ls=[20.0], radiusS=1e-6, raL=268.0,
            decL=-29.0)
        assert_psbl_orbit_frames(
            m, "t0_p", "beta_p", T0, 0.4, "prim_center", 0.001, -0.002)
    return None


def test_psbl_param4_com_and_param2_midpoint_agree():
    """Geometric Param2 built from Param4's midpoint values matches Param4."""
    for mod, cls_com in _both_modules("PSBL_PhotAstrom_noPar_EllOrbs_Param4"):
        m_com = cls_com(
            t0_com=T0, u0_amp_com=0.15, tE=60.0, thetaE=3.0, piS=0.125,
            piE_E=0.08, piE_N=-0.06, xS0_com_E=[0.001], xS0_com_N=[-0.002],
            muS_E=-2.0, muS_N=1.5, q=0.5, omega_pri=40.0, big_omega_sec=110.0,
            i=60.0, e=0.3, tp=T0 - 900.0, a=2.0, b_sff=[0.8], mag_src=[19.0],
            dmag_Lp_Ls=[20.0], raL=268.0, decL=-29.0)

        # Geometric inputs take xS0 at t0_mid.
        xS_mid = m_com.get_source_astrometry_unlensed(np.array([m_com.t0_mid]))
        cls_geo = getattr(mod, "PSBL_PhotAstrom_noPar_EllOrbs_Param2")
        m_geo = cls_geo(
            t0=m_com.t0_mid, u0_amp=m_com.u0_amp_mid, tE=60.0, thetaE=3.0,
            piS=0.125, piE_E=0.08, piE_N=-0.06, xS0_E=xS_mid[:, 0],
            xS0_N=xS_mid[:, 1], muS_E=-2.0, muS_N=1.5, q=0.5, omega_pri=40.0,
            big_omega_sec=110.0, i=60.0, e=0.3, tp=T0 - 900.0, a=2.0,
            b_sff=[0.8], mag_src=[19.0], dmag_Lp_Ls=[20.0], raL=268.0,
            decL=-29.0)

        np.testing.assert_allclose(
            [m_geo.t0_com, m_geo.u0_amp_com], [T0, 0.15], rtol=0, atol=1e-8)

        t = np.linspace(T0 - 120, T0 + 120, 241)
        np.testing.assert_allclose(m_geo.get_amplification(t),
                                   m_com.get_amplification(t), rtol=1e-8, atol=0)
        np.testing.assert_allclose(m_geo.get_resolved_lens_astrometry(t),
                                   m_com.get_resolved_lens_astrometry(t),
                                   rtol=0, atol=1e-10)
        np.testing.assert_allclose(m_geo.get_source_astrometry_unlensed(t),
                                   m_com.get_source_astrometry_unlensed(t),
                                   rtol=0, atol=1e-10)
    return None


def test_psbl_param8_com_and_param3_midpoint_agree():
    """Geometric Param3 built from Param8's midpoint values matches Param8."""
    for mod, cls_com in _both_modules("PSBL_PhotAstrom_noPar_EllOrbs_Param8"):
        m_com = cls_com(
            t0_com=T0, u0_amp_com=0.15, tE=60.0, log10_thetaE=np.log10(3.0),
            piS=0.125, piE_E=0.08, piE_N=-0.06, xS0_com_E=[0.001],
            xS0_com_N=[-0.002], omega_pri=40.0, big_omega_sec=110.0, i=60.0,
            e=0.3, tp=T0 - 900.0, a=2.0, muS_E=-2.0, muS_N=1.5, q=0.5,
            b_sff=[0.8], mag_src=[19.0], dmag_Lp_Ls=[20.0], raL=268.0,
            decL=-29.0)

        xS_mid = m_com.get_source_astrometry_unlensed(np.array([m_com.t0_mid]))
        cls_geo = getattr(mod, "PSBL_PhotAstrom_noPar_EllOrbs_Param3")
        m_geo = cls_geo(
            t0=m_com.t0_mid, u0_amp=m_com.u0_amp_mid, tE=60.0,
            log10_thetaE=np.log10(3.0), piS=0.125, piE_E=0.08, piE_N=-0.06,
            xS0_E=xS_mid[:, 0], xS0_N=xS_mid[:, 1], muS_E=-2.0, muS_N=1.5,
            q=0.5, omega_pri=40.0, big_omega_sec=110.0, i=60.0, e=0.3,
            tp=T0 - 900.0, a=2.0, b_sff=[0.8],
            mag_base=[19.0 + 2.5 * np.log10(0.8)], dmag_Lp_Ls=[20.0],
            raL=268.0, decL=-29.0)

        np.testing.assert_allclose(
            [m_geo.t0_com, m_geo.u0_amp_com], [T0, 0.15], rtol=0, atol=1e-8)

        t = np.linspace(T0 - 120, T0 + 120, 241)
        np.testing.assert_allclose(m_geo.get_amplification(t),
                                   m_com.get_amplification(t), rtol=1e-8, atol=0)
        np.testing.assert_allclose(m_geo.get_resolved_lens_astrometry(t),
                                   m_com.get_resolved_lens_astrometry(t),
                                   rtol=0, atol=1e-10)
        np.testing.assert_allclose(m_geo.get_source_astrometry_unlensed(t),
                                   m_com.get_source_astrometry_unlensed(t),
                                   rtol=0, atol=1e-10)
    return None


# ----------------------------------------------------------------------
# Phot-only Keplerian PSBL / FSBL
# ----------------------------------------------------------------------
def relative_los(m, t):
    """
    Line-of-sight offset of the secondary from the primary at time t.

    Parameters
    ----------
    m : model instance
        Keplerian PSBL model (phot-only or PhotAstrom).
    t : float
        Time (MJD).

    Returns
    -------
    z : float
        Secondary minus primary along the line of sight (Einstein radii).
    """
    # Phot-only stores (w, o); PhotAstrom stores (omega_pri, big_omega_sec).
    if hasattr(m, "w") and hasattr(m, "o"):
        omega, big_omega = m.w, m.o
    else:
        omega, big_omega = m.omega_pri, m.big_omega_sec

    orb = model.keplerian_lens_orbit(omega, big_omega, m.i, m.e, m.tp, m.p,
                                    m.aleph, m.aleph_sec)
    orb.oal2xy(np.array([t]))

    # Orbit-plane coordinates at t (Thiele-Innes C, H give the LOS).
    mean_anom = 2.0 * np.pi * (t - m.tp) / m.p
    ecc_anom = orb.eccen_anomaly(np.array([mean_anom]), m.e)[0]
    X = np.cos(ecc_anom) - m.e
    Y = np.sqrt(1.0 - m.e ** 2) * np.sin(ecc_anom)
    z = ((orb.conC2 - orb.conC) * X + (orb.conH2 - orb.conH) * Y)

    # PhotAstrom aleph is in mas; convert LOS to Einstein radii.
    if hasattr(m, "thetaE_amp") and not hasattr(m, "w"):
        z = z / m.thetaE_amp

    return z


def assert_phot_orbit_state_at_t0(m, phi):
    """
    Check sep, alpha, r_s and velocities at the phot-only input t0.

    Parameters
    ----------
    m : model instance
        Phot-only Keplerian PSBL / FSBL model.
    phi : float
        Input binary-axis angle relative to piE (deg).

    Returns
    -------
    None
    """
    dt = 1e-3
    lens = m.get_resolved_lens_astrometry(np.array([T0 - dt, T0, T0 + dt]))
    rel = lens[:, 1] - lens[:, 0]
    rel_dot = (rel[2] - rel[0]) / (2 * dt) * 365.25  # per year

    # Expected state: separation along s_hat (PA alpha + 180).
    alpha = np.deg2rad(m.alpha)
    s_hat = -np.array([np.sin(alpha), np.cos(alpha)])
    pa_hat = np.array([-np.cos(alpha), np.sin(alpha)])
    np.testing.assert_allclose(rel[1], m.sep * s_hat, rtol=0, atol=1e-9)
    np.testing.assert_allclose(rel_dot, m.sep * (m.v_para * s_hat
                                                 + m.v_perp * pa_hat),
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.sep_t0, m.sep, rtol=1e-9)

    # Line of sight: offset r_s * sep and rate v_rad * sep.
    np.testing.assert_allclose(relative_los(m, T0), m.r_s * m.sep, atol=1e-9)
    z_dot = (relative_los(m, T0 + dt) - relative_los(m, T0 - dt)) / (2 * dt)
    np.testing.assert_allclose(z_dot * 365.25, m.v_rad * m.sep, atol=1e-5)

    pa_piE = np.arctan2(m.piE[0], m.piE[1])
    np.testing.assert_allclose(np.rad2deg(alpha - pa_piE), phi, atol=1e-9)
    return None


def test_psbl_phot_ellorbs_param1_fitter_order():
    """PSBL Phot EllOrbs Param1 builds from its fitter_param_names."""
    for mod, cls in _both_modules("PSBL_Phot_noPar_EllOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=35.0, v_para=0.3, v_perp=0.4, v_rad=-0.6, r_s=0.5,
            a_s=1.5, b_sff=[0.8], mag_src=[19.0], raL=268.0, decL=-29.0)
        assert m.t0 == T0 and m.u0_amp == 0.15 and m.sep == 1.2
        np.testing.assert_allclose(m.piE, [0.08, -0.06])
        np.testing.assert_allclose(m.aleph / m.aleph_sec, m.q, rtol=1e-12)
    return None


def test_psbl_phot_circorbs_param1_fitter_order():
    """PSBL Phot CircOrbs Param1 builds from its fitter_param_names."""
    for mod, cls in _both_modules("PSBL_Phot_noPar_CircOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=35.0, v_para=0.3, v_perp=0.4, v_rad=-0.6,
            b_sff=[0.8], mag_src=[19.0], raL=268.0, decL=-29.0)
        assert m.t0 == T0 and m.e == 0.0
        np.testing.assert_allclose(m.aleph / m.aleph_sec, m.q, rtol=1e-12)
    return None


def test_fsbl_phot_ellorbs_param1_fitter_order():
    """FSBL Phot EllOrbs Param1 builds from its fitter_param_names."""
    for mod, cls in _both_modules("FSBL_Phot_noPar_EllOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=35.0, v_para=0.3, v_perp=0.4, v_rad=-0.6, r_s=0.5,
            a_s=1.5, radiusS=1e-3, b_sff=[0.8], mag_src=[19.0], raL=268.0,
            decL=-29.0)
        assert m.radiusS == 1e-3
        np.testing.assert_allclose(m.aleph / m.aleph_sec, m.q, rtol=1e-12)
    return None


def test_psbl_phot_ellorbs_param1_state_at_t0():
    """PSBL Phot EllOrbs Param1 reproduces sep, alpha and velocities at t0."""
    for mod, cls in _both_modules("PSBL_Phot_noPar_EllOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=35.0, v_para=0.3, v_perp=0.4, v_rad=-0.6, r_s=0.5,
            a_s=1.5, b_sff=[0.8], mag_src=[19.0], raL=268.0, decL=-29.0)
        assert_phot_orbit_state_at_t0(m, phi=35.0)
    return None


def test_psbl_phot_ellorbs_param1_state_at_t0_flipped():
    """PSBL Phot EllOrbs Param1 with flipped velocities and phi."""
    for mod, cls in _both_modules("PSBL_Phot_noPar_EllOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=200.0, v_para=-0.3, v_perp=-0.4, v_rad=-0.6, r_s=0.5,
            a_s=1.5, b_sff=[0.8], mag_src=[19.0], raL=268.0, decL=-29.0)
        assert_phot_orbit_state_at_t0(m, phi=200.0)
    return None


def test_psbl_phot_ellorbs_param1_near_circular():
    """PSBL Phot EllOrbs Param1 with a_s just above 1."""
    for mod, cls in _both_modules("PSBL_Phot_noPar_EllOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=35.0, v_para=0.3, v_perp=0.4, v_rad=-0.6, r_s=0.5,
            a_s=1.0001, b_sff=[0.8], mag_src=[19.0], raL=268.0, decL=-29.0)
        assert_phot_orbit_state_at_t0(m, phi=35.0)
    return None


def test_psbl_phot_ellorbs_param1_circular_a_s():
    """PSBL Phot EllOrbs Param1 with a_s = 1 (circular)."""
    for mod, cls in _both_modules("PSBL_Phot_noPar_EllOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=35.0, v_para=0.3, v_perp=0.4, v_rad=-0.6, r_s=0.5,
            a_s=1.0, b_sff=[0.8], mag_src=[19.0], raL=268.0, decL=-29.0)
        assert_phot_orbit_state_at_t0(m, phi=35.0)
    return None


def test_psbl_phot_circorbs_param1_state_at_t0():
    """PSBL Phot CircOrbs Param1 reproduces the state at t0."""
    for mod, cls in _both_modules("PSBL_Phot_noPar_CircOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=35.0, v_para=0.3, v_perp=0.4, v_rad=-0.6,
            b_sff=[0.8], mag_src=[19.0], raL=268.0, decL=-29.0)
        assert_phot_orbit_state_at_t0(m, phi=35.0)
    return None


def test_fsbl_phot_ellorbs_param1_state_at_t0():
    """FSBL Phot EllOrbs Param1 reproduces the state at t0."""
    for mod, cls in _both_modules("FSBL_Phot_noPar_EllOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=-70.0, v_para=0.3, v_perp=0.4, v_rad=-0.6, r_s=0.5,
            a_s=1.5, radiusS=1e-3, b_sff=[0.8], mag_src=[19.0], raL=268.0,
            decL=-29.0)
        assert_phot_orbit_state_at_t0(m, phi=-70.0)
    return None


def test_psbl_phot_ellorbs_param1_frame_definitions():
    """Phot-only: u0 about the midpoint at t0, u0_com about the CoM."""
    for mod, cls in _both_modules("PSBL_Phot_noPar_EllOrbs_Param1"):
        m = cls(
            t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5,
            sep=1.2, phi=35.0, v_para=0.3, v_perp=0.4, v_rad=-0.6, r_s=0.5,
            a_s=1.5, b_sff=[0.8], mag_src=[19.0], raL=268.0, decL=-29.0)

        def rel(t, frame):
            """Source minus a lens point at time t (Einstein units)."""
            w, z1, z2 = m.get_complex_pos(np.array([t]))
            z = {"geom_mid": 0.5 * (z1 + z2),
                 "COM": m.m1 * z1 + m.m2 * z2}[frame]
            d = (w - z)[0]
            return np.array([d.real, d.imag])

        np.testing.assert_allclose(rel(m.t0, "geom_mid"), m.u0, rtol=0,
                                   atol=1e-9)
        np.testing.assert_allclose(rel(m.t0_com, "COM"), m.u0_com, rtol=0,
                                   atol=1e-9)
        assert abs(rel(m.t0_com, "COM") @ m.thetaE_hat) < 1e-9
    return None


def _is_photastrom(m):
    """
    True if the model returns absolute sky positions (arcsec).

    Parameters
    ----------
    m : model instance
        Keplerian PSBL model.

    Returns
    -------
    is_pa : bool
    """
    return hasattr(m, "thetaE_amp") and not hasattr(m, "w")


def _primary_mass_fraction(m):
    """
    Primary mass fraction ``mLp / (mLp + mLs)``.

    Parameters
    ----------
    m : model instance
        Binary-lens model with ``mLp`` / ``mLs``, ``q``, or phot ``m1``.

    Returns
    -------
    w_p : float
    """
    # Prefer physical masses: PhotAstrom also defines m1 as a polynomial
    # coefficient, not the mass fraction.
    if hasattr(m, "mLp") and hasattr(m, "mLs"):
        return float(m.mLp / (m.mLp + m.mLs))

    if hasattr(m, "q"):
        return float(1.0 / (1.0 + m.q))

    return float(m.m1)


def _bagle_positions_einstein(m, t):
    """
    Unlensed source, lenses and CoM in Einstein radii.

    Parameters
    ----------
    m : model instance
        Keplerian PSBL model.
    t : array_like
        Times (MJD).

    Returns
    -------
    src : numpy array, shape = [N_times, 2]
        Unlensed source (Einstein radii).
    lens : numpy array, shape = [N_times, 2, 2]
        Primary and secondary lens positions.
    com : numpy array, shape = [N_times, 2]
        Lens center of mass.
    """
    t = np.asarray(t, dtype=float)
    lens = m.get_resolved_lens_astrometry(t)
    src = m.get_source_astrometry_unlensed(t)

    # PhotAstrom returns arcseconds; convert to Einstein radii.
    if _is_photastrom(m):
        scale = m.thetaE_amp * 1e-3
        lens = lens / scale
        src = src / scale

    w_p = _primary_mass_fraction(m)
    com = w_p * lens[:, 0] + (1.0 - w_p) * lens[:, 1]

    return src, lens, com


def _lens_rel_einstein(m, t_arr):
    """
    Secondary minus primary lens positions in Einstein radii.

    Parameters
    ----------
    m : model instance
        Keplerian PSBL model.
    t_arr : array_like
        Times (MJD).

    Returns
    -------
    rel : numpy array, shape = [len(t_arr), 2]
        Relative sky positions (Einstein radii).
    """
    _, lens, _ = _bagle_positions_einstein(m, t_arr)

    return lens[:, 1] - lens[:, 0]


def _vbm_frame_from_lenses(lens):
    """
    Instantaneous VBM binary axes from primary and secondary positions.

    Parameters
    ----------
    lens : numpy array, shape = [N_times, 2, 2]
        Primary and secondary (Einstein radii).

    Returns
    -------
    s : numpy array, shape = [N_times]
        Projected separation.
    ex : numpy array, shape = [N_times, 2]
        Unit vector primary -> secondary.
    ey : numpy array, shape = [N_times, 2]
        Unit vector rotated 90 deg (VBM y).
    """
    rel = lens[:, 1] - lens[:, 0]
    s = np.hypot(rel[:, 0], rel[:, 1])
    ex = rel / s[:, None]
    ey = np.stack([-ex[:, 1], ex[:, 0]], axis=1)

    return s, ex, ey


def bagle_in_vbm_frame(m, t):
    """
    BAGLE source and separation in VBM's instantaneous binary frame.

    Source coordinates are relative to the lens CoM, with ``y1`` along
    primary -> secondary and ``y2`` along the VBM y axis.

    Parameters
    ----------
    m : model instance
        Keplerian PSBL model.
    t : array_like
        Times (MJD).

    Returns
    -------
    y1 : numpy array, shape = [N_times]
        Source coordinate along the binary axis.
    y2 : numpy array, shape = [N_times]
        Source coordinate along VBM y.
    sep : numpy array, shape = [N_times]
        Projected lens separation (Einstein radii).
    """
    src, lens, com = _bagle_positions_einstein(m, t)
    sep, ex, ey = _vbm_frame_from_lenses(lens)
    d = src - com
    y1 = np.sum(d * ex, axis=1)
    y2 = np.sum(d * ey, axis=1)

    return y1, y2, sep


def vbm_sky_from_frame(m, t, y1, y2, sep):
    """
    Reconstruct sky/Einstein positions from VBM ``(y1, y2, sep)``.

    Uses BAGLE's CoM track and instantaneous binary orientation so the
    comparison is in absolute coordinates.

    Parameters
    ----------
    m : model instance
        Keplerian PSBL model (supplies CoM and orientation).
    t : array_like
        Times (MJD).
    y1, y2, sep : array_like
        VBM source coordinates and separation (Einstein radii).

    Returns
    -------
    src : numpy array, shape = [N_times, 2]
        Reconstructed unlensed source.
    lens : numpy array, shape = [N_times, 2, 2]
        Reconstructed primary and secondary.
    com : numpy array, shape = [N_times, 2]
        BAGLE lens CoM (shared origin).
    """
    _, lens_b, com = _bagle_positions_einstein(m, t)
    _, ex, ey = _vbm_frame_from_lenses(lens_b)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    sep = np.asarray(sep, dtype=float)

    # Source relative to CoM in the instantaneous VBM axes.
    src = com + y1[:, None] * ex + y2[:, None] * ey

    # Lenses on the binary axis about the CoM.
    w_p = _primary_mass_fraction(m)
    w_s = 1.0 - w_p
    lens = np.empty_like(lens_b)
    lens[:, 0] = com - w_s * sep[:, None] * ex
    lens[:, 1] = com + w_p * sep[:, None] * ex

    return src, lens, com


def vbm_kepler_compare(m, t=None):
    """
    BAGLE vs VBMicrolensing Keplerian photometry and geometry.

    Maps the BAGLE relative lens state at ``t0_com`` into VBM's
    ``BinaryLightCurveKepler`` parameters (CoM closest approach, no parallax).

    Parameters
    ----------
    m : model instance
        Keplerian PSBL phot-only or PhotAstrom model (noPar).
    t : array_like, optional
        Times (MJD). Default: ``t0_com ± 1.5 tE``.

    Returns
    -------
    result : dict
        ``t``, ``a_bagle``, ``a_vbm``, ``y1_bagle``, ``y2_bagle``, ``sep_bagle``,
        ``y1_vbm``, ``y2_vbm``, ``sep_vbm``, plus Einstein-radius positions
        ``src_bagle``, ``lens_bagle``, ``com_bagle``, ``src_vbm``, ``lens_vbm``.
    """
    VBM = pytest.importorskip("VBMicrolensing")

    # Relative state (secondary - primary) at t0_com, per day.
    tc, dt = m.t0_com, 1e-3
    rel = _lens_rel_einstein(m, [tc - dt, tc, tc + dt])
    rel_dot = (rel[2] - rel[0]) / (2 * dt)
    z = relative_los(m, tc)
    z_dot = (relative_los(m, tc + dt) - relative_los(m, tc - dt)) / (2 * dt)

    # VBM axes at t0_com: x from primary to secondary.
    s = np.hypot(*rel[1])
    ex = rel[1] / s
    ey = np.array([-ex[1], ex[0]])
    w1 = rel_dot @ ex / s
    w2 = -(rel_dot @ ey) / s

    # Semi-major axis in Einstein radii (phot: already; PhotAstrom: mas).
    a_ein = m.a if hasattr(m, "w") else m.a / m.thetaE_amp
    ar = a_ein / (s * np.sqrt(1.0 + (z / s) ** 2))

    v = np.array([m.thetaE_hat @ ex, m.thetaE_hat @ ey])
    alpha = np.arctan2(-v[1], -v[0])

    if t is None:
        t = np.linspace(tc - 1.5 * m.tE, tc + 1.5 * m.tE, 401)
    else:
        t = np.asarray(t, dtype=float)

    shift = 50000.0
    vbm = VBM.VBMicrolensing()
    vbm.Tol = 1e-6
    vbm.RelTol = 1e-7
    vbm.SetObjectCoordinates("17:52:00 -29:00:00")
    par = [np.log(s), np.log(m.q), abs(m.u0_amp_com), alpha, np.log(1e-5),
           np.log(m.tE), tc - shift, 0.0, 0.0, w1, w2, z_dot / s, z / s, ar]
    out = vbm.BinaryLightCurveKepler(par, list(t - shift))
    a_vbm = np.asarray(out[0], dtype=float)
    y1_vbm = np.asarray(out[1], dtype=float)
    y2_vbm = np.asarray(out[2], dtype=float)
    sep_vbm = np.asarray(out[3], dtype=float)

    a_bagle = np.asarray(m.get_amplification(t), dtype=float)
    y1_bagle, y2_bagle, sep_bagle = bagle_in_vbm_frame(m, t)
    src_bagle, lens_bagle, com_bagle = _bagle_positions_einstein(m, t)
    src_vbm, lens_vbm, _ = vbm_sky_from_frame(m, t, y1_vbm, y2_vbm, sep_vbm)

    return {
        "t": t,
        "a_bagle": a_bagle,
        "a_vbm": a_vbm,
        "y1_bagle": y1_bagle,
        "y2_bagle": y2_bagle,
        "sep_bagle": sep_bagle,
        "y1_vbm": y1_vbm,
        "y2_vbm": y2_vbm,
        "sep_vbm": sep_vbm,
        "src_bagle": src_bagle,
        "lens_bagle": lens_bagle,
        "com_bagle": com_bagle,
        "src_vbm": src_vbm,
        "lens_vbm": lens_vbm,
    }


def vbm_kepler_lightcurve(m, t=None):
    """
    BAGLE and VBMicrolensing Keplerian amplifications on a common time grid.

    Parameters
    ----------
    m : model instance
        Keplerian PSBL phot-only or PhotAstrom model (noPar).
    t : array_like, optional
        Times (MJD). Default: ``t0_com ± 1.5 tE``.

    Returns
    -------
    t : numpy array
        Evaluation times (MJD).
    a_bagle : numpy array
        BAGLE amplifications.
    a_vbm : numpy array
        VBM ``BinaryLightCurveKepler`` amplifications.
    """
    res = vbm_kepler_compare(m, t=t)

    return res["t"], res["a_bagle"], res["a_vbm"]


def _plot_save_or_show(fig, outfile=None, show=False):
    """
    Save and/or show a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalize.
    outfile : path-like, optional
        PNG destination.
    show : bool, optional
        If True, call ``plt.show()``.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=130)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return None


def plot_vbm_kepler_compare(t, a_bagle, a_vbm, mag_bagle, mag_vbm, title,
                            outfile=None, show=False):
    """
    Plot amplification and magnitude light curves vs VBM with residuals.

    Parameters
    ----------
    t : array_like
        Times (MJD).
    a_bagle, a_vbm : array_like
        Amplifications.
    mag_bagle, mag_vbm : array_like
        Magnitudes (unblended).
    title : str
        Figure title.
    outfile : path-like, optional
        PNG destination. Created parent dirs if needed.
    show : bool, optional
        If True, call ``plt.show()``.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    t = np.asarray(t, dtype=float)
    a_bagle = np.asarray(a_bagle, dtype=float)
    a_vbm = np.asarray(a_vbm, dtype=float)
    mag_bagle = np.asarray(mag_bagle, dtype=float)
    mag_vbm = np.asarray(mag_vbm, dtype=float)

    frac = a_vbm / a_bagle - 1.0
    dmag = mag_vbm - mag_bagle
    t_rel = t - T0

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 9.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0, 2.0, 1.0]})

    # Amplification light curves.
    axes[0].plot(t_rel, a_bagle, "k-", lw=1.8, label="BAGLE")
    axes[0].plot(t_rel, a_vbm, "C3--", lw=1.4, label="VBM")
    axes[0].set_ylabel("Amplification")
    axes[0].set_title(title)
    axes[0].legend(frameon=False)

    # Fractional amplification residual.
    axes[1].plot(t_rel, frac, "C0-", lw=1.2)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_ylabel(r"$(A_{\mathrm{VBM}}/A_{\mathrm{BAGLE}})-1$")
    axes[1].set_title(
        f"amp frac: mean={frac.mean():.3e}, max={np.max(np.abs(frac)):.3e}",
        fontsize=11)

    # Magnitude light curves (fainter up).
    axes[2].plot(t_rel, mag_bagle, "k-", lw=1.8, label="BAGLE")
    axes[2].plot(t_rel, mag_vbm, "C3--", lw=1.4, label="VBM")
    axes[2].set_ylabel("Mag")
    axes[2].invert_yaxis()
    axes[2].legend(frameon=False)

    # Magnitude residual.
    axes[3].plot(t_rel, dmag, "C0-", lw=1.2)
    axes[3].axhline(0.0, color="0.5", lw=0.8)
    axes[3].set_xlabel(r"$t - 57100$ (days)")
    axes[3].set_ylabel(r"$m_{\mathrm{VBM}}-m_{\mathrm{BAGLE}}$")
    axes[3].set_title(
        f"mag diff: mean={dmag.mean():.3e}, max={np.max(np.abs(dmag)):.3e}",
        fontsize=11)

    _plot_save_or_show(fig, outfile=outfile, show=show)

    return None


def plot_vbm_kepler_astrometry(res, m, title, outfile=None, show=False):
    """
    Plot source/lens trajectories and VBM-frame geometry vs VBM.

    Parameters
    ----------
    res : dict
        Output of :func:`vbm_kepler_compare`.
    m : model instance
        Model used for unit scaling (Einstein radii or mas).
    title : str
        Figure title.
    outfile : path-like, optional
        PNG destination.
    show : bool, optional
        If True, call ``plt.show()``.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    t = res["t"]
    t_rel = t - T0

    # PhotAstrom plots in mas; phot-only stays in Einstein radii.
    if _is_photastrom(m):
        scale = m.thetaE_amp
        unit = "mas"
    else:
        scale = 1.0
        unit = r"$\theta_E$"

    src_b = res["src_bagle"] * scale
    lens_b = res["lens_bagle"] * scale
    com_b = res["com_bagle"] * scale
    src_v = res["src_vbm"] * scale
    lens_v = res["lens_vbm"] * scale

    # Subtract the CoM at t0 so the plane plot is centered.
    i0 = int(np.argmin(np.abs(t - m.t0_com)))
    origin = com_b[i0]
    src_b = src_b - origin
    lens_b = lens_b - origin
    com_b = com_b - origin
    src_v = src_v - origin
    lens_v = lens_v - origin

    y1_b, y2_b, sep_b = res["y1_bagle"], res["y2_bagle"], res["sep_bagle"]
    y1_v, y2_v, sep_v = res["y1_vbm"], res["y2_vbm"], res["sep_vbm"]
    dy1 = y1_v - y1_b
    dy2 = y2_v - y2_b
    dsep = sep_v - sep_b

    fig = plt.figure(figsize=(11.0, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 1.2, 1.2],
                          width_ratios=[1.35, 1.0])

    # Sky / Einstein-plane trajectories.
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(src_b[:, 0], src_b[:, 1], "k-", lw=1.6, label="source BAGLE")
    ax.plot(src_v[:, 0], src_v[:, 1], "C3--", lw=1.2, label="source VBM")
    ax.plot(lens_b[:, 0, 0], lens_b[:, 0, 1], "C0-", lw=1.4, label="L1 BAGLE")
    ax.plot(lens_v[:, 0, 0], lens_v[:, 0, 1], "C0--", lw=1.1, label="L1 VBM")
    ax.plot(lens_b[:, 1, 0], lens_b[:, 1, 1], "C2-", lw=1.4, label="L2 BAGLE")
    ax.plot(lens_v[:, 1, 0], lens_v[:, 1, 1], "C2--", lw=1.1, label="L2 VBM")
    ax.plot(com_b[:, 0], com_b[:, 1], color="0.4", ls=":", lw=1.0, label="CoM")
    ax.plot(src_b[i0, 0], src_b[i0, 1], "ko", ms=5)
    ax.plot(lens_b[i0, 0, 0], lens_b[i0, 0, 1], "C0o", ms=5)
    ax.plot(lens_b[i0, 1, 0], lens_b[i0, 1, 1], "C2o", ms=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"East - CoM({unit})")
    ax.set_ylabel(f"North - CoM({unit})")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")

    # Source track in the instantaneous VBM (y1, y2) plane.
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(y1_b, y2_b, "k-", lw=1.6, label="BAGLE")
    ax.plot(y1_v, y2_v, "C3--", lw=1.2, label="VBM")
    ax.plot(y1_b[i0], y2_b[i0], "ko", ms=5)
    ax.axhline(0.0, color="0.7", lw=0.7)
    ax.axvline(0.0, color="0.7", lw=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$y_1$ (binary axis)")
    ax.set_ylabel(r"$y_2$")
    ax.set_title("Source in VBM frame")
    ax.legend(frameon=False, fontsize=8)

    # Time series: y1, y2, sep.
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_rel, y1_b, "k-", lw=1.5, label="BAGLE")
    ax.plot(t_rel, y1_v, "C3--", lw=1.2, label="VBM")
    ax.set_ylabel(r"$y_1$")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(
        f"y1 max|d|={np.max(np.abs(dy1)):.3e}", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_rel, dy1, "C0-", lw=1.1)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_ylabel(r"$y_{1,\mathrm{VBM}}-y_{1,\mathrm{BAGLE}}$")

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t_rel, y2_b, "k-", lw=1.5, label="BAGLE $y_2$")
    ax.plot(t_rel, y2_v, "C3--", lw=1.2, label="VBM $y_2$")
    ax.plot(t_rel, sep_b, "C2-", lw=1.3, label="BAGLE sep")
    ax.plot(t_rel, sep_v, "C2--", lw=1.1, label="VBM sep")
    ax.set_xlabel(r"$t - 57100$ (days)")
    ax.set_ylabel(r"$y_2$, sep")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(
        f"y2 max|d|={np.max(np.abs(dy2)):.3e}; "
        f"sep max|d|={np.max(np.abs(dsep)):.3e}",
        fontsize=10)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t_rel, dy2, "C0-", lw=1.1, label=r"$\Delta y_2$")
    ax.plot(t_rel, dsep, "C2-", lw=1.1, label=r"$\Delta$sep")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$t - 57100$ (days)")
    ax.set_ylabel("residual")
    ax.legend(frameon=False, fontsize=8)

    _plot_save_or_show(fig, outfile=outfile, show=show)

    return None


def assert_matches_vbm_kepler(m, plot_lc=False, plot_astro=False, title=None,
                              outfile=None, outfile_astro=None,
                              rtol_amp=1e-4, atol_mag=1e-3, atol_pos=1e-6):
    """
    Assert BAGLE Keplerian photometry and geometry match VBM.

    Parameters
    ----------
    m : model instance
        Keplerian PSBL phot-only or PhotAstrom model.
    plot_lc : bool, optional
        If True, plot light curves and residuals (and show the figure).
    plot_astro : bool, optional
        If True, plot source/lens trajectories and VBM-frame geometry.
    title : str, optional
        Plot title. Defaults to the model class name.
    outfile : path-like, optional
        PNG path for the photometry figure.
    outfile_astro : path-like, optional
        PNG path for the astrometry figure.
    rtol_amp : float, optional
        Max allowed ``|A_vbm / A_bagle - 1|``.
    atol_mag : float, optional
        Max allowed ``|mag_vbm - mag_bagle|`` (magnitudes).
    atol_pos : float, optional
        Max allowed absolute error in ``y1``, ``y2`` and ``sep``
        (Einstein radii).

    Returns
    -------
    None
    """
    res = vbm_kepler_compare(m)
    t = res["t"]
    a_bagle, a_vbm = res["a_bagle"], res["a_vbm"]

    # Amplification light curve.
    assert np.max(np.abs(a_vbm / a_bagle - 1.0)) < rtol_amp

    # Magnitude light curve (unblended source; VBM A -> mag_src - 2.5 log10 A).
    mag_bagle = np.asarray(m.get_photometry(t), dtype=float)
    mag_src = float(np.asarray(m.mag_src).ravel()[0])
    mag_vbm = mag_src - 2.5 * np.log10(a_vbm)
    assert np.max(np.abs(mag_vbm - mag_bagle)) < atol_mag

    # Source position and lens separation in the VBM binary frame.
    np.testing.assert_allclose(res["y1_vbm"], res["y1_bagle"], atol=atol_pos,
                               rtol=0)
    np.testing.assert_allclose(res["y2_vbm"], res["y2_bagle"], atol=atol_pos,
                               rtol=0)
    np.testing.assert_allclose(res["sep_vbm"], res["sep_bagle"], atol=atol_pos,
                               rtol=0)

    # Reconstructed sky/Einstein positions (source and both lenses).
    np.testing.assert_allclose(res["src_vbm"], res["src_bagle"], atol=atol_pos,
                               rtol=0)
    np.testing.assert_allclose(res["lens_vbm"], res["lens_bagle"],
                               atol=atol_pos, rtol=0)

    if title is None:
        title = type(m).__name__ + " vs VBM BinaryLightCurveKepler"

    if plot_lc:
        if outfile is None:
            outfile = PLOT_DIR / (type(m).__name__ + ".png")
        plot_vbm_kepler_compare(t, a_bagle, a_vbm, mag_bagle, mag_vbm, title,
                                outfile=outfile, show=True)

    if plot_astro:
        if outfile_astro is None:
            outfile_astro = PLOT_DIR / (type(m).__name__ + "_astro.png")
        plot_vbm_kepler_astrometry(res, m, title, outfile=outfile_astro,
                                   show=True)

    # Phot-only: at input t0, projected rates match v_para and v_perp.
    if hasattr(m, "v_para") and hasattr(m, "v_perp"):
        dt = 1e-3
        rel = _lens_rel_einstein(m, [m.t0 - dt, m.t0, m.t0 + dt])
        rel_dot = (rel[2] - rel[0]) / (2 * dt) * 365.25
        ex = rel[1] / np.hypot(*rel[1])
        ey = np.array([-ex[1], ex[0]])
        np.testing.assert_allclose(
            [rel_dot @ ex / m.sep, -(rel_dot @ ey) / m.sep],
            [m.v_para, m.v_perp], rtol=1e-5)

    return None


def test_psbl_phot_ellorbs_param1_matches_vbm_kepler(plot_lc=False,
                                                     plot_astro=False):
    """PSBL Phot EllOrbs Param1 light curve equals VBM BinaryLightCurveKepler."""
    m = model.PSBL_Phot_noPar_EllOrbs_Param1(
        t0=T0, u0_amp=0.5, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5, sep=1.1,
        phi=35.0, v_para=1.0, v_perp=2.0, v_rad=-1.5, r_s=0.5, a_s=1.4,
        b_sff=[1.0], mag_src=[19.0], raL=268.0, decL=-29.0)
    assert_matches_vbm_kepler(m, plot_lc=plot_lc, plot_astro=plot_astro)
    return None


def test_psbl_phot_ellorbs_param1_matches_vbm_kepler_retrograde(plot_lc=False,
                                                                plot_astro=False):
    """PSBL Phot EllOrbs Param1 vs VBM with retrograde velocities."""
    m = model.PSBL_Phot_noPar_EllOrbs_Param1(
        t0=T0, u0_amp=0.5, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5, sep=1.1,
        phi=200.0, v_para=-0.5, v_perp=-3.0, v_rad=1.0, r_s=-0.3, a_s=2.0,
        b_sff=[1.0], mag_src=[19.0], raL=268.0, decL=-29.0)
    assert_matches_vbm_kepler(m, plot_lc=plot_lc, plot_astro=plot_astro)
    return None


def test_psbl_phot_circorbs_param1_matches_vbm_kepler(plot_lc=False,
                                                     plot_astro=False):
    """PSBL Phot CircOrbs Param1 light curve equals VBM BinaryLightCurveKepler."""
    m = model.PSBL_Phot_noPar_CircOrbs_Param1(
        t0=T0, u0_amp=0.5, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5, sep=1.1,
        phi=-60.0, v_para=0.8, v_perp=2.5, v_rad=-1.6, b_sff=[1.0],
        mag_src=[19.0], raL=268.0, decL=-29.0)
    assert_matches_vbm_kepler(m, plot_lc=plot_lc, plot_astro=plot_astro)
    return None


def test_psbl_photastrom_ellorbs_param1_matches_vbm_kepler(plot_lc=False,
                                                           plot_astro=False):
    """PSBL PhotAstrom EllOrbs Param1 vs VBM BinaryLightCurveKepler."""
    m = model.PSBL_PhotAstrom_noPar_EllOrbs_Param1(
        mLp=3.0, mLs=1.5, t0_com=T0, xS0_com_E=[0.001], xS0_com_N=[-0.002],
        beta_com=0.4, muL_E=2.0, muL_N=-1.0, omega_pri=40.0,
        big_omega_sec=110.0, i=60.0, e=0.3, tp=T0 - 900.0, a=2.0,
        muS_E=-3.0, muS_N=-6.0, dL=2400.0, dS=8000.0, b_sff=[1.0],
        mag_src=[19.0], dmag_Lp_Ls=[20.0], raL=268.0, decL=-29.0)
    assert_matches_vbm_kepler(m, plot_lc=plot_lc, plot_astro=plot_astro)
    return None


def test_psbl_photastrom_circorbs_param1_matches_vbm_kepler(plot_lc=False,
                                                           plot_astro=False):
    """PSBL PhotAstrom CircOrbs Param1 vs VBM BinaryLightCurveKepler."""
    m = model.PSBL_PhotAstrom_noPar_CircOrbs_Param1(
        mLp=3.0, mLs=1.5, t0_com=T0, xS0_com_E=[0.001], xS0_com_N=[-0.002],
        beta_com=0.4, muL_E=2.0, muL_N=-1.0, omega_pri=40.0,
        big_omega_sec=110.0, i=60.0, tp=T0 - 900.0, a=2.0,
        muS_E=-3.0, muS_N=-6.0, dL=2400.0, dS=8000.0, b_sff=[1.0],
        mag_src=[19.0], dmag_Lp_Ls=[20.0], raL=268.0, decL=-29.0)
    assert_matches_vbm_kepler(m, plot_lc=plot_lc, plot_astro=plot_astro)
    return None


# ----------------------------------------------------------------------
# Binary source + binary lens (BSBL / BFSBL) frames
# ----------------------------------------------------------------------
def bsbl_points(m, t, fi=1):
    """
    Source and lens reference points of a BSBL model at t (mas).

    Parameters
    ----------
    m : model instance
        BSBL / BFSBL PhotAstrom model.
    t : float or array_like
        Time(s) (MJD).
    fi : int, optional
        Filter index.

    Returns
    -------
    src : dict
        ``'p'`` (primary) and ``'com'`` source positions, shape [N, 2].
    lens : dict
        ``'p'``, ``'mid'`` and ``'com'`` lens positions, shape [N, 2].
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    S = m.get_resolved_source_astrometry_unlensed(t, filt_idx=fi) * 1e3
    L = m.get_resolved_lens_astrometry(t, filt_idx=fi) * 1e3

    q = m.mLs / m.mLp
    src = {"p": S[:, 0], "com": (S[:, 0] + m.qS * S[:, 1]) / (1 + m.qS)}
    lens = {"p": L[:, 0], "mid": 0.5 * (L[:, 0] + L[:, 1]),
            "com": (L[:, 0] + q * L[:, 1]) / (1 + q)}

    return src, lens


def assert_bsbl_pair(m, t0, u0_amp, src_pt, lens_pt):
    """
    At t0 the source point minus the lens point is u0_amp thetaE, perpendicular.

    Parameters
    ----------
    m : model instance
    t0 : float
    u0_amp : float
    src_pt : {'p', 'com'}
    lens_pt : {'p', 'mid', 'com'}

    Returns
    -------
    None
    """
    src, lens = bsbl_points(m, t0)
    rel = (src[src_pt] - lens[lens_pt])[0] / m.thetaE_amp
    u0_hat_pos = model.u0_hat_from_thetaE_hat(m.thetaE_hat, 1.0)
    np.testing.assert_allclose(rel, u0_amp * u0_hat_pos, rtol=0, atol=1e-9)
    return None


def assert_bsbl_keplerian_frames(m, input_is_primary=False):
    """
    Check CoM internal frame, lazy midpoint/primary, and axes at t0.

    Parameters
    ----------
    m : model instance
        BSBL / BFSBL Keplerian PhotAstrom model.
    input_is_primary : bool, optional
        If True, inputs are ``t0_p`` / ``xS0_p``; else ``t0_com`` / ``xS0_com``.

    Returns
    -------
    None
    """
    # The CoM-CoM pair is a true closest approach at t0_com.
    tt = np.linspace(m.t0_com - 300, m.t0_com + 300, 120001)
    src, lens = bsbl_points(m, tt)
    d = np.hypot(*(src["com"] - lens["com"]).T)
    assert abs(tt[np.argmin(d)] - m.t0_com) < 0.01
    assert_bsbl_pair(m, m.t0_com, m.u0_amp_com, "com", "com")

    # Internal frame is the CoM pair.
    assert m.t0 == m.t0_com and m.u0_amp == m.u0_amp_com
    src, lens = bsbl_points(m, m.t0_com)
    np.testing.assert_allclose(m.xS0_com[1] * 1e3, src["com"][0], atol=1e-9)
    np.testing.assert_allclose(m.xL0_com[1] * 1e3, lens["com"][0], atol=1e-9)

    if input_is_primary:
        assert m.t0_p == T0
        src, _ = bsbl_points(m, T0)
        np.testing.assert_allclose(src["p"][0], [2.0, 0.5], atol=1e-9)
    else:
        assert m.t0_com == T0
        np.testing.assert_allclose(m.xS0_com[1] * 1e3, [2.0, 0.5], atol=1e-9)

    # Lazy midpoint and primary pairs.
    assert_bsbl_pair(m, m.t0_mid, m.u0_amp_mid, "p", "mid")
    _, lens = bsbl_points(m, m.t0_mid)
    np.testing.assert_allclose(m.xL0_mid[1] * 1e3, lens["mid"][0], atol=1e-9)
    assert_bsbl_pair(m, m.t0_p, m.beta_p / m.thetaE_amp, "p", "p")

    np.testing.assert_allclose(m.qS, m.mSs / m.mSp, rtol=1e-12)
    S = m.get_resolved_source_astrometry_unlensed(np.array([m.t0]), 1)[0]
    L = m.get_resolved_lens_astrometry(np.array([m.t0]), 1)[0]
    dl, ds = L[0] - L[1], S[1] - S[0]
    np.testing.assert_allclose(m.alphaL_rad, np.arctan2(dl[0], dl[1]), atol=1e-9)
    np.testing.assert_allclose(np.cos(m.alphaS_rad - np.arctan2(ds[0], ds[1])),
                               1.0, atol=1e-12)
    return None


def test_bsbl_ellorbs_param1_com_inputs():
    """BSBL EllOrbs Param1: CoM-CoM inputs."""
    for mod, cls in _both_modules("BSBL_PhotAstrom_noPar_EllOrbs_Param1"):
        m = cls(
            mLp=10.0, mLs=4.0, t0_com=T0, xS0_com_E=[1e-3, 2e-3],
            xS0_com_N=[-1e-3, 5e-4], beta_com=1.2, muL_E=-3.0, muL_N=1.5,
            muS_E=1.0, muS_N=-0.5, dL=2000.0, dS=8000.0, omegaL_pri=40.0,
            big_omegaL_sec=20.0, iL=55.0, eL=0.3, tpL=T0 + 30.0, aL=3.5,
            omegaS_pri=100.0, big_omegaS_sec=70.0, iS=30.0, eS=0.2, pS=2500.0,
            tpS=T0 - 100.0, alephS=1.0, aleph_secS=1.6, mag_src_pri=[17.0, 18.0],
            mag_src_sec=[18.0, 19.0], b_sff=[0.9, 0.8], dmag_Lp_Ls=[0.0, 0.0])
        assert_bsbl_keplerian_frames(m, input_is_primary=False)
    return None


def test_bsbl_ellorbs_param2_com_inputs():
    """BSBL EllOrbs Param2: CoM-CoM inputs with u0_amp_com."""
    for mod, cls in _both_modules("BSBL_PhotAstrom_noPar_EllOrbs_Param2"):
        m = cls(
            t0_com=T0, u0_amp_com=0.25, tE=250.0, thetaE=4.0, piS=0.12,
            piE_E=0.03, piE_N=0.02, q=0.4, xS0_com_E=[1e-3, 2e-3],
            xS0_com_N=[-1e-3, 5e-4], muS_E=1.0, muS_N=-0.5, omegaL_pri=40.0,
            big_omegaL_sec=20.0, iL=55.0, eL=0.3, tpL=T0 + 30.0, aL=3.5,
            omegaS_pri=100.0, big_omegaS_sec=70.0, iS=30.0, eS=0.2, pS=2500.0,
            tpS=T0 - 100.0, alephS=1.0, aleph_secS=1.6, fratio_bin=[0.4, 0.4],
            mag_base=[16.8, 17.9], b_sff=[0.9, 0.8], dmag_Lp_Ls=[0.0, 0.0])
        assert_bsbl_keplerian_frames(m, input_is_primary=False)
    return None


def test_bsbl_ellorbs_param3_primary_inputs():
    """BSBL EllOrbs Param3: primary-primary inputs."""
    for mod, cls in _both_modules("BSBL_PhotAstrom_noPar_EllOrbs_Param3"):
        m = cls(
            mLp=10.0, mLs=4.0, t0_p=T0, xS0_p_E=[1e-3, 2e-3],
            xS0_p_N=[-1e-3, 5e-4], beta_p=1.2, muL_E=-3.0, muL_N=1.5,
            muS_E=1.0, muS_N=-0.5, dL=2000.0, dS=8000.0, omegaL_pri=40.0,
            big_omegaL_sec=20.0, iL=55.0, eL=0.3, tpL=T0 + 30.0, aL=3.5,
            omegaS_pri=100.0, big_omegaS_sec=70.0, iS=30.0, eS=0.2,
            tpS=T0 - 100.0, aS=2.6, mSp=1.0, mSs=0.6,
            mag_src_pri=[17.0, 18.0], mag_src_sec=[18.0, 19.0],
            b_sff=[0.9, 0.8], dmag_Lp_Ls=[0.0, 0.0])
        assert_bsbl_keplerian_frames(m, input_is_primary=True)
    return None


def test_bsbl_circorbs_param1_com_inputs():
    """BSBL CircOrbs Param1: CoM-CoM inputs."""
    for mod, cls in _both_modules("BSBL_PhotAstrom_noPar_CircOrbs_Param1"):
        m = cls(
            mLp=10.0, mLs=4.0, t0_com=T0, xS0_com_E=[1e-3, 2e-3],
            xS0_com_N=[-1e-3, 5e-4], beta_com=1.2, muL_E=-3.0, muL_N=1.5,
            muS_E=1.0, muS_N=-0.5, dL=2000.0, dS=8000.0, omegaL_pri=40.0,
            big_omegaL_sec=20.0, iL=55.0, tpL=T0 + 30.0, sepL=3.5,
            omegaS_pri=100.0, big_omegaS_sec=70.0, iS=30.0, pS=2500.0,
            tpS=T0 - 100.0, alephS=1.0, aleph_secS=1.6, mag_src_pri=[17.0, 18.0],
            mag_src_sec=[18.0, 19.0], b_sff=[0.9, 0.8], dmag_Lp_Ls=[0.0, 0.0])
        assert_bsbl_keplerian_frames(m, input_is_primary=False)
    return None


def test_bfsbl_ellorbs_param1_com_inputs():
    """BFSBL EllOrbs Param1: CoM-CoM inputs (model_jax only)."""
    for mod, cls in _both_modules("BFSBL_PhotAstrom_noPar_EllOrbs_Param1"):
        m = cls(
            mLp=10.0, mLs=4.0, t0_com=T0, xS0_com_E=[1e-3, 2e-3],
            xS0_com_N=[-1e-3, 5e-4], beta_com=1.2, muL_E=-3.0, muL_N=1.5,
            muS_E=1.0, muS_N=-0.5, dL=2000.0, dS=8000.0, omegaL_pri=40.0,
            big_omegaL_sec=20.0, iL=55.0, eL=0.3, tpL=T0 + 30.0, aL=3.5,
            omegaS_pri=100.0, big_omegaS_sec=70.0, iS=30.0, eS=0.2, pS=2500.0,
            tpS=T0 - 100.0, alephS=1.0, aleph_secS=1.6, mag_src_pri=[17.0, 18.0],
            mag_src_sec=[18.0, 19.0], b_sff=[0.9, 0.8], dmag_Lp_Ls=[0.0, 0.0],
            radiusS_pri=1e-6, radiusS_sec=1e-6)
        assert_bsbl_keplerian_frames(m, input_is_primary=False)
    return None


def test_bsbl_static_param2_primary_inputs():
    """Static BSBL Param2: primary inputs, midpoint internal frame."""
    for mod, cls in _both_modules("BSBL_PhotAstrom_noPar_Param2"):
        m = cls(
            mLp=10.0, mLs=4.0, t0_p=T0, xS0_p_E=[1e-3, 2e-3],
            xS0_p_N=[-1e-3, 5e-4], beta_p=1.2, muL_E=-3.0, muL_N=1.5,
            muS_E=1.0, muS_N=-0.5, dL=2000.0, dS=8000.0, sepL=3.5, alphaL=70.0,
            sepS=2.5, alphaS=200.0, qS=0.6, mag_src_pri=[17.0, 18.0],
            mag_src_sec=[18.0, 19.0], b_sff=[0.9, 0.8], dmag_Lp_Ls=[0.0, 0.0])

        src, _ = bsbl_points(m, m.t0_p)
        np.testing.assert_allclose(src["p"][0], [2.0, 0.5], atol=1e-9)
        assert_bsbl_pair(m, m.t0_p, m.beta_p / m.thetaE_amp, "p", "p")
        assert_bsbl_pair(m, m.t0, m.u0_amp, "p", "mid")
        assert m.t0_mid == m.t0 and m.u0_amp_mid == m.u0_amp
        assert_bsbl_pair(m, m.t0_com, m.u0_amp_com, "com", "com")
        src, lens = bsbl_points(m, m.t0_com)
        np.testing.assert_allclose(m.xS0_com[1] * 1e3, src["com"][0], atol=1e-9)
        np.testing.assert_allclose(m.xL0_com[1] * 1e3, lens["com"][0], atol=1e-9)
    return None


def test_bfsbl_static_param2_primary_inputs():
    """Static BFSBL Param2: primary inputs (model_jax only)."""
    for mod, cls in _both_modules("BFSBL_PhotAstrom_noPar_Param2"):
        m = cls(
            mLp=10.0, mLs=4.0, t0_p=T0, xS0_p_E=[1e-3, 2e-3],
            xS0_p_N=[-1e-3, 5e-4], beta_p=1.2, muL_E=-3.0, muL_N=1.5,
            muS_E=1.0, muS_N=-0.5, dL=2000.0, dS=8000.0, sepL=3.5, alphaL=70.0,
            sepS=2.5, alphaS=200.0, qS=0.6, mag_src_pri=[17.0, 18.0],
            mag_src_sec=[18.0, 19.0], b_sff=[0.9, 0.8], dmag_Lp_Ls=[0.0, 0.0],
            radiusS_pri=1e-6, radiusS_sec=1e-6)

        src, _ = bsbl_points(m, m.t0_p)
        np.testing.assert_allclose(src["p"][0], [2.0, 0.5], atol=1e-9)
        assert_bsbl_pair(m, m.t0_p, m.beta_p / m.thetaE_amp, "p", "p")
        assert_bsbl_pair(m, m.t0, m.u0_amp, "p", "mid")
        assert_bsbl_pair(m, m.t0_com, m.u0_amp_com, "com", "com")
    return None


# ----------------------------------------------------------------------
# frame_convert.convert_u0_amp_t0_bsbl_orbit round trips
# ----------------------------------------------------------------------
def test_bsbl_orbit_converter_round_trips():
    """Pairs convert to the CoM and back; one source matches the PSBL wrapper."""
    from bagle import frame_convert as fc

    orbL = model.keplerian_lens_orbit(40.0, 20.0, 55.0, 0.3, T0 + 30.0, 900.0,
                                      2e-3, 1.5e-3)
    orbS = model.keplerian_lens_orbit(100.0, 70.0, 30.0, 0.2, T0 - 100.0, 2500.0,
                                      1e-3, 1.6e-3)
    th_hat = np.array([0.6, 0.8])
    kw = dict(tE=250.0, thetaE_hat=th_hat, thetaE_orb=4e-3)

    for lens_pt, src_pt in [("geom_mid", "prim_center"),
                            ("prim_center", "prim_center"),
                            ("COM", "prim_center"), ("geom_mid", "COM")]:
        t_c, u_c = fc.convert_u0_amp_t0_bsbl_orbit(
            T0, 0.3, orbL=orbL, orbS=orbS, lens_in=lens_pt, lens_out="COM",
            src_in=src_pt, src_out="COM", **kw)
        t_b, u_b = fc.convert_u0_amp_t0_bsbl_orbit(
            t_c, u_c, orbL=orbL, orbS=orbS, lens_in="COM", lens_out=lens_pt,
            src_in="COM", src_out=src_pt, **kw)
        np.testing.assert_allclose([t_b, u_b], [T0, 0.3], rtol=0, atol=1e-8)

    for pts in [("COM", "geom_mid"), ("geom_mid", "COM"),
                ("prim_center", "COM")]:
        a = fc.convert_u0_amp_t0_psbl_orbit(T0, -0.2, 250.0, th_hat, orbL, 4e-3,
                                            *pts)
        b = fc.convert_u0_amp_t0_bsbl_orbit(T0, -0.2, 250.0, th_hat, orbL, None,
                                            4e-3, lens_in=pts[0],
                                            lens_out=pts[1])
        np.testing.assert_array_equal(a, b)
    return None


# ----------------------------------------------------------------------
# BSPL Keplerian: input xS0 is the primary at t0
# ----------------------------------------------------------------------
def assert_bspl_xS0_at_t0(m, xS0_E, xS0_N):
    """
    Check that the primary at t0 matches the input xS0.

    Parameters
    ----------
    m : model instance
        BSPL Keplerian PhotAstrom model.
    xS0_E, xS0_N : float
        Input primary East / North at t0 (arcsec).

    Returns
    -------
    None
    """
    xS = m.get_resolved_source_astrometry_unlensed(np.array([m.t0]))[0]
    xS0 = np.array([xS0_E, xS0_N], dtype=float)
    np.testing.assert_allclose(xS[0], xS0, atol=1e-12)
    np.testing.assert_allclose(m.xS0_pri[0], xS0, atol=1e-12)
    np.testing.assert_allclose(m.xS0_sec[0], xS[1], atol=1e-12)

    orb = orbits.Orbit()
    orb.w, orb.o, orb.i, orb.e = m.omega_pri, m.big_omega_sec, m.i, m.e
    orb.tp, orb.p = m.tp, m.p
    orb.aleph, orb.aleph2 = m.aleph * 1e-3, m.aleph_sec * 1e-3
    x, y, x2, y2 = orb.oal2xy(np.array([m.t0]))
    d_pri = np.array([x[0], y[0]])
    np.testing.assert_allclose(m.xS0_com[0], xS0 - d_pri, atol=1e-12)

    np.testing.assert_allclose(m.sep, m.aleph + m.aleph_sec, rtol=1e-12)
    np.testing.assert_allclose(
        m.sep_t0, np.hypot(x2[0] - x[0], y2[0] - y[0]) * 1e3, atol=1e-9)
    return None


def test_bspl_ellorbs_param1_xS0_at_t0():
    """BSPL EllOrbs Param1: primary at t0 equals input xS0."""
    for mod, cls in _both_modules("BSPL_PhotAstrom_noPar_EllOrbs_Param1"):
        m = cls(
            mL=8.0, t0=T0, beta=1.0, dL=4000.0, dL_dS=0.5, xS0_E=[0.002],
            xS0_N=[-0.001], muL_E=0.0, muL_N=0.0, muS_E=4.0, muS_N=-1.0,
            omega_pri=30.0, big_omega_sec=10.0, i=90.0, e=0.6, p=1000.0, tp=T0,
            aleph=3.0, aleph_sec=8.0, mag_src_pri=[16.0], mag_src_sec=[17.0],
            b_sff=[1.0])
        assert_bspl_xS0_at_t0(m, 0.002, -0.001)
    return None


def test_bspl_ellorbs_param2_xS0_at_t0():
    """BSPL EllOrbs Param2: primary at t0 equals input xS0."""
    for mod, cls in _both_modules("BSPL_PhotAstrom_noPar_EllOrbs_Param2"):
        m = cls(
            t0=T0, u0_amp=0.3, tE=200.0, thetaE=3.0, piS=0.125, piE_E=0.05,
            piE_N=0.0, omega_pri=30.0, big_omega_sec=10.0, i=90.0, e=0.6,
            p=1000.0, tp=T0, aleph=3.0, aleph_sec=8.0, muS_E=4.0, muS_N=-1.0,
            xS0_E=[0.002], xS0_N=[-0.001], fratio_bin=[0.4], mag_base=[16.0],
            b_sff=[1.0])
        assert_bspl_xS0_at_t0(m, 0.002, -0.001)
    return None


def test_bspl_circorbs_param1_xS0_at_t0():
    """BSPL CircOrbs Param1: primary at t0 equals input xS0."""
    for mod, cls in _both_modules("BSPL_PhotAstrom_noPar_CircOrbs_Param1"):
        m = cls(
            mL=8.0, t0=T0, beta=1.0, dL=4000.0, dL_dS=0.5, xS0_E=[0.002],
            xS0_N=[-0.001], muL_E=0.0, muL_N=0.0, muS_E=4.0, muS_N=-1.0,
            omega_pri=30.0, big_omega_sec=10.0, i=90.0, p=1000.0, tp=T0,
            aleph=3.0, aleph_sec=8.0, mag_src_pri=[16.0], mag_src_sec=[17.0],
            b_sff=[1.0])
        assert_bspl_xS0_at_t0(m, 0.002, -0.001)
    return None


def test_bspl_circorbs_param2_xS0_at_t0():
    """BSPL CircOrbs Param2: primary at t0 equals input xS0."""
    for mod, cls in _both_modules("BSPL_PhotAstrom_noPar_CircOrbs_Param2"):
        m = cls(
            t0=T0, u0_amp=0.3, tE=200.0, thetaE=3.0, piS=0.125, piE_E=0.05,
            piE_N=0.0, omega_pri=30.0, big_omega_sec=10.0, i=90.0, p=1000.0,
            tp=T0, aleph=3.0, aleph_sec=8.0, muS_E=4.0, muS_N=-1.0,
            xS0_E=[0.002], xS0_N=[-0.001], fratio_bin=[0.4], mag_base=[16.0],
            b_sff=[1.0])
        assert_bspl_xS0_at_t0(m, 0.002, -0.001)
    return None


def _assert_bspl_amp_vs_vbm_psplmag(m):
    """
    Compare BSPL amplification to flux-weighted VBM PSPLMag on BAGLE |u|.

    VBMicrolensing has circular xallarap binary-source models, not a full
    Thiele-Innes Keplerian source orbit.

    Parameters
    ----------
    m : model instance
        BSPL Keplerian PhotAstrom model.

    Returns
    -------
    None
    """
    VBM = pytest.importorskip("VBMicrolensing")

    t = np.linspace(m.t0 - 1.5 * m.tE, m.t0 + 1.5 * m.tE, 251)
    u = m.get_u(t)
    u_pri = np.linalg.norm(u[:, 0], axis=1)
    u_sec = np.linalg.norm(u[:, 1], axis=1)

    vbm = VBM.VBMicrolensing()
    a_pri = np.array([vbm.PSPLMag(float(ui)) for ui in u_pri])
    a_sec = np.array([vbm.PSPLMag(float(ui)) for ui in u_sec])

    f1 = 10.0 ** (-0.4 * m.mag_src_pri[0])
    f2 = 10.0 ** (-0.4 * m.mag_src_sec[0])
    a_vbm = (f1 * a_pri + f2 * a_sec) / (f1 + f2)
    a_bagle = np.asarray(m.get_amplification(t), dtype=float)

    np.testing.assert_allclose(a_bagle, a_vbm, rtol=1e-10, atol=1e-12)
    return None


def test_bspl_ellorbs_param1_amp_vs_vbm_psplmag():
    """BSPL EllOrbs Param1 amplification matches flux-weighted VBM PSPLMag."""
    for mod, cls in _both_modules("BSPL_PhotAstrom_noPar_EllOrbs_Param1"):
        m = cls(
            mL=8.0, t0=T0, beta=1.0, dL=4000.0, dL_dS=0.5, xS0_E=[0.002],
            xS0_N=[-0.001], muL_E=0.0, muL_N=0.0, muS_E=4.0, muS_N=-1.0,
            omega_pri=30.0, big_omega_sec=10.0, i=90.0, e=0.6, p=1000.0, tp=T0,
            aleph=3.0, aleph_sec=8.0, mag_src_pri=[16.0], mag_src_sec=[17.0],
            b_sff=[1.0])
        _assert_bspl_amp_vs_vbm_psplmag(m)
    return None


def test_bspl_ellorbs_param2_amp_vs_vbm_psplmag():
    """BSPL EllOrbs Param2 amplification matches flux-weighted VBM PSPLMag."""
    for mod, cls in _both_modules("BSPL_PhotAstrom_noPar_EllOrbs_Param2"):
        m = cls(
            t0=T0, u0_amp=0.3, tE=200.0, thetaE=3.0, piS=0.125, piE_E=0.05,
            piE_N=0.0, omega_pri=30.0, big_omega_sec=10.0, i=90.0, e=0.6,
            p=1000.0, tp=T0, aleph=3.0, aleph_sec=8.0, muS_E=4.0, muS_N=-1.0,
            xS0_E=[0.002], xS0_N=[-0.001], fratio_bin=[0.4], mag_base=[16.0],
            b_sff=[1.0])
        _assert_bspl_amp_vs_vbm_psplmag(m)
    return None
