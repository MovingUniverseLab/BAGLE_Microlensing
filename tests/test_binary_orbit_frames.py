"""
Reference-frame definitions for Keplerian (EllOrbs / CircOrbs) binary lenses.

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

These tests read the lens and source positions back from the models and
check each definition, then check that a CoM-input class and a
geometric-input class built from its derived values give the same light curve
and astrometry.
"""
from __future__ import annotations

import numpy as np
import pytest

from bagle import model, model_jax

T0 = 57100.0

# Shared orbit (tp well before t0, so the axis at tp differs from the axis at t0).
ORBIT = dict(omega_pri=40.0, big_omega_sec=110.0, i=60.0, e=0.3,
             tp=T0 - 900.0, a=2.0)

# thetaE / piE parameterizations (Param2, 3, 4, 8).
REDUCED = dict(tE=60.0, thetaE=3.0, log10_thetaE=np.log10(3.0), piS=0.125,
               piE_E=0.08, piE_N=-0.06, xS0_E=[0.001], xS0_N=[-0.002],
               xS0_com_E=[0.001], xS0_com_N=[-0.002],
               muS_E=-2.0, muS_N=1.5, q=0.5, b_sff=[0.8], mag_src=[19.0],
               mag_base=[19.0 + 2.5 * np.log10(0.8)], dmag_Lp_Ls=[20.0],
               radiusS=1e-6)

# Mass / distance parameterizations (Param1, 7).
PHYSICAL = dict(mLp=3.0, mLs=1.5, xS0_com_E=[0.001], xS0_com_N=[-0.002],
                xS0_p_E=[0.001], xS0_p_N=[-0.002], muL_E=2.0,
                muL_N=-1.0, muS_E=-3.0, muS_N=-6.0, dL=2400.0, dS=8000.0,
                b_sff=[0.8], mag_src=[19.0], dmag_Lp_Ls=[20.0], radiusS=1e-6)

# (leaf class, constructor values, input frame, input t0 / impact names).
CASES = [
    ("PSBL_PhotAstrom_noPar_EllOrbs_Param1",
     dict(t0_com=T0, beta_com=0.4), "COM", "t0_com", "beta_com"),
    ("PSBL_PhotAstrom_noPar_EllOrbs_Param2",
     dict(t0=T0, u0_amp=0.15), "geom_mid", "t0", "u0_amp"),
    ("PSBL_PhotAstrom_noPar_EllOrbs_Param3",
     dict(t0=T0, u0_amp=0.15), "geom_mid", "t0", "u0_amp"),
    ("PSBL_PhotAstrom_noPar_EllOrbs_Param4",
     dict(t0_com=T0, u0_amp_com=0.15), "COM", "t0_com", "u0_amp_com"),
    ("PSBL_PhotAstrom_noPar_EllOrbs_Param7",
     dict(t0_p=T0, beta_p=0.4), "prim_center", "t0_p", "beta_p"),
    ("PSBL_PhotAstrom_noPar_EllOrbs_Param8",
     dict(t0_com=T0, u0_amp_com=0.15), "COM", "t0_com", "u0_amp_com"),
    ("PSBL_PhotAstrom_noPar_CircOrbs_Param4",
     dict(t0_com=T0, u0_amp_com=-0.15), "COM", "t0_com", "u0_amp_com"),
    ("FSBL_PhotAstrom_noPar_EllOrbs_Param1",
     dict(t0_com=T0, beta_com=-0.4), "COM", "t0_com", "beta_com"),
    ("FSBL_PhotAstrom_noPar_EllOrbs_Param4",
     dict(t0_com=T0, u0_amp_com=0.15), "COM", "t0_com", "u0_amp_com"),
    ("FSBL_PhotAstrom_noPar_EllOrbs_Param7",
     dict(t0_p=T0, beta_p=0.4), "prim_center", "t0_p", "beta_p"),
]

MODULES = pytest.mark.parametrize("mod", [model, model_jax],
                                  ids=["model", "model_jax"])


def build(mod, name, frame_vals):
    """
    Orbiting binary-lens model from the shared parameter sets.

    Parameters
    ----------
    mod : module
        :mod:`bagle.model` or :mod:`bagle.model_jax`.
    name : str
        Composed (``noPar``) class name.
    frame_vals : dict
        Input-frame closest approach (e.g. ``t0_com`` and ``u0_amp_com``).

    Returns
    -------
    m : model instance
    """
    if not hasattr(mod, name):
        pytest.skip(f"{name} is not in {mod.__name__}")
    cls = getattr(mod, name)
    vals = {**(PHYSICAL if "mLp" in cls.fitter_param_names else REDUCED),
            **ORBIT, **frame_vals}

    # Positional constructor arguments, in fitter_param_names order. The
    # CircOrbs classes have no 'e'; finite-source classes add radiusS.
    names = [n for n in cls.fitter_param_names if n != "e" or "Ell" in name]
    m = cls(*[vals[n] for n in names], raL=268.0, decL=-29.0)

    return m


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


@MODULES
@pytest.mark.parametrize("name,frame_vals,frame,t_name,u_name", CASES)
def test_orbit_frame_definitions(mod, name, frame_vals, frame, t_name, u_name):
    """Geometric, CoM and input-frame closest approaches match the geometry."""
    m = build(mod, name, frame_vals)

    # Input values are kept as given.
    assert getattr(m, t_name) == frame_vals[t_name]
    assert getattr(m, u_name) == frame_vals[u_name]

    # The input xS0 (xS0_E, xS0_com_E or xS0_p_E) is the source position at
    # the input reference time.
    src = m.get_source_astrometry_unlensed(np.array([frame_vals[t_name]]))
    vals = PHYSICAL if "mLp" in type(m).fitter_param_names else REDUCED
    x_names = [n for n in type(m).fitter_param_names if n.startswith("xS0")]
    np.testing.assert_allclose(np.ravel(src),
                               [vals[x_names[0]][0], vals[x_names[1]][0]],
                               rtol=0, atol=1e-12)
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

    # Internal frame: the midpoint pair for midpoint inputs, else the CoM.
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


@MODULES
@pytest.mark.parametrize("com_name,geo_name", [
    ("PSBL_PhotAstrom_noPar_EllOrbs_Param4", "PSBL_PhotAstrom_noPar_EllOrbs_Param2"),
    ("PSBL_PhotAstrom_noPar_EllOrbs_Param8", "PSBL_PhotAstrom_noPar_EllOrbs_Param3"),
])
def test_com_and_geometric_inputs_agree(mod, com_name, geo_name):
    """A geometric-input model built from a CoM model's _mid values matches it."""
    m_com = build(mod, com_name, dict(t0_com=T0, u0_amp_com=0.15))

    # Geometric inputs take xS0 at t0_mid (the lazy midpoint epoch).
    xS_mid = m_com.get_source_astrometry_unlensed(np.array([m_com.t0_mid]))
    vals = {**REDUCED, **ORBIT, "t0": m_com.t0_mid, "u0_amp": m_com.u0_amp_mid,
            "xS0_E": xS_mid[:, 0], "xS0_N": xS_mid[:, 1]}
    cls = getattr(mod, geo_name)
    m_geo = cls(*[vals[n] for n in cls.fitter_param_names], raL=268.0,
                decL=-29.0)

    # The CoM closest approach comes back unchanged.
    np.testing.assert_allclose([m_geo.t0_com, m_geo.u0_amp_com], [T0, 0.15],
                               rtol=0, atol=1e-8)

    # Same magnification and lens / source astrometry.
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


# Phot-only orbit inputs, keyed by fitter_param_names.
PHOT = dict(t0=T0, u0_amp=0.15, tE=60.0, piE_E=0.08, piE_N=-0.06, q=0.5, sep=1.2,
            phi=35.0, v_para=0.3, v_perp=0.4, v_rad=-0.6, r_s=0.5, a_s=1.5,
            radiusS=1e-3, b_sff=[0.8], mag_src=[19.0], gamma=[0.4])

PHOT_CLASSES = ["PSBL_Phot_noPar_EllOrbs_Param1", "PSBL_Phot_noPar_CircOrbs_Param1",
                "FSBL_Phot_noPar_EllOrbs_Param1", "FSBL_Phot_noPar_CircOrbs_Param1",
                "FSBL_Limb_Phot_Par_EllOrbs_Param1",
                "FSBL_Limb_Phot_Par_CircOrbs_Param1"]


@MODULES
@pytest.mark.parametrize("name", PHOT_CLASSES)
def test_phot_orbit_positional_fitter_order(mod, name):
    """Phot-only orbit classes build positionally from fitter_param_names."""
    if not hasattr(mod, name):
        pytest.skip(f"{name} is not in {mod.__name__}")
    cls = getattr(mod, name)
    m = cls(*[PHOT[n] for n in cls.fitter_param_names], raL=268.0, decL=-29.0)

    # Each fitter parameter lands on the attribute of the same name (piE is
    # stored as a vector).
    stored = {"piE_E": m.piE[0], "piE_N": m.piE[1]}
    for n in cls.fitter_param_names:
        np.testing.assert_allclose(np.ravel(stored.get(n, getattr(m, n, None))),
                                   np.ravel(PHOT[n]), err_msg=n)

    # The semi-major axes about the CoM split a in the mass ratio.
    np.testing.assert_allclose(m.aleph / m.aleph_sec, m.q, rtol=1e-12)
    return None


def relative_los(m, t):
    """
    Line-of-sight offset of the secondary from the primary at time t.

    Parameters
    ----------
    m : model instance
        Phot-only Keplerian model.
    t : float
        Time (MJD).

    Returns
    -------
    z : float
        Secondary minus primary along the line of sight (Einstein radii),
        from the orbit's Thiele-Innes constants.
    """
    orb = model.keplerian_lens_orbit(m.w, m.o, m.i, m.e, m.tp, m.p, m.aleph,
                                     m.aleph_sec)
    orb.oal2xy(np.array([t]))

    # Orbit-plane coordinates at t.
    mean_anom = 2.0 * np.pi * (t - m.tp) / m.p
    ecc_anom = orb.eccen_anomaly(np.array([mean_anom]), m.e)[0]
    X = np.cos(ecc_anom) - m.e
    Y = np.sqrt(1.0 - m.e ** 2) * np.sin(ecc_anom)

    return ((orb.conC2 - orb.conC) * X + (orb.conH2 - orb.conH) * Y)


@MODULES
@pytest.mark.parametrize("name,state", [
    ("PSBL_Phot_noPar_EllOrbs_Param1", {}),
    ("PSBL_Phot_noPar_EllOrbs_Param1", dict(v_para=-0.3, v_perp=-0.4, phi=200.0)),
    ("PSBL_Phot_noPar_EllOrbs_Param1", dict(v_para=0.3, v_rad=-0.6, a_s=1.0001)),
    ("PSBL_Phot_noPar_EllOrbs_Param1", dict(v_para=0.3, v_rad=-0.6, a_s=1.0)),
    ("PSBL_Phot_noPar_CircOrbs_Param1", {}),
    ("FSBL_Phot_noPar_EllOrbs_Param1", dict(phi=-70.0)),
], ids=["eccentric", "eccentric-flipped", "near-circular", "circular",
        "CircOrbs", "FSBL"])
def test_phot_orbit_reproduces_state_at_t0(mod, name, state):
    """The orbit reproduces sep, alpha, r_s and the velocities at t0."""
    if not hasattr(mod, name):
        pytest.skip(f"{name} is not in {mod.__name__}")
    cls = getattr(mod, name)
    vals = {**PHOT, **state}
    m = cls(*[vals[n] for n in cls.fitter_param_names], raL=268.0, decL=-29.0)

    # Secondary minus primary at t0 and +-dt (Einstein radii).
    dt = 1e-3
    lens = m.get_resolved_lens_astrometry(np.array([T0 - dt, T0, T0 + dt]))
    rel = lens[:, 1] - lens[:, 0]
    rel_dot = (rel[2] - rel[0]) / (2 * dt) * 365.25  # per year

    # Expected state: separation along s_hat (PA alpha + 180), velocity
    # along s_hat and the direction of increasing position angle.
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

    # The axis angle is the input phi relative to the piE direction.
    pa_piE = np.arctan2(m.piE[0], m.piE[1])
    np.testing.assert_allclose(np.rad2deg(alpha - pa_piE), vals["phi"],
                               atol=1e-9)
    return None


@MODULES
def test_phot_orbit_frame_definitions(mod):
    """Phot-only orbits: u0 about the midpoint at t0, u0_com about the CoM."""
    cls = mod.PSBL_Phot_noPar_EllOrbs_Param1
    m = cls(*[PHOT[n] for n in cls.fitter_param_names], raL=268.0, decL=-29.0)

    def rel(t, frame):
        """Source minus a lens point at time t (Einstein units)."""
        w, z1, z2 = m.get_complex_pos(np.array([t]))
        z = {"geom_mid": 0.5 * (z1 + z2),
             "COM": m.m1 * z1 + m.m2 * z2}[frame]
        d = (w - z)[0]
        return np.array([d.real, d.imag])

    # Geometric midpoint at t0 and the CoM at t0_com.
    np.testing.assert_allclose(rel(m.t0, "geom_mid"), m.u0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(rel(m.t0_com, "COM"), m.u0_com, rtol=0,
                               atol=1e-9)
    assert abs(rel(m.t0_com, "COM") @ m.thetaE_hat) < 1e-9
    return None


@pytest.mark.parametrize("name,state", [
    ("PSBL_Phot_noPar_EllOrbs_Param1",
     dict(phi=35.0, v_para=1.0, v_perp=2.0, v_rad=-1.5, r_s=0.5, a_s=1.4)),
    ("PSBL_Phot_noPar_EllOrbs_Param1",
     dict(phi=200.0, v_para=-0.5, v_perp=-3.0, v_rad=1.0, r_s=-0.3, a_s=2.0)),
    ("PSBL_Phot_noPar_CircOrbs_Param1",
     dict(phi=-60.0, v_para=0.8, v_perp=2.5, v_rad=-1.6)),
], ids=["eccentric", "eccentric-retro", "circular"])
def test_phot_orbit_matches_vbm_kepler(name, state):
    """Phot-only Keplerian light curve equals VBMicrolensing's Kepler model.

    VBM's BinaryLightCurveKepler takes the state at its t0 (the CoM closest
    approach): s, w1 = (ds/dt)/s, w2 = dalpha/dt, w3 = (dsz/dt)/s, sz/s and
    a/r3D, with rates per day. Those are read from our orbit at t0_com; with
    the on-sky axes below, w1 and w2 are our v_para and v_perp conventions.
    """
    VBM = pytest.importorskip("VBMicrolensing")
    vals = dict(PHOT, u0_amp=0.5, sep=1.1, b_sff=[1.0], **state)
    cls = getattr(model, name)
    m = cls(*[vals[n] for n in cls.fitter_param_names], raL=268.0, decL=-29.0)

    # Relative state (secondary - primary) at t0_com, per day.
    tc, dt = m.t0_com, 1e-3
    lens = m.get_resolved_lens_astrometry(np.array([tc - dt, tc, tc + dt]))
    rel = lens[:, 1] - lens[:, 0]
    rel_dot = (rel[2] - rel[0]) / (2 * dt)
    z = relative_los(m, tc)
    z_dot = (relative_los(m, tc + dt) - relative_los(m, tc - dt)) / (2 * dt)

    # VBM axes: x from primary to secondary, y = -(increasing position angle).
    s = np.hypot(*rel[1])
    ex = rel[1] / s
    ey = np.array([-ex[1], ex[0]])
    w1 = rel_dot @ ex / s
    w2 = -(rel_dot @ ey) / s
    ar = m.a / (s * np.sqrt(1.0 + (z / s) ** 2))

    # Source track about the CoM in VBM axes (as in test_binary_frames_vs_vbm).
    v = np.array([m.thetaE_hat @ ex, m.thetaE_hat @ ey])
    alpha = np.arctan2(-v[1], -v[0])

    # VBM needs coordinates (parallax is off) and HJD - 2450000 times.
    shift = 50000.0
    vbm = VBM.VBMicrolensing()
    vbm.Tol = 1e-6
    vbm.RelTol = 1e-7
    vbm.SetObjectCoordinates("17:52:00 -29:00:00")
    t = np.linspace(tc - 1.5 * m.tE, tc + 1.5 * m.tE, 401)
    par = [np.log(s), np.log(m.q), abs(m.u0_amp_com), alpha, np.log(1e-5),
           np.log(m.tE), tc - shift, 0.0, 0.0, w1, w2, z_dot / s, z / s, ar]
    a_vbm = np.array(vbm.BinaryLightCurveKepler(par, list(t - shift))[0])

    a_bagle = m.get_amplification(t)
    assert np.max(np.abs(a_vbm / a_bagle - 1)) < 1e-4

    # At our own reference epoch t0, the same VBM rates are v_para and v_perp
    # (per year), which pins the v_perp sign convention.
    lens = m.get_resolved_lens_astrometry(np.array([m.t0 - dt, m.t0, m.t0 + dt]))
    rel = lens[:, 1] - lens[:, 0]
    rel_dot = (rel[2] - rel[0]) / (2 * dt) * 365.25
    ex = rel[1] / np.hypot(*rel[1])
    ey = np.array([-ex[1], ex[0]])
    np.testing.assert_allclose([rel_dot @ ex / m.sep, -(rel_dot @ ey) / m.sep],
                               [m.v_para, m.v_perp], rtol=1e-5)
    return None


# ----------------------------------------------------------------------
# Binary source + binary lens (BSBL / BFSBL) frames.
# ----------------------------------------------------------------------
BSBL_VALS = dict(mLp=10.0, mLs=4.0, t0_com=T0, t0_p=T0, beta_com=1.2, beta_p=1.2,
                 u0_amp_com=0.25, tE=250.0, thetaE=4.0, piS=0.12, piE_E=0.03,
                 piE_N=0.02, q=0.4, muL_E=-3.0, muL_N=1.5, muS_E=1.0, muS_N=-0.5,
                 dL=2000.0, dS=8000.0, omegaL_pri=40.0, big_omegaL_sec=20.0,
                 iL=55.0, eL=0.3, tpL=T0 + 30.0, aL=3.5, sepL=3.5,
                 omegaS_pri=100.0, big_omegaS_sec=70.0, iS=30.0, eS=0.2,
                 pS=2500.0, tpS=T0 - 100.0, alephS=1.0, aleph_secS=1.6, aS=2.6,
                 mSp=1.0, mSs=0.6, mag_src_pri=[17.0, 18.0],
                 mag_src_sec=[18.0, 19.0], b_sff=[0.9, 0.8],
                 dmag_Lp_Ls=[0.0, 0.0], fratio_bin=[0.4, 0.4],
                 mag_base=[16.8, 17.9], radiusS_pri=1e-6, radiusS_sec=1e-6,
                 alphaL=70.0, sepS=2.5, alphaS=200.0, qS=0.6,
                 xS0_com_E=[1e-3, 2e-3], xS0_com_N=[-1e-3, 5e-4],
                 xS0_p_E=[1e-3, 2e-3], xS0_p_N=[-1e-3, 5e-4])

BSBL_KEP = [f"{fam}_PhotAstrom_noPar_{kind}Orbs_Param{n}"
            for fam in ("BSBL", "BFSBL") for n in (1, 2, 3)
            for kind in ("Ell", "Circ")]


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

    # CoMs weighted by the mass ratios.
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


@MODULES
@pytest.mark.parametrize("name", BSBL_KEP)
def test_bsbl_keplerian_frames(mod, name):
    """Input, internal (CoM), lazy midpoint and primary frames of BSBL orbits."""
    if not hasattr(mod, name):
        pytest.skip(f"{name} is not in {mod.__name__}")
    cls = getattr(mod, name)
    m = cls(*[BSBL_VALS[n] for n in cls.fitter_param_names])

    # The CoM-CoM pair is a true closest approach at t0_com.
    tt = np.linspace(m.t0_com - 300, m.t0_com + 300, 120001)
    src, lens = bsbl_points(m, tt)
    d = np.hypot(*(src["com"] - lens["com"]).T)
    assert abs(tt[np.argmin(d)] - m.t0_com) < 0.01
    assert_bsbl_pair(m, m.t0_com, m.u0_amp_com, "com", "com")

    # Internal frame is the CoM pair; xS0_com / xL0_com are the CoMs there.
    assert m.t0 == m.t0_com and m.u0_amp == m.u0_amp_com
    src, lens = bsbl_points(m, m.t0_com)
    np.testing.assert_allclose(m.xS0_com[1] * 1e3, src["com"][0], atol=1e-9)
    np.testing.assert_allclose(m.xL0_com[1] * 1e3, lens["com"][0], atol=1e-9)

    # The inputs are honored in their own frame.
    if "t0_p" in cls.fitter_param_names:
        assert m.t0_p == T0
        src, _ = bsbl_points(m, T0)
        np.testing.assert_allclose(src["p"][0], [2.0, 0.5], atol=1e-9)
    else:
        assert m.t0_com == T0
        np.testing.assert_allclose(m.xS0_com[1] * 1e3, [2.0, 0.5], atol=1e-9)

    # Lazy midpoint and primary pairs follow the instantaneous definition.
    assert_bsbl_pair(m, m.t0_mid, m.u0_amp_mid, "p", "mid")
    _, lens = bsbl_points(m, m.t0_mid)
    np.testing.assert_allclose(m.xL0_mid[1] * 1e3, lens["mid"][0], atol=1e-9)
    assert_bsbl_pair(m, m.t0_p, m.beta_p / m.thetaE_amp, "p", "p")

    # Source mass ratio, and the axes at t0 (lens: sec -> pri; source:
    # pri -> sec).
    np.testing.assert_allclose(m.qS, m.mSs / m.mSp, rtol=1e-12)
    src, lens = bsbl_points(m, m.t0)
    S = m.get_resolved_source_astrometry_unlensed(np.array([m.t0]), 1)[0]
    L = m.get_resolved_lens_astrometry(np.array([m.t0]), 1)[0]
    dl, ds = L[0] - L[1], S[1] - S[0]
    np.testing.assert_allclose(m.alphaL_rad, np.arctan2(dl[0], dl[1]), atol=1e-9)
    np.testing.assert_allclose(np.cos(m.alphaS_rad - np.arctan2(ds[0], ds[1])),
                               1.0, atol=1e-12)
    return None


@MODULES
@pytest.mark.parametrize("name", ["BSBL_PhotAstrom_noPar_Param2",
                                  "BFSBL_PhotAstrom_noPar_Param2"])
def test_bsbl_static_param2_frames(mod, name):
    """Static BSBL Param2: primary inputs, midpoint internal frame, CoM pair."""
    if not hasattr(mod, name):
        pytest.skip(f"{name} is not in {mod.__name__}")
    cls = getattr(mod, name)
    m = cls(*[BSBL_VALS[n] for n in cls.fitter_param_names])

    # xS0_p is the primary source at t0_p; the primary pair holds there.
    src, _ = bsbl_points(m, m.t0_p)
    np.testing.assert_allclose(src["p"][0], [2.0, 0.5], atol=1e-9)
    assert_bsbl_pair(m, m.t0_p, m.beta_p / m.thetaE_amp, "p", "p")

    # Internal midpoint pair (signed u0_amp), with _mid aliases.
    assert_bsbl_pair(m, m.t0, m.u0_amp, "p", "mid")
    assert m.t0_mid == m.t0 and m.u0_amp_mid == m.u0_amp

    # CoM-CoM pair from qS and the lens masses.
    assert_bsbl_pair(m, m.t0_com, m.u0_amp_com, "com", "com")
    src, lens = bsbl_points(m, m.t0_com)
    np.testing.assert_allclose(m.xS0_com[1] * 1e3, src["com"][0], atol=1e-9)
    np.testing.assert_allclose(m.xL0_com[1] * 1e3, lens["com"][0], atol=1e-9)
    return None


# ----------------------------------------------------------------------
# frame_convert.convert_u0_amp_t0_bsbl_orbit round trips.
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

    for lens_pt, src_pt in [("geom_mid", "prim_center"), ("prim_center", "prim_center"),
                            ("COM", "prim_center"), ("geom_mid", "COM")]:
        t_c, u_c = fc.convert_u0_amp_t0_bsbl_orbit(
            T0, 0.3, orbL=orbL, orbS=orbS, lens_in=lens_pt, lens_out="COM",
            src_in=src_pt, src_out="COM", **kw)
        t_b, u_b = fc.convert_u0_amp_t0_bsbl_orbit(
            t_c, u_c, orbL=orbL, orbS=orbS, lens_in="COM", lens_out=lens_pt,
            src_in="COM", src_out=src_pt, **kw)
        np.testing.assert_allclose([t_b, u_b], [T0, 0.3], rtol=0, atol=1e-8)

    # A single source reproduces the PSBL wrapper exactly.
    for pts in [("COM", "geom_mid"), ("geom_mid", "COM"), ("prim_center", "COM")]:
        a = fc.convert_u0_amp_t0_psbl_orbit(T0, -0.2, 250.0, th_hat, orbL, 4e-3,
                                            *pts)
        b = fc.convert_u0_amp_t0_bsbl_orbit(T0, -0.2, 250.0, th_hat, orbL, None,
                                            4e-3, lens_in=pts[0], lens_out=pts[1])
        np.testing.assert_array_equal(a, b)
    return None
