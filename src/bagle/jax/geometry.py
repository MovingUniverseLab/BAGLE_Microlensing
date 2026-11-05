"""Derive microlensing geometry from fitter parameter vectors (JAX)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from astropy import constants as _const
from astropy import units as _u

from bagle.jax.orbits import oal2xy

from bagle.jax_physics import (
    _DAYS_PER_YEAR,
    _EINSTEIN_M_PER_MSUN,
    _PC_M,
    _PI_MAS_PER_PC,
    _RAD_TO_MAS,
    derive_pspl_photastrom_param1_geometry,
    derive_pspl_static_geometry,
    derive_psbl_static_geometry,
    u0_hat_from_thetaE_hat_jax,
)

# kappa = 4 G / (c^2 AU) in mas / Msun, so thetaE^2 = kappa * mL * piRel.
_KAPPA_MAS_MSUN = float(
    (4.0 * _const.G / (_const.c ** 2 * _u.AU)).to(
        _u.mas / _u.Msun, equivalencies=_u.dimensionless_angles()).value
)

# Period (days) of a 1 AU orbit around 1 Msun: P = this * sqrt(a_AU^3 / M).
_KEPLER_P_DAYS = float(
    (2.0 * np.pi * np.sqrt(_u.AU ** 3 / (_const.G * _u.Msun))).to(_u.day).value
)


def mag_src_from_base(mag_base, b_sff):
    return mag_base - 2.5 * jnp.log10(b_sff)


def mag_src_from_fitter(mag_value, mag_fitter: str, b_sff):
    if mag_fitter == "mag_base":
        return mag_src_from_base(mag_value, b_sff)
    return mag_value


def derive_pspl_photastrom_reduced(
    t0,
    u0_amp,
    tE,
    thetaE_amp,
    piS,
    piE_E,
    piE_N,
    xS0_E,
    xS0_N,
    muS_E,
    muS_N,
):
    """Geometry for PSPL PhotAstrom Param2 / Astrom Param4 style parameters."""
    piE = jnp.stack([piE_E, piE_N])
    piE_amp = jnp.linalg.norm(piE)
    piRel = piE_amp * thetaE_amp
    muRel_amp = thetaE_amp / (tE / _DAYS_PER_YEAR)
    mL = thetaE_amp**2 / (piRel * _KAPPA_MAS_MSUN)
    piL = piRel + piS
    dL = _PI_MAS_PER_PC / piL
    dS = _PI_MAS_PER_PC / piS
    thetaE_hat = piE / piE_amp
    muRel = muRel_amp * thetaE_hat
    muS = jnp.stack([muS_E, muS_N])
    muL = muS - muRel
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, u0_amp)
    u0 = jnp.abs(u0_amp) * u0_hat
    xS0 = jnp.stack([xS0_E, xS0_N])
    thetaS0 = u0 * thetaE_amp
    xL0 = xS0 - thetaS0 * 1e-3
    return (
        u0,
        thetaE_hat,
        tE,
        piE_E,
        piE_N,
        xS0,
        xL0,
        muS,
        muL,
        thetaE_amp,
        piS,
        piL,
    )


def derive_pspl_photastrom_log10_thetaE(
    t0,
    u0_amp,
    tE,
    log10_thetaE,
    piS,
    piE_E,
    piE_N,
    xS0_E,
    xS0_N,
    muS_E,
    muS_N,
):
    thetaE_amp = 10.0**log10_thetaE
    return derive_pspl_photastrom_reduced(
        t0,
        u0_amp,
        tE,
        thetaE_amp,
        piS,
        piE_E,
        piE_N,
        xS0_E,
        xS0_N,
        muS_E,
        muS_N,
    )


def derive_pspl_phot_log(
    t0,
    u0_amp,
    log_tE,
    log_piE,
    phi_muRel_deg,
):
    """PSPL_PhotParam3: log tE, log piE, phi_muRel."""
    tE = 10.0**log_tE
    piE_amp = 10.0**log_piE
    phi = phi_muRel_deg * jnp.pi / 180.0
    piE_E = piE_amp * jnp.sin(phi)
    piE_N = piE_amp * jnp.cos(phi)
    u0, thetaE_hat, _ = derive_pspl_static_geometry(u0_amp, piE_E, piE_N)
    return u0, thetaE_hat, tE, piE_E, piE_N


def derive_psbl_param4_heliocentric(
    t0_com,
    u0_amp_com,
    tE,
    thetaE,
    piE_E,
    piE_N,
    q,
    sep,
    alpha_deg,
):
    """Heliocentric t0, u0_amp from COM-frame PSBL PhotAstrom Param4 inputs."""
    alpha_rad = alpha_deg * jnp.pi / 180.0
    phi_rad = alpha_rad - jnp.arctan2(piE_E, piE_N)
    qeff = (1.0 - q) / (1.0 + q)
    t0 = t0_com - 0.5 * qeff * tE * sep * jnp.cos(phi_rad) / thetaE
    u0_amp = u0_amp_com - 0.5 * qeff * sep * jnp.sin(phi_rad) / thetaE
    return t0, u0_amp


def derive_psbl_photastrom_param1(
    mLp,
    mLs,
    t0,
    xS0_E,
    xS0_N,
    beta,
    muL_E,
    muL_N,
    muS_E,
    muS_N,
    dL,
    dS,
    sep,
    alpha_deg,
):
    """
    Static PSBL PhotAstrom Param1 geometry (mas, arcsec).

    Notes
    -----
    ``sep`` is the projected binary separation in mas. Companion offsets
    ``xL*_over_theta`` are in Einstein radii using the absolute on-sky
    binary angle ``alpha`` (degrees East of North), matching host
    ``get_resolved_lens_astrometry``. Masses ``m1``/``m2`` are fractions;
    convert to arcsec^2 via ``m_i * (thetaE_amp * 1e-3)**2`` for absolute
    Witt solutions.
    """
    mL = mLp + mLs
    xS0 = jnp.stack([xS0_E, xS0_N])
    muL = jnp.stack([muL_E, muL_N])
    muS = jnp.stack([muS_E, muS_N])
    inv_dist_diff = 1.0 / dL - 1.0 / dS
    piRel = _PI_MAS_PER_PC * inv_dist_diff
    piS = _PI_MAS_PER_PC / dS
    piL = _PI_MAS_PER_PC / dL
    muRel = muS - muL
    muRel_amp = jnp.linalg.norm(muRel)
    thetaE_amp = jnp.sqrt(_EINSTEIN_M_PER_MSUN * mL * inv_dist_diff / _PC_M) * _RAD_TO_MAS
    thetaE_hat = muRel / muRel_amp
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, beta)
    u0_amp = beta / thetaE_amp
    u0 = jnp.abs(u0_amp) * u0_hat
    piE_amp = piRel / thetaE_amp
    piE_E = piE_amp * thetaE_hat[0]
    piE_N = piE_amp * thetaE_hat[1]
    tE = (thetaE_amp / muRel_amp) * _DAYS_PER_YEAR
    thetaS0 = u0 * thetaE_amp
    xL0 = xS0 - thetaS0 * 1e-3

    # PhotAstrom Param1: sep is in mas; alpha is East of North (absolute).
    # Companion offsets in Einstein radii match host get_resolved_lens_astrometry.
    alpha_rad = alpha_deg * jnp.pi / 180.0
    half_sep_ein = 0.5 * sep / thetaE_amp
    xL1_over_theta = jnp.stack(
        [half_sep_ein * jnp.sin(alpha_rad),
         half_sep_ein * jnp.cos(alpha_rad)]
    )
    xL2_over_theta = -xL1_over_theta

    # Mass fractions for Einstein-unit photometry; convert to arcsec^2 for
    # absolute-sky Witt solutions via m_i * (thetaE_amp * 1e-3)**2.
    q = mLs / mLp
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    return (
        u0,
        thetaE_hat,
        tE,
        piE_E,
        piE_N,
        xS0,
        xL0,
        muS,
        muL,
        thetaE_amp,
        piS,
        piL,
        m1,
        m2,
        xL1_over_theta,
        xL2_over_theta,
        mLp,
        mLs,
    )


def derive_pstl_photastrom_param1(
    mLp, mLs, mLt, t0, xS0_E, xS0_N, beta, muL_E, muL_N, muS_E, muS_N,
    dL, dS, sep12, sep13, alpha12_deg, alpha13_deg,
):
    """
    Static PSTL PhotAstrom Param1 geometry.

    ``thetaE`` follows the PSBL parent: it is computed from
    ``mLp + mLs`` only. ``m3`` is ``mLt / (mLp + mLs)`` in those units.
    ``sep12`` / ``sep13`` are mas; ``alpha12`` / ``alpha13`` are East of
    North.

    Parameters
    ----------
    mLp, mLs, mLt : float
        Lens masses (Msun).
    t0 : float
        Closest-approach time (MJD).
    xS0_E, xS0_N : float
        Source position at ``t0`` (arcsec).
    beta : float
        Impact parameter (mas).
    muL_E, muL_N, muS_E, muS_N : float
        Proper motions (mas/yr).
    dL, dS : float
        Distances (pc).
    sep12, sep13 : float
        1-2 and 1-3 projected separations (mas).
    alpha12_deg, alpha13_deg : float
        Position angles East of North (deg).

    Returns
    -------
    geometry
        PSBL Param1 fields plus ``m3`` and ``xL3_over_theta``.
    """
    geom = derive_psbl_photastrom_param1(
        mLp, mLs, t0, xS0_E, xS0_N, beta, muL_E, muL_N, muS_E, muS_N,
        dL, dS, sep12, alpha12_deg)
    (u0, thetaE_hat, tE, piE_E, piE_N, xS0, xL0, muS, muL, thetaE_amp,
     piS, piL, m1, m2, xL1, xL2, mLp_o, mLs_o) = geom

    # Tertiary mass in units of thetaE(mLp+mLs)^2; lens 3 offset from lens 1.
    m3 = mLt / (mLp + mLs)
    a13 = alpha13_deg * jnp.pi / 180.0
    off13 = (sep13 / thetaE_amp) * jnp.stack(
        [jnp.sin(a13), jnp.cos(a13)])
    xL3 = xL1 + off13

    return (
        u0, thetaE_hat, tE, piE_E, piE_N, xS0, xL0, muS, muL,
        thetaE_amp, piS, piL, m1, m2, m3, xL1, xL2, xL3, mLp_o, mLs_o, mLt,
    )


def pstl_keplerian_lens_offsets(t, mLp, mLs, mLt, dL, omega_pri, omega_sec,
                                big_omega_sec, big_omega_ter, i_12, e_12,
                                tp_12, a_12, i_23, e_23, tp_23, a_23):
    """
    Keplerian PSTL lens offsets from the system CoM (arcsec).

    Matches the ``orbitFlag`` branch of
    :meth:`bagle.model.PSTL_PhotAstrom.get_resolved_lens_astrometry`:
    orbit 1-2 places the primary and secondary; orbit 2-3 places the
    tertiary at its "secondary" position.

    Parameters
    ----------
    t : array_like, shape (N_times,)
        Times (MJD).
    mLp, mLs, mLt : float
        Lens masses (Msun).
    dL : float
        Lens distance (pc).
    omega_pri, omega_sec : float
        Arguments of periastron for orbits 1-2 and 2-3 (deg).
    big_omega_sec, big_omega_ter : float
        Ascending-node longitudes for orbits 1-2 and 2-3 (deg).
    i_12, e_12, tp_12, a_12 : float
        Inclination (deg), eccentricity, periastron time (MJD), and
        semi-major axis (mas) of orbit 1-2.
    i_23, e_23, tp_23, a_23 : float
        Same for orbit 2-3.

    Returns
    -------
    off1, off2, off3 : jnp.ndarray, shape (N_times, 2)
        Primary, secondary, and tertiary offsets from the CoM (arcsec).
    alpha12_rad : jnp.ndarray, shape ()
        Position angle of the 1-2 axis at ``tp_12`` (rad, East of North).
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)

    # Semi-major axes of each body about its pair barycenter (mas).
    aleph_sec_12 = (mLp / (mLp + mLs)) * a_12
    aleph_12 = a_12 - aleph_sec_12
    aleph_sec_23 = (mLs / (mLs + mLt)) * a_23
    aleph_23 = a_23 - aleph_sec_23

    # Periods from Kepler's third law (days).
    p_12 = _KEPLER_P_DAYS * jnp.sqrt((dL * a_12 * 1e-3) ** 3 / (mLp + mLs))
    p_23 = _KEPLER_P_DAYS * jnp.sqrt((dL * a_23 * 1e-3) ** 3 / (mLs + mLt))

    # Orbit 1-2: primary and secondary (aleph in arcsec).
    x1, y1, x2, y2 = oal2xy(
        t, omega_pri, big_omega_sec, i_12, e_12, p_12, tp_12,
        aleph_12 * 1e-3, aleph_sec_12 * 1e-3)

    # Orbit 2-3: the tertiary is the "secondary" of this orbit.
    _, _, x3, y3 = oal2xy(
        t, omega_sec, big_omega_ter, i_23, e_23, p_23, tp_23,
        aleph_23 * 1e-3, aleph_sec_23 * 1e-3)

    off1 = jnp.stack([x1, y1], axis=-1)
    off2 = jnp.stack([x2, y2], axis=-1)
    off3 = jnp.stack([x3, y3], axis=-1)

    # Snapshot 1-2 angle at periastron (host Param ``alpha``).
    xa, ya, xb, yb = oal2xy(
        jnp.atleast_1d(tp_12), omega_pri, big_omega_sec, i_12, e_12, p_12,
        tp_12, aleph_12 * 1e-3, aleph_sec_12 * 1e-3)
    alpha12_rad = jnp.arctan2(xa - xb, ya - yb)[0]

    return off1, off2, off3, alpha12_rad


def derive_pstl_photastrom_ellorbs_param1(
    t, mLp, mLs, mLt, t0_com, xS0_E, xS0_N, beta_com, muL_E, muL_N,
    omega_pri, omega_sec, big_omega_sec, big_omega_ter,
    i_12, e_12, tp_12, a_12, i_23, e_23, tp_23, a_23,
    muS_E, muS_N, dL, dS,
):
    """
    Keplerian PSTL PhotAstrom EllOrbs Param1 sky tracks (traced).

    ``thetaE`` is from ``mLp + mLs`` (PSBL parent). The source track uses
    the geometric-midpoint ``t0`` from ``fc.convert_u0_t0_psbl`` (COM ->
    geom_mid), as the host does; the lens CoM moves from ``t0_com``.

    Parameters
    ----------
    t : array_like, shape (N_times,)
        Times (MJD).
    mLp, mLs, mLt : float
        Lens masses (Msun).
    t0_com : float
        Closest approach to the 1-2 CoM (MJD).
    xS0_E, xS0_N : float
        Source position (arcsec).
    beta_com : float
        Impact parameter about the 1-2 CoM (mas).
    muL_E, muL_N, muS_E, muS_N : float
        Proper motions (mas/yr).
    omega_pri, omega_sec, big_omega_sec, big_omega_ter : float
        Orbit angles (deg).
    i_12, e_12, tp_12, a_12, i_23, e_23, tp_23, a_23 : float
        Orbit elements (deg, -, MJD, mas).
    dL, dS : float
        Distances (pc).

    Returns
    -------
    t0 : jnp.ndarray
        Reference time shared by the source and lens tracks (MJD).
    xS0, xL0 : jnp.ndarray, shape (2,)
        Source and lens-CoM positions at ``t0`` (arcsec).
    muS, muL : jnp.ndarray, shape (2,)
        Proper motions (mas/yr).
    thetaE_amp : jnp.ndarray
        Einstein radius of ``mLp + mLs`` (mas).
    piS, piL : jnp.ndarray
        Parallaxes (mas).
    m1, m2, m3 : jnp.ndarray
        Masses in units of ``thetaE^2``.
    off1, off2, off3 : jnp.ndarray, shape (N_times, 2)
        Lens offsets from the CoM in Einstein radii.
    """
    # Static PSBL Param1 quantities about the 1-2 CoM (sep/alpha unused).
    geom = derive_psbl_photastrom_param1(
        mLp, mLs, t0_com, xS0_E, xS0_N, beta_com, muL_E, muL_N,
        muS_E, muS_N, dL, dS, a_12, 0.0)
    (_u0, _th, tE, piE_E, piE_N, xS0, xL0_com, muS, muL, thetaE_amp,
     piS, piL, m1, m2, _x1, _x2, _mLp, _mLs) = geom
    m3 = mLt / (mLp + mLs)

    # Lens offsets about the CoM (arcsec) and the snapshot 1-2 angle.
    off1, off2, off3, alpha12_rad = pstl_keplerian_lens_offsets(
        t, mLp, mLs, mLt, dL, omega_pri, omega_sec, big_omega_sec,
        big_omega_ter, i_12, e_12, tp_12, a_12, i_23, e_23, tp_23, a_23)

    # COM -> geometric-midpoint t0 (fc.convert_u0_t0_psbl).
    q = mLs / mLp
    phi = alpha12_rad - jnp.arctan2(piE_E, piE_N)
    d = a_12 * (1.0 - q) / (2.0 * (1.0 + q)) / thetaE_amp
    sign = jnp.where(q > 1.0, -1.0, 1.0)
    t0 = t0_com + sign * tE * d * jnp.cos(phi)

    # Re-anchor the lens CoM to t0 so one t0 drives both tracks.
    xL0 = xL0_com + muL * ((t0 - t0_com) / _DAYS_PER_YEAR) * 1e-3

    # Offsets in Einstein radii for pstl_sky_images.
    thetaE_as = thetaE_amp * 1e-3
    return (t0, xS0, xL0, muS, muL, thetaE_amp, piS, piL, m1, m2, m3,
            off1 / thetaE_as, off2 / thetaE_as, off3 / thetaE_as)


def derive_pstl_photastrom_ellorbs_param2(
    t, t0, u0_amp, tE, thetaE, piS, piE_E, piE_N, xS0_E, xS0_N,
    muS_E, muS_N, q_12, q_13, omega_pri, omega_sec, big_omega_sec,
    big_omega_ter, i_12, e_12, tp_12, a_12, i_23, e_23, tp_23, a_23,
):
    """
    Keplerian PSTL PhotAstrom EllOrbs Param2 sky tracks (traced).

    The host sets ``t0_com = t0`` and ``xL0_com = xL0`` for this
    parameterization, so the lens CoM moves from the input ``t0``.

    Parameters
    ----------
    t : array_like, shape (N_times,)
        Times (MJD).
    t0, u0_amp, tE, thetaE, piS, piE_E, piE_N : float
        Reduced PSPL parameters (MJD, thetaE, days, mas, mas, -, -).
    xS0_E, xS0_N : float
        Source position at ``t0`` (arcsec).
    muS_E, muS_N : float
        Source proper motion (mas/yr).
    q_12, q_13 : float
        Mass ratios M2/M1 and M3/M1.
    omega_pri, omega_sec, big_omega_sec, big_omega_ter : float
        Orbit angles (deg).
    i_12, e_12, tp_12, a_12, i_23, e_23, tp_23, a_23 : float
        Orbit elements (deg, -, MJD, mas).

    Returns
    -------
    tuple
        Same layout as :func:`derive_pstl_photastrom_ellorbs_param1`.
    """
    (_u0, _th, tE, _pE, _pN, xS0, xL0, muS, muL, thetaE_amp,
     piS, piL) = derive_pspl_photastrom_reduced(
        t0, u0_amp, tE, thetaE, piS, piE_E, piE_N, xS0_E, xS0_N,
        muS_E, muS_N)

    # Lens masses from thetaE and piRel; the pair 1-2 sets thetaE.
    piRel = piL - piS
    mL = thetaE_amp ** 2 / (piRel * _KAPPA_MAS_MSUN)
    mLp = mL / (1.0 + q_12)
    mLs = mLp * q_12
    mLt = mLp * q_13
    dL = _PI_MAS_PER_PC / piL

    # Masses in units of thetaE(mLp+mLs)^2.
    m1 = 1.0 / (1.0 + q_12)
    m2 = q_12 / (1.0 + q_12)
    m3 = q_13 / (1.0 + q_12)

    off1, off2, off3, _alpha = pstl_keplerian_lens_offsets(
        t, mLp, mLs, mLt, dL, omega_pri, omega_sec, big_omega_sec,
        big_omega_ter, i_12, e_12, tp_12, a_12, i_23, e_23, tp_23, a_23)

    thetaE_as = thetaE_amp * 1e-3
    return (t0, xS0, xL0, muS, muL, thetaE_amp, piS, piL, m1, m2, m3,
            off1 / thetaE_as, off2 / thetaE_as, off3 / thetaE_as)


def unpack_base_params(names: tuple[str, ...], vec):
    vec = jnp.asarray(vec, dtype=jnp.float64).reshape(-1)
    return {n: vec[i] for i, n in enumerate(names)}


def derive_geometry_from_layout(layout_id: str, eval_kind: str, base_vec, names: tuple[str, ...]):
    """Dispatch geometry derivation from a base fitter parameter vector.

    Parameters
    ----------
    layout_id : str
        Unused layout identifier (kept for call-site compatibility).
    eval_kind : str
        Geometry family key (e.g. ``pspl_photastrom_physical``).
    base_vec : array_like
        Packed base-fitter parameter vector.
    names : tuple of str
        Parameter names parallel to ``base_vec``.

    Returns
    -------
    tuple
        Derived geometry fields for the requested ``eval_kind``.
    """
    p = unpack_base_params(names, base_vec)

    # BSPL CircOrbs Param1 samples t0_com but physical geometry uses t0.
    if "t0" not in p and "t0_com" in p:
        p["t0"] = p["t0_com"]
    if "t0" not in p and "t0_p" in p:
        p["t0"] = p["t0_p"]
    if "t0" not in p and "t0_prim" in p:
        p["t0"] = p["t0_prim"]

    if eval_kind == "pspl_photastrom_physical":
        return derive_pspl_photastrom_param1_geometry(
            p["mL"],
            p["t0"],
            p["beta"],
            p["dL"],
            p["dL_dS"],
            p["xS0_E"],
            p["xS0_N"],
            p["muL_E"],
            p["muL_N"],
            p["muS_E"],
            p["muS_N"],
        )
    if eval_kind in ("pspl_photastrom_reduced", "pspl_astrom_reduced"):
        te_key = "log10_thetaE" if "log10_thetaE" in p else "thetaE"
        theta = 10.0 ** p[te_key] if te_key == "log10_thetaE" else p[te_key]
        if "log_piE" in p and "phi_muRel" in p:
            piE_amp = 10.0 ** p["log_piE"]
            phi = p["phi_muRel"] * jnp.pi / 180.0
            piE_E = piE_amp * jnp.sin(phi)
            piE_N = piE_amp * jnp.cos(phi)
        elif "piEN_piEE" in p:
            piE_E = p["piE_E"]
            piE_N = p["piE_E"] * p["piEN_piEE"]
        else:
            piE_E = p["piE_E"]
            piE_N = p["piE_N"]
        return derive_pspl_photastrom_reduced(
            p["t0"],
            p["u0_amp"],
            p["tE"],
            theta,
            p["piS"],
            piE_E,
            piE_N,
            p["xS0_E"],
            p["xS0_N"],
            p["muS_E"],
            p["muS_N"],
        )
    if eval_kind in ("pspl_phot_static", "pspl_phot_log"):
        if "log_tE" in p:
            u0, thetaE_hat, tE, piE_E, piE_N = derive_pspl_phot_log(
                p["t0"],
                p["u0_amp"],
                p["log_tE"],
                p["log_piE"],
                p["phi_muRel"],
            )
        else:
            u0, thetaE_hat, _ = derive_pspl_static_geometry(
                p["u0_amp"], p["piE_E"], p["piE_N"]
            )
            tE = p["tE"]
            piE_E = p["piE_E"]
            piE_N = p["piE_N"]
        return ("pspl_phot", u0, thetaE_hat, tE, piE_E, piE_N)
    if eval_kind.startswith("psbl_phot"):
        phi_key = "phi" if "phi" in p else "alpha"
        u0_amp = p["u0_amp"]
        t0 = p["t0"]
        if "t0_com" in p:
            t0, u0_amp = derive_psbl_param4_heliocentric(
                p["t0_com"],
                p["u0_amp_com"],
                p["tE"],
                p["thetaE"],
                p["piE_E"],
                p["piE_N"],
                p["q"],
                p["sep"],
                p[phi_key],
            )
        m1, m2, u0, thetaE_hat, xL1, xL2, _ = derive_psbl_static_geometry(
            u0_amp,
            p["piE_E"],
            p["piE_N"],
            p["q"],
            p["sep"],
            p[phi_key],
        )
        return (
            "psbl_phot",
            u0,
            thetaE_hat,
            t0,
            p["tE"],
            m1,
            m2,
            xL1,
            xL2,
            p["piE_E"],
            p["piE_N"],
        )
    if eval_kind == "bspl_phot":
        piE = jnp.stack([p["piE_E"], p["piE_N"]])
        piE_amp = jnp.linalg.norm(piE)
        thetaE_hat = piE / piE_amp
        u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, p["u0_amp"])
        u0_pri = jnp.abs(p["u0_amp"]) * u0_hat
        phi_rad = p["phi"] * jnp.pi / 180.0
        phi_piE = jnp.arctan2(p["piE_E"], p["piE_N"])
        phi_rho1 = phi_piE + phi_rad
        sep_vec = p["sep"] * jnp.stack([jnp.sin(phi_rho1), jnp.cos(phi_rho1)])
        u0_amp_sec = p["u0_amp"] + jnp.dot(sep_vec, u0_hat)
        u0_sec = u0_amp_sec * u0_hat
        return (
            "bspl_phot",
            u0_pri,
            u0_sec,
            thetaE_hat,
            p["t0"],
            p["t0"],
            p["tE"],
            p["piE_E"],
            p["piE_N"],
        )
    if eval_kind.startswith("psbl_photastrom"):
        return derive_psbl_photastrom_param1(
            p["mLp"],
            p["mLs"],
            p["t0"],
            p["xS0_E"],
            p["xS0_N"],
            p["beta"],
            p["muL_E"],
            p["muL_N"],
            p["muS_E"],
            p["muS_N"],
            p["dL"],
            p["dS"],
            p["sep"],
            p["alpha"],
        )
    # FSPL and FSBL finite-source layouts reuse the single-lens geometry.
    if eval_kind in ("fsbl_photastrom", "fspl_photastrom"):
        if "mL" in p:
            return derive_pspl_photastrom_param1_geometry(
                p["mL"],
                p["t0"],
                p["beta"],
                p["dL"],
                p["dL_dS"],
                p["xS0_E"],
                p["xS0_N"],
                p["muL_E"],
                p["muL_N"],
                p["muS_E"],
                p["muS_N"],
            )
        te_key = "log10_thetaE" if "log10_thetaE" in p else "thetaE"
        theta = 10.0 ** p[te_key] if te_key == "log10_thetaE" else p[te_key]
        if "log_piE" in p and "phi_muRel" in p:
            piE_amp = 10.0 ** p["log_piE"]
            phi = p["phi_muRel"] * jnp.pi / 180.0
            piE_E = piE_amp * jnp.sin(phi)
            piE_N = piE_amp * jnp.cos(phi)
        elif "piEN_piEE" in p:
            piE_E = p["piE_E"]
            piE_N = p["piE_E"] * p["piEN_piEE"]
        else:
            piE_E = p["piE_E"]
            piE_N = p["piE_N"]
        return derive_pspl_photastrom_reduced(
            p["t0"],
            p["u0_amp"],
            p["tE"],
            theta,
            p["piS"],
            piE_E,
            piE_N,
            p["xS0_E"],
            p["xS0_N"],
            p["muS_E"],
            p["muS_N"],
        )
    if eval_kind in ("fsbl_phot", "fspl_phot"):
        u0, thetaE_hat, _ = derive_pspl_static_geometry(
            p["u0_amp"], p["piE_E"], p["piE_N"]
        )
        return ("pspl_phot", u0, thetaE_hat, p["tE"], p["piE_E"], p["piE_N"])
    raise NotImplementedError(f"derive_geometry for eval_kind={eval_kind!r}")
