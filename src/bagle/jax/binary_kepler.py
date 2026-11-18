"""Traced likelihoods for PSBL and FSBL Keplerian (Circ / Ell) orbits.

PhotAstrom classes are evaluated in the lens center-of-mass frame: the
source and the lens CoM share ``t0_com``, and ``oal2xy`` supplies the
primary and secondary offsets. Photometry-only classes stay in Einstein
radii. The relative state at ``t0`` is turned into orbital elements, the
geometric midpoint is moved to the CoM, and the existing sky-track
kernels are called with ``thetaE = 1000`` mas so one arcsec equals one
Einstein radius.
"""
from __future__ import annotations

import jax.numpy as jnp

from bagle.jax.geometry import _KEPLER_P_DAYS, _KAPPA_MAS_MSUN
from bagle.jax.orbits import oal2xy
from bagle.jax_physics import (
    _DAYS_PER_YEAR,
    _EINSTEIN_M_PER_MSUN,
    _PC_M,
    _PI_MAS_PER_PC,
    _RAD_TO_MAS,
    psbl_log_likely_astrometry,
    psbl_log_likely_photometry_tracks,
    u0_hat_from_thetaE_hat_jax,
)

# Phot-only tracks are Einstein radii. thetaE = 1000 mas makes the
# arcsec conversion in the sky-track kernels a no-op (1 mas = 1e-3 arcsec).
_THETA_E_FAKE_MAS = 1000.0


def _elements_from_state(t0, r_enz, v_enz, a_rel):
    """
    Keplerian elements from a relative state, matching ``orbits.py``.

    Parameters
    ----------
    t0 : float
        Epoch of the state (MJD).
    r_enz, v_enz : array_like, shape (3,)
        Secondary minus primary, East / North / line of sight.
        ``v`` is in the length unit of ``r`` per year.
    a_rel : float
        Semi-major axis of the relative orbit, same length unit as ``r``.

    Returns
    -------
    e, i_deg, big_omega_deg, omega_pri_deg, p_days, tp : float
        Elements for :func:`bagle.jax.orbits.oal2xy`.
    """
    # Thiele-Innes order is North, East, line of sight.
    r = jnp.stack([r_enz[1], r_enz[0], r_enz[2]])
    v = jnp.stack([v_enz[1], v_enz[0], v_enz[2]])
    r_amp = jnp.linalg.norm(r)

    # Vis-viva gravitational parameter (length^3 / yr^2) and mean motion.
    gm = jnp.dot(v, v) / (2.0 / r_amp - 1.0 / a_rel)
    n_ang = jnp.sqrt(gm / a_rel ** 3)

    # Orbit normal -> inclination and node.
    h = jnp.cross(r, v)
    w_hat = h / jnp.linalg.norm(h)
    incl = jnp.arccos(jnp.clip(w_hat[2], -1.0, 1.0))
    big_omega = jnp.arctan2(w_hat[0], -w_hat[1])

    # Eccentricity vector. A circular orbit uses the position at t0.
    e_vec = jnp.cross(v, h) / gm - r / r_amp
    e_amp = jnp.linalg.norm(e_vec)
    circular = e_amp <= 1e-12
    e = jnp.where(circular, 0.0, e_amp)
    p_hat = jnp.where(circular, r / r_amp, e_vec / jnp.maximum(e_amp, 1e-30))

    # Argument of periastron of the relative orbit, then of the primary.
    node = jnp.stack([jnp.cos(big_omega), jnp.sin(big_omega), 0.0])
    omega_rel = jnp.arctan2(
        jnp.dot(p_hat, jnp.cross(w_hat, node)), jnp.dot(p_hat, node))
    q_hat = jnp.cross(w_hat, p_hat)
    nu = jnp.arctan2(jnp.dot(r, q_hat), jnp.dot(r, p_hat))
    ecc_anom = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 - e) * jnp.sin(0.5 * nu),
        jnp.sqrt(1.0 + e) * jnp.cos(0.5 * nu))
    mean_anom = ecc_anom - e * jnp.sin(ecc_anom)
    tp = t0 - mean_anom / n_ang * _DAYS_PER_YEAR
    p_days = 2.0 * jnp.pi / n_ang * _DAYS_PER_YEAR
    omega_pri = jnp.mod(jnp.rad2deg(omega_rel) - 180.0, 360.0)
    big_omega_deg = jnp.mod(jnp.rad2deg(big_omega), 360.0)

    return e, jnp.rad2deg(incl), big_omega_deg, omega_pri, p_days, tp


def _com_shift(t_in, u0_amp, thetaE_hat, tE, d_ein):
    """
    Move a midpoint or primary reference onto the lens CoM.

    Parameters
    ----------
    t_in : float
        Input reference epoch (MJD).
    u0_amp : float
        Signed impact parameter about the input point (Einstein radii).
    thetaE_hat : array_like, shape (2,)
        Unit relative proper motion (East, North).
    tE : float
        Einstein crossing time (days).
    d_ein : array_like, shape (2,)
        Offset of the input point from the CoM at ``t_in`` (Einstein radii).
        Zero when the input is already the CoM.

    Returns
    -------
    t0_com : float
        Closest-approach time of the source to the lens CoM (MJD).
    u0_com : array_like, shape (2,)
        Impact-parameter vector about the CoM (Einstein radii).
    """
    # Host convert_u0_amp_t0_psbl_orbit, input -> COM (explicit, no root).
    u_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, 1.0)
    u_in = u0_amp * u_hat
    u_com = u_in + d_ein
    along = jnp.dot(u_com, thetaE_hat)
    t0_com = t_in - tE * along
    u0_com = u_com - along * thetaE_hat

    return t0_com, u0_com


def _point_offset(t_in, omega, big_omega, incl, ecc, period, tp,
                  aleph, aleph_sec, which):
    """
    Offset of the midpoint or the primary from the CoM (``aleph`` units).

    Parameters
    ----------
    t_in : float
        Epoch (MJD).
    omega, big_omega, incl, ecc, period, tp, aleph, aleph_sec : float
        :func:`bagle.jax.orbits.oal2xy` elements. ``aleph`` units are the
        caller's (arcsec or Einstein radii).
    which : {'com', 'prim', 'mid'}
        Reference point.

    Returns
    -------
    offset : jnp.ndarray, shape (2,)
        East / North offset from the CoM.
    """
    if which == 'com':
        return jnp.zeros(2)

    # One epoch. oal2xy returns primary (x, y) and secondary (x2, y2).
    x, y, x2, y2 = oal2xy(
        jnp.atleast_1d(t_in), omega, big_omega, incl, ecc, period, tp,
        aleph, aleph_sec)
    primary = jnp.stack([x[0], y[0]])
    if which == 'prim':
        return primary

    return 0.5 * (primary + jnp.stack([x2[0], y2[0]]))


def _mass_distance_geometry(p):
    """
    Einstein radius and motions from lens masses and distances.

    Parameters
    ----------
    p : dict
        Fitter parameters. Requires ``mLp``, ``mLs``, ``dL``, ``dS``,
        ``muL_*`` and ``muS_*``.

    Returns
    -------
    thetaE_amp, tE, thetaE_hat, muS, muL, piS, piL, m1, m2, dL, mL
        Geometry used by the CoM sky tracks. ``m1`` and ``m2`` are mass
        fractions. ``thetaE_amp`` is mas, parallaxes are mas.
    """
    m_lp = p['mLp']
    m_ls = p['mLs']
    m_l = m_lp + m_ls
    d_l = p['dL']
    d_s = p['dS']
    inv = 1.0 / d_l - 1.0 / d_s

    # Same Einstein-radius expression as derive_psbl_photastrom_param1.
    theta = jnp.sqrt(
        _EINSTEIN_M_PER_MSUN * m_l * inv / _PC_M) * _RAD_TO_MAS
    pi_s = _PI_MAS_PER_PC / d_s
    pi_l = _PI_MAS_PER_PC / d_l
    mu_l = jnp.stack([p['muL_E'], p['muL_N']])
    mu_s = jnp.stack([p['muS_E'], p['muS_N']])
    mu_rel = mu_s - mu_l
    mu_amp = jnp.linalg.norm(mu_rel)
    theta_hat = mu_rel / mu_amp
    t_e = (theta / mu_amp) * _DAYS_PER_YEAR
    q = m_ls / m_lp
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)

    return theta, t_e, theta_hat, mu_s, mu_l, pi_s, pi_l, m1, m2, d_l, m_l


def _reduced_geometry(p):
    """
    Einstein radius and motions from ``tE``, ``thetaE`` and ``piE``.

    Parameters
    ----------
    p : dict
        Fitter parameters. ``thetaE`` or ``log10_thetaE`` is the Einstein
        radius in mas. ``q`` is the mass ratio.

    Returns
    -------
    Same as :func:`_mass_distance_geometry`.
    """
    if 'log10_thetaE' in p:
        theta = 10.0 ** p['log10_thetaE']
    else:
        theta = p['thetaE']

    pi_e = jnp.stack([p['piE_E'], p['piE_N']])
    pi_e_amp = jnp.linalg.norm(pi_e)
    theta_hat = pi_e / pi_e_amp
    t_e = p['tE']
    pi_s = p['piS']
    pi_rel = pi_e_amp * theta
    pi_l = pi_rel + pi_s
    mu_amp = theta / (t_e / _DAYS_PER_YEAR)
    mu_rel = mu_amp * theta_hat
    mu_s = jnp.stack([p['muS_E'], p['muS_N']])
    mu_l = mu_s - mu_rel
    q = p['q']
    m_l = theta ** 2 / (pi_rel * _KAPPA_MAS_MSUN)
    m_lp = m_l / (1.0 + q)
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    d_l = _PI_MAS_PER_PC / pi_l

    return (theta, t_e, theta_hat, mu_s, mu_l, pi_s, pi_l, m1, m2,
            d_l, m_l)


def _input_frame(p, theta):
    """
    Input reference time, signed u0 and source position.

    Parameters
    ----------
    p : dict
        Fitter parameters.
    theta : float
        Einstein radius (mas). ``beta`` inputs are mas; ``u0_amp`` inputs
        are already Einstein radii.

    Returns
    -------
    t_in, u0_amp, xS0, which : float, float, shape (2,), str
        ``which`` is ``'com'``, ``'prim'`` or ``'mid'``.
    """
    if 't0_com' in p:
        t_in = p['t0_com']
        if 'beta_com' in p:
            u0_amp = p['beta_com'] / theta
        else:
            u0_amp = p['u0_amp_com']
        x_s0 = jnp.stack([p['xS0_com_E'], p['xS0_com_N']])
        which = 'com'
    elif 't0_p' in p:
        t_in = p['t0_p']
        u0_amp = p['beta_p'] / theta
        x_s0 = jnp.stack([p['xS0_p_E'], p['xS0_p_N']])
        which = 'prim'
    else:
        t_in = p['t0']
        u0_amp = p['u0_amp']
        x_s0 = jnp.stack([p['xS0_E'], p['xS0_N']])
        which = 'mid'

    return t_in, u0_amp, x_s0, which


def photastrom_tracks(p, t):
    """
    CoM-frame sky tracks for a Keplerian binary lens.

    Parameters
    ----------
    p : dict
        Filter-local fitter parameters (PhotAstrom Circ or Ell).
    t : array_like, shape (N_times,)
        Observation times (MJD).

    Returns
    -------
    args : tuple
        ``(t0, xS0, xL0, muS, muL, thetaE_amp, off1, off2, m1, m2,
        piS, piL, rho)``. Offsets are Einstein radii, shape
        ``(N_times, 2)``. ``rho`` is the source radius in Einstein radii,
        or ``None`` for a point source.
    """
    if 'mLp' in p:
        (theta, t_e, theta_hat, mu_s, mu_l, pi_s, pi_l, m1, m2,
         d_l, m_l) = _mass_distance_geometry(p)
    else:
        (theta, t_e, theta_hat, mu_s, mu_l, pi_s, pi_l, m1, m2,
         d_l, m_l) = _reduced_geometry(p)

    # Circular classes omit e. a is the relative semi-major axis (mas).
    ecc = p['e'] if 'e' in p else 0.0
    a_mas = p['a']
    m_lp = m_l / (1.0 + (m2 / m1))
    aleph_sec = (m_lp / m_l) * a_mas
    aleph = a_mas - aleph_sec
    period = _KEPLER_P_DAYS * jnp.sqrt((d_l * a_mas * 1e-3) ** 3 / m_l)

    # Orbit offsets are arcsec; the sky kernels want Einstein radii.
    theta_as = theta * 1e-3
    aleph_as = aleph * 1e-3
    aleph_sec_as = aleph_sec * 1e-3
    t_in, u0_amp, x_s0, which = _input_frame(p, theta)
    d_as = _point_offset(
        t_in, p['omega_pri'], p['big_omega_sec'], p['i'], ecc, period,
        p['tp'], aleph_as, aleph_sec_as, which)
    t0, u0_com = _com_shift(t_in, u0_amp, theta_hat, t_e, d_as / theta_as)

    # Source position at t0_com, then the lens CoM from the impact parameter.
    dt_yr = (t0 - t_in) / _DAYS_PER_YEAR
    x_s0 = x_s0 + mu_s * dt_yr * 1e-3
    x_l0 = x_s0 - u0_com * theta * 1e-3

    x, y, x2, y2 = oal2xy(
        t, p['omega_pri'], p['big_omega_sec'], p['i'], ecc, period,
        p['tp'], aleph_as, aleph_sec_as)
    off1 = jnp.stack([x, y], axis=1) / theta_as
    off2 = jnp.stack([x2, y2], axis=1) / theta_as

    # PhotAstrom radiusS is arcsec. Phot-only never reaches this function.
    rho = None
    if 'radiusS' in p:
        rho = p['radiusS'] * 1e3 / theta

    return (t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2, m1, m2,
            pi_s, pi_l, rho)


def _phot_only_state(p, t):
    """
    Einstein-radius tracks for a photometry-only Keplerian binary.

    Parameters
    ----------
    p : dict
        Filter-local fitter parameters. Circular classes omit ``r_s``
        and ``a_s`` (``r_s = -v_para / v_rad``, ``a_s = 1``).
    t : array_like, shape (N_times,)
        Observation times (MJD).

    Returns
    -------
    args : tuple
        Arguments for the sky-track kernels with ``thetaE = 1000`` mas,
        plus ``rho`` (Einstein radii, or ``None``).
    """
    pi_e = jnp.stack([p['piE_E'], p['piE_N']])
    pi_e_amp = jnp.linalg.norm(pi_e)
    theta_hat = pi_e / pi_e_amp
    t_e = p['tE']
    q = p['q']

    # Binary axis at t0: alpha = PA(piE) + phi (secondary -> primary).
    alpha = jnp.arctan2(p['piE_E'], p['piE_N']) + jnp.deg2rad(p['phi'])
    if 'r_s' in p:
        r_s = p['r_s']
        a_s = p['a_s']
    else:
        r_s = -p['v_para'] / p['v_rad']
        a_s = 1.0

    # Relative position and velocity of the secondary (Einstein radii).
    s_hat = -jnp.stack([jnp.sin(alpha), jnp.cos(alpha), 0.0])
    pa_hat = jnp.stack([-jnp.cos(alpha), jnp.sin(alpha), 0.0])
    z_hat = jnp.array([0.0, 0.0, 1.0])
    sep = p['sep']
    r_vec = sep * (s_hat + r_s * z_hat)
    v_vec = sep * (
        p['v_para'] * s_hat + p['v_perp'] * pa_hat + p['v_rad'] * z_hat)
    a_rel = a_s * jnp.linalg.norm(r_vec)
    ecc, incl, big_omega, omega, period, tp = _elements_from_state(
        p['t0'], r_vec, v_vec, a_rel)

    # Split the relative orbit about the CoM (Einstein radii).
    aleph_sec = a_rel / (1.0 + q)
    aleph = a_rel - aleph_sec
    d_mid = _point_offset(
        p['t0'], omega, big_omega, incl, ecc, period, tp, aleph,
        aleph_sec, 'mid')
    t0, u0_com = _com_shift(p['t0'], p['u0_amp'], theta_hat, t_e, d_mid)

    x, y, x2, y2 = oal2xy(
        t, omega, big_omega, incl, ecc, period, tp, aleph, aleph_sec)
    off1 = jnp.stack([x, y], axis=1)
    off2 = jnp.stack([x2, y2], axis=1)

    # thetaE = 1000 mas => positions in "arcsec" are Einstein radii.
    scale = _THETA_E_FAKE_MAS
    mu_s = theta_hat * (_DAYS_PER_YEAR / t_e) * scale
    mu_l = jnp.zeros(2)
    x_s0 = u0_com
    x_l0 = jnp.zeros(2)
    pi_s = -pi_e_amp * scale
    pi_l = jnp.asarray(0.0)
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    rho = p['radiusS'] if 'radiusS' in p else None

    return (t0, x_s0, x_l0, mu_s, mu_l, scale, off1, off2, m1, m2,
            pi_s, pi_l, rho)


def _gamma_of(p):
    """Limb-darkening coefficient, or ``None`` for a uniform source."""
    if 'gamma' in p:
        return p['gamma']

    return None


def _phot_only_fsbl_lnl(t, t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2,
                         m1, m2, rho, mag_src, b_sff, mag_obs, mag_err,
                         pi_s, pi_l, parallax_vectors=None, gamma=None,
                         n_outline=None, gp_params=None, fixed_jitter=True):
    """
    Phot-only FSBL Keplerian photometry, matching the host contour frame.

    Parameters
    ----------
    t, t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2, m1, m2 : see
        :func:`bagle.jax.fsbl.fsbl_sky_tracks`
        ``theta`` is 1000 mas, so the returned positions are Einstein radii.
    rho, mag_src, b_sff, mag_obs, mag_err : see
        :func:`kepler_log_likely_photometry`
    pi_s, pi_l, parallax_vectors, gamma, n_outline, gp_params, fixed_jitter :
        optional
        Parallax, limb darkening and the GP path.

    Returns
    -------
    lnL : float
        Summed photometric log-likelihood.
    """
    from bagle.jax.fsbl import fsbl_magnification_centroid, fsbl_sky_tracks
    from bagle.jax_physics import (
        gaussian_log_likelihood_sum,
        pspl_photometry_from_amp,
    )

    # Sky tracks with thetaE = 1000 mas: numbers are Einstein radii.
    x_s, x_l1, x_l2 = fsbl_sky_tracks(
        t, t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2,
        parallax_vectors=parallax_vectors, piS=pi_s, piL=pi_l)

    # Host phot-only: source stays in the CoM frame; lenses are recentered.
    origin = 0.5 * (x_l1 + x_l2)
    zeta = x_s[:, 0] + 1j * x_s[:, 1]
    z1 = (x_l1[:, 0] - origin[:, 0]) + 1j * (x_l1[:, 1] - origin[:, 1])
    z2 = (x_l2[:, 0] - origin[:, 0]) + 1j * (x_l2[:, 1] - origin[:, 1])
    kw = {} if n_outline is None else {'n_init': int(n_outline)}
    amp, _cent = fsbl_magnification_centroid(
        zeta, rho, z1, z2, m1, m2, gamma=gamma, **kw)
    mag_model = pspl_photometry_from_amp(amp, mag_src, b_sff)

    if gp_params is None:
        return gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)

    from bagle.jax.gp import gp_log_likely_photometry

    return gp_log_likely_photometry(
        t, mag_obs, mag_err, mag_model, gp_params, fixed_jitter=fixed_jitter)


def kepler_log_likely_photometry(p, t, mag_obs, mag_err, b_sff, mag_src,
                                 parallax_vectors=None, gp_params=None,
                                 fixed_jitter=True, n_outline=None):
    """
    Photometric log-likelihood for one Keplerian binary-lens parameterization.

    Parameters
    ----------
    p : dict
        Filter-local fitter parameters.
    t, mag_obs, mag_err : array_like, shape (N_times,)
        Times and photometry.
    b_sff, mag_src : float
        Filter-local blend and source magnitude.
    parallax_vectors : array_like, shape (N_times, 2), optional
        Parallax table. ``None`` skips parallax.
    gp_params : dict or None, optional
        GP hyperparameters.
    fixed_jitter : bool, optional
        Passed to the GP path.
    n_outline : int or None, optional
        Finite-source limb samples. ``None`` keeps the kernel default
        (adaptive for FSBL).

    Returns
    -------
    lnL : float
        Summed photometric log-likelihood.
    """
    if 'v_para' in p:
        state = _phot_only_state(p, t)
    else:
        state = photastrom_tracks(p, t)

    (t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2, m1, m2,
     pi_s, pi_l, rho) = state

    if rho is None:
        return psbl_log_likely_photometry_tracks(
            t, t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2, m1, m2,
            mag_src, b_sff, mag_obs, mag_err,
            parallax_vectors=parallax_vectors, piS=pi_s, piL=pi_l,
            gp_params=gp_params, fixed_jitter=fixed_jitter)

    # Phot-only FSBL. The host contour leaves the source in the CoM
    # frame and recenters only the lenses (``_contour_frame`` skips the
    # shift when ``astrometryFlag`` is false). PhotAstrom does shift the
    # source, which ``fsbl_log_likely_photometry_tracks`` already does.
    if 'v_para' in p:
        return _phot_only_fsbl_lnl(
            t, t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2, m1, m2,
            rho, mag_src, b_sff, mag_obs, mag_err, pi_s, pi_l,
            parallax_vectors=parallax_vectors, gamma=_gamma_of(p),
            n_outline=n_outline, gp_params=gp_params,
            fixed_jitter=fixed_jitter)

    from bagle.jax.fsbl import fsbl_log_likely_photometry_tracks

    return fsbl_log_likely_photometry_tracks(
        t, t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2, m1, m2, rho,
        mag_src, b_sff, mag_obs, mag_err,
        parallax_vectors=parallax_vectors, piS=pi_s, piL=pi_l,
        gamma=_gamma_of(p), n_outline=n_outline, gp_params=gp_params,
        fixed_jitter=fixed_jitter)


def kepler_log_likely_astrometry(p, t, x_obs, y_obs, x_err, y_err,
                                 b_sff=1.0, mag_src=18.0, dmag_Lp_Ls=20.0,
                                 parallax_vectors=None, n_outline=None):
    """
    Astrometric log-likelihood for one Keplerian binary-lens parameterization.

    Parameters
    ----------
    p : dict
        Filter-local fitter parameters (PhotAstrom).
    t, x_obs, y_obs, x_err, y_err : array_like, shape (N_times,)
        Times and East / North astrometry (arcsec).
    b_sff, mag_src, dmag_Lp_Ls : float, optional
        Blend, source magnitude and lens magnitude difference.
    parallax_vectors : array_like, shape (N_times, 2), optional
        Parallax table.
    n_outline : int or None, optional
        Finite-source limb samples.

    Returns
    -------
    lnL : float
        Summed astrometric log-likelihood.
    """
    (t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2, m1, m2,
     pi_s, pi_l, rho) = photastrom_tracks(p, t)

    if rho is None:
        return psbl_log_likely_astrometry(
            t, t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2, m1, m2,
            mag_src, b_sff, x_obs, y_obs, x_err, y_err,
            dmag_Lp_Ls=dmag_Lp_Ls, parallax_vectors=parallax_vectors,
            piS=pi_s, piL=pi_l)

    from bagle.jax.fsbl import fsbl_log_likely_astrometry

    return fsbl_log_likely_astrometry(
        t, t0, x_s0, x_l0, mu_s, mu_l, theta, off1, off2, m1, m2, rho,
        mag_src, b_sff, x_obs, y_obs, x_err, y_err,
        dmag_Lp_Ls=dmag_Lp_Ls, parallax_vectors=parallax_vectors,
        piS=pi_s, piL=pi_l, gamma=_gamma_of(p), n_outline=n_outline)


class _KeplerBinaryLnL:
    """Classmethods copied onto PSBL and FSBL CircOrbs / EllOrbs classes."""

    @classmethod
    def jax_log_likely_photometry(cls, vec, t, mag_obs, mag_err, b_sff,
                                  mag_src, parallax_vectors=None,
                                  gp_params=None, fixed_jitter=True,
                                  n_outline=None):
        """
        Traced Keplerian photometry for this class's parameter vector.

        Parameters
        ----------
        vec : array_like
            Packed ``fitter_param_names`` (filter-local).
        t, mag_obs, mag_err : array_like, shape (N_times,)
            Times and photometry.
        b_sff, mag_src : float
            Filter-local blend and source magnitude.
        parallax_vectors, gp_params, fixed_jitter, n_outline : optional
            See :func:`kepler_log_likely_photometry`.

        Returns
        -------
        lnL : float
            Summed photometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))

        # The fitter passes the photometry slot unchanged. mag_base
        # classes store the baseline, not the unlensed source.
        if ('mag_base' in cls.fitter_param_names
                and 'mag_src' not in cls.fitter_param_names):
            mag_src = mag_src - 2.5 * jnp.log10(jnp.maximum(b_sff, 1e-12))

        return kepler_log_likely_photometry(
            p, t, mag_obs, mag_err, b_sff, mag_src,
            parallax_vectors=parallax_vectors, gp_params=gp_params,
            fixed_jitter=fixed_jitter, n_outline=n_outline)

    @classmethod
    def jax_log_likely_astrometry(cls, vec, t, x_obs, y_obs, x_err, y_err,
                                  b_sff=1.0, mag_src=18.0, dmag_Lp_Ls=20.0,
                                  parallax_vectors=None, n_outline=None):
        """
        Traced Keplerian astrometry for this class's parameter vector.

        Parameters
        ----------
        vec : array_like
            Packed ``fitter_param_names`` (filter-local).
        t, x_obs, y_obs, x_err, y_err : array_like, shape (N_times,)
            Times and East / North astrometry (arcsec).
        b_sff, mag_src, dmag_Lp_Ls : float, optional
            Blend, source magnitude and lens magnitude difference.
        parallax_vectors, n_outline : optional
            Parallax table and finite-source limb samples.

        Returns
        -------
        lnL : float
            Summed astrometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))

        return kepler_log_likely_astrometry(
            p, t, x_obs, y_obs, x_err, y_err, b_sff=b_sff, mag_src=mag_src,
            dmag_Lp_Ls=dmag_Lp_Ls, parallax_vectors=parallax_vectors,
            n_outline=n_outline)
