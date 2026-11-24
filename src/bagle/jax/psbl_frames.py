"""
Traced PSBL and FSBL likelihoods for the parameterizations that were gaps.

Every class is reduced to the geometric-midpoint Param1 geometry, then
evaluated with the existing point-source or finite-source kernel. Linear
and accelerated secondary motion uses the per-epoch lens offsets.
"""
import jax.numpy as jnp

from bagle.jax.geometry import (
    _DAYS_PER_YEAR,
    _EINSTEIN_M_PER_MSUN,
    _KAPPA_MAS_MSUN,
    _PC_M,
    _PI_MAS_PER_PC,
    _RAD_TO_MAS,
    derive_psbl_photastrom_param1,
)
from bagle.jax_physics import (
    orbiting_lens_offsets,
    psbl_log_likely_astrometry,
    psbl_log_likely_photometry,
    psbl_log_likely_photometry_tracks,
    u0_hat_from_thetaE_hat_jax,
)


def _shift_to_midpoint(t0, u0_amp, tE, theta, hat, q, sep, phi, frame):
    """
    Move a closest approach onto the geometric midpoint.

    Parameters
    ----------
    t0 : float
        Closest-approach time in ``frame`` (MJD).
    u0_amp : float
        Signed impact parameter in ``frame`` (Einstein radii).
    tE : float
        Einstein time (days).
    theta : float
        Einstein radius (mas).
    hat : array, shape (2,)
        Unit relative proper motion (East, North).
    q : float
        Mass ratio ``mLs / mLp``.
    sep : float
        Primary-secondary separation (mas).
    phi : float
        Binary axis relative to ``hat`` (rad).
    frame : {'geom_mid', 'prim_center', 'COM'}
        Reference point of the input closest approach.

    Returns
    -------
    t0_mid, u0_mid : float
        Closest approach of the geometric midpoint.
    """
    if frame == 'geom_mid':
        return t0, u0_amp

    # Offset of the input point from the midpoint, toward the primary.
    if frame == 'prim_center':
        offset_in = 0.5 * sep / theta
    else:
        offset_in = 0.5 * (1.0 - q) / (1.0 + q) * sep / theta
    d = 0.0 - offset_in

    # Same shift as bagle.frame_convert.convert_u0_t0_psbl.
    u0_hat = u0_hat_from_thetaE_hat_jax(hat, 1.0)
    u0 = u0_amp * u0_hat
    perp = jnp.stack([hat[1], -hat[0]])
    t0_mid = t0 + d * jnp.cos(phi) * tE
    u0_mid = jnp.dot(u0 - d * jnp.sin(phi) * perp, u0_hat)

    return t0_mid, u0_mid


def _reduced_physical(p):
    """
    Masses, distances, and midpoint closest approach from reduced parameters.

    Parameters
    ----------
    p : dict
        Named fitter parameters. ``thetaE`` or ``log10_thetaE``, and
        ``piE_N`` or ``piEN_piEE``.

    Returns
    -------
    args : tuple
        Inputs of :func:`derive_psbl_photastrom_param1`.
    """
    if 'thetaE' in p:
        theta = p['thetaE']
    else:
        theta = 10.0 ** p['log10_thetaE']

    pi_e = p['piE_E']
    if 'piE_N' in p:
        pi_n = p['piE_N']
    else:
        pi_n = p['piEN_piEE'] * pi_e

    pi_vec = jnp.stack([pi_e, pi_n])
    pi_amp = jnp.linalg.norm(pi_vec)
    hat = pi_vec / pi_amp
    pi_rel = pi_amp * theta
    m_l = theta ** 2 / (pi_rel * _KAPPA_MAS_MSUN)
    m_lp = m_l / (1.0 + p['q'])
    m_ls = m_lp * p['q']
    mu_rel = hat * (theta * _DAYS_PER_YEAR / p['tE'])
    mu_s = jnp.stack([p['muS_E'], p['muS_N']])
    mu_l = mu_s - mu_rel
    pi_l = pi_rel + p['piS']
    d_l = _PI_MAS_PER_PC / pi_l
    d_s = _PI_MAS_PER_PC / p['piS']

    # Which lens point the input closest approach refers to.
    if 't0_prim' in p:
        t0_in = p['t0_prim']
        u0_in = p['u0_amp_prim']
        frame = 'prim_center'
    elif 't0_com' in p:
        t0_in = p['t0_com']
        u0_in = p['u0_amp_com']
        frame = 'COM'
    else:
        t0_in = p['t0']
        u0_in = p['u0_amp']
        frame = 'geom_mid'

    phi = p['alpha'] * jnp.pi / 180.0 - jnp.arctan2(pi_e, pi_n)
    t0, u0_amp = _shift_to_midpoint(
        t0_in, u0_in, p['tE'], theta, hat, p['q'], p['sep'], phi, frame)
    beta = u0_amp * theta

    return (
        m_lp, m_ls, t0, p['xS0_E'], p['xS0_N'], beta,
        mu_l[0], mu_l[1], p['muS_E'], p['muS_N'],
        d_l, d_s, p['sep'], p['alpha'],
    )


def _physical(p):
    """
    Midpoint Param1 arguments from a mass-distance parameterization.

    Parameters
    ----------
    p : dict
        Named fitter parameters, including ``mLp`` and ``mLs``.

    Returns
    -------
    args : tuple
        Inputs of :func:`derive_psbl_photastrom_param1`.
    """
    if 't0' in p:
        return (
            p['mLp'], p['mLs'], p['t0'], p['xS0_E'], p['xS0_N'], p['beta'],
            p['muL_E'], p['muL_N'], p['muS_E'], p['muS_N'],
            p['dL'], p['dS'], p['sep'], p['alpha'],
        )

    # Primary-lens closest approach (Param7). thetaE from the masses.
    mu_s = jnp.stack([p['muS_E'], p['muS_N']])
    mu_l = jnp.stack([p['muL_E'], p['muL_N']])
    mu_rel = mu_s - mu_l
    mu_amp = jnp.linalg.norm(mu_rel)
    hat = mu_rel / mu_amp
    inv = 1.0 / p['dL'] - 1.0 / p['dS']
    theta = jnp.sqrt(
        _EINSTEIN_M_PER_MSUN * (p['mLp'] + p['mLs']) * inv / _PC_M
    ) * _RAD_TO_MAS
    t_e = (theta / mu_amp) * _DAYS_PER_YEAR
    q = p['mLs'] / p['mLp']
    phi = p['alpha'] * jnp.pi / 180.0 - jnp.arctan2(hat[0], hat[1])
    t0, u0_amp = _shift_to_midpoint(
        p['t0_p'], p['beta_p'] / theta, t_e, theta, hat, q, p['sep'],
        phi, 'prim_center')

    return (
        p['mLp'], p['mLs'], t0, p['xS0_E'], p['xS0_N'], u0_amp * theta,
        p['muL_E'], p['muL_N'], p['muS_E'], p['muS_N'],
        p['dL'], p['dS'], p['sep'], p['alpha'],
    )


def _geometry(p):
    """
    Midpoint geometry, and per-epoch lens offsets when the secondary moves.

    Parameters
    ----------
    p : dict
        Named fitter parameters. ``p['_t']`` is the time grid, used only
        for the orbit offsets.

    Returns
    -------
    t0 : float
        Midpoint closest-approach time (MJD).
    geom : tuple
        Output of :func:`derive_psbl_photastrom_param1`.
    off1, off2 : array or None
        Primary and secondary offsets (Einstein radii), or ``None``
        for a static binary.
    """
    if 'mLp' in p:
        args = _physical(p)
    else:
        args = _reduced_physical(p)

    geom = derive_psbl_photastrom_param1(*args)
    t0 = args[2]
    if 'delta_muL_sec_E' not in p:
        return t0, geom, None, None

    # Secondary motion about the lens origin, at the midpoint t0.
    theta = geom[9]
    dmu = jnp.stack([p['delta_muL_sec_E'], p['delta_muL_sec_N']])
    acc = None
    if 'accLsec_E' in p:
        acc = jnp.stack([p['accLsec_E'], p['accLsec_N']])
    off1, off2 = orbiting_lens_offsets(
        p['_t'], t0, p['sep'], p['alpha'], theta, dmu, accL_sec=acc)

    return t0, geom, off1, off2


def _source_mag(cls, mag_src, b_sff):
    """
    Source magnitude, converting ``mag_base`` when that is the fitted name.

    Parameters
    ----------
    cls : type
        Model class (for ``fitter_param_names``).
    mag_src : float
        Filter slot. This is ``mag_base`` when that name is fitted.
    b_sff : float
        Source flux fraction.

    Returns
    -------
    mag : float
        Unlensed source magnitude.
    """
    names = cls.fitter_param_names
    if 'mag_base' in names and 'mag_src' not in names:
        return mag_src - 2.5 * jnp.log10(jnp.maximum(b_sff, 1e-12))

    return mag_src


class _PSBLFrameLnL:
    """Traced likelihoods for the PSBL and FSBL classes that were gaps."""

    @classmethod
    def jax_log_likely_photometry(cls, vec, t, mag_obs, mag_err, b_sff,
                                  mag_src, parallax_vectors=None,
                                  gp_params=None, fixed_jitter=True,
                                  n_outline=None):
        """
        Traced photometric log-likelihood in the midpoint frame.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, mag_obs, mag_err : array_like, float, shape = (N_times,)
            Times and photometry.
        b_sff : float
            Source flux fraction.
        mag_src : float
            Unlensed source magnitude, or ``mag_base`` when that is fitted.
        parallax_vectors, gp_params, fixed_jitter : optional
            Parallax table and GP inputs.
        n_outline : int or None, optional
            Limb samples. ``None`` keeps the finite-source kernel adaptive.
            Point-source classes ignore it.

        Returns
        -------
        lnL : float
            Summed photometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        p['_t'] = t
        t0, geom, off1, off2 = _geometry(p)
        mag = _source_mag(cls, mag_src, b_sff)
        (u0, hat, t_e, pi_e, pi_n, x_s0, x_l0, mu_s, mu_l, theta,
         pi_s, pi_l, m1, m2, x_l1, x_l2, _, _) = geom

        # Finite source: radiusS is arcsec on the phot+astrom classes.
        if 'radiusS' in p:
            from bagle.jax.fsbl import (
                fsbl_log_likely_photometry,
                fsbl_log_likely_photometry_tracks,
            )

            rho = p['radiusS'] * 1e3 / theta
            gamma = p.get('gamma')
            if off1 is None:
                z1 = x_l1[0] + 1j * x_l1[1]
                z2 = x_l2[0] + 1j * x_l2[1]
                return fsbl_log_likely_photometry(
                    t, t0, t_e, u0, hat, z1, z2, m1, m2,
                    rho, n_outline, mag, b_sff, mag_obs, mag_err,
                    parallax_vectors=parallax_vectors,
                    piE_E=pi_e, piE_N=pi_n, gp_params=gp_params,
                    fixed_jitter=fixed_jitter, gamma=gamma)

            return fsbl_log_likely_photometry_tracks(
                t, t0, x_s0, x_l0, mu_s, mu_l, theta,
                off1, off2, m1, m2, rho, mag, b_sff, mag_obs, mag_err,
                parallax_vectors=parallax_vectors, piS=pi_s, piL=pi_l,
                gamma=gamma, n_outline=n_outline, gp_params=gp_params,
                fixed_jitter=fixed_jitter)

        if off1 is None:
            return psbl_log_likely_photometry(
                t, t0, t_e, u0, hat, x_l1, x_l2, m1, m2,
                mag, b_sff, mag_obs, mag_err,
                parallax_vectors=parallax_vectors, piE_E=pi_e, piE_N=pi_n,
                gp_params=gp_params, fixed_jitter=fixed_jitter)

        return psbl_log_likely_photometry_tracks(
            t, t0, x_s0, x_l0, mu_s, mu_l, theta,
            off1, off2, m1, m2, mag, b_sff, mag_obs, mag_err,
            parallax_vectors=parallax_vectors, piS=pi_s, piL=pi_l,
            gp_params=gp_params, fixed_jitter=fixed_jitter)

    @classmethod
    def jax_log_likely_astrometry(cls, vec, t, x_obs, y_obs, x_err, y_err,
                                  b_sff=1.0, mag_src=18.0, dmag_Lp_Ls=20.0,
                                  parallax_vectors=None, n_outline=None):
        """
        Traced astrometric log-likelihood in the midpoint frame.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, x_obs, y_obs, x_err, y_err : array_like
            Times and East/North astrometry (arcsec).
        b_sff, mag_src, dmag_Lp_Ls : float, optional
            Blend, source magnitude, and lens magnitude difference.
            The fitter has already turned ``mag_base`` into ``mag_src``.
        parallax_vectors : array_like or None, optional
            Parallax table.
        n_outline : int or None, optional
            Limb samples. ``None`` keeps the finite-source kernel adaptive.

        Returns
        -------
        lnL : float
            Summed astrometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        p['_t'] = t
        t0, geom, off1, off2 = _geometry(p)
        (_, _, _, _, _, x_s0, x_l0, mu_s, mu_l, theta,
         pi_s, pi_l, m1, m2, x_l1, x_l2, _, _) = geom

        x1 = x_l1 if off1 is None else off1
        x2 = x_l2 if off2 is None else off2

        if 'radiusS' in p:
            from bagle.jax.fsbl import fsbl_log_likely_astrometry

            rho = p['radiusS'] * 1e3 / theta
            return fsbl_log_likely_astrometry(
                t, t0, x_s0, x_l0, mu_s, mu_l, theta,
                x1, x2, m1, m2, rho, mag_src, b_sff, x_obs, y_obs,
                x_err, y_err, dmag_Lp_Ls=dmag_Lp_Ls,
                parallax_vectors=parallax_vectors, piS=pi_s, piL=pi_l,
                gamma=p.get('gamma'), n_outline=n_outline)

        return psbl_log_likely_astrometry(
            t, t0, x_s0, x_l0, mu_s, mu_l, theta,
            x1, x2, m1, m2, mag_src, b_sff, x_obs, y_obs, x_err, y_err,
            dmag_Lp_Ls=dmag_Lp_Ls, parallax_vectors=parallax_vectors,
            piS=pi_s, piL=pi_l)
