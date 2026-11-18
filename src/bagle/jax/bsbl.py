"""
Traced point-source binary-lens, binary-source likelihoods (BSBL).

The host evaluates each source with the binary-lens image solver and
flux-weights the two sources. These kernels do the same thing from a packed
parameter vector, for the static, linear, accelerated and Keplerian
families. Finite-source BFSBL stays in :mod:`bagle.jax.bfs`.
"""
import jax.numpy as jnp

from bagle.jax.geometry import (
    _DAYS_PER_YEAR,
    _EINSTEIN_M_PER_MSUN,
    _KAPPA_MAS_MSUN,
    _KEPLER_P_DAYS,
    _PC_M,
    _PI_MAS_PER_PC,
    _RAD_TO_MAS,
    u0_hat_from_thetaE_hat_jax,
)
from bagle.jax.orbits import oal2xy
from bagle.jax.fsbl import resolved_lens_fluxes_jax
from bagle.jax_physics import (
    flux2mag_jax,
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    mag2flux_jax,
    psbl_all_arrays,
)


def _sky(east, north):
    """
    Pack an East, North pair.

    Parameters
    ----------
    east, north : float, shape = ()
        Sky components.

    Returns
    -------
    vec : jnp.ndarray, float, shape = (2,)
        ``[east, north]``.
    """
    return jnp.stack([east, north])


def _sep_vec(sep, alpha_deg):
    """
    Separation vector, East of North.

    Parameters
    ----------
    sep : float, shape = ()
        Separation.
    alpha_deg : float, shape = ()
        Position angle (degrees East of North).

    Returns
    -------
    vec : jnp.ndarray, float, shape = (2,)
        ``sep * [sin, cos]``.
    """
    alpha = alpha_deg * jnp.pi / 180.0

    return sep * jnp.stack([jnp.sin(alpha), jnp.cos(alpha)])


def _geometry(p):
    """
    Shared lens mass, parallax and relative motion.

    Parameters
    ----------
    p : dict
        Named fitter parameters.

    Returns
    -------
    thetaE, tE, hat, muS, muL, piS, piL, m1, m2 : float and shape ``(2,)``
        Einstein radius (mas), timescale (days), relative-motion direction,
        proper motions (mas/yr), parallaxes (mas) and lens mass fractions.
    """
    muS = _sky(p['muS_E'], p['muS_N']) if 'muS_E' in p else None

    if 'mLp' in p:
        # Mass-distance parameterization.
        inv = 1.0 / p['dL'] - 1.0 / p['dS']
        thetaE = jnp.sqrt(
            _EINSTEIN_M_PER_MSUN * (p['mLp'] + p['mLs']) * inv / _PC_M
        ) * _RAD_TO_MAS
        muL = _sky(p['muL_E'], p['muL_N'])
        muRel = muS - muL
        hat = muRel / jnp.linalg.norm(muRel)
        tE = (thetaE / jnp.linalg.norm(muRel)) * _DAYS_PER_YEAR
        piS = _PI_MAS_PER_PC / p['dS']
        piL = _PI_MAS_PER_PC / p['dL']
        q = p['mLs'] / p['mLp']
    else:
        # Reduced parameterization (thetaE, tE, piE, q).
        thetaE = p['thetaE']
        tE = p['tE']
        hat = _sky(p['piE_E'], p['piE_N'])
        hat = hat / jnp.linalg.norm(hat)
        muRel = hat * (thetaE / (tE / _DAYS_PER_YEAR))
        muL = muS - muRel
        piS = p['piS']
        piL = jnp.linalg.norm(_sky(p['piE_E'], p['piE_N'])) * thetaE + piS
        q = p['q']

    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)

    return thetaE, tE, hat, muS, muL, piS, piL, m1, m2


def _period(d_pc, a_mas, mass):
    """
    Keplerian period.

    Parameters
    ----------
    d_pc : float, shape = ()
        Distance (pc).
    a_mas : float, shape = ()
        Semi-major axis of the relative orbit (mas).
    mass : float, shape = ()
        Total mass (solar masses).

    Returns
    -------
    period : jnp.ndarray, float, shape = ()
        Period (days).
    """
    a_au = d_pc * a_mas * 1e-3

    return _KEPLER_P_DAYS * jnp.sqrt((a_au ** 3) / mass)


def _orbit_offsets(t, omega, big_omega, inc, ecc, period, tp, aleph, aleph2):
    """
    Primary and secondary offsets from their centre of mass.

    Parameters
    ----------
    t : array_like, float, shape = (N_times,)
        Times (MJD).
    omega, big_omega, inc : float, shape = ()
        Argument of periastron, longitude of the ascending node and
        inclination (degrees).
    ecc : float, shape = ()
        Eccentricity.
    period : float, shape = ()
        Period (days).
    tp : float, shape = ()
        Periastron time (MJD).
    aleph, aleph2 : float, shape = ()
        Primary and secondary semi-major axes, in the caller's units.

    Returns
    -------
    off1, off2 : jnp.ndarray, float, shape = (N_times, 2)
        ``[east, north]`` offsets.
    """
    # oal2xy returns East, North of the primary, then of the secondary.
    x, y, x2, y2 = oal2xy(
        t, omega, big_omega, inc, ecc, period, tp, aleph, aleph2)
    off1 = jnp.stack([x, y], axis=-1)
    off2 = jnp.stack([x2, y2], axis=-1)

    return off1, off2


def _phot_only_positions(p, t, parallax):
    """
    Photometry-only source and lens positions in Einstein radii.

    Parameters
    ----------
    p : dict
        Named parameters.
    t : array_like, float, shape = (N_times,)
        Times (MJD).
    parallax : array_like or None
        Parallax table.

    Returns
    -------
    xS1, xS2, xL1, xL2 : jnp.ndarray, float, shape = (N_times, 2)
        Positions in Einstein radii.
    m1, m2 : float, shape = ()
        Lens mass fractions (already in Einstein-radius units).
    """
    piE = _sky(p['piE_E'], p['piE_N'])
    hat = piE / jnp.linalg.norm(piE)
    u0_hat = u0_hat_from_thetaE_hat_jax(hat, p['u0_amp'])
    u0 = jnp.abs(p['u0_amp']) * u0_hat

    # Both sources share the primary closest approach.
    tau = ((t - p['t0']) / p['tE'])[:, None]
    u1 = u0 + tau * hat
    pa = jnp.arctan2(piE[0], piE[1])
    u2 = u1 + _sep_vec(p['sep_S'], jnp.rad2deg(pa) + p['phi_S'])

    if parallax is not None:
        shift = jnp.linalg.norm(piE) * parallax
        u1 = u1 - shift
        u2 = u2 - shift

    # Static lenses about the geometric centre.
    half = 0.5 * _sep_vec(p['sep_L'], jnp.rad2deg(pa) + p['phi_L'])
    xL1 = jnp.broadcast_to(half, u1.shape)
    xL2 = -xL1
    q = p['q']

    return u1, u2, xL1, xL2, 1.0 / (1.0 + q), q / (1.0 + q)


def _midpoint_frame(p, thetaE, tE, hat):
    """
    Move a static primary-lens closest approach to the lens midpoint.

    Parameters
    ----------
    p : dict
        Named parameters (``t0_p``, ``beta_p``, ``sepL``, ``alphaL``).
    thetaE, tE : float, shape = ()
        Einstein radius (mas) and timescale (days).
    hat : array_like, float, shape = (2,)
        Direction of the relative proper motion.

    Returns
    -------
    t0 : float
        Closest approach of the primary source to the lens midpoint.
    u0 : jnp.ndarray, float, shape = (2,)
        Closest-approach vector in Einstein radii.
    """
    # Primary sits +sep/2 from the midpoint (Einstein radii).
    d = -0.5 * p['sepL'] / thetaE
    phi = p['alphaL'] * jnp.pi / 180.0 - jnp.arctan2(hat[0], hat[1])
    perp = jnp.stack([hat[1], -hat[0]])
    u0_amp = p['beta_p'] / thetaE
    u0_in = jnp.abs(u0_amp) * u0_hat_from_thetaE_hat_jax(hat, p['beta_p'])
    t0 = p['t0_p'] + d * jnp.cos(phi) * tE
    u0 = u0_in - d * jnp.sin(phi) * perp

    return t0, u0


def _primary_to_com(p, tE, hat, thetaE, muS, lens_off, src_off):
    """
    Move a primary-primary instant to the CoM-CoM closest approach.

    Parameters
    ----------
    p : dict
        Named parameters.
    tE : float, shape = ()
        Einstein timescale (days).
    hat : array_like, float, shape = (2,)
        Relative-motion direction.
    thetaE : float, shape = ()
        Einstein radius (mas).
    muS : array_like, float, shape = (2,)
        Source proper motion (mas/yr).
    lens_off, src_off : array_like, float, shape = (2,)
        Primary lens and primary source offsets from their centres of mass
        at ``t0_p`` (arcsec).

    Returns
    -------
    t0 : float
        CoM-CoM closest approach (MJD).
    xS0, xL0 : jnp.ndarray, float, shape = (2,)
        Source and lens centres of mass at ``t0`` (arcsec).
    """
    u0_amp = p['beta_p'] / thetaE
    u_in = jnp.abs(u0_amp) * u0_hat_from_thetaE_hat_jax(hat, p['beta_p'])
    # Source primary minus lens primary, in Einstein radii.
    u_com = u_in - (src_off - lens_off) / (thetaE * 1e-3)
    along = jnp.dot(u_com, hat)
    t0 = p['t0_p'] - tE * along
    u0 = u_com - along * hat

    # Source CoM at t0_p, then carry it to t0 along muS.
    xS_p = _sky(p['xS0_p_E'], p['xS0_p_N'])
    xS_com = xS_p - src_off
    xS0 = xS_com + muS * ((t0 - p['t0_p']) / _DAYS_PER_YEAR) * 1e-3
    xL0 = xS0 - u0 * thetaE * 1e-3

    return t0, xS0, xL0


def _photastrom_positions(p, t, parallax):
    """
    Photometry and astrometry source and lens tracks.

    Parameters
    ----------
    p : dict
        Named parameters.
    t : array_like, float, shape = (N_times,)
        Times (MJD).
    parallax : array_like or None
        Parallax table (arcsec is applied here).

    Returns
    -------
    xS1, xS2, xL1, xL2 : jnp.ndarray, float, shape = (N_times, 2)
        Sky positions (arcsec), parallax included.
    thetaE, m1, m2 : float, shape = ()
        Einstein radius (mas) and lens mass fractions.
    """
    thetaE, tE, hat, muS, muL, piS, piL, m1, m2 = _geometry(p)
    theta_as = thetaE * 1e-3
    kepler = 'omegaL_pri' in p
    linear = 'delta_muL_sec_E' in p
    accelerated = 'accLsec_E' in p

    if kepler:
        # Lens orbit about the lens centre of mass.
        aL = p['aL'] if 'aL' in p else p['sepL']
        eL = p['eL'] if 'eL' in p else 0.0
        # Reduced classes have thetaE and piE, not masses and distances.
        if 'mLp' in p:
            dL = p['dL']
            m_lp = p['mLp']
            m_l = p['mLp'] + p['mLs']
        else:
            pi_amp = jnp.linalg.norm(_sky(p['piE_E'], p['piE_N']))
            pi_rel = pi_amp * thetaE
            dL = _PI_MAS_PER_PC / piL
            m_l = thetaE ** 2 / (pi_rel * _KAPPA_MAS_MSUN)
            m_lp = m_l / (1.0 + p['q'])
        pL = _period(dL, aL, m_l)
        aleph_sec = (m_lp / m_l) * aL
        aleph = aL - aleph_sec
        # Source orbit. Param3 gives aS and the source masses; Param1 gives
        # the two semi-major axes and the period.
        if 'aS' in p:
            eS = p['eS'] if 'eS' in p else 0.0
            pS = _period(p['dS'], p['aS'], p['mSp'] + p['mSs'])
            alephS_sec = (p['mSp'] / (p['mSp'] + p['mSs'])) * p['aS']
            alephS = p['aS'] - alephS_sec
        else:
            eS = p['eS'] if 'eS' in p else 0.0
            pS = p['pS']
            alephS = p['alephS']
            alephS_sec = p['aleph_secS']

        if 't0_p' in p:
            # Offsets of the two primaries at the input instant.
            lens_at = _orbit_offsets(
                jnp.array([p['t0_p']]), p['omegaL_pri'], p['big_omegaL_sec'],
                p['iL'], eL, pL, p['tpL'], aleph * 1e-3, aleph_sec * 1e-3)
            src_at = _orbit_offsets(
                jnp.array([p['t0_p']]), p['omegaS_pri'], p['big_omegaS_sec'],
                p['iS'], eS, pS, p['tpS'], alephS * 1e-3, alephS_sec * 1e-3)
            t0, xS0, xL0 = _primary_to_com(
                p, tE, hat, thetaE, muS, lens_at[0][0], src_at[0][0])
        else:
            # Inputs are already the CoM-CoM closest approach.
            t0 = p['t0_com']
            signed = p['beta_com'] if 'beta_com' in p else (
                p['u0_amp_com'] * thetaE)
            u0 = jnp.abs(signed / thetaE) * u0_hat_from_thetaE_hat_jax(
                hat, signed)
            xS0 = _sky(p['xS0_com_E'], p['xS0_com_N'])
            xL0 = xS0 - u0 * theta_as

        dt = ((t - t0) / _DAYS_PER_YEAR)[:, None]
        xL_com = xL0 + dt * muL * 1e-3
        xS_com = xS0 + dt * muS * 1e-3
        offL1, offL2 = _orbit_offsets(
            t, p['omegaL_pri'], p['big_omegaL_sec'], p['iL'], eL, pL,
            p['tpL'], aleph * 1e-3, aleph_sec * 1e-3)
        offS1, offS2 = _orbit_offsets(
            t, p['omegaS_pri'], p['big_omegaS_sec'], p['iS'], eS, pS,
            p['tpS'], alephS * 1e-3, alephS_sec * 1e-3)
        xL1 = xL_com + offL1
        xL2 = xL_com + offL2
        xS1 = xS_com + offS1
        xS2 = xS_com + offS2
    else:
        # Static, linear or accelerated: midpoint lens, primary source.
        if 't0_p' in p:
            t0, u0 = _midpoint_frame(p, thetaE, tE, hat)
            xS_p = _sky(p['xS0_p_E'], p['xS0_p_N'])
            xS0 = xS_p + muS * ((t0 - p['t0_p']) / _DAYS_PER_YEAR) * 1e-3
        else:
            t0 = p['t0']
            u0 = jnp.abs(p['beta'] / thetaE) * u0_hat_from_thetaE_hat_jax(
                hat, p['beta'])
            xS0 = _sky(p['xS0_E'], p['xS0_N'])
        xL0 = xS0 - u0 * theta_as
        # sepS and sepL are milliarcseconds.
        dxS = _sep_vec(p['sepS'], p['alphaS']) * 1e-3
        half = 0.5 * _sep_vec(p['sepL'], p['alphaL']) * 1e-3

        dt = ((t - t0) / _DAYS_PER_YEAR)[:, None]
        xL_mid = xL0 + dt * muL * 1e-3
        xL1 = xL_mid + half
        xL2 = xL_mid - half
        xS1 = xS0 + dt * muS * 1e-3
        mu2 = muS
        if linear:
            mu2 = muS + _sky(p['delta_muS_sec_E'], p['delta_muS_sec_N'])
            xL2 = xL2 + dt * _sky(p['delta_muL_sec_E'],
                                  p['delta_muL_sec_N']) * 1e-3
        xS2 = xS0 + dxS + dt * mu2 * 1e-3
        if accelerated:
            kick = (0.5 * dt ** 2) * 1e-3
            xS2 = xS2 + kick * _sky(p['accSsec_E'], p['accSsec_N'])
            xL2 = xL2 + kick * _sky(p['accLsec_E'], p['accLsec_N'])

    if parallax is not None:
        xS1 = xS1 + piS * parallax * 1e-3
        xS2 = xS2 + piS * parallax * 1e-3
        xL1 = xL1 + piL * parallax * 1e-3
        xL2 = xL2 + piL * parallax * 1e-3

    return xS1, xS2, xL1, xL2, thetaE, m1, m2


def _images(xS, xL1, xL2, m1, m2):
    """
    Binary-lens images of one source.

    Parameters
    ----------
    xS, xL1, xL2 : array_like, float, shape = (N_times, 2)
        Source and lens positions, in the same units.
    m1, m2 : float, shape = ()
        Lens masses in the square of those units.

    Returns
    -------
    images : jnp.ndarray, float, shape = (N_times, 5, 2)
        Image positions. Invalid roots are zero.
    amp : jnp.ndarray, float, shape = (N_times, 5)
        Image magnifications. Invalid roots are zero.
    """
    w = xS[:, 0] + 1j * xS[:, 1]
    z1 = xL1[:, 0] + 1j * xL1[:, 1]
    z2 = xL2[:, 0] + 1j * xL2[:, 1]
    image, amp = psbl_all_arrays(w, z1, z2, m1, m2, 1e-8)

    # Invalid roots must not enter the flux sum.
    ok = jnp.isfinite(amp) & jnp.isfinite(image.real) & jnp.isfinite(image.imag)
    amp = jnp.where(ok, amp, 0.0)
    xy = jnp.stack([image.real, image.imag], axis=-1)
    images = jnp.where(ok[:, :, None], xy, 0.0)

    return images, amp


def _source_mags(p, mag_src):
    """
    Primary and secondary source magnitudes.

    Parameters
    ----------
    p : dict
        Named parameters.
    mag_src : tuple, float or None
        ``(mag_pri, mag_sec)`` from the fitter, or a baseline magnitude.

    Returns
    -------
    mag_pri, mag_sec : float
        Unlensed source magnitudes.
    """
    if isinstance(mag_src, (tuple, list)):
        return mag_src[0], mag_src[1]

    if 'mag_base' in p:
        # mag_base is the blended baseline of both sources.
        base = p['mag_base'] - 2.5 * jnp.log10(jnp.maximum(p['b_sff'], 1e-12))
        fr = p['fratio_bin']
        mag_pri = base + 2.5 * jnp.log10(1.0 + fr)
        mag_sec = base + 2.5 * jnp.log10(1.0 + 1.0 / fr)

        return mag_pri, mag_sec

    return p['mag_src_pri'], p['mag_src_sec']


def bsbl_log_likely_photometry(p, t, mag_obs, mag_err, b_sff, mag_src,
                               parallax_vectors=None, gp_params=None,
                               fixed_jitter=True):
    """
    BSBL photometric log-likelihood.

    Parameters
    ----------
    p : dict
        Named fitter parameters.
    t, mag_obs, mag_err : array_like, float, shape = (N_times,)
        Times and photometry.
    b_sff : float
        Source flux fraction.
    mag_src : tuple, float or None
        Source magnitudes, or a baseline magnitude.
    parallax_vectors : array_like, optional
        Parallax table.
    gp_params, fixed_jitter : optional
        Unused; these classes have no GP.

    Returns
    -------
    lnL : jnp.ndarray, float, shape = ()
        Summed photometric log-likelihood.
    """
    del gp_params, fixed_jitter

    if 'sep_S' in p:
        xS1, xS2, xL1, xL2, m1, m2 = _phot_only_positions(
            p, t, parallax_vectors)
    else:
        xS1, xS2, xL1, xL2, thetaE, m1, m2 = _photastrom_positions(
            p, t, parallax_vectors)
        scale = (thetaE * 1e-3) ** 2
        m1 = m1 * scale
        m2 = m2 * scale

    _, amp1 = _images(xS1, xL1, xL2, m1, m2)
    _, amp2 = _images(xS2, xL1, xL2, m1, m2)
    mag_pri, mag_sec = _source_mags(p, mag_src)

    # Unclipped blend, matching the host event photometry.
    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    flux = (f1 * jnp.sum(amp1, axis=1) + f2 * jnp.sum(amp2, axis=1)
            + (f1 + f2) * (1.0 - b_sff) / b_sff)
    mag = flux2mag_jax(flux)
    lnL = gaussian_log_likelihood_sum(mag, mag_obs, mag_err)

    return lnL


def bsbl_log_likely_astrometry(p, t, x_obs, y_obs, x_err, y_err, b_sff,
                               mag_src, dmag_Lp_Ls=20.0,
                               parallax_vectors=None):
    """
    BSBL astrometric log-likelihood.

    Parameters
    ----------
    p : dict
        Named fitter parameters.
    t, x_obs, y_obs, x_err, y_err : array_like, float
        Times and East / North astrometry.
    b_sff : float
        Source flux fraction.
    mag_src : tuple or None
        ``(mag_pri, mag_sec)``.
    dmag_Lp_Ls : float, optional
        Primary minus secondary lens magnitude.
    parallax_vectors : array_like, optional
        Parallax table.

    Returns
    -------
    lnL : jnp.ndarray, float, shape = ()
        Summed astrometric log-likelihood.
    """
    xS1, xS2, xL1, xL2, thetaE, m1, m2 = _photastrom_positions(
        p, t, parallax_vectors)
    scale = (thetaE * 1e-3) ** 2
    img1, amp1 = _images(xS1, xL1, xL2, m1 * scale, m2 * scale)
    img2, amp2 = _images(xS2, xL1, xL2, m1 * scale, m2 * scale)
    mag_pri, mag_sec = _source_mags(p, mag_src)
    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    fL1, fL2 = resolved_lens_fluxes_jax(f1 + f2, b_sff, dmag_Lp_Ls)

    # Flux-weighted centroid of every image plus the two lenses.
    numer = (jnp.sum(img1 * amp1[:, :, None] * f1, axis=1)
             + jnp.sum(img2 * amp2[:, :, None] * f2, axis=1)
             + xL1 * fL1 + xL2 * fL2)
    denom = (f1 * jnp.sum(amp1, axis=1) + f2 * jnp.sum(amp2, axis=1)
             + fL1 + fL2)
    pos = numer / denom[:, None]
    lnL = gaussian_astrometry_log_likelihood_sum(
        pos, x_obs, y_obs, x_err, y_err)

    return lnL


class _BSBLLnL:
    """Traced BSBL likelihoods shared by descriptor copy."""

    @classmethod
    def jax_log_likely_photometry(cls, vec, t, mag_obs, mag_err, b_sff,
                                  mag_src, parallax_vectors=None,
                                  gp_params=None, fixed_jitter=True):
        """
        Traced BSBL photometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, mag_obs, mag_err : array_like, float, shape = (N_times,)
            Times and photometry.
        b_sff : float
            Filter-local source flux fraction.
        mag_src : tuple, float or None
            Source magnitudes from the fitter.
        parallax_vectors, gp_params, fixed_jitter : optional
            Parallax table and unused GP inputs.

        Returns
        -------
        lnL : float
            Summed photometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        lnL = bsbl_log_likely_photometry(
            p, t, mag_obs, mag_err, b_sff, mag_src,
            parallax_vectors=parallax_vectors, gp_params=gp_params,
            fixed_jitter=fixed_jitter)

        return lnL

    @classmethod
    def jax_log_likely_astrometry(cls, vec, t, x_obs, y_obs, x_err, y_err,
                                  b_sff, mag_src, dmag_Lp_Ls=20.0,
                                  parallax_vectors=None):
        """
        Traced BSBL astrometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, x_obs, y_obs, x_err, y_err : array_like, float
            Times and East / North astrometry.
        b_sff : float
            Filter-local source flux fraction.
        mag_src : tuple or None
            ``(mag_pri, mag_sec)``.
        dmag_Lp_Ls : float, optional
            Primary minus secondary lens magnitude.
        parallax_vectors : array_like, optional
            Parallax table.

        Returns
        -------
        lnL : float
            Summed astrometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        if 'dmag_Lp_Ls' in p:
            dmag_Lp_Ls = p['dmag_Lp_Ls']
        lnL = bsbl_log_likely_astrometry(
            p, t, x_obs, y_obs, x_err, y_err, b_sff, mag_src,
            dmag_Lp_Ls=dmag_Lp_Ls, parallax_vectors=parallax_vectors)

        return lnL
