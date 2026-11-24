"""
Traced static PSTL PhotAstrom likelihoods for Param2 and Param3.

Param1 is mass-distance. These two are the reduced forms: ``thetaE`` or
``log10_thetaE``, with ``mag_src`` or ``mag_base``. The geometry is the
Param1 geometry after the masses and distances are recovered.
"""
import jax.numpy as jnp

from bagle.jax.geometry import (
    _DAYS_PER_YEAR,
    _KAPPA_MAS_MSUN,
    _PI_MAS_PER_PC,
    derive_pstl_photastrom_param1,
)
from bagle.jax.pstl import (
    pstl_log_likely_astrometry,
    pstl_log_likely_photometry_tracks,
)


def _param1_args(p):
    """
    Mass-distance arguments for :func:`derive_pstl_photastrom_param1`.

    Parameters
    ----------
    p : dict
        Named Param2 or Param3 parameters.

    Returns
    -------
    args : tuple
        ``(mLp, mLs, mLt, t0, xS0_E, xS0_N, beta, muL_E, muL_N, muS_E,
        muS_N, dL, dS, sep12, sep13, alpha12, alpha13)``.
    """
    theta = p['thetaE'] if 'thetaE' in p else 10.0 ** p['log10_thetaE']
    piE = jnp.stack([p['piE_E'], p['piE_N']])
    pi_amp = jnp.linalg.norm(piE)
    hat = piE / pi_amp

    # thetaE is the Einstein radius of the primary plus the secondary.
    pi_rel = pi_amp * theta
    m_12 = theta ** 2 / (pi_rel * _KAPPA_MAS_MSUN)
    m_lp = m_12 / (1.0 + p['q_12'])
    m_ls = m_lp * p['q_12']
    m_lt = m_lp * p['q_13']

    mu_rel = hat * (theta * _DAYS_PER_YEAR / p['tE'])
    mu_l = jnp.stack([p['muS_E'], p['muS_N']]) - mu_rel
    pi_l = pi_rel + p['piS']

    return (
        m_lp, m_ls, m_lt, p['t0'], p['xS0_E'], p['xS0_N'],
        p['u0_amp'] * theta, mu_l[0], mu_l[1], p['muS_E'], p['muS_N'],
        _PI_MAS_PER_PC / pi_l, _PI_MAS_PER_PC / p['piS'],
        p['sep12'], p['sep13'], p['alpha12'], p['alpha13'],
    )


def _geom(p):
    """
    Static PSTL geometry.

    Parameters
    ----------
    p : dict
        Named parameters.

    Returns
    -------
    geom : tuple
        Output of :func:`derive_pstl_photastrom_param1`.
    """
    return derive_pstl_photastrom_param1(*_param1_args(p))


class _PSTLReducedLnL:
    """Traced Param2 / Param3 likelihoods."""

    @classmethod
    def jax_log_likely_photometry(cls, vec, t, mag_obs, mag_err, b_sff,
                                  mag_src, parallax_vectors=None,
                                  gp_params=None, fixed_jitter=True):
        """
        Traced static PSTL photometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, mag_obs, mag_err : array_like, float, shape = (N_times,)
            Times and photometry.
        b_sff : float
            Source flux fraction.
        mag_src : float
            Source magnitude, or the baseline for ``mag_base``.
        parallax_vectors, gp_params, fixed_jitter : optional
            Parallax table and unused GP inputs.

        Returns
        -------
        lnL : float
            Summed photometric log-likelihood.
        """
        del gp_params, fixed_jitter

        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        if 'mag_base' in p and 'mag_src' not in p:
            mag_src = mag_src - 2.5 * jnp.log10(jnp.maximum(b_sff, 1e-12))
        geom = _geom(p)
        (_u0, _th, _tE, _e, _n, xS0, xL0, muS, muL, theta, piS, piL,
         m1, m2, m3, xL1, xL2, xL3, _a, _b, _c) = geom
        lnL = pstl_log_likely_photometry_tracks(
            t, p['t0'], xS0, xL0, muS, muL, theta, xL1, xL2, xL3,
            m1, m2, m3, mag_src, b_sff, mag_obs, mag_err,
            parallax_vectors=parallax_vectors, piS=piS, piL=piL)

        return lnL

    @classmethod
    def jax_log_likely_astrometry(cls, vec, t, x_obs, y_obs, x_err, y_err,
                                  b_sff=1.0, mag_src=18.0,
                                  dmag_Lp_Ls=20.0, dmag_Lp_Lt=-20.0,
                                  parallax_vectors=None):
        """
        Traced static PSTL astrometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, x_obs, y_obs, x_err, y_err : array_like, float
            Times and East / North astrometry.
        b_sff : float, optional
            Source flux fraction.
        mag_src : float, optional
            Unlensed source magnitude. The fitter converts ``mag_base``.
        dmag_Lp_Ls, dmag_Lp_Lt : float, optional
            Primary minus companion lens magnitudes.
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
        if 'dmag_Lp_Lt' in p:
            dmag_Lp_Lt = p['dmag_Lp_Lt']
        geom = _geom(p)
        (_u0, _th, _tE, _e, _n, xS0, xL0, muS, muL, theta, piS, piL,
         m1, m2, m3, xL1, xL2, xL3, _a, _b, _c) = geom
        lnL = pstl_log_likely_astrometry(
            t, p['t0'], xS0, xL0, muS, muL, theta, xL1, xL2, xL3,
            m1, m2, m3, mag_src, b_sff, x_obs, y_obs, x_err, y_err,
            dmag_Lp_Ls=dmag_Lp_Ls, dmag_Lp_Lt=dmag_Lp_Lt,
            parallax_vectors=parallax_vectors, piS=piS, piL=piL)

        return lnL
