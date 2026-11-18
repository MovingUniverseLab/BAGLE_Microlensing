"""
Traced PSBL PhotAstrom Param2 likelihood.

Param2 is the reduced form of Param1 (``thetaE``, ``u0_amp``, ``q``).
Only the GP classes use this in Step 7. The other Param2 classes stay
without a registered likelihood.
"""
import jax.numpy as jnp

from bagle.jax.geometry import (
    _DAYS_PER_YEAR,
    _KAPPA_MAS_MSUN,
    _PI_MAS_PER_PC,
    derive_psbl_photastrom_param1,
)
from bagle.jax_physics import (
    psbl_log_likely_astrometry,
    psbl_log_likely_photometry,
)


def _geom(p):
    """
    Param1 geometry from reduced Param2 parameters.

    Parameters
    ----------
    p : dict
        Named Param2 parameters.

    Returns
    -------
    geom : tuple
        Output of :func:`derive_psbl_photastrom_param1`.
    """
    theta = p['thetaE']
    piE = jnp.stack([p['piE_E'], p['piE_N']])
    pi_amp = jnp.linalg.norm(piE)
    hat = piE / pi_amp
    pi_rel = pi_amp * theta
    m_l = theta ** 2 / (pi_rel * _KAPPA_MAS_MSUN)
    m_lp = m_l / (1.0 + p['q'])
    m_ls = m_lp * p['q']
    mu_rel = hat * (theta * _DAYS_PER_YEAR / p['tE'])
    mu_l = jnp.stack([p['muS_E'], p['muS_N']]) - mu_rel
    pi_l = pi_rel + p['piS']

    return derive_psbl_photastrom_param1(
        m_lp, m_ls, p['t0'], p['xS0_E'], p['xS0_N'], p['u0_amp'] * theta,
        mu_l[0], mu_l[1], p['muS_E'], p['muS_N'],
        _PI_MAS_PER_PC / pi_l, _PI_MAS_PER_PC / p['piS'],
        p['sep'], p['alpha'])


class _PSBLReducedLnL:
    """Traced reduced PSBL likelihoods, including the GP photometry path."""

    @classmethod
    def jax_log_likely_photometry(cls, vec, t, mag_obs, mag_err, b_sff,
                                  mag_src, parallax_vectors=None,
                                  gp_params=None, fixed_jitter=True):
        """
        Traced PSBL Param2 photometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, mag_obs, mag_err : array_like, float, shape = (N_times,)
            Times and photometry.
        b_sff : float
            Source flux fraction.
        mag_src : float
            Unlensed source magnitude.
        parallax_vectors, gp_params, fixed_jitter : optional
            Parallax table and GP inputs.

        Returns
        -------
        lnL : float
            Summed photometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        geom = _geom(p)
        (u0, thetaE_hat, tE, piE_E, piE_N, _, _, _, _, _, _, _,
         m1, m2, xL1, xL2, _, _) = geom
        lnL = psbl_log_likely_photometry(
            t, p['t0'], tE, u0, thetaE_hat, xL1, xL2, m1, m2,
            mag_src, b_sff, mag_obs, mag_err,
            parallax_vectors=parallax_vectors, piE_E=piE_E,
            piE_N=piE_N, gp_params=gp_params, fixed_jitter=fixed_jitter)

        return lnL

    @classmethod
    def jax_log_likely_astrometry(cls, vec, t, x_obs, y_obs, x_err, y_err,
                                  b_sff=1.0, mag_src=18.0, dmag_Lp_Ls=20.0,
                                  parallax_vectors=None):
        """
        Traced PSBL Param2 astrometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, x_obs, y_obs, x_err, y_err : array_like, float
            Times and East / North astrometry.
        b_sff : float, optional
            Source flux fraction.
        mag_src : float, optional
            Unlensed source magnitude.
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
        geom = _geom(p)
        (_, _, _, _, _, xS0, xL0, muS, muL, theta, piS, piL,
         m1, m2, xL1, xL2, _, _) = geom
        lnL = psbl_log_likely_astrometry(
            t, p['t0'], xS0, xL0, muS, muL, theta, xL1, xL2, m1, m2,
            mag_src, b_sff, x_obs, y_obs, x_err, y_err,
            dmag_Lp_Ls=dmag_Lp_Ls, parallax_vectors=parallax_vectors,
            piS=piS, piL=piL)

        return lnL
