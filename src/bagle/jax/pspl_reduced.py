"""
Traced PSPL PhotAstrom likelihoods for the reduced parameterizations.

Param2 already has its own methods. These cover Param4 (baseline magnitude),
Param5 (``piEN_piEE`` and ``log10_thetaE``), Param6 (``log_piE``), Param7
(``log10_piE``), Astrom Param4, and Param4 geoproj.
"""
import jax.numpy as jnp

from bagle.jax.frame_convert import geo_phot_to_helio_jax
from bagle.jax.geometry import _piE_EN, derive_pspl_photastrom_reduced
from bagle.jax_physics import (
    pspl_log_likely_astrometry,
    pspl_log_likely_photometry,
)


def _theta(p):
    """
    Einstein radius in mas.

    Parameters
    ----------
    p : dict
        Named parameters.

    Returns
    -------
    thetaE : float
        Einstein radius (mas).
    """
    if 'thetaE' in p:
        return p['thetaE']

    return 10.0 ** p['log10_thetaE']


def _geometry(p):
    """
    Reduced PSPL geometry from a heliocentric parameter dict.

    Parameters
    ----------
    p : dict
        Named parameters, with ``t0``, ``u0_amp``, ``piE_E`` and ``piE_N``.

    Returns
    -------
    geom : tuple
        Output of :func:`derive_pspl_photastrom_reduced`.
    """
    piE_E, piE_N = _piE_EN(p)

    return derive_pspl_photastrom_reduced(
        p['t0'], p['u0_amp'], p['tE'], _theta(p), p['piS'], piE_E, piE_N,
        p['xS0_E'], p['xS0_N'], p['muS_E'], p['muS_N'])


def _helio(p, raL, decL, t0par):
    """
    Copy geoproj photometry parameters into the heliocentric frame.

    Parameters
    ----------
    p : dict
        Named geoproj parameters.
    raL, decL : float
        Event coordinates (degrees).
    t0par : float
        Geocentric reference time (MJD).

    Returns
    -------
    q : dict
        Parameters with heliocentric ``t0``, ``u0_amp``, ``tE``, ``piE_E``
        and ``piE_N``.
    """
    x_geo = jnp.stack([
        p['t0_geotr'], p['u0_amp_geotr'], p['tE_geotr'],
        p['piE_E_geotr'], p['piE_N_geotr'],
        jnp.asarray(t0par, dtype=jnp.float64),
    ])
    t0, u0, tE, piE_E, piE_N = geo_phot_to_helio_jax(x_geo, raL, decL)
    q = dict(p)
    q['t0'] = t0
    q['u0_amp'] = u0
    q['tE'] = tE
    q['piE_E'] = piE_E
    q['piE_N'] = piE_N

    return q


def _source_mag(p, mag_src, b_sff):
    """
    Source magnitude from the fitter argument.

    Parameters
    ----------
    p : dict
        Named parameters.
    mag_src : float
        Source magnitude, or the baseline when the class fits ``mag_base``.
    b_sff : float
        Source flux fraction.

    Returns
    -------
    mag : float
        Unlensed source magnitude.
    """
    if 'mag_base' in p and 'mag_src' not in p:
        return mag_src - 2.5 * jnp.log10(jnp.maximum(b_sff, 1e-12))

    return mag_src


def _phot(p, t, mag_obs, mag_err, b_sff, mag_src, parallax_vectors,
          gp_params, fixed_jitter):
    """
    Photometric log-likelihood for one heliocentric parameter dict.

    Parameters
    ----------
    p : dict
        Heliocentric named parameters.
    t, mag_obs, mag_err : array_like
        Times and photometry.
    b_sff, mag_src : float
        Blend and source magnitude (or baseline).
    parallax_vectors, gp_params, fixed_jitter : optional
        Parallax table and GP inputs.

    Returns
    -------
    lnL : jnp.ndarray
        Summed photometric log-likelihood.
    """
    u0, thetaE_hat, tE, piE_E, piE_N, *_ = _geometry(p)
    mag = _source_mag(p, mag_src, b_sff)
    lnL = pspl_log_likely_photometry(
        t, p['t0'], tE, u0, thetaE_hat, mag, b_sff, mag_obs, mag_err,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N,
        gp_params=gp_params, fixed_jitter=fixed_jitter)

    return lnL


def _ast(p, t, x_obs, y_obs, x_err, y_err, b_sff, parallax_vectors):
    """
    Astrometric log-likelihood for one heliocentric parameter dict.

    Parameters
    ----------
    p : dict
        Heliocentric named parameters.
    t, x_obs, y_obs, x_err, y_err : array_like
        Times and East / North astrometry.
    b_sff : float
        Source flux fraction.
    parallax_vectors : array_like or None
        Parallax table.

    Returns
    -------
    lnL : jnp.ndarray
        Summed astrometric log-likelihood.
    """
    _, _, _, _, _, xS0, xL0, muS, muL, thetaE, piS, piL = _geometry(p)
    lnL = pspl_log_likely_astrometry(
        t, p['t0'], xS0, xL0, muS, muL, thetaE, b_sff,
        x_obs, y_obs, x_err, y_err,
        parallax_vectors=parallax_vectors, piS=piS, piL=piL)

    return lnL


class _PSPLReducedLnL:
    """Heliocentric reduced PSPL likelihoods."""

    @classmethod
    def jax_log_likely_photometry(cls, vec, t, mag_obs, mag_err, b_sff,
                                  mag_src, parallax_vectors=None,
                                  gp_params=None, fixed_jitter=True):
        """
        Traced PSPL photometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, mag_obs, mag_err : array_like, float, shape = (N_times,)
            Times and photometry.
        b_sff : float
            Source flux fraction.
        mag_src : float
            Source magnitude, or the baseline for ``mag_base`` classes.
        parallax_vectors, gp_params, fixed_jitter : optional
            Parallax table and GP inputs.

        Returns
        -------
        lnL : float
            Summed photometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        lnL = _phot(p, t, mag_obs, mag_err, b_sff, mag_src,
                    parallax_vectors, gp_params, fixed_jitter)

        return lnL

    @classmethod
    def jax_log_likely_astrometry(cls, vec, t, x_obs, y_obs, x_err, y_err,
                                  b_sff=1.0, parallax_vectors=None):
        """
        Traced PSPL astrometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, x_obs, y_obs, x_err, y_err : array_like, float
            Times and East / North astrometry.
        b_sff : float, optional
            Source flux fraction.
        parallax_vectors : array_like, optional
            Parallax table.

        Returns
        -------
        lnL : float
            Summed astrometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        lnL = _ast(p, t, x_obs, y_obs, x_err, y_err, b_sff,
                   parallax_vectors)

        return lnL


class _PSPLGeoLnL:
    """Geoproj PSPL likelihoods. Photometry parameters are converted first."""

    @classmethod
    def jax_log_likely_photometry(cls, vec, t, mag_obs, mag_err, b_sff,
                                  mag_src, parallax_vectors=None,
                                  gp_params=None, fixed_jitter=True,
                                  raL=None, decL=None, t0par=None):
        """
        Traced geoproj PSPL photometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, mag_obs, mag_err : array_like, float, shape = (N_times,)
            Times and photometry.
        b_sff : float
            Source flux fraction.
        mag_src : float
            Baseline magnitude for this filter.
        parallax_vectors, gp_params, fixed_jitter : optional
            Parallax table and GP inputs.
        raL, decL : float
            Event coordinates (degrees).
        t0par : float
            Geocentric reference time (MJD).

        Returns
        -------
        lnL : float
            Summed photometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        q = _helio(p, raL, decL, t0par)
        lnL = _phot(q, t, mag_obs, mag_err, b_sff, mag_src,
                    parallax_vectors, gp_params, fixed_jitter)

        return lnL

    @classmethod
    def jax_log_likely_astrometry(cls, vec, t, x_obs, y_obs, x_err, y_err,
                                  b_sff=1.0, parallax_vectors=None,
                                  raL=None, decL=None, t0par=None):
        """
        Traced geoproj PSPL astrometric log-likelihood.

        Parameters
        ----------
        vec : array_like, float, shape = (N_params,)
            Filter-local vector in ``fitter_param_names`` order.
        t, x_obs, y_obs, x_err, y_err : array_like, float
            Times and East / North astrometry.
        b_sff : float, optional
            Source flux fraction.
        parallax_vectors : array_like, optional
            Parallax table.
        raL, decL : float
            Event coordinates (degrees).
        t0par : float
            Geocentric reference time (MJD).

        Returns
        -------
        lnL : float
            Summed astrometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        q = _helio(p, raL, decL, t0par)
        lnL = _ast(q, t, x_obs, y_obs, x_err, y_err, b_sff,
                   parallax_vectors)

        return lnL
