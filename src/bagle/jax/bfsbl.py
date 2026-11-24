"""
Traced finite-source binary-lens, binary-source likelihoods (BFSBL).

Sky tracks are the BSBL tracks. Each source is then a finite disk in
:func:`bagle.jax.bfs.bfsbl_sky_model`. Static PhotAstrom Param1 already has
its own methods and is not overwritten.
"""
import jax.numpy as jnp

from bagle.jax.bfs import (
    _two_source_photometry,
    bfsbl_sky_model,
)
from bagle.jax.bsbl import (
    _phot_only_positions,
    _photastrom_positions,
    _sep_vec,
    _source_mags,
)
from bagle.jax.fsbl import resolved_lens_fluxes_jax
from bagle.jax_physics import (
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    mag2flux_jax,
)


# 1000 mas = 1 arcsec, so an Einstein-radius coordinate is unchanged.
_THETA_FAKE_MAS = 1000.0


def _finite_sources(p, t, parallax, n_outline):
    """
    Magnifications and lensed centroids of both finite sources.

    Parameters
    ----------
    p : dict
        Named fitter parameters.
    t : array_like, float, shape = (N_times,)
        Times (MJD).
    parallax : array_like or None
        Parallax table.
    n_outline : int or None
        Initial limb samples. ``None`` keeps the kernel default.

    Returns
    -------
    amp1, amp2 : jnp.ndarray, float, shape = (N_times,)
        Magnification of each source.
    x1, x2, xL1, xL2 : jnp.ndarray, float, shape = (N_times, 2)
        Lensed source centroids and lens positions (arcsec, or Einstein
        radii for photometry-only).
    theta, m1, m2 : float, shape = ()
        Einstein radius (mas) and lens mass fractions.
    """
    # Orbiting lenses need a per-epoch caustic. Static lenses do not.
    static = 'omegaL_pri' not in p and 'delta_muL_sec_E' not in p
    gamma_pri = p['gamma_pri'] if 'gamma_pri' in p else None
    gamma_sec = p['gamma_sec'] if 'gamma_sec' in p else None

    if 'sep_S' in p:
        # Photometry-only positions are already Einstein radii.
        xS1, xS2, xL1, xL2, m1, m2 = _phot_only_positions(
            p, t, parallax)
        theta = _THETA_FAKE_MAS
        rho_pri = p['radiusS_pri']
        rho_sec = p['radiusS_sec']
    else:
        xS1, xS2, xL1, xL2, theta, m1, m2 = _photastrom_positions(
            p, t, parallax)
        # radiusS is arcsec on the sky.
        rho_pri = p['radiusS_pri'] * 1e3 / theta
        rho_sec = p['radiusS_sec'] * 1e3 / theta

    amp1, amp2, x1, x2 = bfsbl_sky_model(
        xS1, xS2, xL1, xL2, theta, m1, m2, rho_pri, rho_sec,
        gamma_pri=gamma_pri, gamma_sec=gamma_sec, n_outline=n_outline,
        static_lens=static)

    return amp1, amp2, x1, x2, xL1, xL2, theta, m1, m2


def bfsbl_log_likely_photometry(p, t, mag_obs, mag_err, b_sff, mag_src,
                                parallax_vectors=None, gp_params=None,
                                fixed_jitter=True, n_outline=None):
    """
    BFSBL photometric log-likelihood.

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
    n_outline : int or None, optional
        Initial limb samples.

    Returns
    -------
    lnL : jnp.ndarray, float, shape = ()
        Summed photometric log-likelihood.
    """
    del gp_params, fixed_jitter

    amp1, amp2, _, _, _, _, _, _, _ = _finite_sources(
        p, t, parallax_vectors, n_outline)
    mag_pri, mag_sec = _source_mags(p, mag_src)
    mag = _two_source_photometry(amp1, amp2, mag_pri, mag_sec, b_sff)
    lnL = gaussian_log_likelihood_sum(mag, mag_obs, mag_err)

    return lnL


def bfsbl_log_likely_astrometry(p, t, x_obs, y_obs, x_err, y_err, b_sff,
                                mag_src, dmag_Lp_Ls=20.0,
                                parallax_vectors=None, n_outline=None):
    """
    BFSBL astrometric log-likelihood.

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
    n_outline : int or None, optional
        Initial limb samples.

    Returns
    -------
    lnL : jnp.ndarray, float, shape = ()
        Summed astrometric log-likelihood.
    """
    amp1, amp2, x1, x2, xL1, xL2, theta, m1, m2 = _finite_sources(
        p, t, parallax_vectors, n_outline)

    # The host contour is about the lens midpoint, then it adds the
    # lens-system origin rather than that midpoint. Keplerian origin is
    # the centre of mass. Otherwise the primary is origin + sepL/2.
    mid = 0.5 * (xL1 + xL2)
    if 'omegaL_pri' in p:
        origin = m1 * xL1 + m2 * xL2
    elif 'sepL' in p:
        half = 0.5 * _sep_vec(p['sepL'], p['alphaL']) * 1e-3
        origin = xL1 - half
    else:
        origin = mid
    x1 = origin + (x1 - mid)
    x2 = origin + (x2 - mid)
    del theta

    mag_pri, mag_sec = _source_mags(p, mag_src)
    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    fL1, fL2 = resolved_lens_fluxes_jax(f1 + f2, b_sff, dmag_Lp_Ls)

    # Flux-weighted centroid of both disks and the luminous lenses.
    numer = (x1 * (f1 * amp1)[:, None] + x2 * (f2 * amp2)[:, None]
             + xL1 * fL1 + xL2 * fL2)
    denom = f1 * amp1 + f2 * amp2 + fL1 + fL2
    pos = numer / denom[:, None]
    lnL = gaussian_astrometry_log_likelihood_sum(
        pos, x_obs, y_obs, x_err, y_err)

    return lnL


class _BFSBLLnL:
    """Traced BFSBL likelihoods shared by descriptor copy."""

    @classmethod
    def jax_log_likely_photometry(cls, vec, t, mag_obs, mag_err, b_sff,
                                  mag_src, parallax_vectors=None,
                                  gp_params=None, fixed_jitter=True,
                                  n_outline=None):
        """
        Traced BFSBL photometric log-likelihood.

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
        n_outline : int or None, optional
            Initial limb samples. The default keeps the adaptive contour.

        Returns
        -------
        lnL : float
            Summed photometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        lnL = bfsbl_log_likely_photometry(
            p, t, mag_obs, mag_err, b_sff, mag_src,
            parallax_vectors=parallax_vectors, gp_params=gp_params,
            fixed_jitter=fixed_jitter, n_outline=n_outline)

        return lnL

    @classmethod
    def jax_log_likely_astrometry(cls, vec, t, x_obs, y_obs, x_err, y_err,
                                  b_sff, mag_src, dmag_Lp_Ls=20.0,
                                  parallax_vectors=None, n_outline=None):
        """
        Traced BFSBL astrometric log-likelihood.

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
        n_outline : int or None, optional
            Initial limb samples.

        Returns
        -------
        lnL : float
            Summed astrometric log-likelihood.
        """
        p = dict(zip(cls.fitter_param_names, jnp.asarray(vec)))
        if 'dmag_Lp_Ls' in p:
            dmag_Lp_Ls = p['dmag_Lp_Ls']
        lnL = bfsbl_log_likely_astrometry(
            p, t, x_obs, y_obs, x_err, y_err, b_sff, mag_src,
            dmag_Lp_Ls=dmag_Lp_Ls, parallax_vectors=parallax_vectors,
            n_outline=n_outline)

        return lnL
