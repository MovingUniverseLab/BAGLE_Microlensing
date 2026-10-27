"""
Traced binary-source finite-source models (BFSPL and BFSBL).

JAX counterparts of the host ``BFSPL`` and ``BFSBL`` classes in
:mod:`bagle.model_jax`. Each source is a finite disk (optionally linearly
limb-darkened) integrated with the traced point-lens kernel
(:func:`bagle.jax.fspl.fspl_images_amps`) or binary-lens contour kernel
(:func:`bagle.jax.fsbl_contour.fsbl_magnification_centroid`). Photometry
and astrometry are flux-weighted sums over the two sources, plus luminous
lenses for astrometry.
"""
from __future__ import annotations

import jax.numpy as jnp

from bagle.jax.fsbl import (
    fsbl_sky_images,
    fsbl_sky_tracks,
    resolved_lens_fluxes_jax,
)
from bagle.jax.fspl import fspl_images_amps
from bagle.jax_physics import (
    _DAYS_PER_YEAR,
    flux2mag_jax,
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    lens_flux_for_astrometry_jax,
    mag2flux_jax,
    pspl_photometry_from_amp,
    pspl_u,
)


def binary_source_offset(sep, alpha_deg):
    """
    Secondary-source sky offset from the primary.

    Parameters
    ----------
    sep : float, shape = ()
        Primary-to-secondary source separation on the sky (mas).
    alpha_deg : float, shape = ()
        Position angle of the secondary, East of North (degrees).

    Returns
    -------
    dxS : jnp.ndarray, float, shape = (2,)
        Secondary minus primary source position (arcsec).
    """
    # Separation vector on the sky (mas -> arcsec).
    alpha = alpha_deg * jnp.pi / 180.0
    sep_vec = sep * jnp.stack([jnp.sin(alpha), jnp.cos(alpha)])

    return sep_vec * 1e-3


def _two_source_flux(amp_pri, amp_sec, mag_pri, mag_sec):
    """
    Source fluxes and the flux-weighted magnification of a binary source.

    Parameters
    ----------
    amp_pri, amp_sec : array_like, float, shape = (N_times,)
        Total magnification of each source.
    mag_pri, mag_sec : float, shape = ()
        Unlensed magnitudes of the two sources.

    Returns
    -------
    f1, f2 : jnp.ndarray, float, shape = ()
        Unlensed source fluxes.
    amp_eff : jnp.ndarray, float, shape = (N_times,)
        ``(f1 A1 + f2 A2) / (f1 + f2)``.
    """
    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    amp_eff = (f1 * amp_pri + f2 * amp_sec) / (f1 + f2)

    return f1, f2, amp_eff


def _two_source_photometry(amp_pri, amp_sec, mag_pri, mag_sec, b_sff):
    """
    Blended unresolved magnitudes of a binary source.

    Parameters
    ----------
    amp_pri, amp_sec : array_like, float, shape = (N_times,)
        Total magnification of each source.
    mag_pri, mag_sec : float, shape = ()
        Unlensed magnitudes of the two sources.
    b_sff : float, shape = ()
        Source flux fraction ``(f1 + f2) / f_total``.

    Returns
    -------
    mag : jnp.ndarray, float, shape = (N_times,)
        Model magnitudes.
    """
    f1, f2, amp_eff = _two_source_flux(amp_pri, amp_sec, mag_pri, mag_sec)

    # flux = f1 A1 + f2 A2 + (f1 + f2)(1 - b) / b, written as one source.
    mag_tot = flux2mag_jax(f1 + f2)

    return pspl_photometry_from_amp(amp_eff, mag_tot, b_sff)


def _phot_log_likelihood(t, mag_model, mag_obs, mag_err, gp_params=None,
                         fixed_jitter=True):
    """
    Gaussian or GP photometric log-likelihood.

    Parameters
    ----------
    t : array_like, float, shape = (N_times,)
        Observation times (MJD).
    mag_model, mag_obs, mag_err : array_like, float, shape = (N_times,)
        Model, observed and uncertainty magnitudes.
    gp_params : dict or None, optional
        GP hyperparameters; ``None`` uses a plain Gaussian likelihood.
    fixed_jitter : bool, optional
        Passed to the GP path.

    Returns
    -------
    lnL : jnp.ndarray, float, shape = ()
        Summed log-likelihood.
    """
    if gp_params is None:
        return gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)

    from bagle.jax.gp import gp_log_likely_photometry

    return gp_log_likely_photometry(t, mag_obs, mag_err, mag_model, gp_params,
                                    fixed_jitter=fixed_jitter)


# -------------------------------------------------------------------------
# BFSPL: binary source, point lens.
# -------------------------------------------------------------------------

def bfspl_images_amps(t, t0, tE, u0_pri, u0_sec, thetaE_hat, rho_pri, rho_sec,
                      n_outline, gamma_pri=None, gamma_sec=None,
                      parallax_vectors=None, piE_E=None, piE_N=None, n_ld=8):
    """
    Lens-relative images and amplifications of both finite sources.

    Parameters
    ----------
    t : array_like, float, shape = (N_times,)
        Observation times (MJD).
    t0, tE : float, shape = ()
        Closest-approach time (MJD) and Einstein time (days).
    u0_pri, u0_sec : array_like, float, shape = (2,)
        Source - lens offsets of the two sources at ``t0`` (Einstein radii);
        ``u0_sec`` includes the separation along ``thetaE_hat``.
    thetaE_hat : array_like, float, shape = (2,)
        Unit relative proper-motion direction.
    rho_pri, rho_sec : float, shape = ()
        Source radii (Einstein radii).
    n_outline : int
        Static limb samples per disk.
    gamma_pri, gamma_sec : float or None, optional
        Linear limb-darkening coefficients; ``None`` is a uniform source.
    parallax_vectors : array_like, float, shape = (N_times, 2), optional
        Parallax table; ``None`` skips parallax.
    piE_E, piE_N : float, optional
        Microlensing parallax components.
    n_ld : int, optional
        Radial Gauss-Legendre nodes per sub-interval.

    Returns
    -------
    img_pri, img_sec : jnp.ndarray, float, shape = (N_times, 2, 2)
        ``[+, -]`` image centroids relative to the lens (Einstein radii).
    amp_pri, amp_sec : jnp.ndarray, float, shape = (N_times, 2)
        Signed image amplifications.
    """
    # Einstein-frame trajectories (same motion, different u0).
    kw = dict(parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N)
    u_pri = pspl_u(t, t0, tE, u0_pri, thetaE_hat, **kw)
    u_sec = pspl_u(t, t0, tE, u0_sec, thetaE_hat, **kw)

    # Finite-source images per source (limb-darkened when gamma is given).
    img_pri, amp_pri = fspl_images_amps(u_pri, rho_pri, n_outline,
                                        gamma=gamma_pri, n_ld=n_ld)
    img_sec, amp_sec = fspl_images_amps(u_sec, rho_sec, n_outline,
                                        gamma=gamma_sec, n_ld=n_ld)

    return img_pri, img_sec, amp_pri, amp_sec


def bfspl_photometry(t, t0, tE, u0_pri, u0_sec, thetaE_hat, rho_pri, rho_sec,
                     n_outline, mag_pri, mag_sec, b_sff, gamma_pri=None,
                     gamma_sec=None, parallax_vectors=None, piE_E=None,
                     piE_N=None, n_ld=8):
    """
    Unresolved BFSPL magnitudes (traced).

    Parameters
    ----------
    t, t0, tE, u0_pri, u0_sec, thetaE_hat, rho_pri, rho_sec, n_outline
        See :func:`bfspl_images_amps`.
    mag_pri, mag_sec : float, shape = ()
        Unlensed source magnitudes.
    b_sff : float, shape = ()
        Source flux fraction.
    gamma_pri, gamma_sec, parallax_vectors, piE_E, piE_N, n_ld : optional
        See :func:`bfspl_images_amps`.

    Returns
    -------
    mag : jnp.ndarray, float, shape = (N_times,)
        Model magnitudes.
    """
    _, _, amp_pri, amp_sec = bfspl_images_amps(
        t, t0, tE, u0_pri, u0_sec, thetaE_hat, rho_pri, rho_sec, n_outline,
        gamma_pri=gamma_pri, gamma_sec=gamma_sec,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N, n_ld=n_ld)

    # Total (signed-sum) magnification of each source.
    A1 = jnp.sum(amp_pri, axis=1)
    A2 = jnp.sum(amp_sec, axis=1)

    return _two_source_photometry(A1, A2, mag_pri, mag_sec, b_sff)


def bfspl_astrometry(t, t0, tE, u0_pri, u0_sec, thetaE_hat, xL0, muL,
                     thetaE_amp, rho_pri, rho_sec, n_outline, mag_pri, mag_sec,
                     b_sff, gamma_pri=None, gamma_sec=None,
                     parallax_vectors=None, piE_E=None, piE_N=None, piL=None,
                     n_ld=8):
    """
    Unresolved BFSPL sky centroid (arcsec, traced).

    Parameters
    ----------
    t, t0, tE, u0_pri, u0_sec, thetaE_hat : see :func:`bfspl_images_amps`
        Source trajectories.
    xL0 : array_like, float, shape = (2,)
        Lens sky position at ``t0`` (arcsec).
    muL : array_like, float, shape = (2,)
        Lens proper motion (mas/yr).
    thetaE_amp : float, shape = ()
        Einstein radius (mas).
    rho_pri, rho_sec, n_outline : see :func:`bfspl_images_amps`
        Source radii and limb samples.
    mag_pri, mag_sec, b_sff : float, shape = ()
        Source magnitudes and source flux fraction.
    gamma_pri, gamma_sec, parallax_vectors, piE_E, piE_N, n_ld : optional
        See :func:`bfspl_images_amps`.
    piL : float, optional
        Lens parallax (mas), used with ``parallax_vectors``.

    Returns
    -------
    pos : jnp.ndarray, float, shape = (N_times, 2)
        East / North centroid (arcsec).

    Notes
    -----
    Each source's images are weighted by that source's own signed
    amplifications, and the clipped lens flux sits at the lens position,
    as in host ``BFSPL_PhotAstrom.get_astrometry``.
    """
    img_pri, img_sec, amp_pri, amp_sec = bfspl_images_amps(
        t, t0, tE, u0_pri, u0_sec, thetaE_hat, rho_pri, rho_sec, n_outline,
        gamma_pri=gamma_pri, gamma_sec=gamma_sec,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N, n_ld=n_ld)

    # Lens sky track (arcsec).
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    dt = ((t - t0) / _DAYS_PER_YEAR).reshape(-1, 1)
    xL = xL0.reshape(1, 2) + dt * muL.reshape(1, 2) * 1e-3
    if parallax_vectors is not None:
        xL = xL + piL * jnp.asarray(parallax_vectors) * 1e-3

    # Image positions on the sky; shapes [N_times, 2 images, 2].
    thetaE_as = thetaE_amp * 1e-3
    sky_pri = xL[:, None, :] + img_pri * thetaE_as
    sky_sec = xL[:, None, :] + img_sec * thetaE_as

    # Source fluxes and clipped lens flux.
    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    fL = lens_flux_for_astrometry_jax(f1 + f2, b_sff)

    # Signed-amplification-weighted sum over all four images plus the lens.
    numer = (f1 * jnp.sum(sky_pri * amp_pri[:, :, None], axis=1)
             + f2 * jnp.sum(sky_sec * amp_sec[:, :, None], axis=1) + fL * xL)
    denom = (f1 * jnp.sum(amp_pri, axis=1) + f2 * jnp.sum(amp_sec, axis=1)
             + fL)

    return numer / denom[:, None]


def bfspl_log_likely_photometry(t, t0, tE, u0_pri, u0_sec, thetaE_hat, rho_pri,
                                rho_sec, n_outline, mag_pri, mag_sec, b_sff,
                                mag_obs, mag_err, gamma_pri=None,
                                gamma_sec=None, parallax_vectors=None,
                                piE_E=None, piE_N=None, gp_params=None,
                                fixed_jitter=True, n_ld=8):
    """
    BFSPL photometric log-likelihood (traced).

    Parameters
    ----------
    t, ..., b_sff : see :func:`bfspl_photometry`
        Forward-model arguments.
    mag_obs, mag_err : array_like, float, shape = (N_times,)
        Observed magnitudes and uncertainties.
    gamma_pri, gamma_sec, parallax_vectors, piE_E, piE_N, n_ld : optional
        See :func:`bfspl_photometry`.
    gp_params, fixed_jitter : optional
        GP likelihood inputs; ``gp_params=None`` is a plain Gaussian.

    Returns
    -------
    lnL : jnp.ndarray, float, shape = ()
        Summed log-likelihood.
    """
    mag_model = bfspl_photometry(
        t, t0, tE, u0_pri, u0_sec, thetaE_hat, rho_pri, rho_sec, n_outline,
        mag_pri, mag_sec, b_sff, gamma_pri=gamma_pri, gamma_sec=gamma_sec,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N, n_ld=n_ld)

    return _phot_log_likelihood(t, mag_model, mag_obs, mag_err,
                                gp_params=gp_params, fixed_jitter=fixed_jitter)


def bfspl_log_likely_astrometry(t, t0, tE, u0_pri, u0_sec, thetaE_hat, xL0,
                                muL, thetaE_amp, rho_pri, rho_sec, n_outline,
                                mag_pri, mag_sec, b_sff, x_obs, y_obs, x_err,
                                y_err, gamma_pri=None, gamma_sec=None,
                                parallax_vectors=None, piE_E=None, piE_N=None,
                                piL=None, n_ld=8):
    """
    BFSPL astrometric log-likelihood (traced).

    Parameters
    ----------
    t, ..., b_sff : see :func:`bfspl_astrometry`
        Forward-model arguments.
    x_obs, y_obs, x_err, y_err : array_like, float, shape = (N_times,)
        Observed East / North positions and uncertainties (arcsec).
    gamma_pri, gamma_sec, parallax_vectors, piE_E, piE_N, piL, n_ld : optional
        See :func:`bfspl_astrometry`.

    Returns
    -------
    lnL : jnp.ndarray, float, shape = ()
        Summed Gaussian astrometric log-likelihood.
    """
    pos = bfspl_astrometry(
        t, t0, tE, u0_pri, u0_sec, thetaE_hat, xL0, muL, thetaE_amp, rho_pri,
        rho_sec, n_outline, mag_pri, mag_sec, b_sff, gamma_pri=gamma_pri,
        gamma_sec=gamma_sec, parallax_vectors=parallax_vectors, piE_E=piE_E,
        piE_N=piE_N, piL=piL, n_ld=n_ld)

    return gaussian_astrometry_log_likelihood_sum(pos, x_obs, y_obs, x_err,
                                                  y_err)


# -------------------------------------------------------------------------
# BFSBL: binary source, binary lens.
# -------------------------------------------------------------------------

def bfsbl_sky_model(xS_pri, xS_sec, xL1, xL2, thetaE_amp, m1, m2, rho_pri,
                    rho_sec, gamma_pri=None, gamma_sec=None, n_outline=None,
                    root_tol=1e-10, static_lens=False):
    """
    Magnifications and lensed sky centroids of both sources (traced).

    Parameters
    ----------
    xS_pri, xS_sec : array_like, float, shape = (N_times, 2)
        Unlensed source sky positions (arcsec).
    xL1, xL2 : array_like, float, shape = (N_times, 2)
        Lens sky positions (arcsec).
    thetaE_amp : float, shape = ()
        Einstein radius of the total lens mass (mas).
    m1, m2 : float, shape = ()
        Lens mass fractions.
    rho_pri, rho_sec : float, shape = ()
        Source radii (Einstein radii).
    gamma_pri, gamma_sec : float or None, optional
        Linear limb-darkening coefficients; both ``None`` (uniform) or both
        set.
    n_outline : int or None, optional
        Initial limb samples per disk (static).
    root_tol : float, optional
        Lens-equation tolerance.
    static_lens : bool, optional
        Lens offsets are the same at every epoch (static Python flag).

    Returns
    -------
    amp_pri, amp_sec : jnp.ndarray, float, shape = (N_times,)
        Total magnification of each source.
    x_pri, x_sec : jnp.ndarray, float, shape = (N_times, 2)
        Lensed sky centroid of each source (arcsec).
    """
    n_t = xS_pri.shape[0]

    # Both sources in one kernel call: rows [primary epochs, secondary].
    xS = jnp.concatenate([xS_pri, xS_sec], axis=0)
    rho = jnp.concatenate([jnp.full((n_t,), rho_pri), jnp.full((n_t,), rho_sec)])
    gamma = None
    if gamma_pri is not None:
        gamma = jnp.concatenate([jnp.full((n_t,), gamma_pri),
                                 jnp.full((n_t,), gamma_sec)])
    xL1_2 = jnp.concatenate([xL1, xL1], axis=0)
    xL2_2 = jnp.concatenate([xL2, xL2], axis=0)

    amp, x_img = fsbl_sky_images(xS, xL1_2, xL2_2, thetaE_amp, m1, m2, rho,
                                 gamma=gamma, n_outline=n_outline,
                                 root_tol=root_tol, static_lens=static_lens)

    return amp[:n_t], amp[n_t:], x_img[:n_t], x_img[n_t:]


def bfsbl_photastrom(t, t0, xS0, dxS, xL0, muS, muL, thetaE_amp,
                     xL1_over_theta, xL2_over_theta, m1, m2, rho_pri, rho_sec,
                     mag_pri, mag_sec, b_sff, dmag_Lp_Ls=20.0, gamma_pri=None,
                     gamma_sec=None, parallax_vectors=None, piS=None,
                     piL=None, n_outline=None, root_tol=1e-10):
    """
    BFSBL magnitudes and unresolved sky centroid (traced).

    Parameters
    ----------
    t : array_like, float, shape = (N_times,)
        Observation times (MJD).
    t0 : float, shape = ()
        Reference time (MJD).
    xS0 : array_like, float, shape = (2,)
        Primary source sky position at ``t0`` (arcsec).
    dxS : array_like, float, shape = (2,)
        Secondary minus primary source position (arcsec).
    xL0 : array_like, float, shape = (2,)
        Lens geometric-centre sky position at ``t0`` (arcsec).
    muS, muL : array_like, float, shape = (2,)
        Source and lens proper motions (mas/yr).
    thetaE_amp : float, shape = ()
        Einstein radius (mas).
    xL1_over_theta, xL2_over_theta : array_like, float, shape = (2,) or
        (N_times, 2)
        Lens offsets from the geometric centre (Einstein radii).
    m1, m2 : float, shape = ()
        Lens mass fractions.
    rho_pri, rho_sec : float, shape = ()
        Source radii (Einstein radii).
    mag_pri, mag_sec, b_sff : float, shape = ()
        Source magnitudes and source flux fraction.
    dmag_Lp_Ls : float, optional
        Primary-minus-secondary lens magnitude difference.
    gamma_pri, gamma_sec : float or None, optional
        Linear limb-darkening coefficients.
    parallax_vectors : array_like, float, shape = (N_times, 2), optional
        Parallax table; ``None`` skips parallax.
    piS, piL : float, optional
        Source and lens parallax (mas).
    n_outline, root_tol : optional
        See :func:`bfsbl_sky_model`.

    Returns
    -------
    mag : jnp.ndarray, float, shape = (N_times,)
        Model magnitudes.
    pos : jnp.ndarray, float, shape = (N_times, 2)
        East / North centroid (arcsec).
    """
    # Primary source and lens tracks; the secondary is a fixed sky offset.
    xS_pri, xL1, xL2 = fsbl_sky_tracks(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, xL1_over_theta,
        xL2_over_theta, parallax_vectors=parallax_vectors, piS=piS, piL=piL)
    xS_sec = xS_pri + dxS.reshape(1, 2)

    # Lensed magnifications and centroids of both sources.
    static = jnp.ndim(xL1_over_theta) == 1 and jnp.ndim(xL2_over_theta) == 1
    A1, A2, x1, x2 = bfsbl_sky_model(
        xS_pri, xS_sec, xL1, xL2, thetaE_amp, m1, m2, rho_pri, rho_sec,
        gamma_pri=gamma_pri, gamma_sec=gamma_sec, n_outline=n_outline,
        root_tol=root_tol, static_lens=static)

    # Blended photometry.
    mag = _two_source_photometry(A1, A2, mag_pri, mag_sec, b_sff)

    # Flux-weighted centroid of both sources and the luminous lenses.
    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    fL1, fL2 = resolved_lens_fluxes_jax(f1 + f2, b_sff, dmag_Lp_Ls)
    numer = (x1 * (f1 * A1)[:, None] + x2 * (f2 * A2)[:, None]
             + xL1 * fL1 + xL2 * fL2)
    denom = f1 * A1 + f2 * A2 + fL1 + fL2
    pos = numer / denom[:, None]

    return mag, pos


def bfsbl_log_likely_photometry(t, t0, xS0, dxS, xL0, muS, muL, thetaE_amp,
                                xL1_over_theta, xL2_over_theta, m1, m2,
                                rho_pri, rho_sec, mag_pri, mag_sec, b_sff,
                                mag_obs, mag_err, gamma_pri=None,
                                gamma_sec=None, parallax_vectors=None,
                                piS=None, piL=None, gp_params=None,
                                fixed_jitter=True, n_outline=None,
                                root_tol=1e-10):
    """
    BFSBL photometric log-likelihood (traced).

    Parameters
    ----------
    t, ..., b_sff : see :func:`bfsbl_photastrom`
        Forward-model arguments.
    mag_obs, mag_err : array_like, float, shape = (N_times,)
        Observed magnitudes and uncertainties.
    gamma_pri, gamma_sec, parallax_vectors, piS, piL : optional
        See :func:`bfsbl_photastrom`.
    gp_params, fixed_jitter : optional
        GP likelihood inputs; ``gp_params=None`` is a plain Gaussian.
    n_outline, root_tol : optional
        See :func:`bfsbl_sky_model`.

    Returns
    -------
    lnL : jnp.ndarray, float, shape = ()
        Summed log-likelihood.
    """
    mag_model, _ = bfsbl_photastrom(
        t, t0, xS0, dxS, xL0, muS, muL, thetaE_amp, xL1_over_theta,
        xL2_over_theta, m1, m2, rho_pri, rho_sec, mag_pri, mag_sec, b_sff,
        gamma_pri=gamma_pri, gamma_sec=gamma_sec,
        parallax_vectors=parallax_vectors, piS=piS, piL=piL,
        n_outline=n_outline, root_tol=root_tol)

    return _phot_log_likelihood(t, mag_model, mag_obs, mag_err,
                                gp_params=gp_params, fixed_jitter=fixed_jitter)


def bfsbl_log_likely_astrometry(t, t0, xS0, dxS, xL0, muS, muL, thetaE_amp,
                                xL1_over_theta, xL2_over_theta, m1, m2,
                                rho_pri, rho_sec, mag_pri, mag_sec, b_sff,
                                x_obs, y_obs, x_err, y_err, dmag_Lp_Ls=20.0,
                                gamma_pri=None, gamma_sec=None,
                                parallax_vectors=None, piS=None, piL=None,
                                n_outline=None, root_tol=1e-10):
    """
    BFSBL astrometric log-likelihood (traced).

    Parameters
    ----------
    t, ..., b_sff : see :func:`bfsbl_photastrom`
        Forward-model arguments.
    x_obs, y_obs, x_err, y_err : array_like, float, shape = (N_times,)
        Observed East / North positions and uncertainties (arcsec).
    dmag_Lp_Ls, gamma_pri, gamma_sec, parallax_vectors, piS, piL : optional
        See :func:`bfsbl_photastrom`.
    n_outline, root_tol : optional
        See :func:`bfsbl_sky_model`.

    Returns
    -------
    lnL : jnp.ndarray, float, shape = ()
        Summed Gaussian astrometric log-likelihood.
    """
    _, pos = bfsbl_photastrom(
        t, t0, xS0, dxS, xL0, muS, muL, thetaE_amp, xL1_over_theta,
        xL2_over_theta, m1, m2, rho_pri, rho_sec, mag_pri, mag_sec, b_sff,
        dmag_Lp_Ls=dmag_Lp_Ls, gamma_pri=gamma_pri, gamma_sec=gamma_sec,
        parallax_vectors=parallax_vectors, piS=piS, piL=piL,
        n_outline=n_outline, root_tol=root_tol)

    return gaussian_astrometry_log_likelihood_sum(pos, x_obs, y_obs, x_err,
                                                  y_err)
