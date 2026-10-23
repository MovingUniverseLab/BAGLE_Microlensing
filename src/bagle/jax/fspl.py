"""FSPL finite-source point-lens helpers (JAX Bozza contour + host wrappers).

The JAX kernel uses a static ``n_outline`` limb, closed-form point-lens
images, and Bozza et al. (2021) contour integrals (area, centroid, parabolic
corrections from the analytic limb derivatives, as in VBMicrolensing and
:mod:`bagle.fspl_contour`) as ``jnp`` reductions, ``vmap``-friendly over
epochs. Traced
``fspl_log_likely_*`` helpers support milestone (c) parameter-class wiring;
host NumPy wrappers remain for parity and notebooks.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax_physics import (
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    lens_flux_for_astrometry_jax,
    mag2flux_jax,
    pspl_amplification_from_u,
    pspl_photometry_from_amp,
    pspl_u,
)

# Singularity floor when a limb point sits on the lens (contour-artifact
# fix from host AMG ``c913b38`` / ``e1bc6ec``).
_U_EPS = 1e-15


def _point_lens_images(zeta, eps=_U_EPS):
    """
    Plus and minus images of complex source points (point lens at origin).

    Parameters
    ----------
    zeta : jnp.ndarray, complex
        Source positions relative to the lens (Einstein radii).
    eps : float, optional
        Floor on ``|zeta|`` so the lens map stays finite on the limb.

    Returns
    -------
    z_plus, z_minus : jnp.ndarray, complex
        Image positions, same shape as ``zeta``.
    """
    # Floor |zeta| at the lens singularity (keep the direction).
    abs_zeta = jnp.abs(zeta)
    zeta = jnp.where(abs_zeta < eps, eps + 0j, zeta)
    abs_zeta = jnp.maximum(abs_zeta, eps)

    # Lens equation zeta = z - 1 / conj(z).
    z_plus = zeta * (1.0 + jnp.sqrt(1.0 + 4.0 / abs_zeta ** 2)) / 2.0
    z_minus = -1.0 / jnp.conj(z_plus)

    return z_plus, z_minus


def _image_derivative_wedge(z, dzeta, rho, eps=_U_EPS):
    """
    Image derivative and curvature wedge along a circular source limb.

    JAX port of :func:`bagle.fspl_contour.point_lens_image_derivatives`.

    Parameters
    ----------
    z : jnp.ndarray, complex
        Image positions relative to the lens.
    dzeta : jnp.ndarray, complex
        Limb derivative ``i rho e^{i theta}``.
    rho : float
        Source radius in Einstein radii.
    eps : float, optional
        Floor on ``|J|`` at the critical curve.

    Returns
    -------
    dz : jnp.ndarray, complex
        ``dz/dtheta``.
    wedge : jnp.ndarray
        ``x' ^ x''`` (Bozza et al. 2021 Eq. 14 with d2 = -2/z^3).
    """
    inv_zb = 1.0 / jnp.conj(z)
    abs2 = z.real * z.real + z.imag * z.imag

    # Jacobian determinant with a floor at the critical curve.
    jac = 1.0 - 1.0 / (abs2 * abs2)
    jac = jnp.where(jnp.abs(jac) < eps, eps, jac)

    # zeta' = z' + conj(z')/zb^2 solved for z'.
    dz = (dzeta - jnp.conj(dzeta) * inv_zb * inv_zb) / jac

    # Eq. 14: x' ^ x'' = [rho^2 + Im(z'^2 zeta' (-2/z^3))] / J.
    wedge = (rho * rho + (dz * dz * dzeta * (-2.0 / (z * z * z))).imag) / jac

    return dz, wedge


def _contour_area_moments(z, dz, wedge, step, parabolic):
    """
    Area and centroid moments of closed, uniformly sampled contours.

    JAX port of :func:`bagle.fspl_contour.contour_area_moments` for a static
    limb (no padding). Trapezoid sums (Bozza et al. 2021 Eqs. 9, 19) plus
    parabolic corrections from the analytic derivatives (Eqs. 10, 21-22 with
    1/2 on the ``x'^2 y'`` terms).

    Parameters
    ----------
    z, dz : jnp.ndarray, complex, shape (N_times, n_outline)
        Contour samples and their limb-angle derivatives.
    wedge : jnp.ndarray, shape (N_times, n_outline)
        ``x' ^ x''`` at each sample.
    step : float
        Uniform limb-angle step ``2 pi / n_outline``.
    parabolic : jnp.ndarray of bool, shape (N_times,)
        Epochs that get the parabolic corrections; the rest use the
        trapezoid sums only.

    Returns
    -------
    area, mom_x, mom_y : jnp.ndarray, shape (N_times,)
        ``oint x dy``, ``1/2 oint x^2 dy`` and ``-1/2 oint y^2 dx``.
    """
    x, y = z.real, z.imag
    xd, yd = dz.real, dz.imag

    # Next sample on the closed contour.
    xn = jnp.roll(x, -1, axis=1)
    yn = jnp.roll(y, -1, axis=1)

    # Trapezoid terms.
    sx, sy = x + xn, y + yn
    area = 0.5 * jnp.sum(sx * (yn - y), axis=1)
    mom_x = jnp.sum(sx * sx * (yn - y), axis=1) / 8.0
    mom_y = -jnp.sum(sy * sy * (xn - x), axis=1) / 8.0

    # Parabolic integrands at each sample; every segment uses both ends.
    f_a = wedge
    f_x = 0.5 * xd * xd * yd + x * wedge
    f_y = 0.5 * yd * yd * xd - y * wedge
    h3 = jnp.where(parabolic, step ** 3 / 24.0, 0.0)

    area = area + h3 * jnp.sum(f_a + jnp.roll(f_a, -1, axis=1), axis=1)
    mom_x = mom_x + h3 * jnp.sum(f_x + jnp.roll(f_x, -1, axis=1), axis=1)
    mom_y = mom_y - h3 * jnp.sum(f_y + jnp.roll(f_y, -1, axis=1), axis=1)

    return area, mom_x, mom_y


@partial(jax.jit, static_argnames=("n_outline",))
def fspl_images_amps(u, rho, n_outline):
    """
    FSPL image centroids and signed amplifications (static limb).

    Parameters
    ----------
    u : array_like
        Source-center separations in Einstein radii, shape ``(N_times, 2)``
        or ``(2,)``.
    rho : float
        Source radius in Einstein radii.
    n_outline : int
        Static limb sample count (JIT static).

    Returns
    -------
    images : jnp.ndarray, shape (N_times, 2, 2)
        Image centroids ``[+, -]`` x ``[E, N]`` relative to the lens.
    amps : jnp.ndarray, shape (N_times, 2)
        Signed image amplifications. The minus entry is negative when the
        lens is inside the source (hole of the ring image); total
        magnification is their sum.
    """
    u = jnp.atleast_2d(jnp.asarray(u, dtype=jnp.float64))
    rho = jnp.asarray(rho, dtype=jnp.float64)

    # Source limb (complex E + iN) and its limb-angle derivative.
    step = 2.0 * jnp.pi / n_outline
    theta = jnp.arange(n_outline, dtype=jnp.float64) * step
    e_it = jnp.exp(1j * theta)[None, :]
    zeta = (u[:, 0] + 1j * u[:, 1])[:, None] + rho * e_it
    dzeta = 1j * rho * e_it

    # The parabolic expansion fails when a limb sample lies within one step
    # of the lens (|J| -> 0 there); those epochs use the trapezoid sums.
    parabolic = ~jnp.any(jnp.abs(zeta) < rho * step, axis=1)

    images = []
    amps = []
    src_area = jnp.pi * rho ** 2

    # Plus image has parity +1, minus image parity -1.
    for z, parity in zip(_point_lens_images(zeta), (1.0, -1.0)):
        dz, wedge = _image_derivative_wedge(z, dzeta, rho)
        area, mom_x, mom_y = _contour_area_moments(z, dz, wedge, step,
                                                   parabolic)
        images.append(jnp.stack([mom_x / area, mom_y / area], axis=-1))
        amps.append(parity * area / src_area)

    return jnp.stack(images, axis=1), jnp.stack(amps, axis=1)


@partial(jax.jit, static_argnames=("n_outline",))
def fspl_magnification_centroid(u, rho, n_outline):
    """
    FSPL total magnification and flux-weighted centroid (Einstein radii).

    Parameters
    ----------
    u : array_like
        Source-center separations in Einstein radii, shape ``(N_times, 2)``
        or ``(2,)``.
    rho : float
        Source radius in Einstein radii.
    n_outline : int
        Static limb sample count (JIT static).

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
        Total finite-source magnification.
    centroid : jnp.ndarray, shape (N_times, 2)
        Flux-weighted image centroid relative to the lens (Einstein radii).
        Uses signed image weights, matching host unresolved astrometry.
    """
    images, amps = fspl_images_amps(u, rho, n_outline)

    # Total magnification is the signed image sum.
    mag = jnp.sum(amps, axis=1)

    # Unresolved centroid with signed image weights (the ring's hole is
    # subtracted when the lens is inside the source).
    centroid = jnp.sum(images * amps[:, :, None], axis=1) / mag[:, None]

    return mag, centroid


def fspl_magnification(u, rho, n_outline):
    """
    FSPL total magnification from separation vectors.

    Parameters
    ----------
    u : array_like
        Source-center separations in Einstein radii, shape ``(N_times, 2)``
        or ``(2,)``.
    rho : float
        Source radius in Einstein radii.
    n_outline : int
        Static limb sample count.

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
        Total finite-source magnification.
    """
    mag, _centroid = fspl_magnification_centroid(u, rho, n_outline)

    return mag


def fspl_magnification_from_trajectory(
    t, t0, tE, u0, thetaE_hat, rho, n_outline,
    parallax_vectors=None, piE_E=None, piE_N=None,
):
    """
    FSPL magnification along a PSPL trajectory (uses ``pspl_u``).

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Closest-approach separation, shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along relative proper motion, shape ``(2,)``.
    rho : float
        Source radius in Einstein radii.
    n_outline : int
        Static limb sample count.
    parallax_vectors, piE_E, piE_N : optional
        Passed through to :func:`bagle.jax_physics.pspl_u`.

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
        Total finite-source magnification.
    centroid : jnp.ndarray, shape (N_times, 2)
        Flux-weighted centroid relative to the lens (Einstein radii).
    """
    # Reuse the PSPL Einstein-frame trajectory helper.
    u = pspl_u(
        t, t0, tE, u0, thetaE_hat,
        parallax_vectors=parallax_vectors,
        piE_E=piE_E,
        piE_N=piE_N,
    )

    return fspl_magnification_centroid(u, rho, n_outline)


def fspl_point_source_limit_mag(u):
    """
    Analytic PSPL magnification for tiny-``rho`` regression tests.

    Parameters
    ----------
    u : array_like
        Source-lens separations, shape ``(N_times, 2)``.

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
        Point-source magnification.
    """
    return pspl_amplification_from_u(u)


# ---------------------------------------------------------------------------
# Host-model wrappers (pre-traced likelihood path)
# ---------------------------------------------------------------------------


def fspl_photometry_from_model(model, t, filt_idx, pvec):
    """
    Host numpy photometry (finite-source).

    Parameters
    ----------
    model : object
        BAGLE FSPL model instance.
    t : array_like
        Times in MJD.
    filt_idx : int
        Filter index.
    pvec : array_like
        Unused parameter vector (API compatibility).

    Returns
    -------
    mag : ndarray
        Model photometry.
    """
    return np.asarray(model.get_photometry(t, filt_idx=filt_idx), dtype=np.float64)


def fspl_amplification_from_model(model, t, filt_idx, pvec):
    """
    Host numpy amplification (finite-source AMG / CI dispatch).

    Parameters
    ----------
    model : object
        BAGLE FSPL model instance.
    t : array_like
        Times in MJD.
    filt_idx : int
        Filter index.
    pvec : array_like
        Unused parameter vector (API compatibility).

    Returns
    -------
    amp : ndarray
        Total amplification.
    """
    return np.asarray(
        model.get_amplification(t, filt_idx=filt_idx), dtype=np.float64
    )


def fspl_astrometry_from_model(model, t, filt_idx, pvec):
    """
    Host numpy astrometry when the model provides it.

    Parameters
    ----------
    model : object
        BAGLE FSPL model instance.
    t : array_like
        Times in MJD.
    filt_idx : int
        Filter index.
    pvec : array_like
        Unused parameter vector (API compatibility).

    Returns
    -------
    ast : ndarray or None
        Astrometry in arcsec, or ``None`` if unsupported.
    """
    if not getattr(model, "astrometryFlag", False):
        return None

    return np.asarray(model.get_astrometry(t, filt_idx=filt_idx), dtype=np.float64)


def finite_source_image_positions(image_arr) -> np.ndarray:
    """
    Convert AMG complex image positions to ``(N_t, N_img, 2)`` float.

    Parameters
    ----------
    image_arr : array_like
        Host image array (complex or real).

    Returns
    -------
    img : ndarray
        Real East/North image positions.
    """
    img = np.asarray(image_arr)
    if np.iscomplexobj(img):
        return np.stack(
            [np.real(img), np.imag(img)], axis=-1
        ).astype(np.float64, copy=False)
    if getattr(img.dtype, "names", None) is not None:
        return img.view("(2,)float")

    return np.asarray(img, dtype=np.float64)


def fspl_resolved_astrometry_from_model(model, t, filt_idx, pvec):
    """
    Finite-source resolved image positions via host ``get_all_arrays``.

    Parameters
    ----------
    model : object
        BAGLE FSPL model instance.
    t : array_like
        Times in MJD.
    filt_idx : int
        Filter index.
    pvec : array_like
        Unused parameter vector (API compatibility).

    Returns
    -------
    img : ndarray, shape (2, N_t, 2)
        Plus/minus image positions (image axis first), matching
        ``PSPL.get_resolved_astrometry``.
    """
    out = model.get_all_arrays(t, filt_idx=filt_idx)
    if isinstance(out, tuple) and len(out) == 4:
        img_arr = out[0]
    else:
        img_arr, _amp_arr = out

    # AMG stores (N_t, 2, 2); public resolved API is (2, N_t, 2).
    return np.swapaxes(finite_source_image_positions(img_arr), 0, 1)


def fspl_resolved_amplification_from_model(model, t, filt_idx, pvec):
    """
    Finite-source resolved amplifications via host AMG ``get_all_arrays``.

    FSPL_PhotAstrom and BFSPL_PhotAstrom use different ``swapaxes``
    conventions in
    :meth:`bagle.model.FSPL_PhotAstrom.get_resolved_amplification`.

    Parameters
    ----------
    model : object
        BAGLE FSPL or BFSPL model instance.
    t : array_like
        Times in MJD.
    filt_idx : int
        Filter index.
    pvec : array_like
        Unused parameter vector (API compatibility).

    Returns
    -------
    amp : ndarray
        Resolved amplifications.
    """
    img_arr, amp_arr = model.get_all_arrays(t, filt_idx=filt_idx)
    amp_arr = np.asarray(amp_arr, dtype=np.float64)
    if type(model).__name__.startswith("BFSPL"):
        return np.swapaxes(amp_arr, 1, 2)

    return np.swapaxes(amp_arr, 0, 1)


def fspl_centroid_shift_from_model(model, t, filt_idx, pvec):
    """
    Finite-source centroid shift (mas) from AMG astrometry minus unlensed.

    Parameters
    ----------
    model : object
        BAGLE FSPL model instance.
    t : array_like
        Times in MJD.
    filt_idx : int
        Filter index.
    pvec : array_like
        Unused parameter vector (API compatibility).

    Returns
    -------
    shift : ndarray or None
        Centroid shift in mas, or ``None`` if astrometry is unsupported.
    """
    if not getattr(model, "astrometryFlag", False):
        return None

    ast = fspl_astrometry_from_model(model, t, filt_idx, pvec)
    unl = np.asarray(
        model.get_astrometry_unlensed(t, filt_idx=filt_idx), dtype=np.float64
    )

    return (ast - unl) * 1e3


# ---------------------------------------------------------------------------
# Traced photometry / astrometry (milestone c)
# ---------------------------------------------------------------------------


def fspl_photometry(
    t, t0, tE, u0, thetaE_hat, rho, n_outline, mag_src, b_sff=None,
    parallax_vectors=None, piE_E=None, piE_N=None,
):
    """
    Unresolved FSPL magnitudes along a PSPL trajectory.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Closest-approach separation, shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along relative proper motion, shape ``(2,)``.
    rho : float
        Source radius in Einstein radii.
    n_outline : int
        Static limb sample count (JIT-static).
    mag_src : float
        Unlensed source magnitude.
    b_sff : float or None, optional
        Source flux fraction.
    parallax_vectors, piE_E, piE_N : optional
        Passed through to :func:`fspl_magnification_from_trajectory`.

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
        Unresolved model magnitudes.
    """
    # Finite-source magnification along the Einstein-frame trajectory.
    amp, _centroid = fspl_magnification_from_trajectory(
        t, t0, tE, u0, thetaE_hat, rho, n_outline,
        parallax_vectors=parallax_vectors,
        piE_E=piE_E,
        piE_N=piE_N,
    )

    return pspl_photometry_from_amp(amp, mag_src, b_sff=b_sff)


def fspl_log_likely_photometry(
    t, t0, tE, u0, thetaE_hat, rho, n_outline, mag_src, b_sff,
    mag_obs, mag_err, parallax_vectors=None, piE_E=None, piE_N=None,
    gp_params=None, fixed_jitter=True,
):
    """
    FSPL photometric Gaussian log-likelihood (traced).

    Parameters
    ----------
    t, t0, tE, u0, thetaE_hat, rho, n_outline, mag_src, b_sff
        Forward-model arguments for :func:`fspl_photometry`.
    mag_obs, mag_err : array_like
        Observed magnitudes and uncertainties.
    parallax_vectors, piE_E, piE_N : optional
        Parallax inputs for the trajectory.
    gp_params : optional
        If set, use the GP photometric likelihood instead of i.i.d. Gaussian.
    fixed_jitter : bool, optional
        Passed to the GP path when ``gp_params`` is not None.

    Returns
    -------
    lnL : float
        Summed log-likelihood.
    """
    mag_model = fspl_photometry(
        t, t0, tE, u0, thetaE_hat, rho, n_outline, mag_src, b_sff=b_sff,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N,
    )

    if gp_params is None:
        lnL = gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)
    else:
        from bagle.jax.gp import gp_log_likely_photometry

        lnL = gp_log_likely_photometry(
            t, mag_obs, mag_err, mag_model, gp_params,
            fixed_jitter=fixed_jitter,
        )

    return lnL


def fspl_astrometry(
    t, t0, xS0, xL0, muS, muL, thetaE_amp, rho, n_outline, mag_src, b_sff,
    parallax_vectors=None, piS=None, piL=None, pi_ref=None,
):
    """
    FSPL flux-weighted unresolved sky centroid (arcsec).

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Reference time (MJD).
    xS0, xL0 : array_like
        Source and lens sky positions at ``t0`` (arcsec), shape ``(2,)``.
    muS, muL : array_like
        Proper motions (mas/yr), shape ``(2,)``.
    thetaE_amp : float
        Einstein radius (mas).
    rho : float
        Source radius in Einstein radii.
    n_outline : int
        Static limb sample count.
    mag_src : float
        Unlensed source magnitude (sets the source flux scale).
    b_sff : float
        Source flux fraction; lens flux is clipped at zero for astrometry.
    parallax_vectors, piS, piL, pi_ref : optional
        Parallax table and amplitudes (mas), matching PSPL astrometry.

    Returns
    -------
    pos : jnp.ndarray, shape (N_times, 2)
        Unresolved East/North centroid (arcsec).
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    dt = ((t - t0) / 365.25).reshape(-1, 1)

    # Unlensed source and lens tracks (arcsec); mu in mas/yr.
    xS = xS0.reshape(1, 2) + dt * muS.reshape(1, 2) * 1e-3
    xL = xL0.reshape(1, 2) + dt * muL.reshape(1, 2) * 1e-3

    if parallax_vectors is not None:
        pvec = jnp.asarray(parallax_vectors, dtype=jnp.float64)
        xS = xS + piS * pvec * 1e-3
        xL = xL + piL * pvec * 1e-3
        if pi_ref is not None:
            shift = jnp.asarray(pi_ref, dtype=jnp.float64) * pvec * 1e-3
            xS = xS + shift
            xL = xL + shift

    # Einstein-frame separation and finite-source images / amps.
    thetaE_as = thetaE_amp * 1e-3
    u_vec = (xS - xL) / thetaE_as
    images, amps = fspl_images_amps(u_vec, rho, n_outline)

    # Absolute image positions on the sky (arcsec).
    img_sky = xL[:, None, :] + images * thetaE_as

    # Lens flux clipped at zero; photometry keeps the unclipped blend.
    fS = mag2flux_jax(mag_src)
    fL = lens_flux_for_astrometry_jax(fS, b_sff)
    # Signed image weights (the ring's hole is subtracted when the lens is
    # inside the source).
    ftot = fL + jnp.sum(fS * amps, axis=1)

    # Flux-weighted centroid of lensed images plus luminous lens.
    pos = jnp.sum(img_sky * (fS * amps)[:, :, None], axis=1)
    pos = pos + fL * xL
    pos = pos / ftot[:, None]

    return pos


def fspl_log_likely_astrometry(
    t, t0, xS0, xL0, muS, muL, thetaE_amp, rho, n_outline, mag_src, b_sff,
    x_obs, y_obs, x_err, y_err, parallax_vectors=None, piS=None, piL=None,
    pi_ref=None,
):
    """
    FSPL absolute-astrometry Gaussian log-likelihood (traced).

    Parameters
    ----------
    t, t0, xS0, xL0, muS, muL, thetaE_amp, rho, n_outline, mag_src, b_sff
        Forward-model arguments for :func:`fspl_astrometry`.
    x_obs, y_obs, x_err, y_err : array_like
        Observed East/North positions and uncertainties (arcsec).
    parallax_vectors, piS, piL, pi_ref : optional
        Parallax inputs for the sky tracks.

    Returns
    -------
    lnL : float
        Summed Gaussian log-likelihood.
    """
    pos_model = fspl_astrometry(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, rho, n_outline, mag_src, b_sff,
        parallax_vectors=parallax_vectors, piS=piS, piL=piL, pi_ref=pi_ref,
    )

    return gaussian_astrometry_log_likelihood_sum(
        pos_model, x_obs, y_obs, x_err, y_err
    )
