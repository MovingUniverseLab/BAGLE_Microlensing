"""FSPL finite-source point-lens helpers (JAX Bozza contour + host wrappers).

The JAX kernel uses a static ``n_outline`` limb, closed-form point-lens
images, and Bozza et al. (2021) contour integrals (area, centroid, parabolic
correction) as ``jnp`` reductions, ``vmap``-friendly over epochs. Host
NumPy wrappers remain for model objects until traced likelihoods land in
milestone (c).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax_physics import pspl_amplification_from_u, pspl_u

# Singularity floor when a limb point sits on the lens (contour-artifact
# fix from host AMG ``c913b38`` / ``e1bc6ec``).
_U_EPS = 1e-15


def point_lens_images_from_u(u_vec, eps=_U_EPS):
    """
    Closed-form plus/minus point-lens image positions from separation ``u``.

    Parameters
    ----------
    u_vec : array_like
        Source-lens separation in Einstein radii. Trailing axis length 2
        (East, North). Leading batch dims are arbitrary.
    eps : float, optional
        Floor on ``|u|`` so the lens map stays finite on the limb.

    Returns
    -------
    u_plus, u_minus : jnp.ndarray
        Image positions relative to the lens, same shape as ``u_vec``.
    """
    u_vec = jnp.asarray(u_vec, dtype=jnp.float64)

    # Separation amplitude with a floor at the lens singularity.
    u_amp = jnp.linalg.norm(u_vec, axis=-1, keepdims=True)
    u_safe = jnp.maximum(u_amp, eps)
    u_hat = u_vec / u_safe

    # Standard point-lens image radii along ``u_hat``.
    sqrt_term = jnp.sqrt(u_safe ** 2 + 4.0)
    u_plus = 0.5 * (u_safe + sqrt_term) * u_hat
    u_minus = 0.5 * (u_safe - sqrt_term) * u_hat

    return u_plus, u_minus


def _wedge_product(aa, bb):
    """
    2-D wedge product ``a_x b_y - a_y b_x`` on the last axis.

    Parameters
    ----------
    aa, bb : jnp.ndarray
        Arrays with trailing shape ``(..., 2)``.

    Returns
    -------
    wp : jnp.ndarray
        Wedge product with the last axis removed.
    """
    return aa[..., 0] * bb[..., 1] - aa[..., 1] * bb[..., 0]


def bozza_contour_integrals(plus, minus, n_outline):
    """
    Bozza area and centroid integrals on closed plus/minus image contours.

    Matches host ``get_all_arrays_CI`` geometry (Green + parabolic terms)
    but keeps signed image areas like host ``amg_contour_integrals`` so the
    total magnification stays physical when ``|u| < rho``.

    Parameters
    ----------
    plus, minus : jnp.ndarray
        Image contour points in Einstein radii, shape
        ``(N_times, n_outline, 2)``.
    n_outline : int
        Static limb sample count (must match the contour axis).

    Returns
    -------
    images : jnp.ndarray, shape (N_times, 2, 2)
        Image centroids ``[+, -]`` x ``[E, N]`` in Einstein radii.
    amps : jnp.ndarray, shape (N_times, 2)
        Signed image amplifications ``[+, -]`` before source-area
        normalization (raw contour areas). Caller divides by
        ``pi * rho**2``.
    """
    # Close each contour by appending the first point.
    plus_c = jnp.concatenate([plus, plus[:, :1, :]], axis=1)
    minus_c = jnp.concatenate([minus, minus[:, :1, :]], axis=1)

    # First and second differences along the closed path.
    d1_plus = jnp.diff(plus_c, axis=1)
    d1_minus = jnp.diff(minus_c, axis=1)
    d2_plus = jnp.diff(
        jnp.concatenate([d1_plus, d1_plus[:, :1, :]], axis=1), axis=1
    )
    d2_minus = jnp.diff(
        jnp.concatenate([d1_minus, d1_minus[:, :1, :]], axis=1), axis=1
    )

    # Pairwise sums for trapezoidal Green terms (n_outline closed edges).
    b2_plus = plus_c[:, :-1, :] + plus_c[:, 1:, :]
    b2_minus = minus_c[:, :-1, :] + minus_c[:, 1:, :]

    # Uniform source-limb angle step. Bozza parabolic terms use contour
    # derivatives w.r.t. theta; convert raw diffs with /d_theta and /d_theta^2.
    # Using bare diffs (host CI) leaves an n-gon area bias that fails A-08
    # at rho <= 1e-8.
    d_theta = 2.0 * jnp.pi / float(n_outline)
    d_angles3 = jnp.full((n_outline,), d_theta ** 3, dtype=jnp.float64)

    z1_plus = d1_plus / d_theta
    z2_plus = d2_plus / (d_theta ** 2)
    z1_minus = d1_minus / d_theta
    z2_minus = d2_minus / (d_theta ** 2)

    # Wrap i+1 so every closed edge gets a parabolic contribution.
    z1_plus_ip1 = jnp.concatenate(
        [z1_plus[:, 1:, :], z1_plus[:, :1, :]], axis=1
    )
    z2_plus_ip1 = jnp.concatenate(
        [z2_plus[:, 1:, :], z2_plus[:, :1, :]], axis=1
    )
    z1_minus_ip1 = jnp.concatenate(
        [z1_minus[:, 1:, :], z1_minus[:, :1, :]], axis=1
    )
    z2_minus_ip1 = jnp.concatenate(
        [z2_minus[:, 1:, :], z2_minus[:, :1, :]], axis=1
    )

    # Parabolic wedge products (Eq. 10 Bozza 2021) on all n edges.
    wp_i_plus = _wedge_product(z1_plus, z2_plus)
    wp_ip1_plus = _wedge_product(z1_plus_ip1, z2_plus_ip1)
    wp_i_minus = _wedge_product(z1_minus, z2_minus)
    wp_ip1_minus = _wedge_product(z1_minus_ip1, z2_minus_ip1)

    # Contour start points for centroid parabolic terms (length n).
    pos_plus_i = plus_c[:, :-1, :]
    pos_plus_ip1 = plus_c[:, 1:, :]
    pos_minus_i = minus_c[:, :-1, :]
    pos_minus_ip1 = minus_c[:, 1:, :]

    # Eq. 9 areas (CI y*dx form).
    a_plus = -0.5 * jnp.sum(b2_plus[:, :, 1] * d1_plus[:, :, 0], axis=1)
    a_minus = 0.5 * jnp.sum(b2_minus[:, :, 1] * d1_minus[:, :, 0], axis=1)

    # Eq. 10 parabolic area corrections.
    a_plus = a_plus + (1.0 / 24.0) * jnp.sum(
        d_angles3 * (wp_i_plus + wp_ip1_plus), axis=1
    )
    a_minus = a_minus - (1.0 / 24.0) * jnp.sum(
        d_angles3 * (wp_i_minus + wp_ip1_minus), axis=1
    )

    # Eq. 19 centroids.
    c_plus_x = 0.125 * jnp.sum(b2_plus[:, :, 0] ** 2 * d1_plus[:, :, 1], axis=1)
    c_plus_y = -0.125 * jnp.sum(
        b2_plus[:, :, 1] ** 2 * d1_plus[:, :, 0], axis=1
    )
    c_minus_x = -0.125 * jnp.sum(
        b2_minus[:, :, 0] ** 2 * d1_minus[:, :, 1], axis=1
    )
    c_minus_y = 0.125 * jnp.sum(
        b2_minus[:, :, 1] ** 2 * d1_minus[:, :, 0], axis=1
    )

    # Eqs. 21–22 parabolic centroid corrections (theta-derivative form).
    c_plus_x = c_plus_x + (1.0 / 24.0) * jnp.sum(
        d_angles3 * (
            (z1_plus[:, :, 0] ** 2 * z1_plus[:, :, 1]
             + pos_plus_i[:, :, 0] * wp_i_plus)
            + (z1_plus_ip1[:, :, 0] ** 2 * z1_plus_ip1[:, :, 1]
               + pos_plus_ip1[:, :, 0] * wp_ip1_plus)
        ),
        axis=1,
    )
    c_plus_y = c_plus_y - (1.0 / 24.0) * jnp.sum(
        d_angles3 * (
            (z1_plus[:, :, 1] ** 2 * z1_plus[:, :, 0]
             + pos_plus_i[:, :, 1] * wp_i_plus)
            + (z1_plus_ip1[:, :, 1] ** 2 * z1_plus_ip1[:, :, 0]
               + pos_plus_ip1[:, :, 1] * wp_ip1_plus)
        ),
        axis=1,
    )
    c_minus_x = c_minus_x - (1.0 / 24.0) * jnp.sum(
        d_angles3 * (
            (z1_minus[:, :, 0] ** 2 * z1_minus[:, :, 1]
             + pos_minus_i[:, :, 0] * wp_i_minus)
            + (z1_minus_ip1[:, :, 0] ** 2 * z1_minus_ip1[:, :, 1]
               + pos_minus_ip1[:, :, 0] * wp_ip1_minus)
        ),
        axis=1,
    )
    c_minus_y = c_minus_y + (1.0 / 24.0) * jnp.sum(
        d_angles3 * (
            (z1_minus[:, :, 1] ** 2 * z1_minus[:, :, 0]
             + pos_minus_i[:, :, 1] * wp_i_minus)
            + (z1_minus_ip1[:, :, 1] ** 2 * z1_minus_ip1[:, :, 0]
               + pos_minus_ip1[:, :, 1] * wp_ip1_minus)
        ),
        axis=1,
    )

    # Image centroids; area magnitude in the denominator (host convention).
    abs_plus = jnp.abs(a_plus)
    abs_minus = jnp.abs(a_minus)
    img_plus = jnp.stack([c_plus_x / abs_plus, c_plus_y / abs_plus], axis=-1)
    img_minus = jnp.stack(
        [c_minus_x / abs_minus, c_minus_y / abs_minus], axis=-1
    )
    images = jnp.stack([img_plus, img_minus], axis=1)
    areas = jnp.stack([a_plus, a_minus], axis=1)

    return images, areas


def fspl_source_outline(u, rho, n_outline):
    """
    Static circular source limb in Einstein radii.

    Parameters
    ----------
    u : array_like
        Source-center separations, shape ``(N_times, 2)``.
    rho : float
        Source radius in Einstein radii.
    n_outline : int
        Number of equally spaced limb samples.

    Returns
    -------
    u_outline : jnp.ndarray, shape (N_times, n_outline, 2)
        Limb separations relative to the lens.
    """
    u = jnp.asarray(u, dtype=jnp.float64)
    u = jnp.atleast_2d(u)

    # Equally spaced angles on the source circumference.
    angles = (jnp.arange(n_outline, dtype=jnp.float64) / n_outline) * (
        2.0 * jnp.pi
    )
    dux = rho * jnp.cos(angles)
    duy = rho * jnp.sin(angles)
    outline = jnp.stack([dux, duy], axis=1)

    # Broadcast center + limb offset.
    u_outline = u[:, None, :] + outline[None, :, :]

    return u_outline


@partial(jax.jit, static_argnames=("n_outline",))
def fspl_images_amps(u, rho, n_outline):
    """
    FSPL image centroids and signed amplifications (static Bozza limb).

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
        Signed image amplifications; total mag is their sum (flipped if
        needed so the sum is non-negative).
    """
    u = jnp.asarray(u, dtype=jnp.float64)
    u = jnp.atleast_2d(u)
    rho = jnp.asarray(rho, dtype=jnp.float64)

    # Limb in the source plane, then closed-form point-lens images.
    u_outline = fspl_source_outline(u, rho, n_outline)
    plus, minus = point_lens_images_from_u(u_outline)

    # Bozza Green + parabolic integrals on the image contours.
    images, areas = bozza_contour_integrals(plus, minus, n_outline)

    # Normalize by source area; keep signs (AMG convention).
    inv_area = 1.0 / (jnp.pi * rho ** 2)
    amps = areas * inv_area

    # Flip both images if the epoch total is negative.
    tot = jnp.sum(amps, axis=1, keepdims=True)
    amps = jnp.where(tot < 0.0, -amps, amps)

    return images, amps


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
        Uses ``|amp|`` weights, matching host unresolved astrometry.
    """
    images, amps = fspl_images_amps(u, rho, n_outline)

    # Total magnification is the signed image sum.
    mag = jnp.sum(amps, axis=1)

    # Unresolved centroid with absolute image weights.
    abs_amps = jnp.abs(amps)
    weight = jnp.sum(abs_amps, axis=1)
    centroid = jnp.sum(
        images * abs_amps[:, :, None], axis=1
    ) / weight[:, None]

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
    img : ndarray
        Resolved image positions.
    """
    out = model.get_all_arrays(t, filt_idx=filt_idx)
    if isinstance(out, tuple) and len(out) == 4:
        img_arr = out[0]
    else:
        img_arr, _amp_arr = out

    return finite_source_image_positions(img_arr)


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
