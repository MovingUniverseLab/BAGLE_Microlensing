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
    step : float or jnp.ndarray, shape (N_times, n_outline)
        Limb-angle step from each sample to the next (the last one closes
        the contour).
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
    h3 = jnp.where(parabolic[:, None], step ** 3 / 24.0, 0.0)

    area = area + jnp.sum(h3 * (f_a + jnp.roll(f_a, -1, axis=1)), axis=1)
    mom_x = mom_x + jnp.sum(h3 * (f_x + jnp.roll(f_x, -1, axis=1)), axis=1)
    mom_y = mom_y - jnp.sum(h3 * (f_y + jnp.roll(f_y, -1, axis=1)), axis=1)

    return area, mom_x, mom_y


def _clustered_limb_angles(zeta_c, rho, n_outline, c_step=0.05, a_max=16.0,
                           n_iter=12):
    """
    Static-count limb angles clustered toward the lens.

    A uniform grid ``psi`` in ``(-pi, pi)`` (half-step offset, so no sample
    sits on the limb point nearest the lens) is mapped to
    ``phi = pi sinh(a psi / pi) / sinh(a)`` measured from that point. The
    smallest step is ``a / sinh(a)`` times the uniform step, so ``a`` is set
    per row to make it about ``c_step * d / rho``, with ``d`` the lens-limb
    distance; far from the limb ``a -> 0`` and the grid is uniform. This is
    a fixed-shape stand-in for the adaptive refinement of
    :func:`bagle.fspl_contour.adaptive_limb_images_amps`. The angles carry
    no gradient: the contour integral does not depend on the
    parameterisation of the limb.

    Parameters
    ----------
    zeta_c : jnp.ndarray, complex, shape (N_rows,)
        Disk centres relative to the lens (Einstein radii).
    rho : jnp.ndarray, shape (N_rows,)
        Disk radii (Einstein radii).
    n_outline : int
        Static limb sample count.
    c_step : float, optional
        Target smallest step in units of ``d / rho``.
    a_max : float, optional
        Cap on the clustering strength (largest step is about ``a_max``
        times the uniform step).
    n_iter : int, optional
        Fixed-point iterations for ``a / sinh(a) = kappa``.

    Returns
    -------
    theta : jnp.ndarray, shape (N_rows, n_outline)
        Increasing limb angles.
    step : jnp.ndarray, shape (N_rows, n_outline)
        Angle step from each sample to the next (the last one wraps).
    """
    zeta_c = jax.lax.stop_gradient(zeta_c)
    rho = jax.lax.stop_gradient(rho)

    # Required clustering a / sinh(a) = kappa from the lens-limb distance.
    dist = jnp.abs(jnp.abs(zeta_c) - rho)
    kappa = jnp.clip(c_step * n_outline * dist / (2.0 * jnp.pi * rho),
                     1e-12, 1.0)

    # Solve a = asinh(a / kappa) by fixed-point iteration (a ~ 0 if uniform).
    a = jnp.full_like(kappa, 3.0)
    for _ in range(n_iter):
        a = jnp.arcsinh(a / kappa)
    a = jnp.where(kappa > 0.95, 1e-3, jnp.clip(a, 1e-3, a_max))[:, None]

    # Uniform psi grid (half-step offset) mapped through the sinh law.
    psi = (-jnp.pi + 2.0 * jnp.pi * (jnp.arange(n_outline) + 0.5)
           / n_outline)[None, :]
    phi = jnp.pi * jnp.sinh(a * psi / jnp.pi) / jnp.sinh(a)

    # Steps to the next sample; the last closes the contour.
    step = jnp.concatenate([phi[:, 1:] - phi[:, :-1],
                            phi[:, :1] + 2.0 * jnp.pi - phi[:, -1:]], axis=1)

    # Measure phi from the limb point nearest the lens.
    theta = jnp.angle(-zeta_c)[:, None] + phi

    return theta, step


def _disk_images_amps(zeta_c, rho, n_outline):
    """
    Uniform-disk images and signed amplifications on a static limb.

    Parameters
    ----------
    zeta_c : jnp.ndarray, complex, shape (N_rows,)
        Disk centres relative to the lens (Einstein radii).
    rho : jnp.ndarray, shape (N_rows,)
        Disk radius of each row (Einstein radii).
    n_outline : int
        Static limb sample count (clustered toward the lens; see
        :func:`_clustered_limb_angles`).

    Returns
    -------
    images : jnp.ndarray, shape (N_rows, 2, 2)
        Image centroids ``[+, -]`` x ``[E, N]`` relative to the lens.
    amps : jnp.ndarray, shape (N_rows, 2)
        Signed image amplifications.
    """
    # A lens exactly on the limb maps to the whole Einstein ring. Move
    # |zeta_c| off rho by 1e-6 rho (about the closest approach the clustered
    # static limb resolves), keeping the side of the limb.
    abs_c = jnp.abs(zeta_c)
    nudge = 1e-6 * rho
    on_limb = jnp.abs(abs_c - rho) < nudge
    dir_c = zeta_c / jnp.where(abs_c > 0, abs_c, 1.0)
    abs_new = jnp.where(abs_c < rho, rho - nudge, rho + nudge)
    zeta_c = jnp.where(on_limb, abs_new * dir_c, zeta_c)

    # Radius as a column for broadcasting over limb samples.
    rho_c = rho[:, None]

    # Source limb (complex E + iN) and its limb-angle derivative.
    theta, step = _clustered_limb_angles(zeta_c, rho, n_outline)
    e_it = jnp.exp(1j * theta)
    zeta = zeta_c[:, None] + rho_c * e_it
    dzeta = 1j * rho_c * e_it

    # The parabolic expansion fails when a limb sample lies within one step
    # of the lens (|J| -> 0 there); those epochs use the trapezoid sums.
    parabolic = ~jnp.any(jnp.abs(zeta) < rho_c * step, axis=1)

    images = []
    amps = []
    src_area = jnp.pi * rho ** 2

    # Plus image has parity +1, minus image parity -1.
    for z, parity in zip(_point_lens_images(zeta), (1.0, -1.0)):
        dz, wedge = _image_derivative_wedge(z, dzeta, rho_c)
        area, mom_x, mom_y = _contour_area_moments(z, dz, wedge, step,
                                                   parabolic)
        images.append(jnp.stack([mom_x / area, mom_y / area], axis=-1))
        amps.append(parity * area / src_area)

    return jnp.stack(images, axis=1), jnp.stack(amps, axis=1)


def _ld_radius_nodes(u_amp, rho, n_ld):
    """
    Graded Gauss-Legendre nodes in ``s`` for linear limb darkening.

    JAX port of :func:`bagle.fspl_contour.ld_radius_nodes` (same nodes).

    Parameters
    ----------
    u_amp : jnp.ndarray, shape (N_times,)
        Source-lens separation (Einstein radii).
    rho : jnp.ndarray, scalar
        Source radius (Einstein radii).
    n_ld : int
        Gauss-Legendre nodes per sub-interval (static).

    Returns
    -------
    s, w : jnp.ndarray, shape (N_times, 2 * n_ld)
        Nodes in ``s = sqrt(1 - r^2/rho^2)`` and weights (sum to 1).
    """
    # Gauss-Legendre on tau in [0, 1] (trace-time constants).
    x, w = np.polynomial.legendre.leggauss(int(n_ld))
    tau = jnp.asarray(0.5 * (x + 1.0))
    w_tau = jnp.asarray(0.5 * w)

    # Kink where the disk edge crosses the lens; 1/2 when outside.
    inside = u_amp < rho
    q2 = jnp.where(inside, (u_amp / rho) ** 2, 0.75)
    s_star = jnp.sqrt(1.0 - q2)[:, None]

    # s = s* -/+ (interval length) * tau^2, graded toward s*.
    s_lo = s_star * (1.0 - tau ** 2)
    w_lo = s_star * 2.0 * tau * w_tau
    s_up = s_star + (1.0 - s_star) * tau ** 2
    w_up = (1.0 - s_star) * 2.0 * tau * w_tau

    return (jnp.concatenate([s_lo, s_up], axis=1),
            jnp.concatenate([w_lo, w_up], axis=1))


@partial(jax.jit, static_argnames=("n_outline", "n_ld"))
def fspl_images_amps(u, rho, n_outline, gamma=None, n_ld=8):
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
    gamma : float or None, optional
        Linear limb-darkening coefficient (Gamma convention). ``None`` is a
        uniform source (one disk per epoch); otherwise the images and
        amplifications are brightness-weighted sums over ``2 n_ld + 1``
        nested disks, as in :func:`bagle.fspl_contour.ld_images_amps`.
    n_ld : int, optional
        Gauss-Legendre nodes per radial sub-interval (JIT static).

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
    zeta_c = u[:, 0] + 1j * u[:, 1]
    n_times = u.shape[0]

    # Uniform source: one disk per epoch.
    if gamma is None:
        return _disk_images_amps(zeta_c, jnp.broadcast_to(rho, (n_times,)),
                                 n_outline)

    # Disk radii and brightness weights per epoch (full disk first).
    gamma = jnp.broadcast_to(jnp.asarray(gamma, dtype=jnp.float64),
                             (n_times,))[:, None]
    s, w_s = _ld_radius_nodes(jnp.abs(zeta_c), rho, n_ld)
    radii = jnp.concatenate([jnp.broadcast_to(rho, (n_times, 1)),
                             rho * jnp.sqrt(1.0 - s ** 2)], axis=1)
    weights = jnp.concatenate([1.0 - gamma, 1.5 * gamma * w_s], axis=1)
    n_disk = radii.shape[1]

    # All (epoch, radius) disks in one static-shape call.
    img, amp = _disk_images_amps(jnp.repeat(zeta_c, n_disk), radii.reshape(-1),
                                 n_outline)
    img = img.reshape(n_times, n_disk, 2, 2)
    amp = amp.reshape(n_times, n_disk, 2)

    # Magnified flux (area units) and first moments, summed over disks.
    flux = jnp.pi * radii[:, :, None] ** 2 * amp
    flux_ld = jnp.sum(weights[:, :, None] * flux, axis=1)
    mom_ld = jnp.sum((weights[:, :, None] * flux)[:, :, :, None] * img, axis=1)

    return mom_ld / flux_ld[:, :, None], flux_ld / (jnp.pi * rho ** 2)


@partial(jax.jit, static_argnames=("n_outline", "n_ld"))
def fspl_magnification_centroid(u, rho, n_outline, gamma=None, n_ld=8):
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
    gamma : float or None, optional
        Linear limb-darkening coefficient; ``None`` for a uniform source.
    n_ld : int, optional
        Radial Gauss-Legendre nodes per sub-interval (JIT static).

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
        Total finite-source magnification.
    centroid : jnp.ndarray, shape (N_times, 2)
        Flux-weighted image centroid relative to the lens (Einstein radii).
        Uses signed image weights, matching host unresolved astrometry.
    """
    images, amps = fspl_images_amps(u, rho, n_outline, gamma=gamma, n_ld=n_ld)

    # Total magnification is the signed image sum.
    mag = jnp.sum(amps, axis=1)

    # Unresolved centroid with signed image weights (the ring's hole is
    # subtracted when the lens is inside the source).
    centroid = jnp.sum(images * amps[:, :, None], axis=1) / mag[:, None]

    return mag, centroid


def fspl_magnification(u, rho, n_outline, gamma=None, n_ld=8):
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
    gamma : float or None, optional
        Linear limb-darkening coefficient; ``None`` for a uniform source.
    n_ld : int, optional
        Radial Gauss-Legendre nodes per sub-interval.

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
        Total finite-source magnification.
    """
    mag, _centroid = fspl_magnification_centroid(u, rho, n_outline,
                                                 gamma=gamma, n_ld=n_ld)

    return mag


def fspl_magnification_from_trajectory(
    t, t0, tE, u0, thetaE_hat, rho, n_outline,
    parallax_vectors=None, piE_E=None, piE_N=None, gamma=None, n_ld=8,
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
    gamma : float or None, optional
        Linear limb-darkening coefficient; ``None`` for a uniform source.
    n_ld : int, optional
        Radial Gauss-Legendre nodes per sub-interval.

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

    return fspl_magnification_centroid(u, rho, n_outline, gamma=gamma,
                                       n_ld=n_ld)


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

    FSPL_PhotAstrom returns ``(2, N_t)`` (image axis first), as in
    :meth:`bagle.model.FSPL_PhotAstrom.get_resolved_amplification`;
    BFSPL_PhotAstrom keeps the ``(N_t, source, image)`` layout of BSPL.

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
        return amp_arr

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
    parallax_vectors=None, piE_E=None, piE_N=None, gamma=None, n_ld=8,
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
    gamma : float or None, optional
        Linear limb-darkening coefficient; ``None`` for a uniform source.
    n_ld : int, optional
        Radial Gauss-Legendre nodes per sub-interval.

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
        gamma=gamma,
        n_ld=n_ld,
    )

    return pspl_photometry_from_amp(amp, mag_src, b_sff=b_sff)


def fspl_log_likely_photometry(
    t, t0, tE, u0, thetaE_hat, rho, n_outline, mag_src, b_sff,
    mag_obs, mag_err, parallax_vectors=None, piE_E=None, piE_N=None,
    gp_params=None, fixed_jitter=True, gamma=None, n_ld=8,
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
    gamma : float or None, optional
        Linear limb-darkening coefficient; ``None`` for a uniform source.
    n_ld : int, optional
        Radial Gauss-Legendre nodes per sub-interval.

    Returns
    -------
    lnL : float
        Summed log-likelihood.
    """
    mag_model = fspl_photometry(
        t, t0, tE, u0, thetaE_hat, rho, n_outline, mag_src, b_sff=b_sff,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N,
        gamma=gamma, n_ld=n_ld,
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
    parallax_vectors=None, piS=None, piL=None, pi_ref=None, gamma=None,
    n_ld=8,
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
    gamma : float or None, optional
        Linear limb-darkening coefficient; ``None`` for a uniform source.
    n_ld : int, optional
        Radial Gauss-Legendre nodes per sub-interval.

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
    images, amps = fspl_images_amps(u_vec, rho, n_outline, gamma=gamma,
                                    n_ld=n_ld)

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
    pi_ref=None, gamma=None, n_ld=8,
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
    gamma : float or None, optional
        Linear limb-darkening coefficient; ``None`` for a uniform source.
    n_ld : int, optional
        Radial Gauss-Legendre nodes per sub-interval.

    Returns
    -------
    lnL : float
        Summed Gaussian log-likelihood.
    """
    pos_model = fspl_astrometry(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, rho, n_outline, mag_src, b_sff,
        parallax_vectors=parallax_vectors, piS=piS, piL=piL, pi_ref=pi_ref,
        gamma=gamma, n_ld=n_ld,
    )

    return gaussian_astrometry_log_likelihood_sum(
        pos_model, x_obs, y_obs, x_err, y_err
    )
