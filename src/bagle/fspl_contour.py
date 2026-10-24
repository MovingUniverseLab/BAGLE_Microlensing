"""
Point-lens finite-source contour integrals with analytic limb derivatives.

Shared by the NumPy (njx) FSPL paths in :mod:`bagle.model` and
:mod:`bagle.model_jax`. The source limb is sampled at angles ``theta`` (uniform
or adaptive), each limb point is mapped to its plus and minus images, and the
image areas (Bozza et al. 2021, Eqs. 9-10) and centroid moments (Eqs. 19-22)
are summed over the sampled contours. The parabolic corrections use the exact
image derivative ``z'`` and curvature wedge ``x' ^ x''`` (Eq. 14) obtained from
the lens equation as in Bozza (2010) and VBMicrolensing, together with the
actual angle step of each segment.

All positions are relative to the lens, in Einstein radii, as complex numbers
``East + i North``.
"""
import numpy as np

# Floor on |zeta| (limb point on the lens) and |J| (image on the critical
# curve) so the lens map and its derivatives stay finite.
_EPS = 1e-15


def point_lens_images(zeta):
    """
    Plus and minus images of source points for a point lens at the origin.

    Parameters
    ----------
    zeta : array_like, complex
        Source positions relative to the lens (Einstein radii).

    Returns
    -------
    z_plus, z_minus : ndarray, complex
        Image positions, same shape as ``zeta``.
    """
    zeta = np.asarray(zeta, dtype=complex)
    zeta = np.where(np.abs(zeta) < _EPS, _EPS + 0j, zeta)

    # Lens equation zeta = z - 1 / conj(z).
    z_plus = zeta * (1.0 + np.sqrt(1.0 + 4.0 / np.abs(zeta) ** 2)) / 2.0
    z_minus = -1.0 / np.conjugate(z_plus)

    return z_plus, z_minus


def point_lens_image_derivatives(z, dzeta, rho):
    """
    Image derivative and curvature wedge along a circular source limb.

    Differentiates the lens equation ``zeta = z - 1/conj(z)`` with respect to
    the limb angle and solves for ``z'``. The wedge ``x' ^ x''`` follows from
    Bozza et al. (2021) Eq. 14 with ``d^2 conj(zeta) / dz^2 = -2 / z^3``,
    which avoids computing ``z''``.

    Parameters
    ----------
    z : array_like, complex
        Image positions relative to the lens.
    dzeta : array_like, complex
        Limb derivative ``i rho e^{i theta}`` of the source point.
    rho : float
        Source radius (``zeta' ^ zeta'' = rho^2`` on a circle).

    Returns
    -------
    dz : ndarray, complex
        ``dz/dtheta``, same shape as ``z``.
    wedge : ndarray, float
        ``x' ^ x'' = Re(z') Im(z'') - Im(z') Re(z'')``.
    """
    zb = np.conjugate(z)

    # Products instead of complex ** (which goes through exp/log).
    inv_zb = 1.0 / zb
    abs2 = z.real * z.real + z.imag * z.imag

    # Jacobian determinant of the lens map at the image.
    jac = 1.0 - 1.0 / (abs2 * abs2)
    jac = np.where(np.abs(jac) < _EPS, _EPS, jac)

    # zeta' = z' + conj(z')/zb^2 and its conjugate, solved for z'.
    dz = (dzeta - np.conjugate(dzeta) * inv_zb * inv_zb) / jac

    # Eq. 14: x' ^ x'' = [rho^2 + Im(z'^2 zeta' d2)] / J with d2 = -2/z^3.
    wedge = (rho * rho + (dz * dz * dzeta * (-2.0 / (z * z * z))).imag) / jac

    return dz, wedge


def contour_area_moments(z, dz, wp, step, counts, parabolic=None):
    """
    Area and centroid moments of closed sampled contours (parabolic order).

    Parameters
    ----------
    z, dz : ndarray, complex, shape (N_times, N_pts)
        Contour samples and their first derivatives with respect to the limb
        angle. Slots at and beyond ``counts`` are padding.
    wp : ndarray, shape (N_times, N_pts)
        Curvature wedge ``x' ^ x''`` at each sample.
    step : ndarray, shape (N_times, N_pts)
        Limb-angle step from each sample to the next (closing segment
        included).
    counts : ndarray of int, shape (N_times,)
        Number of valid samples per epoch.
    parabolic : ndarray of bool, shape (N_times,), optional
        Epochs that get the parabolic corrections (default: all). Epochs set
        to False use the trapezoid sums only.

    Returns
    -------
    area, mom_x, mom_y : ndarray, shape (N_times,)
        ``oint x dy``, ``1/2 oint x^2 dy`` and ``-1/2 oint y^2 dx`` (positive
        area for counter-clockwise contours). The centroid is
        ``(mom_x, mom_y) / area`` for either traversal direction, and shifts
        exactly with a translation of the contour.

    Notes
    -----
    Trapezoid sums over each segment plus the parabolic corrections
    ``(1/24) [f_i + f_{i+1}] dtheta^3`` with ``f = x' ^ x''`` for the area
    (Bozza et al. 2021 Eq. 10) and ``f = x'^2 y'/2 + x (x' ^ x'')`` and
    ``-(y'^2 x'/2 - y (x' ^ x''))`` for the moments. The factor 1/2 on the
    ``x'^2 y'`` terms makes the moments converge at the same O(dtheta^4) rate
    as the area; with coefficient 1 (Eqs. 21-22 as implemented previously)
    they are only O(dtheta^2) and not rotation invariant.
    """
    n_times, n_pts = z.shape
    k = np.arange(n_pts)[None, :]
    rows = np.arange(n_times)
    last = counts - 1

    # Padding is masked; the last valid sample closes onto the first.
    valid = k < counts[:, None]

    def take_next(arr):
        out = np.roll(arr, -1, axis=1)
        out[rows, last] = arr[:, 0]
        return out

    # Real components of the samples and their first derivatives.
    x, y = z.real, z.imag
    xd, yd = dz.real, dz.imag

    # Per-sample parabolic integrands (area, x-moment, y-moment). Each
    # segment uses them at both ends.
    f_a = wp
    f_x = 0.5 * xd * xd * yd + x * wp
    f_y = 0.5 * yd * yd * xd - y * wp

    # Next sample along the contour.
    xn, yn = take_next(x), take_next(y)

    # Parabolic weights dtheta^3 / 24 and trapezoid weights; zero on padding.
    use_parab = valid if parabolic is None else valid & parabolic[:, None]
    h3 = np.where(use_parab, step ** 3 / 24.0, 0.0)
    w = valid.astype(float)

    # Trapezoid terms (Eqs. 9 and 19).
    sx = x + xn
    sy = y + yn
    dy_seg = w * (yn - y)
    dx_seg = w * (xn - x)
    area = 0.5 * np.sum(sx * dy_seg, axis=1)
    mom_x = np.sum(sx * sx * dy_seg, axis=1) / 8.0
    mom_y = -np.sum(sy * sy * dx_seg, axis=1) / 8.0

    # Parabolic corrections at both ends of each segment.
    area += np.sum(h3 * (f_a + take_next(f_a)), axis=1)
    mom_x += np.sum(h3 * (f_x + take_next(f_x)), axis=1)
    mom_y -= np.sum(h3 * (f_y + take_next(f_y)), axis=1)

    return area, mom_x, mom_y


def _take_next(arr, rows, last):
    """
    Next sample along each closed, padded contour.

    Parameters
    ----------
    arr : ndarray, shape (N_times, N_pts)
    rows : ndarray of int, shape (N_times,)
        ``arange(N_times)``.
    last : ndarray of int, shape (N_times,)
        Index of the last valid sample per epoch.

    Returns
    -------
    out : ndarray, shape (N_times, N_pts)
        ``arr`` shifted by one sample, with the last valid sample wrapping to
        the first.
    """
    out = np.roll(arr, -1, axis=1)
    out[rows, last] = arr[:, 0]

    return out


def _limb_terms(zeta_c, rho, theta, counts, errors=False):
    """
    Images, signed magnifications and (optionally) per-segment errors.

    Parameters
    ----------
    zeta_c : ndarray, complex, shape (N_times,)
        Source centre relative to the lens (Einstein radii).
    rho : float or ndarray, shape (N_times,)
        Source radius in Einstein radii (one value or one per row).
    theta : ndarray, shape (N_times, N_pts)
        Increasing limb angles; slots at and beyond ``counts`` are padding.
    counts : ndarray of int, shape (N_times,)
        Number of valid limb samples per epoch.
    errors : bool, optional
        Also return the per-segment error estimates used for refinement.

    Returns
    -------
    images : ndarray, shape (N_times, 2, 2)
        Plus / minus image centroids relative to the lens.
    amps : ndarray, shape (N_times, 2)
        Signed magnifications ``[+, -]``.
    seg_err : ndarray, shape (N_times, N_pts)
        Only if ``errors``: estimated area error of each segment (summed
        over both images, weighted by the lever arm ``max(1, |z|)`` so it
        also bounds the moment error). ``inf`` where a sample lies within
        one of its own steps of the lens.
    abs_area : ndarray, shape (N_times,)
        Only if ``errors``: ``|A_+| + |A_-|`` (raw image areas).
    step : ndarray, shape (N_times, N_pts)
        Only if ``errors``: limb-angle step from each sample to the next.
    """
    n_times, n_pts = theta.shape
    k = np.arange(n_pts)[None, :]
    rows = np.arange(n_times)
    last = counts - 1
    valid = k < counts[:, None]

    # One radius per row (column vector for broadcasting over samples).
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (n_times,))
    rho_c = rho[:, None]

    # Padding repeats the first angle so every slot maps to a finite image.
    theta = np.where(valid, theta, theta[:, :1])

    # Angle step to the next sample; the last valid one closes the contour.
    theta_next = np.where(k < counts[:, None] - 1,
                          np.roll(theta, -1, axis=1),
                          theta[:, :1] + 2.0 * np.pi)
    step = theta_next - theta

    # Source limb and its derivative with respect to theta.
    e_it = np.exp(1j * theta)
    zeta = zeta_c[:, None] + rho_c * e_it
    dzeta = 1j * rho_c * e_it

    # The parabolic expansion fails when a limb sample lies within one of
    # its own steps of the lens (|J| -> 0 there); such epochs use the
    # trapezoid sums (and are refined first by the adaptive sampler).
    close = valid & (np.abs(zeta) < rho_c * step)
    parabolic = ~np.any(close, axis=1)

    images = np.zeros((n_times, 2, 2), dtype=float)
    amps = np.zeros((n_times, 2), dtype=float)
    src_area = np.pi * rho ** 2
    if errors:
        seg_err = np.zeros((n_times, n_pts), dtype=float)
        abs_area = np.zeros(n_times, dtype=float)

    # Plus image has parity +1, minus image parity -1.
    for i_img, (z, parity) in enumerate(zip(point_lens_images(zeta), (1, -1))):
        dz, wp = point_lens_image_derivatives(z, dzeta, rho_c)
        area, mom_x, mom_y = contour_area_moments(z, dz, wp, step, counts,
                                                  parabolic=parabolic)

        images[:, i_img, 0] = mom_x / area
        images[:, i_img, 1] = mom_y / area
        amps[:, i_img] = parity * area / src_area

        if errors:
            # Segment error estimates (cf. Bozza 2010): change of x' ^ x''
            # across the segment (parabolic truncation) and mismatch between
            # the chord and the mean end tangent (non-parabolic arcs).
            zn = _take_next(z, rows, last)
            dzn = _take_next(dz, rows, last)
            wpn = _take_next(wp, rows, last)
            chord = zn - z
            e_curv = step ** 3 / 24.0 * np.abs(wpn - wp)
            e_chord = np.abs(chord - 0.5 * step * (dz + dzn)) * np.abs(chord)
            lever = np.maximum(1.0, np.abs(0.5 * (z + zn)))
            seg_err += np.where(valid, (e_curv + e_chord) * lever, 0.0)
            abs_area += np.abs(area)

    if not errors:
        return images, amps

    # Force refinement where a sample sits within its own step of the lens.
    seg_err = np.where(close, np.inf, seg_err)

    return images, amps, seg_err, abs_area, step


def adaptive_limb_images_amps(zeta_c, rho, n_init=64, tol=1e-4, max_pts=2048,
                              max_iter=40):
    """
    FSPL images and magnifications with error-controlled limb sampling.

    Starts from ``n_init`` equally spaced limb points and repeatedly bisects
    every segment whose error estimate exceeds its share of the budget
    ``tol * (|A_+| + |A_-|)``, until each epoch meets the budget, reaches
    ``max_pts`` samples, or ``max_iter`` passes. Only unconverged epochs are
    recomputed. Refinement concentrates where the limb passes close to the
    lens, so the lens crossing the limb is resolved without oversampling the
    rest of the contour.

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_times,)
        Source centre relative to the lens (Einstein radii).
    rho : float or array_like, shape (N_times,)
        Source radius in Einstein radii (one value or one per row).
    n_init : int, optional
        Initial number of uniform limb samples.
    tol : float, optional
        Relative error budget on the image areas. The estimate is
        conservative: the achieved magnification and centroid errors are
        typically 10-100 times smaller.
    max_pts : int, optional
        Maximum number of limb samples per epoch.
    max_iter : int, optional
        Maximum number of refinement passes.

    Returns
    -------
    images : ndarray, shape (N_times, 2, 2)
        Plus / minus image centroids relative to the lens (Einstein radii).
    amps : ndarray, shape (N_times, 2)
        Signed magnifications ``[+, -]``.
    """
    zeta_c = np.atleast_1d(np.asarray(zeta_c, dtype=complex)).reshape(-1)
    n_times = zeta_c.size
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (n_times,))
    images = np.zeros((n_times, 2, 2), dtype=float)
    amps = np.zeros((n_times, 2), dtype=float)
    if n_times == 0:
        return images, amps

    # A limb point exactly on the lens maps to the whole Einstein ring and
    # no finite sampling resolves it. Move |zeta_c| off rho by a relative
    # 1e-9 (the magnification and centroid are continuous in |u|).
    nudge = 1e-9 * rho
    abs_c = np.abs(zeta_c)
    on_limb = np.abs(abs_c - rho) < nudge
    if np.any(on_limb):
        dir_c = np.where(abs_c > 0, zeta_c / np.where(abs_c > 0, abs_c, 1.0),
                         1.0 + 0j)
        zeta_c = np.where(on_limb, (rho + nudge) * dir_c, zeta_c)

    # Sorted limb angles per epoch; +inf marks unused slots.
    theta = np.full((n_times, max_pts), np.inf)
    theta[:, :n_init] = 2.0 * np.pi * np.arange(n_init) / n_init
    counts = np.full(n_times, n_init)
    active = np.arange(n_times)

    for n_iter in range(max_iter + 1):
        # Only unconverged epochs, trimmed to their current sample count.
        width = int(counts[active].max())
        th_a = theta[active, :width]
        c_a = counts[active]
        img, amp, err, abs_area, step = _limb_terms(
            zeta_c[active], rho[active], th_a, c_a, errors=True)
        images[active] = img
        amps[active] = amp

        # Epochs still over budget and with room to refine.
        need = ((np.sum(err, axis=1) > tol * abs_area) & (c_a < max_pts)
                & (n_iter < max_iter))
        if not np.any(need):
            break

        # Bisect every segment above an equal share of the budget.
        k = np.arange(width)[None, :]
        share = tol * abs_area / c_a
        flag = need[:, None] & (k < c_a[:, None]) & (err > share[:, None])

        # Respect capacity by keeping the largest-error segments.
        rows = np.arange(active.size)
        room = max_pts - c_a
        order = np.argsort(-np.where(flag, err, -1.0), axis=1)
        ranks = np.empty_like(order)
        ranks[rows[:, None], order] = np.arange(width)[None, :]
        flag &= ranks < room[:, None]

        # Insert segment midpoints (the closing segment wraps past 2 pi).
        th_valid = np.where(np.isfinite(th_a), th_a, 0.0)
        new = np.where(flag, np.mod(th_valid + 0.5 * step, 2.0 * np.pi),
                       np.inf)
        merged = np.sort(np.concatenate([th_a, new], axis=1), axis=1)
        n_new = flag.sum(axis=1)
        new_width = min(max_pts, width + int(n_new.max()))
        theta[active, :new_width] = merged[:, :new_width]
        counts[active] = c_a + n_new
        active = active[need]

    return images, amps


def limb_images_amps(zeta_c, rho, theta, counts):
    """
    FSPL plus/minus image centroids and signed magnifications.

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_times,)
        Source centre relative to the lens (Einstein radii).
    rho : float or array_like, shape (N_times,)
        Source radius in Einstein radii (one value or one per row).
    theta : array_like, shape (N_times, N_pts)
        Increasing limb angles in ``[0, 2 pi)`` for each epoch. Slots at and
        beyond ``counts`` are padding (any value).
    counts : array_like of int, shape (N_times,)
        Number of valid limb samples per epoch.

    Returns
    -------
    images : ndarray, shape (N_times, 2, 2)
        Plus / minus image centroids ``[+, -] x [E, N]`` relative to the
        lens (Einstein radii).
    amps : ndarray, shape (N_times, 2)
        Signed magnifications ``[+, -]``. The minus entry is negative when the
        lens is inside the source (the hole of the ring image); the total
        magnification is their sum.
    """
    zeta_c = np.asarray(zeta_c, dtype=complex).reshape(-1)
    theta = np.asarray(theta, dtype=float)
    counts = np.asarray(counts, dtype=int).reshape(-1)

    return _limb_terms(zeta_c, rho, theta, counts)


def uniform_limb_images_amps(zeta_c, rho, n_outline):
    """
    FSPL images and magnifications on ``n_outline`` equally spaced limb points.

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_times,)
        Source centre relative to the lens (Einstein radii).
    rho : float or array_like, shape (N_times,)
        Source radius in Einstein radii (one value or one per row).
    n_outline : int
        Number of limb samples.

    Returns
    -------
    images : ndarray, shape (N_times, 2, 2)
        Plus / minus image centroids relative to the lens (Einstein radii).
    amps : ndarray, shape (N_times, 2)
        Signed magnifications ``[+, -]``.
    """
    zeta_c = np.atleast_1d(np.asarray(zeta_c, dtype=complex)).reshape(-1)
    n_pts = int(n_outline)

    # Same uniform limb angles for every epoch.
    theta = np.broadcast_to(2.0 * np.pi * np.arange(n_pts) / n_pts,
                            (zeta_c.size, n_pts))
    counts = np.full(zeta_c.size, n_pts)

    return limb_images_amps(zeta_c, rho, theta, counts)


def gamma_to_u(gamma):
    """
    Linear limb-darkening coefficient: Gamma convention to u convention.

    ``I(mu) / I(1) = 1 - u (1 - mu)`` with ``u = 3 Gamma / (2 + Gamma)``
    (An et al. 2002). VBMicrolensing's ``a1`` is ``u``.

    Parameters
    ----------
    gamma : float or array_like
        Flux-conserving coefficient Gamma.

    Returns
    -------
    u : float or ndarray
        Central-intensity coefficient u.
    """
    gamma = np.asarray(gamma, dtype=float)

    return 3.0 * gamma / (2.0 + gamma)


def ld_radius_nodes(u_amp, rho, n_ld=8):
    """
    Quadrature nodes in ``s = sqrt(1 - r^2/rho^2)`` for linear limb darkening.

    The Gamma-law magnified flux is
    ``(1 - Gamma) F(rho) + (3/2) Gamma * int_0^1 F(rho sqrt(1 - s^2)) ds``
    with ``F(r) = pi r^2 A(u, r)`` the magnified flux of a uniform disk of
    radius ``r``. The integrand has a weak (logarithmic) singularity where
    the disk edge crosses the lens, ``s* = sqrt(1 - u^2/rho^2)`` for
    ``u < rho``. Each epoch uses two sub-intervals ``[0, s*]`` and
    ``[s*, 1]`` with Gauss-Legendre nodes graded quadratically toward ``s*``
    (``s* = 1/2`` when the lens is outside the source, where the integrand
    is smooth), so the node count is the same for every epoch.

    Parameters
    ----------
    u_amp : array_like, shape (N_times,)
        Source-lens separation (Einstein radii).
    rho : float or array_like, shape (N_times,)
        Source radius (Einstein radii).
    n_ld : int, optional
        Gauss-Legendre nodes per sub-interval.

    Returns
    -------
    s : ndarray, shape (N_times, 2 * n_ld)
        Nodes in ``s``.
    w : ndarray, shape (N_times, 2 * n_ld)
        Quadrature weights (sum to 1 per epoch).
    """
    u_amp = np.atleast_1d(np.asarray(u_amp, dtype=float))
    rho = np.broadcast_to(np.asarray(rho, dtype=float), u_amp.shape)

    # Gauss-Legendre on tau in [0, 1].
    x, w = np.polynomial.legendre.leggauss(int(n_ld))
    tau = 0.5 * (x + 1.0)
    w_tau = 0.5 * w

    # Kink location; s* = 1/2 (no kink) when the lens is outside the disk.
    inside = u_amp < rho
    q2 = np.where(inside, (u_amp / rho) ** 2, 0.75)
    s_star = np.sqrt(1.0 - q2)[:, None]

    # s = s* -/+ (interval length) * tau^2, graded toward s*.
    s_lo = s_star * (1.0 - tau ** 2)
    w_lo = s_star * 2.0 * tau * w_tau
    s_up = s_star + (1.0 - s_star) * tau ** 2
    w_up = (1.0 - s_star) * 2.0 * tau * w_tau

    return (np.concatenate([s_lo, s_up], axis=1),
            np.concatenate([w_lo, w_up], axis=1))


def ld_images_amps(zeta_c, rho, gamma, disk_images_amps, n_ld=8):
    """
    Linear (Gamma-law) limb-darkened FSPL images and magnifications.

    The source surface brightness is
    ``I(r) proportional to 1 - Gamma (1 - (3/2) sqrt(1 - r^2/rho^2))``
    (Yoo et al. 2004), normalised to the uniform-source flux. The magnified
    flux and first moments of each image are integrated over concentric
    uniform disks (see :func:`ld_radius_nodes`), each computed by
    ``disk_images_amps``.

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_times,)
        Source centre relative to the lens (Einstein radii).
    rho : float
        Source radius in Einstein radii.
    gamma : float
        Linear limb-darkening coefficient (Gamma convention).
    disk_images_amps : callable
        ``(zeta_rows, rho_rows) -> (images, amps)`` for uniform disks, e.g.
        :func:`adaptive_limb_images_amps` or :func:`uniform_limb_images_amps`
        with their sampling arguments bound.
    n_ld : int, optional
        Gauss-Legendre nodes per sub-interval (``2 * n_ld + 1`` disks per
        epoch).

    Returns
    -------
    images : ndarray, shape (N_times, 2, 2)
        Brightness-weighted plus / minus image centroids relative to the
        lens (Einstein radii).
    amps : ndarray, shape (N_times, 2)
        Signed magnifications ``[+, -]``; the total magnification is their
        sum.
    """
    zeta_c = np.atleast_1d(np.asarray(zeta_c, dtype=complex)).reshape(-1)
    n_times = zeta_c.size

    # Disk radii per epoch: the full disk, then the quadrature nodes.
    s, w_s = ld_radius_nodes(np.abs(zeta_c), rho, n_ld)
    radii = np.concatenate([np.full((n_times, 1), rho),
                            rho * np.sqrt(1.0 - s ** 2)], axis=1)
    weights = np.concatenate([np.full((n_times, 1), 1.0 - gamma),
                              1.5 * gamma * w_s], axis=1)
    n_disk = radii.shape[1]

    # All (epoch, radius) uniform disks in one call.
    img, amp = disk_images_amps(np.repeat(zeta_c, n_disk), radii.ravel())
    img = img.reshape(n_times, n_disk, 2, 2)
    amp = amp.reshape(n_times, n_disk, 2)

    # Magnified flux (area units) and first moment of each image and disk.
    flux = np.pi * radii[:, :, None] ** 2 * amp
    mom = flux[:, :, :, None] * img

    # Brightness-weighted sums over the disks.
    flux_ld = np.sum(weights[:, :, None] * flux, axis=1)
    mom_ld = np.sum(weights[:, :, None, None] * mom, axis=1)

    return mom_ld / flux_ld[:, :, None], flux_ld / (np.pi * rho ** 2)
