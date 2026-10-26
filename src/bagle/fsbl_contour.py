"""
Binary-lens finite-source contour integrals (VBBL / Bozza style).

NumPy host / reference path for FSBL: sample the source limb, solve the
Witt quintic for image positions, match images between consecutive limb
samples, and accumulate signed image areas and first moments with
trapezoid + parabolic corrections (Bozza 2010; Bozza et al. 2021). Image
pair creation and annihilation use Gould & Gaucherel (1997) chords.

Lens positions and masses follow the VBMicrolensing convention: origin at
the centre of mass, ``m1 = 1/(1+q)`` at ``x = -s q/(1+q)``,
``m2 = q/(1+q)`` at ``x = +s/(1+q)``. All coordinates are complex
``East + i North`` in Einstein radii.

Limb darkening reuses :func:`bagle.fspl_contour.ld_images_amps` with a
uniform-disk callable bound to this integrator. The radial integral over
nested disks is split at the radii where a circle about the source centre
touches the caustic (the magnification has a derivative jump there), and
each piece gets smoothstep-graded Gauss-Legendre nodes.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from bagle import fspl_contour

# Floor on |J| and lens-equation residuals so derivatives stay finite.
_EPS = 1e-15
_ROOT_TOL = 1e-10
_N_IMG = 5

# Limb darkening: Gauss-Legendre nodes per radial interval, caustic-tangency
# split points per epoch, and critical-curve phase samples.
_N_LD = 8
_N_BRK = 2
_N_PHI = 512


def lens_positions_cm(s, q):
    """
    Binary lens positions and masses in the centre-of-mass frame.

    Parameters
    ----------
    s : float
        Projected separation in Einstein radii.
    q : float
        Mass ratio ``m2 / m1``.

    Returns
    -------
    z1, z2 : complex
        Primary and secondary positions (East + i North).
    m1, m2 : float
        Mass fractions with ``m1 + m2 = 1``.
    """
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    z1 = complex(-s * q / (1.0 + q), 0.0)
    z2 = complex(s / (1.0 + q), 0.0)

    return z1, z2, m1, m2


def kappa_jac(z, z1, z2, m1, m2):
    """
    Complex deflection derivative ``kappa`` and Jacobian determinant.

    ``kappa = sum_i m_i / (z - z_i)^2`` and ``J = 1 - |kappa|^2``.

    Parameters
    ----------
    z : array_like, complex
        Image position(s).
    z1, z2 : complex or array_like, complex
        Lens positions.
    m1, m2 : float or array_like
        Lens masses.

    Returns
    -------
    kappa : ndarray, complex
        Same shape as ``z``.
    jac : ndarray, float
        Jacobian determinant, same shape as ``z``.
    """
    z = np.asarray(z, dtype=complex)
    # Avoid 0/0 at the lens positions (unused / nan slots).
    d1 = z - z1
    d2 = z - z2
    d1 = np.where(np.isfinite(z) & (np.abs(d1) >= _EPS), d1, _EPS + 0j)
    d2 = np.where(np.isfinite(z) & (np.abs(d2) >= _EPS), d2, _EPS + 0j)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = m1 / d1 ** 2 + m2 / d2 ** 2
    kappa = np.where(np.isfinite(z), kappa, 0.0 + 0j)
    jac = 1.0 - np.abs(kappa) ** 2

    return kappa, jac


def kappa_prime(z, z1, z2, m1, m2):
    """
    Derivative ``d kappa / dz = sum_i -2 m_i / (z - z_i)^3``.

    Parameters
    ----------
    z : array_like, complex
        Image position(s).
    z1, z2 : complex or array_like, complex
        Lens positions.
    m1, m2 : float or array_like
        Lens masses.

    Returns
    -------
    kappa_p : ndarray, complex
        Same shape as ``z``.
    """
    z = np.asarray(z, dtype=complex)
    d1 = z - z1
    d2 = z - z2
    d1 = np.where(np.abs(d1) < _EPS, _EPS + 0j, d1)
    d2 = np.where(np.abs(d2) < _EPS, _EPS + 0j, d2)

    return -2.0 * m1 / d1 ** 3 - 2.0 * m2 / d2 ** 3


def lens_map(z, z1, z2, m1, m2):
    """
    Binary lens map ``zeta(z) = z - conj(sum m_i / (z - z_i))``.

    Parameters
    ----------
    z : array_like, complex
        Image position(s).
    z1, z2 : complex or array_like, complex
        Lens positions.
    m1, m2 : float or array_like
        Lens masses.

    Returns
    -------
    zeta : ndarray, complex
        Source position(s) corresponding to ``z``.
    """
    z = np.asarray(z, dtype=complex)
    alpha = m1 / (z - z1) + m2 / (z - z2)

    return z - np.conjugate(alpha)


def image_derivatives(z, dzeta, rho, z1, z2, m1, m2):
    """
    Image derivative and curvature wedge along a circular source limb.

    Parameters
    ----------
    z : array_like, complex
        Image positions.
    dzeta : array_like, complex
        Limb derivative ``i rho e^{i theta}``.
    rho : float or array_like
        Source radius (``zeta' ^ zeta'' = rho^2`` on a circle).
    z1, z2 : complex or array_like, complex
        Lens positions.
    m1, m2 : float or array_like
        Lens masses.

    Returns
    -------
    dz : ndarray, complex
        ``dz / dtheta``, same shape as ``z``.
    wedge : ndarray, float
        ``x' ^ x''`` (Bozza et al. 2021 Eq. 14).
    jac : ndarray, float
        Jacobian determinant at ``z``.
    """
    kappa, jac = kappa_jac(z, z1, z2, m1, m2)
    jac_safe = np.where(np.abs(jac) < _EPS, _EPS, jac)

    # z' = (zeta' - conj(kappa) conj(zeta')) / J.
    dz = (dzeta - np.conjugate(kappa) * np.conjugate(dzeta)) / jac_safe

    # Wedge from kappa' without forming z''.
    kap_p = kappa_prime(z, z1, z2, m1, m2)
    wedge = (rho * rho + (dz * dz * dzeta * kap_p).imag) / jac_safe

    return dz, wedge, jac


def _quintic_coeffs(w, z1, z2, m1, m2):
    """
    Witt quintic coefficients ``a5..a0`` (high degree to constant).

    Parameters
    ----------
    w : ndarray, complex, shape (N,)
        Source positions.
    z1, z2 : complex
        Lens positions (scalars).
    m1, m2 : float
        Lens masses.

    Returns
    -------
    coeffs : ndarray, complex, shape (N, 6)
        ``[a5, a4, a3, a2, a1, a0]`` per source.
    """
    w = np.asarray(w, dtype=complex).reshape(-1)
    wbar = np.conjugate(w)
    z1bar = np.conjugate(z1)
    z2bar = np.conjugate(z2)

    a5 = (wbar - z1bar) * (wbar - z2bar)
    a4 = (-((w + 2 * (z1 + z2)) * wbar ** 2) - m2 * z2bar
          - z1bar * (m1 + (w + 2 * (z1 + z2)) * z2bar)
          + wbar * (m1 + m2 + (w + 2 * (z1 + z2)) * (z1bar + z2bar)))
    a3 = ((z1 ** 2 + 4 * z1 * z2 + z2 ** 2 + 2 * w * (z1 + z2)) * wbar ** 2
          + (m1 * (w - z1) + m2 * (w + 2 * z1 + z2)) * z2bar
          + z1bar * (m2 * (w - z2) + m1 * (w + z1 + 2 * z2)
                     + (z1 ** 2 + 4 * z1 * z2 + z2 ** 2
                        + 2 * w * (z1 + z2)) * z2bar)
          - wbar * (2 * (m2 * (w + z1) + m1 * (w + z2))
                    + (z1 ** 2 + 4 * z1 * z2 + z2 ** 2
                       + 2 * w * (z1 + z2)) * (z1bar + z2bar)))
    a2 = (-((m1 + m2) * (m1 * (w - z1) + m2 * (w - z2)))
          - (2 * z1 * z2 * (z1 + z2)
             + w * (z1 ** 2 + 4 * z1 * z2 + z2 ** 2)) * wbar ** 2
          - (m2 * (w - z2) * (2 * z1 + z2)
             + m1 * (w * z1 + 2 * (w + z1) * z2 + z2 ** 2)) * z1bar
          - (m2 * w * (2 * z1 + z2) + m1 * (w - z1) * (z1 + 2 * z2)
             + m2 * z1 * (z1 + 2 * z2)
             + 2 * z1 * z2 * (z1 + z2) * z1bar
             + w * (z1 ** 2 + 4 * z1 * z2 + z2 ** 2) * z1bar) * z2bar
          + wbar * (z1 * (2 * m1 * w + 4 * m2 * w - m1 * z1 + m2 * z1)
                    + 2 * (2 * m1 + m2) * w * z2 + (m1 - m2) * z2 ** 2
                    + (2 * z1 * z2 * (z1 + z2)
                       + w * (z1 ** 2 + 4 * z1 * z2 + z2 ** 2))
                    * (z1bar + z2bar)))
    a1 = (2 * m1 ** 2 * w * z2 + 2 * m1 * m2 * w * z2 - m1 * m2 * z2 ** 2
          - 2 * m1 * w * z2 ** 2 * wbar + m1 * w * z2 ** 2 * z1bar
          + m1 * w * z2 ** 2 * z2bar
          + z1 ** 2 * (-(m1 * m2) - 2 * m2 * w * wbar + 2 * m1 * z2 * wbar
                       + 2 * w * z2 * wbar ** 2 + z2 ** 2 * wbar ** 2
                       + m2 * (w - z2) * z1bar - 2 * w * z2 * wbar * z1bar
                       - z2 ** 2 * wbar * z1bar + m2 * w * z2bar
                       - 2 * m1 * z2 * z2bar + m2 * z2 * z2bar
                       - 2 * w * z2 * wbar * z2bar - z2 ** 2 * wbar * z2bar
                       + 2 * w * z2 * z1bar * z2bar
                       + z2 ** 2 * z1bar * z2bar)
          + z1 * (2 * m1 * m2 * w + 2 * m2 ** 2 * (w - z2) - 2 * m1 ** 2 * z2
                  - 2 * m1 * m2 * z2 - 4 * m1 * w * z2 * wbar
                  - 4 * m2 * w * z2 * wbar + 2 * m2 * z2 ** 2 * wbar
                  + 2 * w * z2 ** 2 * wbar ** 2 + 2 * m1 * w * z2 * z1bar
                  + 2 * m2 * (w - z2) * z2 * z1bar + m1 * z2 ** 2 * z1bar
                  - 2 * w * z2 ** 2 * wbar * z1bar + 2 * m1 * w * z2 * z2bar
                  + 2 * m2 * w * z2 * z2bar - m1 * z2 ** 2 * z2bar
                  - 2 * w * z2 ** 2 * wbar * z2bar
                  + 2 * w * z2 ** 2 * z1bar * z2bar))
    a0 = ((m2 * z1 + m1 * z2) * (m1 * (-w + z1) * z2 + m2 * z1 * (-w + z2))
          + z1 * z2 * (-(w * z1 * z2 * wbar ** 2)
                      - (m2 * z1 * (w - z2) + m1 * w * z2) * z1bar
                      - (m2 * w * z1 + m1 * (w - z1) * z2
                         + w * z1 * z2 * z1bar) * z2bar
                      + wbar * (2 * m2 * w * z1 + 2 * m1 * w * z2
                                - (m1 + m2) * z1 * z2
                                + w * z1 * z2 * (z1bar + z2bar))))

    return np.stack([a5, a4, a3, a2, a1, a0], axis=1)


def _companion_roots(coeffs):
    """
    Roots of monic quintics via companion-matrix eigenvalues.

    Parameters
    ----------
    coeffs : ndarray, complex, shape (N, 6)
        ``[a5, a4, ..., a0]`` per polynomial.

    Returns
    -------
    roots : ndarray, complex, shape (N, 5)
        Roots per polynomial; NaN rows where the coefficients are not finite
        (e.g. NaN lens geometry), so bad inputs propagate instead of raising.
    """
    a5 = coeffs[:, 0]
    # Avoid divide-by-zero for degenerate polynomials.
    a5_safe = np.where(np.abs(a5) < _EPS, _EPS + 0j, a5)
    with np.errstate(invalid='ignore', divide='ignore'):
        c = -coeffs[:, 1:] / a5_safe[:, None]

    n = coeffs.shape[0]
    C = np.zeros((n, 5, 5), dtype=complex)
    C[:, 0, :] = c
    C[:, 1, 0] = 1.0
    C[:, 2, 1] = 1.0
    C[:, 3, 2] = 1.0
    C[:, 4, 3] = 1.0

    # eigvals raises on NaN / inf; solve only the finite rows.
    good = np.all(np.isfinite(c), axis=1)
    if np.all(good):
        return np.linalg.eigvals(C)
    roots = np.full((n, 5), np.nan + 1j * np.nan)
    if np.any(good):
        roots[good] = np.linalg.eigvals(C[good])

    return roots


def _newton_polish(z, w, z1, z2, m1, m2, n_iter=2):
    """
    Newton polish of image roots using the linearized lens map.

    Parameters
    ----------
    z : ndarray, complex, shape (N, 5)
        Candidate image positions.
    w : ndarray, complex, shape (N,)
        True source positions.
    z1, z2 : complex
        Lens positions.
    m1, m2 : float
        Lens masses.
    n_iter : int, optional
        Number of Newton steps.

    Returns
    -------
    z : ndarray, complex, shape (N, 5)
        Polished roots.
    """
    for _ in range(int(n_iter)):
        zeta = lens_map(z, z1, z2, m1, m2)
        delta = w[:, None] - zeta
        kappa, jac = kappa_jac(z, z1, z2, m1, m2)
        jac_safe = np.where(np.abs(jac) < _EPS, _EPS, jac)
        dz = (delta - np.conjugate(kappa) * np.conjugate(delta)) / jac_safe
        z = z + dz

    return z


def binary_images(w, z1, z2, m1, m2, root_tol=_ROOT_TOL, n_polish=2):
    """
    Physical binary-lens images of source points (Witt quintic).

    Parameters
    ----------
    w : array_like, complex, shape (N,)
        Source positions (Einstein radii).
    z1, z2 : complex
        Lens positions.
    m1, m2 : float
        Lens masses.
    root_tol : float, optional
        Maximum lens-equation residual for a physical image.
    n_polish : int, optional
        Newton polish iterations after the companion-matrix solve.

    Returns
    -------
    z_img : ndarray, complex, shape (N, 5)
        Image positions; unused slots are ``nan``.
    parity : ndarray, float, shape (N, 5)
        ``sign(J)`` for physical images (0 in unused slots).
    n_phys : ndarray of int, shape (N,)
        Number of physical images (3 or 5 when the solve succeeds).
    """
    w = np.atleast_1d(np.asarray(w, dtype=complex)).reshape(-1)
    n = w.size
    if n == 0:
        return (np.zeros((0, _N_IMG), dtype=complex),
                np.zeros((0, _N_IMG), dtype=float),
                np.zeros(0, dtype=int))

    coeffs = _quintic_coeffs(w, z1, z2, m1, m2)
    z_img = _companion_roots(coeffs)
    z_img = _newton_polish(z_img, w, z1, z2, m1, m2, n_iter=n_polish)

    # Lens-equation residual; keep only physical roots.
    resid = np.abs(lens_map(z_img, z1, z2, m1, m2) - w[:, None])
    scale = np.maximum(1.0, np.abs(w))
    tol = np.maximum(root_tol * scale, 1e-8)[:, None]
    good = resid <= tol

    # Drop near-duplicate roots (eigvals can repeat).
    for i in range(_N_IMG):
        for j in range(i + 1, _N_IMG):
            close = good[:, i] & good[:, j] & (
                np.abs(z_img[:, i] - z_img[:, j]) < 1e-8)
            good[close, j] = False

    _, jac = kappa_jac(z_img, z1, z2, m1, m2)
    parity = np.where(good, np.sign(jac.real), 0.0)
    z_out = np.where(good, z_img, np.nan + 0j)
    n_phys = good.sum(axis=1).astype(int)

    return z_out, parity, n_phys


def _segment_area_moments(z0, z1, dz0, dz1, wp0, wp1, step, parabolic):
    """
    Trapezoid + parabolic area and moment contribution of one segment.

    Parameters
    ----------
    z0, z1 : complex or ndarray, complex
        Segment end-points.
    dz0, dz1 : complex or ndarray, complex
        ``dz/dtheta`` at the ends.
    wp0, wp1 : float or ndarray
        Curvature wedge at the ends.
    step : float or ndarray
        Limb-angle step.
    parabolic : bool or ndarray of bool
        Use parabolic corrections where True.

    Returns
    -------
    area, mom_x, mom_y : float or ndarray
        Contribution of this segment (positive for CCW contours).
    """
    x0, y0 = np.real(z0), np.imag(z0)
    x1, y1 = np.real(z1), np.imag(z1)
    xd0, yd0 = np.real(dz0), np.imag(dz0)
    xd1, yd1 = np.real(dz1), np.imag(dz1)

    # Trapezoid (Bozza et al. 2021 Eqs. 9 and 19).
    dy = y1 - y0
    dx = x1 - x0
    sx = x0 + x1
    sy = y0 + y1
    area = 0.5 * sx * dy
    mom_x = (sx * sx * dy) / 8.0
    mom_y = -(sy * sy * dx) / 8.0

    # Parabolic corrections; factor 1/2 on x'^2 y' for O(dtheta^4) moments.
    h3 = np.where(parabolic, step ** 3 / 24.0, 0.0)
    f_a0, f_a1 = wp0, wp1
    f_x0 = 0.5 * xd0 * xd0 * yd0 + x0 * wp0
    f_x1 = 0.5 * xd1 * xd1 * yd1 + x1 * wp1
    f_y0 = 0.5 * yd0 * yd0 * xd0 - y0 * wp0
    f_y1 = 0.5 * yd1 * yd1 * xd1 - y1 * wp1
    area = area + h3 * (f_a0 + f_a1)
    mom_x = mom_x + h3 * (f_x0 + f_x1)
    mom_y = mom_y - h3 * (f_y0 + f_y1)

    return area, mom_x, mom_y


def _chord_area_moments(z_neg, z_pos):
    """
    Area and moment of a creation/annihilation chord (trapezoid only).

    The chord is traversed from the negative-parity image to the
    positive-parity image so the signed area closes the newly created
    (or annihilated) image pair (Gould & Gaucherel 1997).

    Parameters
    ----------
    z_neg, z_pos : complex or ndarray, complex
        Negative- and positive-parity image positions.

    Returns
    -------
    area, mom_x, mom_y : float or ndarray
    """
    # Zero derivatives / wedges: trapezoid only, no parabolic term.
    return _segment_area_moments(z_neg, z_pos, 0j, 0j, 0.0, 0.0, 0.0, False)


def _match_images(z_a, p_a, z_b, p_b):
    """
    Match images of two consecutive limb samples (parity-respecting).

    Brute-forces the 120 permutations of five slots. Missing images are
    padded with ``nan``. The cost is the sum of squared distances over
    matched physical pairs; parity mismatches are forbidden.

    Parameters
    ----------
    z_a, z_b : ndarray, complex, shape (5,)
        Image positions at the two samples.
    p_a, p_b : ndarray, float, shape (5,)
        Parities (``+/-1`` or 0 if unused).

    Returns
    -------
    order_b : ndarray of int, shape (5,)
        Permutation of ``b`` slots aligned to ``a`` (``-1`` = unmatched).
    cost : float
        Matching cost (inf if no parity-consistent match exists).
    """
    good_a = np.isfinite(z_a) & (p_a != 0)
    good_b = np.isfinite(z_b) & (p_b != 0)
    n_a = int(good_a.sum())
    n_b = int(good_b.sum())
    if n_a == 0 and n_b == 0:
        return np.full(_N_IMG, -1, dtype=int), 0.0

    idx_a = np.where(good_a)[0]
    idx_b = np.where(good_b)[0]
    best_cost = np.inf
    best_order = np.full(_N_IMG, -1, dtype=int)

    # Exhaustive match among the physical images only.
    if n_a <= n_b:
        # Assign each a-image to a unique b-image.
        for perm in permutations(range(n_b), n_a):
            cost = 0.0
            ok = True
            order = np.full(_N_IMG, -1, dtype=int)
            used_b = set()
            for ia, ib_local in enumerate(perm):
                ia_slot = idx_a[ia]
                ib_slot = idx_b[ib_local]
                if p_a[ia_slot] != p_b[ib_slot]:
                    ok = False
                    break
                d = z_a[ia_slot] - z_b[ib_slot]
                cost += (d.real * d.real + d.imag * d.imag)
                order[ia_slot] = ib_slot
                used_b.add(ib_local)
            if not ok:
                continue
            # Unmatched b images (creation) stay unmatched in order.
            if cost < best_cost:
                best_cost = cost
                best_order = order
    else:
        # Assign each b-image to a unique a-image (annihilation leftovers).
        for perm in permutations(range(n_a), n_b):
            cost = 0.0
            ok = True
            order = np.full(_N_IMG, -1, dtype=int)
            for ib, ia_local in enumerate(perm):
                ia_slot = idx_a[ia_local]
                ib_slot = idx_b[ib]
                if p_a[ia_slot] != p_b[ib_slot]:
                    ok = False
                    break
                d = z_a[ia_slot] - z_b[ib_slot]
                cost += (d.real * d.real + d.imag * d.imag)
                order[ia_slot] = ib_slot
            if not ok:
                continue
            if cost < best_cost:
                best_cost = cost
                best_order = order

    return best_order, best_cost


def _limb_sample_images(zeta_c, rho, theta, z1, z2, m1, m2, root_tol):
    """
    Images, derivatives and wedges for one epoch's limb samples.

    Parameters
    ----------
    zeta_c : complex
        Source centre.
    rho : float
        Source radius.
    theta : ndarray, float, shape (N_pts,)
        Limb angles.
    z1, z2 : complex
        Lens positions.
    m1, m2 : float
        Lens masses.
    root_tol : float
        Lens-equation tolerance.

    Returns
    -------
    z_img : ndarray, complex, shape (N_pts, 5)
    parity : ndarray, float, shape (N_pts, 5)
    dz : ndarray, complex, shape (N_pts, 5)
    wedge : ndarray, float, shape (N_pts, 5)
    jac : ndarray, float, shape (N_pts, 5)
    n_phys : ndarray of int, shape (N_pts,)
    """
    e_it = np.exp(1j * theta)
    zeta = zeta_c + rho * e_it
    dzeta = 1j * rho * e_it

    z_img, parity, n_phys = binary_images(zeta, z1, z2, m1, m2,
                                          root_tol=root_tol)
    dz = np.zeros_like(z_img)
    wedge = np.zeros(z_img.shape, dtype=float)
    jac = np.zeros(z_img.shape, dtype=float)

    good = np.isfinite(z_img) & (parity != 0)
    if np.any(good):
        dz_g, wp_g, jac_g = image_derivatives(
            z_img[good], np.broadcast_to(dzeta[:, None], z_img.shape)[good],
            rho, z1, z2, m1, m2)
        dz[good] = dz_g
        wedge[good] = wp_g
        jac[good] = jac_g

    return z_img, parity, dz, wedge, jac, n_phys


def _integrate_epoch(theta, z_img, parity, dz, wedge, jac, rho):
    """
    Closed-contour area and moments for one epoch.

    Parameters
    ----------
    theta : ndarray, float, shape (N_pts,)
        Sorted limb angles in ``[0, 2 pi)``.
    z_img, parity, dz, wedge, jac : ndarrays
        Per-sample image data from :func:`_limb_sample_images`.
    rho : float
        Source radius.

    Returns
    -------
    area_signed : float
        Total signed image area (magnification * pi rho^2).
    mom_x, mom_y : float
        First moments of the signed area distribution.
    seg_err : ndarray, float, shape (N_pts,)
        Per-segment error estimates for adaptive refinement.
    """
    n_pts = theta.size
    area = 0.0
    mom_x = 0.0
    mom_y = 0.0
    seg_err = np.zeros(n_pts, dtype=float)

    for i in range(n_pts):
        j = (i + 1) % n_pts
        step = theta[j] - theta[i]
        if j == 0:
            step += 2.0 * np.pi

        z_a, p_a = z_img[i], parity[i]
        z_b, p_b = z_img[j], parity[j]
        order, cost = _match_images(z_a, p_a, z_b, p_b)

        # Matched continuum segments (same parity).
        matched_a = []
        matched_b = []
        for ia in range(_N_IMG):
            ib = order[ia]
            if ib < 0 or p_a[ia] == 0:
                continue
            matched_a.append(ia)
            matched_b.append(ib)
            # Fall back to trapezoid-only near critical curves.
            parabolic = (np.abs(jac[i, ia]) > 1e-6
                         and np.abs(jac[j, ib]) > 1e-6)
            a, mx, my = _segment_area_moments(
                z_a[ia], z_b[ib], dz[i, ia], dz[j, ib],
                wedge[i, ia], wedge[j, ib], step, parabolic)
            # Signed by image parity.
            area += p_a[ia] * a
            mom_x += p_a[ia] * mx
            mom_y += p_a[ia] * my

            # Bozza 2010 segment error (curvature change + chord mismatch).
            chord = z_b[ib] - z_a[ia]
            e_curv = step ** 3 / 24.0 * np.abs(wedge[j, ib] - wedge[i, ia])
            e_chord = (np.abs(chord - 0.5 * step * (dz[i, ia] + dz[j, ib]))
                       * np.abs(chord))
            lever = max(1.0, abs(0.5 * (z_a[ia] + z_b[ib])))
            seg_err[i] += (e_curv + e_chord) * lever

        # Creation (3 -> 5): unmatched positive and negative at sample j.
        used_b = set(matched_b)
        new_pos = [k for k in range(_N_IMG)
                   if k not in used_b and p_b[k] > 0 and np.isfinite(z_b[k])]
        new_neg = [k for k in range(_N_IMG)
                   if k not in used_b and p_b[k] < 0 and np.isfinite(z_b[k])]
        for ip, ineg in zip(new_pos, new_neg):
            a, mx, my = _chord_area_moments(z_b[ineg], z_b[ip])
            area += a
            mom_x += mx
            mom_y += my
            gap = z_b[ip] - z_b[ineg]
            # Large error for creation chords (~ |gap|^3).
            seg_err[i] += np.abs(gap) ** 3

        # Annihilation (5 -> 3): unmatched at sample i.
        used_a = set(matched_a)
        old_pos = [k for k in range(_N_IMG)
                   if k not in used_a and p_a[k] > 0 and np.isfinite(z_a[k])]
        old_neg = [k for k in range(_N_IMG)
                   if k not in used_a and p_a[k] < 0 and np.isfinite(z_a[k])]
        for ip, ineg in zip(old_pos, old_neg):
            a, mx, my = _chord_area_moments(z_a[ip], z_a[ineg])
            # Chord positive -> negative at the earlier sample (weight +1).
            area += a
            mom_x += mx
            mom_y += my
            gap = z_a[ip] - z_a[ineg]
            seg_err[i] += np.abs(gap) ** 3

        # No consistent match: force refinement of this segment.
        if not np.isfinite(cost):
            seg_err[i] = np.inf

    return area, mom_x, mom_y, seg_err


def uniform_limb_images_amps(zeta_c, rho, s, q, n_outline=64,
                             z1=None, z2=None, m1=None, m2=None,
                             root_tol=_ROOT_TOL):
    """
    FSBL images and magnifications on equally spaced limb samples.

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_times,)
        Source centres relative to the binary CM (Einstein radii).
    rho : float or array_like, shape (N_times,)
        Source radius in Einstein radii.
    s : float
        Binary separation (used when ``z1`` is None).
    q : float
        Mass ratio (used when ``z1`` is None).
    n_outline : int, optional
        Number of limb samples.
    z1, z2, m1, m2 : optional
        Override lens geometry / masses (CM frame).
    root_tol : float, optional
        Lens-equation tolerance.

    Returns
    -------
    images : ndarray, shape (N_times, 2)
        Flux-weighted centroid ``[E, N]`` relative to the CM (all images
        combined). Shape matches a single unresolved centroid; use
        :func:`adaptive_limb_images_amps` for per-image moments when needed.
    amps : ndarray, shape (N_times,)
        Total magnification.
    """
    zeta_c = np.atleast_1d(np.asarray(zeta_c, dtype=complex)).reshape(-1)
    n_times = zeta_c.size
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (n_times,))
    if z1 is None:
        z1, z2, m1, m2 = lens_positions_cm(s, q)

    theta = 2.0 * np.pi * np.arange(int(n_outline)) / int(n_outline)
    images = np.zeros((n_times, 2), dtype=float)
    amps = np.zeros(n_times, dtype=float)

    for i in range(n_times):
        z_img, parity, dz, wedge, jac, _ = _limb_sample_images(
            zeta_c[i], rho[i], theta, z1, z2, m1, m2, root_tol)
        area, mx, my, _ = _integrate_epoch(
            theta, z_img, parity, dz, wedge, jac, rho[i])
        src = np.pi * rho[i] ** 2
        amps[i] = area / src if src > 0 else 0.0
        if area != 0:
            images[i, 0] = mx / area
            images[i, 1] = my / area

    return images, amps


def adaptive_limb_images_amps(zeta_c, rho, s, q, n_init=64, tol=1e-4,
                              max_pts=1024, max_iter=40, z1=None, z2=None,
                              m1=None, m2=None, root_tol=_ROOT_TOL):
    """
    FSBL magnification and centroid with error-controlled limb sampling.

    Starts from ``n_init`` equally spaced limb points and bisects segments
    whose error estimate exceeds an equal share of ``tol * |area|``, until
    each epoch meets the budget, reaches ``max_pts``, or ``max_iter``.

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_times,)
        Source centres relative to the binary CM (Einstein radii).
    rho : float or array_like, shape (N_times,)
        Source radius in Einstein radii.
    s, q : float
        Binary separation and mass ratio (ignored if ``z1`` given).
    n_init : int, optional
        Initial uniform limb samples.
    tol : float, optional
        Relative error budget on the image area.
    max_pts : int, optional
        Maximum limb samples per epoch.
    max_iter : int, optional
        Maximum refinement passes.
    z1, z2, m1, m2 : optional
        Override lens geometry / masses.
    root_tol : float, optional
        Lens-equation tolerance.

    Returns
    -------
    images : ndarray, shape (N_times, 2)
        Flux-weighted centroids ``[E, N]`` relative to the CM.
    amps : ndarray, shape (N_times,)
        Total magnifications.
    """
    zeta_c = np.atleast_1d(np.asarray(zeta_c, dtype=complex)).reshape(-1)
    n_times = zeta_c.size
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (n_times,))
    if z1 is None:
        z1, z2, m1, m2 = lens_positions_cm(s, q)

    # Allow scalar or per-epoch lens geometry (orbital motion).
    z1 = np.broadcast_to(np.asarray(z1, dtype=complex), (n_times,))
    z2 = np.broadcast_to(np.asarray(z2, dtype=complex), (n_times,))
    m1 = np.broadcast_to(np.asarray(m1, dtype=float), (n_times,))
    m2 = np.broadcast_to(np.asarray(m2, dtype=float), (n_times,))

    images = np.zeros((n_times, 2), dtype=float)
    amps = np.zeros(n_times, dtype=float)
    if n_times == 0:
        return images, amps

    for i in range(n_times):
        theta = 2.0 * np.pi * np.arange(n_init) / n_init
        z1_i, z2_i = complex(z1[i]), complex(z2[i])
        m1_i, m2_i = float(m1[i]), float(m2[i])
        for n_iter in range(max_iter + 1):
            z_img, parity, dz, wedge, jac, _ = _limb_sample_images(
                zeta_c[i], rho[i], theta, z1_i, z2_i, m1_i, m2_i, root_tol)
            area, mx, my, seg_err = _integrate_epoch(
                theta, z_img, parity, dz, wedge, jac, rho[i])
            abs_area = abs(area)
            if (np.sum(seg_err) <= tol * max(abs_area, _EPS)
                    or theta.size >= max_pts or n_iter == max_iter):
                src = np.pi * rho[i] ** 2
                amps[i] = area / src if src > 0 else 0.0
                if area != 0:
                    images[i, 0] = mx / area
                    images[i, 1] = my / area
                break

            # Bisect every segment above an equal share of the budget.
            share = tol * max(abs_area, _EPS) / theta.size
            flag = seg_err > share
            room = max_pts - theta.size
            if room <= 0 or not np.any(flag):
                src = np.pi * rho[i] ** 2
                amps[i] = area / src if src > 0 else 0.0
                if area != 0:
                    images[i, 0] = mx / area
                    images[i, 1] = my / area
                break

            # Keep the largest-error segments within capacity.
            order = np.argsort(-np.where(flag, seg_err, -1.0))
            keep = np.zeros_like(flag)
            keep[order[:room]] = flag[order[:room]]

            step = np.diff(theta, append=theta[0] + 2.0 * np.pi)
            new = np.mod(theta[keep] + 0.5 * step[keep], 2.0 * np.pi)
            theta = np.sort(np.concatenate([theta, new]))

    return images, amps


def caustic_samples(z1, z2, m1, m2, n_phi=_N_PHI):
    """
    Sample the binary-lens caustics with tangents and neighbour matching.

    Critical points solve ``kappa(z) = m1/(z-z1)^2 + m2/(z-z2)^2 =
    exp(-i phi)`` (a quartic in ``z``). The caustic point is the lens map
    of each critical point, and its tangent is
    ``d zeta / d phi = zdot + exp(i phi) conj(zdot)`` with
    ``zdot = -i exp(-i phi) / kappa'(z)``.

    Parameters
    ----------
    z1, z2 : complex or array_like, complex, shape (N_geom,)
        Lens positions.
    m1, m2 : float or array_like, shape (N_geom,)
        Lens mass fractions.
    n_phi : int, optional
        Samples of the critical-curve phase ``phi`` in ``[0, 2 pi)``.

    Returns
    -------
    zeta : ndarray, complex, shape (N_geom, n_phi, 4)
        Caustic points (one per quartic root).
    zeta_dot : ndarray, complex, shape (N_geom, n_phi, 4)
        Caustic tangents ``d zeta / d phi``.
    nxt : ndarray, int, shape (N_geom, n_phi, 4)
        Index of the matching root at the next phase sample (wrapping
        from the last sample to the first).

    Notes
    -----
    Matching is pairwise between neighbouring phases (no global branch
    labels), which is all that sign changes along a branch need.
    """
    z1 = np.atleast_1d(np.asarray(z1, dtype=complex))
    z2 = np.atleast_1d(np.asarray(z2, dtype=complex))
    m1 = np.atleast_1d(np.asarray(m1, dtype=float))
    m2 = np.atleast_1d(np.asarray(m2, dtype=float))
    z1, z2, m1, m2 = np.broadcast_arrays(z1, z2, m1, m2)
    n_geom = z1.size

    # Shapes below: geometry [N_geom, 1], phase [1, n_phi].
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    e = np.exp(-1j * phi)[None, :]
    z1c, z2c = z1[:, None], z2[:, None]
    m1c, m2c = m1[:, None], m2[:, None]

    # Monic quartic (z-z1)^2 (z-z2)^2 - [m1 (z-z2)^2 + m2 (z-z1)^2] / e.
    a1, a0 = -2.0 * z1c, z1c ** 2
    b1, b0 = -2.0 * z2c, z2c ** 2
    p3 = np.broadcast_to(a1 + b1, (n_geom, n_phi))
    p2 = a0 + b0 + a1 * b1 - (m1c + m2c) / e
    p1 = a1 * b0 + a0 * b1 - (m1c * b1 + m2c * a1) / e
    p0 = a0 * b0 - (m1c * b0 + m2c * a0) / e

    # Companion-matrix roots, shape [N_geom, n_phi, 4].
    comp = np.zeros((n_geom, n_phi, 4, 4), dtype=complex)
    comp[..., 0, :] = -np.stack([p3, p2, p1, p0], axis=-1)
    comp[..., 1, 0] = comp[..., 2, 1] = comp[..., 3, 2] = 1.0

    # Non-finite geometry gives NaN roots (eigvals would raise).
    good = np.all(np.isfinite(comp), axis=(-2, -1))
    z = np.full((n_geom, n_phi, 4), np.nan + 1j * np.nan)
    if np.any(good):
        z[good] = np.linalg.eigvals(comp[good])

    # Caustic points and tangents.
    kappa, _ = kappa_jac(z, z1c[..., None], z2c[..., None], m1c[..., None],
                         m2c[..., None])
    kappa_p = kappa_prime(z, z1c[..., None], z2c[..., None], m1c[..., None],
                          m2c[..., None])
    zdot = -1j * e[..., None] / kappa_p
    zeta = z - np.conj(m1c[..., None] / (z - z1c[..., None])
                       + m2c[..., None] / (z - z2c[..., None]))
    zeta_dot = zdot + np.conj(e)[..., None] * np.conj(zdot)

    # Pairwise matching to the next phase sample (24 permutations).
    perms = np.array(list(permutations(range(4))))
    z_next = np.roll(z, -1, axis=1)
    cost = np.sum(np.abs(z_next[..., perms] - z[..., None, :]) ** 2, axis=-1)
    nxt = perms[np.argmin(cost, axis=-1)]

    return zeta, zeta_dot, nxt


def caustic_tangency_radii(zeta_c, zeta, zeta_dot, nxt):
    """
    Radii at which circles around each source centre touch the caustics.

    A circle of radius ``r`` around ``c`` is tangent to the caustic where
    ``h = Re[conj(zeta - c) d zeta/d phi]`` changes sign along a branch
    (distance extrema, including cusps, where the tangent reverses).

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_times,)
        Source centres.
    zeta, zeta_dot, nxt : ndarray, shape (N_geom, n_phi, 4)
        Output of :func:`caustic_samples`; ``N_geom`` is 1 (static lens)
        or ``N_times``.

    Returns
    -------
    r : ndarray, float, shape (N_times, n_phi * 4)
        Tangency radii, ``NaN`` where a phase interval has no sign change.
    """
    zeta_c = np.atleast_1d(np.asarray(zeta_c, dtype=complex))
    c = zeta_c[:, None, None]

    # Distance and its phase derivative (times distance) at every sample.
    h = np.real(np.conj(zeta - c) * zeta_dot)
    dist = np.abs(zeta - c)

    # Same quantities at the matched root of the next phase sample.
    nxt = np.broadcast_to(nxt, h.shape)
    h_next = np.take_along_axis(np.roll(h, -1, axis=1), nxt, axis=-1)
    d_next = np.take_along_axis(np.roll(dist, -1, axis=1), nxt, axis=-1)

    # Linear interpolation of the zero of h inside each flipped interval.
    flip = np.sign(h) * np.sign(h_next) < 0
    with np.errstate(invalid='ignore', divide='ignore'):
        f = h / (h - h_next)
    r = np.where(flip, dist + f * (d_next - dist), np.nan)

    return r.reshape(zeta_c.size, -1)


def ld_break_nodes(r_tan, rho, n_ld=_N_LD, n_brk=_N_BRK):
    """
    Radial limb-darkening nodes split at caustic-tangency radii.

    The uniform-disk flux ``F(r)`` has a derivative jump where a circle of
    radius ``r`` touches a fold or passes a cusp. The ``s`` interval
    ``[0, 1]`` (``s = sqrt(1 - r^2/rho^2)``) is split at the ``n_brk``
    largest tangency radii inside the source; unused split slots bisect
    the longest interval. Each interval carries ``n_ld`` Gauss-Legendre
    nodes graded toward both ends (``s = a + (b - a)(3 t^2 - 2 t^3)``).

    Parameters
    ----------
    r_tan : array_like, float, shape (N_times, N_cand)
        Candidate tangency radii (``NaN`` = none), e.g. from
        :func:`caustic_tangency_radii`.
    rho : float or array_like, shape (N_times,)
        Source radius.
    n_ld : int, optional
        Gauss-Legendre nodes per interval.
    n_brk : int, optional
        Split points per epoch (static count).

    Returns
    -------
    s : ndarray, float, shape (N_times, (n_brk + 1) * n_ld)
        Nodes in ``s``.
    w : ndarray, float, shape (N_times, (n_brk + 1) * n_ld)
        Weights (sum to 1 per epoch).
    """
    r_tan = np.atleast_2d(np.asarray(r_tan, dtype=float))
    n_times = r_tan.shape[0]
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (n_times,))

    # Keep the n_brk largest radii strictly inside the source.
    inside = np.isfinite(r_tan) & (r_tan > 0) & (r_tan < rho[:, None])
    r_in = np.where(inside, r_tan, -np.inf)
    r_top = -np.sort(-r_in, axis=1)[:, :n_brk]
    if r_top.shape[1] < n_brk:
        pad = np.full((n_times, n_brk - r_top.shape[1]), -np.inf)
        r_top = np.concatenate([r_top, pad], axis=1)
    valid = np.isfinite(r_top)
    with np.errstate(invalid='ignore'):
        s_top = np.sqrt(1.0 - (r_top / rho[:, None]) ** 2)

    # Interval edges in s; unfilled slots sit at 1 (zero-length).
    edges = np.concatenate([np.zeros((n_times, 1)),
                            np.where(valid, s_top, 1.0),
                            np.ones((n_times, 1))], axis=1)
    edges = np.sort(edges, axis=1)

    # Spend unused slots bisecting the longest interval.
    n_valid = np.sum(valid, axis=1)
    rows = np.arange(n_times)
    for j in range(n_brk):
        need = n_valid <= j
        gaps = np.diff(edges, axis=1)
        k = np.argmax(gaps, axis=1)
        mid = 0.5 * (edges[rows, k] + edges[rows, k + 1])
        edges[:, -1] = np.where(need, mid, edges[:, -1])
        edges = np.sort(edges, axis=1)
        edges[:, -1] = 1.0

    # Graded Gauss-Legendre nodes on each interval.
    x, w = np.polynomial.legendre.leggauss(int(n_ld))
    tau = 0.5 * (x + 1.0)
    w_tau = 0.5 * w
    g = 3.0 * tau ** 2 - 2.0 * tau ** 3
    dg = 6.0 * tau * (1.0 - tau)
    a, b = edges[:, :-1, None], edges[:, 1:, None]
    s = a + (b - a) * g
    ws = (b - a) * dg * w_tau

    return s.reshape(n_times, -1), ws.reshape(n_times, -1)


def ld_images_amps(zeta_c, rho, gamma, s, q, n_ld=_N_LD, n_init=64, tol=1e-4,
                   max_pts=1024, z1=None, z2=None, m1=None, m2=None,
                   n_brk=_N_BRK, n_phi=_N_PHI):
    """
    Linear (Gamma-law) limb-darkened FSBL magnification and centroid.

    The magnified flux is a weighted sum over nested uniform disks (see
    :func:`bagle.fspl_contour.ld_images_amps`). The radial nodes are split
    where circles around the source centre touch the caustics
    (:func:`ld_break_nodes`), so each radial interval is smooth.

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_times,)
        Source centres relative to the binary CM.
    rho : float
        Source radius in Einstein radii.
    gamma : float
        Linear limb-darkening coefficient (Gamma convention).
    s, q : float
        Binary separation and mass ratio.
    n_ld : int, optional
        Gauss-Legendre nodes per radial interval; each epoch uses
        ``(n_brk + 1) * n_ld + 1`` uniform disks.
    n_init, tol, max_pts : optional
        Forwarded to the uniform-disk adaptive integrator.
    z1, z2, m1, m2 : optional
        Override lens geometry / masses (scalars or per epoch).
    n_brk : int, optional
        Caustic-tangency split points per epoch.
    n_phi : int, optional
        Critical-curve phase samples used to locate the tangencies.

    Returns
    -------
    images : ndarray, shape (N_times, 2)
        Brightness-weighted centroids relative to the CM.
    amps : ndarray, shape (N_times,)
        Total magnifications.
    """
    zeta_c = np.atleast_1d(np.asarray(zeta_c, dtype=complex)).reshape(-1)
    n_times = zeta_c.size
    if z1 is None:
        z1, z2, m1, m2 = lens_positions_cm(s, q)

    # Per-epoch lens geometry (broadcast scalars).
    z1_arr = np.broadcast_to(np.asarray(z1, dtype=complex), (n_times,))
    z2_arr = np.broadcast_to(np.asarray(z2, dtype=complex), (n_times,))
    m1_arr = np.broadcast_to(np.asarray(m1, dtype=float), (n_times,))
    m2_arr = np.broadcast_to(np.asarray(m2, dtype=float), (n_times,))

    def disk_images_amps(zeta_rows, rho_rows):
        # Nested disks repeat each epoch; tile the lens geometry to match.
        n_disk = int(np.asarray(zeta_rows).size // n_times)
        z1_r = np.repeat(z1_arr, n_disk)
        z2_r = np.repeat(z2_arr, n_disk)
        m1_r = np.repeat(m1_arr, n_disk)
        m2_r = np.repeat(m2_arr, n_disk)
        # fspl_contour.ld_images_amps expects (N, 2, 2) images and (N, 2) amps.
        img, amp = adaptive_limb_images_amps(
            zeta_rows, rho_rows, s, q, n_init=n_init, tol=tol,
            max_pts=max_pts, z1=z1_r, z2=z2_r, m1=m1_r, m2=m2_r)
        n = amp.shape[0]
        images = np.zeros((n, 2, 2), dtype=float)
        images[:, 0, :] = img
        amps = np.zeros((n, 2), dtype=float)
        amps[:, 0] = amp

        return images, amps

    if gamma == 0:
        img, amp = adaptive_limb_images_amps(
            zeta_c, rho, s, q, n_init=n_init, tol=tol, max_pts=max_pts,
            z1=z1_arr, z2=z2_arr, m1=m1_arr, m2=m2_arr)
        return img, amp

    # Caustic-tangency radii: one caustic for a static lens, else per epoch
    # (in chunks to bound the companion-matrix memory).
    static = all(np.all(a == a[0]) for a in (z1_arr, z2_arr, m1_arr, m2_arr))
    if static:
        caus = caustic_samples(z1_arr[0], z2_arr[0], m1_arr[0], m2_arr[0],
                               n_phi=n_phi)
        r_tan = caustic_tangency_radii(zeta_c, *caus)
    else:
        chunk = 64
        r_tan = np.concatenate([
            caustic_tangency_radii(
                zeta_c[i:i + chunk],
                *caustic_samples(z1_arr[i:i + chunk], z2_arr[i:i + chunk],
                                 m1_arr[i:i + chunk], m2_arr[i:i + chunk],
                                 n_phi=n_phi))
            for i in range(0, n_times, chunk)], axis=0)

    # Radial nodes split at the tangencies, then the nested-disk sum.
    # The empty second image slot gives a discarded 0/0 centroid.
    nodes = ld_break_nodes(r_tan, rho, n_ld=n_ld, n_brk=n_brk)
    with np.errstate(invalid='ignore', divide='ignore'):
        images2, amps2 = fspl_contour.ld_images_amps(
            zeta_c, rho, gamma, disk_images_amps, n_ld=n_ld,
            radius_nodes=nodes)

    return images2[:, 0, :], amps2[:, 0]
