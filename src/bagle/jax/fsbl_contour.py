"""
Traced binary-lens finite-source contour integrals (JAX port).

JAX port of :mod:`bagle.fsbl_contour`: sample the source limb, solve the
Witt quintic for the images, match images between consecutive limb samples,
and accumulate parity-signed image areas and first moments with trapezoid
plus parabolic corrections (Bozza 2010; Bozza et al. 2021). Image pair
creation and annihilation are closed with Gould & Gaucherel (1997) chords.

Differences from the host integrator, all needed for fixed shapes:

* Images are matched within parity groups. A binary lens always has one
  more negative- than positive-parity image (1 + 2 or 2 + 3), so each
  sample is laid out as ``[+, +, -, -, -]`` slots and only 2 x 6
  permutations are scored instead of 120.
* The limb is refined with a fixed schedule: ``n_init`` uniform angles,
  then ``n_pass`` passes that bisect the ``n_add`` segments with the
  largest error estimate (Bozza 2010 estimator, as on the host). The
  passes are unrolled, so every array shape is static.
* Roots on the starting limb come from the companion matrix; samples added
  by refinement are warm-started from their neighbour's quintic roots with
  a few Ehrlich-Aberth iterations (no per-pass LAPACK call). Root finding
  runs under ``stop_gradient`` and is followed by one differentiable
  Newton step on the lens equation, which gives the implicit-function
  derivative of each image.
* Limb darkening splits the radial integral at a fixed number (``n_brk``)
  of caustic-tangency radii, the largest ones inside the source; unused
  split points bisect the longest interval.

Lens positions and masses follow the VBMicrolensing convention used by the
host module (origin at the centre of mass). Coordinates are complex
``East + i North`` in Einstein radii.
"""
from __future__ import annotations

from functools import partial
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax_physics import quintic_coefficients

# Floor on |J| and lens distances so derivatives stay finite.
_EPS = 1e-15
_ROOT_TOL = 1e-10
_N_IMG = 5

# Smallest |J| at which a segment end still gets parabolic corrections.
_J_PARABOLIC = 1e-3

# Stand-in position for empty image slots (far from both lenses, J ~ 1).
_Z_FAR = 1.0e3 + 0.0j

# Default limb schedule (initial samples, passes, samples added per pass).
_N_INIT = 64
_N_PASS = 16
_N_ADD = 8

# Parity of each canonical slot: two positive, then three negative.
# NumPy (not jnp) so importing this module inside a trace leaks nothing.
_SLOT_SIGN = np.array([1.0, 1.0, -1.0, -1.0, -1.0])

# Permutations within the positive (slots 0-1) and negative (2-4) groups.
_PERM_POS = np.array([[0, 1], [1, 0]])
_PERM_NEG = np.array([[2, 3, 4], [2, 4, 3], [3, 2, 4],
                      [3, 4, 2], [4, 2, 3], [4, 3, 2]])

# Limb darkening: Gauss-Legendre nodes per radial interval, caustic-tangency
# split points per epoch, and critical-curve phase samples (as on the host).
_N_LD = 8
_N_BRK = 2
_N_PHI = 512

# All orderings of the four critical-curve roots.
_PERM4 = np.array(list(permutations(range(4))))


def _safe_dist(d):
    """
    Floor complex lens distances away from zero.

    Parameters
    ----------
    d : jnp.ndarray, complex
        Image minus lens position.

    Returns
    -------
    d : jnp.ndarray, complex
        Same shape; entries with ``|d| < _EPS`` replaced by ``_EPS``.
    """
    return jnp.where(jnp.abs(d) < _EPS, _EPS + 0j, d)


def _kappa_terms(z, z1, z2, m1, m2):
    """
    Deflection, its derivatives and the Jacobian at image positions.

    Parameters
    ----------
    z : jnp.ndarray, complex
        Image positions.
    z1, z2 : complex
        Lens positions (centre-of-mass frame).
    m1, m2 : float
        Lens mass fractions.

    Returns
    -------
    alpha : jnp.ndarray, complex
        ``sum_i m_i / (z - z_i)`` (lens map is ``z - conj(alpha)``).
    kappa : jnp.ndarray, complex
        ``sum_i m_i / (z - z_i)^2``.
    kappa_p : jnp.ndarray, complex
        ``d kappa / dz = sum_i -2 m_i / (z - z_i)^3``.
    jac : jnp.ndarray, float
        Jacobian determinant ``1 - |kappa|^2``.
    """
    d1 = _safe_dist(z - z1)
    d2 = _safe_dist(z - z2)

    alpha = m1 / d1 + m2 / d2
    kappa = m1 / d1 ** 2 + m2 / d2 ** 2
    kappa_p = -2.0 * m1 / d1 ** 3 - 2.0 * m2 / d2 ** 3
    jac = 1.0 - (kappa.real ** 2 + kappa.imag ** 2)

    return alpha, kappa, kappa_p, jac


def _newton_step(z, w, z1, z2, m1, m2):
    """
    One Newton step on the binary lens equation.

    Solves the linearised map ``dzeta = dz + conj(kappa) conj(dz)`` for the
    image shift that moves ``zeta(z)`` onto ``w``. Applied to a converged,
    gradient-stopped root this step carries the implicit-function
    derivative of the image with respect to the source and lens.

    Parameters
    ----------
    z : jnp.ndarray, complex, shape (N, 5)
        Candidate image positions.
    w : jnp.ndarray, complex, shape (N,)
        Source positions.
    z1, z2 : complex
        Lens positions.
    m1, m2 : float
        Lens mass fractions.

    Returns
    -------
    z : jnp.ndarray, complex, shape (N, 5)
        Updated image positions.
    """
    alpha, kappa, _, jac = _kappa_terms(z, z1, z2, m1, m2)
    jac = jnp.where(jnp.abs(jac) < _EPS, _EPS, jac)

    # Residual of the lens equation zeta(z) = z - conj(alpha).
    delta = w[:, None] - (z - jnp.conj(alpha))

    # Invert the 2x2 real linearisation of the map.
    dz = (delta - jnp.conj(kappa) * jnp.conj(delta)) / jac

    return z + dz


def _aberth_roots(coeffs, z, n_iter):
    """
    Ehrlich-Aberth refinement of all quintic roots (warm start).

    Parameters
    ----------
    coeffs : jnp.ndarray, complex, shape (N, 6)
        ``[a5, a4, ..., a0]`` per polynomial (high degree first).
    z : jnp.ndarray, complex, shape (N, 5)
        Starting roots, e.g. the roots at a neighbouring limb sample.
    n_iter : int
        Number of simultaneous iterations (static).

    Returns
    -------
    z : jnp.ndarray, complex, shape (N, 5)
        Refined roots.
    """
    # Polynomial and derivative coefficients, broadcast over the roots.
    c = coeffs[:, None, :]
    dc = c[:, :, :-1] * jnp.arange(_N_IMG, 0, -1)

    # Mask for the off-diagonal pairs in the Aberth sum.
    off = ~jnp.eye(_N_IMG, dtype=bool)

    for _ in range(int(n_iter)):
        # Horner evaluation of p(z) and p'(z); shape (N, 5).
        p = jnp.zeros_like(z)
        for k in range(_N_IMG + 1):
            p = p * z + c[:, :, k]
        dp = jnp.zeros_like(z)
        for k in range(_N_IMG):
            dp = dp * z + dc[:, :, k]

        # Newton ratio corrected by the repulsion of the other roots.
        ratio = p / jnp.where(dp == 0, _EPS + 0j, dp)
        diff = z[:, :, None] - z[:, None, :]
        diff = jnp.where(off, diff, 1.0)
        repulse = jnp.sum(jnp.where(off, 1.0 / diff, 0.0), axis=2)
        denom = 1.0 - ratio * repulse
        z = z - ratio / jnp.where(denom == 0, _EPS + 0j, denom)

    return z


def _binary_images(w, z1, z2, m1, m2, root_tol=_ROOT_TOL, n_polish=2,
                   z_init=None, n_aberth=24):
    """
    Physical binary-lens images of source points (Witt quintic).

    JAX port of :func:`bagle.fsbl_contour.binary_images`.

    Parameters
    ----------
    w : jnp.ndarray, complex, shape (N,)
        Source positions (Einstein radii).
    z1, z2 : complex
        Lens positions.
    m1, m2 : float
        Lens mass fractions.
    root_tol : float, optional
        Maximum lens-equation residual for a physical image (relative to
        ``max(1, |w|)``, floored at ``1e-8``).
    n_polish : int, optional
        Gradient-free Newton steps on the lens equation after the
        polynomial solve.
    z_init : jnp.ndarray, complex, shape (N, 5), optional
        Quintic roots at nearby source points. When given, the roots are
        refined from them with ``n_aberth`` Ehrlich-Aberth iterations;
        otherwise they come from companion-matrix eigenvalues (one LAPACK
        call, which carries a large fixed cost on CPU).
    n_aberth : int, optional
        Aberth iterations for the warm start (static).

    Returns
    -------
    z : jnp.ndarray, complex, shape (N, 5)
        Image positions (all slots finite; check ``good``).
    good : jnp.ndarray, bool, shape (N, 5)
        True for physical, non-duplicate images.
    parity : jnp.ndarray, float, shape (N, 5)
        ``sign(J)`` of each image.
    roots : jnp.ndarray, complex, shape (N, 5)
        All five quintic roots before the lens-equation polish (no
        gradient), for warm-starting nearby solves.
    """
    n = w.shape[0]

    # Root finding carries no gradient; the last Newton step does.
    w_sg = jax.lax.stop_gradient(w)
    z1_sg, z2_sg = jax.lax.stop_gradient(z1), jax.lax.stop_gradient(z2)
    m1_sg, m2_sg = jax.lax.stop_gradient(m1), jax.lax.stop_gradient(m2)

    # Quintic coefficients per source point; shape (N, 6), high degree first.
    coeffs = jnp.stack(quintic_coefficients(w_sg, z1_sg, z2_sg, m1_sg, m2_sg),
                       axis=1)
    a5 = coeffs[:, 0]
    a5 = jnp.where(jnp.abs(a5) < _EPS, _EPS + 0j, a5)
    coeffs = coeffs / a5[:, None]

    if z_init is None:
        # Companion matrix of the monic quintic; eigenvalues are the roots.
        comp = jnp.zeros((n, _N_IMG, _N_IMG), dtype=jnp.complex128)
        comp = comp.at[:, 0, :].set(-coeffs[:, 1:])
        comp = comp.at[:, jnp.arange(1, _N_IMG),
                       jnp.arange(_N_IMG - 1)].set(1.0)
        roots = jnp.linalg.eigvals(comp)
    else:
        roots = _aberth_roots(coeffs, jax.lax.stop_gradient(z_init),
                              n_aberth)
    roots = jax.lax.stop_gradient(roots)

    # Gradient-free polish, then one differentiable step.
    z = roots
    for _ in range(int(n_polish)):
        z = _newton_step(z, w_sg, z1_sg, z2_sg, m1_sg, m2_sg)
    z = jax.lax.stop_gradient(z)
    z = _newton_step(z, w, z1, z2, m1, m2)

    # Keep roots that satisfy the lens equation.
    alpha, _, _, jac = _kappa_terms(z, z1, z2, m1, m2)
    resid = jnp.abs(z - jnp.conj(alpha) - w[:, None])
    tol = jnp.maximum(root_tol * jnp.maximum(1.0, jnp.abs(w)), 1e-8)
    good = jax.lax.stop_gradient(resid <= tol[:, None])

    # Drop near-duplicate roots (eigvals can repeat), lower slot wins.
    for i in range(_N_IMG):
        for j in range(i + 1, _N_IMG):
            close = good[:, i] & good[:, j] & (
                jnp.abs(z[:, i] - z[:, j]) < 1e-8)
            good = good.at[:, j].set(good[:, j] & ~close)

    parity = jax.lax.stop_gradient(jnp.sign(jac))

    return z, good, parity, roots


def _limb_samples(zeta_c, rho, theta, z1, z2, m1, m2, root_tol, z_init=None):
    """
    Canonical image layout, derivatives and wedges on limb samples.

    Each sample's images are placed in slots ``[+, +, -, -, -]``; empty
    slots hold :data:`_Z_FAR` and are flagged invalid.

    Parameters
    ----------
    zeta_c : complex
        Source centre (centre-of-mass frame, Einstein radii).
    rho : float
        Source radius (Einstein radii).
    theta : jnp.ndarray, float, shape (N,)
        Limb angles (no gradient).
    z1, z2 : complex
        Lens positions.
    m1, m2 : float
        Lens mass fractions.
    root_tol : float
        Lens-equation tolerance.
    z_init : jnp.ndarray, complex, shape (N, 5), optional
        Quintic roots at neighbouring samples for a warm start (see
        :func:`_binary_images`).

    Returns
    -------
    sample : tuple of jnp.ndarray, each shape (N, 5)
        ``(z, valid, dz, wedge, jac)``: image positions in canonical slots,
        True where a slot holds a physical image of the slot's parity,
        ``dz / dtheta`` along the limb, the wedge ``x' ^ x''`` (Bozza et al.
        2021 Eq. 14) and the Jacobian determinant.
    roots : jnp.ndarray, complex, shape (N, 5)
        Raw quintic roots (no gradient).
    """
    # Source limb points and their limb-angle derivative.
    e_it = jnp.exp(1j * theta)
    w = zeta_c + rho * e_it
    dzeta = 1j * rho * e_it

    z, good, parity, roots = _binary_images(w, z1, z2, m1, m2,
                                            root_tol=root_tol, z_init=z_init)

    # Physical positive images first, then physical negative images.
    # Stable argsort keeps the root order within each group.
    is_pos = good & (parity > 0)
    is_neg = good & (parity < 0)
    idx_pos = jnp.argsort(jnp.where(is_pos, 0, 1), axis=1, stable=True)[:, :2]
    idx_neg = jnp.argsort(jnp.where(is_neg, 0, 1), axis=1, stable=True)[:, :3]
    idx = jnp.concatenate([idx_pos, idx_neg], axis=1)

    # Gather into the canonical layout; a slot is valid if its parity fits.
    z = jnp.take_along_axis(z, idx, axis=1)
    valid = jnp.concatenate(
        [jnp.take_along_axis(is_pos, idx_pos, axis=1),
         jnp.take_along_axis(is_neg, idx_neg, axis=1)], axis=1)
    z = jnp.where(valid, z, _Z_FAR)

    # z' = (zeta' - conj(kappa) conj(zeta')) / J.
    _, kappa, kappa_p, jac = _kappa_terms(z, z1, z2, m1, m2)
    jac_safe = jnp.where(jnp.abs(jac) < _EPS, _EPS, jac)
    dzeta = dzeta[:, None]
    dz = (dzeta - jnp.conj(kappa) * jnp.conj(dzeta)) / jac_safe

    # Wedge from kappa' without forming z''.
    wedge = (rho * rho + (dz * dz * dzeta * kappa_p).imag) / jac_safe

    return (z, valid, dz, wedge, jac), roots


def _segment_terms(z0, z1, dz0, dz1, wp0, wp1, step, parabolic):
    """
    Trapezoid plus parabolic area and moments of contour segments.

    JAX port of :func:`bagle.fsbl_contour._segment_area_moments`.

    Parameters
    ----------
    z0, z1 : jnp.ndarray, complex
        Segment end-points.
    dz0, dz1 : jnp.ndarray, complex
        ``dz / dtheta`` at the ends.
    wp0, wp1 : jnp.ndarray, float
        Curvature wedge at the ends.
    step : jnp.ndarray, float
        Limb-angle step, broadcastable to ``z0``.
    parabolic : jnp.ndarray, bool
        Use the parabolic corrections where True.

    Returns
    -------
    area, mom_x, mom_y : jnp.ndarray, float
        ``oint x dy``, ``1/2 oint x^2 dy`` and ``-1/2 oint y^2 dx`` of each
        segment.
    """
    x0, y0 = z0.real, z0.imag
    x1, y1 = z1.real, z1.imag
    xd0, yd0 = dz0.real, dz0.imag
    xd1, yd1 = dz1.real, dz1.imag

    # Trapezoid (Bozza et al. 2021 Eqs. 9 and 19).
    dx, dy = x1 - x0, y1 - y0
    sx, sy = x0 + x1, y0 + y1
    area = 0.5 * sx * dy
    mom_x = sx * sx * dy / 8.0
    mom_y = -sy * sy * dx / 8.0

    # Parabolic corrections; factor 1/2 on x'^2 y' for the moments.
    h3 = jnp.where(parabolic, step ** 3 / 24.0, 0.0)
    f_x0 = 0.5 * xd0 * xd0 * yd0 + x0 * wp0
    f_x1 = 0.5 * xd1 * xd1 * yd1 + x1 * wp1
    f_y0 = 0.5 * yd0 * yd0 * xd0 - y0 * wp0
    f_y1 = 0.5 * yd1 * yd1 * xd1 - y1 * wp1
    area = area + h3 * (wp0 + wp1)
    mom_x = mom_x + h3 * (f_x0 + f_x1)
    mom_y = mom_y - h3 * (f_y0 + f_y1)

    return area, mom_x, mom_y


def _match_next(z_a, va, z_b, vb):
    """
    Match each sample's images onto the next sample's (parity groups).

    Parameters
    ----------
    z_a, z_b : jnp.ndarray, complex, shape (N, 5)
        Canonical images at the segment start and end.
    va, vb : jnp.ndarray, bool, shape (N, 5)
        Slot validity at the start and end.

    Returns
    -------
    order : jnp.ndarray, int, shape (N, 5)
        Slot of ``z_b`` matched to each slot of ``z_a``.
    """
    z_a = jax.lax.stop_gradient(z_a)
    z_b = jax.lax.stop_gradient(z_b)

    def group_cost(perms):
        # Shape of zb_p, vb_p: (N, n_perm, n_group).
        slots = perms[0]
        zb_p = z_b[:, perms]
        vb_p = vb[:, perms]
        za_g = z_a[:, None, slots]
        va_g = va[:, None, slots]
        both = va_g & vb_p
        d2 = jnp.where(both, jnp.abs(za_g - zb_p) ** 2, 0.0)

        # Match as many physical pairs as possible, then minimise the
        # summed d^2 among those permutations (lexicographic, so no
        # penalty constant swamps the distances).
        n_match = jnp.sum(both, axis=2)
        most = n_match == jnp.max(n_match, axis=1, keepdims=True)
        cost = jnp.where(most, jnp.sum(d2, axis=2), jnp.inf)
        best = jnp.argmin(cost, axis=1)

        return jnp.asarray(perms)[best]

    return jnp.concatenate([group_cost(_PERM_POS), group_cost(_PERM_NEG)],
                           axis=1)


def _limb_steps(theta):
    """
    Angle step from each sorted limb sample to the next (closed limb).

    Parameters
    ----------
    theta : jnp.ndarray, float, shape (N_pts,)
        Sorted limb angles in ``[0, 2 pi)``.

    Returns
    -------
    step : jnp.ndarray, float, shape (N_pts,)
        ``theta[i + 1] - theta[i]``; the last step wraps by ``2 pi``.
    """
    step = jnp.roll(theta, -1) - theta

    return step.at[-1].add(2.0 * jnp.pi)


def _segment_sums(step, sample_a, sample_b):
    """
    Area, moments and error estimate of limb segments ``a -> b``.

    Parameters
    ----------
    step : jnp.ndarray, float, shape (N_seg,)
        Limb-angle step of each segment.
    sample_a, sample_b : tuple of jnp.ndarray, each shape (N_seg, 5)
        ``(z, valid, dz, wedge, jac)`` from :func:`_limb_samples` at the
        segment start and end.

    Returns
    -------
    area, mom_x, mom_y : jnp.ndarray, float, shape (N_seg,)
        Parity-signed contributions of each segment, including any image
        creation or annihilation chord.
    seg_err : jnp.ndarray, float, shape (N_seg,)
        Error estimate of each segment (no gradient).
    """
    z, valid, dz, wedge, jac = sample_a
    step_c = step[:, None]

    # Align the end sample's slots with the start sample's images.
    order = _match_next(z, valid, sample_b[0], sample_b[1])
    take = lambda a: jnp.take_along_axis(a, order, axis=1)
    z_b, vb, dz_b, wp_b, jac_b = [take(a) for a in sample_b]

    # Matched continuum segments, signed by slot parity. The parabolic
    # terms grow like 1/J^3, so they are dropped for any end within
    # _J_PARABOLIC of the critical curve (the value and its gradient
    # otherwise spike when a fixed-budget sample lands near J = 0).
    both = valid & vb
    parabolic = (both & (jnp.abs(jac) > _J_PARABOLIC)
                 & (jnp.abs(jac_b) > _J_PARABOLIC))
    a, mx, my = _segment_terms(z, z_b, dz, dz_b, wedge, wp_b, step_c,
                               parabolic)
    sign = jnp.where(both, _SLOT_SIGN[None, :], 0.0)
    area = jnp.sum(sign * a, axis=1)
    mom_x = jnp.sum(sign * mx, axis=1)
    mom_y = jnp.sum(sign * my, axis=1)

    # Bozza 2010 segment error (curvature change plus chord mismatch).
    z_sg, zb_sg = jax.lax.stop_gradient(z), jax.lax.stop_gradient(z_b)
    chord = zb_sg - z_sg
    e_curv = step_c ** 3 / 24.0 * jnp.abs(
        jax.lax.stop_gradient(wp_b - wedge))
    e_chord = jnp.abs(chord - 0.5 * step_c * jax.lax.stop_gradient(
        dz + dz_b)) * jnp.abs(chord)
    lever = jnp.maximum(1.0, jnp.abs(0.5 * (z_sg + zb_sg)))
    seg_err = jnp.sum(jnp.where(both, (e_curv + e_chord) * lever, 0.0),
                      axis=1)

    def chord_pair(new, z_src):
        # One new (or lost) image in each parity group closes a chord.
        n_pos = jnp.sum(new[:, :2], axis=1)
        n_neg = jnp.sum(new[:, 2:], axis=1)
        z_pos = jnp.sum(jnp.where(new[:, :2], z_src[:, :2], 0.0), axis=1)
        z_neg = jnp.sum(jnp.where(new[:, 2:], z_src[:, 2:], 0.0), axis=1)
        ok = (n_pos == 1) & (n_neg == 1)

        return ok, z_pos, z_neg

    zero = jnp.zeros_like(step)
    parab_off = jnp.zeros(step.shape, dtype=bool)

    # Creation (3 -> 5): chord from the new negative to the new positive.
    ok_c, zp_c, zn_c = chord_pair(vb & ~valid, z_b)
    a_c, mx_c, my_c = _segment_terms(zn_c, zp_c, 0j * zero, 0j * zero,
                                     zero, zero, zero, parab_off)

    # Annihilation (5 -> 3): chord from the lost positive to the negative.
    ok_a, zp_a, zn_a = chord_pair(valid & ~vb, z)
    a_a, mx_a, my_a = _segment_terms(zp_a, zn_a, 0j * zero, 0j * zero,
                                     zero, zero, zero, parab_off)

    area = area + jnp.where(ok_c, a_c, 0.0) + jnp.where(ok_a, a_a, 0.0)
    mom_x = mom_x + jnp.where(ok_c, mx_c, 0.0) + jnp.where(ok_a, mx_a, 0.0)
    mom_y = mom_y + jnp.where(ok_c, my_c, 0.0) + jnp.where(ok_a, my_a, 0.0)

    # Chords get a large error (~ |gap|^3) until the limb is refined.
    gap_c = jnp.abs(jax.lax.stop_gradient(zp_c - zn_c))
    gap_a = jnp.abs(jax.lax.stop_gradient(zp_a - zn_a))
    seg_err = seg_err + jnp.where(ok_c, gap_c ** 3, 0.0)
    seg_err = seg_err + jnp.where(ok_a, gap_a ** 3, 0.0)

    # A segment whose image counts do not close needs refinement too.
    n_a = jnp.sum(valid, axis=1)
    n_b = jnp.sum(vb, axis=1)
    unclosed = (n_a != n_b) & ~(ok_c | ok_a)
    seg_err = jnp.where(unclosed, 1.0, seg_err)
    seg_err = jnp.nan_to_num(seg_err, nan=1.0, posinf=1.0)

    return area, mom_x, mom_y, seg_err


def _contour_sums(theta, z, valid, dz, wedge, jac):
    """
    Closed-contour area, moments and segment errors for one disk.

    Parameters
    ----------
    theta : jnp.ndarray, float, shape (N_pts,)
        Sorted limb angles in ``[0, 2 pi)``.
    z, valid, dz, wedge, jac : jnp.ndarray, shape (N_pts, 5)
        Canonical per-sample image data from :func:`_limb_samples`.

    Returns
    -------
    area : float
        Total parity-signed image area (magnification times ``pi rho^2``).
    mom_x, mom_y : float
        First moments of the signed image area.
    seg_err : jnp.ndarray, float, shape (N_pts,)
        Error estimate of each segment (sample ``i`` to ``i + 1``), without
        gradient.
    """
    sample_a = (z, valid, dz, wedge, jac)
    sample_b = tuple(jnp.roll(a, -1, axis=0) for a in sample_a)
    area, mom_x, mom_y, seg_err = _segment_sums(_limb_steps(theta), sample_a,
                                                sample_b)

    return jnp.sum(area), jnp.sum(mom_x), jnp.sum(mom_y), seg_err


def _disk_moments_one(zeta_c, rho, z1, z2, m1, m2, n_init, n_pass, n_add,
                      root_tol):
    """
    Signed image area and moments of one uniform disk (refined limb).

    Parameters
    ----------
    zeta_c : complex
        Disk centre (centre-of-mass frame, Einstein radii).
    rho : float
        Disk radius (Einstein radii).
    z1, z2 : complex
        Lens positions.
    m1, m2 : float
        Lens mass fractions.
    n_init : int
        Initial uniform limb samples (static).
    n_pass : int
        Refinement passes (static).
    n_add : int
        Segments bisected per pass (static).
    root_tol : float
        Lens-equation tolerance.

    Returns
    -------
    moments : jnp.ndarray, float, shape (3,)
        ``[area, mom_x, mom_y]``; the magnification is
        ``area / (pi rho^2)`` and the centroid ``mom / area``.
    """
    sample = partial(_limb_samples, zeta_c, rho, z1=z1, z2=z2, m1=m1, m2=m2,
                     root_tol=root_tol)

    # Uniform starting limb; the pool stays sorted in theta, and seg_err[i]
    # is the error of the segment from sample i to sample i + 1.
    theta = 2.0 * jnp.pi * jnp.arange(n_init) / n_init
    pool, roots = sample(theta)
    _, _, _, seg_err = _contour_sums(theta, *pool)

    # The passes are unrolled, so the pool grows by n_add (static) each time.
    for _ in range(n_pass):
        # Bisect the n_add segments with the largest error.
        _, worst = jax.lax.top_k(seg_err, n_add)
        n_pts = theta.shape[0]
        half = 0.5 * _limb_steps(theta)[worst]
        th_new = jax.lax.stop_gradient(
            jnp.mod(theta[worst] + half, 2.0 * jnp.pi))

        # Warm-start the new roots from the left neighbour's quintic roots.
        new, roots_new = sample(th_new, z_init=roots[worst])

        # Only the two halves of each bisected segment need new errors.
        left = tuple(a[worst] for a in pool)
        right = tuple(a[(worst + 1) % n_pts] for a in pool)
        err_left = _segment_sums(half, left, new)[3]
        err_right = _segment_sums(half, new, right)[3]
        seg_err = seg_err.at[worst].set(err_left)

        # Merge the new samples (and their segment errors) in angle order.
        order = jnp.argsort(jnp.concatenate([theta, th_new]))
        theta = jnp.concatenate([theta, th_new])[order]
        seg_err = jnp.concatenate([seg_err, err_right])[order]
        roots = jnp.concatenate([roots, roots_new], axis=0)[order]
        pool = [jnp.concatenate([a, b], axis=0)[order]
                for a, b in zip(pool, new)]

    # Final contour integral on the refined limb.
    area, mom_x, mom_y, _ = _contour_sums(theta, *pool)

    return jnp.stack([area, mom_x, mom_y])


@partial(jax.jit, static_argnames=("n_init", "n_pass", "n_add",
                                   "batch_size"))
def fsbl_disk_moments(zeta_c, rho, z1, z2, m1, m2, n_init=_N_INIT,
                      n_pass=_N_PASS, n_add=_N_ADD, root_tol=_ROOT_TOL,
                      batch_size=1024):
    """
    Signed image area and first moments of uniform disks (binary lens).

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_rows,)
        Disk centres relative to the binary centre of mass (Einstein radii).
    rho : array_like, float, shape (N_rows,) or ()
        Disk radii (Einstein radii).
    z1, z2 : array_like, complex, shape (N_rows,) or ()
        Lens positions (centre-of-mass frame).
    m1, m2 : array_like, float, shape (N_rows,) or ()
        Lens mass fractions.
    n_init : int, optional
        Initial uniform limb samples per disk (JIT static).
    n_pass : int, optional
        Refinement passes (JIT static).
    n_add : int, optional
        Segments bisected per pass (JIT static). Each disk uses
        ``n_init + n_pass * n_add`` limb samples.
    root_tol : float, optional
        Lens-equation tolerance for physical images.
    batch_size : int, optional
        Disks evaluated together per ``lax.map`` step (bounds memory;
        JIT static).

    Returns
    -------
    moments : jnp.ndarray, float, shape (N_rows, 3)
        ``[area, mom_x, mom_y]`` per disk.
    """
    if int(n_add) > int(n_init):
        raise ValueError("n_add must not exceed n_init.")

    zeta_c = jnp.atleast_1d(jnp.asarray(zeta_c, dtype=jnp.complex128))
    n_rows = zeta_c.shape[0]

    # One value per disk for every geometric input.
    bc = lambda a, dt: jnp.broadcast_to(jnp.asarray(a, dtype=dt), (n_rows,))
    rows = (zeta_c, bc(rho, jnp.float64), bc(z1, jnp.complex128),
            bc(z2, jnp.complex128), bc(m1, jnp.float64), bc(m2, jnp.float64))

    one = partial(_disk_moments_one, n_init=int(n_init), n_pass=int(n_pass),
                  n_add=int(n_add), root_tol=root_tol)

    # Batched map over disks (vmap within a batch).
    return jax.lax.map(lambda r: one(*r), rows,
                       batch_size=min(int(batch_size), n_rows))


def _caustic_samples(z1, z2, m1, m2, n_phi):
    """
    Caustic points, tangents and neighbour matching (traced port).

    JAX counterpart of :func:`bagle.fsbl_contour.caustic_samples`.

    Parameters
    ----------
    z1, z2 : jnp.ndarray, complex, shape (N_geom,)
        Lens positions.
    m1, m2 : jnp.ndarray, float, shape (N_geom,)
        Lens mass fractions.
    n_phi : int
        Critical-curve phase samples (static).

    Returns
    -------
    zeta, zeta_dot : jnp.ndarray, complex, shape (N_geom, n_phi, 4)
        Caustic points and tangents ``d zeta / d phi``.
    nxt : jnp.ndarray, int, shape (N_geom, n_phi, 4)
        Matching root index at the next phase sample.
    """
    n_geom = z1.shape[0]

    # Shapes below: geometry [N_geom, 1], phase [1, n_phi].
    phi = 2.0 * jnp.pi * jnp.arange(n_phi) / n_phi
    e = jnp.exp(-1j * phi)[None, :]
    z1c, z2c = z1[:, None], z2[:, None]
    m1c, m2c = m1[:, None], m2[:, None]

    # Monic quartic (z-z1)^2 (z-z2)^2 - [m1 (z-z2)^2 + m2 (z-z1)^2] / e.
    a1, a0 = -2.0 * z1c, z1c ** 2
    b1, b0 = -2.0 * z2c, z2c ** 2
    p3 = jnp.broadcast_to(a1 + b1, (n_geom, n_phi))
    p2 = a0 + b0 + a1 * b1 - (m1c + m2c) / e
    p1 = a1 * b0 + a0 * b1 - (m1c * b1 + m2c * a1) / e
    p0 = a0 * b0 - (m1c * b0 + m2c * a0) / e

    # Companion-matrix roots, shape [N_geom, n_phi, 4].
    comp = jnp.zeros((n_geom, n_phi, 4, 4), dtype=jnp.complex128)
    comp = comp.at[..., 0, :].set(-jnp.stack([p3, p2, p1, p0], axis=-1))
    comp = comp.at[..., 1, 0].set(1.0).at[..., 2, 1].set(1.0)
    comp = comp.at[..., 3, 2].set(1.0)
    z = jnp.linalg.eigvals(comp)

    # Caustic points and tangents.
    lens = (z1c[..., None], z2c[..., None], m1c[..., None], m2c[..., None])
    alpha, _, kappa_p, _ = _kappa_terms(z, *lens)
    zdot = -1j * e[..., None] / kappa_p
    zeta = z - jnp.conj(alpha)
    zeta_dot = zdot + jnp.conj(e)[..., None] * jnp.conj(zdot)

    # Pairwise matching to the next phase sample (24 permutations).
    z_next = jnp.roll(z, -1, axis=1)
    cost = jnp.sum(jnp.abs(z_next[..., _PERM4] - z[..., None, :]) ** 2,
                   axis=-1)
    nxt = jnp.asarray(_PERM4)[jnp.argmin(cost, axis=-1)]

    return zeta, zeta_dot, nxt


def _tangency_radii(zeta_c, zeta, zeta_dot, nxt):
    """
    Caustic-tangency radii around each source centre (traced port).

    JAX counterpart of :func:`bagle.fsbl_contour.caustic_tangency_radii`.

    Parameters
    ----------
    zeta_c : jnp.ndarray, complex, shape (N_times,)
        Source centres.
    zeta, zeta_dot, nxt : jnp.ndarray, shape (N_geom, n_phi, 4)
        Output of :func:`_caustic_samples`; ``N_geom`` is 1 or ``N_times``.

    Returns
    -------
    r : jnp.ndarray, float, shape (N_times, n_phi * 4)
        Tangency radii, ``-inf`` where a phase interval has no sign change.
    """
    c = zeta_c[:, None, None]

    # Distance and its phase derivative (times distance) at every sample.
    h = jnp.real(jnp.conj(zeta - c) * zeta_dot)
    dist = jnp.abs(zeta - c)

    # Same quantities at the matched root of the next phase sample.
    nxt = jnp.broadcast_to(nxt, h.shape)
    h_next = jnp.take_along_axis(jnp.roll(h, -1, axis=1), nxt, axis=-1)
    d_next = jnp.take_along_axis(jnp.roll(dist, -1, axis=1), nxt, axis=-1)

    # Linear interpolation of the zero of h inside each flipped interval.
    flip = jnp.sign(h) * jnp.sign(h_next) < 0
    f = h / jnp.where(flip, h - h_next, 1.0)
    r = jnp.where(flip, dist + f * (d_next - dist), -jnp.inf)

    return r.reshape(zeta_c.shape[0], -1)


def _ld_break_nodes(r_tan, rho, n_ld, n_brk):
    """
    Radial limb-darkening nodes split at caustic-tangency radii (traced).

    JAX counterpart of :func:`bagle.fsbl_contour.ld_break_nodes`.

    Parameters
    ----------
    r_tan : jnp.ndarray, float, shape (N_times, N_cand)
        Candidate tangency radii (``-inf`` = none).
    rho : jnp.ndarray, float, shape (N_times,)
        Source radii.
    n_ld, n_brk : int
        Nodes per interval and split points per epoch (static).

    Returns
    -------
    s, w : jnp.ndarray, float, shape (N_times, (n_brk + 1) * n_ld)
        Nodes in ``s = sqrt(1 - r^2/rho^2)`` and weights (sum to 1).
    """
    n_times = r_tan.shape[0]

    # The n_brk largest radii strictly inside the source.
    inside = (r_tan > 0) & (r_tan < rho[:, None])
    r_top, _ = jax.lax.top_k(jnp.where(inside, r_tan, -jnp.inf), n_brk)
    valid = jnp.isfinite(r_top)
    s_top = jnp.sqrt(jnp.clip(1.0 - (r_top / rho[:, None]) ** 2, 0.0, 1.0))

    # Interval edges in s; unfilled slots sit at 1 (zero-length).
    edges = jnp.concatenate([jnp.zeros((n_times, 1)),
                             jnp.where(valid, s_top, 1.0),
                             jnp.ones((n_times, 1))], axis=1)
    edges = jnp.sort(edges, axis=1)

    # Spend unused slots bisecting the longest interval (static loop).
    n_valid = jnp.sum(valid, axis=1)
    rows = jnp.arange(n_times)
    for j in range(n_brk):
        k = jnp.argmax(jnp.diff(edges, axis=1), axis=1)
        mid = 0.5 * (edges[rows, k] + edges[rows, k + 1])
        edges = edges.at[:, -1].set(jnp.where(n_valid <= j, mid, 1.0))
        edges = jnp.sort(edges, axis=1).at[:, -1].set(1.0)

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


def ld_radius_nodes(zeta_c, rho, z1, z2, m1, m2, n_ld=_N_LD, n_brk=_N_BRK,
                    n_phi=_N_PHI):
    """
    Caustic-aware radial limb-darkening nodes for a binary lens (traced).

    Parameters
    ----------
    zeta_c : jnp.ndarray, complex, shape (N_times,)
        Source centres.
    rho : float or jnp.ndarray, float, shape () or (N_times,)
        Source radii.
    z1, z2 : complex or jnp.ndarray, shape () or (N_times,)
        Lens positions; scalars (static lens) compute the caustic once.
    m1, m2 : float or jnp.ndarray, shape () or (N_times,)
        Lens mass fractions.
    n_ld, n_brk, n_phi : int, optional
        Nodes per interval, split points per epoch and critical-curve
        phase samples (static).

    Returns
    -------
    s, w : jnp.ndarray, float, shape (N_times, (n_brk + 1) * n_ld)
        Nodes in ``s`` and weights, held constant under differentiation.

    Notes
    -----
    The integrand is continuous at each split, so moving a split point
    changes the integral only at second order; the nodes are therefore
    built under ``stop_gradient``.
    """
    sg = jax.lax.stop_gradient
    geom = [jnp.atleast_1d(sg(jnp.asarray(a))) for a in (z1, z2, m1, m2)]
    z1g = geom[0].astype(jnp.complex128)
    z2g = geom[1].astype(jnp.complex128)
    m1g = geom[2].astype(jnp.float64)
    m2g = geom[3].astype(jnp.float64)
    n_geom = max(a.shape[0] for a in geom)
    bc = lambda a: jnp.broadcast_to(a, (n_geom,))

    # One caustic for a static lens, else one per epoch.
    caus = _caustic_samples(bc(z1g), bc(z2g), bc(m1g), bc(m2g), int(n_phi))
    zeta_c = jnp.atleast_1d(sg(jnp.asarray(zeta_c)))
    r_tan = _tangency_radii(zeta_c, *caus)

    # Source radius per epoch.
    rho = jnp.broadcast_to(sg(jnp.asarray(rho, dtype=jnp.float64)),
                           zeta_c.shape)
    s, w = _ld_break_nodes(r_tan, rho, int(n_ld), int(n_brk))

    return sg(s), sg(w)


@partial(jax.jit, static_argnames=("n_init", "n_pass", "n_add", "n_ld",
                                   "n_brk", "n_phi", "batch_size"))
def fsbl_magnification_centroid(zeta_c, rho, z1, z2, m1, m2, gamma=None,
                                n_ld=_N_LD, n_init=_N_INIT, n_pass=_N_PASS,
                                n_add=_N_ADD, root_tol=_ROOT_TOL,
                                batch_size=1024, n_brk=_N_BRK, n_phi=_N_PHI):
    """
    FSBL magnification and unresolved image centroid (traced).

    JAX counterpart of :func:`bagle.fsbl_contour.adaptive_limb_images_amps`
    (``gamma=None``) and :func:`bagle.fsbl_contour.ld_images_amps`.

    Parameters
    ----------
    zeta_c : array_like, complex, shape (N_times,)
        Source centres relative to the binary centre of mass (Einstein
        radii).
    rho : float or array_like, shape (N_times,)
        Source radius (Einstein radii).
    z1, z2 : complex or array_like, shape (N_times,)
        Lens positions (centre-of-mass frame); per-epoch values allow
        orbital motion.
    m1, m2 : float or array_like, shape (N_times,)
        Lens mass fractions.
    gamma : float, array_like or None, optional
        Linear limb-darkening coefficient (Gamma convention). ``None`` is a
        uniform source; otherwise ``(n_brk + 1) n_ld + 1`` nested uniform
        disks per epoch are combined as in
        :func:`bagle.fspl_contour.ld_images_amps`, with radial nodes split
        at the caustic-tangency radii (:func:`ld_radius_nodes`).
    n_ld : int, optional
        Gauss-Legendre nodes per radial interval (JIT static).
    n_init, n_pass, n_add : int, optional
        Limb schedule per disk (see :func:`fsbl_disk_moments`).
    root_tol : float, optional
        Lens-equation tolerance for physical images.
    batch_size : int, optional
        Disks per ``lax.map`` step (JIT static).
    n_brk, n_phi : int, optional
        Caustic-tangency split points per epoch and critical-curve phase
        samples (JIT static). Scalar lens inputs compute one caustic;
        per-epoch inputs compute one per epoch.

    Returns
    -------
    mag : jnp.ndarray, float, shape (N_times,)
        Total finite-source magnification.
    centroid : jnp.ndarray, float, shape (N_times, 2)
        Brightness-weighted unresolved image centroid ``[E, N]`` relative to
        the centre of mass (Einstein radii).
    """
    zeta_c = jnp.atleast_1d(jnp.asarray(zeta_c, dtype=jnp.complex128))
    n_times = zeta_c.shape[0]
    rho = jnp.broadcast_to(jnp.asarray(rho, dtype=jnp.float64), (n_times,))

    # Lens inputs as given (scalar = static caustic), then one per epoch.
    lens_in = (z1, z2, m1, m2)
    bc = lambda a, dt: jnp.broadcast_to(jnp.asarray(a, dtype=dt), (n_times,))
    z1, z2 = bc(z1, jnp.complex128), bc(z2, jnp.complex128)
    m1, m2 = bc(m1, jnp.float64), bc(m2, jnp.float64)
    kw = dict(n_init=n_init, n_pass=n_pass, n_add=n_add, root_tol=root_tol,
              batch_size=batch_size)

    # Uniform source: one disk per epoch.
    if gamma is None:
        mom = fsbl_disk_moments(zeta_c, rho, z1, z2, m1, m2, **kw)
        mag = mom[:, 0] / (jnp.pi * rho ** 2)
        return mag, mom[:, 1:] / mom[:, :1]

    # Disk radii and brightness weights per epoch (full disk first).
    gamma = bc(gamma, jnp.float64)[:, None]
    s, w_s = ld_radius_nodes(zeta_c, rho, *lens_in, n_ld=n_ld, n_brk=n_brk,
                             n_phi=n_phi)
    radii = jnp.concatenate([rho[:, None],
                             rho[:, None] * jnp.sqrt(1.0 - s ** 2)], axis=1)
    weights = jnp.concatenate([1.0 - gamma, 1.5 * gamma * w_s], axis=1)
    n_disk = radii.shape[1]

    # All (epoch, radius) disks in one call, epoch-major.
    rep = lambda a: jnp.repeat(a, n_disk)
    mom = fsbl_disk_moments(rep(zeta_c), radii.reshape(-1), rep(z1), rep(z2),
                            rep(m1), rep(m2), **kw)
    mom = mom.reshape(n_times, n_disk, 3)

    # Brightness-weighted sums of the signed areas and moments.
    mom_ld = jnp.sum(weights[:, :, None] * mom, axis=1)
    mag = mom_ld[:, 0] / (jnp.pi * rho ** 2)

    return mag, mom_ld[:, 1:] / mom_ld[:, :1]
