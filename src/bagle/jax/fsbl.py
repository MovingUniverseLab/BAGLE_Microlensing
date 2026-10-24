"""FSBL finite-source binary-lens helpers (static limb d1 + adaptive d2).

Milestone (d1): fixed ``n_outline`` limb, Witt (1995) quintic roots via the
companion matrix (shared with :mod:`bagle.jax_physics`), and Bozza-style
parity-weighted polygonal areas. No ``jnp.insert`` and no ``jax.random``.

Milestone (d2): microlux-style adaptive contour with fixed buffers,
linear-sum-assignment image matching, and ``stop_gradient`` on the adaptive
step. Full analytic ``custom_jvp`` / derivative-aware estimator (G-04) can
land on top of this driver.

Attribution: Witt (1995) quintic; contour / segment ideas after Bartolić
``caustics`` (inspiration only; no vendored code); host FSBL in
``model_jax.py``; d2 adaptive strategy and LSA after Ren and Zhu (2025)
microlux (MIT License, Copyright (c) 2024 Haibin Ren; see
:mod:`bagle.jax.linear_sum_assignment`).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from bagle.jax.linear_sum_assignment import find_nearest
from bagle.jax_physics import (
    gaussian_log_likelihood_sum,
    psbl_image_positions,
    psbl_source_position,
    pspl_photometry_from_amp,
)

# Static number of Witt images (binary lens).
_N_IMAGES = 5


def det_and_jac(z, z1, z2, m1, m2):
    """
    Binary-lens Jacobian determinant and parity at image positions.

    Parameters
    ----------
    z : jnp.ndarray, complex, shape = (N_limb, N_images) or (N_times, N_limb, N_images)
        Complex image positions (East + i North) in Einstein radii.
        ``N_images = 5`` for the Witt quintic.
    z1, z2 : complex or jnp.ndarray, shape = () or (N_times,)
        Complex lens positions (East + i North), broadcast onto ``z``.
    m1, m2 : float or jnp.ndarray, shape = () or broadcastable to ``z``
        Lens mass fractions (sum to 1).

    Returns
    -------
    J : jnp.ndarray, float, shape = same as ``z``
        Jacobian determinant ``1 - |dw/dz|^2``.
    parity : jnp.ndarray, float, shape = same as ``z``
        ``sign(Re J)`` in ``{-1, +1}``.
    """
    # Cast inputs; shapes broadcast through the elementwise ops below.
    z = jnp.asarray(z, dtype=jnp.complex128)
    z1 = jnp.asarray(z1, dtype=jnp.complex128)
    z2 = jnp.asarray(z2, dtype=jnp.complex128)
    m1 = jnp.asarray(m1, dtype=jnp.float64)
    m2 = jnp.asarray(m2, dtype=jnp.float64)

    # Complex derivative of the inverse lens map.
    dwdz = m1 / (z - z1) ** 2 + m2 / (z - z2) ** 2

    # Jacobian determinant; parity is the image sign.
    J = 1.0 - jnp.abs(dwdz) ** 2
    parity = jnp.sign(jnp.real(J))

    return J, parity


def polygonal_area_with_parity(pts, valid, theta, parity_edge):
    """
    Signed polygonal area with Bozza parabolic terms, weighted by parity.

    Port of ``FSBL.polygonal_area_with_parity`` for a single image track.

    Parameters
    ----------
    pts : jnp.ndarray, complex, shape = (N_limb,)
        Image-plane contour samples for one image track
        (East + i North).
    valid : jnp.ndarray, bool, shape = (N_limb,)
        True where the root is a physical image.
    theta : jnp.ndarray, float, shape = (N_limb,)
        Source-limb angles (uniform spacing in d1).
    parity_edge : jnp.ndarray, float, shape = (N_limb,)
        Local image parity at each sample (``{-1, +1}``).

    Returns
    -------
    area : float, shape = ()
        Parity-weighted signed area of the track.
    """
    # Closed contour: N_limb samples with uniform limb angle spacing.
    n = pts.shape[0]
    dtheta = theta[1] - theta[0]
    k = jnp.arange(n)
    kp1 = (k + 1) % n

    # Chord from sample k to k+1 (wraps around the limb).
    delta = pts[kp1] - pts

    # Edge parity in {-1, +1}; treat zeros as +1 on valid edges.
    p_edge = jnp.where(valid, jnp.sign(jnp.real(parity_edge)), 0.0)
    p_edge = jnp.where(p_edge == 0.0, 1.0, p_edge)

    # Drop contributions across discontinuous jumps near caustics.
    big_jump = (delta.real ** 2 + delta.imag ** 2) > (0.1 ** 2)
    arc_valid = valid & jnp.roll(valid, -1) & (~big_jump)

    # Trapezoidal Green term (parity-weighted signed area).
    term_t = jnp.imag(jnp.conj(pts) * delta) * p_edge
    a_t = 0.5 * jnp.sum(jnp.where(arc_valid, term_t, 0.0))

    # Parabolic Bozza correction only on smooth interior edges.
    parabolic_ok = (
        jnp.roll(valid, 1)
        & valid
        & jnp.roll(valid, -1)
        & jnp.roll(valid, -2)
        & (~jnp.roll(big_jump, 1))
        & (~big_jump)
        & (~jnp.roll(big_jump, -1))
    )

    # First and second derivatives along the closed contour.
    d1 = jnp.diff(pts, append=pts[0])
    z_p = (d1 + jnp.roll(d1, 1)) / (2.0 * dtheta)
    z_pp = jnp.diff(d1, prepend=d1[-1]) / (dtheta ** 2)

    # ap1: curvature term; ap2: mixed delta / z_p term.
    ap1_term = jnp.imag(jnp.conj(z_p) * z_pp) + jnp.imag(
        jnp.conj(z_p[kp1]) * z_pp[kp1]
    )
    ap1_term = ap1_term * p_edge
    ap1 = (1.0 / 24.0) * jnp.sum(
        jnp.where(parabolic_ok, ap1_term * dtheta ** 3, 0.0)
    )

    ap2_term = jnp.imag(jnp.conj(delta) * (z_p[kp1] - z_p)) * dtheta
    ap2_term = ap2_term * p_edge
    ap2 = (1.0 / 12.0) * jnp.sum(jnp.where(parabolic_ok, ap2_term, 0.0))

    return a_t + 0.5 * (ap1 + ap2)


def _first_nonzero(mask):
    """
    Index of the first True entry, or 0 if none.

    Parameters
    ----------
    mask : jnp.ndarray, bool, shape = (N,)
        1-d boolean mask (e.g. free assignment slots).

    Returns
    -------
    idx : int, shape = ()
        Index of the first True, or 0 if all False.
    """
    # argmax returns the first True for bool arrays (False < True).
    return jnp.argmax(mask)


def match_points(a, b):
    """
    Greedy nearest-neighbor permutation from ``a`` onto ``b``.

    Static ``n = len(a)`` (five Witt images). Matches host
    ``FSBL.match_points`` / caustics ``utils.match_points``.

    Parameters
    ----------
    a : jnp.ndarray, complex, shape = (N_images,)
        Previous limb image positions (East + i North).
        ``N_images = 5`` for the Witt quintic.
    b : jnp.ndarray, complex, shape = (N_images,)
        Current limb image positions (same length as ``a``).

    Returns
    -------
    idcs : jnp.ndarray, int32, shape = (N_images,)
        Indices into ``b`` that align with ``a``
        (``b[idcs]`` matches ``a`` order).
    """
    n = a.shape[0]

    # Pairwise distances: shape (N_images, N_images); row i is |b - a_i|.
    dist = jnp.abs(b[None, :] - a[:, None])
    order = jnp.argsort(dist, axis=1)

    # Greedy assignment without reuse (n is static = 5).
    used = jnp.zeros(n, dtype=bool)
    idcs = jnp.zeros(n, dtype=jnp.int32)

    def body(i, carry):
        # Assign image i to the nearest still-free partner in b.
        used_i, idcs_i = carry
        row = order[i]
        free = ~used_i[row]
        j = _first_nonzero(free)
        idx = row[j].astype(jnp.int32)
        used_i = used_i.at[idx].set(True)
        idcs_i = idcs_i.at[i].set(idx)
        return used_i, idcs_i

    _, idcs = jax.lax.fori_loop(0, n, body, (used, idcs))

    return idcs


def permute_images(z, z_mask, z_parity):
    """
    Track images across the limb by nearest-neighbor matching.

    Parameters
    ----------
    z : jnp.ndarray, complex, shape = (N_images, N_limb)
        Image positions at each limb sample (East + i North).
        ``N_images = 5`` for the Witt quintic.
    z_mask : jnp.ndarray, bool, shape = (N_images, N_limb)
        True where the root is a physical image.
    z_parity : jnp.ndarray, float, shape = (N_images, N_limb)
        Image parity ``sign(Re J)`` at each sample.

    Returns
    -------
    z_o : jnp.ndarray, complex, shape = (N_images, N_limb)
        Image tracks reordered so each row is continuous in limb angle.
    z_mask_o : jnp.ndarray, bool, shape = (N_images, N_limb)
        Permuted validity mask.
    z_parity_o : jnp.ndarray, float, shape = (N_images, N_limb)
        Permuted parity.
    """
    # Pack channels for scan: shape (3, n_images, n_outline).
    xs = jnp.stack([z, z_mask.astype(z.dtype), z_parity.astype(z.dtype)])

    def apply_match(carry, xs_slice):
        # xs_slice: (3, n_images) at one limb sample.
        z_slice, z_mask_slice, z_parity_slice = xs_slice

        # Match current images onto the previous limb sample (carry).
        idcs = match_points(carry, z_slice)

        # Reorder all channels; new carry is the permuted z.
        z_perm = z_slice[idcs]
        out = jnp.stack(
            [z_perm, z_mask_slice[idcs], z_parity_slice[idcs]]
        )
        return z_perm, out

    # Seed carry with the first limb sample's image positions.
    init = xs[0, :, 0]

    # Scan over outline: xs_scan shape (n_outline, 3, n_images).
    xs_scan = jnp.moveaxis(xs, -1, 0)
    _, xs_out = jax.lax.scan(apply_match, init, xs_scan)

    # xs_out: (n_outline, 3, n_images) -> three (n_outline, n_images) arrays.
    z_o, z_mask_o, z_parity_o = jnp.moveaxis(xs_out, 1, 0)

    # Back to (n_images, n_outline); mask/parity were stored as complex.
    return (
        z_o.T,
        z_mask_o.real.astype(bool).T,
        z_parity_o.real.T,
    )


def images_on_source_limb_static(w0, z1, z2, m1, m2, rho, n_outline,
                                 root_tol=1e-10):
    """
    Solve the Witt quintic on a fixed source-limb outline (d1).

    Parameters
    ----------
    w0 : complex, shape = ()
        Source center in Einstein radii (East + i North).
    z1, z2 : complex, shape = ()
        Lens positions relative to the binary center of mass.
    m1, m2 : float, shape = ()
        Mass fractions (sum to 1).
    rho : float, shape = ()
        Source radius in Einstein radii.
    n_outline : int
        Static limb sample count ``N_limb`` (concrete Python int for jit).
    root_tol : float, optional, shape = ()
        Lens-equation root mask tolerance.

    Returns
    -------
    z : jnp.ndarray, complex, shape = (N_images, N_limb)
        Image positions (East + i North); ``N_images = 5``.
    z_mask : jnp.ndarray, bool, shape = (N_images, N_limb)
        True where the root is a physical image.
    z_parity : jnp.ndarray, float, shape = (N_images, N_limb)
        Image parity ``sign(Re J)`` at each sample.
    theta : jnp.ndarray, float, shape = (N_limb,)
        Limb angles on ``[-pi, pi)``.
    """
    # Concrete int so jit can treat N_limb as static.
    n_outline = int(n_outline)

    # Closed limb: [-pi, pi) with the last sample just below +pi.
    theta = jnp.linspace(-jnp.pi, jnp.pi, n_outline, endpoint=False)
    w_limb = w0 + rho * jnp.exp(1j * theta)

    # Broadcast lenses to the limb length for the shared PSBL solver.
    z1_b = jnp.broadcast_to(jnp.asarray(z1, dtype=jnp.complex128), (n_outline,))
    z2_b = jnp.broadcast_to(jnp.asarray(z2, dtype=jnp.complex128), (n_outline,))
    m1_b = jnp.broadcast_to(jnp.asarray(m1, dtype=jnp.float64), (n_outline,))
    m2_b = jnp.broadcast_to(jnp.asarray(m2, dtype=jnp.float64), (n_outline,))
    tol_b = jnp.broadcast_to(
        jnp.asarray(root_tol, dtype=jnp.float64), (n_outline,)
    )

    # Witt roots: shape (N_limb, N_images); NaN marks non-physical roots.
    z_all = psbl_image_positions(
        w_limb, z1_b, z2_b, m1_b, m2_b, tol_b, check_sols=True
    )

    # Contour code expects (N_images, N_limb).
    z = z_all.T
    z_mask = (~jnp.isnan(z_all.real)).T

    # Parity from the Jacobian at each image sample.
    _J, parity = det_and_jac(z_all, z1, z2, m1, m2)
    z_parity = parity.T

    return z, z_mask, z_parity, theta


def magnification_one_static(w0, z1, z2, m1, m2, rho, n_outline,
                             root_tol=1e-10):
    """
    Extended-source magnification for one epoch (static limb, d1).

    Parameters
    ----------
    w0 : complex, shape = ()
        Source center in Einstein radii (East + i North).
    z1, z2 : complex, shape = ()
        Lens positions relative to the binary center of mass.
    m1, m2 : float, shape = ()
        Mass fractions (sum to 1).
    rho : float, shape = ()
        Source radius in Einstein radii.
    n_outline : int
        Static limb sample count ``N_limb``.
    root_tol : float, optional, shape = ()
        Lens-equation root mask tolerance.

    Returns
    -------
    mag : float, shape = ()
        Total finite-source magnification.
    """
    # Solve limb roots then order image tracks by nearest neighbor.
    z_raw, z_mask, z_parity, theta = images_on_source_limb_static(
        w0, z1, z2, m1, m2, rho, n_outline, root_tol=root_tol
    )
    z_perm, z_mask_perm, z_parity_perm = permute_images(
        z_raw, z_mask, z_parity
    )

    # Normalize by source area so mag -> point-source limit as rho -> 0.
    inv_src = 1.0 / (jnp.pi * rho ** 2)

    def area_k(k):
        # Skip empty tracks (all roots invalid for this image).
        n_valid = jnp.sum(z_mask_perm[k, :])
        area = jax.lax.cond(
            n_valid > 0,
            lambda: polygonal_area_with_parity(
                z_perm[k],
                z_mask_perm[k, :],
                theta,
                z_parity_perm[k, :],
            ),
            lambda: jnp.array(0.0, dtype=jnp.float64),
        )
        return area

    # Shape of areas: (N_images,).
    areas = jax.vmap(area_k)(jnp.arange(_N_IMAGES))

    # Host uses abs(sum of parity-weighted closed/open integrals).
    mag = jnp.abs(jnp.sum(areas)) * inv_src

    return mag


def fsbl_magnification(w0, z1, z2, m1, m2, rho, n_outline, root_tol=1e-10):
    """
    Vectorized static-limb FSBL magnification over epochs.

    Parameters
    ----------
    w0 : array_like, complex, shape = (N_times,)
        Source centers in Einstein radii (East + i North).
    z1, z2 : array_like, complex, shape = (N_times,) or ()
        Lens positions (COM frame), broadcast to ``N_times``.
    m1, m2 : array_like, float, shape = (N_times,) or ()
        Mass fractions, broadcast to ``N_times``.
    rho : float, shape = ()
        Source radius in Einstein radii (shared across epochs).
    n_outline : int
        Static limb count ``N_limb`` (concrete int).
    root_tol : float, optional, shape = ()
        Root mask tolerance.

    Returns
    -------
    mag : jnp.ndarray, float, shape = (N_times,)
        Total magnification at each epoch.
    """
    # Shape of w0: (N_times,).
    w0 = jnp.atleast_1d(jnp.asarray(w0, dtype=jnp.complex128))
    n = w0.shape[0]

    # Broadcast lens geometry to one value per epoch.
    z1 = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(z1, dtype=jnp.complex128)).ravel()[:n]
        if jnp.size(z1) > 1
        else jnp.atleast_1d(jnp.asarray(z1, dtype=jnp.complex128)),
        (n,),
    )
    z2 = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(z2, dtype=jnp.complex128)).ravel()[:n]
        if jnp.size(z2) > 1
        else jnp.atleast_1d(jnp.asarray(z2, dtype=jnp.complex128)),
        (n,),
    )
    m1 = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(m1, dtype=jnp.float64)), (n,)
    )
    m2 = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(m2, dtype=jnp.float64)), (n,)
    )

    # Freeze rho / N_limb / tol as Python scalars for the static jit path.
    mag_one = partial(
        magnification_one_static,
        rho=float(rho),
        n_outline=int(n_outline),
        root_tol=float(root_tol),
    )

    # One magnification per epoch.
    return jax.vmap(mag_one)(w0, z1, z2, m1, m2)


def fsbl_magnification_from_u(u_vec, z1, z2, m1, m2, rho, n_outline,
                              root_tol=1e-10):
    """
    FSBL magnification from real ``u`` vectors (East, North).

    Parameters
    ----------
    u_vec : array_like, float, shape = (N_times, 2)
        Source-lens COM separation in Einstein radii
        (``[..., 0]`` East, ``[..., 1]`` North).
    z1, z2 : array_like, complex, shape = (N_times,) or ()
        Lens positions (COM frame), broadcast to ``N_times``.
    m1, m2 : array_like, float, shape = (N_times,) or ()
        Mass fractions, broadcast to ``N_times``.
    rho : float, shape = ()
        Source radius in Einstein radii.
    n_outline : int
        Static limb count ``N_limb``.
    root_tol : float, optional, shape = ()
        Root mask tolerance.

    Returns
    -------
    mag : jnp.ndarray, float, shape = (N_times,)
        Total magnification at each epoch.
    """
    # Pack real (East, North) into complex source centers.
    u_vec = jnp.asarray(u_vec, dtype=jnp.float64)
    w0 = u_vec[..., 0] + 1j * u_vec[..., 1]

    return fsbl_magnification(
        w0, z1, z2, m1, m2, rho, n_outline, root_tol=root_tol
    )


# ---------------------------------------------------------------------------
# Milestone (d2) adaptive contour (microlux-style fixed buffers)
# ---------------------------------------------------------------------------

# Fixed buffers matching microlux defaults (Ren and Zhu 2025).
_D2_N_START = 30
_D2_N_BUFFER = 480
# Gap-based refinement: add up to N_PER midpoints per iteration.
_D2_N_PER = 4
_D2_N_ITER = 40


def match_points_lsa(a, b, parity_a, parity_b):
    """
    Linear-sum-assignment match from ``a`` onto ``b`` (d2).

    Parameters
    ----------
    a, b : jnp.ndarray, complex, shape = (N_images,)
        Previous and current limb image positions.
    parity_a, parity_b : jnp.ndarray, float, shape = (N_images,)
        Image parities at those samples.

    Returns
    -------
    idcs : jnp.ndarray, int, shape = (N_images,)
        Indices into ``b`` that align with ``a``.
    """
    return find_nearest(a, parity_a, b, parity_b)


def permute_images_lsa(z, z_mask, z_parity):
    """
    Track images across the limb by linear-sum assignment (d2).

    Parameters
    ----------
    z : jnp.ndarray, complex, shape = (N_images, N_limb)
        Image positions at each limb sample.
    z_mask : jnp.ndarray, bool, shape = (N_images, N_limb)
        Validity mask.
    z_parity : jnp.ndarray, float, shape = (N_images, N_limb)
        Image parity at each sample.

    Returns
    -------
    z_o : jnp.ndarray, complex, shape = (N_images, N_limb)
        Continuously ordered image tracks.
    z_mask_o : jnp.ndarray, bool, shape = (N_images, N_limb)
        Permuted validity mask.
    z_parity_o : jnp.ndarray, float, shape = (N_images, N_limb)
        Permuted parity.
    """
    # Pack channels: shape (3, N_images, N_limb).
    xs = jnp.stack([z, z_mask.astype(z.dtype), z_parity.astype(z.dtype)])

    def apply_match(carry, xs_slice):
        # carry: (z_prev, parity_prev); xs_slice: (3, N_images).
        z_prev, parity_prev = carry
        z_slice, z_mask_slice, z_parity_slice = xs_slice

        idcs = match_points_lsa(z_prev, z_slice, parity_prev, z_parity_slice)

        z_perm = z_slice[idcs]
        p_perm = z_parity_slice[idcs]
        out = jnp.stack([z_perm, z_mask_slice[idcs], p_perm])
        return (z_perm, p_perm), out

    init = (xs[0, :, 0], xs[2, :, 0])
    xs_scan = jnp.moveaxis(xs, -1, 0)
    _, xs_out = jax.lax.scan(apply_match, init, xs_scan)

    z_o, z_mask_o, z_parity_o = jnp.moveaxis(xs_out, 1, 0)

    return (
        z_o.T,
        z_mask_o.real.astype(bool).T,
        z_parity_o.real.T,
    )


def _solve_limb_buffer(w0, z1, z2, m1, m2, rho, theta, n_valid, root_tol):
    """
    Solve Witt roots on a packed limb buffer (first ``n_valid`` angles).

    Parameters
    ----------
    w0, z1, z2, m1, m2, rho, root_tol
        Same as :func:`images_on_source_limb_static`.
    theta : jnp.ndarray, float, shape = (N_buffer,)
        Limb angles; only ``theta[:n_valid]`` are used.
    n_valid : int
        Number of active limb samples (concrete or traced).

    Returns
    -------
    z : jnp.ndarray, complex, shape = (N_images, N_buffer)
        Image positions; inactive columns are NaN.
    z_mask : jnp.ndarray, bool, shape = (N_images, N_buffer)
        Validity mask (False for inactive columns).
    z_parity : jnp.ndarray, float, shape = (N_images, N_buffer)
        Parity; inactive columns are 0.
    """
    n_buffer = theta.shape[0]
    active = jnp.arange(n_buffer) < n_valid

    # Inactive slots get a dummy source; their roots are masked out.
    th = jnp.where(active, theta, 0.0)
    w_limb = w0 + rho * jnp.exp(1j * th)

    z1_b = jnp.broadcast_to(jnp.asarray(z1, dtype=jnp.complex128), (n_buffer,))
    z2_b = jnp.broadcast_to(jnp.asarray(z2, dtype=jnp.complex128), (n_buffer,))
    m1_b = jnp.broadcast_to(jnp.asarray(m1, dtype=jnp.float64), (n_buffer,))
    m2_b = jnp.broadcast_to(jnp.asarray(m2, dtype=jnp.float64), (n_buffer,))
    tol_b = jnp.broadcast_to(
        jnp.asarray(root_tol, dtype=jnp.float64), (n_buffer,)
    )

    z_all = psbl_image_positions(
        w_limb, z1_b, z2_b, m1_b, m2_b, tol_b, check_sols=True
    )
    # Shape (N_images, N_buffer).
    z = z_all.T
    z_mask = (~jnp.isnan(z_all.real)).T & active[None, :]
    z = jnp.where(z_mask, z, jnp.nan + 1j * jnp.nan)

    _J, parity = det_and_jac(z_all, z1, z2, m1, m2)
    z_parity = jnp.where(z_mask, parity.T, 0.0)

    return z, z_mask, z_parity


def _mag_from_limb(z, z_mask, z_parity, theta, n_valid, rho, n_fill,
                   use_lsa=True):
    """
    Parity-weighted polygonal magnification on a packed limb buffer.

    Parameters
    ----------
    z, z_mask, z_parity : jnp.ndarray
        Shapes (N_images, N_buffer).
    theta : jnp.ndarray, float, shape = (N_buffer,)
        Limb angles (active samples packed at the front).
    n_valid : int
        Active sample count (may be traced).
    rho : float
        Source radius in Einstein radii.
    n_fill : int
        Concrete upper bound on active samples
        (``min(n_start + n_iter * n_per, n_buffer)``).
    use_lsa : bool, optional
        If True, match tracks with linear-sum assignment; else greedy.

    Returns
    -------
    mag : float, shape = ()
        Total finite-source magnification.
    """
    # Concrete slice so permute/area see a fixed limb length.
    n_fill = int(n_fill)
    z_f = z[:, :n_fill]
    mask_f = z_mask[:, :n_fill]
    parity_f = z_parity[:, :n_fill]
    th_f = theta[:n_fill]

    # Drop slots beyond the traced n_valid; zero-fill for safe arithmetic.
    active = jnp.arange(n_fill) < n_valid
    mask_f = mask_f & active[None, :]
    z_f = jnp.where(mask_f, z_f, 0.0 + 0.0j)
    parity_f = jnp.where(mask_f, parity_f, 0.0)
    th_f = jnp.where(active, th_f, th_f[0])

    # stop_gradient on limb spacing (adaptive step size / parabolic scale).
    dth_mean = jax.lax.stop_gradient(
        jnp.mean(jnp.diff(th_f, append=th_f[0] + 2.0 * jnp.pi))
    )
    th_area = th_f[0] + dth_mean * jnp.arange(n_fill, dtype=jnp.float64)

    if use_lsa:
        z_perm, z_mask_perm, z_parity_perm = permute_images_lsa(
            z_f, mask_f, parity_f
        )
    else:
        z_perm, z_mask_perm, z_parity_perm = permute_images(
            z_f, mask_f, parity_f
        )

    inv_src = 1.0 / (jnp.pi * rho ** 2)

    def area_k(k):
        n_ok = jnp.sum(z_mask_perm[k, :])
        area = jax.lax.cond(
            n_ok > 0,
            lambda: polygonal_area_with_parity(
                z_perm[k],
                z_mask_perm[k, :],
                th_area,
                z_parity_perm[k, :],
            ),
            lambda: jnp.array(0.0, dtype=jnp.float64),
        )
        return area

    areas = jax.vmap(area_k)(jnp.arange(_N_IMAGES))
    mag = jnp.abs(jnp.sum(areas)) * inv_src

    return mag


def images_on_source_limb_adaptive(w0, z1, z2, m1, m2, rho, n_start=_D2_N_START,
                                   n_buffer=_D2_N_BUFFER, n_iter=_D2_N_ITER,
                                   n_per=_D2_N_PER, root_tol=1e-10):
    """
    Adaptive limb samples in a fixed buffer (gap-based refinement).

    Starts with ``n_start`` uniform angles and, for ``n_iter`` passes, inserts
    up to ``n_per`` midpoints in the largest image-plane gaps. Decisions use
    ``stop_gradient`` so the adaptive schedule does not pollute derivatives.
    No ``jnp.insert`` and no ``jax.random``.

    Parameters
    ----------
    w0 : complex, shape = ()
        Source center in Einstein radii (East + i North).
    z1, z2 : complex, shape = ()
        Lens positions relative to the binary center of mass.
    m1, m2 : float, shape = ()
        Mass fractions (sum to 1).
    rho : float, shape = ()
        Source radius in Einstein radii.
    n_start : int, optional
        Initial limb samples.
    n_buffer : int, optional
        Static buffer capacity.
    n_iter : int, optional
        Number of refinement passes.
    n_per : int, optional
        Midpoints added per pass (microlux ``N_max``-style cap).
    root_tol : float, optional, shape = ()
        Root mask tolerance.

    Returns
    -------
    z : jnp.ndarray, complex, shape = (N_images, N_buffer)
        Image positions on the refined limb.
    z_mask : jnp.ndarray, bool, shape = (N_images, N_buffer)
        Validity mask.
    z_parity : jnp.ndarray, float, shape = (N_images, N_buffer)
        Image parity.
    theta : jnp.ndarray, float, shape = (N_buffer,)
        Limb angles (inactive slots are ``+inf``).
    n_valid : int
        Number of active limb samples.
    """
    n_start = int(n_start)
    n_buffer = int(n_buffer)
    n_iter = int(n_iter)
    n_per = int(n_per)

    # Packed buffer: finite angles in front, +inf padding at the end.
    theta0 = jnp.full((n_buffer,), jnp.inf, dtype=jnp.float64)
    theta0 = theta0.at[:n_start].set(
        jnp.linspace(-jnp.pi, jnp.pi, n_start, endpoint=False)
    )
    n_valid0 = n_start

    def refine_body(carry, _unused):
        theta, n_valid = carry

        z, z_mask, z_parity = _solve_limb_buffer(
            w0, z1, z2, m1, m2, rho, theta, n_valid, root_tol
        )

        # Match tracks before measuring image gaps (roots are unordered).
        z_pack = jnp.where(z_mask, z, 0.0 + 0.0j)
        z_perm, z_mask_perm, _z_parity_perm = permute_images_lsa(
            z_pack, z_mask, z_parity
        )

        # Interval scores: matched image-plane gap and source-angle gap.
        # Shape of delta_z: (N_images, N_buffer-1).
        delta_z = jnp.abs(z_perm[:, 1:] - z_perm[:, :-1])
        pair_ok = z_mask_perm[:, 1:] & z_mask_perm[:, :-1]
        delta_z = jnp.where(pair_ok, delta_z, 0.0)
        gap_active = jnp.arange(n_buffer - 1) < (n_valid - 1)
        delta_z_max = jnp.max(delta_z, axis=0)

        # Source-angle spacing on the packed active samples.
        dth = theta[1:] - theta[:-1]
        last = jnp.maximum(n_valid - 1, 0)
        dth_wrap = (theta[0] + 2.0 * jnp.pi) - theta[last]

        # Combine scores; angle term prevents pathological clustering.
        score = jnp.maximum(delta_z_max, dth)
        score = jnp.where(gap_active, score, -1.0)
        score = score.at[last - 1].set(
            jnp.maximum(score[last - 1], dth_wrap)
        )

        # Indices of the largest gaps (stop_gradient: adaptive schedule).
        idcs = jax.lax.stop_gradient(jnp.argsort(score)[::-1][:n_per])

        # Midpoints; wrap picks use theta[0]+2pi as the right endpoint.
        th_a = theta[idcs]
        th_b = jnp.where(
            idcs == last - 1,
            theta[0] + 2.0 * jnp.pi,
            theta[idcs + 1],
        )
        th_new = 0.5 * (th_a + th_b)
        th_new = jnp.where(th_new > jnp.pi, th_new - 2.0 * jnp.pi, th_new)

        # Room left in the buffer.
        n_add = jnp.minimum(n_per, n_buffer - n_valid)
        th_new = jnp.where(jnp.arange(n_per) < n_add, th_new, jnp.inf)

        # Scatter midpoints at dynamic indices (slices must be static in JAX).
        write_at = n_valid + jnp.arange(n_per)
        write_at = jnp.minimum(write_at, n_buffer - 1)
        theta2 = theta.at[write_at].set(th_new)

        # Re-sort so finite angles pack to the front.
        order = jnp.argsort(theta2)
        theta_out = theta2[order]
        n_valid_out = jnp.minimum(n_valid + n_add, n_buffer)

        return (theta_out, n_valid_out), None

    (theta, n_valid), _ = jax.lax.scan(
        refine_body, (theta0, n_valid0), None, length=n_iter
    )

    z, z_mask, z_parity = _solve_limb_buffer(
        w0, z1, z2, m1, m2, rho, theta, n_valid, root_tol
    )

    return z, z_mask, z_parity, theta, n_valid


def magnification_one_adaptive(w0, z1, z2, m1, m2, rho, n_start=_D2_N_START,
                               n_buffer=_D2_N_BUFFER, n_iter=_D2_N_ITER,
                               n_per=_D2_N_PER, root_tol=1e-10):
    """
    Extended-source magnification with microlux-style adaptive limb (d2).

    Parameters
    ----------
    w0 : complex, shape = ()
        Source center in Einstein radii (East + i North).
    z1, z2 : complex, shape = ()
        Lens positions relative to the binary center of mass.
    m1, m2 : float, shape = ()
        Mass fractions (sum to 1).
    rho : float, shape = ()
        Source radius in Einstein radii.
    n_start : int, optional
        Initial limb samples before refinement.
    n_buffer : int, optional
        Static buffer capacity for refined samples
        (padded limb length ``N_limb <= n_buffer``).
    n_iter : int, optional
        Gap-based refinement passes.
    n_per : int, optional
        Midpoints added per pass.
    root_tol : float, optional, shape = ()
        Root mask tolerance.

    Returns
    -------
    mag : float, shape = ()
        Total finite-source magnification.

    Notes
    -----
    Uses fixed buffers and LSA image matching after Ren and Zhu (2025)
    microlux (MIT). Analytic ``custom_jvp`` / derivative-aware error control
    for caustic crossings (plan G-04) is not yet wired.
    """
    n_start_i = int(n_start)
    n_buffer_i = int(n_buffer)
    n_iter_i = int(n_iter)
    n_per_i = int(n_per)
    n_fill = min(n_start_i + n_iter_i * n_per_i, n_buffer_i)

    z, z_mask, z_parity, theta, n_valid = images_on_source_limb_adaptive(
        w0, z1, z2, m1, m2, rho,
        n_start=n_start_i, n_buffer=n_buffer_i, n_iter=n_iter_i,
        n_per=n_per_i, root_tol=root_tol,
    )

    return _mag_from_limb(
        z, z_mask, z_parity, theta, n_valid, rho, n_fill
    )


def fsbl_magnification_adaptive(w0, z1, z2, m1, m2, rho, n_start=_D2_N_START,
                                n_buffer=_D2_N_BUFFER, n_iter=_D2_N_ITER,
                                n_per=_D2_N_PER, root_tol=1e-10):
    """
    Vectorized adaptive-limb FSBL magnification (d2).

    Parameters
    ----------
    w0 : array_like, complex, shape = (N_times,)
        Source centers in Einstein radii (East + i North).
    z1, z2 : array_like, complex, shape = (N_times,) or ()
        Lens positions (COM frame), broadcast to ``N_times``.
    m1, m2 : array_like, float, shape = (N_times,) or ()
        Mass fractions, broadcast to ``N_times``.
    rho : float, shape = ()
        Source radius in Einstein radii.
    n_start, n_buffer, n_iter, n_per : int, optional
        Adaptive limb controls (see :func:`magnification_one_adaptive`).
    root_tol : float, optional, shape = ()
        Root mask tolerance.

    Returns
    -------
    mag : jnp.ndarray, float, shape = (N_times,)
        Total magnification at each epoch.
    """
    # Shape of w0: (N_times,).
    w0 = jnp.atleast_1d(jnp.asarray(w0, dtype=jnp.complex128))
    n = w0.shape[0]

    z1 = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(z1, dtype=jnp.complex128)).ravel()[:n]
        if jnp.size(z1) > 1
        else jnp.atleast_1d(jnp.asarray(z1, dtype=jnp.complex128)),
        (n,),
    )
    z2 = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(z2, dtype=jnp.complex128)).ravel()[:n]
        if jnp.size(z2) > 1
        else jnp.atleast_1d(jnp.asarray(z2, dtype=jnp.complex128)),
        (n,),
    )
    m1 = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(m1, dtype=jnp.float64)), (n,)
    )
    m2 = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(m2, dtype=jnp.float64)), (n,)
    )

    # Freeze adaptive controls as Python scalars for the static jit path.
    mag_one = partial(
        magnification_one_adaptive,
        rho=float(rho),
        n_start=int(n_start),
        n_buffer=int(n_buffer),
        n_iter=int(n_iter),
        n_per=int(n_per),
        root_tol=float(root_tol),
    )

    return jax.vmap(mag_one)(w0, z1, z2, m1, m2)


def fsbl_log_likely_photometry(t, t0, tE, u0, thetaE_hat, z1, z2, m1, m2,
                               rho, n_outline, mag_src, b_sff, mag_obs,
                               mag_err, parallax_vectors=None, piE_E=0.0,
                               piE_N=0.0, gp_params=None, fixed_jitter=True,
                               root_tol=1e-10, gamma=None):
    """
    Traced photometric log-likelihood with static-limb FSBL magnification.

    Notes
    -----
    The authoritative host path is :mod:`bagle.fsbl_contour` (adaptive
    Bozza/VBBL contours). This traced kernel is the legacy static-limb
    d1 path; a full JAX port of ``fsbl_contour`` (per-row ``rho``,
    creation/annihilation chords, nested-disk ``gamma``) remains open.
    ``gamma`` is accepted for API compatibility and currently ignored
    (uniform source).

    Parameters
    ----------
    t : array_like, float, shape = (N_times,)
        Observation times (MJD).
    t0 : float, shape = ()
        Time of closest approach.
    tE : float, shape = ()
        Einstein crossing time.
    u0 : array_like, float, shape = (2,)
        Closest-approach vector (East, North) in Einstein radii.
    thetaE_hat : array_like, float, shape = (2,)
        Unit lens proper-motion direction (East, North).
    z1, z2 : complex or array_like, shape = () or (N_times,)
        Lens positions in Einstein radii (COM frame).
    m1, m2 : float, shape = ()
        Mass fractions (sum to 1).
    rho : float, shape = ()
        Source radius in Einstein radii.
    n_outline : int
        Static limb count ``N_limb``.
    mag_src : float or array_like, shape = () or (N_times,)
        Source magnitude (unlensed).
    b_sff : float or array_like, shape = () or (N_times,)
        Source flux fraction.
    mag_obs : array_like, float, shape = (N_times,)
        Observed magnitudes.
    mag_err : array_like, float, shape = (N_times,)
        Magnitude uncertainties.
    parallax_vectors : array_like, float, shape = (N_times, 2), optional
        Precomputed parallax vectors (East, North); ``None`` skips parallax.
    piE_E, piE_N : float, optional, shape = ()
        Parallax components when ``parallax_vectors`` is used.
    gp_params : optional
        GP hyperparameters; ``None`` uses a plain Gaussian likelihood.
    fixed_jitter : bool, optional
        Passed to the GP path when ``gp_params`` is not ``None``.
    root_tol : float, optional, shape = ()
        Root mask tolerance for the Witt solver.
    gamma : float, optional
        Linear limb-darkening coefficient (unused until the contour port).

    Returns
    -------
    lnL : float, shape = ()
        Summed Gaussian photometric log-likelihood.
    """
    # Complex source track in Einstein radii (East + i North).
    # Shape of w0: (N_times,).
    w0 = psbl_source_position(
        t, t0, tE, u0, thetaE_hat,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N,
    )

    # Static-limb FSBL magnification, then blended model magnitudes.
    amp = fsbl_magnification(
        w0, z1, z2, m1, m2, rho, n_outline, root_tol=root_tol
    )
    mag_model = pspl_photometry_from_amp(amp, mag_src, b_sff)

    # Plain Gaussian or GP residual likelihood.
    if gp_params is None:
        return gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)

    from bagle.jax.gp import gp_log_likely_photometry

    return gp_log_likely_photometry(
        t, mag_obs, mag_err, mag_model, gp_params, fixed_jitter=fixed_jitter
    )
