"""FSBL finite-source binary-lens helpers (static limb d1 + adaptive d2).

Milestone (d1): fixed ``n_outline`` limb, Witt (1995) quintic roots via the
companion matrix (shared with :mod:`bagle.jax_physics`), and Bozza-style
parity-weighted polygonal areas. No ``jnp.insert`` and no ``jax.random``.

Milestone (d2): microlux-style adaptive contour with ``custom_jvp`` roots
and assignment matching lands on top of the same area driver.

Attribution: Witt (1995) quintic; contour / segment ideas after Bartolić
``caustics`` (inspiration only; no vendored code); host FSBL in
``model_jax.py``; d2 after Ren and Zhu (2025) microlux once licensed.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

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
# Milestone (d2) adaptive contour (stubs; filled after microlux license check)
# ---------------------------------------------------------------------------

# Fixed buffers matching microlux defaults (Ren and Zhu 2025).
_D2_N_START = 30
_D2_N_BUFFER = 480


def magnification_one_adaptive(w0, z1, z2, m1, m2, rho, n_start=_D2_N_START,
                               n_buffer=_D2_N_BUFFER, root_tol=1e-10):
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
    root_tol : float, optional, shape = ()
        Root mask tolerance.

    Returns
    -------
    mag : float, shape = ()
        Total finite-source magnification.

    Notes
    -----
    Not implemented yet. Plan (d2): implicit-differentiation roots
    (``custom_jvp``), ``stop_gradient`` on the adaptive step, a
    derivative-aware error estimate, and linear-sum-assignment image
    matching. Until then callers should use
    :func:`magnification_one_static`.
    """
    # Stub until d2 adaptive contour lands (microlux license check).
    raise NotImplementedError(
        "FSBL d2 adaptive contour is not implemented yet; "
        "use magnification_one_static (d1)."
    )


def fsbl_magnification_adaptive(w0, z1, z2, m1, m2, rho, n_start=_D2_N_START,
                                n_buffer=_D2_N_BUFFER, root_tol=1e-10):
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
    n_start, n_buffer : int, optional
        Adaptive limb start count and static buffer capacity.
    root_tol : float, optional, shape = ()
        Root mask tolerance.

    Returns
    -------
    mag : jnp.ndarray, float, shape = (N_times,)
        Total magnification at each epoch.
    """
    # Stub: will vmap magnification_one_adaptive once d2 is implemented.
    raise NotImplementedError(
        "FSBL d2 adaptive contour is not implemented yet; "
        "use fsbl_magnification (d1)."
    )


def fsbl_log_likely_photometry(t, t0, tE, u0, thetaE_hat, z1, z2, m1, m2,
                               rho, n_outline, mag_src, b_sff, mag_obs,
                               mag_err, parallax_vectors=None, piE_E=0.0,
                               piE_N=0.0, gp_params=None, fixed_jitter=True,
                               root_tol=1e-10):
    """
    Traced photometric log-likelihood with static-limb FSBL magnification.

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
