"""JAX hyper-Catalan image-position tracker.

Same step as :mod:`bagle.hyper_catalan`: shifted series (eq. 9), a
local quadratic or cubic when roots cluster, then Newton. Epochs are
a ``lax.scan``. Gradients use the implicit root formula
``dz = -P_theta / P_z`` and do not flow through the first-epoch
companion eigendecomposition.
"""

from functools import partial

import jax
import jax.numpy as jnp

from bagle.hyper_catalan import (
    _CLUSTER_SEP,
    _COLLAPSE_SEP,
    _MAX_SUB,
    _RES_TOL,
    _terms,
)
from bagle.jax_physics import quintic_coefficients, quintic_roots_companion


def _horner(coeffs, z):
    """
    ``P(z)`` and ``P'(z)`` by Horner.

    Parameters
    ----------
    coeffs : array, complex, shape (degree + 1,)
        Low-to-high coefficients.
    z : array, complex, shape (n_roots,)
        Evaluation points.

    Returns
    -------
    p, dp : arrays, complex, shape (n_roots,)
        Value and derivative.
    """
    n = coeffs.shape[0] - 1
    p = jnp.broadcast_to(coeffs[n], z.shape).astype(jnp.complex128)
    dp = jnp.zeros(z.shape, dtype=jnp.complex128)

    for k in range(n - 1, -1, -1):
        dp = dp * z + p
        p = p * z + coeffs[k]

    return p, dp


def _taylor_many(coeffs, z0):
    """
    Taylor coefficients about each seed.

    Parameters
    ----------
    coeffs : array, complex, shape (degree + 1,)
        Low-to-high coefficients.
    z0 : array, complex, shape (n_roots,)
        Expansion points.

    Returns
    -------
    b : array, complex, shape (n_roots, degree + 1)
        ``b[j] = P^{(j)}(z0) / j!``.
    """
    n_roots = z0.shape[0]
    degree = coeffs.shape[0] - 1
    b = jnp.repeat(coeffs[None, :], n_roots, axis=0)

    for j in range(degree):
        for k in range(degree - 1, j - 1, -1):
            b = b.at[:, k].add(z0 * b[:, k + 1])

    return b


def _hc_displacement(b, terms):
    """
    Truncated hyper-Catalan displacement (eq. 9).

    Parameters
    ----------
    b : array, complex, shape (n_roots, degree + 1)
        Taylor coefficients about the seeds.
    terms : sequence
        Multi-index table from :func:`bagle.hyper_catalan._terms`.

    Returns
    -------
    y : array, complex, shape (n_roots,)
        Predicted displacement. Zero where the linear coefficient
        vanishes.
    """
    a0 = b[:, 0]
    a1 = b[:, 1]
    c1 = -a1
    safe = jnp.abs(a1) > 1e-18
    c1_safe = jnp.where(safe, c1, 1.0 + 0.0j)
    series = jnp.ones(b.shape[0], dtype=jnp.complex128)

    for ms, k_part, weight, c_coef in terms:
        prod_b = jnp.ones(b.shape[0], dtype=jnp.complex128)

        for i, m in enumerate(ms):
            if m:
                prod_b = prod_b * b[:, i + 2] ** m

        term = c_coef * (a0 ** (weight - 1)) * prod_b / (c1_safe ** k_part)
        series = series + term

    y = (a0 / c1_safe) * series
    y = jnp.where(safe & jnp.isfinite(y), y, 0.0 + 0.0j)

    return y


def _newton_capped(coeffs, z, n_newton, cap):
    """
    Newton updates limited by a per-root step cap.

    Parameters
    ----------
    coeffs : array, complex, shape (degree + 1,)
        Low-to-high coefficients.
    z : array, complex, shape (n_roots,)
        Starting roots.
    n_newton : int
        Update count.
    cap : array, float, shape (n_roots,)
        Maximum ``|dz|`` of one update.

    Returns
    -------
    z_out : array, complex, shape (n_roots,)
        Updated roots.
    """

    def _body(_, zz):
        p, dp = _horner(coeffs, zz)
        ok = jnp.abs(dp) > 1e-18
        step = jnp.where(ok, p / jnp.where(ok, dp, 1.0), 0.0)
        mag = jnp.abs(step)
        shrink = jnp.minimum(1.0, cap / jnp.maximum(mag, 1e-30))
        trial = zz - step * shrink

        return jnp.where(jnp.isfinite(trial), trial, zz)

    return jax.lax.fori_loop(0, int(n_newton), _body, z)


def _quadratic_shifts(b0, b1, b2):
    """
    Roots of ``b2 t^2 + b1 t + b0``.

    Parameters
    ----------
    b0, b1, b2 : complex
        Local Taylor coefficients.

    Returns
    -------
    t1, t2 : complex
        Displacements from the expansion point.
    """
    scale = jnp.maximum(jnp.abs(b1), jnp.maximum(jnp.abs(b0), 1e-30))
    linear = jnp.abs(b2) < 1e-18 * scale
    b1_safe = jnp.where(jnp.abs(b1) > 1e-18, b1, 1.0 + 0.0j)
    t_lin = -b0 / b1_safe
    disc = b1 * b1 - 4.0 * b2 * b0
    sqrt_disc = jnp.sqrt(disc + 0.0j)
    sigma = jnp.where(jnp.abs(b1) > 0.0, b1 / jnp.abs(b1), 1.0 + 0.0j)
    q = -0.5 * (b1 + sigma * sqrt_disc)
    tiny_q = jnp.abs(q) < 1e-30
    b2_safe = jnp.where(jnp.abs(b2) > 0.0, b2, 1.0 + 0.0j)
    t_mid = -b1 / (2.0 * b2_safe)
    t1 = jnp.where(tiny_q, t_mid, q / b2_safe)
    t2 = jnp.where(tiny_q, t_mid, b0 / jnp.where(tiny_q, 1.0, q))
    both = linear & (jnp.abs(b1) < 1e-18)
    t1 = jnp.where(both, 0.0, jnp.where(linear, t_lin, t1))
    t2 = jnp.where(both, 0.0, jnp.where(linear, t_lin, t2))

    return t1, t2


def _cubic_shifts(b0, b1, b2, b3):
    """
    Eq. 36 seeds for a local cubic, then a short Newton polish.

    Parameters
    ----------
    b0, b1, b2, b3 : complex
        Local Taylor coefficients.

    Returns
    -------
    t : array, complex, shape (3,)
        Displacements from the expansion point.
    """
    scale = jnp.maximum(jnp.abs(b3), jnp.abs(b2))
    scale = jnp.maximum(scale, jnp.abs(b1))
    scale = jnp.maximum(scale, jnp.abs(b0))
    scale = jnp.maximum(scale, 1e-30)
    dropped = jnp.abs(b3) < 1e-18 * scale
    q1, q2 = _quadratic_shifts(b0, b1, b2)
    t_drop = jnp.array([q1, q2, q1])

    a_coef = b2 / jnp.where(dropped, 1.0, b3)
    b_coef = b1 / jnp.where(dropped, 1.0, b3)
    c_coef = b0 / jnp.where(dropped, 1.0, b3)
    h = a_coef / 3.0
    p = b_coef - a_coef * a_coef / 3.0
    q = c_coef + (2.0 * a_coef ** 3) / 27.0 - a_coef * b_coef / 3.0
    m = -q
    omega = jnp.exp(2j * jnp.pi / 3.0)
    tiny_m = jnp.abs(m) < 1e-16 * (1.0 + jnp.abs(p))
    s_branch = jnp.sqrt(-p + 0.0j)
    s_tiny = jnp.array([0.0 + 0.0j, s_branch, -s_branch])

    def _one(k):
        t0 = (omega ** k) * (m ** (1.0 / 3.0))
        safe = jnp.abs(t0) > 1e-18
        sk = jnp.where(safe, t0 - p / jnp.where(safe, 3.0 * t0, 1.0), 0.0)

        return sk

    s_card = jnp.array([_one(0), _one(1), _one(2)])
    ss = jnp.where(tiny_m, s_tiny, s_card)

    def _polish(_, ss):
        f = ss ** 3 + p * ss + q
        fp = 3.0 * ss ** 2 + p
        ok = jnp.abs(fp) > 1e-18
        step = jnp.where(ok, f / jnp.where(ok, fp, 1.0), 0.0)

        return ss - step

    ss = jax.lax.fori_loop(0, 4, _polish, ss)
    t = ss - h

    return jnp.where(dropped, t_drop, t)


def _assign_pair(prev, seeds):
    """
    Match two seeds onto two previous roots.

    Parameters
    ----------
    prev, seeds : arrays, complex, shape (2,)
        Previous roots and new local roots.

    Returns
    -------
    ordered : array, complex, shape (2,)
        ``seeds`` in the minimum-distance order.
    """
    keep = jnp.abs(seeds[0] - prev[0]) + jnp.abs(seeds[1] - prev[1])
    swap = jnp.abs(seeds[1] - prev[0]) + jnp.abs(seeds[0] - prev[1])
    use_swap = swap < keep
    ordered = jnp.where(
        use_swap,
        jnp.array([seeds[1], seeds[0]]),
        seeds,
    )

    return ordered


def _assign_triple(prev, seeds):
    """
    Match three seeds onto three previous roots.

    Parameters
    ----------
    prev, seeds : arrays, complex, shape (3,)
        Previous roots and new local roots.

    Returns
    -------
    ordered : array, complex, shape (3,)
        ``seeds`` in the minimum-distance order.
    """
    perms = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    best_cost = None
    best = seeds

    for perm in perms:
        ordered = jnp.array([seeds[perm[0]], seeds[perm[1]], seeds[perm[2]]])
        cost = jnp.abs(ordered - prev).sum()
        take = True if best_cost is None else cost < best_cost
        # ``take`` is traced. Blend.
        if best_cost is None:
            best_cost = cost
            best = ordered
        else:
            best = jnp.where(cost < best_cost, ordered, best)
            best_cost = jnp.where(cost < best_cost, cost, best_cost)

    return best


def _labels(z, sep):
    """
    Min-index label of each cluster closer than ``sep``.

    Parameters
    ----------
    z : array, complex, shape (n_roots,)
        Seeds.
    sep : float
        Link length.

    Returns
    -------
    parent : array, int, shape (n_roots,)
        Cluster label (smallest index in the cluster).
    """
    n = z.shape[0]
    dist = jnp.abs(z[:, None] - z[None, :])
    close = (dist < sep) | jnp.eye(n, dtype=bool)
    parent = jnp.arange(n)

    def _relax(_, parent):
        labs = jnp.where(close, parent[None, :], n)

        return jnp.min(labs, axis=1)

    parent = jax.lax.fori_loop(0, n - 1, _relax, parent)

    return parent


def _poly_scale(coeffs, z):
    """Per-root scale ``sum |a_k| max(1, |z|)^k``."""
    zabs = jnp.maximum(jnp.abs(z), 1.0)
    scale = jnp.zeros(z.shape, dtype=jnp.float64)
    zk = jnp.ones(z.shape, dtype=jnp.float64)

    for k in range(coeffs.shape[0]):
        scale = scale + jnp.abs(coeffs[k]) * zk
        zk = zk * zabs

    return jnp.maximum(scale, 1e-30)


def _deriv_scale(coeffs, z):
    """Per-root scale ``sum k |a_k| max(1, |z|)^{k-1}``."""
    zabs = jnp.maximum(jnp.abs(z), 1.0)
    scale = jnp.zeros(z.shape, dtype=jnp.float64)
    zk = jnp.ones(z.shape, dtype=jnp.float64)

    for k in range(1, coeffs.shape[0]):
        scale = scale + k * jnp.abs(coeffs[k]) * zk
        zk = zk * zabs

    return jnp.maximum(scale, 1e-30)


def _collapsed(coeffs, z):
    """
    True when two slots sit on one simple root.

    Parameters
    ----------
    coeffs : array, complex, shape (degree + 1,)
        Target polynomial.
    z : array, complex, shape (n_roots,)
        Candidate roots.

    Returns
    -------
    collapsed : bool
        Glue flag. A genuine near-double root is not flagged.
    """
    _p, dp = _horner(coeffs, z)
    dscale = _deriv_scale(coeffs, z)
    n = z.shape[0]
    flag = jnp.array(False)

    for i in range(n):
        for j in range(i + 1, n):
            sep = jnp.abs(z[i] - z[j])
            near = sep < 1e-3
            dscale_ij = jnp.maximum(
                jnp.minimum(dscale[i], dscale[j]), 1e-30
            )
            dp_ij = jnp.minimum(jnp.abs(dp[i]), jnp.abs(dp[j]))
            glued = near & (
                (sep * dscale_ij < 0.02 * dp_ij) | (sep < _COLLAPSE_SEP)
            )
            glued = glued & (dp_ij > 1e-8 * dscale_ij)
            flag = flag | glued

    return flag


def _quality_ok(coeffs, z):
    """
    True when residuals are small and no slot has glued.

    Parameters
    ----------
    coeffs : array, complex, shape (degree + 1,)
        Target polynomial.
    z : array, complex, shape (n_roots,)
        Candidate roots.

    Returns
    -------
    ok : bool
        Acceptance flag for one substep budget.
    """
    p, _dp = _horner(coeffs, z)
    scale = _poly_scale(coeffs, z)
    res = jnp.max(jnp.abs(p) / scale)
    finite = jnp.all(jnp.isfinite(z)) & jnp.isfinite(res)
    ok = finite & (res < _RES_TOL) & (~_collapsed(coeffs, z))

    return ok


def _polish_cluster(coeffs, ordered, n_newton):
    """
    Uncapped Newton on a fold or cusp seed.

    Parameters
    ----------
    coeffs : array, complex, shape (degree + 1,)
        Full polynomial.
    ordered : array, complex, shape (d,)
        Local seeds, already slot-matched.
    n_newton : int
        Minimum updates. At least twelve are applied, matching
        ``hyper_catalan._polish_separated``.

    Returns
    -------
    z_out : array, complex, shape (d,)
        Polished seeds. A step that glues the slots is dropped.
    """
    n_pass = max(int(n_newton), 2) + 10
    cap = jnp.full(ordered.shape[0], 1e6)

    def _body(_, zz):
        trial = _newton_capped(coeffs, zz, 1, cap)
        bad = _collapsed(coeffs, trial) | (~jnp.all(jnp.isfinite(trial)))

        return jnp.where(bad, zz, trial)

    return jax.lax.fori_loop(0, n_pass, _body, ordered)


def _step_roots(coeffs, z_prev, terms, n_newton):
    """
    One hyper-Catalan / geode step.

    Parameters
    ----------
    coeffs : array, complex, shape (degree + 1,)
        New polynomial, low-to-high.
    z_prev : array, complex, shape (n_roots,)
        Previous roots.
    terms : sequence
        Baked multi-index table.
    n_newton : int
        Newton updates.

    Returns
    -------
    z_new : array, complex, shape (n_roots,)
        Continued roots, same slot order.
    """
    n = z_prev.shape[0]
    dist = jnp.abs(z_prev[:, None] - z_prev[None, :])
    dist = dist + jnp.eye(n) * jnp.inf
    nn = jnp.min(dist, axis=1)
    cap = jnp.maximum(0.5 * nn, 1e-8)
    b_all = _taylor_many(coeffs, z_prev)
    y = _hc_displacement(b_all, terms)
    mag = jnp.abs(y)
    shrink = jnp.minimum(1.0, cap / jnp.maximum(mag, 1e-30))
    y = jnp.where(jnp.isfinite(y), y * shrink, 0.0)
    z_hc = _newton_capped(coeffs, z_prev + y, int(n_newton), cap)

    parent = _labels(z_prev, _CLUSTER_SEP)
    counts = jax.vmap(lambda lab: jnp.sum(parent == lab))(parent)
    z_out = z_hc

    # Folds: every static pair, applied only for the cluster's
    # smallest index so the update happens once.
    for i in range(n):
        for j in range(i + 1, n):
            is_fold = (
                (parent[i] == i)
                & (parent[j] == i)
                & (counts[i] == 2)
            )
            members = jnp.array([z_prev[i], z_prev[j]])
            centroid = jnp.mean(members)
            b = _taylor_many(coeffs, jnp.array([centroid]))[0]
            t1, t2 = _quadratic_shifts(b[0], b[1], b[2])
            seeds = centroid + jnp.array([t1, t2])
            ordered = _assign_pair(members, seeds)
            polished = _polish_cluster(coeffs, ordered, n_newton)
            z_out = z_out.at[i].set(
                jnp.where(is_fold, polished[0], z_out[i])
            )
            z_out = z_out.at[j].set(
                jnp.where(is_fold, polished[1], z_out[j])
            )

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                is_cusp = (
                    (parent[i] == i)
                    & (parent[j] == i)
                    & (parent[k] == i)
                    & (counts[i] == 3)
                )
                members = jnp.array(
                    [z_prev[i], z_prev[j], z_prev[k]]
                )
                centroid = jnp.mean(members)
                b = _taylor_many(coeffs, jnp.array([centroid]))[0]
                shifts = _cubic_shifts(b[0], b[1], b[2], b[3])
                ordered = _assign_triple(members, centroid + shifts)
                polished = _polish_cluster(coeffs, ordered, n_newton)
                z_out = z_out.at[i].set(
                    jnp.where(is_cusp, polished[0], z_out[i])
                )
                z_out = z_out.at[j].set(
                    jnp.where(is_cusp, polished[1], z_out[j])
                )
                z_out = z_out.at[k].set(
                    jnp.where(is_cusp, polished[2], z_out[k])
                )

    return z_out


def _final_polish(coeffs, z, n_newton):
    """
    Uncapped Newton after an accepted step.

    Parameters
    ----------
    coeffs : array, complex, shape (degree + 1,)
        Target polynomial.
    z : array, complex, shape (n_roots,)
        Accepted roots.
    n_newton : int
        Count added to the six extra passes.

    Returns
    -------
    z_out : array, complex, shape (n_roots,)
        Polished roots. A pass that glues is discarded.
    """
    n_pass = max(int(n_newton), 2) + 6
    cap = jnp.full(z.shape[0], 1e6)

    def _body(_, zz):
        trial = _newton_capped(coeffs, zz, 1, cap)
        bad = _collapsed(coeffs, trial) | (~jnp.all(jnp.isfinite(trial)))

        return jnp.where(bad, zz, trial)

    return jax.lax.fori_loop(0, n_pass, _body, z)


def _advance_along(sample, z_prev, terms, n_newton, n_substeps, max_sub):
    """
    Adaptive segments along ``sample(t)``.

    Parameters
    ----------
    sample : callable
        ``sample(t)`` returns coefficients at fraction ``t``.
    z_prev : array, complex, shape (n_roots,)
        Roots at ``t = 0``.
    terms : sequence
        Baked series table.
    n_newton : int
        Newton updates per segment.
    n_substeps : int
        First segment count.
    max_sub : int
        Largest segment count.

    Returns
    -------
    z_new : array, complex, shape (n_roots,)
        Roots of ``sample(1)``.
    """
    a_new = sample(1.0)
    n0 = max(int(n_substeps), 1)

    def _walk(n_sub, z0):
        def _body(k, zz):
            t = (k + 1) / n_sub
            a = sample(t)

            return _step_roots(a, zz, terms, n_newton)

        return jax.lax.fori_loop(0, n_sub, _body, z0)

    z_try = _walk(jnp.int32(n0), z_prev)
    ok = _quality_ok(a_new, z_try)
    lead = jnp.abs(a_new[-1])
    others = jnp.max(jnp.abs(a_new[:-1]))
    singular = lead < 1e-12 * jnp.maximum(others, 1e-30)
    done = ok | singular
    state = (jnp.int32(n0), z_try, done)

    def _cond(state):
        n_sub, _z, done = state

        return (~done) & (n_sub < max_sub)

    def _body(state):
        n_sub, _z, _done = state
        n_sub = n_sub * 2
        z2 = _walk(n_sub, z_prev)
        ok2 = _quality_ok(a_new, z2)

        return n_sub, z2, ok2

    _n_sub, z_out, _done = jax.lax.while_loop(_cond, _body, state)

    return _final_polish(a_new, z_out, n_newton)


def _low_coeffs(w, z1, z2, m1, m2):
    """
    Witt quintic, low-to-high, at one epoch.

    Parameters
    ----------
    w, z1, z2 : complex
        Source and lens positions.
    m1, m2 : float
        Mass fractions.

    Returns
    -------
    coeffs : array, complex, shape (6,)
        ``(a0, a1, a2, a3, a4, a5)``.
    """
    a5, a4, a3, a2, a1, a0 = quintic_coefficients(
        jnp.reshape(w, (1,)),
        jnp.reshape(z1, (1,)),
        jnp.reshape(z2, (1,)),
        m1,
        m2,
    )

    return jnp.stack([a0[0], a1[0], a2[0], a3[0], a4[0], a5[0]])


def _initial_roots(coeffs, n_newton):
    """
    Companion-matrix seed, Newton polished, lex-sorted.

    Parameters
    ----------
    coeffs : array, complex, shape (6,)
        Low-to-high quintic coefficients.
    n_newton : int
        Polish count.

    Returns
    -------
    z : array, complex, shape (5,)
        Sorted roots of the first epoch.
    """
    roots = quintic_roots_companion(
        coeffs[5], coeffs[4], coeffs[3], coeffs[2], coeffs[1], coeffs[0]
    )
    cap = jnp.full(5, 1e6)
    roots = _newton_capped(coeffs, roots, max(int(n_newton), 2), cap)
    order = jnp.lexsort((jnp.imag(roots), jnp.real(roots)))

    return roots[order]


def _track_quintics(w, z1, z2, m1, m2, hc_order, n_newton, n_substeps):
    """
    Track five quintic roots over epochs.

    Parameters
    ----------
    w, z1, z2 : arrays, complex, shape (n_times,)
        Source and lens positions.
    m1, m2 : arrays, float, shape (n_times,)
        Mass fractions.
    hc_order, n_newton, n_substeps : int
        Static tracker settings.

    Returns
    -------
    z : array, complex, shape (n_times, 5)
        Continuous roots, spurious ones included.
    """
    terms = _terms(5, int(hc_order))
    w = jnp.asarray(w, dtype=jnp.complex128).reshape(-1)
    z1 = jnp.asarray(z1, dtype=jnp.complex128).reshape(-1)
    z2 = jnp.asarray(z2, dtype=jnp.complex128).reshape(-1)
    m1 = jnp.asarray(m1, dtype=jnp.float64).reshape(-1)
    m2 = jnp.asarray(m2, dtype=jnp.float64).reshape(-1)
    n = w.shape[0]
    z1 = jnp.broadcast_to(z1, (n,))
    z2 = jnp.broadcast_to(z2, (n,))
    m1 = jnp.broadcast_to(m1, (n,))
    m2 = jnp.broadcast_to(m2, (n,))

    z0 = _initial_roots(_low_coeffs(w[0], z1[0], z2[0], m1[0], m2[0]), n_newton)

    def _epoch(z_prev, idx):
        z_new = _advance_geometry(
            w[idx], z1[idx], z2[idx], m1[idx], m2[idx],
            w[idx + 1], z1[idx + 1], z2[idx + 1], m1[idx + 1], m2[idx + 1],
            z_prev, terms, n_newton, n_substeps,
        )

        return z_new, z_new

    # Empty scan (one epoch) yields shape (0, 5) and concat keeps z0.
    _z_last, z_rest = jax.lax.scan(_epoch, z0, jnp.arange(n - 1))

    return jnp.concatenate([z0[None, :], z_rest], axis=0)


def _implicit_dz(coeffs, dcoeffs, z):
    """
    ``dz = -dP / P'`` at each simple root.

    Parameters
    ----------
    coeffs : array, complex, shape (n_times, 6)
        Low-to-high coefficients. Used only for ``P'``.
    dcoeffs : array, complex, shape (n_times, 6)
        Directional derivative of the coefficients.
    z : array, complex, shape (n_times, 5)
        Roots.

    Returns
    -------
    dz : array, complex, shape (n_times, 5)
        Root tangents. Zero where ``|P'|`` is tiny.
    """
    powers = z[:, :, None] ** jnp.arange(6)
    d_p = jnp.sum(dcoeffs[:, None, :] * powers, axis=-1)

    def _dp_row(a, zz):
        _p, dp = _horner(a, zz)

        return dp

    dp = jax.vmap(_dp_row)(coeffs, z)
    # The predicate has to be a primal. Do not fold ``dz`` into it:
    # a tangent-dependent ``where`` has no transpose.
    tiny = jax.lax.stop_gradient(jnp.abs(dp) < 1e-12)
    denom = jnp.where(tiny, 1.0 + 0.0j, dp)
    dz = jnp.where(tiny, 0.0 + 0.0j, -d_p / denom)

    return dz


@partial(jax.custom_jvp, nondiff_argnums=(5, 6, 7, 8))
def psbl_image_positions_hc(
    w, z1, z2, m1, m2, root_tol, check_sols, hc_order, n_newton
):
    """
    Binary-lens image positions by hyper-Catalan tracking.

    Parameters
    ----------
    w, z1, z2 : array_like, complex, shape (n_times,)
        Source and lens positions in Einstein radii.
    m1, m2 : array_like, float
        Mass fractions, scalar or per epoch.
    root_tol : float
        Lens-equation tolerance used when ``check_sols`` is True.
    check_sols : bool
        Mask non-physical roots with NaN. Gradient callers should
        leave this False and zero bad amplifications instead.
    hc_order : int
        Series truncation ``K``. Static.
    n_newton : int
        Newton updates per step. Static.

    Returns
    -------
    images : array, complex, shape (n_times, 5)
        Tracked roots. Slot order is continuous in time.

    Notes
    -----
    The tangent is ``dz = -P_theta / P_z`` (implicit function
    theorem). The companion eigendecomposition that seeds epoch 0
    is not differentiated.
    """
    w = jnp.asarray(w, dtype=jnp.complex128).reshape(-1)
    n = w.shape[0]
    z1 = jnp.broadcast_to(jnp.asarray(z1, dtype=jnp.complex128).reshape(-1), (n,))
    z2 = jnp.broadcast_to(jnp.asarray(z2, dtype=jnp.complex128).reshape(-1), (n,))
    m1 = jnp.broadcast_to(jnp.asarray(m1, dtype=jnp.float64).reshape(-1), (n,))
    m2 = jnp.broadcast_to(jnp.asarray(m2, dtype=jnp.float64).reshape(-1), (n,))
    z = _track_quintics(w, z1, z2, m1, m2, hc_order, n_newton, 1)

    def _mask(z_arr):
        from bagle.jax_physics import _mask_psbl_roots

        return _mask_psbl_roots(z_arr, w, z1, z2, m1, m2, root_tol)

    # check_sols is a concrete bool (nondiff). 
    if check_sols:
        return _mask(z)

    return z


@psbl_image_positions_hc.defjvp
def _psbl_image_positions_hc_jvp(
    root_tol, check_sols, hc_order, n_newton, primals, tangents
):
    """Implicit JVP. ``root_tol`` does not enter the root locations."""
    w, z1, z2, m1, m2 = primals
    dw, dz1, dz2, dm1, dm2 = tangents
    w = jnp.asarray(w, dtype=jnp.complex128).reshape(-1)
    n = w.shape[0]
    z1 = jnp.broadcast_to(jnp.asarray(z1, dtype=jnp.complex128).reshape(-1), (n,))
    z2 = jnp.broadcast_to(jnp.asarray(z2, dtype=jnp.complex128).reshape(-1), (n,))
    m1 = jnp.broadcast_to(jnp.asarray(m1, dtype=jnp.float64).reshape(-1), (n,))
    m2 = jnp.broadcast_to(jnp.asarray(m2, dtype=jnp.float64).reshape(-1), (n,))
    dw = jnp.broadcast_to(jnp.asarray(dw, dtype=jnp.complex128).reshape(-1), (n,))
    dz1 = jnp.broadcast_to(jnp.asarray(dz1, dtype=jnp.complex128).reshape(-1), (n,))
    dz2 = jnp.broadcast_to(jnp.asarray(dz2, dtype=jnp.complex128).reshape(-1), (n,))
    dm1 = jnp.broadcast_to(jnp.asarray(dm1, dtype=jnp.float64).reshape(-1), (n,))
    dm2 = jnp.broadcast_to(jnp.asarray(dm2, dtype=jnp.float64).reshape(-1), (n,))

    # Calling the custom_jvp primal (not _track_quintics) keeps the
    # adaptive scan and the companion seed out of the linearization.
    z = psbl_image_positions_hc(
        w, z1, z2, m1, m2, root_tol, False, hc_order, n_newton
    )
    z = jax.lax.stop_gradient(z)

    def _pack(ww, uu, vv, aa, bb):
        return jax.vmap(_low_coeffs)(ww, uu, vv, aa, bb)

    coeffs, dcoeffs = jax.jvp(
        _pack, (w, z1, z2, m1, m2), (dw, dz1, dz2, dm1, dm2)
    )
    dz = _implicit_dz(jax.lax.stop_gradient(coeffs), dcoeffs, z)

    if check_sols:
        from bagle.jax_physics import _psbl_invalid_root_mask

        bad = jax.lax.stop_gradient(
            _psbl_invalid_root_mask(z, w, z1, z2, m1, m2, root_tol)
        )
        z_out = jnp.where(bad, jnp.nan + 0.0j, z)
        dz = jnp.where(bad, 0.0, dz)

        return z_out, dz

    return z, dz


psbl_image_positions_hc_jit = jax.jit(
    psbl_image_positions_hc,
    static_argnames=("check_sols", "hc_order", "n_newton"),
)


def _advance_geometry(
    w_a, z1_a, z2_a, m1_a, m2_a,
    w_b, z1_b, z2_b, m1_b, m2_b,
    z_prev, terms, n_newton, n_substeps=1,
):
    """
    Track roots from one lens geometry to the next.

    Substeps interpolate the Witt coefficients. A straight source
    segment makes a fast spurious root swap Newton basins before
    the pair is close enough to treat as a fold.

    Parameters
    ----------
    w_a, z1_a, z2_a, m1_a, m2_a
        Geometry at the sample that produced ``z_prev``.
    w_b, z1_b, z2_b, m1_b, m2_b
        Target geometry.
    z_prev : array, complex, shape (5,)
        Roots at the previous geometry.
    terms : sequence
        Baked series table.
    n_newton : int
        Newton updates per segment.
    n_substeps : int, optional
        First segment count. Default 1.

    Returns
    -------
    z_new : array, complex, shape (5,)
        Roots at the target geometry.
    """
    a_prev = _low_coeffs(w_a, z1_a, z2_a, m1_a, m2_a)
    a_new = _low_coeffs(w_b, z1_b, z2_b, m1_b, m2_b)

    def _sample(t):
        return (1.0 - t) * a_prev + t * a_new

    return _advance_along(
        _sample, z_prev, terms, n_newton, n_substeps, _MAX_SUB
    )


def _limb_primal(w0, z1, z2, m1, m2, rho, n_outline, hc_order, n_newton):
    """
    Ordered limb roots. Shape ``(n_epochs, n_limb, 5)``.

    Limb point 0 of epoch 0 is the only global solve. Later limb
    points continue the previous angle. Epoch ``e`` limb point 0
    continues epoch ``e - 1`` limb point 0.
    """
    terms = _terms(5, int(hc_order))
    w0 = jnp.asarray(w0, dtype=jnp.complex128).reshape(-1)
    n = w0.shape[0]
    z1 = jnp.broadcast_to(jnp.asarray(z1, dtype=jnp.complex128).reshape(-1), (n,))
    z2 = jnp.broadcast_to(jnp.asarray(z2, dtype=jnp.complex128).reshape(-1), (n,))
    m1 = jnp.broadcast_to(jnp.asarray(m1, dtype=jnp.float64).reshape(-1), (n,))
    m2 = jnp.broadcast_to(jnp.asarray(m2, dtype=jnp.float64).reshape(-1), (n,))
    theta = jnp.linspace(-jnp.pi, jnp.pi, int(n_outline), endpoint=False)
    phase = jnp.exp(1j * theta)

    def _around(z_at_0, w_limb, zz1, zz2, mm1, mm2):
        def _step(z_prev, k):
            z_new = _advance_geometry(
                w_limb[k - 1], zz1, zz2, mm1, mm2,
                w_limb[k], zz1, zz2, mm1, mm2,
                z_prev, terms, n_newton,
            )

            return z_new, z_new

        _last, rest = jax.lax.scan(
            _step, z_at_0, jnp.arange(1, int(n_outline))
        )

        return jnp.concatenate([z_at_0[None, :], rest], axis=0)

    w_limb0 = w0[0] + rho * phase
    z00 = _initial_roots(
        _low_coeffs(w_limb0[0], z1[0], z2[0], m1[0], m2[0]), n_newton
    )
    limb0 = _around(z00, w_limb0, z1[0], z2[0], m1[0], m2[0])

    def _epoch(carry, idx):
        z_prev0, w_prev, z1p, z2p, m1p, m2p = carry
        w_limb = w0[idx] + rho * phase
        z_at_0 = _advance_geometry(
            w_prev, z1p, z2p, m1p, m2p,
            w_limb[0], z1[idx], z2[idx], m1[idx], m2[idx],
            z_prev0, terms, n_newton,
        )
        limb = _around(
            z_at_0, w_limb, z1[idx], z2[idx], m1[idx], m2[idx]
        )
        new_carry = (
            z_at_0, w_limb[0], z1[idx], z2[idx], m1[idx], m2[idx]
        )

        return new_carry, limb

    carry0 = (z00, w_limb0[0], z1[0], z2[0], m1[0], m2[0])
    _carry, limbs = jax.lax.scan(_epoch, carry0, jnp.arange(1, n))

    return jnp.concatenate([limb0[None, :, :], limbs], axis=0)


def _limb_coeffs(w0, z1, z2, m1, m2, rho, n_outline):
    """
    Witt coefficients at every limb point.

    Parameters
    ----------
    w0 : array, complex, shape (n_epochs,)
        Source centers.
    z1, z2 : array, complex, shape (n_epochs,)
        Lens positions.
    m1, m2 : array, float, shape (n_epochs,)
        Mass fractions.
    rho : float
        Source radius.
    n_outline : int
        Limb samples. Static.

    Returns
    -------
    coeffs : array, complex, shape (n_epochs, n_limb, 6)
        Low-to-high quintic coefficients.
    """
    theta = jnp.linspace(-jnp.pi, jnp.pi, int(n_outline), endpoint=False)
    phase = jnp.exp(1j * theta)
    w_limb = w0[:, None] + rho * phase[None, :]

    def _one(e, k):
        return _low_coeffs(w_limb[e, k], z1[e], z2[e], m1[e], m2[e])

    n = w0.shape[0]
    coeffs = jax.vmap(
        lambda e: jax.vmap(lambda k: _one(e, k))(jnp.arange(int(n_outline)))
    )(jnp.arange(n))

    return coeffs


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def fsbl_limb_positions_hc(
    w0, z1, z2, m1, m2, rho, n_outline, hc_order, n_newton
):
    """
    Finite-source limb roots, ordered by continuation.

    Parameters
    ----------
    w0 : array, complex, shape (n_epochs,)
        Source centers.
    z1, z2 : array, complex, shape (n_epochs,) or ()
        Lens positions.
    m1, m2 : array, float
        Mass fractions.
    rho : float
        Source radius in Einstein radii.
    n_outline : int
        Limb samples. Static.
    hc_order, n_newton : int
        Tracker settings. Static.

    Returns
    -------
    z : array, complex, shape (n_epochs, n_limb, 5)
        Image positions. Index 2 is the continuous image slot.

    Notes
    -----
    Reverse mode uses a custom VJP. The tracker itself is not
    differentiated; ``dz = -dP / P'`` at the saved roots. Forward
    mode is not implemented.
    """
    return _limb_primal(
        w0, z1, z2, m1, m2, rho, n_outline, hc_order, n_newton
    )


def _fsbl_limb_positions_hc_fwd(
    w0, z1, z2, m1, m2, rho, n_outline, hc_order, n_newton
):
    """
    Forward pass saved for the implicit VJP.

    Parameters
    ----------
    w0, z1, z2, m1, m2, rho, n_outline, hc_order, n_newton
        Same as :func:`fsbl_limb_positions_hc`.

    Returns
    -------
    z : array, complex, shape (n_epochs, n_limb, 5)
        Tracked roots.
    res : tuple
        Inputs and stopped roots for the backward rule.
    """
    z = _limb_primal(
        w0, z1, z2, m1, m2, rho, n_outline, hc_order, n_newton
    )

    return z, (w0, z1, z2, m1, m2, rho, jax.lax.stop_gradient(z))


def _fsbl_limb_positions_hc_bwd(
    n_outline, hc_order, n_newton, res, g
):
    """
    Cotangent of the limb geometry from ``dz = -dP / P'``.

    Parameters
    ----------
    n_outline, hc_order, n_newton : int
        Static tracker settings.
    res : tuple
        Saved forward inputs and roots.
    g : array, complex, shape (n_epochs, n_limb, 5)
        Cotangent of the roots.

    Returns
    -------
    cotangents : tuple
        Cotangents of ``(w0, z1, z2, m1, m2, rho)``.
    """
    del hc_order, n_newton
    w0, z1, z2, m1, m2, rho, z = res

    def _epoch(x, n, dtype):
        # Broadcast inside the differentiated function so a scalar
        # mass comes back as a scalar cotangent.
        x = jnp.asarray(x, dtype=dtype)

        if x.ndim == 0:
            return jnp.broadcast_to(x, (n,))

        return jnp.broadcast_to(jnp.reshape(x, (-1,)), (n,))

    def _implicit(ww, uu, vv, aa, bb, rr):
        ww = jnp.asarray(ww, dtype=jnp.complex128).reshape(-1)
        n = ww.shape[0]
        uu = _epoch(uu, n, jnp.complex128)
        vv = _epoch(vv, n, jnp.complex128)
        aa = _epoch(aa, n, jnp.float64)
        bb = _epoch(bb, n, jnp.float64)
        rr = jnp.asarray(rr, dtype=jnp.float64)
        coeffs = _limb_coeffs(ww, uu, vv, aa, bb, rr, n_outline)

        def _one(a, zz):
            p, dp = _horner(a, zz)
            tiny = jax.lax.stop_gradient(jnp.abs(dp) < 1e-12)
            denom = jnp.where(tiny, 1.0 + 0.0j, dp)
            step = jnp.where(tiny, 0.0 + 0.0j, p / denom)

            return zz - step

        return jax.vmap(jax.vmap(_one))(coeffs, z)

    _y, vjp = jax.vjp(_implicit, w0, z1, z2, m1, m2, rho)
    dw0, dz1, dz2, dm1, dm2, drho = vjp(g)

    return dw0, dz1, dz2, dm1, dm2, drho


fsbl_limb_positions_hc.defvjp(
    _fsbl_limb_positions_hc_fwd, _fsbl_limb_positions_hc_bwd
)
