"""
Hyper-Catalan root tracking for lens polynomials.

The step is the shifted-polynomial series of Berloff & Berloff
(ApJ, 1003, 232, 2026), eq. 8-9, polished by Newton on ``P``.
Near a multiple root the same paper's geode step (Sec. 3) is done
numerically: a local quadratic (fold) or a depressed cubic with the
Cardano correction (eq. 36, cusp). The global companion solve is
used only to seed the first sample.

Coefficients are low-to-high (index = power) and the degree is
arbitrary, so a later triple-lens decic can call the same step.
"""

import itertools

import numpy as np


# Absolute separation (root units) that opens a fold or cusp cluster.
# Only genuinely close pairs are geodes; a merely-nearest pair is not.
_CLUSTER_SEP = 0.02

# Two slots this close, sitting on a simple root, have glued.
_COLLAPSE_SEP = 1e-5

# Max |P| / coeff-scale accepted without another bisection.
_RES_TOL = 1e-9

# Cap on the adaptive coefficient subdivision (powers of two).
_MAX_SUB = 2048

_FACT = [1]


def _fact(n):
    """
    Integer factorial as a Python int.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    f : int
        ``n!``.
    """
    while len(_FACT) <= n:
        _FACT.append(_FACT[-1] * len(_FACT))

    return _FACT[n]


def _multi_index_terms(degree, order):
    """
    Hyper-Catalan multi-indices with paper weight ``w <= order``.

    Weight ``w = 1 + m2 + 2 m3 + 3 m4 + ...`` (eq. 9). The empty
    index is the leading term and is not listed.

    Parameters
    ----------
    degree : int
        Polynomial degree. Parts run from 2 through ``degree``.
    order : int
        Maximum weight ``w`` kept in the truncation (``K``).

    Returns
    -------
    terms : list of tuple
        Each entry is ``(ms, k_part, w, C)`` where ``ms[i]`` is the
        count of part ``i + 2``, ``k_part`` is the partition sum,
        ``w`` is the paper weight, and ``C`` is the HC multinomial.
    """
    terms = []

    if order < 2 or degree < 2:
        return terms

    # ms[i] counts how many times the part (i + 2) is used.
    n_parts = degree - 1

    def _rec(part_index, budget, ms):
        if part_index == n_parts:
            if any(m > 0 for m in ms):
                k_part = 0
                w_extra = 0
                mfact = 1

                for i, m in enumerate(ms):
                    p = i + 2
                    k_part += p * m
                    w_extra += (p - 1) * m
                    mfact *= _fact(m)

                w = 1 + w_extra
                c_coef = _fact(k_part) / (_fact(w) * mfact)
                terms.append((tuple(ms), k_part, w, c_coef))

            return None

        step = (part_index + 2) - 1
        max_m = budget // step

        for m in range(max_m + 1):
            ms.append(m)
            _rec(part_index + 1, budget - m * step, ms)
            ms.pop()

        return None

    _rec(0, order - 1, [])

    return terms


_TERM_CACHE = {}


def _terms(degree, order):
    """
    Cached multi-index table for one degree and truncation.

    Parameters
    ----------
    degree : int
        Polynomial degree.
    order : int
        Truncation weight ``K``.

    Returns
    -------
    terms : list of tuple
        See :func:`_multi_index_terms`.
    """
    key = (int(degree), int(order))

    if key not in _TERM_CACHE:
        _TERM_CACHE[key] = _multi_index_terms(key[0], key[1])

    return _TERM_CACHE[key]


def taylor_shift(coeffs, z0):
    """
    Taylor coefficients of ``P`` about ``z0``.

    ``b[j] = P^{(j)}(z0) / j!`` (eq. 8), low-to-high.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Low-to-high coefficients of ``P``.
    z0 : complex or array_like, shape (n_roots,)
        Expansion point. An array shifts about every point at once.

    Returns
    -------
    b : ndarray, complex
        Shape ``(degree + 1,)`` for a scalar ``z0``, or
        ``(n_roots, degree + 1)`` when ``z0`` is an array.

    Notes
    -----
    Repeated synthetic division. The leading coefficient is unchanged.
    """
    a = np.asarray(coeffs, dtype=np.complex128)
    z = np.asarray(z0, dtype=np.complex128)
    n = a.shape[0] - 1
    scalar = z.ndim == 0

    if scalar:
        b = a.copy()

        for j in range(n):
            for k in range(n - 1, j - 1, -1):
                b[k] = b[k] + z * b[k + 1]

        return b

    # One row per root. The inner loops are the polynomial degree.
    b = np.broadcast_to(a, (z.shape[0], a.shape[0])).copy()

    for j in range(n):
        for k in range(n - 1, j - 1, -1):
            b[:, k] = b[:, k] + z * b[:, k + 1]

    return b


def horner_value_deriv(coeffs, z):
    """
    Evaluate ``P(z)`` and ``P'(z)`` by Horner.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Low-to-high coefficients.
    z : complex or array_like, shape (n_roots,)
        Evaluation point(s).

    Returns
    -------
    p : ndarray, complex
        ``P(z)``. Scalar shape if ``z`` is scalar.
    dp : ndarray, complex
        ``P'(z)``, same shape as ``p``.
    """
    a = np.asarray(coeffs, dtype=np.complex128)
    zz = np.asarray(z, dtype=np.complex128)
    n = a.shape[0] - 1
    p = np.array(a[n], dtype=np.complex128) + np.zeros_like(zz)
    dp = np.zeros_like(zz, dtype=np.complex128)

    for k in range(n - 1, -1, -1):
        dp = dp * zz + p
        p = p * zz + a[k]

    return p, dp


def newton_polish(coeffs, z, n_newton):
    """
    Apply a fixed number of Newton updates ``z <- z - P/P'``.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Low-to-high coefficients of ``P``.
    z : array_like, complex, shape (n_roots,)
        Starting roots.
    n_newton : int
        Number of updates. ``0`` returns ``z`` unchanged.

    Returns
    -------
    z_out : ndarray, complex, shape (n_roots,)
        Polished roots. A vanishing derivative leaves that root put.
    """
    zz = np.array(z, dtype=np.complex128)

    for _ in range(int(n_newton)):
        p, dp = horner_value_deriv(coeffs, zz)
        ok = np.abs(dp) > 1e-18
        step = np.zeros_like(zz)
        step = np.where(ok, p / np.where(ok, dp, 1.0), 0.0)
        zz = zz - step
        zz = np.where(np.isfinite(zz), zz, z)

    return zz


def hc_increment(b, order):
    """
    Truncated hyper-Catalan displacement for a shifted polynomial.

    Solves ``b0 + b1 y + b2 y^2 + ... = 0`` for the root that
    vanishes with ``b0`` (Appendix A; eq. 9 for a quintic).

    Parameters
    ----------
    b : array_like, complex, shape (n_roots, degree + 1)
        Taylor coefficients about the current seeds, low-to-high.
    order : int
        Truncation weight ``K``. ``K = 1`` keeps only ``y = -b0/b1``.

    Returns
    -------
    y : ndarray, complex, shape (n_roots,)
        Predicted displacement. ``0`` where the linear coefficient
        vanishes, so the caller can fall through to Newton or geode.
    """
    bb = np.atleast_2d(np.asarray(b, dtype=np.complex128))
    n_roots, n_coef = bb.shape
    degree = n_coef - 1
    a0 = bb[:, 0]
    a1 = bb[:, 1]
    c1 = -a1
    safe = np.abs(a1) > 1e-18
    c1_safe = np.where(safe, c1, 1.0 + 0.0j)
    series = np.ones(n_roots, dtype=np.complex128)

    for ms, k_part, w, c_coef in _terms(degree, int(order)):
        prod_b = np.ones(n_roots, dtype=np.complex128)

        for i, m in enumerate(ms):
            if m:
                prod_b = prod_b * bb[:, i + 2] ** m

        term = c_coef * (a0 ** (w - 1)) * prod_b / (c1_safe ** k_part)
        series = series + term

    y = (a0 / c1_safe) * series
    y = np.where(safe & np.isfinite(y), y, 0.0 + 0.0j)

    return y


def _assign_nearest(prev, cand):
    """
    Match candidate roots onto the previous seeds.

    Parameters
    ----------
    prev : array_like, complex, shape (d,)
        Previous roots in the cluster, in slot order.
    cand : array_like, complex, shape (d,)
        New local roots, unordered.

    Returns
    -------
    ordered : ndarray, complex, shape (d,)
        ``cand`` permuted to the minimum total distance to ``prev``.
    """
    prev = np.asarray(prev, dtype=np.complex128)
    cand = np.asarray(cand, dtype=np.complex128)
    d = prev.shape[0]
    best_perm = None
    best_cost = np.inf

    for perm in itertools.permutations(range(d)):
        cost = 0.0

        for i in range(d):
            cost += abs(cand[perm[i]] - prev[i])

        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    ordered = np.array(
        [cand[best_perm[i]] for i in range(d)], dtype=np.complex128
    )

    return ordered


def _quadratic_shifts(b0, b1, b2):
    """
    Roots of the local quadratic ``b2 t^2 + b1 t + b0``.

    Parameters
    ----------
    b0, b1, b2 : complex
        Local Taylor coefficients about the cluster centroid.

    Returns
    -------
    t : ndarray, complex, shape (2,)
        Displacements from the centroid. A vanishing ``b2`` drops
        the solve to a linear factor plus a copy of that root.
    """
    if abs(b2) < 1e-18 * max(abs(b1), abs(b0), 1e-30):
        if abs(b1) < 1e-18:
            return np.zeros(2, dtype=np.complex128)

        t = -b0 / b1

        return np.array([t, t], dtype=np.complex128)

    # Stable quadratic formula (complex sign of b1).
    disc = b1 * b1 - 4.0 * b2 * b0
    sqrt_disc = np.sqrt(disc + 0.0j)

    if abs(b1) > 0.0:
        sigma = b1 / abs(b1)
    else:
        sigma = 1.0 + 0.0j

    q = -0.5 * (b1 + sigma * sqrt_disc)

    if abs(q) < 1e-30:
        t1 = -b1 / (2.0 * b2)
        t2 = t1
    else:
        t1 = q / b2
        t2 = b0 / q

    return np.array([t1, t2], dtype=np.complex128)


def _cubic_eq36_shifts(b0, b1, b2, b3):
    """
    Local cusp seeds from the depressed cubic and eq. 36.

    The cubic ``b3 t^3 + b2 t^2 + b1 t + b0`` is depressed, the
    geode value is ``m = -q``, and the three seeds are
    ``s = w^k m^{1/3} - p / (3 w^k m^{1/3})`` (eq. 36). A short
    Newton loop on the depressed cubic finishes the local solve.

    Parameters
    ----------
    b0, b1, b2, b3 : complex
        Local Taylor coefficients about the cluster centroid.

    Returns
    -------
    t : ndarray, complex, shape (3,)
        Displacements from the centroid.
    """
    scale = max(abs(b3), abs(b2), abs(b1), abs(b0), 1e-30)

    if abs(b3) < 1e-18 * scale:
        pair = _quadratic_shifts(b0, b1, b2)

        return np.array([pair[0], pair[1], pair[0]], dtype=np.complex128)

    # Depress: t = s - h, h = A/3, A = b2/b3.
    a_coef = b2 / b3
    b_coef = b1 / b3
    c_coef = b0 / b3
    h = a_coef / 3.0
    p = b_coef - a_coef * a_coef / 3.0
    q = c_coef + (2.0 * a_coef ** 3) / 27.0 - a_coef * b_coef / 3.0
    m = -q
    omega = np.exp(2j * np.pi / 3.0)

    # On the caustic m ~ 0 and the roots are 0, +/- sqrt(-p).
    if abs(m) < 1e-16 * (1.0 + abs(p)):
        s1 = np.sqrt(-p + 0.0j)
        ss = [0.0 + 0.0j, s1, -s1]
    else:
        ss = []

        for k in range(3):
            t0 = (omega ** k) * (m ** (1.0 / 3.0))

            if abs(t0) < 1e-18:
                ss.append(0.0 + 0.0j)
            else:
                ss.append(t0 - p / (3.0 * t0))

    # Polish on the depressed cubic so the three basins stay apart.
    for _ in range(4):
        updated = []

        for s in ss:
            f = s ** 3 + p * s + q
            fp = 3.0 * s ** 2 + p

            if abs(fp) < 1e-18:
                updated.append(s)
            else:
                updated.append(s - f / fp)

        ss = updated

    t = np.array([s - h for s in ss], dtype=np.complex128)

    return t


def _poly_scale(coeffs, z):
    """
    Per-root coefficient scale ``sum |a_k| |z|^k``.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Low-to-high coefficients.
    z : array_like, complex, shape (n_roots,)
        Evaluation points.

    Returns
    -------
    scale : ndarray, float, shape (n_roots,)
        Positive scale used to judge ``|P(z)|``.
    """
    a = np.asarray(coeffs, dtype=np.complex128)
    zz = np.asarray(z, dtype=np.complex128)
    zabs = np.maximum(np.abs(zz), 1.0)
    scale = np.zeros(zz.shape, dtype=np.float64)
    zk = np.ones(zz.shape, dtype=np.float64)

    for k in range(a.shape[0]):
        scale = scale + np.abs(a[k]) * zk
        zk = zk * zabs

    return np.maximum(scale, 1e-30)


def _deriv_scale(coeffs, z):
    """
    Per-root derivative scale ``sum k |a_k| |z|^{k-1}``.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Low-to-high coefficients.
    z : array_like, complex, shape (n_roots,)
        Evaluation points.

    Returns
    -------
    scale : ndarray, float, shape (n_roots,)
        Positive scale used to judge ``|P'(z)|``.
    """
    a = np.asarray(coeffs, dtype=np.complex128)
    zz = np.asarray(z, dtype=np.complex128)
    zabs = np.maximum(np.abs(zz), 1.0)
    scale = np.zeros(zz.shape, dtype=np.float64)
    zk = np.ones(zz.shape, dtype=np.float64)

    for k in range(1, a.shape[0]):
        scale = scale + k * np.abs(a[k]) * zk
        zk = zk * zabs

    return np.maximum(scale, 1e-30)


def _clusters(z, sep):
    """
    Groups of roots closer than ``sep``.

    Parameters
    ----------
    z : array_like, complex, shape (n_roots,)
        Current seeds.
    sep : float
        Absolute separation that links two seeds.

    Returns
    -------
    groups : list of list of int
        Each group has at least two indices. Singletons are omitted.
    """
    zz = np.asarray(z, dtype=np.complex128)
    n = zz.shape[0]
    parent = list(range(n))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]

        return i

    def _union(i, j):
        ri = _find(i)
        rj = _find(j)

        if ri != rj:
            parent[rj] = ri

        return None

    for i in range(n):
        for j in range(i + 1, n):
            if abs(zz[i] - zz[j]) < sep:
                _union(i, j)

    groups = {}

    for i in range(n):
        groups.setdefault(_find(i), []).append(i)

    out = [g for g in groups.values() if len(g) >= 2]

    return out


def _nn_distance(z):
    """
    Distance from each root to its nearest neighbor.

    Parameters
    ----------
    z : array_like, complex, shape (n_roots,)
        Roots.

    Returns
    -------
    nn : ndarray, float, shape (n_roots,)
        Nearest-neighbor distance. A single root gets ``1``.
    """
    zz = np.asarray(z, dtype=np.complex128)
    n = zz.shape[0]

    if n < 2:
        return np.ones(n, dtype=np.float64)

    dist = np.abs(zz[:, None] - zz[None, :])
    np.fill_diagonal(dist, np.inf)

    return dist.min(axis=1)


def _newton_capped(coeffs, z, n_newton, cap):
    """
    Newton updates whose steps are limited by ``cap``.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Low-to-high coefficients.
    z : array_like, complex, shape (n_roots,)
        Starting roots.
    n_newton : int
        Number of updates.
    cap : array_like, float, shape (n_roots,)
        Maximum ``|dz|`` of one update, per root.

    Returns
    -------
    z_out : ndarray, complex, shape (n_roots,)
        Updated roots. A vanishing derivative leaves that root put.
    """
    zz = np.array(z, dtype=np.complex128)
    cap = np.asarray(cap, dtype=np.float64)

    for _ in range(int(n_newton)):
        p, dp = horner_value_deriv(coeffs, zz)
        ok = np.abs(dp) > 1e-18
        step = np.where(ok, p / np.where(ok, dp, 1.0), 0.0)
        mag = np.abs(step)
        shrink = np.minimum(1.0, cap / np.maximum(mag, 1e-30))
        trial = zz - step * shrink
        zz = np.where(np.isfinite(trial), trial, zz)

    return zz


def _polish_separated(coeffs, seeds, n_newton, frac=0.3):
    """
    Newton polish of a fold or cusp seed.

    The local quadratic (or cubic) is only the first guess. Higher
    coefficients shift that guess by a few thousandths, and a close
    pair needs several Newton steps to land inside 1e-10. Steps are
    not length-capped: a cap of half the seed gap stalls when the
    quadratic roots sit closer together than the true pair. A step
    that glues the slots onto one simple root is dropped.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Full polynomial.
    seeds : array_like, complex, shape (d,)
        Local quadratic or cubic seeds, already slot-matched.
    n_newton : int
        Minimum Newton updates. At least twelve are applied.
    frac : float, optional
        Unused. Kept so callers that still pass it do not break.

    Returns
    -------
    z_out : ndarray, complex, shape (d,)
        Polished seeds.
    """
    del frac

    z_cur = np.asarray(seeds, dtype=np.complex128).copy()
    n_pass = max(int(n_newton), 2) + 10

    for _ in range(n_pass):
        trial = newton_polish(coeffs, z_cur, 1)

        if not np.all(np.isfinite(trial)):
            return z_cur

        _ok, _res, collapsed = _root_quality(coeffs, trial)

        if collapsed:
            return z_cur

        z_cur = trial

    return z_cur


def _root_quality(coeffs, z):
    """
    Decide whether a candidate set is still a full set of roots.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Low-to-high coefficients of the target polynomial.
    z : array_like, complex, shape (n_roots,)
        Candidate roots.

    Returns
    -------
    ok : bool
        True when every residual is small and no two slots have
        glued onto one simple root.
    res : float
        Largest ``|P(z)|`` divided by the coefficient scale.
    collapsed : bool
        True when a pair sits on a simple root.
    """
    zz = np.asarray(z, dtype=np.complex128)
    p, dp = horner_value_deriv(coeffs, zz)
    scale = _poly_scale(coeffs, zz)
    res = float(np.max(np.abs(p) / scale))
    dscale = _deriv_scale(coeffs, zz)
    collapsed = False
    n = zz.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            sep = abs(zz[i] - zz[j])

            if sep >= 1e-3:
                continue

            # A real close pair has |P'| ~ (deriv scale) * separation.
            # Two copies of one simple root have |P'| much larger
            # than that, because the missing partner sits far away.
            dscale_ij = max(min(float(dscale[i]), float(dscale[j])), 1e-30)
            dp_ij = min(abs(dp[i]), abs(dp[j]))

            if sep * dscale_ij < 0.02 * dp_ij or sep < _COLLAPSE_SEP:
                if dp_ij > 1e-8 * dscale_ij:
                    collapsed = True

    finite = bool(np.all(np.isfinite(zz))) and np.isfinite(res)
    ok = finite and (res < _RES_TOL) and (not collapsed)

    return ok, res, collapsed


def step_roots(coeffs, z_prev, hc_order=2, n_newton=2):
    """
    Advance every root of one polynomial sample.

    Isolated roots use the truncated HC series (eq. 9) and then
    ``n_newton`` Newton steps, with the predictor and the Newton
    step capped by half the distance to the nearest neighbor.
    A fold (two roots closer than ``_CLUSTER_SEP``) is the local
    quadratic about their centroid. A cusp (three roots) uses the
    eq. 36 seeds. Newton on a cluster is not allowed to glue the
    slots back into one basin. No global ``numpy.roots`` call is
    made here.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Low-to-high coefficients of the new polynomial.
    z_prev : array_like, complex, shape (n_roots,)
        Roots at the previous sample, spurious ones included.
        ``n_roots`` must equal the degree.
    hc_order : int, optional
        Series truncation weight ``K``. Default 2.
    n_newton : int, optional
        Newton updates on ``P`` after the predictor. Default 2.

    Returns
    -------
    z_new : ndarray, complex, shape (n_roots,)
        Tracked roots, same slot order as ``z_prev``.

    Notes
    -----
    Groups of four or more close roots are left to the isolated
    HC step. A binary lens only needs pairs (folds) and triples
    (cusps). The separation threshold is absolute in the same
    units as the roots.
    """
    a = np.asarray(coeffs, dtype=np.complex128)
    z_prev = np.asarray(z_prev, dtype=np.complex128)
    n = z_prev.shape[0]
    z_out = z_prev.copy()
    used = np.zeros(n, dtype=bool)
    groups = _clusters(z_prev, _CLUSTER_SEP)

    # Cusps (3) before folds (2). Larger clumps are not a local jet.
    for size in (3, 2):
        for group in groups:
            if len(group) != size:
                continue

            idx = np.array(group, dtype=int)
            centroid = np.mean(z_prev[idx])
            b = taylor_shift(a, centroid)

            if size == 2:
                shifts = _quadratic_shifts(b[0], b[1], b[2])
            else:
                shifts = _cubic_eq36_shifts(b[0], b[1], b[2], b[3])

            seeds = _assign_nearest(z_prev[idx], centroid + shifts)
            z_out[idx] = _polish_separated(
                a, seeds, max(int(n_newton), 2)
            )
            used[idx] = True

    iso = np.where(~used)[0]

    if iso.size == 0:
        return z_out

    # Eq. 9 on the isolated slots. Cap the jump by the gap to the
    # nearest other seed so a predictor cannot leap into that basin.
    nn = _nn_distance(z_prev)
    cap = np.maximum(0.5 * nn[iso], 1e-8)
    b_iso = taylor_shift(a, z_prev[iso])
    y = hc_increment(b_iso, hc_order)
    mag = np.abs(y)
    shrink = np.minimum(1.0, cap / np.maximum(mag, 1e-30))
    y = np.where(np.isfinite(y), y * shrink, 0.0)
    z_out[iso] = _newton_capped(a, z_prev[iso] + y, n_newton, cap)

    return z_out


def _walk_samples(sample, z_prev, n_sub, hc_order, n_newton):
    """
    Apply :func:`step_roots` at ``n_sub`` interior fractions.

    Parameters
    ----------
    sample : callable
        ``sample(t)`` returns low-to-high coefficients at fraction
        ``t`` in ``(0, 1]``. ``t = 1`` is the target polynomial.
    z_prev : array_like, complex, shape (n_roots,)
        Roots at ``t = 0``.
    n_sub : int
        Number of equal segments.
    hc_order : int
        Series truncation weight ``K``.
    n_newton : int
        Newton updates per segment.

    Returns
    -------
    z_new : ndarray, complex, shape (n_roots,)
        Roots after the segment at ``t = 1``.
    """
    z_cur = np.asarray(z_prev, dtype=np.complex128)

    for s in range(1, int(n_sub) + 1):
        t = s / float(n_sub)
        a = np.asarray(sample(t), dtype=np.complex128)
        z_cur = step_roots(a, z_cur, hc_order=hc_order, n_newton=n_newton)

    return z_cur


def advance_along(
    sample,
    z_prev,
    hc_order=2,
    n_newton=2,
    n_substeps=1,
    adaptive=True,
    max_substeps=_MAX_SUB,
):
    """
    Track roots along a callable coefficient path.

    Parameters
    ----------
    sample : callable
        ``sample(t)`` returns coefficients at fraction ``t``.
        ``sample(0)`` is the polynomial that ``z_prev`` solves and
        ``sample(1)`` is the target. A lens track should rebuild
        the Witt quintic at the linearly interpolated source and
        lenses. A bare polynomial track may interpolate
        coefficients.
    z_prev : array_like, complex, shape (n_roots,)
        Roots at ``sample(0)``.
    hc_order : int, optional
        Series truncation weight ``K``.
    n_newton : int, optional
        Newton updates per substep.
    n_substeps : int, optional
        Segments tried first. Default 1.
    adaptive : bool, optional
        Double the segment count, up to ``max_substeps``, when the
        target residual is large or two slots have glued.
    max_substeps : int, optional
        Largest segment count.

    Returns
    -------
    z_new : ndarray, complex, shape (n_roots,)
        Roots of ``sample(1)``.

    Notes
    -----
    A failed attempt is thrown away and restarted from ``z_prev``.
    Lost roots do not fall back onto a global solve.
    """
    z_prev = np.asarray(z_prev, dtype=np.complex128)
    a_new = np.asarray(sample(1.0), dtype=np.complex128)
    n_sub = max(int(n_substeps), 1)
    limit = max(int(max_substeps), n_sub)
    lead = abs(a_new[-1])
    others = float(np.max(np.abs(a_new[:-1]))) if a_new.size > 1 else 1.0

    # One root escapes through infinity when the leading coefficient
    # vanishes (source sitting on a lens). Extra segments cannot
    # follow that in the finite plane.
    if lead < 1e-12 * max(others, 1e-30):
        adaptive = False

    z_try = _walk_samples(sample, z_prev, n_sub, hc_order, n_newton)
    ok, _res, _col = _root_quality(a_new, z_try)
    advance_along.last_substeps = n_sub

    if adaptive and not ok:
        while n_sub < limit:
            n_sub = min(n_sub * 2, limit)
            z_try = _walk_samples(
                sample, z_prev, n_sub, hc_order, n_newton
            )
            ok, _res, _col = _root_quality(a_new, z_try)
            advance_along.last_substeps = n_sub

            if ok or n_sub >= limit:
                break

    return _final_polish(a_new, z_try, n_newton)


def _final_polish(coeffs, z, n_newton):
    """
    Extra Newton passes once a step has been accepted.

    Close roots converge slowly, so a residual under ``_RES_TOL``
    can still leave a position error above 1e-10. A few un-capped
    Newton steps finish those. A step that glues a pair is dropped
    and the last separated iterate is kept.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Target polynomial.
    z : array_like, complex, shape (n_roots,)
        Accepted roots.
    n_newton : int
        Minimum extra updates. At least six are applied.

    Returns
    -------
    z_out : ndarray, complex, shape (n_roots,)
        Polished roots. Stops early if a step would collapse.
    """
    z_cur = np.asarray(z, dtype=np.complex128)
    n_pass = max(int(n_newton), 2) + 6

    for _ in range(n_pass):
        z2 = newton_polish(coeffs, z_cur, 1)
        _ok, _res, collapsed = _root_quality(coeffs, z2)

        if collapsed or not np.all(np.isfinite(z2)):
            return z_cur

        z_cur = z2

    return z_cur


def advance_roots(
    coeffs_prev,
    coeffs_new,
    z_prev,
    hc_order=2,
    n_newton=2,
    n_substeps=1,
    adaptive=True,
    max_substeps=_MAX_SUB,
):
    """
    Step from one coefficient sample to the next.

    Parameters
    ----------
    coeffs_prev : array_like, complex, shape (degree + 1,)
        Coefficients at the sample that produced ``z_prev``.
    coeffs_new : array_like, complex, shape (degree + 1,)
        Coefficients at the new sample.
    z_prev : array_like, complex, shape (n_roots,)
        Roots at ``coeffs_prev``.
    hc_order : int, optional
        Series truncation weight ``K``.
    n_newton : int, optional
        Newton updates per substep.
    n_substeps : int, optional
        Number of equal coefficient interpolations to try first.
        ``1`` (default) is a single step.
    adaptive : bool, optional
        If True and the step glues two slots onto one simple root
        or leaves a large residual, double the substep count up to
        ``max_substeps``. A fold needs this: the local quadratic
        only separates the pair while they are still apart.
    max_substeps : int, optional
        Largest subdivision tried when ``adaptive`` is True.

    Returns
    -------
    z_new : ndarray, complex, shape (n_roots,)
        Roots of ``coeffs_new``.

    Notes
    -----
    Substeps linearly interpolate coefficients,
    ``(1 - t) P_prev + t P_new``. The binary-lens tracker uses this
    path: a straight source segment swings a spurious root so fast
    that Newton basins swap. :func:`advance_along` is for a caller
    that already has its own coefficient schedule.
    """
    a_prev = np.asarray(coeffs_prev, dtype=np.complex128)
    a_new = np.asarray(coeffs_new, dtype=np.complex128)

    def _sample(t):
        return (1.0 - t) * a_prev + t * a_new

    z_new = advance_along(
        _sample,
        z_prev,
        hc_order=hc_order,
        n_newton=n_newton,
        n_substeps=n_substeps,
        adaptive=adaptive,
        max_substeps=max_substeps,
    )
    advance_roots.last_substeps = advance_along.last_substeps

    return z_new


def sort_roots(z):
    """
    Order roots by increasing real part, then imaginary part.

    Parameters
    ----------
    z : array_like, complex, shape (n_roots,)
        Unordered roots.

    Returns
    -------
    z_sorted : ndarray, complex, shape (n_roots,)
        Lexicographic order. Used only on the global first-sample seed
        so two implementations start on the same branch.
    """
    zz = np.asarray(z, dtype=np.complex128)
    order = np.lexsort((np.imag(zz), np.real(zz)))

    return zz[order]


def initial_roots(coeffs, n_newton=2):
    """
    Global seed for the first sample, via :func:`numpy.roots`.

    Parameters
    ----------
    coeffs : array_like, complex, shape (degree + 1,)
        Low-to-high coefficients.
    n_newton : int, optional
        Newton polish applied after ``numpy.roots``.

    Returns
    -------
    z : ndarray, complex, shape (degree,)
        All polynomial roots, sorted by real then imaginary part.

    Notes
    -----
    This is the only call to the global solver. Later samples use
    :func:`step_roots`.
    """
    a = np.asarray(coeffs, dtype=np.complex128)
    # numpy.roots wants highest degree first.
    raw = np.roots(a[::-1])
    z = sort_roots(raw)
    z = newton_polish(a, z, max(int(n_newton), 2))
    z = sort_roots(z)

    return z


def track_polynomial_roots(
    coeffs, hc_order=2, n_newton=2, n_substeps=1
):
    """
    Track every root of a sequence of polynomials.

    Parameters
    ----------
    coeffs : array_like, complex, shape (n_samples, degree + 1)
        Low-to-high coefficients at each sample. Sample 0 is seeded
        with :func:`numpy.roots`; later samples are HC / geode steps.
    hc_order : int, optional
        Series truncation weight ``K``. Default 2.
    n_newton : int, optional
        Newton updates on ``P`` per step. Default 2.
    n_substeps : int, optional
        Internal coefficient substeps tried first. Default 1.
        A rejected step (glued slots or a large residual) doubles
        this count up to ``_MAX_SUB``. That refinement is what
        carries a fold: the quadratic split has to fire while the
        pair is still separated.

    Returns
    -------
    z : ndarray, complex, shape (n_samples, degree)
        All roots, spurious included, continuous in the sample index.

    Notes
    -----
    Slot ``i`` at sample ``t + 1`` is the continuation of slot ``i``
    at sample ``t``. The first sample is sorted by real, then
    imaginary, part.
    """
    aa = np.asarray(coeffs, dtype=np.complex128)

    if aa.ndim != 2:
        raise ValueError("coeffs must have shape (n_samples, degree + 1)")

    n_samples, n_coef = aa.shape
    degree = n_coef - 1
    z = np.empty((n_samples, degree), dtype=np.complex128)
    z[0] = initial_roots(aa[0], n_newton=n_newton)

    for t in range(1, n_samples):
        z[t] = advance_roots(
            aa[t - 1],
            aa[t],
            z[t - 1],
            hc_order=hc_order,
            n_newton=n_newton,
            n_substeps=n_substeps,
        )

    return z
