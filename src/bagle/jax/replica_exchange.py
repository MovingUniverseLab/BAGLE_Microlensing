"""Tempered HMC replica exchange with non-reversible DEO swaps.

The cold rung (``β=1``) is the posterior. Hot rungs use the tempered
density ``β lnL + ln prior``. Swaps follow the deterministic even/odd
schedule of Syed et al. (2019). Log evidence is the thermodynamic
integral of ``<lnL>_β``.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


def temperature_ladder(n_temperatures):
    """Inverse temperatures, cold rung first.

    Parameters
    ----------
    n_temperatures : int
        Number of rungs, including ``β=1`` and ``β=0``.

    Returns
    -------
    betas : ndarray, shape (n_temperatures,)
        Log-spaced positive rungs from 1 down to a hot beta, then 0.
        Index 0 is the cold chain. The hot beta is ``1e-3``, or
        warmer when there are only a few rungs.

    Notes
    -----
    Neighboring positive rungs keep a constant ratio. The quadratic
    ``linspace ** 2`` ladder leaves its widest gaps beside the cold
    posterior, which is where replica swaps were failing. The hot
    rung stays near ``1e-3`` so the slice down to ``β=0`` does not
    dominate the thermodynamic integral.
    """
    n_temperatures = int(n_temperatures)
    if n_temperatures < 2:
        raise ValueError(
            'replica exchange needs n_temperatures >= 2 '
            '(include the cold posterior and a hotter rung).'
        )
    if n_temperatures == 2:
        betas = np.array([1.0, 0.0], dtype=np.float64)
        return np.ascontiguousarray(betas)

    # Hottest positive rung. β=0 stays so the TI integral covers [0, 1].
    # Half a decade per step until the ladder reaches 1e-3. A wider
    # gap into β=0 lets the prior-averaged lnL dominate the trapezoid.
    n_positive = n_temperatures - 1
    decades = 0.5 * float(n_positive - 1)
    beta_min = max(1.0e-3, 10.0 ** (-decades))
    positive = np.geomspace(1.0, beta_min, n_positive)
    betas = np.concatenate([positive, np.array([0.0])])
    return np.ascontiguousarray(betas, dtype=np.float64)


def _adapt_temperature_ladder(betas, accept, attempt, target=0.25):
    """Nudge interior rungs so swap acceptance moves toward ``target``.

    Parameters
    ----------
    betas : ndarray, shape (n_temps,)
        Cold-first inverse temperatures. ``betas[0]`` stays 1,
        ``betas[-1]`` stays 0, and the hottest positive rung stays.
    accept, attempt : array_like, shape (n_temps - 1,)
        Swap counts on each neighboring gap from one warmup chunk.
    target : float, optional
        Desired swap acceptance. Default 0.25.

    Returns
    -------
    betas : ndarray, shape (n_temps,)
        Updated ladder, strictly decreasing, same endpoints.
    """
    betas = np.asarray(betas, dtype=np.float64).copy()
    n_temps = int(betas.size)
    # Need an interior positive rung that is free to move.
    if n_temps < 4:
        return betas

    attempt = np.asarray(attempt, dtype=np.float64).reshape(-1)
    accept = np.asarray(accept, dtype=np.float64).reshape(-1)
    rate = np.ones(n_temps - 1, dtype=np.float64)
    counted = attempt > 0.0
    rate[counted] = accept[counted] / np.maximum(attempt[counted], 1.0)

    positive = betas[:-1]
    log_beta = np.log(positive)
    gaps = np.diff(log_beta)
    # Mild factors so one noisy chunk cannot collapse a gap.
    scales = np.clip(rate[: gaps.size] / float(target), 0.85, 1.15)
    gaps = gaps * scales
    span = float(np.sum(gaps))
    target_span = float(log_beta[-1] - log_beta[0])
    if span != 0.0:
        gaps = gaps * (target_span / span)

    new_log = np.empty_like(log_beta)
    new_log[0] = 0.0
    new_log[1:] = np.cumsum(gaps)
    new_pos = np.exp(new_log)
    new_pos[0] = 1.0
    new_pos[-1] = positive[-1]
    for i in range(1, new_pos.size - 1):
        ceiling = new_pos[i - 1] * (1.0 - 1.0e-4)
        floor = new_pos[-1] * (1.0 + 1.0e-4)
        new_pos[i] = float(np.clip(new_pos[i], floor, ceiling))

    out = np.concatenate([new_pos, np.array([0.0])])
    return np.ascontiguousarray(out, dtype=np.float64)


def adjacent_temperature_pairs(n_temps, offset):
    """Neighboring rungs for one DEO parity.

    Parameters
    ----------
    n_temps : int
        Number of inverse temperatures.
    offset : int
        0 for even pairs ``(0, 1), (2, 3), ...``; 1 for odd pairs.

    Returns
    -------
    pairs : ndarray, shape (n_pairs, 2), dtype int32
        Empty array with shape ``(0, 2)`` when no pair starts at
        ``offset``.
    """
    left = np.arange(int(offset), int(n_temps) - 1, 2, dtype=np.int32)
    if left.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.column_stack([left, left + 1]).astype(np.int32)


def thermodynamic_logz(betas, mean_lnL):
    """Trapezoidal thermodynamic integral of mean log-likelihood.

    Parameters
    ----------
    betas : array_like, shape (n_temps,)
        Inverse temperatures, in any order.
    mean_lnL : array_like, shape (n_temps,)
        ``<lnL>_β`` on the same rungs.

    Returns
    -------
    logz : float
        ``∫_0^1 <lnL>_β dβ``. ``nan`` when fewer than two rungs are
        finite.
    """
    betas = np.asarray(betas, dtype=float).reshape(-1)
    mean_lnL = np.asarray(mean_lnL, dtype=float).reshape(-1)
    if betas.size < 2 or not np.any(np.isfinite(mean_lnL)):
        return float(np.nan)

    order = np.argsort(betas)
    y = mean_lnL[order]
    x = betas[order]
    # NumPy 2 dropped ``np.trapz``; the pairwise form is the same rule.
    logz = np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return float(logz)


def _update_round_trips(labels, seen_cold, armed, n_trips):
    """Count cold → hot → cold tours of each replica.

    Parameters
    ----------
    labels : jax.Array, shape (n_ladders, n_temps)
        Replica id currently sitting on each rung.
    seen_cold : jax.Array, shape (n_ladders, n_temps), bool
        Indexed by replica id. True after that replica has sat on
        the cold rung.
    armed : jax.Array, shape (n_ladders, n_temps), bool
        Indexed by replica id. True after a cold visit and a later
        hot visit, until the replica returns to the cold rung.
    n_trips : jax.Array, shape (n_ladders,), int32
        Completed round trips per ladder.

    Returns
    -------
    seen_cold, armed, n_trips
        Updated tour state. A trip is counted only when an armed
        replica is on the cold rung.
    """
    idx = jnp.arange(labels.shape[0])
    cold_id = labels[:, 0]
    hot_id = labels[:, -1]

    # Finish a tour before marking the current cold visit.
    done = armed[idx, cold_id]
    n_trips = n_trips + done.astype(jnp.int32)
    armed = armed.at[idx, cold_id].set(
        jnp.where(done, False, armed[idx, cold_id])
    )
    seen_cold = seen_cold.at[idx, cold_id].set(True)

    # Arm replicas that have already been cold and are now hot.
    can_arm = seen_cold[idx, hot_id]
    armed = armed.at[idx, hot_id].set(armed[idx, hot_id] | can_arm)
    return seen_cold, armed, n_trips


def _deo_swap(pos, lnL, labels, key, betas, pairs, n_gaps):
    """Swap positions on one DEO parity.

    Parameters
    ----------
    pos : jax.Array, shape (n_ladders, n_temps, n_dim)
        Replica positions.
    lnL : jax.Array, shape (n_ladders, n_temps)
        Untempered log-likelihood at ``pos``. Used only for the
        swap ratio. It is not a gradient and does not depend on β.
    labels : jax.Array, shape (n_ladders, n_temps)
        Replica ids. They move with the positions.
    key : jax.Array
        PRNG key for the swap uniforms.
    betas : jax.Array, shape (n_temps,)
        Inverse temperatures. Index 0 is cold.
    pairs : ndarray, shape (n_pairs, 2)
        Rungs to exchange. An empty array is a no-op.
    n_gaps : int
        ``n_temps - 1``. Gap ``g`` joins rungs ``g`` and ``g + 1``.

    Returns
    -------
    pos, labels, gap_accept, gap_attempt
        Positions and labels after accepted swaps. Accept and attempt
        counts have shape ``(n_gaps,)``. Tempered gradients are not
        an input and are not returned.

    Notes
    -----
    The acceptance ratio is ``(β_i - β_j) (lnL_j - lnL_i)``. The prior
    terms cancel. Callers must recompute the tempered gradient at the
    destination β on the next HMC step.
    """
    zeros = jnp.zeros((n_gaps,), dtype=jnp.float64)
    if int(pairs.shape[0]) == 0:
        return pos, labels, zeros, zeros

    i = jnp.asarray(pairs[:, 0], dtype=jnp.int32)
    j = jnp.asarray(pairs[:, 1], dtype=jnp.int32)
    log_alpha = (betas[i] - betas[j]) * (lnL[:, j] - lnL[:, i])
    log_u = jnp.log(jax.random.uniform(key, shape=log_alpha.shape))
    accept = (log_u < log_alpha) & jnp.isfinite(log_alpha)

    # Exchange positions only. Do not carry a β-dependent density.
    pos_i = pos[:, i, :]
    pos_j = pos[:, j, :]
    pos = pos.at[:, i, :].set(jnp.where(accept[..., None], pos_j, pos_i))
    pos = pos.at[:, j, :].set(jnp.where(accept[..., None], pos_i, pos_j))

    lab_i = labels[:, i]
    lab_j = labels[:, j]
    labels = labels.at[:, i].set(jnp.where(accept, lab_j, lab_i))
    labels = labels.at[:, j].set(jnp.where(accept, lab_i, lab_j))

    n_ladders = jnp.asarray(pos.shape[0], dtype=jnp.float64)
    accepted = jnp.sum(accept, axis=0).astype(jnp.float64)
    gap_accept = zeros.at[i].add(accepted)
    gap_attempt = zeros.at[i].set(jnp.full(i.shape, n_ladders))
    return pos, labels, gap_accept, gap_attempt


def _make_hmc_deo_chunk(
    lnL_fn, log_prior_fn, betas, even_pairs, odd_pairs, prior_width
):
    """JIT one chunk of tempered HMC and one DEO swap per step.

    Parameters
    ----------
    lnL_fn, log_prior_fn : callable
        Scalar log-density of one vector, shape ``(n_dim,)``.
    betas : array_like, shape (n_temps,)
        Inverse temperatures. Index 0 is ``β=1``.
    even_pairs, odd_pairs : array_like, shape (n_pairs, 2)
        DEO pair lists. Even steps use ``even_pairs`` only.
    prior_width : array_like, shape (n_dim,)
        Per-parameter scale. The leapfrog step is
        ``step_scale * prior_width``.

    Returns
    -------
    chunk : callable
        ``chunk(pos, step_scale, key, labels, seen_cold, armed,
        n_trips, step_index0, betas, n_steps, n_leapfrog)``.
        ``n_steps`` and ``n_leapfrog`` are static. ``betas`` is an
        argument so warmup can retune the ladder without retracing.

    Notes
    -----
    Each HMC proposal evaluates the tempered gradient at the rung's
    own β. After a swap, that gradient is dropped. The next proposal
    rebuilds it from the new position at the destination β.
    """
    width = jnp.asarray(np.asarray(prior_width), dtype=jnp.float64)
    n_gaps = int(np.asarray(betas).shape[0]) - 1
    even_pairs = np.asarray(even_pairs, dtype=np.int32)
    odd_pairs = np.asarray(odd_pairs, dtype=np.int32)

    def _one(theta):
        """Value and gradient of lnL and the log prior.

        Parameters
        ----------
        theta : jax.Array, shape (n_dim,)
            Physical parameters.

        Returns
        -------
        lnL, gL, log_prior, gP, ok
            ``ok`` is False when a value or gradient is non-finite.
            Non-finite values are clipped so the leapfrog stays defined
            and the proposal can be rejected.
        """
        lnL, gL = jax.value_and_grad(lnL_fn)(theta)
        log_prior, gP = jax.value_and_grad(log_prior_fn)(theta)
        ok = (
            jnp.isfinite(lnL)
            & jnp.isfinite(log_prior)
            & jnp.all(jnp.isfinite(gL))
            & jnp.all(jnp.isfinite(gP))
        )
        lnL = jnp.where(jnp.isfinite(lnL), lnL, -1.0e300)
        log_prior = jnp.where(jnp.isfinite(log_prior), log_prior, -1.0e300)
        gL = jnp.where(jnp.isfinite(gL), gL, 0.0)
        gP = jnp.where(jnp.isfinite(gP), gP, 0.0)
        return lnL, gL, log_prior, gP, ok

    def _batch(pos):
        """Vectorized density and gradient on the whole ladder.

        Parameters
        ----------
        pos : jax.Array, shape (n_ladders, n_temps, n_dim)
            Replica positions.

        Returns
        -------
        lnL, gL, log_prior, gP, ok
            ``lnL`` / ``log_prior`` / ``ok`` have shape
            ``(n_ladders, n_temps)``. Gradients match ``pos``.
        """
        flat = pos.reshape((-1, pos.shape[-1]))
        lnL, gL, log_prior, gP, ok = jax.vmap(_one)(flat)
        shape = pos.shape[:-1]
        return (
            lnL.reshape(shape),
            gL.reshape(pos.shape),
            log_prior.reshape(shape),
            gP.reshape(pos.shape),
            ok.reshape(shape),
        )

    def _values(pos):
        """Untempered lnL and log prior, without gradients.

        Parameters
        ----------
        pos : jax.Array, shape (n_ladders, n_temps, n_dim)
            Replica positions after a swap.

        Returns
        -------
        lnL, log_prior : jax.Array, shape (n_ladders, n_temps)
            Recomputed at the current positions. Non-finite entries
            are clipped to ``-1e300``.
        """
        flat = pos.reshape((-1, pos.shape[-1]))
        lnL = jax.vmap(lnL_fn)(flat).reshape(pos.shape[:-1])
        log_prior = jax.vmap(log_prior_fn)(flat).reshape(pos.shape[:-1])
        lnL = jnp.where(jnp.isfinite(lnL), lnL, -1.0e300)
        log_prior = jnp.where(
            jnp.isfinite(log_prior), log_prior, -1.0e300
        )
        return lnL, log_prior

    def _hmc(pos, step_scale, key, n_leapfrog, betas):
        """One tempered HMC proposal on every rung.

        Parameters
        ----------
        pos : jax.Array, shape (n_ladders, n_temps, n_dim)
            Current positions.
        step_scale : jax.Array, shape (n_ladders, n_temps)
            Leapfrog step in units of ``prior_width``.
        key : jax.Array
            PRNG key.
        n_leapfrog : int
            Number of leapfrog steps. Static.
        betas : jax.Array, shape (n_temps,)
            Inverse temperatures for this chunk. Index 0 is cold.
            Passed in so warmup can retune the ladder without
            retracing.

        Returns
        -------
        pos, lnL, log_prior, accept, key
            Accepted state. ``accept`` has shape
            ``(n_ladders, n_temps)``. The tempered gradient is not
            returned.
        """
        eps = step_scale[..., None] * width
        key, k_r, k_u = jax.random.split(key, 3)
        r = jax.random.normal(k_r, pos.shape)

        # Fresh gradient at this rung's β. Nothing is reused from a
        # swap that may have moved the particle onto this rung.
        lnL, gL, log_prior, gP, ok = _batch(pos)
        gU = -betas[None, :, None] * gL - gP
        gU = jnp.where(ok[..., None], gU, 0.0)

        def _energy(lnL_now, lp_now, r_now):
            """Hamiltonian of the tempered density.

            Parameters
            ----------
            lnL_now, lp_now : jax.Array, shape (n_ladders, n_temps)
                Untempered densities.
            r_now : jax.Array, shape (n_ladders, n_temps, n_dim)
                Momentum.

            Returns
            -------
            H : jax.Array, shape (n_ladders, n_temps)
                ``U + K``, with non-finite entries replaced by a
                large positive value so the proposal is rejected.
            """
            potential = -betas[None, :] * lnL_now - lp_now
            kinetic = 0.5 * jnp.sum(r_now * r_now, axis=-1)
            energy = potential + kinetic
            return jnp.where(jnp.isfinite(energy), energy, 1.0e300)

        H0 = _energy(lnL, log_prior, r)

        def _leap(carry, _unused):
            """One velocity-Verlet step at fixed β.

            Parameters
            ----------
            carry
                ``pos, r, gU, lnL, log_prior, ok``.
            _unused
                Scan placeholder.

            Returns
            -------
            carry, None
                Updated state. ``ok`` stays False after a non-finite
                point so the whole trajectory is rejected.
            """
            pos_l, r_l, gU_l, lnL_l, lp_l, ok_l = carry
            r_l = r_l - 0.5 * eps * gU_l
            pos_l = pos_l + eps * r_l
            lnL_l, gL_l, lp_l, gP_l, ok_new = _batch(pos_l)
            gU_l = -betas[None, :, None] * gL_l - gP_l
            gU_l = jnp.where(jnp.isfinite(gU_l), gU_l, 0.0)
            r_l = r_l - 0.5 * eps * gU_l
            finite = (
                ok_new
                & jnp.all(jnp.isfinite(pos_l), axis=-1)
                & jnp.all(jnp.isfinite(r_l), axis=-1)
            )
            ok_l = ok_l & finite
            return (pos_l, r_l, gU_l, lnL_l, lp_l, ok_l), None

        final, _hist = jax.lax.scan(
            _leap, (pos, r, gU, lnL, log_prior, ok), None, length=n_leapfrog
        )
        pos_f, r_f, _gU_f, lnL_f, lp_f, ok_f = final
        log_alpha = H0 - _energy(lnL_f, lp_f, r_f)
        log_u = jnp.log(jax.random.uniform(k_u, shape=log_alpha.shape))
        accept = (log_u < log_alpha) & ok_f & jnp.isfinite(log_alpha)

        pos_out = jnp.where(accept[..., None], pos_f, pos)
        lnL_out = jnp.where(accept, lnL_f, lnL)
        lp_out = jnp.where(accept, lp_f, log_prior)
        return pos_out, lnL_out, lp_out, accept, key

    def chunk(
        pos, step_scale, key, labels, seen_cold, armed, n_trips,
        step_index0, betas, n_steps, n_leapfrog,
    ):
        """Advance every ladder by ``n_steps`` HMC+DEO updates.

        Parameters
        ----------
        pos : jax.Array, shape (n_ladders, n_temps, n_dim)
            Current positions.
        step_scale : jax.Array, shape (n_ladders, n_temps)
            Per-rung leapfrog step in units of the prior width.
        key : jax.Array
            PRNG key.
        labels, seen_cold, armed, n_trips
            Replica-tour state. See :func:`_update_round_trips`.
        step_index0 : int
            Global step index of the first update in this chunk.
            DEO parity is ``step_index0 + k``.
        betas : jax.Array, shape (n_temps,)
            Inverse temperatures used for this chunk.
        n_steps, n_leapfrog : int
            Chunk length and leapfrog count. Both static.

        Returns
        -------
        pos, key, labels, seen_cold, armed, n_trips, n_hmc,
        cold, lnL_hist, gap_accept, gap_attempt
            ``n_hmc`` has shape ``(n_ladders, n_temps)``.
            ``cold`` is ``(n_steps, n_ladders, n_dim)``.
            ``lnL_hist`` is ``(n_steps, n_ladders, n_temps)`` after
            the swap recomputation. Gap arrays are sums over the chunk
            with shape ``(n_gaps,)``.
        """
        def body(carry, k):
            """One HMC proposal and one DEO parity.

            Parameters
            ----------
            carry
                Position, key, and replica-tour state, plus the HMC
                accept counter.
            k : int
                Index inside this chunk.

            Returns
            -------
            carry, hist
                ``hist`` is the cold position, recomputed lnL, and
                this step's gap accept / attempt counts.
            """
            (
                pos_b, key_b, labels_b, seen_b, armed_b, trips_b, n_hmc,
            ) = carry
            key_b, k_hmc, k_swap = jax.random.split(key_b, 3)
            pos_b, lnL_b, _lp_b, hmc_acc, key_b = _hmc(
                pos_b, step_scale, k_hmc, n_leapfrog, betas
            )
            n_hmc = n_hmc + hmc_acc.astype(jnp.float64)

            # Even steps swap (0,1), (2,3), ...; odd steps the others.
            # Only one parity runs. Positions move; gradients do not.
            do_even = ((step_index0 + k) % 2) == 0

            def _even(args):
                """Even DEO pairs.

                Parameters
                ----------
                args : tuple
                    ``pos, lnL, labels``.

                Returns
                -------
                pos, labels, gap_accept, gap_attempt
                    See :func:`_deo_swap`.
                """
                pos_e, lnL_e, labels_e = args
                return _deo_swap(
                    pos_e, lnL_e, labels_e, k_swap, betas,
                    even_pairs, n_gaps,
                )

            def _odd(args):
                """Odd DEO pairs.

                Parameters
                ----------
                args : tuple
                    ``pos, lnL, labels``.

                Returns
                -------
                pos, labels, gap_accept, gap_attempt
                    See :func:`_deo_swap`.
                """
                pos_o, lnL_o, labels_o = args
                return _deo_swap(
                    pos_o, lnL_o, labels_o, k_swap, betas,
                    odd_pairs, n_gaps,
                )

            pos_b, labels_b, gap_acc, gap_att = jax.lax.cond(
                do_even, _even, _odd, (pos_b, lnL_b, labels_b)
            )
            # Recompute at the post-swap positions. The HMC gradient
            # was built at the pre-swap β and is not reused.
            lnL_b, _lp_b = _values(pos_b)
            seen_b, armed_b, trips_b = _update_round_trips(
                labels_b, seen_b, armed_b, trips_b
            )
            cold = pos_b[:, 0, :]
            new_carry = (
                pos_b, key_b, labels_b, seen_b, armed_b, trips_b, n_hmc,
            )
            return new_carry, (cold, lnL_b, gap_acc, gap_att)

        n_hmc0 = jnp.zeros(pos.shape[:2], dtype=jnp.float64)
        carry0 = (
            pos, key, labels, seen_cold, armed, n_trips, n_hmc0,
        )
        final, hist = jax.lax.scan(
            body, carry0, jnp.arange(n_steps)
        )
        pos, key, labels, seen_cold, armed, n_trips, n_hmc = final
        cold, lnL_hist, gap_acc_h, gap_att_h = hist
        gap_accept = jnp.sum(gap_acc_h, axis=0)
        gap_attempt = jnp.sum(gap_att_h, axis=0)
        return (
            pos, key, labels, seen_cold, armed, n_trips, n_hmc,
            cold, lnL_hist, gap_accept, gap_attempt,
        )

    return jax.jit(chunk, static_argnames=('n_steps', 'n_leapfrog'))


def replica_exchange_sample(
    lnL_fn,
    log_prior_fn,
    init_pos,
    prior_width,
    n_tune,
    n_draws,
    n_leapfrog=6,
    seed=0,
    verbose=False,
):
    """Sample the cold chain of an HMC replica-exchange ladder.

    Parameters
    ----------
    lnL_fn : callable
        JAX log-likelihood of one parameter vector.
    log_prior_fn : callable
        JAX log-prior of one parameter vector.
    init_pos : array_like, shape (n_ladders, n_temps, n_dim)
        Starting positions, one replica per ladder and rung.
    prior_width : array_like, shape (n_dim,)
        Central prior width of each parameter. Sets the HMC step.
    n_tune : int
        Warmup steps per replica. Discarded. Step sizes adapt here.
    n_draws : int
        Steps kept per cold chain after warmup.
    n_leapfrog : int, optional
        Leapfrog steps per HMC proposal. Default 6.
    seed : int, optional
        PRNG seed.
    verbose : bool, optional
        Print warmup acceptance, swap rates, and logZ.

    Returns
    -------
    samples : ndarray, shape (n_ladders * n_draws, n_dim)
        Cold-chain parameter draws.
    logz : float
        Thermodynamic-integration estimate of log evidence.
    swap_accept_rate : ndarray, shape (n_temps - 1,)
        DEO swap acceptance during the sampling phase. Entry ``g``
        is the gap between rungs ``g`` and ``g + 1``.
    n_round_trips : int
        Cold → hot → cold tours completed during the sampling phase,
        summed over ladders.

    Notes
    -----
    Swap rates and round trips are counted only after warmup. The
    replica labels themselves keep moving through warmup so a tour
    can finish during sampling. During warmup the positive rungs are
    nudged, in log spacing, toward equal swap acceptance. The ladder
    is frozen before the samples that enter the thermodynamic
    integral.
    """
    init_pos = np.asarray(init_pos, dtype=np.float64)
    if init_pos.ndim != 3:
        raise ValueError(
            'init_pos must have shape (n_ladders, n_temps, n_dim), '
            f'got {init_pos.shape}.'
        )
    n_ladders, n_temps, n_dim = init_pos.shape
    prior_width = np.asarray(prior_width, dtype=np.float64).reshape(-1)
    if prior_width.shape[0] != n_dim:
        raise ValueError(
            f'prior_width has length {prior_width.shape[0]}, '
            f'expected {n_dim}.'
        )
    if int(n_ladders) < 1:
        raise ValueError('replica exchange needs at least one ladder.')

    betas = temperature_ladder(n_temps)
    even_pairs = adjacent_temperature_pairs(n_temps, 0)
    odd_pairs = adjacent_temperature_pairs(n_temps, 1)
    chunk = _make_hmc_deo_chunk(
        lnL_fn, log_prior_fn, betas, even_pairs, odd_pairs, prior_width
    )

    # Larger steps on hotter rungs. Tune rescales each rung.
    base = 0.05 * (1.0 + 2.0 * (1.0 - betas))
    step_scale = np.broadcast_to(base, (n_ladders, n_temps)).copy()
    step_scale = np.asarray(step_scale, dtype=np.float64)

    pos = jnp.asarray(init_pos, dtype=jnp.float64)
    key = jax.random.PRNGKey(int(seed))
    labels = np.broadcast_to(
        np.arange(n_temps, dtype=np.int32), (n_ladders, n_temps)
    ).copy()
    labels = jnp.asarray(labels)
    seen_cold = jnp.zeros((n_ladders, n_temps), dtype=bool)
    armed = jnp.zeros((n_ladders, n_temps), dtype=bool)
    n_trips = jnp.zeros((n_ladders,), dtype=jnp.int32)

    def _phase(pos, step_scale, key, labels, seen_cold, armed, n_trips,
               step_index, n_steps, betas, adapt, record):
        """Run one phase in fixed-size JIT chunks.

        Parameters
        ----------
        pos, step_scale, key, labels, seen_cold, armed, n_trips
            Replica state. ``step_scale`` is a NumPy array.
        step_index : int
            Global DEO index at the start of the phase.
        n_steps : int
            Number of HMC+DEO updates.
        betas : ndarray, shape (n_temps,)
            Inverse temperatures. Updated in place of the return
            value when ``adapt`` is True.
        adapt : bool
            Rescale ``step_scale`` from the chunk's HMC accept rate,
            and retune interior betas toward equal swap acceptance.
        record : bool
            Keep cold-chain draws, lnL sums, and swap counts.

        Returns
        -------
        pos, step_scale, key, labels, seen_cold, armed, n_trips,
        step_index, betas, cold, lnL_sum, n_kept, gap_accept, gap_attempt
            ``cold`` is ``(n_recorded, n_ladders, n_dim)`` or None.
            ``lnL_sum`` has shape ``(n_temps,)``.
        """
        left = int(n_steps)
        cold_parts = []
        lnL_sum = np.zeros(n_temps, dtype=np.float64)
        n_kept = 0
        gap_accept = np.zeros(n_temps - 1, dtype=np.float64)
        gap_attempt = np.zeros(n_temps - 1, dtype=np.float64)
        chunk_size = 10

        while left > 0:
            this = chunk_size if left >= chunk_size else left
            (
                pos, key, labels, seen_cold, armed, n_trips, n_hmc,
                cold, lnL_hist, acc, att,
            ) = chunk(
                pos,
                jnp.asarray(step_scale),
                key,
                labels,
                seen_cold,
                armed,
                n_trips,
                step_index,
                jnp.asarray(betas, dtype=jnp.float64),
                n_steps=this,
                n_leapfrog=int(n_leapfrog),
            )
            step_index += int(this)
            left -= int(this)

            if adapt:
                rate = np.asarray(n_hmc) / float(this)
                factor = np.ones(rate.shape, dtype=np.float64)
                factor = np.where(rate > 0.85, 1.15, factor)
                factor = np.where(rate < 0.55, 0.80, factor)
                step_scale = np.clip(step_scale * factor, 1.0e-4, 0.50)
                betas = _adapt_temperature_ladder(
                    betas, np.asarray(acc), np.asarray(att)
                )
                if verbose:
                    print(
                        'replica HMC accept '
                        f'{float(np.mean(rate)):.2f}'
                    )
            if record:
                cold_parts.append(np.asarray(cold))
                lnL_sum += np.asarray(lnL_hist).sum(axis=(0, 1))
                n_kept += int(this) * int(n_ladders)
                gap_accept += np.asarray(acc, dtype=np.float64)
                gap_attempt += np.asarray(att, dtype=np.float64)

        cold_out = None
        if cold_parts:
            cold_out = np.concatenate(cold_parts, axis=0)
        return (
            pos, step_scale, key, labels, seen_cold, armed, n_trips,
            step_index, betas, cold_out, lnL_sum, n_kept,
            gap_accept, gap_attempt,
        )

    (
        pos, step_scale, key, labels, seen_cold, armed, n_trips,
        step_index, betas, _cold, _sum, _n, _acc, _att,
    ) = _phase(
        pos, step_scale, key, labels, seen_cold, armed, n_trips,
        0, int(n_tune), betas, adapt=True, record=False,
    )
    trips_after_tune = int(np.asarray(n_trips).sum())
    # Freeze the ladder for the recorded phase so the TI weights
    # match the betas that produced ``<lnL>``.
    (
        pos, step_scale, key, labels, seen_cold, armed, n_trips,
        step_index, betas, cold, lnL_sum, n_kept, gap_accept, gap_attempt,
    ) = _phase(
        pos, step_scale, key, labels, seen_cold, armed, n_trips,
        step_index, int(n_draws), betas, adapt=False, record=True,
    )

    if cold is None:
        samples = np.zeros((0, n_dim), dtype=float)
    else:
        # (n_draws, n_ladders, n_dim) -> stack chains.
        samples = np.asarray(cold, dtype=float).reshape(-1, n_dim)

    if n_kept <= 0:
        logz = float(np.nan)
    else:
        mean_lnL = lnL_sum / float(n_kept)
        logz = thermodynamic_logz(betas, mean_lnL)

    rate = np.full(n_temps - 1, np.nan, dtype=np.float64)
    ok = gap_attempt > 0.0
    rate[ok] = gap_accept[ok] / gap_attempt[ok]
    n_round_trips = int(np.asarray(n_trips).sum()) - trips_after_tune

    if verbose:
        print(f'replica exchange logZ ≈ {logz:.3f}')
        print(
            'replica betas '
            + np.array2string(np.asarray(betas), precision=3)
        )
        print(f'replica swap accept {np.array2string(rate, precision=2)}')
        print(f'replica round trips {n_round_trips}')

    return samples, logz, rate, n_round_trips
