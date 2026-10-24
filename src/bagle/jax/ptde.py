# Copyright (c) 2026, Jason Eastman
#
# BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Parallel tempering with differential evolution (PTDE), in JAX.

Jason Eastman's synchronous PTDE sampler, ported from EXOZIPPy (BSD-3)
so BAGLE can compare the efficiency of his method:

https://github.com/jdeast/EXOZIPPy/tree/master/src/exozippy/samplers

The move, the ladder, the DEO swaps, the gamma update, and the
post-hoc burn-in follow ``ptde.py``, ``_common.py``, ``ladder.py``,
and ``convergence.py`` there. Cite Eastman when using this sampler.

The cold rung is index 0 and has temperature ``T = 1``. Hotter rungs
have ``T > 1``. The Metropolis ratio tempers the full unconstrained
log posterior by ``1/T``, which is his raw-parameter convention
(``ptde.py`` acceptance, the ``lp / T`` factor).

Deviations from his code, all approved for this port:

* The state lives in BAGLE unconstrained coordinates. The target is
  ``log L + log prior + log|det J|`` from ``_unconstrained_target``.
* Starts are the caller's positions. The solver draws them from the
  prior with ``_init_positions`` and copies that population onto every
  rung. There is no MAP start and no polish.
* The run length is fixed. There is no live early stop. Gamma and the
  ladder still adapt between jitted ``lax.scan`` windows during tune.
* No asynchronous sampler, no hot-rung thinning, no hot-chain store.
* The legacy random-pair swap schedule is not implemented. Swaps are
  the DEO schedule only.
* The RNG is JAX, so the stream is not bit-identical to NumPy. Each
  chain draws two distinct partners from its rung. Same move.
* ``n_chains < d + 2`` raises. His code warns and will still run down
  to 3 chains, and it floors the default at 4. The default here is
  ``max(2d, d+2)``, which matches his ``max(2d, 4)`` for ``d >= 2``.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


# Epsilon jitter on the ter Braak move, in unconstrained units.
# Without it the chain cannot leave the affine hull of the starts.
DE_JITTER = 1.0e-4

# T=1 acceptance the gamma adapter aims at. Not the NUTS 0.80 target.
TARGET_ACCEPT = 0.20

# Largest factor one tune window may multiply or divide gamma by.
GAMMA_CLIP_FACTOR = 10.0

# Shortest tune window. A shorter window makes the ladder follow noise.
MIN_ADAPT_WINDOW = 50

# Default geometric ladder. EXOFASTv2 parity, as in his sampler.
DEFAULT_N_TEMPS = 8
DEFAULT_T_MAX = 200.0

# Burn-in fractions scanned by find_burnin. Capped at 0.75 so a quarter
# of the trace always survives.
_BURNIN_FRACS = (
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.75,
)

# Thin to this many draws per chain before R-hat / ESS.
_STAT_DRAW_BUDGET = 2000

# Below this many retained draws per chain the statistic is noise.
_MIN_DRAWS_PER_CHAIN = 16

# getburnndx: do not drop chains if fewer than this many would remain.
_MIN_GOOD_CHAINS = 3


def geometric_ladder(n_temps, T_max):
    """Geometric temperature ladder, cold rung first.

    Parameters
    ----------
    n_temps : int
        Number of rungs ``K``, including the cold target.
    T_max : float
        Hottest temperature. The cold rung is ``T = 1``.

    Returns
    -------
    temperatures : ndarray, shape (n_temps,)
        ``T_k = T_max^(k/(K-1))``. Eastman,
        ``exozippy.samplers.ladder._geometric_ladder``.

    Notes
    -----
    A single rung returns ``[1]``. The ladder is in temperature, not
    inverse temperature. Hotter rungs have larger ``T``.
    """
    n_temps = int(n_temps)
    if n_temps < 1:
        raise ValueError(f'n_temps must be >= 1, got {n_temps}')
    if n_temps == 1:
        return np.array([1.0], dtype=np.float64)

    # k / (K-1) runs from 0 to 1, so the ends are 1 and T_max.
    exponent = np.arange(n_temps, dtype=np.float64) / float(n_temps - 1)
    temperatures = np.asarray(T_max, dtype=np.float64) ** exponent
    return temperatures


def resolve_n_temps(n_temps, n_params, T_max):
    """Resolve ``n_temps``, including the string ``'auto'``.

    Parameters
    ----------
    n_temps : int or str
        A rung count, or ``'auto'``.
    n_params : int
        Dimension ``d`` of the unconstrained state.
    T_max : float
        Hottest temperature. Used only for ``'auto'``.

    Returns
    -------
    n_temps : int
        ``'auto'`` is ``max(8, ceil(sqrt(d/2) * ln(T_max)))``.
        Otherwise the integer that was passed.

    Notes
    -----
    Eastman, ``exozippy.samplers.ladder.resolve_n_temps``. The
    ``sqrt(d/2)`` factor is the adjacent-rung overlap argument: the
    mean log posterior shifts by about ``(d/2) ln r`` between rungs
    while it fluctuates by ``sqrt(d/2)``.
    """
    if isinstance(n_temps, str):
        if n_temps.strip().lower() != 'auto':
            raise ValueError(
                f"n_temps must be an integer or 'auto', got {n_temps!r}"
            )
        # Natural log, matching his np.log.
        width = np.sqrt(float(n_params) / 2.0) * np.log(float(T_max))
        return max(8, int(np.ceil(width)))

    n_temps = int(n_temps)
    if n_temps < 1:
        raise ValueError(f'n_temps must be >= 1, got {n_temps}')
    return n_temps


def resolve_n_chains(n_chains, n_params):
    """Resolve the DE population size on one rung.

    Parameters
    ----------
    n_chains : int or None
        Requested chains. ``None`` selects the default.
    n_params : int
        Dimension ``d``.

    Returns
    -------
    n_chains : int
        Default ``max(2d, d+2)``. An explicit count below ``d+2``
        raises. A DE move needs two other members, and ``d+2`` members
        are what it takes to span ``R^d``.

    Notes
    -----
    Eastman's default is ``max(2d, 4)`` and he only warns below
    ``d+2``. This port treats ``d+2`` as a hard minimum, which is the
    same number for every ``d >= 2``.
    """
    n_params = int(n_params)
    if n_params < 1:
        raise ValueError(f'n_params must be >= 1, got {n_params}')

    floor = n_params + 2
    if n_chains is None:
        return max(2 * n_params, floor)

    n_chains = int(n_chains)
    if n_chains < floor:
        raise ValueError(
            f'n_chains={n_chains} is below d+2={floor}. DE difference '
            'vectors then cannot span the parameter space.'
        )
    return n_chains


def deo_pairs(round_idx, n_temps):
    """Adjacent rung pairs for one DEO swap round.

    Parameters
    ----------
    round_idx : int
        Swap-round counter. Even rounds take pairs ``(0, 1), (2, 3),
        ...``. Odd rounds take ``(1, 2), (3, 4), ...``.
    n_temps : int
        Number of rungs.

    Returns
    -------
    pairs : list of tuple of int
        ``(k, k+1)`` pairs. Empty when that parity has no pair, which
        is the odd round of a 2-rung ladder.

    Notes
    -----
    Eastman, ``exozippy.samplers.ladder._deo_pairs``. The pairs in one
    round are disjoint, so they can be attempted together. The
    alternating offset is what makes the index process non-reversible
    (Syed et al. 2022).
    """
    start = 0 if int(round_idx) % 2 == 0 else 1
    return [
        (k, k + 1) for k in range(start, int(n_temps) - 1, 2)
    ]


def next_gamma(gamma, acceptance, target_accept, clip=GAMMA_CLIP_FACTOR):
    """One gamma update from a window acceptance rate.

    Parameters
    ----------
    gamma : float
        Current DE scale. Must be positive.
    acceptance : float
        Window acceptance at ``T = 1``, hops excluded.
    target_accept : float
        Target acceptance. His default is 0.20.
    clip : float, optional
        Largest multiplicative step. Default 10.

    Returns
    -------
    gamma_new : float
        ``gamma * (acceptance / target)**0.5``, clipped to
        ``[gamma/clip, gamma*clip]``. A non-positive acceptance
        shrinks by ``clip`` instead of stalling.

    Notes
    -----
    Eastman, ``exozippy.samplers._common.next_gamma``. The square root
    damps the overshoot of a one-window-lagged proportional rule.
    The sampler itself skips the update when the window accepted
    nothing; the shrink branch is what his polish driver uses.
    """
    gamma = float(gamma)
    clip = float(clip)
    shrink = 1.0 / clip
    if float(acceptance) > 0.0:
        scale = (float(acceptance) / float(target_accept)) ** 0.5
    else:
        scale = shrink

    updated = gamma * scale
    lo = gamma * shrink
    hi = gamma * clip
    return float(np.clip(updated, lo, hi))


def adaptation_window(tune, log_every, min_window=MIN_ADAPT_WINDOW):
    """Steps between tune-phase adaptation windows.

    Parameters
    ----------
    tune : int
        Configured tune length.
    log_every : int
        Logging cadence. The window is never longer than this.
    min_window : int, optional
        Shortest window. Default 50.

    Returns
    -------
    adapt_every : int
        Eastman, ``exozippy.samplers.ptde.adaptation_window``.
        ``min(log_every, max(min_window, tune // 20))``, or
        ``log_every`` when ``tune <= 0``.
    """
    tune = int(tune)
    log_every = max(1, int(log_every))
    if tune <= 0:
        return log_every
    return min(log_every, max(int(min_window), tune // 20))


def update_ladder_barrier(temperatures, swap_accept, swap_propose):
    """Re-space the ladder so each rung carries the same barrier.

    Parameters
    ----------
    temperatures : array_like, shape (n_temps,)
        Current temperatures. Ends stay put.
    swap_accept, swap_propose : array_like, shape (n_temps - 1,)
        Swap counts on each adjacent gap for one tune window.

    Returns
    -------
    temperatures : ndarray, shape (n_temps,)
        New temperatures. ``T_0 = 1`` and ``T_{K-1} = T_max`` are
        unchanged. Fewer than 3 rungs, or a window with no proposals,
        returns the input.

    Notes
    -----
    Eastman, ``exozippy.samplers.ladder._update_ladder_barrier``.
    The per-gap rejection rate estimates the local communication
    barrier (Syed et al. 2022). Interior rungs are placed at equal
    barrier fractions by interpolating coldness ``beta = 1/T``.
    Only call this during tune. Moving the ladder after tune breaks
    invariance, same rule as the gamma adaptation.
    """
    temperatures = np.asarray(temperatures, dtype=np.float64)
    n_temps = int(temperatures.size)
    if n_temps < 3:
        return temperatures.copy()

    prop = np.asarray(swap_propose, dtype=np.float64).reshape(-1)
    acc = np.asarray(swap_accept, dtype=np.float64).reshape(-1)
    measured = prop > 0.0
    if not np.any(measured):
        return temperatures.copy()

    # An unmeasured gap is not a rejection. Interpolate it from the
    # gaps that were actually proposed this window.
    rejection = np.zeros(prop.shape, dtype=np.float64)
    rejection[measured] = np.clip(
        1.0 - acc[measured] / prop[measured], 0.0, 1.0
    )
    if not np.all(measured):
        pair_idx = np.arange(prop.size)
        rejection[~measured] = np.interp(
            pair_idx[~measured], pair_idx[measured], rejection[measured]
        )

    # Cumulative barrier at each rung. Lambda[0] = 0.
    barrier = np.concatenate([[0.0], np.cumsum(rejection)])
    total = float(barrier[-1])
    if total <= 0.0:
        return temperatures.copy()

    # Nudge flat segments so np.interp's x is strictly increasing.
    for k in range(1, n_temps):
        if barrier[k] <= barrier[k - 1]:
            barrier[k] = barrier[k - 1] + 1.0e-9

    beta = 1.0 / temperatures
    targets = np.linspace(0.0, barrier[-1], n_temps)
    new_beta = np.interp(targets, barrier, beta)

    # Pin the cold target and the hottest rung.
    new_beta[0] = beta[0]
    new_beta[-1] = beta[-1]
    return 1.0 / new_beta


def ladder_health(temperatures, swap_accept, swap_propose):
    """Communication barrier of a finished ladder.

    Parameters
    ----------
    temperatures : array_like, shape (n_temps,)
        Temperatures, cold first. Only the length is used.
    swap_accept, swap_propose : array_like, shape (n_temps - 1,)
        Swap counts. Zero-proposal gaps are interpolated, not scored
        as total rejections.

    Returns
    -------
    lam : float or None
        ``Lambda = sum of adjacent-rung rejection rates``. ``None``
        when there is nothing to measure.

    Notes
    -----
    Eastman, ``exozippy.samplers.ladder.ladder_health_report``. Under
    DEO the round-trip rate approaches ``1/(2 + 2 Lambda)`` once the
    ladder is longer than the barrier. ``n_temps - 1 < 2 Lambda``
    means the ladder, not the draw count, is the bottleneck.
    """
    n_temps = int(np.asarray(temperatures).size)
    prop = np.asarray(swap_propose, dtype=np.float64).reshape(-1)
    if n_temps < 2 or float(np.sum(prop)) <= 0.0:
        return None

    acc = np.asarray(swap_accept, dtype=np.float64).reshape(-1)
    measured = prop > 0.0
    rejection = np.zeros(prop.shape, dtype=np.float64)
    rejection[measured] = np.clip(
        1.0 - acc[measured] / prop[measured], 0.0, 1.0
    )
    if not np.all(measured):
        pair_idx = np.arange(prop.size)
        rejection[~measured] = np.interp(
            pair_idx[~measured], pair_idx[measured], rejection[measured]
        )
    return float(np.sum(np.clip(rejection, 0.0, 1.0)))


def good_chain_mask(lp):
    """Chains that reached the good-likelihood region.

    Parameters
    ----------
    lp : array_like, shape (n_chains, n_draws)
        Untempered log posterior of each cold chain.

    Returns
    -------
    mask : ndarray, shape (n_chains,), bool
        True for chains whose maximum log posterior crossed the
        threshold.
    reliable : bool
        False when fewer than ``_MIN_GOOD_CHAINS`` passed and every
        chain was kept instead.

    Notes
    -----
    Eastman's getburnndx rule
    (``exozippy.samplers.convergence.good_chain_mask``). The best
    chain is the one with the highest single log posterior. The
    threshold is that chain's own median. A chain is good if its
    maximum ever crossed the threshold. If too few survive, keep
    them all rather than trust a tiny survivor set.
    """
    lp = np.asarray(lp, dtype=np.float64)
    n_chains = int(lp.shape[0])
    if lp.ndim != 2 or n_chains < 2:
        return np.ones(n_chains, dtype=bool), True

    chain_max = np.nanmax(lp, axis=1)
    if not np.any(np.isfinite(chain_max)):
        return np.ones(n_chains, dtype=bool), False

    best = int(np.nanargmax(chain_max))
    threshold = np.nanmedian(lp[best])
    mask = chain_max >= threshold
    if int(np.sum(mask)) < min(_MIN_GOOD_CHAINS, n_chains):
        return np.ones(n_chains, dtype=bool), False
    return mask, True


def _thin_stride(n_draws):
    """Stride that leaves at most ``_STAT_DRAW_BUDGET`` draws.

    Parameters
    ----------
    n_draws : int
        Draws per chain.

    Returns
    -------
    stride : int
        At least 1.
    """
    return max(1, int(n_draws) // _STAT_DRAW_BUDGET)


def _rhat_ess(posterior):
    """Worst rank R-hat and worst bulk ESS.

    Parameters
    ----------
    posterior : dict
        Name -> ndarray, shape ``(n_chains, n_draws)`` or with extra
        trailing dimensions.

    Returns
    -------
    stats : tuple or None
        ``(max_rhat, min_ess, worst_rhat_var, worst_ess_var)``.
        ``None`` when arviz cannot form the statistic.

    Notes
    -----
    Eastman, ``exozippy.samplers.convergence._rhat_ess``. Rank
    split-R-hat and bulk ESS are arviz's defaults (Vehtari et al.
    2021). The methods are passed explicitly so a later arviz default
    cannot switch the statistic.
    """
    try:
        import arviz as az
    except ImportError:
        return None

    try:
        idata = az.from_dict({'posterior': posterior})
        rhat_ds = az.rhat(idata, method='rank')
        ess_ds = az.ess(idata, method='bulk')
    except Exception:
        return None

    max_rhat = -np.inf
    worst_rhat = None
    for name, values in rhat_ds.data_vars.items():
        arr = np.asarray(values)
        if arr.size and not np.all(np.isnan(arr)):
            worst = float(np.nanmax(arr))
            if worst > max_rhat:
                max_rhat = worst
                worst_rhat = str(name)

    min_ess = np.inf
    worst_ess = None
    for name, values in ess_ds.data_vars.items():
        arr = np.asarray(values)
        if arr.size and not np.all(np.isnan(arr)):
            worst = float(np.nanmin(arr))
            if worst < min_ess:
                min_ess = worst
                worst_ess = str(name)

    if not np.isfinite(max_rhat) or not np.isfinite(min_ess):
        return None
    return max_rhat, min_ess, worst_rhat, worst_ess


def find_burnin(posterior, lp=None):
    """Burn-in that maximizes the worst parameter's bulk ESS.

    Parameters
    ----------
    posterior : dict
        Name -> ndarray, shape ``(n_chains, n_draws, ...)``.
    lp : array_like, shape (n_chains, n_draws), optional
        Cold-chain log posterior. Used to drop stuck chains before
        the scan. ``None`` keeps every chain.

    Returns
    -------
    diag : dict
        ``burnin`` is the number of leading draws to drop, in the
        original (unthinned) index. ``max_rhat`` and ``min_ess`` are
        the rank R-hat and bulk ESS after that trim. ``good_mask``
        selects the chains that were judged. A trace that is too
        short to score returns ``burnin=0`` and NaN statistics.

    Notes
    -----
    Eastman, ``exozippy.samplers.convergence.find_burnin``. The scan
    is over a fixed grid of fractions. Trimming removes a drifting
    transient, which raises ESS, until further trimming only discards
    samples and ESS falls. The peak is that knee. Thresholds are not
    used to discard a finished run.
    """
    names = list(posterior)
    any_arr = np.asarray(posterior[names[0]])
    n_chains = int(any_arr.shape[0])
    n_draws = int(any_arr.shape[1])

    if lp is not None:
        good_mask, good_reliable = good_chain_mask(lp)
    else:
        good_mask = np.ones(n_chains, dtype=bool)
        good_reliable = True
    good_idx = np.nonzero(good_mask)[0]

    diag = {
        'burnin': 0,
        'burnin_frac': 0.0,
        'max_rhat': float('nan'),
        'min_ess': float('nan'),
        'worst_rhat_var': None,
        'worst_ess_var': None,
        'good_mask': good_mask,
        'good_reliable': good_reliable,
        'n_chains_used': int(good_idx.size),
        'n_draws': int(n_draws),
    }
    if good_idx.size < 2 or n_draws < 2 * _MIN_DRAWS_PER_CHAIN:
        return diag

    # Thin once, then slice. A larger burn-in then simply leaves fewer
    # draws, which is the tradeoff the ESS rule is meant to see.
    thin = _thin_stride(n_draws)
    thinned = {
        name: np.asarray(posterior[name])[good_idx][:, ::thin]
        for name in names
    }
    n_thin = int(thinned[names[0]].shape[1])

    best = None
    for frac in _BURNIN_FRACS:
        burn = int(frac * n_thin)
        if n_thin - burn < _MIN_DRAWS_PER_CHAIN:
            break
        sliced = {name: arr[:, burn:] for name, arr in thinned.items()}
        stat = _rhat_ess(sliced)
        if stat is None:
            continue
        rhat, ess, worst_rhat, worst_ess = stat
        if best is None or ess > best['min_ess']:
            best = {
                'burnin': int(burn * thin),
                'burnin_frac': float(frac),
                'max_rhat': float(rhat),
                'min_ess': float(ess),
                'worst_rhat_var': worst_rhat,
                'worst_ess_var': worst_ess,
            }

    if best is not None:
        diag.update(best)
    return diag


def _record_round_trips(direction, trips):
    """Count cold -> hot -> cold tours after one swap round.

    Parameters
    ----------
    direction : jax.Array, shape (n_temps, n_chains), int32
        Last extreme rung visited by the configuration now in that
        slot. ``+1`` has been at the cold end, ``-1`` has been at the
        hot end, ``0`` has been at neither. Tags move with the state
        when a swap is accepted.
    trips : jax.Array, scalar int32
        Completed round trips so far.

    Returns
    -------
    direction : jax.Array, shape (n_temps, n_chains), int32
        Hot rung set to ``-1``. A cold slot that arrived tagged ``-1``
        is retagged ``+1``.
    trips : jax.Array, scalar int32
        Incremented once per cold slot that just completed a tour.

    Notes
    -----
    Eastman, ``exozippy.samplers.ladder._record_round_trips``. A cold
    slot already tagged ``+1`` is not counted again, so calling this
    every round does not double-count a chain that stays cold.
    """
    hot = direction.shape[0] - 1
    direction = direction.at[hot].set(jnp.int32(-1))

    cold = direction[0]
    arrived = cold == jnp.int32(-1)
    # Keep the counter int32. A bare sum promotes to int64 here.
    n_new = jnp.sum(arrived, dtype=jnp.int32)
    trips = jnp.asarray(trips + n_new, dtype=jnp.int32)
    cold = jnp.where(arrived, jnp.int32(1), cold)
    direction = direction.at[0].set(cold)
    return direction, trips


def _window_runner(logdensity, de_jitter, de_mode_hop, swap_interval):
    """Build the jitted multi-step update.

    Parameters
    ----------
    logdensity : callable
        Scalar log posterior of one vector, shape ``(n_dim,)``.
    de_jitter : float
        Jitter scale. ``0`` disables it.
    de_mode_hop : float
        Probability of a ``gamma = 1`` proposal. ``0`` disables hops.
    swap_interval : int
        Attempt a DEO round every this many steps.

    Returns
    -------
    run : callable
        Jitted ``run(pos, lp, direction, trips, swap_round, gamma,
        temperatures, keys)``. ``keys`` has shape ``(n_steps, 2)``.
    """
    jitter = jnp.float64(de_jitter)
    hop_prob = jnp.float64(de_mode_hop)
    interval = jnp.int32(swap_interval)

    @jax.jit
    def run(
        pos, lp, direction, trips, swap_round, gamma, temperatures, keys,
        step_offset,
    ):
        """Advance every rung for ``keys.shape[0]`` steps.

        Parameters
        ----------
        pos : jax.Array, shape (n_temps, n_chains, n_dim)
            Current population.
        lp : jax.Array, shape (n_temps, n_chains)
            Untempered log posterior at ``pos``.
        direction : jax.Array, shape (n_temps, n_chains), int32
            Round-trip tags.
        trips : jax.Array, scalar int32
            Round trips so far.
        swap_round : jax.Array, scalar int32
            DEO round counter. Even values attempt even pairs.
        gamma : jax.Array, scalar
            DE scale, constant for this window.
        temperatures : jax.Array, shape (n_temps,)
            Ladder, constant for this window.
        keys : jax.Array, shape (n_steps, 2)
            One PRNG key per step.
        step_offset : jax.Array, scalar int32
            Global step index at the start of this window. The swap
            interval is counted from the start of the run, not from
            the start of the window.

        Returns
        -------
        pos, lp, direction, trips, swap_round
            State after the last step.
        outs : tuple
            Per-step ``cold_pos`` ``(n_steps, n_chains, n_dim)``,
            ``cold_lp`` ``(n_steps, n_chains)``, ``accept`` and ``hop``
            ``(n_steps, n_temps, n_chains)``, and swap counts
            ``(n_steps, n_temps - 1)``.
        """
        n_temps, n_chains, dim = pos.shape
        n_gaps = n_temps - 1

        def propose(pop, key):
            """DE proposal for every rung and chain.

            Parameters
            ----------
            pop : jax.Array, shape (n_temps, n_chains, n_dim)
                Population the difference vectors are drawn from.
            key : jax.Array, shape (2,)
                PRNG key.

            Returns
            -------
            prop : jax.Array, shape (n_temps, n_chains, n_dim)
                ``x_i + g (x_j1 - x_j2) + jitter N(0, I)``.
                ``j1 != j2 != i``, same rung. No crossover.
                ``g`` is ``gamma``, or ``1`` on a mode hop.
            hop : jax.Array, shape (n_temps, n_chains), bool
                Which proposals used ``g = 1``.
            """
            rung_keys = jax.random.split(key, n_temps)

            def one_rung(pop_k, key_r):
                """Proposals for one rung.

                Parameters
                ----------
                pop_k : jax.Array, shape (n_chains, n_dim)
                    This rung's population.
                key_r : jax.Array, shape (2,)
                    PRNG key.

                Returns
                -------
                prop_k : jax.Array, shape (n_chains, n_dim)
                    Proposals.
                hop_k : jax.Array, shape (n_chains,), bool
                    Mode-hop flags.
                """
                chain_keys = jax.random.split(key_r, n_chains)

                def one_chain(key_c, i):
                    """One ter Braak proposal.

                    Parameters
                    ----------
                    key_c : jax.Array, shape (2,)
                        PRNG key.
                    i : jax.Array, scalar int32
                        Index of the chain being updated.

                    Returns
                    -------
                    prop_i : jax.Array, shape (n_dim,)
                        Proposed position.
                    hop_i : jax.Array, scalar bool
                        True when this draw used ``gamma = 1``.
                    """
                    k_a, k_b, k_noise, k_hop = jax.random.split(key_c, 4)

                    # Two distinct indices in 0..n-2, then shift so
                    # neither lands on i. Same support as sampling
                    # two partners from the other members.
                    n_pool = n_chains - 1
                    a = jax.random.randint(k_a, (), 0, n_pool)
                    b = jax.random.randint(k_b, (), 0, n_chains - 2)
                    b = jnp.where(b >= a, b + 1, b)
                    j1 = jnp.where(a >= i, a + 1, a)
                    j2 = jnp.where(b >= i, b + 1, b)

                    hop_u = jax.random.uniform(k_hop, dtype=jnp.float64)
                    hop_i = (hop_prob > 0.0) & (hop_u < hop_prob)
                    scale = jnp.where(hop_i, jnp.float64(1.0), gamma)

                    # No crossover: every coordinate uses the same
                    # partners and the same scale.
                    step = scale * (pop_k[j1] - pop_k[j2])
                    noise = jax.random.normal(
                        k_noise, (dim,), dtype=jnp.float64
                    )
                    step = step + jitter * noise
                    return pop_k[i] + step, hop_i

                index = jnp.arange(n_chains, dtype=jnp.int32)
                return jax.vmap(one_chain)(chain_keys, index)

            return jax.vmap(one_rung)(pop, rung_keys)

        def metropolis(pop, logp, prop, prop_lp, key):
            """Temper the full log posterior by ``1/T`` and accept.

            Parameters
            ----------
            pop : jax.Array, shape (n_temps, n_chains, n_dim)
                Current positions.
            logp : jax.Array, shape (n_temps, n_chains)
                Untempered log posterior.
            prop : jax.Array, shape (n_temps, n_chains, n_dim)
                Proposals.
            prop_lp : jax.Array, shape (n_temps, n_chains)
                Untempered log posterior of the proposals.
            key : jax.Array, shape (2,)
                PRNG key.

            Returns
            -------
            pop, logp : jax.Array
                Updated state. A non-finite proposal is rejected.
            accept : jax.Array, shape (n_temps, n_chains), bool
                Which proposals replaced the current state.
            """
            u = jax.random.uniform(
                key, (n_temps, n_chains), dtype=jnp.float64
            )
            # His acceptance: exp(min(0, (lp_new - lp_old) / T)).
            # T = 1 is the posterior. Larger T flattens it.
            delta = (prop_lp - logp) / temperatures[:, None]
            log_alpha = jnp.minimum(jnp.float64(0.0), delta)
            accept = (
                jnp.isfinite(prop_lp)
                & jnp.isfinite(log_alpha)
                & (u < jnp.exp(log_alpha))
            )
            pop = jnp.where(accept[:, :, None], prop, pop)
            logp = jnp.where(accept, prop_lp, logp)
            return pop, logp, accept

        def deo_round(pop, logp, tags, round_idx, key):
            """One DEO round: even or odd disjoint pairs, one permutation.

            Parameters
            ----------
            pop : jax.Array, shape (n_temps, n_chains, n_dim)
                Positions after the Metropolis step.
            logp : jax.Array, shape (n_temps, n_chains)
                Matching log posteriors.
            tags : jax.Array, shape (n_temps, n_chains), int32
                Round-trip tags. They swap with the state.
            round_idx : jax.Array, scalar int32
                DEO round counter carried across steps. Even values
                attempt pairs ``(0, 1), (2, 3), ...``.
            key : jax.Array, shape (2,)
                PRNG key.

            Returns
            -------
            pop, logp, tags
                After the attempted swaps.
            n_acc, n_prop : jax.Array, shape (n_temps - 1,), int32
                Accepted and proposed swaps on each adjacent gap.
                A gap that this parity does not touch stays zero.
            """
            key_perm, key_u = jax.random.split(key)
            # One permutation for the whole round. Cold chain i is
            # paired with hot chain perm[i] on every attempted gap.
            perm = jax.random.permutation(key_perm, n_chains)
            n_acc = jnp.zeros((n_gaps,), dtype=jnp.int32)
            n_prop = jnp.zeros((n_gaps,), dtype=jnp.int32)
            # The carried round index, not the value from the start of
            # the scan. Even and odd rounds have to alternate.
            parity = jnp.mod(round_idx, jnp.int32(2))
            max_pairs = max(n_temps // 2, 1)

            def pair_body(i, state):
                """Attempt pair ``(parity + 2i, parity + 2i + 1)``.

                Parameters
                ----------
                i : jax.Array, scalar int32
                    Pair slot.
                state : tuple
                    ``pop, logp, tags, n_acc, n_prop``.

                Returns
                -------
                state : tuple
                    Updated after this pair, or unchanged when the
                    slot is past the last rung.
                """
                pop_s, logp_s, tags_s, acc_s, prop_s = state
                left = parity + jnp.int32(2) * jnp.asarray(i, dtype=jnp.int32)
                valid = left < jnp.int32(n_gaps)
                left_i = jnp.clip(left, 0, n_temps - 1)
                right_i = jnp.clip(left + jnp.int32(1), 0, n_temps - 1)
                gap_i = jnp.clip(left, 0, max(n_gaps - 1, 0))

                pos_c = pop_s[left_i]
                pos_h = pop_s[right_i]
                pos_h_perm = pos_h[perm]
                lp_c = logp_s[left_i]
                lp_h = logp_s[right_i][perm]
                tag_c = tags_s[left_i]
                tag_h = tags_s[right_i][perm]

                t_c = temperatures[left_i]
                t_h = temperatures[right_i]
                # (lp_hot - lp_cold) * (1/T_cold - 1/T_hot).
                log_a = (lp_h - lp_c) * (1.0 / t_c - 1.0 / t_h)
                u = jax.random.uniform(
                    jax.random.fold_in(key_u, i),
                    (n_chains,),
                    dtype=jnp.float64,
                )
                log_alpha = jnp.minimum(jnp.float64(0.0), log_a)
                accept = valid & jnp.isfinite(log_alpha) & (
                    u < jnp.exp(log_alpha)
                )

                new_c = jnp.where(accept[:, None], pos_h_perm, pos_c)
                new_h_perm = jnp.where(accept[:, None], pos_c, pos_h_perm)
                new_h = pos_h.at[perm].set(new_h_perm)
                new_lp_c = jnp.where(accept, lp_h, lp_c)
                new_lp_h_perm = jnp.where(accept, lp_c, lp_h)
                new_lp_h = logp_s[right_i].at[perm].set(new_lp_h_perm)
                new_tag_c = jnp.where(accept, tag_h, tag_c)
                new_tag_h_perm = jnp.where(accept, tag_c, tag_h)
                new_tag_h = tags_s[right_i].at[perm].set(new_tag_h_perm)

                def commit(arrays):
                    """Write this pair's rows back into the ladder.

                    Parameters
                    ----------
                    arrays : tuple
                        ``pop, logp, tags``.

                    Returns
                    -------
                    arrays : tuple
                        With the cold and hot rows of this pair updated.
                    """
                    pop_i, logp_i, tags_i = arrays
                    pop_i = pop_i.at[left_i].set(new_c)
                    pop_i = pop_i.at[right_i].set(new_h)
                    logp_i = logp_i.at[left_i].set(new_lp_c)
                    logp_i = logp_i.at[right_i].set(new_lp_h)
                    tags_i = tags_i.at[left_i].set(new_tag_c)
                    tags_i = tags_i.at[right_i].set(new_tag_h)
                    return pop_i, logp_i, tags_i

                pop_s, logp_s, tags_s = jax.lax.cond(
                    valid,
                    commit,
                    lambda arrays: arrays,
                    (pop_s, logp_s, tags_s),
                )
                n_hit = jnp.sum(accept, dtype=jnp.int32)
                acc_s = acc_s.at[gap_i].add(
                    jnp.where(valid, n_hit, jnp.int32(0)).astype(jnp.int32)
                )
                prop_s = prop_s.at[gap_i].add(
                    jnp.where(
                        valid, jnp.int32(n_chains), jnp.int32(0)
                    ).astype(jnp.int32)
                )
                return pop_s, logp_s, tags_s, acc_s, prop_s

            pop, logp, tags, n_acc, n_prop = jax.lax.fori_loop(
                0,
                max_pairs,
                pair_body,
                (pop, logp, tags, n_acc, n_prop),
            )
            return pop, logp, tags, n_acc, n_prop

        def one_step(carry, key):
            """One synchronous PTDE step.

            Parameters
            ----------
            carry : tuple
                ``pos, lp, direction, trips, swap_round, step_idx``.
            key : jax.Array, shape (2,)
                PRNG key.

            Returns
            -------
            carry : tuple
                Updated state and the next step index.
            out : tuple
                Cold position and log posterior, accept and hop
                flags, and this step's swap counts.
            """
            pop, logp, tags, n_trips, rnd, step_idx = carry
            k_prop, k_mh, k_swap = jax.random.split(key, 3)

            # All rungs and chains propose from the pre-step population.
            # Nothing is updated until every log posterior is in.
            prop, hop = propose(pop, k_prop)
            flat = prop.reshape((n_temps * n_chains, dim))
            prop_lp = jax.vmap(logdensity)(flat)
            prop_lp = prop_lp.reshape((n_temps, n_chains))
            pop, logp, accept = metropolis(
                pop, logp, prop, prop_lp, k_mh
            )

            zeros = jnp.zeros((n_gaps,), dtype=jnp.int32)

            def swap_yes(operand):
                """DEO round, then the round-trip tag update.

                Parameters
                ----------
                operand : tuple
                    ``pop, logp, tags, trips, swap_round, key``.

                Returns
                -------
                state : tuple
                    Updated population, tags, trip count, round
                    counter, and per-gap swap counts.
                """
                pop_s, logp_s, tags_s, trips_s, rnd_s, key_s = operand
                pop_s, logp_s, tags_s, n_acc, n_prop = deo_round(
                    pop_s, logp_s, tags_s, rnd_s, key_s
                )
                # Tags move during the swaps. The extreme-rung update
                # runs even when this parity attempted no pairs, which
                # is the odd round of a 2-rung ladder.
                tags_s, trips_s = _record_round_trips(tags_s, trips_s)
                rnd_s = rnd_s + jnp.int32(1)
                return pop_s, logp_s, tags_s, trips_s, rnd_s, n_acc, n_prop

            def swap_no(operand):
                """Leave the ladder where the Metropolis step put it.

                Parameters
                ----------
                operand : tuple
                    Same layout as ``swap_yes``.

                Returns
                -------
                state : tuple
                    Unchanged population and zero swap counts.
                """
                pop_s, logp_s, tags_s, trips_s, rnd_s, _key_s = operand
                return pop_s, logp_s, tags_s, trips_s, rnd_s, zeros, zeros

            do_swap = jnp.equal(
                jnp.mod(step_idx + jnp.int32(1), interval), jnp.int32(0)
            )
            if n_temps > 1:
                pop, logp, tags, n_trips, rnd, n_acc, n_prop = jax.lax.cond(
                    do_swap,
                    swap_yes,
                    swap_no,
                    (pop, logp, tags, n_trips, rnd, k_swap),
                )
            else:
                n_acc, n_prop = zeros, zeros

            carry = (pop, logp, tags, n_trips, rnd, step_idx + jnp.int32(1))
            out = (pop[0], logp[0], accept, hop, n_acc, n_prop)
            return carry, out

        step0 = jnp.asarray(step_offset, dtype=jnp.int32)
        init = (pos, lp, direction, trips, swap_round, step0)
        final, outs = jax.lax.scan(one_step, init, keys)
        pos, lp, direction, trips, swap_round, _step = final
        return pos, lp, direction, trips, swap_round, outs

    return run


def _posterior_dict(draws):
    """Pack cold draws as one arviz variable per coordinate.

    Parameters
    ----------
    draws : ndarray, shape (n_chains, n_draws, n_dim)
        Cold-rung positions.

    Returns
    -------
    posterior : dict
        ``'x{i}' -> ndarray, shape (n_chains, n_draws)``.
    """
    draws = np.asarray(draws, dtype=np.float64)
    posterior = {}
    for i in range(draws.shape[-1]):
        posterior[f'x{i}'] = draws[:, :, i]
    return posterior


def _window_rates(accept, hop):
    """T=1 acceptance with mode hops removed.

    Parameters
    ----------
    accept : jax.Array, shape (n_steps, n_temps, n_chains)
        Metropolis accept flags.
    hop : jax.Array, shape (n_steps, n_temps, n_chains)
        Mode-hop flags. Hops are excluded from the gamma rate because
        they are deliberately oversized and mostly rejected.

    Returns
    -------
    rate : float
        ``(n_accept - n_hop_accept) / (n_propose - n_hop_propose)``
        on rung 0. Zero proposals yields ``0``.
    """
    accept0 = np.asarray(accept[:, 0, :])
    hop0 = np.asarray(hop[:, 0, :])
    n_acc = float(np.sum(accept0))
    n_hop_acc = float(np.sum(accept0 & hop0))
    n_prop = float(accept0.size)
    n_hop_prop = float(np.sum(hop0))
    denom = n_prop - n_hop_prop
    if denom <= 0.0:
        return 0.0
    return (n_acc - n_hop_acc) / denom


def _sum_swaps(swap_accept, swap_propose):
    """Sum per-step swap counts over a window.

    Parameters
    ----------
    swap_accept, swap_propose : array_like, shape (n_steps, n_gaps)
        Counts returned by one scan.

    Returns
    -------
    accept, propose : ndarray, shape (n_gaps,)
        Totals. An empty gap axis (one rung) stays length 0.
    """
    accept = np.asarray(swap_accept, dtype=np.float64)
    propose = np.asarray(swap_propose, dtype=np.float64)
    if accept.ndim == 1:
        return accept, propose
    return accept.sum(axis=0), propose.sum(axis=0)


def sample_ptde(
    logdensity,
    positions,
    *,
    n_temps=DEFAULT_N_TEMPS,
    T_max=DEFAULT_T_MAX,
    n_draws=1000,
    n_tune=500,
    gamma=None,
    target_accept=TARGET_ACCEPT,
    adapt_gamma=True,
    adapt_ladder=False,
    de_jitter=DE_JITTER,
    de_mode_hop=0.0,
    swap_interval=1,
    seed=0,
    verbose=False,
):
    """Run synchronous PTDE and post-process the cold rung.

    Parameters
    ----------
    logdensity : callable
        Untempered log posterior of one unconstrained vector, shape
        ``(n_dim,)``. For a BAGLE model this is likelihood + prior +
        log-Jacobian. The sampler divides differences by ``T``.
    positions : array_like, shape (n_chains, n_dim)
        Cold-rung start. Copied onto every hotter rung. Needs at
        least ``d + 2`` chains.
    n_temps : int or str, optional
        Rungs, or ``'auto'``. Default 8.
    T_max : float, optional
        Hottest temperature. Default 200.
    n_draws : int, optional
        Cold-rung steps kept after tune. Default 1000.
    n_tune : int, optional
        Steps discarded as tune, during which gamma and, if asked,
        the ladder adapt. Default 500.
    gamma : float or None, optional
        DE scale. ``None`` uses ``2.38 / sqrt(2d)``.
    target_accept : float, optional
        Gamma adaptation target on the cold rung. Default 0.20.
    adapt_gamma : bool, optional
        Adapt gamma during tune. Default True.
    adapt_ladder : bool, optional
        Barrier-equalizing ladder updates during tune. Default False,
        which keeps the geometric ladder. The ladder is fixed once
        tune ends.
    de_jitter : float, optional
        Jitter scale. Default ``1e-4``.
    de_mode_hop : float, optional
        Probability of a ``gamma = 1`` proposal. Default 0. Must be
        in ``[0, 1)``. Hops are left out of the gamma acceptance.
    swap_interval : int, optional
        Steps between DEO rounds. Default 1.
    seed : int, optional
        PRNG seed. Default 0.
    verbose : bool, optional
        Print gamma and ladder updates. Default False.

    Returns
    -------
    result : dict
        ``samples`` is the cold rung after the good-chain cut and the
        ESS-maximizing burn-in, shape ``(n_kept * n_draws_kept, d)``,
        chain-major. ``cold_draws`` is the untrimmed cold rung, shape
        ``(n_chains, n_draws, d)``. ``mean_accept`` is the cold-rung
        draw-phase acceptance. ``step_size`` and ``gamma`` are the
        final DE scale. ``swap_accept_rate`` has shape ``(n_temps-1,)``
        and is measured on the draw phase only. ``round_trips`` counts
        the whole run. ``logz`` is NaN: this sampler does not estimate
        evidence. ``rhat_max`` and ``ess_min`` are rank R-hat and bulk
        ESS after the trim.

    Notes
    -----
    Jason Eastman, EXOZIPPy ``ptde_sample``. Proposals on one step all
    read the population from the start of the step. The draw phase is
    one ``lax.scan``. Tune adaptation runs in Python between scans.
    """
    jax.config.update('jax_enable_x64', True)

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2:
        raise ValueError(
            'positions must have shape (n_chains, n_dim), '
            f'got {positions.shape}'
        )
    n_chains, n_dim = positions.shape
    resolve_n_chains(n_chains, n_dim)

    hop = float(de_mode_hop or 0.0)
    if not 0.0 <= hop < 1.0:
        raise ValueError(
            f'de_mode_hop must be in [0, 1), got {hop}'
        )
    if int(swap_interval) < 1:
        raise ValueError(
            f'swap_interval must be >= 1, got {swap_interval}'
        )
    if int(n_draws) < 1:
        raise ValueError(f'n_draws must be >= 1, got {n_draws}')
    if int(n_tune) < 0:
        raise ValueError(f'n_tune must be >= 0, got {n_tune}')

    n_temps = resolve_n_temps(n_temps, n_dim, T_max)
    temperatures = geometric_ladder(n_temps, T_max)
    if gamma is None:
        gamma = 2.38 / np.sqrt(2.0 * n_dim)
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError(f'gamma must be >= 0, got {gamma}')

    n_draws = int(n_draws)
    n_tune = int(n_tune)
    total_steps = n_tune + n_draws
    log_every = max(1, total_steps // 20)
    adapt_every = adaptation_window(n_tune, log_every)

    # Same start on every rung. His initvals are replicated that way.
    pos = np.broadcast_to(positions, (n_temps, n_chains, n_dim)).copy()
    pos = jnp.asarray(pos, dtype=jnp.float64)

    @jax.jit
    def eval_all(pop):
        """Log posterior of every slot.

        Parameters
        ----------
        pop : jax.Array, shape (n_temps, n_chains, n_dim)
            Population.

        Returns
        -------
        logp : jax.Array, shape (n_temps, n_chains)
            Untempered log posterior.
        """
        flat = pop.reshape((-1, pop.shape[-1]))
        values = jax.vmap(logdensity)(flat)
        return values.reshape(pop.shape[:-1])

    lp = eval_all(pos)
    direction = jnp.zeros((n_temps, n_chains), dtype=jnp.int32)
    trips = jnp.int32(0)
    swap_round = jnp.int32(0)
    key = jax.random.PRNGKey(int(seed))
    runner = _window_runner(
        logdensity, float(de_jitter), hop, int(swap_interval)
    )

    def run_steps(
        pos, lp, direction, trips, swap_round, gamma, temps,
        n_steps, key, step_offset,
    ):
        """Scan ``n_steps`` and return the new state plus outputs.

        Parameters
        ----------
        pos, lp, direction, trips, swap_round
            Sampler state. See ``sample_ptde``.
        gamma : float
            DE scale for this window.
        temps : ndarray, shape (n_temps,)
            Ladder for this window.
        n_steps : int
            Steps to take.
        key : jax.Array, shape (2,)
            PRNG key. One child is consumed.
        step_offset : int
            Global step index at the first step of this window.

        Returns
        -------
        pos, lp, direction, trips, swap_round, key, outs
            Updated state, the unused key, and the scan outputs.
        """
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, int(n_steps))
        gamma_j = jnp.asarray(gamma, dtype=jnp.float64)
        temps_j = jnp.asarray(temps, dtype=jnp.float64)
        pos, lp, direction, trips, swap_round, outs = runner(
            pos,
            lp,
            direction,
            trips,
            swap_round,
            gamma_j,
            temps_j,
            keys,
            jnp.int32(step_offset),
        )
        return pos, lp, direction, trips, swap_round, key, outs

    # Tune windows. Adaptation reads only the window that just finished,
    # which is his counter reset at each boundary.
    n_full = n_tune // adapt_every if adapt_every > 0 else 0
    remainder = n_tune - n_full * adapt_every
    step_offset = 0
    for _window in range(n_full):
        pos, lp, direction, trips, swap_round, key, outs = run_steps(
            pos, lp, direction, trips, swap_round,
            gamma, temperatures, adapt_every, key, step_offset,
        )
        step_offset += adapt_every
        _cold, _cold_lp, accept, hop_flag, sw_acc, sw_prop = outs
        rate = _window_rates(accept, hop_flag)
        gap_acc, gap_prop = _sum_swaps(sw_acc, sw_prop)

        if adapt_gamma and rate > 0.0:
            gamma_new = next_gamma(gamma, rate, target_accept)
            # His loop only commits a change above 1 percent.
            if abs(gamma_new - gamma) / max(gamma, 1.0e-30) > 0.01:
                if verbose:
                    print(
                        f'PTDE gamma: {gamma:.4f} -> {gamma_new:.4f} '
                        f'(T=1 accept={rate:.3f}, '
                        f'target={float(target_accept):.2f})'
                    )
                gamma = gamma_new

        if (
            adapt_ladder
            and n_temps > 2
            and float(np.sum(gap_prop)) > 0.0
        ):
            new_T = update_ladder_barrier(
                temperatures, gap_acc, gap_prop
            )
            if np.all(np.isfinite(new_T)) and not np.allclose(
                new_T, temperatures
            ):
                if verbose:
                    text = ', '.join(f'{t:.1f}' for t in new_T)
                    print(f'PTDE ladder (barrier-equalized): T=[{text}]')
                temperatures = np.asarray(new_T, dtype=np.float64)

    # A partial window does not adapt. Its counts are dropped, matching
    # the tune -> draw reset, which starts the reported rates at zero.
    if remainder > 0:
        pos, lp, direction, trips, swap_round, key, _outs = run_steps(
            pos, lp, direction, trips, swap_round,
            gamma, temperatures, remainder, key, step_offset,
        )
        step_offset += remainder

    pos, lp, direction, trips, swap_round, key, outs = run_steps(
        pos, lp, direction, trips, swap_round,
        gamma, temperatures, n_draws, key, step_offset,
    )
    cold_pos, cold_lp, accept, hop_flag, sw_acc, sw_prop = outs

    # Scan axis is step, then chain. Diagnostics want chain, then draw.
    cold_draws = np.swapaxes(np.asarray(cold_pos, dtype=np.float64), 0, 1)
    cold_lp_np = np.swapaxes(np.asarray(cold_lp, dtype=np.float64), 0, 1)
    draw_accept = _window_rates(accept, hop_flag)
    gap_acc, gap_prop = _sum_swaps(sw_acc, sw_prop)
    swap_rate = np.zeros(gap_acc.shape, dtype=np.float64)
    measured = gap_prop > 0.0
    swap_rate[measured] = gap_acc[measured] / gap_prop[measured]

    posterior = _posterior_dict(cold_draws)
    diag = find_burnin(posterior, cold_lp_np)
    good = np.asarray(diag['good_mask'], dtype=bool)
    burnin = int(diag['burnin'])
    kept = cold_draws[good, burnin:]
    if kept.size == 0:
        kept = cold_draws

    n_kept = int(kept.shape[0])
    n_kept_draws = int(kept.shape[1])
    # Chain-major flat layout, same as the other BlackJAX helpers.
    samples = np.reshape(kept, (n_kept * n_kept_draws, n_dim))

    result = {
        'samples': samples,
        'cold_draws': cold_draws,
        'cold_lp': cold_lp_np,
        'final_pos': np.asarray(pos, dtype=np.float64),
        'loglikes': None,
        'weights': None,
        'logz': float(np.nan),
        'mean_accept': float(draw_accept),
        'step_size': float(gamma),
        'rhat_max': float(diag['max_rhat']),
        'ess_min': float(diag['min_ess']),
        'L': float(np.nan),
        'beta': float(np.nan),
        'gamma': float(gamma),
        'round_trips': int(trips),
        'swap_accept_rate': swap_rate,
        'swap_accept': gap_acc,
        'swap_propose': gap_prop,
        'temperatures': np.asarray(temperatures, dtype=np.float64),
        'burnin': burnin,
        'burnin_frac': float(diag['burnin_frac']),
        'good_mask': good,
        'good_reliable': bool(diag['good_reliable']),
        'n_chains_kept': n_kept,
        'n_draws_kept': n_kept_draws,
        'n_temps': int(n_temps),
        'n_chains': int(n_chains),
        'ladder_lambda': ladder_health(temperatures, gap_acc, gap_prop),
    }
    if verbose:
        print(
            f'PTDE draw accept={draw_accept:.3f}  gamma={gamma:.4f}  '
            f'round_trips={int(trips)}  '
            f'rhat={diag["max_rhat"]:.4f}  ess={diag["min_ess"]:.1f}  '
            f'burnin={burnin}'
        )
    return result
