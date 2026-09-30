"""BlackJAX NUTS, MCLMC, adaptive SMC, and nested sampling.

Samplers call a JAX log-density built by the fitter. Several chains are
mapped with ``jax.vmap``. When more than one device is visible and the
chain count divides evenly, the same update is also ``jax.pmap``'d
across devices. Nested sampling keeps one live set; BlackJAX maps the
slice updates over those live points.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


def _blackjax():
    """Import BlackJAX or raise an install hint.

    Returns
    -------
    module
        The ``blackjax`` module.
    """
    try:
        import blackjax
    except ImportError as error:
        raise ImportError(
            'BlackJAX sampling requires blackjax. Install with: '
            'pip install "bagle[blackjax]" '
            '(or pip install "blackjax>=1.6,<1.7").'
        ) from error
    return blackjax


def _map_over_chains(one_chain, keys, positions):
    """Map ``one_chain`` across chains, and across devices when possible.

    Parameters
    ----------
    one_chain : callable
        ``one_chain(key, position) -> (value, diag)``. ``diag`` is a
        pytree. ``key`` has shape ``(2,)``.
    keys : jax.Array, shape (n_chains, 2)
        One PRNG key per chain.
    positions : jax.Array
        Leading axis is the chain. Remaining axes are that chain's
        state, for example ``(n_dim,)`` or ``(n_particles, n_dim)``.

    Returns
    -------
    value : jax.Array
        ``one_chain`` outputs stacked on a chain axis of size
        ``n_chains``.
    diag : pytree
        Diagnostics stacked the same way.
    """
    n_chains = int(positions.shape[0])
    n_devices = int(jax.device_count())
    use_pmap = (
        n_devices > 1
        and n_chains >= n_devices
        and n_chains % n_devices == 0
    )

    if not use_pmap:
        return jax.vmap(one_chain)(keys, positions)

    per = n_chains // n_devices
    keys_d = keys.reshape((n_devices, per) + keys.shape[1:])
    pos_d = positions.reshape((n_devices, per) + positions.shape[1:])

    def _on_device(keys_local, pos_local):
        """Vmap chains that share one device.

        Parameters
        ----------
        keys_local : jax.Array, shape (n_local, 2)
            Keys for this device.
        pos_local : jax.Array
            Positions for this device.

        Returns
        -------
        value, diag
            Stacked over the local chains.
        """
        return jax.vmap(one_chain)(keys_local, pos_local)

    value, diag = jax.pmap(_on_device)(keys_d, pos_d)
    value = value.reshape((n_chains,) + value.shape[2:])
    diag = jax.tree.map(
        lambda leaf: leaf.reshape((n_chains,) + leaf.shape[2:]),
        diag,
    )
    return value, diag


def _chain_diagnostics(samples):
    """Rank-normalized R-hat and bulk ESS of MCMC draws.

    Parameters
    ----------
    samples : array_like, shape (n_chains, n_draws, n_dim)
        Posterior draws. R-hat needs at least two chains and four
        draws. ESS needs four draws.

    Returns
    -------
    rhat_max : float
        Maximum split R-hat across parameters. ``nan`` when it cannot
        be computed.
    ess_min : float
        Minimum effective sample size across parameters. ``nan`` when
        it cannot be computed.
    """
    blackjax = _blackjax()
    samples = np.asarray(samples, dtype=float)
    n_chains, n_draws, _n_dim = samples.shape
    rhat_max = np.nan
    ess_min = np.nan
    traced = jnp.asarray(samples)

    if n_draws >= 4:
        try:
            ess = np.asarray(blackjax.ess(traced), dtype=float)
        except (ValueError, TypeError):
            ess = None
        if ess is not None and np.any(np.isfinite(ess)):
            ess_min = float(np.nanmin(ess))

    if n_chains >= 2 and n_draws >= 4:
        try:
            rhat = np.asarray(blackjax.rhat(traced), dtype=float)
        except (ValueError, TypeError):
            rhat = None
        if rhat is not None and np.any(np.isfinite(rhat)):
            rhat_max = float(np.nanmax(rhat))

    return rhat_max, ess_min


def _mean_finite(values):
    """Mean of the finite entries of ``values``.

    Parameters
    ----------
    values : array_like
        Diagnostics, one value per chain or a scalar.

    Returns
    -------
    mean : float
        ``nan`` when no entry is finite.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(np.nan)
    return float(np.mean(arr))


def _empty_result():
    """Diagnostic fields shared by every sampler.

    Returns
    -------
    result : dict
        ``logz``, ``mean_accept``, ``step_size``, ``rhat_max``,
        ``ess_min``, and ``L`` start as ``nan``. ``loglikes`` and
        ``weights`` start as ``None``.
    """
    return {
        'samples': None,
        'loglikes': None,
        'weights': None,
        'logz': float(np.nan),
        'mean_accept': float(np.nan),
        'step_size': float(np.nan),
        'rhat_max': float(np.nan),
        'ess_min': float(np.nan),
        'L': float(np.nan),
        'beta': float(np.nan),
    }


def sample_nuts(
    logdensity_fn,
    positions,
    n_warmup,
    n_draws,
    seed,
    target_accept=0.80,
    max_num_doublings=10,
    initial_step_size=0.5,
    inverse_mass_matrix=None,
):
    """Window-adapted BlackJAX NUTS, one chain per row of ``positions``.

    Parameters
    ----------
    logdensity_fn : callable
        Scalar JAX log posterior ``log p(theta)``, ``theta`` shape
        ``(n_dim,)``.
    positions : array_like, shape (n_chains, n_dim)
        Initial positions, one chain per row.
    n_warmup : int
        Window-adaptation steps per chain.
    n_draws : int
        Draws kept per chain after adaptation.
    seed : int
        PRNG seed.
    target_accept : float, optional
        Dual-averaging target acceptance. Default 0.80.
    max_num_doublings : int, optional
        NUTS trajectory cap. Default 10.
    initial_step_size : float, optional
        Step size at the start of adaptation, in mass-matrix units.
        Default 0.5.
    inverse_mass_matrix : array_like, shape (n_dim,), optional
        Diagonal inverse-mass seed. ``None`` starts from the identity.

    Returns
    -------
    result : dict
        ``samples`` has shape ``(n_chains * n_draws, n_dim)`` in chain
        order. ``mean_accept`` and ``step_size`` are averages over
        chains. ``rhat_max`` and ``ess_min`` summarize the chains.
        ``logz`` is ``nan``.
    """
    blackjax = _blackjax()
    positions = jnp.asarray(positions, dtype=jnp.float64)
    n_chains = int(positions.shape[0])
    n_warmup = int(n_warmup)
    n_draws = int(n_draws)
    doublings = int(max_num_doublings)

    adapt_kwargs = dict(
        initial_step_size=float(initial_step_size),
        target_acceptance_rate=float(target_accept),
        max_num_doublings=doublings,
    )
    if inverse_mass_matrix is not None:
        adapt_kwargs['initial_inverse_mass_matrix'] = jnp.asarray(
            inverse_mass_matrix, dtype=jnp.float64
        )

    # Built once. vmap/pmap trace ``warmup.run`` per chain.
    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity_fn, **adapt_kwargs
    )
    kernel = blackjax.nuts.build_kernel()
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), n_chains)

    def one_chain(key, position):
        """Adapt, then draw, for a single chain.

        Parameters
        ----------
        key : jax.Array, shape (2,)
            PRNG key.
        position : jax.Array, shape (n_dim,)
            Starting position.

        Returns
        -------
        draws : jax.Array, shape (n_draws, n_dim)
            Posterior positions.
        diag : dict
            Mean acceptance and adapted step size.
        """
        warm_key, draw_key = jax.random.split(key)
        (state, params), _info = warmup.run(warm_key, position, n_warmup)

        def draw(state, key):
            """One NUTS transition.

            Parameters
            ----------
            state : blackjax HMC state
                Current chain state.
            key : jax.Array, shape (2,)
                PRNG key.

            Returns
            -------
            state, (position, acceptance)
                Updated state, the new position, and the trajectory
                acceptance probability.
            """
            state, info = kernel(
                key,
                state,
                logdensity_fn,
                params['step_size'],
                params['inverse_mass_matrix'],
                doublings,
            )
            return state, (state.position, info.acceptance_rate)

        draw_keys = jax.random.split(draw_key, n_draws)
        _state, (draws, accept) = jax.lax.scan(draw, state, draw_keys)
        diag = {
            'accept': jnp.mean(accept),
            'step_size': params['step_size'],
        }
        return draws, diag

    draws, diag = _map_over_chains(one_chain, keys, positions)
    draws_np = np.asarray(draws, dtype=float)
    rhat_max, ess_min = _chain_diagnostics(draws_np)

    result = _empty_result()
    result['samples'] = draws_np.reshape(-1, draws_np.shape[-1])
    result['mean_accept'] = _mean_finite(diag['accept'])
    result['step_size'] = _mean_finite(diag['step_size'])
    result['rhat_max'] = rhat_max
    result['ess_min'] = ess_min
    return result


def sample_mclmc(
    logdensity_fn,
    positions,
    n_draws,
    seed,
    num_effective_samples=50,
    inverse_mass_matrix=None,
    initial_step_size=0.1,
):
    """MCLMC with BlackJAX's ``mclmc_find_L_and_step_size`` tuner.

    Parameters
    ----------
    logdensity_fn : callable
        Scalar JAX log posterior, ``theta`` shape ``(n_dim,)``.
        ``n_dim`` must be at least 2.
    positions : array_like, shape (n_chains, n_dim)
        Initial positions.
    n_draws : int
        Draws kept per chain after tuning. The tuner length is a
        fraction of this count.
    seed : int
        PRNG seed.
    num_effective_samples : int, optional
        ESS target inside the L adaptation. Default 50.
    inverse_mass_matrix : array_like, shape (n_dim,), optional
        Diagonal inverse-mass seed for the tuner. ``None`` starts from
        the identity.
    initial_step_size : float, optional
        Step-size seed in mass-matrix units. Default 0.1.

    Returns
    -------
    result : dict
        ``samples`` has shape ``(n_chains * n_draws, n_dim)``.
        ``mean_accept`` is the fraction of steps that stayed finite.
        ``step_size`` and ``L`` are chain averages. ``logz`` is ``nan``.
    """
    blackjax = _blackjax()
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    positions = jnp.asarray(positions, dtype=jnp.float64)
    n_chains, n_dim = int(positions.shape[0]), int(positions.shape[1])
    if int(n_dim) < 2:
        raise ValueError(
            'MCLMC needs at least 2 parameters, got %d.' % int(n_dim)
        )

    n_draws = int(n_draws)
    n_ess = max(2, int(num_effective_samples))
    if inverse_mass_matrix is None:
        imm = jnp.ones((n_dim,), dtype=jnp.float64)
    else:
        imm = jnp.asarray(inverse_mass_matrix, dtype=jnp.float64)
    # Prior-width preconditioning keeps the first tuner steps on scale.
    init_params = MCLMCAdaptationState(
        L=jnp.asarray(float(np.sqrt(n_dim)), dtype=jnp.float64),
        step_size=jnp.asarray(float(initial_step_size), dtype=jnp.float64),
        inverse_mass_matrix=imm,
    )
    kernel = blackjax.mclmc.build_kernel()
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), int(n_chains))

    def one_chain(key, position):
        """Tune L and the step size, then draw.

        Parameters
        ----------
        key : jax.Array, shape (2,)
            PRNG key.
        position : jax.Array, shape (n_dim,)
            Starting position.

        Returns
        -------
        draws : jax.Array, shape (n_draws, n_dim)
            Posterior positions.
        diag : dict
            Finite-step fraction, step size, and momentum-decoherence
            scale ``L``.
        """
        init_key, tune_key, draw_key = jax.random.split(key, 3)
        state = blackjax.mclmc.init(position, logdensity_fn, init_key)
        state, params, _n_tune = blackjax.mclmc_find_L_and_step_size(
            kernel,
            num_steps=n_draws,
            state=state,
            rng_key=tune_key,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=True,
            num_effective_samples=n_ess,
            params=init_params,
        )

        def draw(state, key):
            """One MCLMC transition.

            Parameters
            ----------
            state : MCLMC integrator state
                Current state.
            key : jax.Array, shape (2,)
                PRNG key.

            Returns
            -------
            state, (position, finite)
                Updated state, the new position, and whether the step
                stayed finite.
            """
            state, info = kernel(
                key,
                state,
                logdensity_fn,
                params.inverse_mass_matrix,
                params.L,
                params.step_size,
            )
            finite = info.nonans.astype(jnp.float64)
            return state, (state.position, finite)

        draw_keys = jax.random.split(draw_key, n_draws)
        _state, (draws, finite) = jax.lax.scan(draw, state, draw_keys)
        diag = {
            'accept': jnp.mean(finite),
            'step_size': params.step_size,
            'L': params.L,
        }
        return draws, diag

    draws, diag = _map_over_chains(one_chain, keys, positions)
    draws_np = np.asarray(draws, dtype=float)
    rhat_max, ess_min = _chain_diagnostics(draws_np)

    result = _empty_result()
    result['samples'] = draws_np.reshape(-1, draws_np.shape[-1])
    result['mean_accept'] = _mean_finite(diag['accept'])
    result['step_size'] = _mean_finite(diag['step_size'])
    result['L'] = _mean_finite(diag['L'])
    result['rhat_max'] = rhat_max
    result['ess_min'] = ess_min
    return result


def sample_smc(
    logprior_fn,
    loglikelihood_fn,
    particles,
    seed,
    step_size,
    inverse_mass_matrix,
    inner='hmc',
    n_mcmc_steps=10,
    n_integration_steps=8,
    max_num_doublings=6,
    target_ess=0.5,
    max_iter=25,
):
    """Adaptive tempered SMC with an HMC or NUTS inner kernel.

    Parameters
    ----------
    logprior_fn : callable
        Scalar JAX log prior, argument shape ``(n_dim,)``.
    loglikelihood_fn : callable
        Scalar JAX log likelihood, argument shape ``(n_dim,)``.
    particles : array_like, shape (n_chains, n_particles, n_dim)
        Initial particles, usually prior draws. Each chain is an
        independent SMC run.
    seed : int
        PRNG seed.
    step_size : float
        Inner-kernel step size, shared by every particle.
    inverse_mass_matrix : array_like, shape (n_dim,)
        Diagonal inverse mass, shared by every particle.
    inner : {'hmc', 'nuts'}, optional
        Inner MCMC kernel. Default ``'hmc'``.
    n_mcmc_steps : int, optional
        Inner MCMC steps per tempering stage. Default 10.
    n_integration_steps : int, optional
        HMC leapfrog steps. Ignored for NUTS. Default 8.
    max_num_doublings : int, optional
        NUTS trajectory cap. Ignored for HMC. Default 6.
    target_ess : float, optional
        Target ESS fraction used to pick the next temperature.
        Default 0.5.
    max_iter : int, optional
        Maximum tempering stages. Default 25.

    Returns
    -------
    result : dict
        ``samples`` has shape ``(n_chains * n_particles, n_dim)`` and
        ``weights`` matches that length. Chain weights each sum to
        ``1 / n_chains``. ``logz`` is the log-mean-exp of the chain
        evidences. ``mean_accept`` averages the inner-kernel
        acceptance over stages that advanced the temperature.
    """
    blackjax = _blackjax()
    from blackjax.smc.resampling import systematic

    particles = jnp.asarray(particles, dtype=jnp.float64)
    n_dim = int(particles.shape[-1])
    inner = str(inner).lower()
    if inner not in ('hmc', 'nuts'):
        raise ValueError(
            "SMC inner kernel must be 'hmc' or 'nuts', got %r." % inner
        )

    # Leading length 1 marks a parameter shared by every particle.
    if inner == 'nuts':
        kernel = blackjax.nuts.build_kernel()

        def mcmc_step(
            rng_key,
            state,
            logdensity,
            step_size,
            inverse_mass_matrix,
            max_num_doublings,
        ):
            """One NUTS proposal on the tempered density.

            Parameters
            ----------
            rng_key : jax.Array
                PRNG key.
            state : HMC state
                Current particle state.
            logdensity : callable
                Tempered log posterior.
            step_size : float
                Integrator step size.
            inverse_mass_matrix : jax.Array, shape (n_dim,)
                Diagonal inverse mass.
            max_num_doublings : int
                Trajectory cap.

            Returns
            -------
            state, info
                Updated state and NUTS info.
            """
            return kernel(
                rng_key,
                state,
                logdensity,
                step_size,
                inverse_mass_matrix,
                max_num_doublings,
            )

        mcmc_init = blackjax.nuts.init
        mcmc_parameters = {
            'step_size': jnp.asarray([step_size], dtype=jnp.float64),
            'inverse_mass_matrix': jnp.asarray(
                inverse_mass_matrix, dtype=jnp.float64
            )[None, :],
            'max_num_doublings': jnp.asarray([int(max_num_doublings)]),
        }
    else:
        kernel = blackjax.hmc.build_kernel()

        def mcmc_step(
            rng_key,
            state,
            logdensity,
            step_size,
            inverse_mass_matrix,
            num_integration_steps,
        ):
            """One HMC proposal on the tempered density.

            Parameters
            ----------
            rng_key : jax.Array
                PRNG key.
            state : HMC state
                Current particle state.
            logdensity : callable
                Tempered log posterior.
            step_size : float
                Integrator step size.
            inverse_mass_matrix : jax.Array, shape (n_dim,)
                Diagonal inverse mass.
            num_integration_steps : int
                Leapfrog steps.

            Returns
            -------
            state, info
                Updated state and HMC info.
            """
            return kernel(
                rng_key,
                state,
                logdensity,
                step_size,
                inverse_mass_matrix,
                num_integration_steps,
            )

        mcmc_init = blackjax.hmc.init
        mcmc_parameters = {
            'step_size': jnp.asarray([step_size], dtype=jnp.float64),
            'inverse_mass_matrix': jnp.asarray(
                inverse_mass_matrix, dtype=jnp.float64
            )[None, :],
            'num_integration_steps': jnp.asarray(
                [int(n_integration_steps)]
            ),
        }

    alg = blackjax.adaptive_tempered_smc(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step,
        mcmc_init,
        mcmc_parameters,
        systematic,
        target_ess=float(target_ess),
        num_mcmc_steps=int(n_mcmc_steps),
    )
    n_chains = int(particles.shape[0])
    max_iter = int(max_iter)
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), n_chains)
    step_size_value = jnp.asarray(step_size, dtype=jnp.float64)

    def one_chain(key, cloud):
        """Anneal one particle cloud from the prior to the posterior.

        Parameters
        ----------
        key : jax.Array, shape (2,)
            PRNG key.
        cloud : jax.Array, shape (n_particles, n_dim)
            Initial particles.

        Returns
        -------
        cloud : jax.Array, shape (n_particles, n_dim)
            Final particles.
        diag : dict
            Normalized weights, log evidence, and mean inner-kernel
            acceptance.
        """
        state = alg.init(cloud)
        stage_keys = jax.random.split(key, max_iter)

        def body(carry, stage_key):
            """One adaptive tempering stage.

            Parameters
            ----------
            carry
                ``(state, logz, done, accept_sum, n_stages)``.
            stage_key : jax.Array, shape (2,)
                PRNG key.

            Returns
            -------
            carry, None
                Updated evidence. Stages after ``β=1`` leave the
                state unchanged.
            """
            state, logz, done, accept_sum, n_stages = carry
            new_state, info = alg.step(stage_key, state)
            accept = jnp.mean(info.update_info.acceptance_rate)
            use = jnp.logical_not(done)
            logz = jnp.where(
                use, logz + info.log_likelihood_increment, logz
            )
            accept_sum = jnp.where(use, accept_sum + accept, accept_sum)
            n_stages = jnp.where(use, n_stages + 1.0, n_stages)
            state = jax.tree.map(
                lambda old, new: jnp.where(done, old, new),
                state,
                new_state,
            )
            done = done | (state.tempering_param >= 1.0 - 1.0e-6)
            return (state, logz, done, accept_sum, n_stages), None

        carry0 = (
            state,
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        (state, logz, _done, accept_sum, n_stages), _ = jax.lax.scan(
            body, carry0, stage_keys
        )
        diag = {
            'weights': state.weights,
            'logz': logz,
            'accept': accept_sum / jnp.maximum(n_stages, 1.0),
            'step_size': step_size_value,
            'beta': state.tempering_param,
        }
        return state.particles, diag

    clouds, diag = _map_over_chains(one_chain, keys, particles)
    clouds_np = np.asarray(clouds, dtype=float)
    weights = np.asarray(diag['weights'], dtype=float)
    # Each chain's weights sum to 1. Give the chains equal weight.
    weights = weights / float(n_chains)
    logz = np.asarray(diag['logz'], dtype=float).reshape(-1)

    result = _empty_result()
    result['samples'] = clouds_np.reshape(-1, n_dim)
    result['weights'] = weights.reshape(-1)
    result['logz'] = float(
        jax.scipy.special.logsumexp(jnp.asarray(logz)) - np.log(n_chains)
    )
    result['mean_accept'] = _mean_finite(diag['accept'])
    result['step_size'] = _mean_finite(diag['step_size'])
    result['beta'] = _mean_finite(diag['beta'])
    return result


def sample_ns(
    logprior_fn,
    loglikelihood_fn,
    positions,
    seed,
    n_inner,
    n_delete,
    max_iter,
    max_slice_steps,
    max_shrinkage,
    f_live,
    n_posterior,
    verbose=False,
):
    """BlackJAX nested slice sampling (``blackjax.ns.nss``).

    Parameters
    ----------
    logprior_fn : callable
        Scalar JAX log prior, argument shape ``(n_dim,)``.
    loglikelihood_fn : callable
        Scalar JAX log likelihood, argument shape ``(n_dim,)``.
    positions : array_like, shape (n_live, n_dim)
        Initial live points, drawn from the prior.
    seed : int
        PRNG seed.
    n_inner : int
        Slice steps per deletion.
    n_delete : int
        Live points deleted at each nested-sampling step.
    max_iter : int
        Maximum nested-sampling steps.
    max_slice_steps : int
        Cap on slice stepping-out expansions.
    max_shrinkage : int
        Cap on slice shrinkage evaluations.
    f_live : float
        Stop when the live-set evidence fraction drops below this.
    n_posterior : int
        Equal-weight posterior draws returned by
        ``blackjax.ns.utils.sample``.
    verbose : bool, optional
        Print logZ every 10 steps. Default False.

    Returns
    -------
    result : dict
        ``samples`` has shape ``(n_posterior, n_dim)``. ``loglikes``
        matches that length. ``logz`` is
        ``logaddexp(logZ_dead, logZ_live)``.
    """
    blackjax = _blackjax()
    from blackjax.ns.utils import finalise
    from blackjax.ns.utils import sample as ns_sample

    # ``blackjax.nss`` is the nested slice sampler in ``blackjax.ns``.
    alg = blackjax.nss(
        logprior_fn,
        loglikelihood_fn,
        num_inner_steps=int(n_inner),
        num_delete=int(n_delete),
        max_steps=int(max_slice_steps),
        max_shrinkage=int(max_shrinkage),
    )

    key = jax.random.PRNGKey(int(seed))
    key, init_key = jax.random.split(key)
    state = alg.init(
        jnp.asarray(positions, dtype=jnp.float64), init_key
    )
    step = jax.jit(alg.step)

    dead = []
    log_floor = np.log(max(float(f_live), 1.0e-16))
    for iteration in range(int(max_iter)):
        key, subkey = jax.random.split(key)
        state, info = step(subkey, state)
        dead.append(info)

        log_z = float(state.integrator.logZ)
        log_z_live = float(state.integrator.logZ_live)
        log_z_tot = float(np.logaddexp(log_z, log_z_live))
        remain = log_z_live - log_z_tot
        if verbose and (iteration % 10 == 0):
            print('blackjax ns step %d: logZ≈%.3f' % (iteration, log_z_tot))
        if np.isfinite(remain) and remain < log_floor:
            break

    final = finalise(state, dead, update_info=False)
    key, sample_key = jax.random.split(key)
    drawn = ns_sample(sample_key, final, shape=int(n_posterior))

    result = _empty_result()
    result['samples'] = np.asarray(drawn.position, dtype=float)
    result['loglikes'] = np.asarray(drawn.loglikelihood, dtype=float)
    result['logz'] = float(np.logaddexp(
        np.asarray(state.integrator.logZ, dtype=float),
        np.asarray(state.integrator.logZ_live, dtype=float),
    ))
    return result
