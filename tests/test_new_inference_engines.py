"""Fast end-to-end checks for the extra microlensing inference engines.

DEMetropolisZ, nautilus, pocoMC, replica exchange, and BlackJAX are run
on a tiny PSPL photometry-only light curve. A shorter
photometry+astrometry run covers the joint likelihood when the engine
is cheap enough.
"""
import os

import numpy as np
import pytest

from bagle import model_fitter_jax as model_fitter
from bagle import model_jax as model
from bagle.model_fitter_jax import (
    MicrolensSolver,
    MicrolensSolverBlackJAX,
    MicrolensSolverImportance,
    MicrolensSolverNumPyro,
    MicrolensSolverPyMC,
)


_TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_OUTPUT_DIR = os.path.join(_TESTS_DIR, 'test_output', 'new_engines')
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


def _tiny_pspl_phot(seed=0, n_obs=28):
    """Synthetic PSPL photometry-only event with a known truth.

    Parameters
    ----------
    seed : int, optional
        Noise seed.
    n_obs : int, optional
        Number of photometric epochs.

    Returns
    -------
    data : dict
        BAGLE data dictionary (photometry only).
    truth : dict
        Fitter-parameter truth, including ``b_sff1`` and ``mag_src1``.
    """
    truth = {
        't0': 57000.0,
        'u0_amp': 0.25,
        'tE': 16.0,
        'piE_E': 0.20,
        'piE_N': 0.18,
        'b_sff1': 0.95,
        'mag_src1': 18.0,
    }
    t = np.linspace(truth['t0'] - 45.0, truth['t0'] + 45.0, int(n_obs))
    pspl = model.PSPL_Phot_noPar_Param1(
        truth['t0'],
        truth['u0_amp'],
        truth['tE'],
        truth['piE_E'],
        truth['piE_N'],
        [truth['b_sff1']],
        [truth['mag_src1']],
    )
    mag = np.asarray(pspl.get_photometry(t), dtype=float).reshape(-1)
    err = np.full(mag.shape, 0.02)
    rng = np.random.default_rng(seed)
    mag_obs = mag + rng.normal(0.0, err)

    data = {
        'target': 'tiny_pspl_phot',
        'phot_data': 'sim',
        'ast_data': 'sim',
        'phot_files': ['tiny_phot1'],
        'ast_files': ['tiny_ast_unused'],
        't_phot1': t,
        'mag1': mag_obs,
        'mag_err1': err,
    }
    return data, truth


def _tiny_pspl_phot_astrom(seed=1):
    """Short PSPL photometry + astrometry event.

    Parameters
    ----------
    seed : int, optional
        Noise seed.

    Returns
    -------
    data : dict
        BAGLE data dictionary with one photometry and one astrometry set.
    truth : dict
        Physical-parameter truth used by ``PSPL_PhotAstrom_noPar_Param1``.
    """
    truth = {
        'mL': 10.0,
        't0': 57000.0,
        'beta': -0.4,
        'dL': 4000.0,
        'dL_dS': 0.5,
        'xS0_E': 0.0,
        'xS0_N': 0.0,
        'muL_E': 0.0,
        'muL_N': -7.0,
        'muS_E': 1.5,
        'muS_N': -0.5,
        'b_sff1': 0.98,
        'mag_src1': 19.0,
    }
    pspl = model.PSPL_PhotAstrom_noPar_Param1(
        truth['mL'], truth['t0'], truth['beta'], truth['dL'],
        truth['dL_dS'], truth['xS0_E'], truth['xS0_N'],
        truth['muL_E'], truth['muL_N'], truth['muS_E'], truth['muS_N'],
        [truth['b_sff1']], [truth['mag_src1']],
    )
    t_phot = np.linspace(truth['t0'] - 40.0, truth['t0'] + 40.0, 24)
    t_ast = np.linspace(truth['t0'] - 60.0, truth['t0'] + 60.0, 8)
    rng = np.random.default_rng(seed)

    mag = np.asarray(pspl.get_photometry(t_phot), dtype=float).reshape(-1)
    mag_err = np.full(mag.shape, 0.03)
    mag_obs = mag + rng.normal(0.0, mag_err)

    pos = np.asarray(pspl.get_astrometry(t_ast), dtype=float)
    ast_err = np.full(pos.shape, 2.0e-4)
    pos_obs = pos + rng.normal(0.0, ast_err)

    data = {
        'target': 'tiny_pspl_joint',
        'phot_data': 'sim',
        'ast_data': 'sim',
        'phot_files': ['tiny_phot1'],
        'ast_files': ['tiny_ast1'],
        't_phot1': t_phot,
        'mag1': mag_obs,
        'mag_err1': mag_err,
        't_ast1': t_ast,
        'xpos1': pos_obs[:, 0],
        'ypos1': pos_obs[:, 1],
        'xpos_err1': ast_err[:, 0],
        'ypos_err1': ast_err[:, 1],
    }
    return data, truth


def _apply_phot_priors(fitter, truth):
    """Put moderately wide priors around the photometry-only truth.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver whose ``priors`` are replaced.
    truth : dict
        Truth dictionary from :func:`_tiny_pspl_phot`.

    Returns
    -------
    None
    """
    # Positive u0 only, so the light curve's sign degeneracy is gone.
    # piE stays away from 0, where the direction vector is undefined.
    fitter.priors['t0'] = model_fitter.make_gen(
        't0', truth['t0'] - 6.0, truth['t0'] + 6.0
    )
    fitter.priors['u0_amp'] = model_fitter.make_gen('u0_amp', 0.05, 0.55)
    fitter.priors['tE'] = model_fitter.make_gen('tE', 8.0, 28.0)
    fitter.priors['piE_E'] = model_fitter.make_gen('piE_E', 0.05, 0.40)
    fitter.priors['piE_N'] = model_fitter.make_gen('piE_N', 0.05, 0.40)
    fitter.priors['b_sff1'] = model_fitter.make_gen('b_sff1', 0.75, 1.0)
    fitter.priors['mag_src1'] = model_fitter.make_gen(
        'mag_src1', truth['mag_src1'] - 0.4, truth['mag_src1'] + 0.4
    )
    return None


def _apply_joint_priors(fitter, truth):
    """Narrow priors for the short photometry+astrometry run.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver whose ``priors`` are replaced.
    truth : dict
        Truth dictionary from :func:`_tiny_pspl_phot_astrom`.

    Returns
    -------
    None
    """
    fitter.priors['mL'] = model_fitter.make_gen('mL', 8.0, 12.0)
    fitter.priors['t0'] = model_fitter.make_gen(
        't0', truth['t0'] - 4.0, truth['t0'] + 4.0
    )
    fitter.priors['beta'] = model_fitter.make_gen('beta', -0.6, -0.2)
    fitter.priors['dL'] = model_fitter.make_gen('dL', 3600.0, 4400.0)
    fitter.priors['dL_dS'] = model_fitter.make_gen('dL_dS', 0.4, 0.6)
    fitter.priors['xS0_E'] = model_fitter.make_gen('xS0_E', -0.01, 0.01)
    fitter.priors['xS0_N'] = model_fitter.make_gen('xS0_N', -0.01, 0.01)
    fitter.priors['muL_E'] = model_fitter.make_gen('muL_E', -0.8, 0.8)
    fitter.priors['muL_N'] = model_fitter.make_gen('muL_N', -8.0, -6.0)
    fitter.priors['muS_E'] = model_fitter.make_gen('muS_E', 0.8, 2.2)
    fitter.priors['muS_N'] = model_fitter.make_gen('muS_N', -1.2, 0.2)
    fitter.priors['b_sff1'] = model_fitter.make_gen('b_sff1', 0.9, 1.0)
    fitter.priors['mag_src1'] = model_fitter.make_gen(
        'mag_src1', 18.7, 19.3
    )
    return None


def _assert_samples(fitter, truth, expect_logz, t0_atol, extra_atol):
    """Check table layout, logZ, and that the mode is near the truth.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver that has already been solved.
    truth : dict
        Parameter truth. Keys that are also fitter parameters are checked
        when they appear in ``extra_atol``.
    expect_logz : bool
        Whether ``summary['logZ']`` must be finite.
    t0_atol : float
        Absolute tolerance on the maximum-likelihood ``t0``.
    extra_atol : dict
        Absolute tolerances for other fitter parameters.

    Returns
    -------
    None
    """
    tab = fitter.load_mnest_results()
    summary = fitter.load_mnest_summary()

    assert len(tab) > 0
    assert 'weights' in tab.colnames
    assert 'logLike' in tab.colnames
    assert np.all(np.isfinite(tab['logLike']))
    assert np.all(np.isfinite(tab['weights']))
    assert np.isclose(float(np.sum(tab['weights'])), 1.0)

    for name in fitter.fitter_param_names:
        assert name in tab.colnames
        assert np.all(np.isfinite(np.asarray(tab[name], dtype=float)))

    # Same on-disk layout the other solvers write.
    assert os.path.exists(fitter.outputfiles_basename + '.txt')
    assert os.path.exists(fitter.outputfiles_basename + '.fits')

    logz = float(summary['logZ'][0])
    if expect_logz:
        assert np.isfinite(logz)
    else:
        assert np.isnan(logz)
    assert np.isfinite(float(summary['maxlogL'][0]))

    best = fitter.get_best_fit(def_best='maxl')
    assert abs(float(best['t0']) - float(truth['t0'])) < t0_atol
    for name, atol in extra_atol.items():
        assert abs(float(best[name]) - float(truth[name])) < atol

    # The best sample should be competitive with the injected truth.
    lnL_best = float(fitter.log_likely(best))
    lnL_truth = float(fitter.log_likely(truth))
    assert lnL_best > lnL_truth - 12.0
    return None


def _assert_replica_diagnostics(fitter):
    """Check DEO swap rates and round trips on the summary row.

    Parameters
    ----------
    fitter : MicrolensSolverNumPyro
        Replica-exchange solver that has already been solved.

    Returns
    -------
    None
    """
    summary = fitter.load_mnest_summary()
    n_gaps = int(fitter.n_temperatures) - 1
    rates = []
    for gap in range(n_gaps):
        rate = float(summary['swap_accept_' + str(gap)][0])
        assert np.isfinite(rate)
        assert 0.0 <= rate <= 1.0
        rates.append(rate)

    trips = int(summary['n_round_trips'][0])
    assert trips >= 0
    assert np.allclose(np.asarray(fitter._swap_accept_rate), rates)
    # At least one neighboring pair should exchange on these ladders.
    assert max(rates) > 0.0
    print(
        'replica swap_accept', rates,
        'round_trips', trips,
        'logZ', float(summary['logZ'][0]),
    )
    return None


def _assert_inside_prior(fitter):
    """Every posterior row lies inside each prior's support.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver that has already been solved.

    Returns
    -------
    None
    """
    from bagle.model_fitter_jax import _prior_bounds

    tab = fitter.load_mnest_results()
    for name in fitter.fitter_param_names:
        low, high = _prior_bounds(fitter.priors[name])
        column = np.asarray(tab[name], dtype=float)
        assert np.all(np.isfinite(column))
        if np.isfinite(low):
            assert np.all(column >= low - 1.0e-8), name
        if np.isfinite(high):
            assert np.all(column <= high + 1.0e-8), name
    return None


def test_unknown_pymc_sampler_rejected():
    """PyMC solver rejects sampler names it does not implement."""
    pytest.importorskip('pymc')
    data, _truth = _tiny_pspl_phot()
    with pytest.raises(ValueError, match='demetropolisz'):
        MicrolensSolverPyMC(
            data,
            model.PSPL_Phot_noPar_Param1,
            outputfiles_basename=os.path.join(TEST_OUTPUT_DIR, 'bad_pymc_'),
            sampler='not_a_sampler',
        )
    return None


def test_unknown_numpyro_sampler_rejected():
    """NumPyro solver rejects sampler names it does not implement."""
    pytest.importorskip('numpyro')
    data, _truth = _tiny_pspl_phot()
    with pytest.raises(ValueError, match='replica_exchange'):
        MicrolensSolverNumPyro(
            data,
            model.PSPL_Phot_noPar_Param1,
            outputfiles_basename=os.path.join(
                TEST_OUTPUT_DIR, 'bad_numpyro_'
            ),
            sampler='not_a_sampler',
        )
    return None


def test_demetropolisz_phot():
    """DEMetropolisZ recovers a tiny PSPL photometry posterior."""
    pytest.importorskip('pymc')
    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'demetropolisz_')
    fitter = MicrolensSolverPyMC(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
        sampler='demetropolis_z',
        draws=200,
        tune=500,
        chains=4,
        cores=1,
        de_scaling=0.01,
        pymc_random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(fitter, truth)
    fitter.solve()

    tab = fitter.load_mnest_results()
    assert len(tab) == 200 * 4
    assert fitter.sampler == 'demetropolisz'
    _assert_samples(
        fitter,
        truth,
        expect_logz=False,
        t0_atol=2.5,
        extra_atol={
            'u0_amp': 0.15,
            'tE': 5.0,
            'mag_src1': 0.3,
            'b_sff1': 0.15,
        },
    )
    return None


def test_replica_exchange_phot():
    """Replica exchange cold chain lands near the photometry truth."""
    pytest.importorskip('numpyro')
    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'replica_')
    fitter = MicrolensSolverNumPyro(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
        sampler='parallel_tempering',
        draws=100,
        tune=120,
        chains=2,
        n_temperatures=4,
        n_leapfrog=4,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(fitter, truth)
    fitter.solve()
    _assert_inside_prior(fitter)

    tab = fitter.load_mnest_results()
    assert len(tab) == 100 * 2
    assert fitter.sampler == 'replica_exchange'
    _assert_replica_diagnostics(fitter)
    _assert_samples(
        fitter,
        truth,
        expect_logz=True,
        t0_atol=2.5,
        extra_atol={
            'u0_amp': 0.15,
            'tE': 5.0,
            'mag_src1': 0.3,
            'b_sff1': 0.15,
        },
    )
    return None


def test_nautilus_phot():
    """Nautilus returns finite logZ and a posterior near the truth."""
    pytest.importorskip('nautilus')
    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'nautilus_')
    fitter = MicrolensSolverImportance(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
        sampler='nautilus',
        n_live=60,
        n_networks=2,
        f_live=0.05,
        n_eff=200,
        n_workers=1,
        posterior_samples=120,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(fitter, truth)
    fitter.solve()

    tab = fitter.load_mnest_results()
    assert len(tab) == 120
    _assert_samples(
        fitter,
        truth,
        expect_logz=True,
        t0_atol=2.5,
        extra_atol={
            'u0_amp': 0.15,
            'tE': 5.0,
            'mag_src1': 0.3,
            'b_sff1': 0.15,
        },
    )
    return None


def test_pocomc_phot():
    """pocoMC returns finite logZ and a posterior near the truth."""
    pytest.importorskip('pocomc')
    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'pocomc_')
    fitter = MicrolensSolverImportance(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
        sampler='pocomc',
        n_effective=48,
        n_active=24,
        n_total=160,
        n_evidence=80,
        flow='nsf3',
        precondition=True,
        n_workers=1,
        posterior_samples=80,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(fitter, truth)
    fitter.solve()

    tab = fitter.load_mnest_results()
    assert len(tab) == 80
    assert np.isfinite(fitter._logZ_err)
    _assert_samples(
        fitter,
        truth,
        expect_logz=True,
        t0_atol=2.5,
        extra_atol={
            'u0_amp': 0.15,
            'tE': 5.0,
            'mag_src1': 0.3,
            'b_sff1': 0.15,
        },
    )
    return None


def test_replica_exchange_phot_astrom():
    """Replica exchange also runs on a tiny photometry+astrometry event."""
    pytest.importorskip('numpyro')
    data, truth = _tiny_pspl_phot_astrom()
    out = os.path.join(TEST_OUTPUT_DIR, 'replica_joint_')
    fitter = MicrolensSolverNumPyro(
        data,
        model.PSPL_PhotAstrom_noPar_Param1,
        outputfiles_basename=out,
        sampler='replica_exchange',
        draws=200,
        tune=250,
        chains=2,
        n_temperatures=4,
        n_leapfrog=4,
        random_seed=1,
        verbose=False,
    )
    _apply_joint_priors(fitter, truth)
    fitter.solve()
    _assert_inside_prior(fitter)
    _assert_replica_diagnostics(fitter)
    _assert_samples(
        fitter,
        truth,
        expect_logz=True,
        t0_atol=3.0,
        extra_atol={'beta': 0.2, 'mag_src1': 0.35},
    )
    return None


def test_demetropolisz_phot_astrom():
    """DEMetropolisZ runs on a tiny photometry+astrometry event."""
    pytest.importorskip('pymc')
    data, truth = _tiny_pspl_phot_astrom()
    out = os.path.join(TEST_OUTPUT_DIR, 'demetropolisz_joint_')
    fitter = MicrolensSolverPyMC(
        data,
        model.PSPL_PhotAstrom_noPar_Param1,
        outputfiles_basename=out,
        sampler='demetropolisz',
        draws=150,
        tune=400,
        chains=4,
        cores=1,
        de_scaling=0.01,
        pymc_random_seed=1,
        verbose=False,
    )
    _apply_joint_priors(fitter, truth)
    fitter.solve()
    _assert_samples(
        fitter,
        truth,
        expect_logz=False,
        t0_atol=3.0,
        extra_atol={'beta': 0.2, 'mag_src1': 0.35},
    )
    return None


def test_nautilus_phot_astrom():
    """Nautilus runs on a tiny photometry+astrometry event."""
    pytest.importorskip('nautilus')
    data, truth = _tiny_pspl_phot_astrom()
    out = os.path.join(TEST_OUTPUT_DIR, 'nautilus_joint_')
    fitter = MicrolensSolverImportance(
        data,
        model.PSPL_PhotAstrom_noPar_Param1,
        outputfiles_basename=out,
        sampler='nautilus',
        n_live=50,
        n_networks=2,
        f_live=0.08,
        n_eff=150,
        n_workers=1,
        posterior_samples=80,
        random_seed=1,
        verbose=False,
    )
    _apply_joint_priors(fitter, truth)
    fitter.solve()
    _assert_samples(
        fitter,
        truth,
        expect_logz=True,
        t0_atol=3.0,
        extra_atol={'beta': 0.2, 'mag_src1': 0.35},
    )
    return None


def test_pocomc_phot_astrom():
    """pocoMC runs on a tiny photometry+astrometry event."""
    pytest.importorskip('pocomc')
    data, truth = _tiny_pspl_phot_astrom()
    out = os.path.join(TEST_OUTPUT_DIR, 'pocomc_joint_')
    fitter = MicrolensSolverImportance(
        data,
        model.PSPL_PhotAstrom_noPar_Param1,
        outputfiles_basename=out,
        sampler='pocomc',
        n_effective=40,
        n_active=20,
        n_total=120,
        n_evidence=60,
        flow='nsf3',
        precondition=True,
        n_workers=1,
        posterior_samples=60,
        random_seed=1,
        verbose=False,
    )
    _apply_joint_priors(fitter, truth)
    fitter.solve()
    _assert_samples(
        fitter,
        truth,
        expect_logz=True,
        t0_atol=3.0,
        extra_atol={'beta': 0.2, 'mag_src1': 0.35},
    )
    return None


def test_nautilus_missing_dependency_message(monkeypatch):
    """Requesting nautilus without the package raises a clear ImportError."""
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == 'nautilus' or name.startswith('nautilus.'):
            raise ImportError('blocked for test')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _blocked)
    with pytest.raises(ImportError, match='nautilus-sampler'):
        model_fitter._import_nautilus()
    return None


def test_pocomc_missing_dependency_message(monkeypatch):
    """Requesting pocoMC without the package raises a clear ImportError."""
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == 'pocomc' or name.startswith('pocomc.'):
            raise ImportError('blocked for test')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _blocked)
    with pytest.raises(ImportError, match='pocomc'):
        model_fitter._import_pocomc()
    return None


def test_evaluate_loglik_jax_batch_matches_point():
    """Batched JAX log-likelihood matches pointwise evaluation."""
    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'batch_lnL_')
    fitter = MicrolensSolver(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
    )
    names = list(fitter.fitter_param_names)
    theta = np.array([float(truth[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(0)
    batch = np.repeat(theta[None, :], 4, axis=0)
    batch[1:] = batch[1:] + rng.normal(0.0, 0.01, size=(3, theta.size))

    point = np.array(
        [float(fitter.evaluate_loglik_jax(batch[i])) for i in range(4)]
    )
    vector = np.asarray(fitter.evaluate_loglik_jax(batch), dtype=float)
    assert vector.shape == (4,)
    assert np.allclose(vector, point, rtol=1e-5, atol=1e-5)

    # Second call must hit the cached jitted vmap.
    cached = fitter._explicit_jax_loglik_vmap_cache
    vector2 = np.asarray(fitter.evaluate_loglik_jax(batch), dtype=float)
    assert fitter._explicit_jax_loglik_vmap_cache[0] is cached[0]
    assert fitter._explicit_jax_loglik_vmap_cache[1] is cached[1]
    assert np.allclose(vector2, point, rtol=1e-5, atol=1e-5)
    return None


def _assert_blackjax_diagnostics(fitter, expect_rhat):
    """Check acceptance, step size, and MCMC diagnostics on the summary.

    Parameters
    ----------
    fitter : MicrolensSolverBlackJAX
        Solver that has already been solved.
    expect_rhat : bool
        Whether R-hat and ESS must be finite (MCMC with enough draws).

    Returns
    -------
    None
    """
    summary = fitter.load_mnest_summary()
    accept = float(summary['mean_accept'][0])
    step = float(summary['step_size'][0])
    print(
        'blackjax', fitter.sampler,
        'accept', accept,
        'step', step,
        'rhat', float(summary['rhat_max'][0]),
        'ess', float(summary['ess_min'][0]),
        'logZ', float(summary['logZ'][0]),
    )
    if fitter.sampler == 'smc':
        beta = float(summary['smc_beta'][0])
        print('smc_beta', beta)
        assert beta > 0.95
    if fitter.sampler in ('nuts', 'mclmc', 'smc'):
        assert np.isfinite(accept)
        assert 0.0 <= accept <= 1.0
        assert np.isfinite(step)
        assert step > 0.0
    if expect_rhat:
        assert np.isfinite(float(summary['rhat_max'][0]))
        assert np.isfinite(float(summary['ess_min'][0]))
    if fitter.sampler == 'mclmc':
        assert np.isfinite(float(summary['mclmc_L'][0]))
        assert float(summary['mclmc_L'][0]) > 0.0
    return None


def test_blackjax_nuts_phot():
    """Window-adapted BlackJAX NUTS recovers a tiny PSPL light curve."""
    pytest.importorskip('blackjax')
    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'blackjax_nuts_')
    fitter = MicrolensSolverBlackJAX(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
        sampler='nuts',
        draws=40,
        tune=80,
        chains=2,
        max_num_doublings=4,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(fitter, truth)
    fitter.solve()
    _assert_inside_prior(fitter)

    tab = fitter.load_mnest_results()
    assert len(tab) == 40 * 2
    assert fitter.sampler == 'nuts'
    _assert_blackjax_diagnostics(fitter, expect_rhat=True)
    _assert_samples(
        fitter,
        truth,
        expect_logz=False,
        t0_atol=2.5,
        extra_atol={
            'u0_amp': 0.15,
            'tE': 5.0,
            'mag_src1': 0.3,
            'b_sff1': 0.15,
        },
    )
    return None


def test_blackjax_mclmc_phot():
    """Tuned BlackJAX MCLMC recovers a tiny PSPL light curve."""
    pytest.importorskip('blackjax')
    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'blackjax_mclmc_')
    fitter = MicrolensSolverBlackJAX(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
        sampler='mclmc',
        draws=40,
        chains=2,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(fitter, truth)
    fitter.solve()
    _assert_inside_prior(fitter)

    tab = fitter.load_mnest_results()
    assert len(tab) == 40 * 2
    _assert_blackjax_diagnostics(fitter, expect_rhat=True)
    _assert_samples(
        fitter,
        truth,
        expect_logz=False,
        t0_atol=2.5,
        extra_atol={
            'u0_amp': 0.2,
            'tE': 6.0,
            'mag_src1': 0.4,
            'b_sff1': 0.2,
        },
    )
    return None


def test_blackjax_smc_phot():
    """Adaptive tempered SMC returns finite logZ near the photometry truth."""
    pytest.importorskip('blackjax')
    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'blackjax_smc_')
    fitter = MicrolensSolverBlackJAX(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
        sampler='smc',
        chains=2,
        smc_inner='hmc',
        smc_particles=16,
        smc_mcmc_steps=4,
        smc_integration_steps=8,
        smc_max_iter=30,
        smc_target_ess=0.4,
        initial_step_size=0.05,
        posterior_samples=40,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(fitter, truth)
    fitter.solve()
    _assert_inside_prior(fitter)

    tab = fitter.load_mnest_results()
    assert len(tab) == 40
    _assert_blackjax_diagnostics(fitter, expect_rhat=False)
    _assert_samples(
        fitter,
        truth,
        expect_logz=True,
        t0_atol=3.0,
        extra_atol={
            'u0_amp': 0.25,
            'tE': 8.0,
            'mag_src1': 0.5,
            'b_sff1': 0.25,
        },
    )
    return None


def test_blackjax_ns_phot():
    """BlackJAX nested sampling returns finite logZ near the truth."""
    pytest.importorskip('blackjax')
    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'blackjax_ns_')
    fitter = MicrolensSolverBlackJAX(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
        sampler='ns',
        n_live=20,
        ns_inner_steps=2,
        ns_num_delete=2,
        ns_max_iter=60,
        ns_max_slice_steps=4,
        ns_max_shrinkage=20,
        f_live=0.15,
        posterior_samples=40,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(fitter, truth)
    fitter.solve()
    _assert_inside_prior(fitter)

    tab = fitter.load_mnest_results()
    assert len(tab) == 40
    print('blackjax ns phot logZ', float(fitter._logZ))
    _assert_samples(
        fitter,
        truth,
        expect_logz=True,
        t0_atol=2.5,
        extra_atol={
            'u0_amp': 0.15,
            'tE': 5.0,
            'mag_src1': 0.3,
            'b_sff1': 0.15,
        },
    )
    return None


def test_blackjax_ns_phot_astrom():
    """BlackJAX nested sampling also runs on photometry+astrometry."""
    pytest.importorskip('blackjax')
    data, truth = _tiny_pspl_phot_astrom()
    out = os.path.join(TEST_OUTPUT_DIR, 'blackjax_ns_joint_')
    fitter = MicrolensSolverBlackJAX(
        data,
        model.PSPL_PhotAstrom_noPar_Param1,
        outputfiles_basename=out,
        sampler='ns',
        n_live=24,
        ns_inner_steps=4,
        ns_num_delete=2,
        ns_max_iter=80,
        ns_max_slice_steps=6,
        ns_max_shrinkage=24,
        f_live=0.1,
        posterior_samples=40,
        random_seed=1,
        verbose=False,
    )
    _apply_joint_priors(fitter, truth)
    fitter.solve()
    _assert_inside_prior(fitter)
    print('blackjax ns joint logZ', float(fitter._logZ))
    _assert_samples(
        fitter,
        truth,
        expect_logz=True,
        t0_atol=3.0,
        extra_atol={'beta': 0.2, 'mag_src1': 0.35},
    )
    return None


def test_jax_log_prior_is_neginf_outside_support():
    """The JAX log prior matches ``log_prob`` inside and is ``-inf`` outside."""
    pytest.importorskip('numpyro')
    import jax
    import jax.numpy as jnp

    from bagle.model_fitter_jax import _build_jax_log_prior
    from bagle.model_fitter_jax import _prior_bounds
    from bagle.model_fitter_jax import scipy_to_numpyro_dist

    data, truth = _tiny_pspl_phot()
    out = os.path.join(TEST_OUTPUT_DIR, 'logprior_')
    fitter = MicrolensSolver(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=out,
    )
    _apply_phot_priors(fitter, truth)
    names = list(fitter.fitter_param_names)
    mids = []
    for name in names:
        low, high = _prior_bounds(fitter.priors[name])
        mids.append(0.5 * (low + high))
    theta = jnp.asarray(mids, dtype=jnp.float64)
    log_prior = _build_jax_log_prior(fitter)

    expected = jnp.asarray(0.0, dtype=jnp.float64)
    for i, name in enumerate(names):
        dist = scipy_to_numpyro_dist(fitter.priors[name])
        expected = expected + dist.log_prob(theta[i])
    inside = log_prior(theta)
    assert np.array_equal(np.asarray(inside), np.asarray(expected))

    outside = theta.at[names.index('b_sff1')].set(1.5)
    val, grad = jax.value_and_grad(log_prior)(outside)
    assert float(val) == -np.inf
    assert np.all(np.isfinite(np.asarray(grad)))

    far = theta.at[names.index('t0')].set(float(truth['t0']) + 100.0)
    val_far, grad_far = jax.value_and_grad(log_prior)(far)
    assert float(val_far) == -np.inf
    assert np.all(np.isfinite(np.asarray(grad_far)))
    return None


def test_temperature_ladder_is_geometric():
    """Replica betas are log-spaced, and warmup can shrink a cold gap."""
    from bagle.jax.replica_exchange import _adapt_temperature_ladder
    from bagle.jax.replica_exchange import temperature_ladder

    betas = temperature_ladder(4)
    assert betas.shape == (4,)
    assert betas[0] == 1.0
    assert betas[-1] == 0.0
    assert np.all(np.diff(betas) < 0.0)
    positive = betas[:-1]
    ratios = positive[:-1] / positive[1:]
    assert np.allclose(ratios, ratios[0])
    quadratic = np.linspace(0.0, 1.0, 4) ** 2
    quadratic = quadratic[::-1]
    assert not np.allclose(betas, quadratic)

    # Cold gap rejects, hotter gaps swap. The interior rung moves up.
    tuned = _adapt_temperature_ladder(
        betas,
        accept=np.array([0.0, 10.0, 10.0]),
        attempt=np.array([10.0, 10.0, 10.0]),
    )
    assert tuned[0] == 1.0
    assert tuned[-1] == 0.0
    assert np.isclose(tuned[-2], betas[-2])
    assert tuned[1] > betas[1]
    assert np.all(np.diff(tuned) < 0.0)
    return None


def test_logz_near_nautilus():
    """BlackJAX NS and replica-exchange logZ sit near the nautilus value."""
    pytest.importorskip('nautilus')
    pytest.importorskip('blackjax')
    pytest.importorskip('numpyro')
    data, truth = _tiny_pspl_phot()

    naut = MicrolensSolverImportance(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=os.path.join(TEST_OUTPUT_DIR, 'logz_naut_'),
        sampler='nautilus',
        n_live=60,
        n_networks=2,
        f_live=0.05,
        n_eff=200,
        n_workers=1,
        posterior_samples=80,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(naut, truth)
    naut.solve()
    logz_naut = float(naut._logZ)

    ns = MicrolensSolverBlackJAX(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=os.path.join(TEST_OUTPUT_DIR, 'logz_ns_'),
        sampler='ns',
        n_live=20,
        ns_inner_steps=2,
        ns_num_delete=2,
        ns_max_iter=60,
        ns_max_slice_steps=4,
        ns_max_shrinkage=20,
        f_live=0.15,
        posterior_samples=40,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(ns, truth)
    ns.solve()
    _assert_inside_prior(ns)
    logz_ns = float(ns._logZ)

    replica = MicrolensSolverNumPyro(
        data,
        model.PSPL_Phot_noPar_Param1,
        outputfiles_basename=os.path.join(TEST_OUTPUT_DIR, 'logz_replica_'),
        sampler='parallel_tempering',
        draws=100,
        tune=120,
        chains=2,
        n_temperatures=8,
        n_leapfrog=4,
        random_seed=0,
        verbose=False,
    )
    _apply_phot_priors(replica, truth)
    replica.solve()
    _assert_inside_prior(replica)
    logz_replica = float(replica._logZ)
    rates = np.asarray(replica._swap_accept_rate, dtype=float)
    print(
        'logZ nautilus', logz_naut,
        'blackjax_ns', logz_ns,
        'replica', logz_replica,
        'swap_accept', rates,
    )
    assert np.isfinite(logz_naut)
    assert np.isfinite(logz_ns)
    assert np.isfinite(logz_replica)
    # Short runs. Loose next to a diverged integral, tight enough
    # to catch a hot rung that dominates the trapezoid.
    assert abs(logz_ns - logz_naut) < 30.0
    assert abs(logz_replica - logz_naut) < 40.0
    assert np.all((rates >= 0.0) & (rates <= 1.0))
    assert float(np.max(rates)) > 0.0
    return None


def test_blackjax_missing_dependency_message(monkeypatch):
    """Requesting BlackJAX without the package raises a clear ImportError."""
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == 'blackjax' or name.startswith('blackjax.'):
            raise ImportError('blocked for test')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _blocked)
    with pytest.raises(ImportError, match='blackjax'):
        model_fitter._import_blackjax()
    return None
