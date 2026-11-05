"""
Fitter smoke test: PSTL PhotAstrom with a luminous tertiary.

Synthetic photometry and astrometry come from ``PSTL_PhotAstrom_noPar_Param1``
with ``dmag_Lp_Lt`` set so the tertiary carries light. The solver must carry
``dmag_Lp_Lt1`` as a fitted parameter with a default prior, match host and
JAX log-likelihoods, and recover both lens magnitude differences with a
fit from an offset start.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp
from scipy.optimize import minimize

from bagle import model_jax as mj
from bagle.model_fitter_jax import MicrolensSolver, build_explicit_jax_loglik_fn

FILT = ('xS0_E', 'xS0_N', 'b_sff', 'mag_src', 'dmag_Lp_Ls', 'dmag_Lp_Lt')

# Luminous lenses: lens light is half the baseline flux, and the tertiary
# is brighter than the secondary so dmag_Lp_Lt moves the centroid.
TRUTH = dict(mLp=10.0, mLs=5.0, mLt=2.0, t0=57000.0, xS0_E=0.0, xS0_N=0.01,
             beta=0.4, muL_E=0.0, muL_N=0.0, sep12=3.0, sep13=2.0,
             muS_E=4.0, muS_N=-3.0, dL=2000.0, dS=4000.0, alpha12=90.0,
             alpha13=45.0, b_sff=0.5, mag_src=18.0, dmag_Lp_Ls=3.0,
             dmag_Lp_Lt=0.5)


def _solver(tmp_path):
    """
    Solver on noiseless synthetic data from the TRUTH model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for MultiNest output files.

    Returns
    -------
    fitter : MicrolensSolver
        Solver with default priors.
    cube : dict
        Truth in ``fitter.fitter_param_names`` (suffixed) order.
    """
    host = mj.PSTL_PhotAstrom_noPar_Param1(
        **{k: ([v] if k in FILT else v) for k, v in TRUTH.items()},
        root_tol=1e-8)

    # Dense photometry and sparser astrometry across the event.
    t_phot = np.linspace(56850.0, 57150.0, 30)
    t_ast = np.linspace(56850.0, 57150.0, 12)
    mag = np.asarray(host.get_photometry(t_phot), dtype=float)
    pos = np.asarray(host.get_astrometry(t_ast), dtype=float)

    data = {
        't_phot1': t_phot,
        'mag1': mag,
        'mag_err1': np.full(t_phot.shape, 0.01),
        't_ast1': t_ast,
        'xpos1': pos[:, 0],
        'ypos1': pos[:, 1],
        'xpos_err1': np.full(t_ast.shape, 5e-5),
        'ypos_err1': np.full(t_ast.shape, 5e-5),
        'phot_data': ['I'],
        'ast_data': ['I'],
    }
    fitter = MicrolensSolver(
        data, mj.PSTL_PhotAstrom_noPar_Param1,
        outputfiles_basename=str(tmp_path / 'pstl_dmag_'),
        n_live_points=20, max_iter=1, dump_callback=None, verbose=False)

    # Filter parameters carry a 1-based suffix in the fitter cube.
    cube = {}
    for name in fitter.fitter_param_names:
        base = name[:-1] if name[:-1] in FILT else name
        cube[name] = TRUTH[base]
    return fitter, cube


def test_dmag_Lp_Lt_is_fitted_with_default_prior(tmp_path):
    """dmag_Lp_Lt1 is a fitted parameter with the default [-20, 20] prior."""
    fitter, _cube = _solver(tmp_path)
    assert 'dmag_Lp_Lt1' in fitter.fitter_param_names
    assert 'dmag_Lp_Ls1' in fitter.fitter_param_names

    prior = fitter.priors['dmag_Lp_Lt1']
    assert prior.ppf(0.0) == pytest.approx(-20.0)
    assert prior.ppf(1.0) == pytest.approx(20.0)
    return None


def test_dmag_Lp_Lt_host_vs_jax_and_recovery(tmp_path):
    """Host/JAX lnL agree; a fit recovers both lens dmags."""
    fitter, cube = _solver(tmp_path)
    names = list(fitter.fitter_param_names)
    fn, _ctx = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None

    # Host and JAX agree at the truth and with a moved tertiary light.
    for dlt in (TRUTH['dmag_Lp_Lt'], TRUTH['dmag_Lp_Lt'] + 2.0):
        trial = dict(cube)
        trial['dmag_Lp_Lt1'] = dlt
        lnL_host = float(fitter.log_likely(trial))
        lnL_jax = float(fitter.evaluate_loglik_jax(trial))
        np.testing.assert_allclose(lnL_jax, lnL_host, rtol=1e-8, atol=1e-6)

    # The tertiary light is constrained: moving it costs likelihood.
    vec_true = jnp.asarray([cube[n] for n in names])
    i_lt = names.index('dmag_Lp_Lt1')
    i_ls = names.index('dmag_Lp_Ls1')
    lnL_true = float(fn(vec_true))
    lnL_off = float(fn(vec_true.at[i_lt].add(1.0)))
    assert lnL_true - lnL_off > 10.0

    # Fit (dmag_Lp_Ls1, dmag_Lp_Lt1) from an offset start, rest at truth.
    # Derivative-free: jit-compiling the PSTL gradient takes about a minute.
    def neg_lnl(x):
        """Negative lnL over the two lens dmags."""
        v = vec_true.at[i_ls].set(x[0]).at[i_lt].set(x[1])
        return -float(fn(v))

    x0 = np.array([TRUTH['dmag_Lp_Ls'] + 1.0, TRUTH['dmag_Lp_Lt'] - 1.0])
    res = minimize(neg_lnl, x0, method='Nelder-Mead',
                   options=dict(xatol=1e-5, fatol=1e-6, maxiter=2000))
    assert res.success
    np.testing.assert_allclose(
        res.x, [TRUTH['dmag_Lp_Ls'], TRUTH['dmag_Lp_Lt']], atol=1e-3)
    return None
