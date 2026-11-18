"""
Step 2: JAX likelihood registry and missing-class errors.

Checks that ``JAX_LIKELIHOOD_CLASSES`` names real concrete classes with the
required ``jax_log_likely_*`` methods, that JAX samplers raise a clear
``NotImplementedError`` for a class outside the list, and that the host
``MicrolensSolver`` still works for that class.
"""
from __future__ import annotations

import numpy as np
import pytest

from bagle import model_jax as mj
from bagle.model_fitter_jax import (
    MicrolensSolver,
    MicrolensSolverBlackJAX,
    MicrolensSolverJaxLike,
    MicrolensSolverNumPyro,
    build_explicit_jax_loglik_fn,
)

# Concrete class with host photometry/astrometry but no JAX likelihood.
_MISSING_CLASS = mj.PSPL_Astrom_Par_Param4
_MISSING_NAME = _MISSING_CLASS.__name__


def _astrom_data():
    """
    Minimal single-filter astrometry data for ``PSPL_Astrom_Par_Param4``.

    Returns
    -------
    data : dict
        Fitter data dictionary.
    cube : dict
        Parameter point in ``fitter_param_names`` order (filter suffixes).
    """
    t0 = 57000.0
    t = np.linspace(t0 - 40.0, t0 + 40.0, 25)
    xpos = 0.001 * (t - t0) / 365.25
    data = {
        't_ast1': t,
        'xpos1': xpos,
        'ypos1': -0.5 * xpos,
        'xpos_err1': np.full_like(t, 1e-4),
        'ypos_err1': np.full_like(t, 1e-4),
        'raL': 268.0,
        'decL': -29.0,
    }
    cube = {
        't0': t0,
        'u0_amp': 0.1,
        'tE': 30.0,
        'thetaE': 1.0,
        'piS': 0.1,
        'piE_E': 0.05,
        'piE_N': 0.05,
        'xS0_E1': 0.0,
        'xS0_N1': 0.0,
        'muS_E': 0.0,
        'muS_N': 0.0,
        'b_sff1': 1.0,
    }

    return data, cube


def test_registry_matches_methods():
    """Every registry name is a concrete class with the required JAX methods."""
    assert isinstance(mj.JAX_LIKELIHOOD_CLASSES, frozenset)
    assert len(mj.JAX_LIKELIHOOD_CLASSES) > 0

    for name in sorted(mj.JAX_LIKELIHOOD_CLASSES):
        assert hasattr(mj, name), f'{name} missing from model_jax'
        cls = getattr(mj, name)
        assert isinstance(cls, type), f'{name} is not a class'

        # Concrete composed models inherit ModelClassABC.
        assert issubclass(cls, mj.ModelClassABC), f'{name} not ModelClassABC'

        need_phot = bool(getattr(cls, 'photometryFlag', False))
        need_ast = bool(getattr(cls, 'astrometryFlag', False))
        if need_phot:
            assert callable(getattr(cls, 'jax_log_likely_photometry', None)), (
                f'{name} missing jax_log_likely_photometry'
            )
        if need_ast:
            assert callable(getattr(cls, 'jax_log_likely_astrometry', None)), (
                f'{name} missing jax_log_likely_astrometry'
            )
        assert need_phot or need_ast, f'{name} has neither phot nor ast flag'

    return None


@pytest.mark.parametrize(
    'sampler',
    ['evaluate', 'jaxlike', 'numpyro', 'blackjax'],
)
def test_missing_jax_lnl_raises(sampler):
    """JAX entry points raise NotImplementedError naming the class."""
    data, cube = _astrom_data()
    match = _MISSING_NAME

    if sampler == 'evaluate':
        fitter = MicrolensSolver(data, _MISSING_CLASS, verbose=False)
        with pytest.raises(NotImplementedError, match=match):
            fitter.evaluate_loglik_jax(cube)
        return None

    if sampler == 'jaxlike':
        fitter = MicrolensSolverJaxLike(data, _MISSING_CLASS, verbose=False)
        with pytest.raises(NotImplementedError, match=match):
            fitter.evaluate_loglik_jax(cube)
        return None

    if sampler == 'numpyro':
        fitter = MicrolensSolverNumPyro(data, _MISSING_CLASS, verbose=False)
        with pytest.raises(NotImplementedError, match=match):
            build_explicit_jax_loglik_fn(fitter)
        return None

    if sampler == 'blackjax':
        fitter = MicrolensSolverBlackJAX(data, _MISSING_CLASS, verbose=False)
        with pytest.raises(NotImplementedError, match=match):
            build_explicit_jax_loglik_fn(fitter)
        return None

    raise AssertionError(f'unknown sampler {sampler!r}')


def test_host_solver_still_works():
    """Host MicrolensSolver.log_likely still works without a JAX likelihood."""
    data, cube = _astrom_data()
    fitter = MicrolensSolver(data, _MISSING_CLASS, verbose=False)
    lnL = fitter.log_likely(cube)
    assert np.isfinite(lnL)

    return None


def test_add_err_raises_clear_error():
    """Fitted add_err / mult_err names raise a clear NotImplementedError."""
    data, cube = _make_pspl_phot_for_add_err()
    fitter = MicrolensSolver(
        data, mj.PSPL_Phot_noPar_Param1, verbose=False,
        add_error_on_photometry=True,
    )
    with pytest.raises(NotImplementedError, match='add_err'):
        build_explicit_jax_loglik_fn(fitter)

    return None


def _make_pspl_phot_for_add_err():
    """
    Photometry data for an add_err fitter probe.

    Returns
    -------
    data : dict
        Fitter data dictionary.
    cube : dict
        Unused placeholder for API symmetry with ``_astrom_data``.
    """
    t0 = 57000.0
    t = np.linspace(t0 - 40.0, t0 + 40.0, 20)
    mag = 18.0 - 0.3 * np.exp(-0.5 * ((t - t0) / 12.0) ** 2)
    data = {
        't_phot1': t,
        'mag1': mag,
        'mag_err1': np.full_like(t, 0.02),
    }

    return data, {}
