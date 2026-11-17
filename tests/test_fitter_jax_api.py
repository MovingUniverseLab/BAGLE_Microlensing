"""API parity for jax_swap_plan Step 6.

Every public function and every public ``MicrolensSolver`` method in
``model_fitter.py`` must exist on ``model_fitter_jax`` with a
signature a ``model_fitter`` call can still bind.

Prior generators are deferred. Their JAX signatures take ``param_name``
and ``stats_pkg`` and are not compared here:

``make_gen``, ``make_norm_gen``, ``make_lognorm_gen``,
``make_log10norm_gen``, ``make_truncnorm_gen``,
``make_truncnorm_gen_with_bounds``, ``make_invgamma_gen``,
``make_t0_gen``, ``make_mag_base_gen``, ``make_mag_src_gen``,
``make_xS0_gen``, ``make_xS0_norm_gen``, ``make_muS_EN_gen``,
``make_muS_EN_norm_gen``, ``make_piS``, ``weighted_quantile``,
and ``MicrolensSolver.make_default_priors``.
"""
from __future__ import annotations

import inspect
import io
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

pytest.importorskip("jax")

import bagle.model_fitter as fitter_np
import bagle.model_fitter_jax as fitter_jax


# Prior builders stay on the deferred signatures. Do not "fix" them
# to match model_fitter.py in this step.
_DEFERRED_FUNCS = frozenset({
    "make_gen",
    "make_norm_gen",
    "make_lognorm_gen",
    "make_log10norm_gen",
    "make_truncnorm_gen",
    "make_truncnorm_gen_with_bounds",
    "make_invgamma_gen",
    "make_t0_gen",
    "make_mag_base_gen",
    "make_mag_src_gen",
    "make_xS0_gen",
    "make_xS0_norm_gen",
    "make_muS_EN_gen",
    "make_muS_EN_norm_gen",
    "make_piS",
    "weighted_quantile",
})

_DEFERRED_METHODS = frozenset({"make_default_priors"})


class _StopAfterBest(Exception):
    """Raised once ``def_best`` has been recorded."""


def _public_functions(mod):
    """Public functions defined on ``mod``.

    Parameters
    ----------
    mod : module

    Returns
    -------
    funcs : dict
        Name to function. Imported callables are excluded.
    """
    funcs = {}
    for name, obj in vars(mod).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
            funcs[name] = obj

    return funcs


def _public_methods(cls):
    """Public callable attributes of ``cls``.

    Parameters
    ----------
    cls : type

    Returns
    -------
    methods : dict
    """
    methods = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        obj = getattr(cls, name)
        if callable(obj):
            methods[name] = obj

    return methods


def _call_still_binds(reference, candidate):
    """True when a positional/keyword call of ``reference`` binds.

    Shared parameters keep order, kind, and default. Extra parameters
    on ``candidate`` must be optional.

    Parameters
    ----------
    reference : callable
    candidate : callable

    Returns
    -------
    ok : bool
    """
    ref_params = list(inspect.signature(reference).parameters.values())
    new_params = list(inspect.signature(candidate).parameters.values())
    if len(new_params) < len(ref_params):
        return False

    for ref_par, new_par in zip(ref_params, new_params):
        if ref_par.name != new_par.name or ref_par.kind != new_par.kind:
            return False
        if ref_par.default != new_par.default:
            return False

    for extra in new_params[len(ref_params):]:
        if extra.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if extra.default is inspect.Parameter.empty:
            return False

    return True


def test_fitter_public_api_matches():
    """Public fitter names from model_fitter exist on model_fitter_jax."""
    missing_funcs = sorted(
        set(_public_functions(fitter_np)) - set(_public_functions(fitter_jax)))
    assert missing_funcs == []

    missing_methods = sorted(
        set(_public_methods(fitter_np.MicrolensSolver))
        - set(_public_methods(fitter_jax.MicrolensSolver)))
    assert missing_methods == []

    # Deferred prior generators are present but not signature-checked.
    bad_funcs = []
    for name, ref in sorted(_public_functions(fitter_np).items()):
        if name in _DEFERRED_FUNCS:
            continue
        cand = _public_functions(fitter_jax)[name]
        if not _call_still_binds(ref, cand):
            bad_funcs.append(name)
    assert bad_funcs == []

    bad_methods = []
    ref_methods = _public_methods(fitter_np.MicrolensSolver)
    jax_methods = _public_methods(fitter_jax.MicrolensSolver)
    for name, ref in sorted(ref_methods.items()):
        if name in _DEFERRED_METHODS:
            continue
        try:
            inspect.signature(ref)
            inspect.signature(jax_methods[name])
        except (ValueError, TypeError):
            continue
        if not _call_still_binds(ref, jax_methods[name]):
            bad_methods.append(name)
    assert bad_methods == []

    return None


def _solver_recording_best():
    """Solver instance whose ``get_best_fit`` records ``def_best``.

    Returns
    -------
    solver : MicrolensSolver
    seen : dict
    """
    seen = {}
    solver = fitter_jax.MicrolensSolver.__new__(fitter_jax.MicrolensSolver)

    def get_best_fit(def_best="maxl"):
        seen["def_best"] = def_best
        return {"t0": 57000.0}

    def to_param_dict(params):
        raise _StopAfterBest()

    solver.get_best_fit = get_best_fit
    solver.to_param_dict = to_param_dict

    return solver, seen


@pytest.mark.parametrize("method_name", [
    "calc_chi2", "calc_chi2_manual", "get_residual",
])
def test_def_best_is_forwarded(method_name):
    """``params='best'`` asks ``get_best_fit`` for the named definition."""
    solver, seen = _solver_recording_best()
    method = getattr(solver, method_name)

    with pytest.raises(_StopAfterBest):
        method(def_best="map")

    assert seen["def_best"] == "map"

    return None


def test_def_best_default_is_maxl():
    """Omitting ``def_best`` selects the maximum-likelihood solution."""
    solver, seen = _solver_recording_best()
    solver.log_likely = lambda params, verbose: -1.0
    solver.calc_chi2 = lambda *args, **kwargs: 2.0

    solver.print_likelihood()
    assert seen["def_best"] == "maxl"

    # The instance calc_chi2 above is a stub for print_likelihood.
    seen.clear()
    with pytest.raises(_StopAfterBest):
        fitter_jax.MicrolensSolver.calc_chi2(solver)
    assert seen["def_best"] == "maxl"

    return None


def test_priors_pickle_round_trip(tmp_path):
    """``write_priors_pkl`` and ``load_priors`` restore the same dict."""
    solver = fitter_jax.MicrolensSolver.__new__(fitter_jax.MicrolensSolver)
    solver.outputfiles_basename = str(tmp_path / "run-")
    solver.priors = {"t0": scipy.stats.norm(loc=57000.0, scale=10.0)}

    solver.write_priors_pkl()
    assert Path(solver.outputfiles_basename + "priors.pkl").is_file()

    solver.priors = None
    solver.load_priors()

    assert set(solver.priors) == {"t0"}
    assert solver.priors["t0"].mean() == pytest.approx(57000.0)
    assert solver.priors["t0"].std() == pytest.approx(10.0)

    return None


def test_make_fdfdt_and_random_prob_match():
    """The two small prior helpers match ``model_fitter``."""
    jax_dist = fitter_jax.make_fdfdt()
    np_dist = fitter_np.make_fdfdt()
    assert jax_dist.mean() == pytest.approx(np_dist.mean())
    assert jax_dist.std() == pytest.approx(np_dist.std())

    value_j, ln_j = fitter_jax.random_prob(jax_dist, 0.3)
    value_n, ln_n = fitter_np.random_prob(np_dist, 0.3)
    assert value_j == pytest.approx(value_n)
    assert ln_j == pytest.approx(ln_n)

    return None


def test_make_sed_distance_gen_matches(tmp_path):
    """The SED distance prior matches the NumPy fitter on the same CSV."""
    csv_path = tmp_path / "sed.csv"
    csv_path.write_text(
        "D_lo_kpc,D_hi_kpc,P_SEDdust_XP\n"
        "1.0,2.0,0.2\n"
        "2.0,3.0,0.5\n"
        "3.0,4.0,0.3\n")

    for param, unit in (("dS", "pc"), ("dS", "kpc"), ("piS", "kpc")):
        dist_j = fitter_jax.make_sed_distance_gen(
            str(csv_path), param=param, distance_unit=unit)
        dist_n = fitter_np.make_sed_distance_gen(
            str(csv_path), param=param, distance_unit=unit)
        grid = np.linspace(dist_n.ppf(0.1), dist_n.ppf(0.9), 11)
        assert dist_j.pdf(grid) == pytest.approx(dist_n.pdf(grid))

    return None


def test_save_params_matches_numpy():
    """``save_params`` writes the same text as ``model_fitter``."""
    class _Model:
        fitter_param_names = ["t0", "b_sff", "xS0_E"]
        filt_param_names = ["b_sff", "xS0_E"]
        phot_param_names = ["mag_src"]
        additional_param_names = ["thetaE_amp"]
        t0 = 57000.25
        b_sff = np.array([0.5, 0.8])
        xS0_E = np.array([0.123456, -0.2])
        mag_src = np.array([18.0])
        thetaE_amp = 1.25

    buf_j = io.StringIO()
    buf_n = io.StringIO()
    fitter_jax.save_params(_Model(), buf_j)
    fitter_np.save_params(_Model(), buf_n)
    assert buf_j.getvalue() == buf_n.getvalue()

    return None
