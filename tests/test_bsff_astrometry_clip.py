"""Host/JAX agreement when b_sff > 1, and the PhotAstrom prior guard."""

import os
from pathlib import Path

import numpy as np
import pytest

from bagle import jax_physics
from bagle import model_jax as model
from bagle.b_sff_prior import check_b_sff_astrom_priors, prior_upper_bound
from bagle.jax.geometry import derive_psbl_photastrom_param1
from bagle.model_fitter_jax import (
    MicrolensSolver,
    build_explicit_jax_loglik_fn,
    make_gen,
    make_norm_gen,
)
from bagle.model_jax import flux2mag, mag2flux


def _psbl_model(b_sff):
    """Static PSBL PhotAstrom model at one source flux fraction.

    Parameters
    ----------
    b_sff : float
        Source flux fraction for the single photometric dataset.

    Returns
    -------
    mod : PSBL_PhotAstrom_noPar_Param1
        Model with a luminous secondary (``dmag_Lp_Ls = 5``).
    """
    mod = model.PSBL_PhotAstrom_noPar_Param1(
        10.0, 5.0, 57000.0, 0.0, 0.0, 2.0,
        0.0, 0.0, 3.0, 0.0,
        3000.0, 8000.0, 10.0, 90.0,
        [b_sff], [14.0], [5.0],
        root_tol=1e-8,
    )
    return mod


def _jax_psbl_astrometry(mod, t, b_sff):
    """JAX PSBL centroid using the same geometry as the JAX log-likelihood.

    Parameters
    ----------
    mod : PSBL_PhotAstrom_noPar_Param1
        Host model. Masses on the model are not passed through; the JAX
        kernel wants mass fractions from ``derive_psbl_photastrom_param1``.
    t : ndarray, shape (N_times,)
        Observation times (MJD).
    b_sff : float
        Source flux fraction.

    Returns
    -------
    pos : ndarray, shape (N_times, 2)
        East / North centroid in arcsec.
    """
    geom = derive_psbl_photastrom_param1(
        mod.mLp, mod.mLs, mod.t0, mod.xS0[0], mod.xS0[1],
        mod.beta, mod.muL[0], mod.muL[1], mod.muS[0], mod.muS[1],
        mod.dL, mod.dS, mod.sep, mod.alpha,
    )
    (
        _u0, _thetaE_hat, _tE, _piE_E, _piE_N, xS0, xL0, muS, muL,
        thetaE_amp, _piS, _piL, m1, m2, xL1, xL2, _mLp, _mLs,
    ) = geom
    pos = jax_physics.psbl_astrometry_param1(
        t, mod.t0, xS0, xL0, muS, muL, thetaE_amp,
        xL1, xL2, m1, m2, float(mod.mag_src[0]), b_sff,
        dmag_Lp_Ls=float(mod.dmag_Lp_Ls[0]),
    )
    return np.asarray(pos)


def _unclipped_photometry(mod, t, b_sff):
    """Unresolved magnitude with the raw (possibly negative) blend.

    Parameters
    ----------
    mod : PSBL
        Model whose image amplifications are reused.
    t : ndarray, shape (N_times,)
        Observation times (MJD).
    b_sff : float
        Source flux fraction. Values above 1 keep a negative blend.

    Returns
    -------
    mag : ndarray, shape (N_times,)
        Unresolved magnitude. Not clipped.
    """
    _img, amp = mod.get_all_arrays(t)
    amp = np.asarray(amp)
    amp_sum = np.sum(np.where(np.isfinite(amp), amp, 0.0), axis=1)

    # Same blend term get_photometry adds, including the negative case.
    f_src = float(np.asarray(mag2flux(mod.mag_src[0])))
    flux = f_src * amp_sum + f_src * (1.0 - b_sff) / b_sff
    mag = np.asarray(flux2mag(flux), dtype=np.float64)
    return mag


def test_host_jax_psbl_astrometry_and_lnl_agree():
    """Host and JAX centroids and lnL match at b_sff 0.8, 1.0, and 1.2."""
    t_ast = np.linspace(56920.0, 57080.0, 8)
    positions = {}

    for b_sff in (0.8, 1.0, 1.2):
        mod = _psbl_model(b_sff)
        host = np.asarray(mod.get_astrometry(t_ast), dtype=np.float64)
        jax_pos = _jax_psbl_astrometry(mod, t_ast, b_sff)
        np.testing.assert_allclose(host, jax_pos, rtol=1e-5, atol=1e-8)
        positions[b_sff] = host

    # b_sff = 1 and b_sff = 1.2 are both a dark lens after the clip.
    np.testing.assert_allclose(
        positions[1.0], positions[1.2], rtol=1e-5, atol=1e-8
    )

    # A luminous lens (b_sff = 0.8) must move the centroid.
    assert not np.allclose(
        positions[0.8], positions[1.0], rtol=1e-5, atol=1e-8
    )

    # Log-likelihood at each blend, including the negative-lens case.
    t_phot = np.linspace(56920.0, 57080.0, 10)
    outdir = "/tmp/bagle_bsff_clip/"
    os.makedirs(outdir, exist_ok=True)

    for b_sff in (0.8, 1.0, 1.2):
        mod = _psbl_model(b_sff)
        mag = np.asarray(mod.get_photometry(t_phot), dtype=np.float64)
        pos = np.asarray(mod.get_astrometry(t_ast), dtype=np.float64)
        data = {
            "t_phot1": t_phot,
            "mag1": mag,
            "mag_err1": np.full_like(t_phot, 0.02),
            "t_ast1": t_ast,
            "xpos1": pos[:, 0],
            "ypos1": pos[:, 1],
            "xpos_err1": np.full(t_ast.shape, 0.001),
            "ypos_err1": np.full(t_ast.shape, 0.001),
            "phot_data": ["I"],
            "ast_data": ["I"],
        }
        fitter = MicrolensSolver(
            data,
            model.PSBL_PhotAstrom_noPar_Param1,
            outputfiles_basename=outdir + f"b{b_sff}_",
            n_live_points=20,
            max_iter=1,
            dump_callback=None,
            verbose=False,
        )
        # Sampling is not run here. The prior only has to be legal.
        fitter.priors["b_sff1"] = make_gen("b_sff1", 0.0, 1.0)

        cube = {
            "mLp": 10.0, "mLs": 5.0, "t0": 57000.0,
            "xS0_E": 0.0, "xS0_N": 0.0, "beta": 2.0,
            "muL_E": 0.0, "muL_N": 0.0, "muS_E": 3.0, "muS_N": 0.0,
            "dL": 3000.0, "dS": 8000.0, "sep": 10.0, "alpha": 90.0,
            "b_sff1": b_sff, "mag_src1": 14.0, "dmag_Lp_Ls1": 5.0,
        }
        missing = [n for n in fitter.fitter_param_names if n not in cube]
        assert missing == []

        fn, _ctx = build_explicit_jax_loglik_fn(fitter)
        assert fn is not None

        lnL_host = float(fitter.log_likely(cube))
        lnL_jax = float(fitter.evaluate_loglik_jax(cube))
        np.testing.assert_allclose(lnL_host, lnL_jax, rtol=1e-6, atol=1e-4)

    return None


def test_photometry_keeps_unclipped_blend():
    """Photometry still adds the raw blend, including b_sff = 1.2."""
    t = np.linspace(56920.0, 57080.0, 10)
    mags = {}

    for b_sff in (0.8, 1.0, 1.2):
        mod = _psbl_model(b_sff)
        got = np.asarray(mod.get_photometry(t), dtype=np.float64)
        expected = _unclipped_photometry(mod, t, b_sff)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

        # JAX photometry kernel must stay on the unclipped blend too.
        geom = derive_psbl_photastrom_param1(
            mod.mLp, mod.mLs, mod.t0, mod.xS0[0], mod.xS0[1],
            mod.beta, mod.muL[0], mod.muL[1], mod.muS[0], mod.muS[1],
            mod.dL, mod.dS, mod.sep, mod.alpha,
        )
        (
            u0, thetaE_hat, tE, piE_E, piE_N, _xS0, _xL0, _muS, _muL,
            _thetaE, _piS, _piL, m1, m2, xL1, xL2, _mLp, _mLs,
        ) = geom
        jax_mag = jax_physics.psbl_photometry(
            t, mod.t0, tE, u0, thetaE_hat, xL1, xL2, m1, m2,
            float(mod.mag_src[0]), b_sff=b_sff,
            piE_E=piE_E, piE_N=piE_N,
        )
        np.testing.assert_allclose(
            got, np.asarray(jax_mag), rtol=1e-5, atol=1e-5
        )
        mags[b_sff] = got

    # Negative blend is still in the photometry: 1.2 is not the same as 1.
    blend = (1.0 - 1.2) / 1.2
    assert blend < 0.0
    assert not np.allclose(mags[1.0], mags[1.2], rtol=1e-6, atol=1e-6)
    return None


def _varying_series(n, scale):
    """A non-flat series so data-driven priors have a finite window.

    Parameters
    ----------
    n : int
        Number of samples.
    scale : float
        Amplitude of the ramp.

    Returns
    -------
    values : ndarray, shape (n,)
        Linear ramp.
    """
    values = np.linspace(-scale, scale, n)
    return values


def _solver_data(with_ast):
    """Tiny photometry, optionally with a matching astrometry dataset.

    Parameters
    ----------
    with_ast : bool
        When True, include one astrometry series paired with the photometry.

    Returns
    -------
    data : dict
        Fitter data dictionary.
    """
    t = np.linspace(57000.0, 57100.0, 20)
    mag = 18.0 + _varying_series(t.size, 0.4)
    data = {
        "t_phot1": t,
        "mag1": mag,
        "mag_err1": np.full(t.shape, 0.02),
        "phot_data": ["I"],
        "phot_files": ["I.dat"],
        "ast_data": [],
        "ast_files": [],
        "target": "test",
    }
    if with_ast:
        t_ast = t[::2]
        data.update({
            "t_ast1": t_ast,
            "xpos1": _varying_series(t_ast.size, 0.01),
            "ypos1": _varying_series(t_ast.size, 0.02),
            "xpos_err1": np.full(t_ast.shape, 0.001),
            "ypos_err1": np.full(t_ast.shape, 0.001),
            "ast_data": ["I"],
            "ast_files": ["I.ast"],
        })
    return data


def _make_solver(tmp_path, with_ast):
    """Build a solver. Combined data uses PSPL PhotAstrom; else phot only.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for MultiNest output files.
    with_ast : bool
        Include astrometry when True.

    Returns
    -------
    fitter : MicrolensSolver
        Solver with default priors installed.
    """
    if with_ast:
        model_class = model.PSPL_PhotAstrom_noPar_Param1
    else:
        model_class = model.PSPL_Phot_noPar_Param1

    fitter = MicrolensSolver(
        _solver_data(with_ast),
        model_class,
        outputfiles_basename=str(tmp_path / "chain_"),
        n_live_points=20,
        max_iter=1,
        dump_callback=None,
        verbose=False,
    )
    return fitter


def test_default_bsff_prior_follows_astrometry(tmp_path):
    """Default b_sff is 1.0 with astrometry and 1.5 for photometry only."""
    (tmp_path / "combined").mkdir()
    (tmp_path / "phot").mkdir()
    combined = _make_solver(tmp_path / "combined", with_ast=True)
    phot = _make_solver(tmp_path / "phot", with_ast=False)

    # A default-constructed combined solver passes without an override.
    assert prior_upper_bound(combined.priors["b_sff1"]) == pytest.approx(1.0)
    assert combined.check_b_sff_astrom_priors() is None

    # Photometry-only keeps the historical template upper bound.
    assert prior_upper_bound(phot.priors["b_sff1"]) == pytest.approx(1.5)
    assert phot.check_b_sff_astrom_priors() is None
    return None


def test_bsff_prior_check_cached_on_repeated_prior(tmp_path, monkeypatch):
    """Repeated Prior calls skip the check until a prior is replaced."""
    import bagle.b_sff_prior as b_sff_prior

    fitter = _make_solver(tmp_path, with_ast=True)
    calls = {"n": 0}
    real = b_sff_prior._validate_b_sff_astrom_priors

    def _counting(solver):
        calls["n"] += 1
        return real(solver)

    monkeypatch.setattr(
        b_sff_prior, "_validate_b_sff_astrom_priors", _counting
    )

    cube = np.full(fitter.n_dims, 0.5)
    fitter.Prior(cube.copy())
    assert calls["n"] == 1

    # Same prior objects: the support read is not repeated.
    fitter.Prior(cube.copy())
    fitter.Prior_copy(cube.copy())
    assert calls["n"] == 1

    # Replacing the prior misses the cache, re-runs, and raises.
    fitter.priors["b_sff1"] = make_gen("b_sff1", 0.0, 1.5)
    with pytest.raises(ValueError, match="b_sff1"):
        fitter.Prior(cube.copy())
    assert calls["n"] == 2
    return None


def test_bsff_prior_upper_bound_on_combined_datasets(tmp_path, monkeypatch):
    """Combined datasets reject user b_sff priors that extend above 1."""
    fitter = _make_solver(tmp_path, with_ast=True)

    # Generated default is already at 1, so solve is not rejected for it.
    assert prior_upper_bound(fitter.priors["b_sff1"]) == pytest.approx(1.0)

    # A user-supplied wider prior is still rejected.
    fitter.priors["b_sff1"] = make_gen("b_sff1", 0.0, 1.5)
    with pytest.raises(ValueError, match="b_sff1") as exc:
        fitter.solve()
    message = str(exc.value)
    assert "1.5" in message
    assert "photometry dataset 1" in message

    # A plain Gaussian is unbounded and must say so.
    fitter.priors["b_sff1"] = make_norm_gen("b_sff1", 0.8, 0.05)
    with pytest.raises(ValueError, match="truncat") as exc:
        fitter.check_b_sff_astrom_priors()
    assert "b_sff1" in str(exc.value)
    assert "unbounded" in str(exc.value)

    # Upper bound of exactly 1 is allowed, and solve() proceeds.
    fitter.priors["b_sff1"] = make_gen("b_sff1", 0.2, 1.0)
    assert fitter.check_b_sff_astrom_priors() is None

    ran = {}

    def _fake_run(*_args, **_kwargs):
        ran["yes"] = True
        return None

    monkeypatch.setattr(
        "bagle.model_fitter_jax.pymultinest.run", _fake_run
    )
    monkeypatch.setattr(
        fitter, "load_mnest_results", lambda remake_fits=True: None
    )
    monkeypatch.setattr(
        fitter, "load_mnest_summary", lambda remake_fits=True: None
    )
    fitter.solve()
    assert ran.get("yes") is True

    # Below 1 is allowed as well.
    fitter.priors["b_sff1"] = make_gen("b_sff1", 0.0, 0.9)
    assert fitter.check_b_sff_astrom_priors() is None
    return None


def test_bsff_prior_ignores_photometry_only(tmp_path, monkeypatch):
    """Photometry-only datasets may keep a b_sff prior above 1."""
    fitter = _make_solver(tmp_path, with_ast=False)
    assert prior_upper_bound(fitter.priors["b_sff1"]) == pytest.approx(1.5)
    fitter.priors["b_sff1"] = make_gen("b_sff1", 0.0, 1.5)
    assert fitter.check_b_sff_astrom_priors() is None

    ran = {}

    def _fake_run(*_args, **_kwargs):
        ran["yes"] = True
        return None

    monkeypatch.setattr(
        "bagle.model_fitter_jax.pymultinest.run", _fake_run
    )
    monkeypatch.setattr(
        fitter, "load_mnest_results", lambda remake_fits=True: None
    )
    monkeypatch.setattr(
        fitter, "load_mnest_summary", lambda remake_fits=True: None
    )
    fitter.solve()
    assert ran.get("yes") is True
    return None


def test_prior_upper_bound_readers():
    """scipy, NumPyro, and PyMC supports, plus an unknown custom prior."""
    import scipy.stats

    uniform = scipy.stats.uniform(loc=0.0, scale=1.5)
    gauss = scipy.stats.norm(loc=0.5, scale=0.2)
    trunc = scipy.stats.truncnorm(-1.0, 2.0, loc=0.5, scale=0.2)
    assert prior_upper_bound(uniform) == pytest.approx(1.5)
    assert np.isinf(prior_upper_bound(gauss))
    assert prior_upper_bound(trunc) == pytest.approx(0.9)

    numpyro = pytest.importorskip("numpyro")
    import numpyro.distributions as dist

    assert prior_upper_bound(dist.Uniform(0.0, 1.0)) == pytest.approx(1.0)
    assert np.isinf(prior_upper_bound(dist.Normal(0.5, 0.2)))
    truncated = dist.TruncatedNormal(0.5, 0.2, low=0.0, high=1.0)
    assert prior_upper_bound(truncated) == pytest.approx(1.0)

    pymc = pytest.importorskip("pymc")
    with pymc.Model():
        uni = pymc.Uniform("b", lower=0.0, upper=1.2)
        nor = pymc.Normal("n", mu=0.5, sigma=0.2)
        trn = pymc.TruncatedNormal(
            "t", mu=0.5, sigma=0.2, lower=0.0, upper=1.0
        )
        assert prior_upper_bound(uni) == pytest.approx(1.2)
        assert np.isinf(prior_upper_bound(nor))
        assert prior_upper_bound(trn) == pytest.approx(1.0)

    class _Custom:
        def ppf(self, unit):
            return unit

    assert np.isinf(prior_upper_bound(_Custom()))

    # Only the astrometry partner is checked. A long string-style map
    # must not inspect photometry datasets past n_ast_sets.
    class _Fitter:
        n_phot_sets = 2
        n_ast_sets = 1
        map_phot_idx_to_ast_idx = [0, 1, 2]
        priors = {
            "b_sff1": scipy.stats.uniform(loc=0.0, scale=1.0),
            "b_sff2": scipy.stats.uniform(loc=0.0, scale=2.0),
        }

    assert check_b_sff_astrom_priors(_Fitter()) is None

    _Fitter.priors["b_sff1"] = scipy.stats.uniform(loc=0.0, scale=1.4)
    with pytest.raises(ValueError, match="b_sff1"):
        check_b_sff_astrom_priors(_Fitter())

    # Astrometry paired with the second photometry dataset.
    class _Fitter2:
        n_phot_sets = 2
        n_ast_sets = 1
        map_phot_idx_to_ast_idx = [1]
        priors = {
            "b_sff1": scipy.stats.uniform(loc=0.0, scale=3.0),
            "b_sff2": scipy.stats.uniform(loc=0.0, scale=1.0),
        }

    assert check_b_sff_astrom_priors(_Fitter2()) is None
    return None


# Injected values wide enough for both PhotAstrom Param1 and phot-only.
_COMPARISON_TRUTH = {
    "mL": 0.5,
    "t0": 57050.0,
    "beta": 0.2,
    "dL": 4000.0,
    "dL_dS": 0.5,
    "xS0_E": 0.0,
    "xS0_N": 0.0,
    "muL_E": 0.0,
    "muL_N": 0.0,
    "muS_E": 0.0,
    "muS_N": 0.0,
    "b_sff": 1.0,
    "mag_src": 18.0,
    "u0_amp": 0.1,
    "tE": 30.0,
    "piE_E": 0.1,
    "piE_N": 0.1,
}


def _load_run_comparison():
    """Import the sampler-comparison script (it is not a package).

    Returns
    -------
    module : module
        ``run_comparison`` loaded from its file path.
    """
    import importlib.util

    path = (
        Path(__file__).resolve().parent
        / "psbl_sampler_compare"
        / "run_comparison.py"
    )
    spec = importlib.util.spec_from_file_location(
        "psbl_run_comparison", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_comparison_caps_bsff_with_astrometry(tmp_path):
    """Narrow and open comparison priors clip b_sff only with astrometry."""
    comp = _load_run_comparison()
    (tmp_path / "ast").mkdir()
    (tmp_path / "phot").mkdir()
    combined = _make_solver(tmp_path / "ast", with_ast=True)
    phot = _make_solver(tmp_path / "phot", with_ast=False)

    # Truth sits on 1, so truth + half-width would exceed the cap.
    comp.apply_narrow_priors(combined, _COMPARISON_TRUTH)
    lo, hi = combined.priors["b_sff1"].support()
    assert hi == pytest.approx(1.0)
    assert lo < hi
    assert lo == pytest.approx(0.95)

    comp.apply_narrow_priors(phot, _COMPARISON_TRUTH)
    _lo, hi = phot.priors["b_sff1"].support()
    assert hi == pytest.approx(1.05)

    comp.apply_open_priors(combined, _COMPARISON_TRUTH)
    lo, hi = combined.priors["b_sff1"].support()
    assert hi == pytest.approx(1.0)
    assert lo < hi

    comp.apply_open_priors(phot, _COMPARISON_TRUTH)
    _lo, hi = phot.priors["b_sff1"].support()
    # Open half-width is 0.35, still under the phot-only 1.5 cap.
    assert hi == pytest.approx(1.35)

    # A zero-width window on the cap stays a non-empty interval.
    lo, hi = comp._clip_bsff_edges(combined, "b_sff1", 1.0, 1.0)
    assert hi == pytest.approx(1.0)
    assert lo < hi
    return None
