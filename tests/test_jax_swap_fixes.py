"""
Regression tests for jax_swap_plan Step 1 (wrong likelihood mappings).

Each test is expected to fail on ``5368a88`` before the corresponding fix
and pass afterward.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from bagle import frame_convert as fc
from bagle import jax_physics
from bagle import model as model_np
from bagle import model_jax as mj
from bagle.jax import gp as gp_mod
from bagle.jax.frame_convert import geo_phot_to_helio_jax
from bagle.model_fitter_jax import (
    MicrolensSolver,
    MicrolensSolverJaxLike,
    build_explicit_jax_loglik_fn,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1.1 Orbit primary/secondary sign
# ---------------------------------------------------------------------------

_ORBIT_CLASSES = [
    "PSBL_PhotAstrom_noPar_LinOrbs_Param1",
    "PSBL_PhotAstrom_Par_LinOrbs_Param1",
    "PSBL_PhotAstrom_noPar_AccOrbs_Param1",
    "PSBL_PhotAstrom_Par_AccOrbs_Param1",
    "FSBL_PhotAstrom_noPar_LinOrbs_Param1",
    "FSBL_PhotAstrom_Par_LinOrbs_Param1",
    "FSBL_PhotAstrom_noPar_AccOrbs_Param1",
    "FSBL_PhotAstrom_Par_AccOrbs_Param1",
    "FSBL_Limb_PhotAstrom_noPar_LinOrbs_Param1",
    "FSBL_Limb_PhotAstrom_Par_LinOrbs_Param1",
    "FSBL_Limb_PhotAstrom_noPar_AccOrbs_Param1",
    "FSBL_Limb_PhotAstrom_Par_AccOrbs_Param1",
]

_ORBIT_KW = dict(
    mLp=0.5, mLs=0.05, t0=57000.0, xS0_E=0.001, xS0_N=-0.002, beta=0.04,
    muL_E=1.0, muL_N=3.0, delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0,
    muS_E=-2.0, muS_N=-1.0, dL=4000.0, dS=8000.0, sep=0.75, alpha=70.0,
    b_sff=0.7, mag_src=19.0, dmag_Lp_Ls=1.5, raL=268.0, decL=-29.0,
)
_ORBIT_FILT = ("xS0_E", "xS0_N", "b_sff", "mag_src", "dmag_Lp_Ls")


def _orbit_setup(class_name):
    """
    Build host model, packed vector, and noise-free data for an orbit class.

    Parameters
    ----------
    class_name : str
        Concrete ``model_jax`` class name.

    Returns
    -------
    cls, vec, t, mag_h, pos_h, pvec, host
    """
    cls = getattr(mj, class_name)
    kw = dict(_ORBIT_KW)
    if "AccOrbs" in class_name:
        kw.update(accLsec_E=2.0, accLsec_N=1.0)
    if "noPar" in class_name:
        kw.pop("raL")
        kw.pop("decL")
    if "FSBL" in class_name:
        kw["radiusS"] = 0.001  # arcsec; small so host stays cheap
        if "Limb" in class_name:
            kw["gamma"] = 0.3

    filt = tuple(n for n in _ORBIT_FILT if n in cls.fitter_param_names)
    if "radiusS" in cls.fitter_param_names:
        filt = filt  # not a filt param
    host_kw = {k: ([v] if k in filt or k in ("gamma",) else v)
               for k, v in kw.items() if k in cls.__init__.__code__.co_varnames
               or k in ("raL", "decL")}
    # Simpler: pass list for known filt names.
    host_kw = {}
    for k, v in kw.items():
        if k in ("raL", "decL") and "noPar" in class_name:
            continue
        if k in filt or k == "gamma":
            host_kw[k] = [v]
        else:
            host_kw[k] = v

    host = cls(**host_kw)
    t = np.linspace(56900.0, 57100.0, 40)
    mag_h = np.asarray(host.get_photometry(t), dtype=np.float64)
    pos_h = np.asarray(host.get_astrometry(t), dtype=np.float64)
    vec = jnp.asarray([kw[n] for n in cls.fitter_param_names], dtype=jnp.float64)
    pvec = None
    if "Par" in class_name and "noPar" not in class_name:
        pvec = jax_physics.precompute_parallax_vectors(
            kw["raL"], kw["decL"], t, obs_location="earth"
        )

    return cls, vec, t, mag_h, pos_h, pvec, host


@pytest.mark.parametrize("class_name", _ORBIT_CLASSES)
def test_orbit_traced_matches_host(class_name):
    """W1: traced orbit lnL matches host at the true parameters."""
    cls, vec, t, mag_h, pos_h, pvec, host = _orbit_setup(class_name)
    err = np.full(t.size, 0.01)
    xerr = np.full(t.size, 1e-4)

    lnL_traced = float(cls.jax_log_likely_photometry(
        vec, jnp.asarray(t), jnp.asarray(mag_h), jnp.asarray(err),
        0.7, 19.0, parallax_vectors=pvec,
    ))
    # Host lnL via Gaussian on noise-free data.
    lnL_host = float(jax_physics.gaussian_log_likelihood_sum(
        jnp.asarray(mag_h), mag_h, err
    ))
    # Noise-free maximum.
    lnL_max = float(-0.5 * np.sum(np.log(2.0 * np.pi * err ** 2)))

    assert abs(lnL_traced - lnL_host) < 1e-5
    assert abs(lnL_traced - lnL_max) < 1e-5

    if hasattr(cls, "jax_log_likely_astrometry"):
        # Host astrometry lnL at the same noise-free positions.
        lnL_ast_host = float(jax_physics.gaussian_astrometry_log_likelihood_sum(
            jnp.asarray(pos_h), pos_h[:, 0], pos_h[:, 1], xerr, xerr
        ))
        ast_kwargs = dict(b_sff=0.7, mag_src=19.0, parallax_vectors=pvec)
        # Luminous-lens PSBL / FSBL need dmag for the centroid.
        import inspect
        if "dmag_Lp_Ls" in inspect.signature(cls.jax_log_likely_astrometry).parameters:
            ast_kwargs["dmag_Lp_Ls"] = 1.5
        lnL_ast = float(cls.jax_log_likely_astrometry(
            vec, jnp.asarray(t),
            jnp.asarray(pos_h[:, 0]), jnp.asarray(pos_h[:, 1]),
            jnp.asarray(xerr), jnp.asarray(xerr),
            **ast_kwargs,
        ))
        assert abs(lnL_ast - lnL_ast_host) < 1e-6


# ---------------------------------------------------------------------------
# 1.2 BFSPL phot-only finite source
# ---------------------------------------------------------------------------

_BFSPL_PHOT = [
    "BFSPL_Phot_noPar_Param1",
    "BFSPL_Phot_Par_Param1",
    "BFSPL_Limb_Phot_noPar_Param1",
    "BFSPL_Limb_Phot_Par_Param1",
]


@pytest.mark.parametrize("class_name", _BFSPL_PHOT)
def test_bfspl_phot_traced_matches_host(class_name):
    """W2: BFSPL phot-only traced lnL uses finite-source kernel."""
    cls = getattr(mj, class_name)
    # Large rho and small u0 so finite-source vs point-source differ a lot.
    kw = dict(
        t0=57000.0, u0_amp=0.02, tE=40.0, piE_E=0.05, piE_N=0.1,
        sep=0.5, phi=30.0, mag_src_pri=18.0, mag_src_sec=19.0, b_sff=1.0,
        radiusS_pri=0.05, radiusS_sec=0.05, raL=268.0, decL=-29.0,
    )
    if "Limb" in class_name:
        kw["gamma_pri"] = 0.4
        kw["gamma_sec"] = 0.3
    if "noPar" in class_name:
        kw.pop("raL")
        kw.pop("decL")

    filt = ("mag_src_pri", "mag_src_sec", "b_sff", "gamma_pri", "gamma_sec")
    host_kw = {
        k: ([v] if k in filt else v)
        for k, v in kw.items()
    }
    host = cls(**host_kw)
    t = np.linspace(56950.0, 57050.0, 40)
    mag_h = np.asarray(host.get_photometry(t), dtype=np.float64)
    err = np.full(t.size, 0.02)
    vec = jnp.asarray([kw[n] for n in cls.fitter_param_names])
    pvec = None
    if "Par" in class_name and "noPar" not in class_name:
        pvec = jax_physics.precompute_parallax_vectors(
            kw["raL"], kw["decL"], t, obs_location="earth"
        )

    mag_src = (kw["mag_src_pri"], kw["mag_src_sec"])
    lnL_traced = float(cls.jax_log_likely_photometry(
        vec, jnp.asarray(t), jnp.asarray(mag_h), jnp.asarray(err),
        1.0, mag_src, parallax_vectors=pvec, n_outline=50,
    ))
    lnL_host = float(jax_physics.gaussian_log_likelihood_sum(
        jnp.asarray(mag_h), mag_h, err
    ))
    assert abs(lnL_traced - lnL_host) < 1e-3

    # Point-source BSPL lnL at the same geometry should disagree strongly.
    bspl_cls = mj.BSPL_Phot_noPar_Param1 if "noPar" in class_name else mj.BSPL_Phot_Par_Param1
    bspl_names = bspl_cls.fitter_param_names
    bspl_vec = jnp.asarray([kw[n] for n in bspl_names])
    lnL_ps = float(bspl_cls.jax_log_likely_photometry(
        bspl_vec, jnp.asarray(t), jnp.asarray(mag_h), jnp.asarray(err),
        1.0, mag_src, parallax_vectors=pvec,
    ))
    assert abs(lnL_traced - lnL_ps) > 10.0


# ---------------------------------------------------------------------------
# 1.3 / 1.4 GP kernel
# ---------------------------------------------------------------------------

def test_gp_kernel_matches_celerite():
    """W3a: tinygp SHO amplitude matches celerite S0*omega0*Q variance."""
    celerite = pytest.importorskip("celerite")
    pytest.importorskip("tinygp")
    from tinygp import GaussianProcess

    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 100.0, 50))
    mag_err = np.full_like(t, 0.02)

    for S0, omega0, rho, sigma in [
        (1e-3, 0.5, 5.0, 0.02),
        (2e-4, 1.2, 10.0, 0.01),
        (5e-3, 0.2, 2.0, 0.03),
    ]:
        gp_params = {
            "gp_log_sigma": math.log(sigma),
            "gp_log_rho": math.log(rho),
            "gp_log_S0": math.log(S0),
            "gp_log_omega0": math.log(omega0),
        }
        kernel, jitter = gp_mod.build_gp_kernel_from_params(
            gp_params, mag_err, fixed_jitter=True
        )
        # Match host: mag_err plus JitterTerm(mean mag_err).
        diag = float(jitter) ** 2 + mag_err ** 2
        tgp = GaussianProcess(kernel, t, diag=diag)
        K_tiny = np.asarray(tgp.covariance)

        Q = gp_mod._GP_QUALITY
        terms = celerite.terms.Matern32Term(math.log(sigma), math.log(rho))
        terms += celerite.terms.SHOTerm(
            math.log(S0), math.log(Q), math.log(omega0)
        )
        terms += celerite.terms.JitterTerm(math.log(float(jitter)))
        cgp = celerite.GP(terms)
        cgp.compute(t, mag_err)
        K_cel = cgp.get_matrix(include_diagonal=True)

        # New SHO amplitude must match celerite much better than sigma=sqrt(S0).
        from tinygp.kernels import quasisep as qk

        kernel_old = qk.Matern32(scale=rho, sigma=sigma) + qk.SHO(
            omega=omega0, quality=Q, sigma=math.sqrt(S0)
        )
        K_old = np.asarray(
            GaussianProcess(kernel_old, t, diag=diag).covariance
        )
        err_new = np.max(np.abs(K_tiny - K_cel))
        err_old = np.max(np.abs(K_old - K_cel))
        assert err_new < 1e-6
        assert err_new < 0.01 * err_old


_GP_TRACED_CLASSES = [
    ("PSPL_Phot_noPar_GP_Param1", False),
    ("PSPL_Phot_Par_GP_Param1", True),
    ("PSBL_Phot_noPar_GP_Param1", False),
    ("PSBL_Phot_Par_GP_Param1", True),
    ("BSPL_Phot_noPar_GP_Param1", False),
    ("BSPL_Phot_Par_GP_Param1", True),
]


@pytest.mark.parametrize("class_name,use_par", _GP_TRACED_CLASSES)
def test_gp_traced_matches_host(class_name, use_par):
    """W3a: traced GP lnL matches host MicrolensSolver lnL."""
    pytest.importorskip("tinygp")
    cls = getattr(mj, class_name)

    t0 = 57000.0
    t = np.linspace(t0 - 60.0, t0 + 60.0, 40)
    # Noise-free-ish photometry from a simple PSPL-like curve.
    mag = 18.0 - 0.3 * np.exp(-0.5 * ((t - t0) / 15.0) ** 2)
    mag_err = np.full_like(t, 0.02)
    data = {
        "t_phot1": t, "mag1": mag, "mag_err1": mag_err,
    }
    if use_par:
        data["raL"] = 268.0
        data["decL"] = -29.0

    fitter = MicrolensSolver(data, cls, verbose=False)
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None

    # True-ish cube: use midpoints of default priors where needed.
    cube = {}
    for name in fitter.fitter_param_names:
        if name.startswith("t0"):
            cube[name] = t0
        elif name.startswith("u0"):
            cube[name] = 0.1
        elif name.startswith("tE"):
            cube[name] = 30.0
        elif "piE" in name:
            cube[name] = 0.05
        elif name.startswith("b_sff"):
            cube[name] = 1.0
        elif "mag_src" in name or name.startswith("mag_"):
            cube[name] = 18.0
        elif name.startswith("gp_log"):
            cube[name] = -2.0
        elif name.startswith("gp_rho") or name == "gp_rho1":
            cube[name] = 5.0
        elif name in ("sep", "sep1"):
            cube[name] = 0.5
        elif name in ("phi", "phi1", "q", "q1"):
            cube[name] = 0.3 if name.startswith("q") else 40.0
        else:
            cube[name] = 0.1

    vec = np.array([cube[n] for n in fitter.fitter_param_names], dtype=np.float64)
    lnL_traced = float(fn(vec))
    lnL_host = float(fitter.log_likely(cube))
    assert np.isfinite(lnL_traced) and np.isfinite(lnL_host)
    # Host celerite vs tinygp can differ at ~1e-4 for these short series.
    assert abs(lnL_traced - lnL_host) < 1e-3


@pytest.mark.parametrize(
    "class_name",
    ["PSPL_Phot_Par_GP_Param1_2", "PSPL_Phot_Par_GP_Param2_3"],
)
def test_gp_alt_s0_names_traced_matches_host(class_name):
    """W3b: gp_log_omega04_S0 / gp_log_omega0_S0 do not KeyError."""
    pytest.importorskip("tinygp")
    cls = getattr(mj, class_name)
    t0 = 57000.0
    t = np.linspace(t0 - 40.0, t0 + 40.0, 30)
    mag = 18.0 + 0.05 * np.sin((t - t0) / 10.0)
    data = {
        "t_phot1": t, "mag1": mag, "mag_err1": np.full_like(t, 0.02),
        "raL": 268.0, "decL": -29.0,
    }
    fitter = MicrolensSolver(data, cls, verbose=False)
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None

    cube = {}
    for name in fitter.fitter_param_names:
        if name.startswith("t0"):
            cube[name] = t0
        elif name.startswith("u0"):
            cube[name] = 0.1
        elif name.startswith("tE"):
            cube[name] = 25.0
        elif "piE" in name:
            cube[name] = 0.05
        elif name.startswith("b_sff"):
            cube[name] = 1.0
        elif "mag" in name:
            cube[name] = 18.0
        elif name.startswith("gp_log"):
            cube[name] = -2.0
        elif "gp_rho" in name:
            cube[name] = 5.0
        else:
            cube[name] = 0.1

    vec = np.array([cube[n] for n in fitter.fitter_param_names], dtype=np.float64)
    # Must not raise KeyError on alt S0 names.
    lnL_traced = float(fn(vec))
    lnL_host = float(fitter.log_likely(cube))
    assert np.isfinite(lnL_traced) and np.isfinite(lnL_host)
    assert abs(lnL_traced - lnL_host) < 1e-2


# ---------------------------------------------------------------------------
# 1.5 geoproj
# ---------------------------------------------------------------------------

def test_geoproj_jax_convert_matches_numpy():
    """W4 helper: JAX geo→helio conversion matches frame_convert."""
    ra, dec = 268.0, -29.0
    t0par = 56000.0
    for t0_g, u0_g, tE_g, piEE_g, piEN_g in [
        (56050.0, -0.2, 135.0, -0.058, 0.11),
        (56020.0, 0.05, 80.0, 0.02, -0.03),
        (55980.0, 0.3, 40.0, 0.1, 0.05),
    ]:
        x = np.array([t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par], dtype=np.float64)
        y_jax = np.asarray(geo_phot_to_helio_jax(x, ra, dec))
        y_np = np.array(fc.convert_helio_geo_phot(
            ra, dec, t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par,
            in_frame="geo", murel_in="LS", murel_out="SL",
            coord_in="tb", coord_out="EN", plot=False,
        ), dtype=np.float64)
        np.testing.assert_allclose(y_jax, y_np, rtol=1e-10, atol=1e-12)


def test_geoproj_traced_matches_host():
    """W4: PSPL_Phot_Par_Param1_geoproj traced lnL matches host photometry."""
    ra, dec = 268.0, -29.0
    t0par = 56000.0
    t0_g, u0_g, tE_g = 56050.0, -0.2, 100.0
    piEE_g, piEN_g = -0.05, 0.08
    b_sff, mag_src = 1.0, 18.0

    # Default geotr priors are deferred; call the classmethod directly.
    host = mj.PSPL_Phot_Par_Param1_geoproj(
        t0_g, u0_g, tE_g, piEE_g, piEN_g, [b_sff], [mag_src],
        t0par, raL=ra, decL=dec,
    )
    t = np.linspace(t0_g - 2 * tE_g, t0_g + 2 * tE_g, 40)
    mag = np.asarray(host.get_photometry(t), dtype=np.float64)
    err = np.full_like(mag, 0.02)
    pvec = jax_physics.precompute_parallax_vectors(
        ra, dec, t, obs_location="earth"
    )
    cls = mj.PSPL_PhotParam1_geoproj
    vec = jnp.asarray([
        t0_g, u0_g, tE_g, piEE_g, piEN_g, b_sff, mag_src
    ], dtype=jnp.float64)
    lnL_traced = float(cls.jax_log_likely_photometry(
        vec, jnp.asarray(t), jnp.asarray(mag), jnp.asarray(err),
        b_sff, mag_src, parallax_vectors=pvec,
        raL=ra, decL=dec, t0par=t0par,
    ))
    lnL_host = float(jax_physics.gaussian_log_likelihood_sum(
        jnp.asarray(mag), mag, err
    ))
    assert abs(lnL_traced - lnL_host) < 1e-6


# ---------------------------------------------------------------------------
# 1.6 catch-all except
# ---------------------------------------------------------------------------

def test_jaxlike_raises_on_bad_key(monkeypatch):
    """Bare KeyError in evaluate_loglik_jax must propagate, not return floor."""
    t0 = 57000.0
    t = np.linspace(t0 - 30.0, t0 + 30.0, 20)
    mag = 18.0 - 0.2 * np.exp(-0.5 * ((t - t0) / 10.0) ** 2)
    data = {
        "t_phot1": t,
        "mag1": mag,
        "mag_err1": np.full_like(t, 0.05),
    }
    fitter = MicrolensSolverJaxLike(
        data, mj.PSPL_Phot_noPar_Param1, verbose=False,
    )
    n = len(fitter.fitter_param_names)
    assert n > 0

    def boom(cube):
        raise KeyError("forced")

    monkeypatch.setattr(fitter, "evaluate_loglik_jax", boom)
    with pytest.raises(KeyError, match="forced"):
        fitter.LogLikelihood(np.full(n, 0.1))


# ---------------------------------------------------------------------------
# 1.7 / 1.8 hygiene
# ---------------------------------------------------------------------------

def test_model_jax_imports_without_tkinter():
    """model_jax must import when tkinter is unavailable."""
    code = (
        "import sys\n"
        "sys.modules['tkinter'] = None\n"
        "sys.path.insert(0, %r)\n"
        "import bagle.model_jax as m\n"
        "assert m is not None\n"
    ) % str(REPO / "src")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()
                       if k != "PYTHONPATH"}),
             "PYTHONPATH": str(REPO / "src")},
    )
    assert proc.returncode == 0, proc.stderr


def test_no_jax_model_imports():
    """Dead bagle.jax_model imports must be gone from model_jax.py."""
    text = (REPO / "src" / "bagle" / "model_jax.py").read_text()
    assert "bagle.jax_model" not in text
