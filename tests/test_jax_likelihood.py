"""Tests for JAX photometry log-likelihood and PyMC gradients."""
import numpy as np
import pytest

from bagle import jax_physics, model_jax as model
from bagle.model_fitter_jax import (
    LogLikelihoodOp,
    MicrolensSolver,
    build_explicit_jax_loglik_fn,
)


def _make_pspl_phot_data():
    t0 = 57000.0
    u0_amp = 0.05
    tE = 45.0
    piE_E = 0.01
    piE_N = 0.02
    b_sff = 0.8
    mag_src = 18.0
    t = np.linspace(t0 - 2 * tE, t0 + 2 * tE, 25)
    mag = 18.0 + 0.05 * np.sin((t - t0) / tE)
    mag_err = np.full_like(t, 0.02)
    data = {
        "t_phot1": t,
        "mag1": mag,
        "mag_err1": mag_err,
    }
    cube = {
        "t0": t0,
        "u0_amp": u0_amp,
        "tE": tE,
        "piE_E": piE_E,
        "piE_N": piE_N,
        "b_sff1": b_sff,
        "mag_src1": mag_src,
    }
    return data, cube


def test_jax_loglik_matches_fitter_pspl():
    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)
    fn, ctx = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None
    assert ctx is None
    names = fitter.fitter_param_names
    vec = np.array([cube[n] for n in names], dtype=np.float64)
    lnL_jax = float(fn(vec))
    lnL_ref = fitter.log_likely(cube)
    np.testing.assert_allclose(lnL_jax, lnL_ref, rtol=1e-9, atol=1e-6)


def test_jax_loglik_grad_finite_pspl():
    jax = pytest.importorskip("jax")
    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None

    names = fitter.fitter_param_names
    vec = np.array([cube[n] for n in names], dtype=np.float64)
    grad = jax.grad(fn)(vec)
    assert np.all(np.isfinite(grad))


def test_loglikelihood_op_jax_grad():
    pytest.importorskip("pymc")
    import pytensor
    import pytensor.tensor as pt

    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)
    names = fitter.fitter_param_names
    op = LogLikelihoodOp(fitter, names, use_jax_grad=True)
    assert op._jax_loglik is not None

    vec = np.array([cube[n] for n in names], dtype=np.float64)
    node = op.make_node(pt.vector())
    storage = [[None]]
    op.perform(node, [vec], storage)
    lnL_op = float(storage[0][0])
    lnL_ref = fitter.log_likely(cube)
    np.testing.assert_allclose(lnL_op, lnL_ref, rtol=1e-9, atol=1e-6)

    param_vec = pt.vector('param_vec', shape=(len(names),))
    logp = op(param_vec)
    grad_sym = pytensor.grad(logp, param_vec)
    g = grad_sym.eval({param_vec: vec})
    assert g.shape == vec.shape
    assert np.all(np.isfinite(g))


def test_gp_model_supported_for_analytic_jax_loglik():
    """GP photometry models use analytic tinygp likelihood (no host_vjp)."""
    pytest.importorskip("tinygp")

    data, cube = _make_pspl_phot_data()
    # GP optional params required by fitter cube expansion
    data = dict(data)
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_GP_Param1, verbose=False)
    assert jax_physics.supports_jax_loglik(fitter) is not None
    fn, ctx = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None
    names = fitter.fitter_param_names
    # Fill missing GP cube entries with defaults from CANONICAL-like values
    vec = []
    for n in names:
        if n in cube:
            vec.append(cube[n])
        elif n.startswith("gp_log"):
            vec.append(-1.0)
        elif n.startswith("gp_"):
            vec.append(1.0)
        else:
            vec.append(0.1)
    vec = np.array(vec, dtype=np.float64)
    lnL = float(fn(vec))
    assert np.isfinite(lnL)


def test_jax_joint_loglik_matches_fitter_pspl_photastrom():
    from bagle import fake_data

    data, params = fake_data.fake_data1(plot=False, verbose=False)
    fitter = MicrolensSolver(
        data, model.PSPL_PhotAstrom_noPar_Param1, verbose=False
    )
    assert jax_physics.supports_jax_joint_loglik(
        fitter
    ) == "PSPL_PhotAstrom_noPar_Param1"
    assert jax_physics.supports_jax_phot_loglik(fitter) is None

    fn, ctx = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None
    assert ctx is None

    vec = fitter.to_param_vector(fitter.to_param_dict(params))
    lnL_jax = float(fn(vec))
    lnL_ref = fitter.log_likely(params)
    np.testing.assert_allclose(lnL_jax, lnL_ref, rtol=1e-8, atol=1e-4)


def test_jax_joint_loglik_grad_finite_pspl_photastrom():
    jax = pytest.importorskip("jax")
    from bagle import fake_data

    data, params = fake_data.fake_data1(plot=False, verbose=False)
    fitter = MicrolensSolver(
        data, model.PSPL_PhotAstrom_noPar_Param1, verbose=False
    )
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None

    vec = fitter.to_param_vector(fitter.to_param_dict(params))
    grad = jax.grad(fn)(vec)
    assert np.all(np.isfinite(grad))


def test_loglikelihood_op_jax_grad_pspl_photastrom():
    pytest.importorskip("pymc")
    import pytensor
    import pytensor.tensor as pt
    from bagle import fake_data

    data, params = fake_data.fake_data1(plot=False, verbose=False)
    fitter = MicrolensSolver(
        data, model.PSPL_PhotAstrom_noPar_Param1, verbose=False
    )
    names = fitter.fitter_param_names
    op = LogLikelihoodOp(fitter, names, use_jax_grad=True)
    assert op._jax_loglik is not None

    vec = fitter.to_param_vector(fitter.to_param_dict(params))
    node = op.make_node(pt.vector())
    storage = [[None]]
    op.perform(node, [vec], storage)
    lnL_op = float(storage[0][0])
    lnL_ref = fitter.log_likely(params)
    np.testing.assert_allclose(lnL_op, lnL_ref, rtol=1e-8, atol=1e-4)

    param_vec = pt.vector('param_vec', shape=(len(names),))
    logp = op(param_vec)
    grad_sym = pytensor.grad(logp, param_vec)
    g = grad_sym.eval({param_vec: vec})
    assert g.shape == vec.shape
    assert np.all(np.isfinite(g))

def test_explicit_phot_param_method_replaces_packing():
    """Phot likelihood dispatches through the explicit Param method."""
    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)
    names = fitter.fitter_param_names
    vec = np.array([cube[n] for n in names], dtype=np.float64)
    assert hasattr(model.PSPL_PhotParam1, "jax_log_likely_photometry")
    assert not hasattr(model.PSPL_PhotParam1, "get_params_for_jax")
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    lnL = float(fn(vec))
    lnL_ref = fitter.log_likely(cube)
    np.testing.assert_allclose(lnL, lnL_ref, rtol=1e-9, atol=1e-6)


def test_explicit_joint_param_methods_replace_packing():
    """Joint likelihood uses explicit photometric and astrometric methods."""
    from bagle import fake_data

    data, params = fake_data.fake_data1(plot=False, verbose=False)
    fitter1 = MicrolensSolver(
        data, model.PSPL_PhotAstrom_noPar_Param1, verbose=False
    )
    vec1 = fitter1.to_param_vector(fitter1.to_param_dict(params))
    assert hasattr(model.PSPL_PhotAstromParam1, "jax_log_likely_photometry")
    assert hasattr(model.PSPL_PhotAstromParam1, "jax_log_likely_astrometry")
    assert not hasattr(model.PSPL_PhotAstromParam1, "get_params_for_jax")
    fn, _ = build_explicit_jax_loglik_fn(fitter1)
    lnL = float(fn(vec1))
    lnL_ref = fitter1.log_likely(params)
    np.testing.assert_allclose(lnL, lnL_ref, rtol=1e-8, atol=1e-4)


def test_explicit_param_loglik_gradient():
    jax = pytest.importorskip("jax")
    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None
    names = fitter.fitter_param_names
    vec = np.array([cube[n] for n in names], dtype=np.float64)
    grad = jax.grad(fn)(vec)
    assert np.all(np.isfinite(grad))


@pytest.mark.parametrize(
    "param_cls",
    [
        model.PSPL_PhotParam1,
        model.PSPL_PhotParam2,
        model.PSPL_PhotParam3,
        model.PSBL_PhotParam1,
        model.BSPL_PhotParam1,
    ],
)
def test_explicit_photometry_param_methods(param_cls):
    assert callable(param_cls.jax_log_likely_photometry)
    assert not hasattr(param_cls, "get_params_for_jax")


@pytest.mark.parametrize(
    "param_cls",
    [
        model.PSPL_PhotAstromParam1,
        model.PSPL_PhotAstromParam2,
        model.PSPL_PhotAstromParam3,
    ],
)
def test_explicit_joint_param_methods(param_cls):
    assert callable(param_cls.jax_log_likely_photometry)
    assert callable(param_cls.jax_log_likely_astrometry)
    assert not hasattr(param_cls, "get_params_for_jax")


def test_explicit_astrometry_param_method():
    assert callable(model.PSPL_AstromParam3.jax_log_likely_astrometry)
    assert not hasattr(model.PSPL_AstromParam3, "get_params_for_jax")


def test_jax_physics_loglik_wrappers_match_explicit_builder():
    """jax_physics build/supports entry points use build_explicit_jax_loglik_fn."""
    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)

    fn_exp, ctx_exp = build_explicit_jax_loglik_fn(fitter)
    fn_phys, ctx_phys = jax_physics.build_jax_loglik_fn(fitter)
    fn_joint, ctx_joint = jax_physics.build_jax_joint_loglik_fn(fitter)

    assert fn_exp is not None
    assert fn_phys is fn_exp
    assert fn_joint is fn_exp
    assert ctx_phys is ctx_exp
    assert ctx_joint is ctx_exp
    assert jax_physics.supports_jax_loglik(fitter) == (
        fitter.model_class.__name__
    )

    names = fitter.fitter_param_names
    vec = np.array([cube[n] for n in names], dtype=np.float64)
    np.testing.assert_allclose(float(fn_phys(vec)), float(fn_exp(vec)))


def test_no_layout_or_packing_modules():
    with pytest.raises(ModuleNotFoundError):
        __import__("bagle.jax.layout_registry")
    with pytest.raises(ModuleNotFoundError):
        __import__("bagle.jax.param_pack")


def test_no_evaluate_or_likelihood_modules():
    with pytest.raises(ModuleNotFoundError):
        __import__("bagle.jax.evaluate")
    with pytest.raises(ModuleNotFoundError):
        __import__("bagle.jax.likelihood")


def _make_reduced_phot_data():
    t = np.linspace(57000.0, 57200.0, 20)
    return {
        "t_phot1": t,
        "mag1": 18.0 + 0.05 * np.sin((t - 57100.0) / 30.0),
        "mag_err1": np.full_like(t, 0.02),
    }


@pytest.mark.parametrize(
    "model_cls,params",
    [
        (
            model.PSPL_Phot_noPar_Param2,
            {
                "t0": 57100.0,
                "u0_amp": 0.05,
                "tE": 45.0,
                "piE_E": 0.01,
                "piE_N": 0.02,
                "b_sff1": 0.8,
                "mag_base1": 18.5,
            },
        ),
        (
            model.PSPL_Phot_noPar_Param3,
            {
                "t0": 57100.0,
                "u0_amp": 0.05,
                "log_tE": np.log10(45.0),
                "log_piE": np.log10(0.022),
                "phi_muRel": 45.0,
                "b_sff1": 0.8,
                "mag_base1": 18.5,
            },
        ),
    ],
)
def test_jax_reduced_phot_loglik(model_cls, params):
    fitter = MicrolensSolver(
        _make_reduced_phot_data(), model_cls, verbose=False
    )
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None
    vec = fitter.to_param_vector(fitter.to_param_dict(params))
    np.testing.assert_allclose(
        float(fn(vec)), fitter.log_likely(params), rtol=1e-8, atol=1e-4
    )


def test_jax_pspl_astrom_param3_loglik():
    from bagle import fake_data

    theta_e = 0.8
    data, _ = fake_data.fake_data2(
        259.5, -29.0, 57100.0, 0.05, 45.0, theta_e, 0.15,
        np.array([0.01, 0.02]), np.array([0.0, 0.0]),
        np.array([1.5, -0.5]), 1.0, 19.0, plot=False
    )
    params = {
        "t0": 57100.0,
        "u0_amp": 0.05,
        "tE": 45.0,
        "log10_thetaE": np.log10(theta_e),
        "piS": 0.15,
        "piE_E": 0.01,
        "piE_N": 0.02,
        "xS0_E": 0.0,
        "xS0_N": 0.0,
        "muS_E": 1.5,
        "muS_N": -0.5,
        "b_sff": 1.0,
    }
    fitter = MicrolensSolver(
        data, model.PSPL_Astrom_Par_Param3, verbose=False
    )
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None
    vec = fitter.to_param_vector(fitter.to_param_dict(params))
    np.testing.assert_allclose(
        float(fn(vec)), fitter.log_likely(params), rtol=1e-8, atol=1e-3
    )


# ---------------------------------------------------------------------------
# Milestone (c): FSPL traced likelihood (A-09, A-10, G-01, G-05)
# ---------------------------------------------------------------------------

_FSPL_N_OUTLINE = 400


def _fspl_phot_host_and_vec(n_outline=_FSPL_N_OUTLINE):
    """Phot-only FSPL host instance, times, and packed Param1 vector."""
    t0 = 57000.0
    kw = dict(
        t0=t0,
        u0_amp=0.005,
        tE=25.0,
        piE_E=0.05,
        piE_N=0.02,
        radiusS=0.02,
        b_sff=[1.0],
        mag_src=[18.0],
        n_outline=n_outline,
    )
    inst = model.FSPL_Phot_noPar_Param1(**kw)
    t = np.linspace(t0 - 40.0, t0 + 40.0, 81)
    vec = np.array(
        [
            kw["t0"],
            kw["u0_amp"],
            kw["tE"],
            kw["piE_E"],
            kw["piE_N"],
            kw["radiusS"],
            kw["b_sff"][0],
            kw["mag_src"][0],
        ],
        dtype=np.float64,
    )
    return inst, t, vec, kw


def test_A09_fspl_traced_mag_matches_host():
    """A-09: traced FSPL magnitudes vs host get_photometry."""
    from bagle.jax.fspl import fspl_photometry
    from bagle.jax.geometry import derive_pspl_static_geometry

    inst, t, _vec, kw = _fspl_phot_host_and_vec()
    host = np.asarray(inst.get_photometry(t), dtype=np.float64)
    u0, thetaE_hat, _ = derive_pspl_static_geometry(
        kw["u0_amp"], kw["piE_E"], kw["piE_N"]
    )
    mag_j = np.asarray(
        fspl_photometry(
            t,
            kw["t0"],
            kw["tE"],
            u0,
            thetaE_hat,
            kw["radiusS"],
            _FSPL_N_OUTLINE,
            kw["mag_src"][0],
            kw["b_sff"][0],
        ),
        dtype=np.float64,
    )

    # Split epochs near / far from the limb (|u| < 2 rho).
    u_amp = np.linalg.norm(np.asarray(inst.get_u(t)), axis=1)
    near = u_amp < 2.0 * kw["radiusS"]
    far = ~near
    rel = np.abs(mag_j - host) / np.maximum(np.abs(host), 1e-12)
    assert far.any() and near.any()
    assert np.max(rel[far]) < 1e-5
    assert np.max(rel[near]) < 1e-3
    return None


def test_G01_fspl_loglik_grad_t0_tE_radiusS():
    """G-01: jax.grad of FSPL phot lnL in t0, tE, radiusS vs central FD."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from bagle.jax.fspl import fspl_log_likely_photometry
    from bagle.jax.geometry import derive_pspl_static_geometry

    inst, t, vec, kw = _fspl_phot_host_and_vec(n_outline=200)
    mag_obs = np.asarray(inst.get_photometry(t), dtype=np.float64) + 0.02
    mag_err = np.full_like(t, 0.03)

    def lnL_of(v):
        p = dict(zip(model.FSPL_PhotParam1.fitter_param_names, v))
        u0, thetaE_hat, _ = derive_pspl_static_geometry(
            p["u0_amp"], p["piE_E"], p["piE_N"]
        )
        return fspl_log_likely_photometry(
            t,
            p["t0"],
            p["tE"],
            u0,
            thetaE_hat,
            p["radiusS"],
            200,
            p["mag_src"],
            p["b_sff"],
            mag_obs,
            mag_err,
        )

    v0 = jnp.asarray(vec, dtype=jnp.float64)
    g = np.asarray(jax.grad(lnL_of)(v0), dtype=np.float64)
    names = list(model.FSPL_PhotParam1.fitter_param_names)
    for name in ("t0", "tE", "radiusS"):
        i = names.index(name)
        assert np.isfinite(g[i]), name
        assert abs(g[i]) > 0.0, name

        # Central finite difference on the traced forward.
        eps = 1e-5 * max(1.0, abs(float(v0[i])))
        vp = v0.at[i].add(eps)
        vm = v0.at[i].add(-eps)
        fd = float((lnL_of(vp) - lnL_of(vm)) / (2.0 * eps))
        np.testing.assert_allclose(g[i], fd, rtol=1e-4, atol=1e-4)
    return None


def _fspl_two_filter_data():
    """Synthetic two-filter phot+astrom data for tied-xS0 tests."""
    t0 = 57000.0
    t = np.linspace(t0 - 30.0, t0 + 30.0, 25)
    # Shared true geometry; filter 2 has a small photometric offset.
    kw = dict(
        mL=10.0,
        t0=t0,
        beta=0.15,
        dL=4000.0,
        dL_dS=0.5,
        xS0_E=[0.0, 0.0],
        xS0_N=[0.0, 0.0],
        muL_E=0.0,
        muL_N=0.0,
        muS_E=3.0,
        muS_N=0.0,
        radiusS=4e-4,
        b_sff=[1.0, 1.0],
        mag_src=[18.0, 18.2],
        n_outline=_FSPL_N_OUTLINE,
    )
    inst = model.FSPL_PhotAstrom_noPar_Param1(**kw)
    mag1 = np.asarray(inst.get_photometry(t, filt_idx=0), dtype=np.float64)
    mag2 = np.asarray(inst.get_photometry(t, filt_idx=1), dtype=np.float64)
    pos1 = np.asarray(inst.get_astrometry(t, filt_idx=0), dtype=np.float64)
    pos2 = np.asarray(inst.get_astrometry(t, filt_idx=1), dtype=np.float64)
    data = {
        "t_phot1": t,
        "mag1": mag1 + 0.01,
        "mag_err1": np.full_like(t, 0.02),
        "t_phot2": t,
        "mag2": mag2 + 0.01,
        "mag_err2": np.full_like(t, 0.02),
        "t_ast1": t,
        "xpos1": pos1[:, 0] + 1e-4,
        "ypos1": pos1[:, 1] - 1e-4,
        "xpos_err1": np.full_like(t, 1e-3),
        "ypos_err1": np.full_like(t, 1e-3),
        "t_ast2": t,
        "xpos2": pos2[:, 0] + 1e-4,
        "ypos2": pos2[:, 1] - 1e-4,
        "xpos_err2": np.full_like(t, 1e-3),
        "ypos_err2": np.full_like(t, 1e-3),
        "raL": 259.0,
        "decL": -29.0,
    }
    return data, kw


def test_A10_fspl_tied_xS0_two_filters():
    """A-10: two-filter FSPL with xS0_2 tied to xS0_1.

    Host photometry/astrometry for filter 2 match filter 1 when origins
    are tied. The JAX likelihood is finite and agrees with an untied
    cube that copies the source xS0 into filter 2 (tie assembly).
    """
    data, kw = _fspl_two_filter_data()
    fitter = MicrolensSolver(
        data, model.FSPL_PhotAstrom_noPar_Param1, verbose=False
    )
    fitter.tie_fit_param("xS0_E2", "xS0_E1")
    fitter.tie_fit_param("xS0_N2", "xS0_N1")

    params = {
        "mL": kw["mL"],
        "t0": kw["t0"],
        "beta": kw["beta"],
        "dL": kw["dL"],
        "dL_dS": kw["dL_dS"],
        "xS0_E1": 0.021,
        "xS0_N1": -0.013,
        "muL_E": kw["muL_E"],
        "muL_N": kw["muL_N"],
        "muS_E": kw["muS_E"],
        "muS_N": kw["muS_N"],
        "radiusS": kw["radiusS"],
        "b_sff1": 1.0,
        "b_sff2": 1.0,
        "mag_src1": 18.0,
        "mag_src2": 18.2,
    }
    mod = fitter.get_model(params)
    # Tied origins: both filters share the source slot.
    np.testing.assert_allclose(mod.xS0[0], [0.021, -0.013])
    np.testing.assert_allclose(mod.xS0[1], [0.021, -0.013])

    t = data["t_phot1"]
    # Host filter photometry differs only by mag_src, not by origin.
    mag1 = np.asarray(mod.get_photometry(t, filt_idx=0), dtype=np.float64)
    mag2 = np.asarray(mod.get_photometry(t, filt_idx=1), dtype=np.float64)
    # Same amplification (shared geometry); magnitude offset from mag_src.
    amp1 = np.asarray(mod.get_amplification(t, filt_idx=0), dtype=np.float64)
    amp2 = np.asarray(mod.get_amplification(t, filt_idx=1), dtype=np.float64)
    np.testing.assert_allclose(amp1, amp2, rtol=1e-5, atol=1e-4)

    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None
    vec = fitter.to_param_vector(fitter.to_param_dict(params))
    lnL_tied = float(fn(vec))
    assert np.isfinite(lnL_tied)

    # Untied fitter with xS0_2 set equal to xS0_1 must match the tied lnL.
    fitter_free = MicrolensSolver(
        data, model.FSPL_PhotAstrom_noPar_Param1, verbose=False
    )
    params_free = dict(params)
    params_free["xS0_E2"] = params["xS0_E1"]
    params_free["xS0_N2"] = params["xS0_N1"]
    fn_free, _ = build_explicit_jax_loglik_fn(fitter_free)
    vec_free = fitter_free.to_param_vector(
        fitter_free.to_param_dict(params_free)
    )
    lnL_free = float(fn_free(vec_free))
    np.testing.assert_allclose(lnL_tied, lnL_free, rtol=1e-5, atol=1e-4)
    return None


def test_G05_fspl_tied_xS0_grad_nonzero():
    """G-05: gradient w.r.t. tied-to xS0 slot is finite and nonzero."""
    jax = pytest.importorskip("jax")

    data, kw = _fspl_two_filter_data()
    fitter = MicrolensSolver(
        data, model.FSPL_PhotAstrom_noPar_Param1, verbose=False
    )
    fitter.tie_fit_param("xS0_E2", "xS0_E1")
    fitter.tie_fit_param("xS0_N2", "xS0_N1")
    params = {
        "mL": kw["mL"],
        "t0": kw["t0"],
        "beta": kw["beta"],
        "dL": kw["dL"],
        "dL_dS": kw["dL_dS"],
        "xS0_E1": 0.021,
        "xS0_N1": -0.013,
        "muL_E": kw["muL_E"],
        "muL_N": kw["muL_N"],
        "muS_E": kw["muS_E"],
        "muS_N": kw["muS_N"],
        "radiusS": kw["radiusS"],
        "b_sff1": 1.0,
        "b_sff2": 1.0,
        "mag_src1": 18.0,
        "mag_src2": 18.2,
    }
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None
    names = list(fitter.fitter_param_names)
    vec = np.asarray(
        fitter.to_param_vector(fitter.to_param_dict(params)),
        dtype=np.float64,
    )
    grad = np.asarray(jax.grad(fn)(vec), dtype=np.float64)
    for name in ("xS0_E1", "xS0_N1"):
        component = float(grad[names.index(name)])
        assert np.isfinite(component), name
        assert component != 0.0, name
    return None

