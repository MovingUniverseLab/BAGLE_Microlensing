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

    names = fitter.fitter_param_names
    vec = np.array([params[n] for n in names], dtype=np.float64)
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

    names = fitter.fitter_param_names
    vec = np.array([params[n] for n in names], dtype=np.float64)
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

    vec = np.array([params[n] for n in names], dtype=np.float64)
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
    names1 = fitter1.fitter_param_names
    vec1 = np.array([params[n] for n in names1], dtype=np.float64)
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


def test_unsupported_binary_astrometry_is_not_advertised():
    assert not hasattr(
        model.PSBL_PhotAstromParam1, "jax_log_likely_astrometry"
    )
    assert not hasattr(
        model.BSPL_PhotAstromParam1, "jax_log_likely_astrometry"
    )


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
    vec = np.array(
        [params[name] for name in fitter.fitter_param_names],
        dtype=np.float64,
    )
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
    }
    fitter = MicrolensSolver(
        data, model.PSPL_Astrom_Par_Param3, verbose=False
    )
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is not None
    vec = np.array(
        [params[name] for name in fitter.fitter_param_names],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        float(fn(vec)), fitter.log_likely(params), rtol=1e-8, atol=1e-3
    )


def test_jax_psbl_joint_rejected_without_binary_astrometry():
    from bagle import fake_data

    data, _, _, _ = fake_data.fake_data_PSBL(
        parallax=False, animate=False
    )
    fitter = MicrolensSolver(
        data, model.PSBL_PhotAstrom_noPar_Param1, verbose=False
    )
    fn, _ = build_explicit_jax_loglik_fn(fitter)
    assert fn is None

