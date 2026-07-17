"""Tests for JAX photometry log-likelihood and PyMC gradients."""
import numpy as np
import pytest

from bagle import jax_physics, model_jax as model
from bagle.model_fitter import LogLikelihoodOp, MicrolensSolver


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
    ctx = jax_physics.build_jax_phot_likelihood_context(fitter)
    assert ctx is not None
    assert ctx.model_kind == "pspl"

    names = fitter.fitter_param_names
    vec = np.array([cube[n] for n in names], dtype=np.float64)
    lnL_jax = float(jax_physics.log_likelihood_phot_from_vec(vec, ctx))
    lnL_ref = fitter.log_likely(cube)
    np.testing.assert_allclose(lnL_jax, lnL_ref, rtol=1e-9, atol=1e-6)


def test_jax_loglik_grad_finite_pspl():
    jax = pytest.importorskip("jax")
    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)
    fn, _ = jax_physics.build_jax_phot_loglik_fn(fitter)
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
    from bagle.jax.likelihood import build_jax_loglik_fn, supports_jax_loglik_for_fitter

    data, cube = _make_pspl_phot_data()
    # GP optional params required by fitter cube expansion
    data = dict(data)
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_GP_Param1, verbose=False)
    assert supports_jax_loglik_for_fitter(fitter) is not None
    fn, ctx = build_jax_loglik_fn(fitter)
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
    assert jax_physics.supports_jax_joint_loglik(fitter) == "PSPL_PhotAstromParam1"
    assert jax_physics.supports_jax_phot_loglik(fitter) is None

    fn, ctx = jax_physics.build_jax_joint_loglik_fn(fitter)
    assert fn is not None
    assert ctx is not None

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
    fn, _ = jax_physics.build_jax_loglik_fn(fitter)
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

def test_build_jax_loglik_uses_get_params_for_jax_phot():
    """Phot lnL path must pack via Param.get_params_for_jax."""
    from unittest import mock

    from bagle.jax.likelihood import build_jax_loglik_fn

    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)
    names = fitter.fitter_param_names
    vec = np.array([cube[n] for n in names], dtype=np.float64)

    with mock.patch.object(
        model.PSPL_PhotParam1,
        "get_params_for_jax",
        wraps=model.PSPL_PhotParam1.get_params_for_jax,
    ) as spy:
        with mock.patch(
            "bagle.jax.geometry.derive_geometry_from_layout",
            side_effect=AssertionError("derive_geometry_from_layout must not be called"),
        ):
            fn, ctx = build_jax_loglik_fn(fitter)
            assert fn is not None
            lnL = float(fn(vec))
        assert spy.called

    lnL_ref = fitter.log_likely(cube)
    np.testing.assert_allclose(lnL, lnL_ref, rtol=1e-9, atol=1e-6)


def test_build_jax_loglik_uses_get_params_for_jax_joint():
    """Joint lnL path packs via PhotAstromParam1.get_params_for_jax."""
    from unittest import mock

    from bagle import fake_data
    from bagle.jax.likelihood import build_jax_loglik_fn

    data, params = fake_data.fake_data1(plot=False, verbose=False)
    fitter1 = MicrolensSolver(
        data, model.PSPL_PhotAstrom_noPar_Param1, verbose=False
    )
    names1 = fitter1.fitter_param_names
    vec1 = np.array([params[n] for n in names1], dtype=np.float64)

    with mock.patch.object(
        model.PSPL_PhotAstromParam1,
        "get_params_for_jax",
        wraps=model.PSPL_PhotAstromParam1.get_params_for_jax,
    ) as spy:
        with mock.patch(
            "bagle.jax.geometry.derive_geometry_from_layout",
            side_effect=AssertionError("derive_geometry_from_layout must not be called"),
        ):
            fn, ctx = build_jax_loglik_fn(fitter1)
            assert fn is not None
            lnL = float(fn(vec1))
        assert spy.called

    lnL_ref = fitter1.log_likely(params)
    np.testing.assert_allclose(lnL, lnL_ref, rtol=1e-8, atol=1e-4)


def test_build_jax_loglik_grad_via_get_params_phot():
    jax = pytest.importorskip("jax")
    from bagle.jax.likelihood import build_jax_loglik_fn

    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)
    fn, _ = build_jax_loglik_fn(fitter)
    assert fn is not None
    names = fitter.fitter_param_names
    vec = np.array([cube[n] for n in names], dtype=np.float64)
    grad = jax.grad(fn)(vec)
    assert np.all(np.isfinite(grad))

