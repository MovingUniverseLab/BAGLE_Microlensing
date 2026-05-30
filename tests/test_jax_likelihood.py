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

    g = op.grad([vec], [np.array(1.0)])
    assert g[0].shape == vec.shape
    assert np.all(np.isfinite(g[0]))


def test_gp_model_not_supported_for_jax_loglik():
    data, cube = _make_pspl_phot_data()
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_GP_Param1, verbose=False)
    assert jax_physics.supports_jax_phot_loglik(fitter) is None


def test_jax_joint_loglik_matches_fitter_pspl_photastrom():
    from bagle import fake_data

    data, params = fake_data.fake_data1(plot=False, verbose=False)
    fitter = MicrolensSolver(
        data, model.PSPL_PhotAstrom_noPar_Param1, verbose=False
    )
    assert jax_physics.supports_jax_joint_loglik(fitter) == "pspl_photastrom_param1"
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

    g = op.grad([vec], [np.array(1.0)])
    assert g[0].shape == vec.shape
    assert np.all(np.isfinite(g[0]))
