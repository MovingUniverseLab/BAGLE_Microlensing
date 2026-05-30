"""Golden tests for registry-backed PSPL reduced Phot / PhotAstrom / Astrom layouts."""
import numpy as np
import pytest

from bagle import jax_physics, model_jax as model


def _phot_data():
    t = np.linspace(57000.0, 57200.0, 20)
    return {
        "t_phot1": t,
        "mag1": 18.0 + 0.05 * np.sin((t - 57100) / 30),
        "mag_err1": np.full_like(t, 0.02),
    }


@pytest.mark.parametrize(
    "model_cls,params",
    [
        (
            model.PSPL_Phot_noPar_Param2,
            dict(
                t0=57100.0,
                u0_amp=0.05,
                tE=45.0,
                piE_E=0.01,
                piE_N=0.02,
                b_sff1=0.8,
                mag_base1=18.5,
            ),
        ),
        (
            model.PSPL_Phot_noPar_Param3,
            dict(
                t0=57100.0,
                u0_amp=0.05,
                log_tE=np.log10(45.0),
                log_piE=np.log10(0.022),
                phi_muRel=45.0,
                b_sff1=0.8,
                mag_base1=18.5,
            ),
        ),
    ],
)
def test_jax_phot_reduced_loglik(model_cls, params):
    pytest.importorskip("pytensor")
    from bagle.model_fitter import MicrolensSolver

    data = _phot_data()
    fitter = MicrolensSolver(data, model_cls, verbose=False)
    fn, ctx = jax_physics.build_jax_loglik_fn(fitter)
    assert fn is not None
    names = fitter.fitter_param_names
    vec = np.array([params[n] for n in names], dtype=np.float64)
    lnL_jax = float(fn(vec))
    lnL_ref = fitter.log_likely(params)
    np.testing.assert_allclose(lnL_jax, lnL_ref, rtol=1e-8, atol=1e-4)


def test_jax_pspl_astrom_param3_loglik():
    pytest.importorskip("pytensor")
    from bagle import fake_data
    from bagle.model_fitter import MicrolensSolver

    ra_l, dec_l = 259.5, -29.0
    theta_e = 0.8
    data, _ = fake_data.fake_data2(
        ra_l,
        dec_l,
        57100.0,
        0.05,
        45.0,
        theta_e,
        0.15,
        np.array([0.01, 0.02]),
        np.array([0.0, 0.0]),
        np.array([1.5, -0.5]),
        1.0,
        19.0,
        plot=False,
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
    fitter = MicrolensSolver(data, model.PSPL_Astrom_Par_Param3, verbose=False)
    fn, _ = jax_physics.build_jax_loglik_fn(fitter)
    assert fn is not None
    names = fitter.fitter_param_names
    vec = np.array([params[n] for n in names], dtype=np.float64)
    lnL_jax = float(fn(vec))
    lnL_ref = fitter.log_likely(params)
    np.testing.assert_allclose(lnL_jax, lnL_ref, rtol=1e-8, atol=1e-3)


def test_jax_psbl_joint_param1_loglik():
    pytest.importorskip("pytensor")
    from bagle import fake_data
    from bagle.model_fitter import MicrolensSolver

    data, params, _, _ = fake_data.fake_data_PSBL(parallax=False, animate=False)
    fitter = MicrolensSolver(data, model.PSBL_PhotAstrom_noPar_Param1, verbose=False)
    fn, _ = jax_physics.build_jax_loglik_fn(fitter)
    assert fn is not None
    names = fitter.fitter_param_names
    vec = np.array([params[n] for n in names], dtype=np.float64)
    lnL_jax = float(fn(vec))
    lnL_ref = fitter.log_likely(params)
    np.testing.assert_allclose(lnL_jax, lnL_ref, rtol=1e-7, atol=1e-3)
