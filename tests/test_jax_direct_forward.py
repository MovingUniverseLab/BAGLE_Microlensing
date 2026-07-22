"""Direct JAX forward path: get_* → jax_physics kernels (no layout registry)."""
from __future__ import annotations

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bagle import jax_physics
from bagle import model as model_np
from bagle import model_jax as model


def _make_pspl_phot_no_par():
    return model.PSPL_Phot_noPar_Param1(
        t0=57000.0,
        u0_amp=0.05,
        tE=45.0,
        piE_E=0.01,
        piE_N=0.02,
        b_sff=np.array([0.8]),
        mag_src=np.array([18.0]),
    )


def _make_pspl_phot_par():
    return model.PSPL_Phot_Par_Param1(
        t0=57000.0,
        u0_amp=0.05,
        tE=45.0,
        piE_E=0.01,
        piE_N=0.02,
        b_sff=np.array([0.8]),
        mag_src=np.array([18.0]),
        raL=17.75,
        decL=-29.0,
    )


def _make_pspl_phot_np(par=False):
    kw = dict(
        t0=57000.0,
        u0_amp=0.05,
        tE=45.0,
        piE_E=0.01,
        piE_N=0.02,
        b_sff=np.array([0.8]),
        mag_src=np.array([18.0]),
    )
    if par:
        return model_np.PSPL_Phot_Par_Param1(
            raL=17.75, decL=-29.0, **kw
        )
    return model_np.PSPL_Phot_noPar_Param1(**kw)


def _times(inst, n=31):
    return np.linspace(inst.t0 - 2 * inst.tE, inst.t0 + 2 * inst.tE, n)


def assert_no_layout_registry(fn):
    """Ensure deleted layout registry is not imported during forward eval."""
    with mock.patch.dict(
        "sys.modules",
        {"bagle.jax.layout_registry": None},
    ):
        return fn()


# Back-compat name used by tests below.
assert_no_resolve_layout = assert_no_layout_registry


def assert_jnp_grad_finite(scalar_fn, primal):
    g = jax.grad(scalar_fn)(primal)
    g_np = np.asarray(g)
    assert np.all(np.isfinite(g_np)), g_np
    return g_np


def test_parallax_helper_shape():
    m = _make_pspl_phot_par()
    t = _times(m)
    pvec = m._parallax_vectors_for_jax(t, 0)
    assert pvec is not None
    assert np.asarray(pvec).shape == (len(t), 2)
    m2 = _make_pspl_phot_no_par()
    assert m2._parallax_vectors_for_jax(t, 0) is None


@pytest.mark.parametrize("par", [False, True])
def test_pspl_get_amplification_matches_numpy(par):
    mj = _make_pspl_phot_par() if par else _make_pspl_phot_no_par()
    mn = _make_pspl_phot_np(par=par)
    t = _times(mj)

    def run():
        return mj.get_amplification(t)

    a_jax = assert_no_resolve_layout(run)
    a_np = mn.get_amplification(t)
    np.testing.assert_allclose(a_jax, a_np, rtol=1e-9, atol=1e-8)


def test_pspl_amplification_kernel_grad():
    mj = _make_pspl_phot_no_par()
    t = jnp.asarray(_times(mj), dtype=jnp.float64)
    u0 = jnp.asarray(mj.u0, dtype=jnp.float64)
    th = jnp.asarray(mj.thetaE_hat, dtype=jnp.float64)
    t0 = jnp.asarray(mj.t0, dtype=jnp.float64)
    tE = jnp.asarray(mj.tE, dtype=jnp.float64)
    piE_E = jnp.asarray(mj.piE[0], dtype=jnp.float64)
    piE_N = jnp.asarray(mj.piE[1], dtype=jnp.float64)

    def f(t0_):
        return jnp.sum(
            jax_physics.pspl_amplification(
                t, t0_, tE, u0, th, piE_E=piE_E, piE_N=piE_N
            )
        )

    g = assert_jnp_grad_finite(f, t0)
    eps = 1e-5
    fd = (float(f(t0 + eps)) - float(f(t0 - eps))) / (2 * eps)
    np.testing.assert_allclose(float(g), fd, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("par", [False, True])
def test_pspl_get_photometry_matches_numpy(par):
    mj = _make_pspl_phot_par() if par else _make_pspl_phot_no_par()
    mn = _make_pspl_phot_np(par=par)
    t = _times(mj)

    def run():
        return mj.get_photometry(t)

    mag_j = assert_no_resolve_layout(run)
    mag_n = mn.get_photometry(t)
    np.testing.assert_allclose(mag_j, mag_n, rtol=1e-9, atol=1e-8)


def test_pspl_photometry_kernel_grad():
    mj = _make_pspl_phot_no_par()
    t = jnp.asarray(_times(mj), dtype=jnp.float64)
    u0 = jnp.asarray(mj.u0, dtype=jnp.float64)
    th = jnp.asarray(mj.thetaE_hat, dtype=jnp.float64)

    def f(t0_):
        return jnp.sum(
            jax_physics.pspl_photometry(
                t, t0_, mj.tE, u0, th, mj.mag_src[0], b_sff=mj.b_sff[0],
                piE_E=mj.piE[0], piE_N=mj.piE[1],
            )
        )

    assert_jnp_grad_finite(f, jnp.asarray(mj.t0, dtype=jnp.float64))


def test_pspl_get_u_matches_numpy():
    mj = _make_pspl_phot_no_par()
    mn = _make_pspl_phot_np(False)
    t = _times(mj)
    u_j = assert_no_resolve_layout(lambda: mj.get_u(t))
    u_n = mn.get_u(t)
    np.testing.assert_allclose(u_j, u_n, rtol=1e-9, atol=1e-8)
    amp_from_u = np.asarray(
        jax_physics.pspl_amplification_from_u(jnp.asarray(u_j))
    )
    np.testing.assert_allclose(amp_from_u, mj.get_amplification(t), rtol=1e-10)


def test_pspl_resolved_amplification_sum():
    mj = _make_pspl_phot_no_par()
    t = _times(mj)
    r = assert_no_resolve_layout(lambda: mj.get_resolved_amplification(t))
    r = np.asarray(r)
    # Phot API: [N_times, N_sources=1, N_images=2]
    assert r.ndim == 3 and r.shape[1:] == (1, 2)
    tot = r[:, 0, 0] + r[:, 0, 1]
    np.testing.assert_allclose(tot, mj.get_amplification(t), rtol=1e-10)


def test_pspl_chi2_photometry():
    mj = _make_pspl_phot_no_par()
    t = _times(mj)
    mag = mj.get_photometry(t)
    err = np.full_like(mag, 0.02)
    chi2 = assert_no_resolve_layout(
        lambda: mj.get_chi2_photometry(t, mag + 0.01, err)
    )
    assert chi2.shape == mag.shape
    assert np.all(chi2 > 0)


def test_explicit_param_likelihood_matches_model_instance():
    mj = _make_pspl_phot_no_par()
    t = _times(mj)
    vec = jnp.array(
        [mj.t0, mj.u0_amp, mj.tE, mj.piE[0], mj.piE[1]], dtype=jnp.float64
    )
    mag_obs = mj.get_photometry(t) + 0.01
    mag_err = np.full_like(mag_obs, 0.02)
    actual = model.PSPL_PhotParam1.jax_log_likely_photometry(
        vec, t, mag_obs, mag_err, mj.b_sff[0], mj.mag_src[0]
    )
    expected = mj.log_likely_photometry(t, mag_obs, mag_err)
    np.testing.assert_allclose(actual, expected)


def test_explicit_param_likelihood_grad():
    mj = _make_pspl_phot_no_par()
    t = _times(mj)
    mag_obs = mj.get_photometry(t) + 0.01
    mag_err = np.full_like(mag_obs, 0.02)
    vec0 = jnp.array(
        [mj.t0, mj.u0_amp, mj.tE, mj.piE[0], mj.piE[1]], dtype=jnp.float64
    )

    def f(v):
        return model.PSPL_PhotParam1.jax_log_likely_photometry(
            v, t, mag_obs, mag_err, mj.b_sff[0], mj.mag_src[0]
        )

    assert_jnp_grad_finite(f, vec0)
