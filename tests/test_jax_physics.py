"""Tests for stateless PSBL JAX physics in bagle.jax_physics."""
import jax
import numpy as np
import pytest

from bagle import jax_physics, model_jax as model


def _sort_roots(z_row):
    z = np.asarray(z_row, dtype=np.complex128)
    order = np.lexsort((z.imag, z.real))
    return z[order]


def _assert_roots_allclose(z_ref, z_test, rtol=1e-9, atol=1e-11):
    z_ref = np.asarray(z_ref, dtype=np.complex128)
    z_test = np.asarray(z_test, dtype=np.complex128)
    assert z_ref.shape == z_test.shape
    for row in range(z_ref.shape[0]):
        ref = _sort_roots(z_ref[row])
        tst = _sort_roots(z_test[row])
        ref_nan = np.isnan(ref.real)
        tst_nan = np.isnan(tst.real)
        np.testing.assert_array_equal(ref_nan, tst_nan, err_msg=f"row {row}: NaN mask mismatch")
        valid = ~ref_nan
        if not np.any(valid):
            continue
        np.testing.assert_allclose(ref[valid].real, tst[valid].real, rtol=rtol, atol=atol)
        np.testing.assert_allclose(ref[valid].imag, tst[valid].imag, rtol=rtol, atol=atol)


@pytest.fixture
def reference_grid():
    z1 = -0.6 + 0.0j
    z2 = 0.6 + 0.0j
    m1 = 0.3
    m2 = 0.7

    w_far = np.array([0.0 + 3.0j, 2.5 + 0.5j, -1.5 - 2.0j], dtype=np.complex128)
    w_near = np.array([0.05 + 0.02j, -0.08 + 0.03j, 0.12 - 0.04j], dtype=np.complex128)
    theta = np.linspace(-np.pi, np.pi, 17, endpoint=False)
    w_limb = 0.1 + 0.05j + 0.02 * np.exp(1j * theta)

    w = np.concatenate([w_far, w_near, w_limb])
    z1_arr = np.full_like(w, z1)
    z2_arr = np.full_like(w, z2)
    return w, z1_arr, z2_arr, m1, m2


def test_pack_unpack_psbl_phot_param1_roundtrip():
    params = {
        "t0": 57000.0,
        "u0_amp": 0.05,
        "tE": 45.0,
        "piE_E": 0.01,
        "piE_N": 0.02,
        "q": 0.3,
        "sep": 1.2,
        "phi": 45.0,
    }
    vec = jax_physics.pack_psbl_phot_param1(params)
    out = jax_physics.unpack_psbl_phot_param1(vec)
    for name in jax_physics.PSBL_PHOT_PARAM1_FITTER_NAMES:
        np.testing.assert_allclose(float(out[name]), params[name])


def test_psbl_image_positions_matches_model(reference_grid):
    w, z1, z2, m1, m2 = reference_grid
    psbl = model.PSBL()
    psbl.root_tol = 1e-8

    z_model = np.asarray(
        psbl.get_image_pos_arr_fast(w, z1, z2, m1, m2, check_sols=False)
    )
    z_jax = np.asarray(
        jax_physics.psbl_image_positions_jit(
            w, z1, z2, m1, m2, 1e-8, check_sols=False
        )
    )
    _assert_roots_allclose(z_model, z_jax)


def test_psbl_photometry_matches_psbl_model():
    params = model.PSBL_PhotParam1(
        t0=57000.0,
        u0_amp=0.05,
        tE=45.0,
        piE_E=0.01,
        piE_N=0.02,
        q=0.3,
        sep=1.2,
        phi=45.0,
        b_sff=np.array([0.8]),
        mag_src=np.array([18.0]),
    )
    psbl = model.PSBL_Phot()
    psbl.orbitFlag = False
    psbl.parallaxFlag = False
    for name in (
        "t0", "u0_amp", "tE", "piE", "q", "sep", "phi", "b_sff", "mag_src",
        "piE_amp", "thetaE_hat", "u0_hat", "u0", "xL1_over_theta",
        "xL2_over_theta", "m1", "m2", "root_tol",
    ):
        setattr(psbl, name, getattr(params, name))

    t = np.linspace(params.t0 - 2 * params.tE, params.t0 + 2 * params.tE, 31)
    mag_model = np.asarray(psbl.get_photometry(t, filt_idx=0))

    fitter_vec = jax_physics.pack_psbl_phot_param1(
        {
            "t0": params.t0,
            "u0_amp": params.u0_amp,
            "tE": params.tE,
            "piE_E": params.piE[0],
            "piE_N": params.piE[1],
            "q": params.q,
            "sep": params.sep,
            "phi": params.phi,
        }
    )
    mag_jax = np.asarray(
        jax_physics.psbl_photometry_from_fitter_vec(
            t,
            fitter_vec,
            mag_src=params.mag_src[0],
            b_sff=params.b_sff[0],
            root_tol=params.root_tol,
        )
    )
    np.testing.assert_allclose(mag_model, mag_jax, rtol=1e-10, atol=1e-8)


def test_mag2flux_flux2mag_jax_matches_model():
    mags = np.array([10.0, 18.0, 30.0, np.nan])
    flux_model = np.asarray(model.mag2flux(mags))
    flux_jax = np.asarray(jax_physics.mag2flux_jax(mags))
    np.testing.assert_allclose(flux_jax, flux_model, rtol=0, equal_nan=True)

    fluxes = np.array([1.0, 0.01, 1e-6, 0.0, -1.0])
    mag_model = np.asarray(model.flux2mag(fluxes))
    mag_jax = np.asarray(jax_physics.flux2mag_jax(fluxes))
    np.testing.assert_allclose(mag_jax, mag_model, rtol=1e-12, equal_nan=True)


def test_pack_unpack_pspl_phot_param1_roundtrip():
    params = {
        "t0": 57000.0,
        "u0_amp": 0.05,
        "tE": 45.0,
        "piE_E": 0.01,
        "piE_N": 0.02,
    }
    vec = jax_physics.pack_pspl_phot_param1(params)
    out = jax_physics.unpack_pspl_phot_param1(vec)
    for name in jax_physics.PSPL_PHOT_PARAM1_FITTER_NAMES:
        np.testing.assert_allclose(float(out[name]), params[name])


def test_pspl_photometry_matches_pspl_model_no_parallax():
    params = model.PSPL_PhotParam1(
        t0=57000.0,
        u0_amp=0.05,
        tE=45.0,
        piE_E=0.01,
        piE_N=0.02,
        b_sff=np.array([0.8]),
        mag_src=np.array([18.0]),
    )
    pspl = model.PSPL_Phot()
    pspl.parallaxFlag = False
    for name in (
        "t0", "u0_amp", "tE", "piE", "b_sff", "mag_src",
        "piE_amp", "thetaE_hat", "u0_hat", "u0",
    ):
        setattr(pspl, name, getattr(params, name))

    t = np.linspace(params.t0 - 2 * params.tE, params.t0 + 2 * params.tE, 31)
    mag_model = np.asarray(pspl.get_photometry(t, filt_idx=0))

    fitter_vec = jax_physics.pack_pspl_phot_param1(
        {
            "t0": params.t0,
            "u0_amp": params.u0_amp,
            "tE": params.tE,
            "piE_E": params.piE[0],
            "piE_N": params.piE[1],
        }
    )
    mag_jax = np.asarray(
        jax_physics.pspl_photometry_from_fitter_vec_jit(
            t,
            fitter_vec,
            mag_src=params.mag_src[0],
            b_sff=params.b_sff[0],
            parallax_vectors=None,
        )
    )
    np.testing.assert_allclose(mag_model, mag_jax, rtol=1e-10, atol=1e-8)


def test_pspl_photometry_matches_pspl_model_with_parallax():
    ra_l = 17.75
    dec_l = -29.0
    params = model.PSPL_Phot_Par_Param1(
        t0=57000.0,
        u0_amp=0.05,
        tE=45.0,
        piE_E=0.01,
        piE_N=0.02,
        b_sff=np.array([0.8]),
        mag_src=np.array([18.0]),
        raL=ra_l,
        decL=dec_l,
    )
    t = np.linspace(params.t0 - 2 * params.tE, params.t0 + 2 * params.tE, 31)
    mag_model = np.asarray(params.get_photometry(t, filt_idx=0))

    pvec = jax_physics.precompute_parallax_vectors(ra_l, dec_l, t)
    fitter_vec = jax_physics.pack_pspl_phot_param1(
        {
            "t0": params.t0,
            "u0_amp": params.u0_amp,
            "tE": params.tE,
            "piE_E": params.piE[0],
            "piE_N": params.piE[1],
        }
    )
    mag_jax = np.asarray(
        jax_physics.pspl_photometry_from_fitter_vec_jit(
            t,
            fitter_vec,
            mag_src=params.mag_src[0],
            b_sff=params.b_sff[0],
            parallax_vectors=pvec,
        )
    )
    np.testing.assert_allclose(mag_model, mag_jax, rtol=1e-10, atol=1e-8)


def test_psbl_photometry_matches_psbl_model_with_parallax():
    ra_l = 17.75
    dec_l = -29.0
    params = model.PSBL_Phot_Par_Param1(
        t0=57000.0,
        u0_amp=0.05,
        tE=45.0,
        piE_E=0.01,
        piE_N=0.02,
        q=0.3,
        sep=1.2,
        phi=45.0,
        b_sff=np.array([0.8]),
        mag_src=np.array([18.0]),
        raL=ra_l,
        decL=dec_l,
    )
    t = np.linspace(params.t0 - 2 * params.tE, params.t0 + 2 * params.tE, 31)
    mag_model = np.asarray(params.get_photometry(t, filt_idx=0))

    pvec = jax_physics.precompute_parallax_vectors(ra_l, dec_l, t)
    fitter_vec = jax_physics.pack_psbl_phot_param1(
        {
            "t0": params.t0,
            "u0_amp": params.u0_amp,
            "tE": params.tE,
            "piE_E": params.piE[0],
            "piE_N": params.piE[1],
            "q": params.q,
            "sep": params.sep,
            "phi": params.phi,
        }
    )
    mag_jax = np.asarray(
        jax_physics.psbl_photometry_from_fitter_vec_jit(
            t,
            fitter_vec,
            mag_src=params.mag_src[0],
            b_sff=params.b_sff[0],
            root_tol=params.root_tol,
            parallax_vectors=pvec,
        )
    )
    np.testing.assert_allclose(mag_model, mag_jax, rtol=1e-10, atol=1e-8)


def test_pspl_trajectory_grad_wrt_piE():
    ra_l = 17.75
    dec_l = -29.0
    t = np.linspace(56900.0, 57100.0, 11)
    pvec = jax_physics.precompute_parallax_vectors(ra_l, dec_l, t)

    def amp_piE_E(piE_E):
        u0, thetaE_hat, _ = jax_physics.derive_pspl_static_geometry(0.05, piE_E, 0.02)
        return jax_physics.pspl_amplification(
            t, 57000.0, 45.0, u0, thetaE_hat,
            parallax_vectors=pvec, piE_E=piE_E, piE_N=0.02,
        )

    grad = jax.grad(lambda x: amp_piE_E(x).sum())(0.01)
    assert np.isfinite(float(grad))
