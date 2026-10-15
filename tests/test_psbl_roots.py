"""Numerical regression tests for PSBL quintic root solvers."""
import numpy as np
import pytest

from bagle import model
import bagle.model_jax as model_jax


def _sort_roots(z_row):
    """Sort one row of complex roots for order-independent comparison."""
    z = np.asarray(z_row, dtype=np.complex128)
    order = np.lexsort((z.imag, z.real))
    return z[order]


def _assert_roots_allclose(z_ref, z_test, rtol=1e-9, atol=1e-11):
    """Compare root arrays allowing permutation within each row."""
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
        np.testing.assert_allclose(
            ref[valid].real, tst[valid].real, rtol=rtol, atol=atol,
            err_msg=f"row {row}: real part mismatch",
        )
        np.testing.assert_allclose(
            ref[valid].imag, tst[valid].imag, rtol=rtol, atol=atol,
            err_msg=f"row {row}: imag part mismatch",
        )


@pytest.fixture
def psbl():
    inst = model.PSBL()
    inst.root_tol = 1e-8
    inst.m1 = 0.3
    inst.m2 = 0.7
    return inst


@pytest.fixture
def psbl_jax():
    inst = model_jax.PSBL()
    inst.root_tol = 1e-8
    inst.m1 = 0.3
    inst.m2 = 0.7
    return inst


@pytest.fixture
def reference_grid():
    """Source positions spanning weak-field, caustic-adjacent, and limb-like paths."""
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


def test_get_image_pos_arr_matches_mpsolve(psbl, reference_grid):
    """Default production path should agree with numpy.roots reference."""
    w, z1, z2, m1, m2 = reference_grid
    z_default = psbl.get_image_pos_arr(w, z1, z2, m1, m2, check_sols=False)
    z_mpsolve = psbl.get_image_pos_arr_mpsolve(w, z1, z2, m1, m2, check_sols=False)
    _assert_roots_allclose(z_mpsolve, z_default)


def test_get_image_pos_arr_fast_matches_mpsolve(psbl_jax, reference_grid):
    """Jitted fast solver should agree with numpy.roots reference."""
    w, z1, z2, m1, m2 = reference_grid
    z_fast = np.asarray(
        psbl_jax.get_image_pos_arr_fast(w, z1, z2, m1, m2, check_sols=False)
    )
    z_mpsolve = psbl_jax.get_image_pos_arr_mpsolve(
        w, z1, z2, m1, m2, check_sols=False
    )
    _assert_roots_allclose(z_mpsolve, z_fast)


def test_get_image_pos_arr_jax_matches_mpsolve(psbl_jax, reference_grid):
    """Non-jitted JAX vmap solver should agree with numpy.roots reference."""
    w, z1, z2, m1, m2 = reference_grid
    z_jax = np.asarray(
        psbl_jax.get_image_pos_arr_jax(w, z1, z2, m1, m2, check_sols=False)
    )
    z_mpsolve = psbl_jax.get_image_pos_arr_mpsolve(
        w, z1, z2, m1, m2, check_sols=False
    )
    _assert_roots_allclose(z_mpsolve, z_jax)


def test_get_image_pos_arr_solution_checking(psbl, reference_grid):
    """Solution masking with check_sols=True should be consistent across solvers."""
    w, z1, z2, m1, m2 = reference_grid
    z_default = psbl.get_image_pos_arr(w, z1, z2, m1, m2, check_sols=True)
    z_mpsolve = psbl.get_image_pos_arr_mpsolve(w, z1, z2, m1, m2, check_sols=True)
    _assert_roots_allclose(z_mpsolve, z_default)


@pytest.mark.skipif(
    not hasattr(model, "vb_sg_roots_quintic_coeffs_high_to_low"),
    reason="VBM Skowron-Gould solver not available",
)
def test_get_image_pos_arr_fast_matches_vbm(psbl_jax, reference_grid):
    """Fast solver should agree with VBM reference when available."""
    w, z1, z2, m1, m2 = reference_grid
    z_fast = np.asarray(
        psbl_jax.get_image_pos_arr_fast(w, z1, z2, m1, m2, check_sols=False)
    )
    z_vbm = psbl_jax.get_image_pos_arr_vbm(w, z1, z2, m1, m2, check_sols=False)
    _assert_roots_allclose(z_vbm, z_fast)


def test_fsbl_images_on_source_limb_uses_fast_solver():
    """FSBL limb sampling should run without error using the jitted root solver."""
    fsbl = model_jax.FSBL()
    fsbl.root_tol = 1e-8
    fsbl.m1 = 0.3
    fsbl.m2 = 0.7

    w0 = 0.05 + 0.02j
    z1 = -0.6 + 0.0j
    z2 = 0.6 + 0.0j
    rho = 0.02

    z, z_mask, z_parity, theta = fsbl.images_on_source_limb(
        w0, z1, z2, fsbl.m1, fsbl.m2, rho, npts_limb=32, niter=2
    )

    assert z.shape[0] == 5
    assert z.shape[1] == theta.shape[0]
    assert z_mask.shape == z.shape
    assert z_parity.shape == z.shape
    assert np.any(z_mask)
