"""Old NumPy reference vs model_jax parity and autodiff smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import bagle.model_jax as model_jax
from model_old_vs_jax_fixtures import (
    build_paired_instances,
    call_method,
    grad_smoke_jax,
    bspl_phot_param1_pairs,
    psbl_gp_param1_pairs,
    psbl_gp_photastrom_param2_pairs,
    psbl_phot_pairs,
    psbl_photastrom_first_pairs,
    psbl_photastrom_gp_param1_pairs,
    psbl_photastrom_param2_pairs,
    psbl_photastrom_param3_phot_pairs,
    psbl_photastrom_par_param1_pairs,
    pspl_gp_pairs,
    pspl_non_gp_pairs,
    time_grid_ast,
    time_grid_phot,
)

RTOL = ATOL = 1e-6
GP_STD_RTOL = GP_STD_ATOL = 1e-5
PHOT_METHODS = {"get_photometry", "get_amplification", "get_photometry_with_gp"}


def _time_grid(method_name: str, instance):
    if method_name in PHOT_METHODS:
        return time_grid_phot(instance)
    return time_grid_ast(instance)


def _assert_parity(old_inst, jax_inst, method_name: str, t: np.ndarray):
    ref_out = call_method(old_inst, method_name, t)
    test_out = call_method(jax_inst, method_name, t)
    if method_name == "get_photometry_with_gp":
        ref_mean, ref_std = ref_out
        test_mean, test_std = test_out
        assert ref_mean.shape == test_mean.shape
        assert ref_std.shape == test_std.shape
        np.testing.assert_allclose(test_mean, ref_mean, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(test_std, ref_std, rtol=GP_STD_RTOL, atol=GP_STD_ATOL)
        return
    ref = np.asarray(ref_out, dtype=np.float64)
    test = np.asarray(test_out, dtype=np.float64)
    assert ref.shape == test.shape, f"shape mismatch {ref.shape} vs {test.shape}"
    if method_name == "get_resolved_astrometry":
        mask = np.isfinite(ref) & np.isfinite(test)
        assert mask.any(), f"no overlapping finite values for {method_name}"
        np.testing.assert_allclose(test[mask], ref[mask], rtol=RTOL, atol=ATOL)
        return
    np.testing.assert_allclose(test, ref, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("class_name,method_name", pspl_non_gp_pairs())
def test_parity_old_vs_jax(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", pspl_non_gp_pairs())
def test_grad_old_vs_jax(class_name, method_name):
    _, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, jax_inst)
    g, init_names = grad_smoke_jax(
        class_name, method_name, jax_inst, t, return_names=True
    )
    assert len(g) == len(init_names), (
        f"grad length {len(g)} != init param count {len(init_names)} "
        f"for {class_name}.{method_name}"
    )
    assert np.all(np.isfinite(g)), f"non-finite grad for {class_name}.{method_name}"
    assert np.linalg.norm(g) > 0.0, f"zero grad norm for {class_name}.{method_name}"


@pytest.mark.parametrize("class_name,method_name", pspl_gp_pairs())
def test_parity_pspl_gp(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", pspl_gp_pairs())
def test_grad_pspl_gp(class_name, method_name):
    _, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, jax_inst)
    g, init_names = grad_smoke_jax(
        class_name, method_name, jax_inst, t, return_names=True
    )
    assert len(g) == len(init_names)
    assert np.all(np.isfinite(g)), f"non-finite grad for {class_name}.{method_name}"
    assert np.linalg.norm(g) > 0.0, f"zero grad norm for {class_name}.{method_name}"


@pytest.mark.parametrize("class_name,method_name", psbl_phot_pairs())
def test_parity_psbl_phot(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", psbl_photastrom_first_pairs())
def test_parity_psbl_photastrom_first(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", psbl_photastrom_par_param1_pairs())
def test_parity_psbl_photastrom_par_param1(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", psbl_photastrom_param2_pairs())
def test_parity_psbl_photastrom_param2(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", psbl_gp_param1_pairs())
def test_parity_psbl_gp_param1(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", psbl_photastrom_param3_phot_pairs())
def test_parity_psbl_photastrom_param3_phot(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", psbl_gp_photastrom_param2_pairs())
def test_parity_psbl_gp_photastrom_param2(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", bspl_phot_param1_pairs())
def test_parity_bspl_phot_param1(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)


@pytest.mark.parametrize("class_name,method_name", psbl_photastrom_gp_param1_pairs())
def test_parity_psbl_photastrom_gp_param1(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    _assert_parity(old_inst, jax_inst, method_name, t)
