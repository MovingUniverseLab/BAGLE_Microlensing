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
    psbl_phot_first_pairs,
    pspl_non_gp_pairs,
    time_grid_ast,
    time_grid_phot,
)

RTOL = ATOL = 1e-6
PHOT_METHODS = {"get_photometry", "get_amplification", "get_photometry_with_gp"}


def _time_grid(method_name: str, instance):
    if method_name in PHOT_METHODS:
        return time_grid_phot(instance)
    return time_grid_ast(instance)


@pytest.mark.parametrize("class_name,method_name", pspl_non_gp_pairs())
def test_parity_old_vs_jax(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    ref = np.asarray(call_method(old_inst, method_name, t), dtype=np.float64)
    test = np.asarray(call_method(jax_inst, method_name, t), dtype=np.float64)
    assert ref.shape == test.shape, f"shape mismatch {ref.shape} vs {test.shape}"
    np.testing.assert_allclose(test, ref, rtol=RTOL, atol=ATOL)


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


@pytest.mark.parametrize("class_name,method_name", psbl_phot_first_pairs())
def test_parity_psbl_first(class_name, method_name):
    old_inst, jax_inst = build_paired_instances(class_name)
    t = _time_grid(method_name, old_inst)
    ref = np.asarray(call_method(old_inst, method_name, t), dtype=np.float64)
    test = np.asarray(call_method(jax_inst, method_name, t), dtype=np.float64)
    assert ref.shape == test.shape
    np.testing.assert_allclose(test, ref, rtol=RTOL, atol=ATOL)
