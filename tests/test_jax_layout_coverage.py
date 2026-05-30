"""Coverage gate: every Phot / PhotAstrom / Astrom class resolves to a JAX layout."""
from __future__ import annotations

import inspect
import re

import pytest

from bagle.jax.layout_registry import LAYOUT_BY_PARAM_MIXIN, check_param_mixin_parity, resolve_layout


def _concrete_model_classes():
    import bagle.model_jax as model

    out = []
    for name, cls in inspect.getmembers(model, inspect.isclass):
        if cls.__module__ != model.__name__:
            continue
        if not re.search(r"_(Phot|PhotAstrom|Astrom)_", name):
            continue
        if name.startswith(("PSPL_", "PSBL_", "BSPL_", "FSPL_", "FSBL_", "BSBL_")):
            if "Param" in name and name.endswith(tuple(f"Param{i}" for i in range(10))):
                continue
            out.append((name, cls))
    return sorted(out, key=lambda x: x[0])


@pytest.mark.parametrize("class_name,model_cls", _concrete_model_classes())
def test_resolve_layout(class_name, model_cls):
    layout = resolve_layout(model_cls)
    assert layout is not None, f"{class_name} has no JAX layout via Param mixin MRO"
    assert layout.base_fitter_names
    assert layout.eval_kind


def test_param_mixin_registry_nonempty():
    assert len(LAYOUT_BY_PARAM_MIXIN) >= 30


def test_param_mixin_parity_reference_vs_jax():
    check_param_mixin_parity()


def test_supports_jax_loglik_pspl_phot():
    pytest.importorskip("pytensor")
    from bagle import jax_physics, model_jax as model
    from bagle.jax.likelihood import supports_jax_loglik_for_fitter
    from bagle.model_fitter import MicrolensSolver

    t = __import__("numpy").linspace(57000, 57200, 10)
    data = {
        "t_phot1": t,
        "mag1": 18.0 + 0.01 * __import__("numpy").sin(t / 50),
        "mag_err1": 0.02,
    }
    fitter = MicrolensSolver(data, model.PSPL_Phot_noPar_Param1, verbose=False)
    assert supports_jax_loglik_for_fitter(fitter) is not None
    assert jax_physics.supports_jax_phot_loglik(fitter) == "pspl"
