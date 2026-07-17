"""Param-mixin capability coverage (replaces layout registry gate)."""
from __future__ import annotations

import inspect
import re

import pytest

from bagle.jax.likelihood import _infer_loglik_mode, _param_mixin_class


_PSPL_JAX_PARAM_MIXINS = (
    "PSPL_PhotParam1",
    "PSPL_PhotParam2",
    "PSPL_PhotParam3",
    "PSPL_PhotAstromParam1",
    "PSPL_PhotAstromParam2",
    "PSPL_PhotAstromParam3",
    "PSPL_AstromParam3",
)


def _param_mixin_classes():
    import bagle.model_jax as model

    out = []
    for name, cls in inspect.getmembers(model, inspect.isclass):
        if cls.__module__ != model.__name__:
            continue
        if "Param" not in name or name == "PSPL_Param":
            continue
        if not getattr(cls, "fitter_param_names", None):
            continue
        out.append((name, cls))
    return sorted(out, key=lambda x: x[0])


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
def test_concrete_model_has_param_mixin(class_name, model_cls):
    mixin = _param_mixin_class(model_cls)
    assert mixin is not None, f"{class_name} has no Param mixin with get_params_for_jax"
    assert hasattr(mixin, "get_params_for_jax")
    assert tuple(mixin.fitter_param_names)
    mode = _infer_loglik_mode(mixin)
    assert mode in ("phot", "joint", "ast"), f"{class_name} mode={mode}"


@pytest.mark.parametrize("mixin_name,cls", _param_mixin_classes())
def test_param_mixin_has_get_params_for_jax(mixin_name, cls):
    assert hasattr(cls, "get_params_for_jax")
    assert getattr(cls, "jax_loglik_backend", None) == "analytic"
    assert cls.fitter_param_names
    # Flags always present on Param mixins
    assert hasattr(cls, "paramPhotFlag")
    assert hasattr(cls, "paramAstromFlag")


@pytest.mark.parametrize("mixin_name", _PSPL_JAX_PARAM_MIXINS)
def test_pspl_param_mixin_packing(mixin_name):
    import bagle.model_jax as model
    import numpy as np

    cls = getattr(model, mixin_name)
    assert hasattr(cls, "get_params_for_jax_from_self")
    n = len(cls.fitter_param_names)
    vec = np.linspace(0.1, 1.0, n)
    # Avoid zeros that blow up geometry (piE amp, etc.)
    for i, name in enumerate(cls.fitter_param_names):
        if "piE" in name or name in ("u0_amp", "tE", "thetaE", "mL", "dL", "dL_dS", "piS"):
            vec[i] = max(vec[i], 0.2)
        if name == "log10_thetaE":
            vec[i] = 0.0
        if name.startswith("log_"):
            vec[i] = -1.0
    packed = cls.get_params_for_jax(vec)
    assert isinstance(packed, dict)
    assert packed


@pytest.mark.parametrize(
    "class_name",
    [
        "PSPL_Phot_noPar_Param1",
        "PSPL_Phot_Par_Param2",
        "PSPL_PhotAstrom_noPar_Param1",
        "PSPL_PhotAstrom_Par_Param2",
        "PSPL_PhotAstrom_noPar_Param3",
    ],
)
def test_pspl_leaf_discovers_param_mixin(class_name):
    import bagle.model_jax as model

    leaf = getattr(model, class_name)
    mixin = _param_mixin_class(leaf)
    assert mixin is not None
    assert hasattr(mixin, "get_params_for_jax")
    assert _infer_loglik_mode(mixin) in ("phot", "joint", "ast")


def test_phot_param_names_for_phot_classes():
    import bagle.model_jax as model

    for name, cls in _concrete_model_classes():
        mixin = _param_mixin_class(cls)
        if mixin is None:
            continue
        if getattr(mixin, "paramPhotFlag", False):
            assert list(getattr(mixin, "phot_param_names", [])), (
                f"{name}/{mixin.__name__} missing phot_param_names"
            )


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


def test_no_layout_registry_module():
    with pytest.raises(ModuleNotFoundError):
        __import__("bagle.jax.layout_registry")


def test_no_evaluate_module():
    with pytest.raises(ModuleNotFoundError):
        __import__("bagle.jax.evaluate")
