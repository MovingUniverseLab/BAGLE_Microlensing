"""``n_outline`` from the data dictionary, and the host FSPL limb kernel.

Items 3.1 and 3.2: ``data['n_outline']`` reaches the model constructor and
the traced likelihood, default 50, one compile per fit.

Item 3.3: the ``model_jax`` host forward model uses the same clustered-limb
kernel as the traced likelihood. ``model.py`` is unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import bagle.model_jax as model_jax
from bagle.jax.fspl import fspl_images_amps, fspl_magnification
from bagle.model_fitter_jax import (
    MicrolensSolver,
    build_explicit_jax_loglik_fn,
)
from test_swap_parity import (
    _construct,
    _epochs,
    _init_values,
    _make_fitter,
    _synthetic_data,
    _t0_of,
)

# Photometry-only classes in item 3.2.
_FSPL_PHOT = (
    "FSPL_Phot_noPar_Param1",
    "FSPL_Phot_Par_Param1",
    "FSPL_Phot_noPar_Param2",
    "FSPL_Phot_Par_Param2",
    "FSPL_Limb_Phot_noPar_Param1",
    "FSPL_Limb_Phot_Par_Param1",
    "FSPL_Limb_Phot_noPar_Param2",
    "FSPL_Limb_Phot_Par_Param2",
)


def _fit(class_name, tmp_path, n_outline=None):
    """
    Build a one-filter solver for one FSPL class.

    Parameters
    ----------
    class_name : str
    tmp_path : path-like
    n_outline : int or None
        Stored on the data dictionary when not None.

    Returns
    -------
    fitter : MicrolensSolver
    values : dict
    data : dict
    """
    cls = getattr(model_jax, class_name)
    values = _init_values(cls, model_jax)
    inst = _construct(cls, model_jax, values)
    t0 = _t0_of(values)
    t = _epochs(t0, 12, 40.0)
    rng = np.random.default_rng(0)
    data = _synthetic_data(inst, class_name, t, rng)
    if n_outline is not None:
        data["n_outline"] = int(n_outline)

    fitter = _make_fitter(MicrolensSolver, data, cls, tmp_path)

    return fitter, values, data


def test_n_outline_reaches_model(tmp_path):
    """``data['n_outline']`` is a fixed parameter; missing means 50."""
    fitter, values, _data = _fit(
        "FSPL_Phot_noPar_Param1", tmp_path, n_outline=100)
    assert fitter.get_model(values).n_outline == 100

    fitter_default, values_default, _data = _fit(
        "FSPL_Phot_noPar_Param1", tmp_path / "default")
    assert fitter_default.get_model(values_default).n_outline == 50

    return None


def test_n_outline_reaches_traced_lnl(tmp_path):
    """Traced lnL uses ``data['n_outline']``, not the old hard-coded 400."""
    cls = model_jax.FSPL_Phot_noPar_Param1
    fitter, values, data = _fit(
        "FSPL_Phot_noPar_Param1", tmp_path, n_outline=100)
    fn, _ctx = build_explicit_jax_loglik_fn(fitter)
    vec = fitter.to_param_vector(fitter.to_param_dict(values))
    traced = float(fn(vec))

    names = list(cls.fitter_param_names)
    packed = dict(zip(names, np.asarray(vec)))
    direct_100 = float(cls.jax_log_likely_photometry(
        vec, data["t_phot1"], data["mag1"], data["mag_err1"],
        packed["b_sff"], packed["mag_src"], n_outline=100))
    direct_50 = float(cls.jax_log_likely_photometry(
        vec, data["t_phot1"], data["mag1"], data["mag_err1"],
        packed["b_sff"], packed["mag_src"], n_outline=50))

    assert abs(traced - direct_100) < 1e-6
    assert abs(traced - direct_50) > 1e-6

    return None


@pytest.mark.parametrize("class_name", _FSPL_PHOT)
def test_fspl_traced_matches_host_same_n(class_name, tmp_path):
    """Traced and host photometry lnL agree at ``n_outline`` 50."""
    fitter, values, _data = _fit(class_name, tmp_path, n_outline=50)
    fn, _ctx = build_explicit_jax_loglik_fn(fitter)
    vec = fitter.to_param_vector(fitter.to_param_dict(values))
    traced = float(fn(vec))
    host = float(fitter.log_likely(values))

    assert abs(traced - host) < 1e-6

    return None


def test_n_outline_no_retrace(tmp_path):
    """Twenty likelihood calls with one ``n_outline`` compile once."""
    fitter, values, _data = _fit(
        "FSPL_Phot_noPar_Param1", tmp_path, n_outline=50)
    fn, _ctx = build_explicit_jax_loglik_fn(fitter)
    base = np.asarray(
        fitter.to_param_vector(fitter.to_param_dict(values)), dtype=float)

    jax.clear_caches()
    fn(base)
    size_after_first = fspl_images_amps._cache_size()

    for i in range(19):
        fn(base + 1e-8 * i)

    assert fspl_images_amps._cache_size() == size_after_first

    return None


def test_host_fspl_amplification_uses_jax_kernel():
    """Host FSPL amplification is the clustered-limb kernel (item 3.3)."""
    cls = model_jax.FSPL_Phot_noPar_Param1
    values = _init_values(cls, model_jax)
    values["n_outline"] = 50
    inst = _construct(cls, model_jax, values)
    t = _epochs(_t0_of(values), 15, 40.0)

    host = np.asarray(inst.get_amplification(t), dtype=float)
    kernel = np.asarray(fspl_magnification(
        inst.get_u(t), inst.source_radius_thetaE_units(), 50))

    np.testing.assert_allclose(host, kernel, rtol=0.0, atol=1e-6)

    return None
