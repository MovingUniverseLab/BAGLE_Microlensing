"""API parity for jax_swap_plan Step 5.

Every class defined in both ``model.py`` and ``model_jax.py`` must
expose the reference public names, and a call written for ``model.py``
must still bind on ``model_jax``. Output checks cover the two
astrometry helpers and ``FSPL.cent``.

Astrometry helpers are compared at ``atol=1e-8`` (arcsec). On the
canonical grid that includes finite-source classes; the largest
residual is about 1e-12 (PSTL photometry). The limb-sampler
difference (item 3.3) is in the traced likelihood, not in these
host forward methods. ``FSPL.cent`` matches at 1e-8 when the lens is
at the origin. With the lens offset, ``model.py`` ignores that
position and the comparison is not held to 1e-8.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("jax")

import bagle.model as model_np
import bagle.model_jax as model_jax
from bagle.jax.migration_tasks import discover_concrete_classes
from model_old_vs_jax_fixtures import build_paired_instances

# Arcsec. Holds for finite-source classes on this grid (measured
# residuals are ~1e-18 for FSPL limb centroids, ~1e-12 for PSTL).
_ATOL = 1e-8

# model.py's unlensed astrometry reads xS0. These phot-only FSPL
# classes have no sky position, so the reference call cannot run.
_FSPL_PHOT_NO_XS0 = frozenset({
    "FSPL_Limb_Phot_Par_Param1",
    "FSPL_Limb_Phot_Par_Param2",
    "FSPL_Limb_Phot_noPar_Param1",
    "FSPL_Limb_Phot_noPar_Param2",
    "FSPL_Phot_Par_Param1",
    "FSPL_Phot_Par_Param2",
    "FSPL_Phot_noPar_Param1",
    "FSPL_Phot_noPar_Param2",
})

# Step 3 added an optional n_outline on these JAX mixins. model.py
# is frozen and does not have it. Calls that omit it still bind.
_N_OUTLINE_EXTRA = {
    "BFSPL_Limb_PhotAstromParam1",
    "BFSPL_Limb_PhotParam1",
    "BFSPL_PhotAstromParam1",
    "BFSPL_PhotParam1",
}

# Derived Param7 quantities that must agree (point source, 1e-8).
_PARAM7_ATTRS = (
    "thetaE_amp",
    "piE_amp",
    "piE",
    "piE_E",
    "piE_N",
    "mL",
    "piL",
    "piRel",
    "muRel",
    "muL",
    "xL0",
    "mag_src",
    "beta",
)


def _module_classes(mod):
    """Classes defined in ``mod``, not ones it imported.

    Parameters
    ----------
    mod : module

    Returns
    -------
    classes : dict
        Name to class.
    """
    found = {}
    for name, obj in vars(mod).items():
        if inspect.isclass(obj) and obj.__module__ == mod.__name__:
            found[name] = obj

    return found


def _public_names(cls):
    """Public attribute names, including inherited ones.

    Parameters
    ----------
    cls : type

    Returns
    -------
    names : set of str
    """
    return {name for name in dir(cls) if not name.startswith("_")}


def _signature_is_callable_as_reference(ref_cls, jax_cls):
    """True when a ``model.py`` call binds on ``model_jax``.

    Shared parameters keep their order, kind, and default. Extra JAX
    parameters must have defaults. The known ``n_outline`` extra is
    the only one allowed.

    Parameters
    ----------
    ref_cls, jax_cls : type

    Returns
    -------
    ok : bool
    detail : str
        Empty when ``ok`` is True.
    """
    ref_sig = inspect.signature(ref_cls.__init__)
    jax_sig = inspect.signature(jax_cls.__init__)
    ref_params = [p for p in ref_sig.parameters.values() if p.name != "self"]
    jax_params = [p for p in jax_sig.parameters.values() if p.name != "self"]
    jax_names = [p.name for p in jax_params]
    ref_names = [p.name for p in ref_params]
    jax_by_name = {p.name: p for p in jax_params}

    # Reference names are a subsequence of the JAX names.
    last = -1
    for param in ref_params:
        if param.name not in jax_by_name:
            return False, f"missing parameter {param.name}"
        pos = jax_names.index(param.name)
        if pos < last:
            return False, f"{param.name} is out of order"
        last = pos
        other = jax_by_name[param.name]
        if param.kind != other.kind or param.default != other.default:
            return False, (
                f"{param.name} kind/default "
                f"{param.kind}/{param.default!r} vs "
                f"{other.kind}/{other.default!r}"
            )

    extras = [p for p in jax_params if p.name not in set(ref_names)]
    allowed = set()
    if ref_cls.__name__ in _N_OUTLINE_EXTRA:
        allowed.add("n_outline")
    for param in extras:
        if param.name in allowed and param.default is not inspect.Parameter.empty:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # *args, **kwargs on both sides is the concrete-class wrapper.
            ref_kinds = {p.kind for p in ref_params}
            if param.kind in ref_kinds:
                continue
        if param.default is inspect.Parameter.empty:
            return False, f"required extra parameter {param.name}"
        return False, f"unexpected extra parameter {param.name}"

    return True, ""


def _epochs(t0):
    """Ten epochs spanning 200 days, including ``t0``.

    Parameters
    ----------
    t0 : float
        Peak time (MJD).

    Returns
    -------
    t : ndarray, shape (10,)
    """
    return np.linspace(t0 - 100.0, t0 + 100.0, 10)


def _astrometry_params():
    """Concrete classes, skipping FSPL phot models with no ``xS0``.

    Returns
    -------
    params : list
    """
    reason = (
        "model.py get_astrometry_unlensed reads xS0; "
        "these FSPL phot-only models have no sky position"
    )
    params = []
    for name in _CONCRETE_SHARED:
        if name in _FSPL_PHOT_NO_XS0:
            params.append(pytest.param(
                name, marks=pytest.mark.skip(reason=reason)))
        else:
            params.append(name)

    return params


_NP_CLASSES = _module_classes(model_np)
_JAX_CLASSES = _module_classes(model_jax)
_SHARED_CLASS_NAMES = sorted(set(_NP_CLASSES) & set(_JAX_CLASSES))
_CONCRETE_SHARED = sorted(
    {name for name, _cls in discover_concrete_classes(model_np)}
    & {name for name, _cls in discover_concrete_classes(model_jax)}
)


@pytest.mark.parametrize("class_name", _SHARED_CLASS_NAMES)
def test_public_api_matches(class_name):
    """JAX public names contain the reference names, and calls bind."""
    ref_cls = _NP_CLASSES[class_name]
    jax_cls = _JAX_CLASSES[class_name]
    missing = sorted(_public_names(ref_cls) - _public_names(jax_cls))
    assert missing == [], f"{class_name} missing {missing}"

    ok, detail = _signature_is_callable_as_reference(ref_cls, jax_cls)
    assert ok, f"{class_name}: {detail}"

    return None


def test_ported_method_signatures():
    """The three ported methods take the same arguments as model.py."""
    pairs = (
        ("get_astrometry_unlensed_noparallax", model_np.PSPL, model_jax.PSPL),
        ("get_astrometry_pm_removed", model_np.PSPL, model_jax.PSPL),
        ("cent", model_np.FSPL, model_jax.FSPL),
    )
    for name, ref_cls, jax_cls in pairs:
        ref_sig = inspect.signature(getattr(ref_cls, name))
        jax_sig = inspect.signature(getattr(jax_cls, name))
        assert ref_sig == jax_sig, f"{name}: {ref_sig} vs {jax_sig}"

    return None


@pytest.mark.parametrize("class_name", _astrometry_params())
def test_astrometry_methods_match(class_name):
    """Compare the two PSPL astrometry helpers at 1e-8 arcsec."""
    ref, jax = build_paired_instances(class_name)
    t0 = float(getattr(ref, "t0", getattr(ref, "t0_com", 57100.0)))
    t = _epochs(t0)

    nopar_ref = ref.get_astrometry_unlensed_noparallax(t)
    nopar_jax = jax.get_astrometry_unlensed_noparallax(t)
    np.testing.assert_allclose(nopar_jax, nopar_ref, rtol=0.0, atol=_ATOL)

    # Unlensed residual is parallax only. No limb sampling.
    unlensed_ref = ref.get_astrometry_pm_removed(t, unlensed=True)
    unlensed_jax = jax.get_astrometry_pm_removed(t, unlensed=True)
    np.testing.assert_allclose(
        unlensed_jax, unlensed_ref, rtol=0.0, atol=_ATOL)

    # Lensed residual includes the image centroid.
    lensed_ref = ref.get_astrometry_pm_removed(t, unlensed=False)
    lensed_jax = jax.get_astrometry_pm_removed(t, unlensed=False)
    np.testing.assert_allclose(
        lensed_jax, lensed_ref, rtol=0.0, atol=_ATOL)

    return None


def test_fspl_phot_helpers_run_without_xS0():
    """JAX evaluates the helpers for an FSPL phot model with no ``xS0``.

    ``model.py`` raises ``AttributeError`` on that class. The JAX
    unlensed path treats a missing sky position as a phot-only model.
    """
    _ref, jax = build_paired_instances("FSPL_Phot_noPar_Param1")
    t = _epochs(float(jax.t0))
    nopar = np.asarray(jax.get_astrometry_unlensed_noparallax(t), dtype=float)
    lensed = np.asarray(
        jax.get_astrometry_pm_removed(t, unlensed=False), dtype=float)
    assert nopar.shape == (len(t), 2)
    assert lensed.shape == (len(t), 2)
    assert np.all(np.isfinite(nopar))
    assert np.all(np.isfinite(lensed))

    return None


def _image_positions():
    """Complex image positions off the critical curve.

    Returns
    -------
    ims : ndarray, complex, shape (3, 2)
        Two images at three epochs. ``|z|`` is away from 1.
    """
    ims = np.array([
        [1.6 + 0.3j, -0.55 + 0.15j],
        [1.3 - 0.4j, -0.80 - 0.25j],
        [2.0 + 0.0j, -0.45 + 0.05j],
    ])

    return ims


def test_fspl_cent_lens_at_origin():
    """With the lens at the origin both centroids use ``1 - 1/|z|^4``."""
    ims = _image_positions()
    ref = model_np.FSPL.__new__(model_np.FSPL)
    jax = model_jax.FSPL.__new__(model_jax.FSPL)
    got_ref = ref.cent(ims, 0.0)
    got_jax = jax.cent(ims, 0.0)
    np.testing.assert_allclose(got_jax, got_ref, rtol=0.0, atol=_ATOL)

    return None


def test_fspl_cent_lens_offset_uses_jax_jacobian():
    """An offset lens is weighted by ``|z - z1|``, not by ``|z|``.

    ``model.py`` drops the lens position, so this is not an ``atol=1e-8``
    comparison against that centroid.
    """
    ims = _image_positions()
    z1 = 0.25 + 0.10j
    jax = model_jax.FSPL.__new__(model_jax.FSPL)
    det = np.abs(jax.detJac(ims, z1))
    expect = np.sum(ims / det, axis=1) / np.sum(1.0 / det, axis=1)
    got = jax.cent(ims, z1)
    np.testing.assert_allclose(got, expect, rtol=0.0, atol=1e-12)

    ref = model_np.FSPL.__new__(model_np.FSPL)
    got_ref = ref.cent(ims, z1)
    # The two formulas disagree once the lens leaves the origin.
    assert np.max(np.abs(np.asarray(got) - np.asarray(got_ref))) > _ATOL

    return None


@pytest.mark.parametrize(
    "class_name",
    ["PSPL_PhotAstrom_noPar_Param7", "PSPL_PhotAstrom_Par_Param7"],
)
def test_param7_derived_quantities(class_name):
    """Param7 amplitudes, masses, and positions match model.py."""
    ref, jax = build_paired_instances(class_name)
    for name in _PARAM7_ATTRS:
        np.testing.assert_allclose(
            np.asarray(getattr(jax, name), dtype=float),
            np.asarray(getattr(ref, name), dtype=float),
            rtol=0.0,
            atol=_ATOL,
        )

    return None
