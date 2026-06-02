"""Fixtures for old-vs-JAX model parity tests."""
from __future__ import annotations

import inspect
import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from bagle.jax.layout_registry import resolve_layout

CANONICAL: dict[str, Any] = {
    "t0": 57100.0,
    "t0_geotr": 57100.0,
    "t0par": 57100.0,
    "u0_amp": 0.05,
    "u0_amp_com": 0.4,
    "u0_amp_geotr": 0.05,
    "tE": 45.0,
    "tE_geotr": 45.0,
    "log_tE": math.log10(45.0),
    "piE_E": 0.01,
    "piE_N": 0.02,
    "piE_E_geotr": 0.01,
    "piE_N_geotr": 0.02,
    "log_piE": math.log10(0.022360679775),
    "phi_muRel": 45.0,
    "piEN_piEE": 0.02,
    "b_sff": 0.8,
    "b_sff1": 0.8,
    "mag_src": 18.5,
    "mag_src_pri": 18.5,
    "mag_src_sec": 19.0,
    "mag_src1": 18.5,
    "mag_base": 18.5,
    "mag_base1": 18.5,
    "mL": 1.0,
    "beta": 0.4,
    "dL": 4000.0,
    "dS": 8000.0,
    "dL_dS": 0.5,
    "xS0_E": 0.0,
    "xS0_N": 0.0,
    "muS_E": 0.0,
    "muS_N": 0.0,
    "muL_E": 1.0,
    "muL_N": -1.0,
    "log10_thetaE": math.log10(0.8),
    "thetaE": 0.8,
    "piS": 0.15,
    "q": 0.3,
    "sep": 1.2,
    "phi": 45.0,
    "alpha": 45.0,
    "s2": 0.02,
    "rho": 0.01,
    "log_rho": math.log10(0.01),
    "root_tol": 1e-8,
    "aleph": 0.4,
    "aleph_sec": 0.8,
    "v_para": 0.01,
    "v_perp": 0.005,
    "v_rad": -0.02,
    "r_s": 0.5,
    "a_s": 1.0,
    "dmag_Lp_Ls": [0.0],
    "mLp": 10.0,
    "mLs": 3.0,
    "sepL": 3.0,
    "alphaL": -35.0,
    "sepS": 0.5,
    "alphaS": 0.0,
    "beta_com": 1.0,
    "beta_p": 0.4,
    "t0_p": 57100.0,
    "t0_com": 57100.0,
    "omega_pri": 90.0,
    "omegaL_pri": 90.0,
    "omegaS_pri": 90.0,
    "big_omega_sec": 0.0,
    "big_omegaL_sec": 0.0,
    "big_omegaS_sec": 0.0,
    "iL": 45.0,
    "tpL": 40.0,
    "eL": 0.1,
    "aL": 1.0,
    "iS": 45.0,
    "eS": 0.1,
    "pS": 3000.0,
    "p": 3000.0,
    "tpS": 40.0,
    "alephS": 0.4,
    "aleph_secS": 0.8,
    "i": 45.0,
    "e": 0.1,
    "tp": 40.0,
    "a": 1.0,
    "fratio_bin": [1.0],
    "radiusS": 1e-3,
    "n_outline": 20,
    "gp_log_sigma": [-1.0],
    "gp_log_rho": [0.5],
    "gp_rho": [math.exp(0.5)],
    "gp_log_S0": [-2.0],
    "gp_log_omega0": [0.0],
    "gp_log_omega04_S0": [-6.0],
    "gp_log_omega0_S0": [-2.0],
    "gp_log_jit_sigma": [math.log(0.02)],
}

PARALLAX_KW = dict(raL=259.5, decL=-29.0, obsLocation="earth")

PHOT_LIKELIHOOD_METHODS = frozenset(
    {"get_chi2_photometry", "log_likely_photometry_each"}
)
AST_LIKELIHOOD_METHODS = frozenset(
    {"get_chi2_astrometry", "log_likely_astrometry_each"}
)

SKIP_CLASS_SUBSTR = (
    "RefPar",
    "LumLens",
)

_GEO_PHOT_TO_HELIO = {
    "t0_geotr": "t0",
    "u0_amp_geotr": "u0_amp",
    "tE_geotr": "tE",
    "piE_E_geotr": "piE_E",
    "piE_N_geotr": "piE_N",
}


def _value_for(name: str) -> Any:
    if name in CANONICAL:
        val = CANONICAL[name]
    elif name.endswith("1") and name[:-1] in CANONICAL:
        v = CANONICAL[name[:-1]]
        val = [v] if name.startswith(("b_sff", "mag_", "gp_")) else v
    else:
        raise KeyError(f"No canonical value for parameter {name!r}")
    if name in (
        "b_sff",
        "mag_src",
        "mag_base",
        "mag_src_pri",
        "mag_src_sec",
    ) and not isinstance(val, (list, tuple)):
        return [val]
    return val


def _param_mixin_cls(model_module, class_name: str):
    cls = getattr(model_module, class_name)
    layout = resolve_layout(cls)
    if layout is None:
        raise ValueError(f"No layout for {class_name}")
    return getattr(model_module, layout.param_mixin)


def _init_mixin_cls(model_module, class_name: str):
    """Param mixin whose ``__init__`` matches the concrete model (includes GP args)."""
    cls = getattr(model_module, class_name)
    for base in cls.__mro__:
        if base.__name__ == class_name:
            continue
        if not hasattr(base, "phot_optional_param_names"):
            continue
        if not any(p.startswith("gp_") for p in base.phot_optional_param_names):
            continue
        sig = inspect.signature(base.__init__)
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
            continue
        return base
    return _param_mixin_cls(model_module, class_name)


def build_init_args(cls: type, model_module) -> tuple[list[Any], dict[str, Any]]:
    mixin = _init_mixin_cls(model_module, cls.__name__)
    sig = inspect.signature(mixin.__init__)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if pname in ("raL", "decL", "obsLocation"):
            continue
        val = _value_for(pname)
        if param.default is not inspect.Parameter.empty:
            kwargs[pname] = val
        else:
            args.append(val)
    if "_Par_" in cls.__name__:
        kwargs.update(PARALLAX_KW)
    return args, kwargs


def _post_init(instance):
    if getattr(instance, "astrometryFlag", False) and not getattr(instance, "photometryFlag", False):
        if not hasattr(instance, "b_sff"):
            instance.b_sff = [1.0]


def build_paired_instances(class_name: str):
    import bagle.model as ref_model
    import bagle.model_jax as jax_model

    if not hasattr(ref_model, class_name):
        return build_jax_eval_paired_instances(class_name)
    ref_cls = getattr(ref_model, class_name)
    jax_cls = getattr(jax_model, class_name)
    ref_args, ref_kw = build_init_args(ref_cls, ref_model)
    jax_args, jax_kw = build_init_args(jax_cls, jax_model)
    ref_inst = ref_cls(*ref_args, **ref_kw)
    jax_inst = jax_cls(*jax_args, **jax_kw)
    _post_init(ref_inst)
    _post_init(jax_inst)
    return ref_inst, jax_inst


def build_jax_eval_paired_instances(class_name: str):
    """Pair native host forward with ``jax/evaluate`` dispatch (jax-only classes)."""
    import bagle.model_jax as jax_model

    jax_cls = getattr(jax_model, class_name)
    jax_args, jax_kw = build_init_args(jax_cls, jax_model)
    native_inst = jax_cls(*jax_args, **jax_kw)
    eval_inst = jax_cls(*jax_args, **jax_kw)
    _post_init(native_inst)
    _post_init(eval_inst)
    return native_inst, eval_inst


def call_method_via_jax_eval(instance, method_name: str, t: np.ndarray):
    """Forward through ``jax/evaluate`` dispatch; fall back to native method."""
    from bagle.jax_model import (
        try_get_amplification,
        try_get_astrometry,
        try_get_astrometry_unlensed,
        try_get_centroid_shift,
        try_get_lens_astrometry,
        try_get_photometry,
        try_get_resolved_astrometry,
    )

    dispatch = {
        "get_photometry": try_get_photometry,
        "get_amplification": try_get_amplification,
        "get_astrometry": try_get_astrometry,
        "get_astrometry_unlensed": try_get_astrometry_unlensed,
        "get_lens_astrometry": try_get_lens_astrometry,
        "get_centroid_shift": try_get_centroid_shift,
        "get_resolved_astrometry": try_get_resolved_astrometry,
    }
    fn = dispatch.get(method_name)
    if fn is not None:
        out = fn(instance, t, filt_idx=0)
        if out is not None:
            return out
    return call_method(instance, method_name, t)


def pspl_non_gp_pairs() -> list[tuple[str, str]]:
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    return sorted(
        (c, m)
        for c, m in applicable_task_pairs(model_jax)
        if c.startswith("PSPL_")
        and "GP" not in c
        and not any(s in c for s in SKIP_CLASS_SUBSTR)
    )


GP_PHOT_METHODS = ("get_photometry", "get_amplification", "get_photometry_with_gp")


def pspl_gp_pairs() -> list[tuple[str, str]]:
    """PSPL GP parity: phot forward + GP + extended PSPL methods where applicable."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = sorted(
        {
            c
            for c, _ in applicable
            if c.startswith("PSPL_")
            and "GP" in c
            and not any(s in c for s in SKIP_CLASS_SUBSTR)
        }
    )
    methods = GP_PHOT_METHODS + PSPL_GP_EXTENDED_METHODS
    return sorted((c, m) for c in classes for m in methods if (c, m) in applicable)


def pspl_gp_param1_pairs() -> list[tuple[str, str]]:
    """Backward-compatible alias: GP ``get_photometry_with_gp`` only."""
    return [(c, m) for c, m in pspl_gp_pairs() if m == "get_photometry_with_gp"]


PSBL_PHOT_METHODS = ("get_photometry", "get_amplification")

PSBL_PHOTASTROM_AST_METHODS = (
    "get_astrometry",
    "get_astrometry_unlensed",
    "get_lens_astrometry",
    "get_centroid_shift",
    "get_resolved_astrometry",
    "get_resolved_lens_astrometry",
)

PSPL_GP_EXTENDED_METHODS = (
    "get_u",
    "get_chi2_photometry",
    "log_likely_photometry_each",
    "get_resolved_amplification",
    "get_source_astrometry_unlensed",
    "get_chi2_astrometry",
    "log_likely_astrometry_each",
)


def psbl_phot_pairs() -> list[tuple[str, str]]:
    """PSBL phot-only parity (static + keplerian orbit, no GP)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    classes = sorted(
        {
            c
            for c, _ in applicable_task_pairs(model_jax)
            if c.startswith("PSBL_Phot_")
            and "PhotAstrom" not in c
            and "GP" not in c
            and not any(s in c for s in SKIP_CLASS_SUBSTR)
        }
    )
    return [(c, m) for c in classes for m in PSBL_PHOT_METHODS]


def psbl_phot_first_pairs() -> list[tuple[str, str]]:
    """Backward-compatible alias for :func:`psbl_phot_pairs`."""
    return psbl_phot_pairs()


def psbl_photastrom_first_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom noPar Param1 phot + core astrometry parity."""
    return _psbl_photastrom_pairs_for_classes(("PSBL_PhotAstrom_noPar_Param1",))


def _psbl_photastrom_pairs_for_classes(class_names: tuple[str, ...]) -> list[tuple[str, str]]:
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    methods = PSBL_PHOT_METHODS + PSBL_PHOTASTROM_AST_METHODS
    return sorted(
        (c, m)
        for c in class_names
        for m in methods
        if (c, m) in applicable
    )


def psbl_photastrom_par_param1_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Par Param1 phot + astrometry parity."""
    return _psbl_photastrom_pairs_for_classes(("PSBL_PhotAstrom_Par_Param1",))


def psbl_photastrom_param2_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param2 (noPar + Par) phot + astrometry parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_Param2", "PSBL_PhotAstrom_Par_Param2")
    methods = PSBL_PHOT_METHODS + PSBL_PHOTASTROM_AST_METHODS
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_gp_param1_pairs() -> list[tuple[str, str]]:
    """PSBL GP Param1 phot forward + ``get_photometry_with_gp``."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_Phot_noPar_GP_Param1", "PSBL_Phot_Par_GP_Param1")
    return sorted(
        (c, m) for c in classes for m in GP_PHOT_METHODS if (c, m) in applicable
    )


def psbl_gp_photastrom_param2_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom GP Param2 phot, GP, and core astrometry parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_GP_Param2", "PSBL_PhotAstrom_Par_GP_Param2")
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
    )
    methods = GP_PHOT_METHODS + core_ast + PSBL_GP_EXTENDED_METHODS
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_photastrom_gp_param1_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom GP Param1 phot, GP, core astrom, and extended likelihoods."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_GP_Param1", "PSBL_PhotAstrom_Par_GP_Param1")
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
    )
    methods = GP_PHOT_METHODS + core_ast + PSBL_GP_EXTENDED_METHODS
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_gp_photastrom_param3_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom GP Param3 — no such classes in model_jax (empty harness)."""
    return []


def psbl_gp_param2_phot_pairs() -> list[tuple[str, str]]:
    """PSBL phot-only GP Param2 — classes do not exist (empty harness)."""
    return []


def bsbl_photastrom_param1_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom Param1 (noPar + Par) phot + core astrometry parity."""
    return _psbl_photastrom_pairs_for_classes(
        ("BSBL_PhotAstrom_noPar_Param1", "BSBL_PhotAstrom_Par_Param1")
    )


def bsbl_photastrom_param2_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom Param2 (noPar + Par) phot + core astrometry parity."""
    return _psbl_photastrom_pairs_for_classes(
        ("BSBL_PhotAstrom_noPar_Param2", "BSBL_PhotAstrom_Par_Param2")
    )


def psbl_photastrom_circorbs_param1_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom CircOrbs Param1 phot + core astrometry."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "PSBL_PhotAstrom_noPar_CircOrbs_Param1",
            "PSBL_PhotAstrom_Par_CircOrbs_Param1",
        )
    )


def psbl_photastrom_ellorbs_param1_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom EllOrbs Param1 phot + core astrometry."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "PSBL_PhotAstrom_noPar_EllOrbs_Param1",
            "PSBL_PhotAstrom_Par_EllOrbs_Param1",
        )
    )


def bspl_photastrom_gp_param1_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom GP Param1 phot, GP, and core astrometry (noPar + Par)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("BSPL_PhotAstrom_noPar_GP_Param1", "BSPL_PhotAstrom_Par_GP_Param1")
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
    )
    methods = GP_PHOT_METHODS + core_ast
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def bsbl_photastrom_circorbs_param1_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom CircOrbs Param1 phot + core astrometry (noPar + Par)."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "BSBL_PhotAstrom_noPar_CircOrbs_Param1",
            "BSBL_PhotAstrom_Par_CircOrbs_Param1",
        )
    )


def fspl_photastrom_param1_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param1 phot + core astrometry (noPar + Par)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("FSPL_PhotAstrom_noPar_Param1", "FSPL_PhotAstrom_Par_Param1")
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_lens_astrometry",)
    )
    methods = PSBL_PHOT_METHODS + core_ast
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


FSPL_PHOTASTROM_EXTENDED_METHODS = (
    "get_u",
    "get_chi2_photometry",
    "log_likely_photometry_each",
    "get_chi2_astrometry",
    "log_likely_astrometry_each",
)


def fspl_photastrom_param1_extended_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param1 extended likelihoods (host AMG phot/ast forward)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("FSPL_PhotAstrom_noPar_Param1", "FSPL_PhotAstrom_Par_Param1")
    return sorted(
        (c, m)
        for c in classes
        for m in FSPL_PHOTASTROM_EXTENDED_METHODS
        if (c, m) in applicable
    )


def fspl_photastrom_param2_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param2 phot + core astrometry (noPar + Par)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("FSPL_PhotAstrom_noPar_Param2", "FSPL_PhotAstrom_Par_Param2")
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_lens_astrometry",)
    )
    methods = PSBL_PHOT_METHODS + core_ast
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def fspl_phot_gp_param1_pairs() -> list[tuple[str, str]]:
    """FSPL phot GP Param1 — no ``FSPL_Phot_*_GP_Param1`` classes in model_jax."""
    return []


def fsbl_phot_param1_pairs() -> list[tuple[str, str]]:
    """FSBL phot-only Param1: jax evaluate dispatch vs native (model_jax only)."""
    return _fsbl_jax_eval_pairs(
        ("FSBL_Phot_noPar_Param1", "FSBL_Phot_Par_Param1")
    )


def _fsbl_jax_eval_pairs(
    class_names: tuple[str, ...],
    methods: tuple[str, ...] | None = None,
) -> list[tuple[str, str]]:
    """FSBL jax-only classes: native host forward vs ``jax/evaluate`` dispatch."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    meth = methods or PSBL_PHOT_METHODS
    return sorted(
        (c, m) for c in class_names for m in meth if (c, m) in applicable
    )


def fsbl_phot_ellorbs_param1_pairs() -> list[tuple[str, str]]:
    """FSBL phot EllOrbs Param1 jax-eval harness (noPar + Par)."""
    return _fsbl_jax_eval_pairs(
        (
            "FSBL_Phot_noPar_EllOrbs_Param1",
            "FSBL_Phot_Par_EllOrbs_Param1",
        )
    )


def fsbl_phot_circorbs_param1_pairs() -> list[tuple[str, str]]:
    """FSBL phot CircOrbs Param1 jax-eval harness (noPar + Par)."""
    return _fsbl_jax_eval_pairs(
        (
            "FSBL_Phot_noPar_CircOrbs_Param1",
            "FSBL_Phot_Par_CircOrbs_Param1",
        )
    )


def fsbl_photastrom_param1_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param1 phot + core astrometry (jax-eval only)."""
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
    )
    return _fsbl_jax_eval_pairs(
        ("FSBL_PhotAstrom_noPar_Param1", "FSBL_PhotAstrom_Par_Param1"),
        methods=PSBL_PHOT_METHODS + core_ast,
    )


def bsbl_phot_param1_pairs() -> list[tuple[str, str]]:
    """BSBL phot-only Param1 — no ``BSBL_Phot_*`` ModelClassABC in model_jax."""
    return []


def bspl_phot_param2_pairs() -> list[tuple[str, str]]:
    """BSPL phot-only Param2 — no ``BSPL_Phot_*_Param2`` classes in model_jax."""
    return []


def bsbl_photastrom_ellorbs_param2_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom EllOrbs Param2 phot + core astrometry."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "BSBL_PhotAstrom_noPar_EllOrbs_Param2",
            "BSBL_PhotAstrom_Par_EllOrbs_Param2",
        )
    )


def bspl_photastrom_ellorbs_param2_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom EllOrbs Param2 phot + core astrometry."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "BSPL_PhotAstrom_noPar_EllOrbs_Param2",
            "BSPL_PhotAstrom_Par_EllOrbs_Param2",
        )
    )


def fspl_photastrom_param1_grad_phot_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param1 phot grad smoke (host AMG finite-difference)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("FSPL_PhotAstrom_noPar_Param1", "FSPL_PhotAstrom_Par_Param1")
    return sorted(
        (c, m)
        for c in classes
        for m in ("get_photometry", "get_amplification")
        if (c, m) in applicable
    )


def bspl_phot_gp_param1_pairs() -> list[tuple[str, str]]:
    """BSPL phot-only GP Param1 parity (noPar + Par)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("BSPL_Phot_noPar_GP_Param1", "BSPL_Phot_Par_GP_Param1")
    return sorted(
        (c, m) for c in classes for m in GP_PHOT_METHODS if (c, m) in applicable
    )


def bspl_photastrom_param2_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom Param2 phot + core astrometry (noPar + Par)."""
    return _psbl_photastrom_pairs_for_classes(
        ("BSPL_PhotAstrom_noPar_Param2", "BSPL_PhotAstrom_Par_Param2")
    )


def psbl_photastrom_param4_phot_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param4 phot-only parity (noPar + Par)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_Param4", "PSBL_PhotAstrom_Par_Param4")
    return sorted(
        (c, m) for c in classes for m in PSBL_PHOT_METHODS if (c, m) in applicable
    )


def psbl_photastrom_param4_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param4 phot + full astrometry (companion-source init)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_Param4", "PSBL_PhotAstrom_Par_Param4")
    methods = PSBL_PHOT_METHODS + PSBL_PHOTASTROM_AST_METHODS
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_photastrom_param4_grad_phot_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param4 phot grad smoke (companion t0/u0 init names)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_Param4", "PSBL_PhotAstrom_Par_Param4")
    return sorted(
        (c, m)
        for c in classes
        for m in ("get_photometry",)
        if (c, m) in applicable
    )


def bsbl_photastrom_ellorbs_param1_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom EllOrbs Param1 phot + core astrometry (noPar + Par)."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "BSBL_PhotAstrom_noPar_EllOrbs_Param1",
            "BSBL_PhotAstrom_Par_EllOrbs_Param1",
        )
    )


def psbl_photastrom_circorbs_param2_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom CircOrbs Param2 phot + core astrometry."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "PSBL_PhotAstrom_noPar_CircOrbs_Param2",
            "PSBL_PhotAstrom_Par_CircOrbs_Param2",
        )
    )


def psbl_photastrom_ellorbs_param2_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom EllOrbs Param2 phot + core astrometry."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "PSBL_PhotAstrom_noPar_EllOrbs_Param2",
            "PSBL_PhotAstrom_Par_EllOrbs_Param2",
        )
    )


def fspl_phot_param2_pairs() -> list[tuple[str, str]]:
    """FSPL phot-only Param2 parity (noPar + Par)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("FSPL_Phot_noPar_Param2", "FSPL_Phot_Par_Param2")
    return sorted(
        (c, m) for c in classes for m in PSBL_PHOT_METHODS if (c, m) in applicable
    )


def bspl_phot_param1_pairs() -> list[tuple[str, str]]:
    """BSPL phot-only Param1 parity (noPar + Par)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("BSPL_Phot_noPar_Param1", "BSPL_Phot_Par_Param1")
    return sorted(
        (c, m) for c in classes for m in PSBL_PHOT_METHODS if (c, m) in applicable
    )


def psbl_phot_param2_pairs() -> list[tuple[str, str]]:
    """PSBL phot-only Param2 parity (noPar + Par, static/orbit as applicable).

    Note: the model hierarchy has no ``PSBL_Phot_*_Param2`` classes; this
    returns an empty list until such classes exist.
    """
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_Phot_noPar_Param2", "PSBL_Phot_Par_Param2")
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOT_METHODS
        if (c, m) in applicable
    )


def psbl_photastrom_param3_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param3 phot + full astrom (log10 thetaE layout)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_Param3", "PSBL_PhotAstrom_Par_Param3")
    methods = PSBL_PHOT_METHODS + PSBL_PHOTASTROM_AST_METHODS
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_phot_grad_pairs() -> list[tuple[str, str]]:
    """PSBL static phot-only classes for grad smoke (root finder => often NaN)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_Phot_noPar_Param1", "PSBL_Phot_Par_Param1")
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOT_METHODS
        if (c, m) in applicable
    )


def bspl_photastrom_param1_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom Param1 phot + core astrometry parity (noPar + Par)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("BSPL_PhotAstrom_noPar_Param1", "BSPL_PhotAstrom_Par_Param1")
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
    )
    methods = PSBL_PHOT_METHODS + core_ast
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_photastrom_param3_phot_pairs() -> list[tuple[str, str]]:
    """Backward-compatible alias: Param3 phot+amp only."""
    return [(c, m) for c, m in psbl_photastrom_param3_pairs() if m in PSBL_PHOT_METHODS]


def time_grid_phot(instance) -> np.ndarray:
    t0 = float(instance.t0)
    tE = float(instance.tE)
    return np.linspace(t0 - 3.0 * tE, t0 + 3.0 * tE, 80)


def time_grid_ast(instance, n: int = 60) -> np.ndarray:
    t0 = float(instance.t0)
    tE = float(instance.tE)
    return np.linspace(t0 - 3.0 * tE, t0 + 3.0 * tE, n)


def call_method(instance, method_name: str, t: np.ndarray):
    method = getattr(instance, method_name)
    sig = inspect.signature(method)
    kwargs: dict[str, Any] = {}
    if "filt_idx" in sig.parameters:
        kwargs["filt_idx"] = 0
    if method_name == "get_photometry_with_gp":
        mag = instance.get_photometry(t) if hasattr(instance, "get_photometry") else np.full_like(t, 18.5)
        err = np.full_like(t, 0.02)
        return method(t, mag, err, filt_idx=0, t_pred=t[:10])
    if method_name in PHOT_LIKELIHOOD_METHODS:
        mag = np.asarray(instance.get_photometry(t), dtype=np.float64)
        mag = mag + 0.05
        err = np.full_like(t, 0.02, dtype=np.float64)
        return method(t, mag, err, **kwargs)
    if method_name in AST_LIKELIHOOD_METHODS:
        pos = np.asarray(instance.get_astrometry(t), dtype=np.float64)
        pos = pos + np.array([0.001, 0.001])
        err = np.full_like(t, 0.001, dtype=np.float64)
        return method(t, pos[:, 0], pos[:, 1], err, err, **kwargs)
    return method(t, **kwargs)


INIT_PARAM_SKIP = frozenset({"self", "raL", "decL", "obsLocation"})
LIST_INIT_PARAMS = frozenset(
    {
        "b_sff",
        "mag_src",
        "mag_base",
        "mag_src_pri",
        "mag_src_sec",
        "dmag_Lp_Ls",
        "gp_log_sigma",
        "gp_log_rho",
        "gp_rho",
        "gp_log_S0",
        "gp_log_omega0",
        "gp_log_omega0_S0",
        "gp_log_omega04_S0",
        "gp_log_jit_sigma",
        "dmag_Lp_Ls",
        "fratio_bin",
    }
)

PSBL_GP_EXTENDED_METHODS = (
    "get_u",
    "get_chi2_photometry",
    "log_likely_photometry_each",
)


def numeric_init_param_names(jax_inst) -> tuple[str, ...]:
    """All numeric ``__init__`` parameters from the Param mixin (filter 0 for lists)."""
    import bagle.model_jax as model_module

    mixin = _init_mixin_cls(model_module, jax_inst.__class__.__name__)
    sig = inspect.signature(mixin.__init__)
    return tuple(p for p in sig.parameters if p not in INIT_PARAM_SKIP)


def pack_init_vector(instance) -> tuple[np.ndarray, tuple[str, ...]]:
    """Pack every numeric Param-mixin ``__init__`` argument into a 1D vector."""
    names = numeric_init_param_names(instance)
    vec = np.array([_scalar_from_instance(instance, n) for n in names], dtype=np.float64)
    return vec, names


def scatter_init_vector(instance, vec, init_names: tuple[str, ...]) -> None:
    """Write a packed init vector back onto a model instance (filter-0 lists)."""
    for i, name in enumerate(init_names):
        val = float(vec[i])
        if name in _ARRAY_COMPONENT:
            attr, idx = _ARRAY_COMPONENT[name]
            arr = np.array(getattr(instance, attr), dtype=np.float64, copy=True)
            arr.reshape(-1)[idx] = val
            setattr(instance, attr, arr)
        elif name in LIST_INIT_PARAMS:
            setattr(instance, name, [val])
        elif name == "thetaE":
            instance.thetaE_amp = val
        elif name == "log10_thetaE":
            instance.thetaE_amp = 10.0 ** val
        else:
            setattr(instance, name, val)


def _scalar_from_instance(instance, name: str) -> float:
    if name in _ARRAY_COMPONENT:
        return _fitter_scalar(instance, name)
    if name == "thetaE" and hasattr(instance, "thetaE_amp"):
        return float(instance.thetaE_amp)
    if name == "log10_thetaE" and hasattr(instance, "thetaE_amp"):
        return float(np.log10(instance.thetaE_amp))
    val = getattr(instance, name, None)
    if val is None and name in CANONICAL:
        val = CANONICAL[name]
    if val is None:
        raise AttributeError(f"{instance.__class__.__name__} has no attribute {name!r}")
    if isinstance(val, dict):
        return float(val[0])
    arr = np.asarray(val).reshape(-1)
    if name in LIST_INIT_PARAMS or arr.size > 1:
        return float(arr[0])
    if arr.size != 1:
        raise TypeError(f"cannot coerce {name!r} to scalar (shape {arr.shape})")
    return float(arr[0])


def _init_value_for_base_name(v, init_names: tuple[str, ...], base_name: str):
    if base_name in init_names:
        return v[init_names.index(base_name)]
    if base_name == "log10_thetaE" and "thetaE" in init_names:
        return jnp.log10(v[init_names.index("thetaE")])
    if base_name == "thetaE" and "log10_thetaE" in init_names:
        return jnp.power(10.0, v[init_names.index("log10_thetaE")])
    raise ValueError(
        f"cannot map base fitter {base_name!r} from init parameters {init_names!r}"
    )


def _base_vec_from_init(v, init_names: tuple[str, ...], base_names: tuple[str, ...]):
    return jnp.stack(
        [_init_value_for_base_name(v, init_names, n) for n in base_names]
    )


def _init_param(v, init_names: tuple[str, ...], name: str, default=None):
    if name not in init_names:
        return default
    return v[init_names.index(name)]


def _mag_from_init(v, init_names: tuple[str, ...], layout, b_sff):
    from bagle.jax.geometry import mag_src_from_fitter

    b_sff_j = jnp.asarray(b_sff, dtype=jnp.float64)
    if layout.mag_fitter == "mag_base" and "mag_base" in init_names:
        mag_v = _init_param(v, init_names, "mag_base")
        return mag_src_from_fitter(mag_v, "mag_base", b_sff_j)
    if "mag_src" in init_names:
        mag_v = _init_param(v, init_names, "mag_src")
        return mag_src_from_fitter(mag_v, "mag_src", b_sff_j)
    return None


_ARRAY_COMPONENT: dict[str, tuple[str, int]] = {
    "piE_E": ("piE", 0),
    "piE_N": ("piE", 1),
    "xS0_E": ("xS0", 0),
    "xS0_N": ("xS0", 1),
    "muS_E": ("muS", 0),
    "muS_N": ("muS", 1),
    "muL_E": ("muL", 0),
    "muL_N": ("muL", 1),
}


def pack_fitter_vector(instance) -> tuple[np.ndarray, list[str]]:
    layout = resolve_layout(instance.__class__)
    names = list(layout.base_fitter_names) if layout else list(instance.fitter_param_names)
    vec = np.array([_fitter_scalar(instance, n) for n in names])
    return vec, names


def _fitter_scalar(instance, name: str) -> float:
    if name in _ARRAY_COMPONENT:
        attr, idx = _ARRAY_COMPONENT[name]
        if hasattr(instance, attr):
            return float(np.asarray(getattr(instance, attr)).reshape(-1)[idx])
    if name == "thetaE" and hasattr(instance, "thetaE_amp"):
        return float(instance.thetaE_amp)
    if name == "log10_thetaE" and hasattr(instance, "thetaE_amp"):
        return float(np.log10(instance.thetaE_amp))
    val = getattr(instance, name, None)
    if val is None and name in CANONICAL:
        val = CANONICAL[name]
    if val is None:
        raise AttributeError(f"{instance.__class__.__name__} has no attribute {name!r}")
    arr = np.asarray(val).reshape(-1)
    if arr.size != 1:
        raise TypeError(f"cannot coerce {name!r} to scalar (shape {arr.shape})")
    return float(arr[0])


def _obs_location(instance) -> str:
    obs = getattr(instance, "obsLocation", "earth")
    if isinstance(obs, (list, tuple, np.ndarray)):
        return str(np.asarray(obs).reshape(-1)[0])
    return str(obs)


def _parallax_vectors(jax_inst, t: np.ndarray):
    if not getattr(jax_inst, "parallaxFlag", False):
        return None
    from bagle.jax_physics import precompute_parallax_vectors

    return precompute_parallax_vectors(
        float(jax_inst.raL),
        float(jax_inst.decL),
        t,
        obs_location=_obs_location(jax_inst),
    )


def _mag_scalar(jax_inst, layout) -> float:
    from bagle.jax.geometry import mag_src_from_fitter

    b_sff = float(np.asarray(jax_inst.b_sff).reshape(-1)[0])
    if layout.mag_fitter == "mag_base":
        mag_base = float(np.asarray(jax_inst.mag_base).reshape(-1)[0])
        return float(
            mag_src_from_fitter(
                jnp.asarray(mag_base), "mag_base", jnp.asarray(b_sff)
            )
        )
    return float(np.asarray(jax_inst.mag_src).reshape(-1)[0])


def _unpack_pspl_geom(eval_kind: str, names: tuple[str, ...], v):
    from bagle.jax.geometry import derive_geometry_from_layout

    out = derive_geometry_from_layout("", eval_kind, v, names)
    if isinstance(out[0], str) and out[0] == "pspl_phot":
        _, u0, thetaE_hat, tE, piE_E, piE_N = out
        p = {names[i]: v[i] for i in range(len(names))}
        return {
            "t0": p["t0"],
            "u0": u0,
            "thetaE_hat": thetaE_hat,
            "tE": tE,
            "piE_E": piE_E,
            "piE_N": piE_N,
        }
    (
        u0,
        thetaE_hat,
        tE,
        piE_E,
        piE_N,
        xS0,
        xL0,
        muS,
        muL,
        thetaE_amp,
        piS,
        piL,
    ) = out
    t0_idx = names.index("t0")
    return {
        "t0": v[t0_idx],
        "u0": u0,
        "thetaE_hat": thetaE_hat,
        "tE": tE,
        "piE_E": piE_E,
        "piE_N": piE_N,
        "xS0": xS0,
        "xL0": xL0,
        "muS": muS,
        "muL": muL,
        "thetaE_amp": thetaE_amp,
        "piS": piS,
        "piL": piL,
    }


def _linear_astrometry_jax(t_j, t0, x0, mu, parallax_vectors, pi):
    dt = ((t_j - t0) / 365.25).reshape(-1, 1)
    pos = x0.reshape(1, 2) + dt * mu.reshape(1, 2) * 1e-3
    if parallax_vectors is not None and pi is not None:
        pos = pos + jnp.asarray(pi, dtype=jnp.float64) * jnp.asarray(
            parallax_vectors, dtype=jnp.float64
        ) * 1e-3
    return pos


def _pspl_astrom_forward(method_name: str, geom: dict, t_j, t0, b_sff, pvec):
    from bagle.jax_physics import pspl_astrometry_param1

    xS0 = geom["xS0"]
    xL0 = geom["xL0"]
    muS = geom["muS"]
    muL = geom["muL"]
    thetaE_amp = geom["thetaE_amp"]
    piS = geom["piS"]
    piL = geom["piL"]
    if method_name == "get_astrometry":
        return pspl_astrometry_param1(
            t_j,
            t0,
            xS0,
            xL0,
            muS,
            muL,
            thetaE_amp,
            b_sff,
            parallax_vectors=pvec,
            piS=piS,
            piL=piL,
        )
    xL = _linear_astrometry_jax(t_j, t0, xL0, muL, pvec, piL)
    if method_name == "get_lens_astrometry":
        return xL
    xS = _linear_astrometry_jax(t_j, t0, xS0, muS, pvec, piS)
    if method_name == "get_astrometry_unlensed":
        b = jnp.asarray(b_sff, dtype=jnp.float64)
        return b * xS + (1.0 - b) * xL
    ast = pspl_astrometry_param1(
        t_j,
        t0,
        xS0,
        xL0,
        muS,
        muL,
        thetaE_amp,
        b_sff,
        parallax_vectors=pvec,
        piS=piS,
        piL=piL,
    )
    unl = jnp.asarray(b_sff, dtype=jnp.float64) * xS + (
        1.0 - jnp.asarray(b_sff, dtype=jnp.float64)
    ) * xL
    return (ast - unl) * 1e3


def _helio_geom_names(base_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_GEO_PHOT_TO_HELIO.get(n, n) for n in base_names)


def _base_vec_geoproj(v, init_names: tuple[str, ...], base_names: tuple[str, ...], ra, dec):
    from bagle.jax.frame_convert import geo_phot_to_helio_jax

    p = {init_names[i]: v[i] for i in range(len(init_names))}
    geo = jnp.stack(
        [
            p["t0_geotr"],
            p["u0_amp_geotr"],
            p["tE_geotr"],
            p["piE_E_geotr"],
            p["piE_N_geotr"],
            p["t0par"],
        ]
    )
    helio = geo_phot_to_helio_jax(geo, float(ra), float(dec))
    helio_map = dict(zip(("t0", "u0_amp", "tE", "piE_E", "piE_N"), helio))
    return jnp.stack([helio_map.get(_GEO_PHOT_TO_HELIO.get(n, n), p[n]) for n in base_names])


def _base_vec_for_layout(v, init_names, base_names, layout, jax_inst):
    if layout.geoproj:
        return _base_vec_geoproj(
            v, init_names, base_names, float(jax_inst.raL), float(jax_inst.decL)
        )
    return _base_vec_from_init(v, init_names, base_names)


def grad_smoke_jax(
    class_name: str,
    method_name: str,
    jax_inst,
    t: np.ndarray,
    *,
    return_names: bool = False,
):
    """Pure-JAX grad smoke w.r.t. all numeric Param-mixin ``__init__`` parameters."""
    import jax

    from bagle.jax.layout_registry import resolve_layout
    from bagle.jax.geometry import derive_geometry_from_layout
    from bagle.jax_physics import (
        gaussian_chi2_astrometry,
        gaussian_chi2_photometry,
        gaussian_log_likelihood_astrometry_each,
        gaussian_log_likelihood_photometry_each,
        psbl_all_arrays,
        psbl_complex_pos_static,
        psbl_photometry,
        psbl_total_amplification,
        pspl_amplification,
        pspl_astrometry_param1,
        pspl_photometry,
        pspl_resolved_amplification,
        pspl_resolved_amplification_from_u,
        pspl_resolved_astrometry,
        pspl_source_astrometry_unlensed,
        pspl_u,
    )

    layout = resolve_layout(jax_inst.__class__)
    if layout is None:
        raise NotImplementedError(f"no layout for {class_name}")

    base_names = tuple(layout.base_fitter_names)
    init_names = numeric_init_param_names(jax_inst)
    t_j = jnp.asarray(t, dtype=jnp.float64)
    pvec = _parallax_vectors(jax_inst, t)
    vec0 = jnp.array(pack_init_vector(jax_inst)[0], dtype=jnp.float64)
    ek = layout.eval_kind
    geom_names = _helio_geom_names(base_names)
    pspl_phot_kinds = (
        "pspl_phot_static",
        "pspl_phot_log",
        "pspl_photastrom_physical",
        "pspl_photastrom_reduced",
    )
    pspl_astrom_kinds = (
        "pspl_photastrom_physical",
        "pspl_photastrom_reduced",
        "pspl_astrom_reduced",
    )

    def _phot_obs():
        mag = jnp.asarray(jax_inst.get_photometry(t), dtype=jnp.float64)
        mag = mag + 0.05
        err = jnp.full_like(mag, 0.02, dtype=jnp.float64)
        return mag, err

    def _ast_obs():
        pos = jnp.asarray(jax_inst.get_astrometry(t), dtype=jnp.float64)
        pos = pos + jnp.array([0.001, 0.001], dtype=jnp.float64)
        err = jnp.full_like(pos[:, 0], 0.001, dtype=jnp.float64)
        return pos[:, 0], pos[:, 1], err, err

    def _pspl_geom(v):
        base = _base_vec_for_layout(v, init_names, base_names, layout, jax_inst)
        return _unpack_pspl_geom(ek, geom_names, base)

    if method_name in ("get_photometry", "get_amplification") and ek in (
        "pspl_phot_static",
        "pspl_phot_log",
        "pspl_photastrom_physical",
        "pspl_photastrom_reduced",
    ):
        geom_names = _helio_geom_names(base_names)

        def forward(v):
            base = _base_vec_for_layout(v, init_names, base_names, layout, jax_inst)
            geom = _unpack_pspl_geom(ek, geom_names, base)
            b_sff = _init_param(v, init_names, "b_sff", 1.0)
            mag = _mag_from_init(v, init_names, layout, b_sff)
            if mag is None:
                mag = jnp.asarray(_mag_scalar(jax_inst, layout), dtype=jnp.float64)
            if method_name == "get_amplification":
                out = pspl_amplification(
                    t_j,
                    geom["t0"],
                    geom["tE"],
                    geom["u0"],
                    geom["thetaE_hat"],
                    parallax_vectors=pvec,
                    piE_E=geom["piE_E"],
                    piE_N=geom["piE_N"],
                )
            else:
                out = pspl_photometry(
                    t_j,
                    geom["t0"],
                    geom["tE"],
                    geom["u0"],
                    geom["thetaE_hat"],
                    mag,
                    b_sff=b_sff,
                    parallax_vectors=pvec,
                    piE_E=geom["piE_E"],
                    piE_N=geom["piE_N"],
                )
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name == "get_photometry_with_gp" and layout.has_gp and (
        ek in pspl_phot_kinds or ek.startswith("psbl_phot")
    ):
        from bagle.jax.gp import _GP_QUALITY, _gp_has_fixed_jitter
        import tinygp
        from tinygp.kernels import quasisep as qk

        t_obs = t_j
        t_pred = t_j[: min(10, t_j.shape[0])]
        mag_obs = jnp.asarray(jax_inst.get_photometry(np.asarray(t)), dtype=jnp.float64)
        mag_err = jnp.full_like(mag_obs, 0.02, dtype=jnp.float64)
        log_jit = jnp.log(jnp.mean(mag_err))
        use_jit_param = "gp_log_jit_sigma" in init_names
        fixed_jitter = _gp_has_fixed_jitter(jax_inst)

        def _gp_log_rho(v):
            if "gp_log_rho" in init_names:
                return _init_param(v, init_names, "gp_log_rho", 0.5)
            return jnp.log(_init_param(v, init_names, "gp_rho", math.exp(0.5)))

        def _gp_log_S0(v):
            if "gp_log_S0" in init_names:
                return _init_param(v, init_names, "gp_log_S0", -2.0)
            if "gp_log_omega04_S0" in init_names:
                log_omega04 = _init_param(v, init_names, "gp_log_omega04_S0", -6.0)
                log_omega0 = _init_param(v, init_names, "gp_log_omega0", 0.0)
                return log_omega04 - 4.0 * log_omega0
            log_omega0_S0 = _init_param(v, init_names, "gp_log_omega0_S0", -2.0)
            log_omega0 = _init_param(v, init_names, "gp_log_omega0", 0.0)
            return log_omega0_S0 - log_omega0

        def forward(v):
            b_sff = _init_param(v, init_names, "b_sff", 1.0)
            mag = _mag_from_init(v, init_names, layout, b_sff)
            if mag is None:
                mag = jnp.asarray(_mag_scalar(jax_inst, layout), dtype=jnp.float64)
            log_sigma = _init_param(v, init_names, "gp_log_sigma", -1.0)
            log_rho = _gp_log_rho(v)
            log_S0 = _gp_log_S0(v)
            log_omega0 = _init_param(v, init_names, "gp_log_omega0", 0.0)
            sigma = jnp.exp(log_sigma)
            rho = jnp.exp(log_rho)
            S0 = jnp.exp(log_S0)
            omega0 = jnp.exp(log_omega0)
            if use_jit_param:
                jitter = jnp.exp(_init_param(v, init_names, "gp_log_jit_sigma", log_jit))
            elif fixed_jitter:
                jitter = jnp.exp(log_jit)
            else:
                jitter = jnp.asarray(0.0, dtype=jnp.float64)
            kernel = qk.Matern32(scale=rho, sigma=sigma) + qk.SHO(
                omega=omega0, quality=_GP_QUALITY, sigma=jnp.sqrt(S0)
            )
            diag = mag_err**2 + jitter**2
            root_tol = float(getattr(jax_inst, "root_tol", 1e-8))

            def mean_fn(x):
                x1 = jnp.atleast_1d(x)
                if ek.startswith("psbl_phot"):
                    base = _base_vec_from_init(v, init_names, base_names)
                    (
                        _tag,
                        u0,
                        thetaE_hat,
                        t0,
                        tE,
                        m1,
                        m2,
                        xL1,
                        xL2,
                        piE_E,
                        piE_N,
                    ) = derive_geometry_from_layout("", ek, base, base_names)
                    m = psbl_photometry(
                        x1,
                        t0,
                        tE,
                        u0,
                        thetaE_hat,
                        xL1,
                        xL2,
                        m1,
                        m2,
                        mag,
                        b_sff=b_sff,
                        parallax_vectors=pvec,
                        piE_E=piE_E,
                        piE_N=piE_N,
                        root_tol=root_tol,
                    )
                else:
                    geom = _pspl_geom(v)
                    m = pspl_photometry(
                        x1,
                        geom["t0"],
                        geom["tE"],
                        geom["u0"],
                        geom["thetaE_hat"],
                        mag,
                        b_sff=b_sff,
                        parallax_vectors=pvec,
                        piE_E=geom["piE_E"],
                        piE_N=geom["piE_N"],
                    )
                return m[0]

            gp = tinygp.GaussianProcess(kernel, t_obs, diag=diag, mean=mean_fn)
            cond = gp.condition(mag_obs, t_pred)
            return jnp.sum(cond.gp.loc)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name in ("get_photometry", "get_amplification") and ek.startswith(
        "psbl_phot"
    ):

        def forward(v):
            base = _base_vec_from_init(v, init_names, base_names)
            (
                _tag,
                u0,
                thetaE_hat,
                t0,
                tE,
                m1,
                m2,
                xL1,
                xL2,
                piE_E,
                piE_N,
            ) = derive_geometry_from_layout("", ek, base, base_names)
            b_sff = _init_param(v, init_names, "b_sff", 1.0)
            mag = _mag_from_init(v, init_names, layout, b_sff)
            if mag is None:
                mag = jnp.asarray(_mag_scalar(jax_inst, layout), dtype=jnp.float64)
            root_tol = float(getattr(jax_inst, "root_tol", 1e-8))
            if method_name == "get_photometry":
                out = psbl_photometry(
                    t_j,
                    t0,
                    tE,
                    u0,
                    thetaE_hat,
                    xL1,
                    xL2,
                    m1,
                    m2,
                    mag,
                    b_sff=b_sff,
                    parallax_vectors=pvec,
                    piE_E=piE_E,
                    piE_N=piE_N,
                    root_tol=root_tol,
                )
            else:
                w, z1, z2 = psbl_complex_pos_static(
                    t_j,
                    t0,
                    tE,
                    u0,
                    thetaE_hat,
                    xL1,
                    xL2,
                    parallax_vectors=pvec,
                    piE_E=piE_E,
                    piE_N=piE_N,
                )
                _, amp_arr = psbl_all_arrays(w, z1, z2, m1, m2, root_tol)
                out = psbl_total_amplification(amp_arr)
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name in (
        "get_astrometry",
        "get_astrometry_unlensed",
        "get_lens_astrometry",
        "get_centroid_shift",
    ) and ek in (
        "pspl_photastrom_physical",
        "pspl_photastrom_reduced",
        "pspl_astrom_reduced",
    ):
        default_b = float(np.asarray(getattr(jax_inst, "b_sff", [1.0])).reshape(-1)[0])
        geom_names = _helio_geom_names(base_names)

        def forward(v):
            base = _base_vec_for_layout(v, init_names, base_names, layout, jax_inst)
            geom = _unpack_pspl_geom(ek, geom_names, base)
            b_sff = _init_param(v, init_names, "b_sff", default_b)
            out = _pspl_astrom_forward(
                method_name, geom, t_j, geom["t0"], b_sff, pvec
            )
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name == "get_u" and ek in set(pspl_phot_kinds) | set(pspl_astrom_kinds):

        def forward(v):
            geom = _pspl_geom(v)
            u = pspl_u(
                t_j,
                geom["t0"],
                geom["tE"],
                geom["u0"],
                geom["thetaE_hat"],
                parallax_vectors=pvec,
                piE_E=geom["piE_E"],
                piE_N=geom["piE_N"],
            )
            return jnp.sum(u)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name == "get_resolved_amplification" and ek in pspl_phot_kinds:

        def forward(v):
            geom = _pspl_geom(v)
            amp = pspl_resolved_amplification(
                t_j,
                geom["t0"],
                geom["tE"],
                geom["u0"],
                geom["thetaE_hat"],
                parallax_vectors=pvec,
                piE_E=geom["piE_E"],
                piE_N=geom["piE_N"],
            )
            return jnp.sum(amp)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name in ("get_chi2_photometry", "log_likely_photometry_each") and ek in (
        pspl_phot_kinds
    ):
        mag_obs, mag_err = _phot_obs()

        def forward(v):
            geom = _pspl_geom(v)
            b_sff = _init_param(v, init_names, "b_sff", 1.0)
            mag = _mag_from_init(v, init_names, layout, b_sff)
            if mag is None:
                mag = jnp.asarray(_mag_scalar(jax_inst, layout), dtype=jnp.float64)
            mag_model = pspl_photometry(
                t_j,
                geom["t0"],
                geom["tE"],
                geom["u0"],
                geom["thetaE_hat"],
                mag,
                b_sff=b_sff,
                parallax_vectors=pvec,
                piE_E=geom["piE_E"],
                piE_N=geom["piE_N"],
            )
            if method_name == "get_chi2_photometry":
                out = gaussian_chi2_photometry(mag_model, mag_obs, mag_err)
            else:
                out = gaussian_log_likelihood_photometry_each(
                    mag_model, mag_obs, mag_err
                )
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name == "get_source_astrometry_unlensed" and ek in pspl_astrom_kinds:

        def forward(v):
            geom = _pspl_geom(v)
            pos = pspl_source_astrometry_unlensed(
                t_j,
                geom["t0"],
                geom["xS0"],
                geom["muS"],
                parallax_vectors=pvec,
                piS=geom["piS"],
            )
            return jnp.sum(pos)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name == "get_resolved_astrometry" and ek in pspl_astrom_kinds:

        def forward(v):
            geom = _pspl_geom(v)
            pos = pspl_resolved_astrometry(
                t_j,
                geom["t0"],
                geom["tE"],
                geom["u0"],
                geom["thetaE_hat"],
                geom["xL0"],
                geom["muL"],
                geom["thetaE_amp"],
                parallax_vectors=pvec,
                piE_E=geom["piE_E"],
                piE_N=geom["piE_N"],
                piL=geom["piL"],
            )
            return jnp.sum(pos)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name in ("get_chi2_astrometry", "log_likely_astrometry_each") and ek in (
        pspl_astrom_kinds
    ):
        x_obs, y_obs, x_err, y_err = _ast_obs()
        default_b = float(np.asarray(getattr(jax_inst, "b_sff", [1.0])).reshape(-1)[0])

        def forward(v):
            geom = _pspl_geom(v)
            b_sff = _init_param(v, init_names, "b_sff", default_b)
            pos_model = pspl_astrometry_param1(
                t_j,
                geom["t0"],
                geom["xS0"],
                geom["xL0"],
                geom["muS"],
                geom["muL"],
                geom["thetaE_amp"],
                b_sff,
                parallax_vectors=pvec,
                piS=geom["piS"],
                piL=geom["piL"],
            )
            if method_name == "get_chi2_astrometry":
                out = gaussian_chi2_astrometry(
                    pos_model, x_obs, y_obs, x_err, y_err
                )
            else:
                out = gaussian_log_likelihood_astrometry_each(
                    pos_model, x_obs, y_obs, x_err, y_err
                )
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if return_names:
            return g, init_names
        return g

    if method_name in ("get_photometry", "get_amplification") and ek.startswith(
        ("fsbl_phot", "fsbl_photastrom")
    ):
        from bagle.jax.fspl import (
            fspl_amplification_from_model,
            fspl_photometry_from_model,
        )

        pvec_np = np.asarray(pvec, dtype=np.float64)
        t_np = np.asarray(t, dtype=np.float64)
        vec0_np = np.asarray(vec0, dtype=np.float64)
        eps = 1e-5

        def _phot_sum(vec_np: np.ndarray) -> float:
            _, inst = build_paired_instances(class_name)
            scatter_init_vector(inst, vec_np, init_names)
            if method_name == "get_photometry":
                out = fspl_photometry_from_model(inst, t_np, 0, pvec_np)
            else:
                out = fspl_amplification_from_model(inst, t_np, 0, pvec_np)
            return float(np.sum(out))

        g = np.zeros(len(vec0_np), dtype=np.float64)
        for i in range(len(vec0_np)):
            vp = vec0_np.copy()
            vm = vec0_np.copy()
            vp[i] += eps
            vm[i] -= eps
            g[i] = (_phot_sum(vp) - _phot_sum(vm)) / (2.0 * eps)
        if return_names:
            return g, init_names
        return g

    raise NotImplementedError(f"grad smoke not wired for {class_name}.{method_name}")
