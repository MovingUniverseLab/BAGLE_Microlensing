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
    "t0_prim": 57100.0,
    "u0_amp_prim": 0.4,
    "piEN_piEE": 2.0,
    "delta_muL_sec_E": 0.01,
    "delta_muL_sec_N": -0.01,
    "delta_muS_sec_E": 0.01,
    "delta_muS_sec_N": -0.01,
    "accLsec_E": 1e-4,
    "accLsec_N": -1e-4,
    "accSsec_E": 1e-4,
    "accSsec_N": -1e-4,
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
    "aS": 1.0,
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
    "log_a": math.log10(1.0),
    "mass_source_p": 1.0,
    "mass_source_s": 1.0,
    "fratio_bin": [1.0],
    "radiusS": 1e-3,
    "radiusS_pri": 1e-3,
    "radiusS_sec": 1e-3,
    "n_outline": 20,
    "n_outline_pri": 20,
    "n_outline_sec": 20,
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
        if pname in ("mag_src_pri", "mag_src_sec") and cls.__name__.startswith(
            "BFSPL"
        ):
            if isinstance(val, (list, tuple)):
                val = val[0]
        if param.default is not inspect.Parameter.empty:
            kwargs[pname] = val
        else:
            args.append(val)
    if "_Par_" in cls.__name__:
        kwargs.update(PARALLAX_KW)
    return args, kwargs


def _ensure_bsbl_primary_source_frame(instance) -> None:
    """Wire ``t0_pri`` / secondary source frame when orbit init omits them."""
    if hasattr(instance, "t0_pri") or not hasattr(instance, "t0"):
        return
    instance.t0_pri = instance.t0
    if hasattr(instance, "xS0"):
        instance.xS0_pri = instance.xS0
    if hasattr(instance, "u0_amp"):
        instance.u0_amp_pri = instance.u0_amp
    if hasattr(instance, "u0"):
        instance.u0_pri = instance.u0
    if not all(
        hasattr(instance, name)
        for name in ("sepS", "alphaS_rad", "u0_hat", "thetaE_amp", "xS0_pri")
    ):
        return
    sepS_vec = instance.sepS * np.array(
        (np.sin(instance.alphaS_rad), np.cos(instance.alphaS_rad))
    )
    instance.u0_amp_sec = instance.u0_amp_pri + (
        np.dot(sepS_vec, instance.u0_hat) / instance.thetaE_amp
    )
    instance.u0_sec = instance.u0_amp_sec * instance.u0_hat
    instance.xS0_sec = instance.xS0_pri + (sepS_vec * 1e-3)


def _ensure_per_filter_arrays(instance) -> None:
    """Wrap scalar photometry attrs so ``[filt_idx]`` indexing works."""
    for name in ("mag_src_pri", "mag_src_sec", "mag_src", "mag_base", "b_sff"):
        if not hasattr(instance, name):
            continue
        val = getattr(instance, name)
        if isinstance(val, (int, float)):
            setattr(instance, name, np.array([val]))
        elif isinstance(val, (list, tuple)):
            setattr(instance, name, np.asarray(val))


def _post_init(instance):
    _ensure_bsbl_primary_source_frame(instance)
    _ensure_per_filter_arrays(instance)
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


def call_method_via_jax_eval(
    instance,
    method_name: str,
    t: np.ndarray,
    *,
    fixed_phot: tuple[np.ndarray, np.ndarray] | None = None,
    fixed_ast: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
):
    """Forward through ``jax/evaluate`` dispatch; fall back to native method."""
    from bagle.jax_model import (
        try_get_amplification,
        try_get_astrometry,
        try_get_astrometry_unlensed,
        try_get_centroid_shift,
        try_get_chi2_astrometry,
        try_get_chi2_photometry,
        try_get_lens_astrometry,
        try_get_log_likely_astrometry_each,
        try_get_log_likely_photometry_each,
        try_get_photometry,
        try_get_resolved_astrometry,
        try_get_u,
    )

    dispatch = {
        "get_photometry": try_get_photometry,
        "get_amplification": try_get_amplification,
        "get_astrometry": try_get_astrometry,
        "get_astrometry_unlensed": try_get_astrometry_unlensed,
        "get_lens_astrometry": try_get_lens_astrometry,
        "get_centroid_shift": try_get_centroid_shift,
        "get_resolved_astrometry": try_get_resolved_astrometry,
        "get_u": try_get_u,
    }
    fn = dispatch.get(method_name)
    if fn is not None:
        out = fn(instance, t, filt_idx=0)
        if out is not None:
            return out
    if method_name in PHOT_LIKELIHOOD_METHODS:
        if fixed_phot is not None:
            mag, err = fixed_phot
        else:
            mag, err = synthetic_phot_obs(instance, t)
        phot_fn = {
            "get_chi2_photometry": try_get_chi2_photometry,
            "log_likely_photometry_each": try_get_log_likely_photometry_each,
        }.get(method_name)
        if phot_fn is not None:
            out = phot_fn(instance, t, mag, err, filt_idx=0)
            if out is not None:
                return out
    if method_name in AST_LIKELIHOOD_METHODS:
        if fixed_ast is not None:
            x_obs, y_obs, x_err, y_err = fixed_ast
        else:
            x_obs, y_obs, x_err, y_err = synthetic_ast_obs(instance, t)
        ast_fn = {
            "get_chi2_astrometry": try_get_chi2_astrometry,
            "log_likely_astrometry_each": try_get_log_likely_astrometry_each,
        }.get(method_name)
        if ast_fn is not None:
            out = ast_fn(
                instance, t, x_obs, y_obs, x_err, y_err, filt_idx=0
            )
            if out is not None:
                return out
    return call_method(
        instance,
        method_name,
        t,
        fixed_phot=fixed_phot,
        fixed_ast=fixed_ast,
    )


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
    methods = GP_PHOT_METHODS + PSPL_GP_EXTENDED_METHODS + PSBL_PHOTASTROM_AST_METHODS
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

PSBL_PHOTASTROM_LIKELIHOOD_METHODS = (
    "get_u",
    "get_chi2_photometry",
    "log_likely_photometry_each",
    "get_chi2_astrometry",
    "log_likely_astrometry_each",
)

PSBL_PHOTASTROM_PHOT_LIKELIHOOD_METHODS = (
    "get_u",
    "get_chi2_photometry",
    "log_likely_photometry_each",
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


def _psbl_photastrom_full_pairs_for_classes(
    class_names: tuple[str, ...],
) -> list[tuple[str, str]]:
    """PSBL PhotAstrom phot + astrom + likelihood parity for named classes."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    methods = (
        PSBL_PHOT_METHODS
        + PSBL_PHOTASTROM_AST_METHODS
        + PSBL_PHOTASTROM_LIKELIHOOD_METHODS
    )
    return sorted(
        (c, m)
        for c in class_names
        for m in methods
        if (c, m) in applicable and hasattr(model_jax, c)
    )


def _psbl_photastrom_orbit_param1_pairs(orbit: str) -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param1 with keplerian orbit (noPar + Par)."""
    no_par = f"PSBL_PhotAstrom_noPar_{orbit}_Param1"
    par = f"PSBL_PhotAstrom_Par_{orbit}_Param1"
    return _psbl_photastrom_full_pairs_for_classes((no_par, par))


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


_BSBL_PHOT_LIKELIHOOD_NO_U = (
    "get_chi2_photometry",
    "log_likely_photometry_each",
)


def bsbl_photastrom_param1_phot_likelihood_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom Param1 phot chi2 / log-likelihood / ``get_u``."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("BSBL_PhotAstrom_noPar_Param1", "BSBL_PhotAstrom_Par_Param1")
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOTASTROM_PHOT_LIKELIHOOD_METHODS
        if (c, m) in applicable
    )


def bsbl_photastrom_linorbs_param1_likelihood_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom LinOrbs Param1 likelihood parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "BSBL_PhotAstrom_noPar_LinOrbs_Param1",
        "BSBL_PhotAstrom_Par_LinOrbs_Param1",
    )
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOTASTROM_LIKELIHOOD_METHODS
        if (c, m) in applicable
    )


def bsbl_photastrom_accorbs_param1_likelihood_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom AccOrbs Param1 likelihood parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "BSBL_PhotAstrom_noPar_AccOrbs_Param1",
        "BSBL_PhotAstrom_Par_AccOrbs_Param1",
    )
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOTASTROM_LIKELIHOOD_METHODS
        if (c, m) in applicable
    )


def psbl_photastrom_param5_likelihood_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Par Param5 likelihood parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_Par_Param5",)
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOTASTROM_LIKELIHOOD_METHODS
        if (c, m) in applicable
    )


def bsbl_photastrom_param2_phot_likelihood_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom Param2 phot chi2 / log-likelihood (no ``get_u``)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("BSBL_PhotAstrom_noPar_Param2", "BSBL_PhotAstrom_Par_Param2")
    return sorted(
        (c, m)
        for c in classes
        for m in _BSBL_PHOT_LIKELIHOOD_NO_U
        if (c, m) in applicable
    )


def psbl_photastrom_circorbs_param1_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom CircOrbs Param1 phot + astrom + likelihoods."""
    return _psbl_photastrom_orbit_param1_pairs("CircOrbs")


def psbl_photastrom_ellorbs_param1_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom EllOrbs Param1 phot + astrom + likelihoods."""
    return _psbl_photastrom_orbit_param1_pairs("EllOrbs")


def psbl_photastrom_accorbs_param1_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom AccOrbs Param1 phot + astrom + likelihoods."""
    return _psbl_photastrom_orbit_param1_pairs("AccOrbs")


def psbl_photastrom_linorbs_param1_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom LinOrbs Param1 phot + astrom + likelihoods."""
    return _psbl_photastrom_orbit_param1_pairs("LinOrbs")


def psbl_photastrom_param7_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param7 static + orbit variants (phot/ast/phot likelihood)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = sorted(
        {
            c
            for c, _ in applicable
            if c.startswith("PSBL_PhotAstrom_")
            and "Param7" in c
            and "GP" not in c
            and not any(s in c for s in SKIP_CLASS_SUBSTR)
            and hasattr(model_jax, c)
        }
    )
    methods = (
        PSBL_PHOT_METHODS
        + PSBL_PHOTASTROM_AST_METHODS
        + PSBL_PHOTASTROM_PHOT_LIKELIHOOD_METHODS
    )
    return sorted((c, m) for c in classes for m in methods if (c, m) in applicable)


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


_BSPL_GP_ORBIT_CLASS_PREFIXES = (
    "BSPL_PhotAstrom_noPar_GP_AccOrbs_Param",
    "BSPL_PhotAstrom_Par_GP_AccOrbs_Param",
    "BSPL_PhotAstrom_noPar_GP_LinOrbs_Param",
    "BSPL_PhotAstrom_Par_GP_LinOrbs_Param",
    "BSPL_PhotAstrom_noPar_GP_Param2",
    "BSPL_PhotAstrom_Par_GP_Param2",
    "BSPL_PhotAstrom_noPar_GP_Param3",
    "BSPL_PhotAstrom_Par_GP_Param3",
)


def bspl_photastrom_gp_orbit_and_param23_pairs() -> list[tuple[str, str]]:
    """BSPL GP LinOrbs/AccOrbs Param1-3 and GP Param2/3 phot/GP/core astrometry."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    class_names = {
        c
        for c, _ in applicable
        if any(c.startswith(p) for p in _BSPL_GP_ORBIT_CLASS_PREFIXES)
    }
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
    )
    methods = GP_PHOT_METHODS + core_ast
    return sorted(
        (c, m) for c in class_names for m in methods if (c, m) in applicable
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


def fspl_photastrom_param2_extended_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param2 extended likelihoods (host AMG phot/ast forward)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("FSPL_PhotAstrom_noPar_Param2", "FSPL_PhotAstrom_Par_Param2")
    return sorted(
        (c, m)
        for c in classes
        for m in FSPL_PHOTASTROM_EXTENDED_METHODS
        if (c, m) in applicable
    )


def fspl_photastrom_param2_resolved_astrometry_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param2 ``get_resolved_astrometry`` parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("FSPL_PhotAstrom_noPar_Param2", "FSPL_PhotAstrom_Par_Param2")
    return sorted(
        (c, "get_resolved_astrometry")
        for c in classes
        if (c, "get_resolved_astrometry") in applicable
    )


def fspl_phot_param2_extended_pairs() -> list[tuple[str, str]]:
    """FSPL phot-only Param2 ``get_u`` and chi2 photometry."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("FSPL_Phot_noPar_Param2", "FSPL_Phot_Par_Param2")
    methods = ("get_u", "get_chi2_photometry", "log_likely_photometry_each")
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def fspl_photastrom_param2_grad_phot_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param2 phot grad smoke (host AMG finite-difference)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("FSPL_PhotAstrom_noPar_Param2", "FSPL_PhotAstrom_Par_Param2")
    return sorted(
        (c, m)
        for c in classes
        for m in ("get_photometry", "get_amplification")
        if (c, m) in applicable
    )


def _psbl_resolved_lens_pairs_for_classes(class_names: tuple[str, ...]) -> list[tuple[str, str]]:
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    return sorted(
        (c, "get_resolved_lens_astrometry")
        for c in class_names
        if (c, "get_resolved_lens_astrometry") in applicable
    )


def psbl_photastrom_param1_resolved_lens_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param1/2 ``get_resolved_lens_astrometry`` parity."""
    return _psbl_resolved_lens_pairs_for_classes(
        (
            "PSBL_PhotAstrom_noPar_Param1",
            "PSBL_PhotAstrom_Par_Param1",
            "PSBL_PhotAstrom_noPar_Param2",
            "PSBL_PhotAstrom_Par_Param2",
        )
    )


def bsbl_photastrom_param1_resolved_lens_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom Param1 ``get_resolved_lens_astrometry`` parity."""
    return _psbl_resolved_lens_pairs_for_classes(
        ("BSBL_PhotAstrom_noPar_Param1", "BSBL_PhotAstrom_Par_Param1")
    )


def psbl_photastrom_param5_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param5 (Par only) phot + full astrometry."""
    return _psbl_photastrom_pairs_for_classes(("PSBL_PhotAstrom_Par_Param5",))


def psbl_photastrom_param6_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param6 static + orbit variants (noPar + Par)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = sorted(
        {
            c
            for c, _ in applicable
            if c.startswith("PSBL_PhotAstrom_")
            and "Param6" in c
            and "GP" not in c
            and not any(s in c for s in SKIP_CLASS_SUBSTR)
        }
    )
    methods = PSBL_PHOT_METHODS + PSBL_PHOTASTROM_AST_METHODS
    return sorted((c, m) for c in classes for m in methods if (c, m) in applicable)


def bsbl_photastrom_gp_param1_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom GP Param1 — no BSBL GP classes in model_jax."""
    return []


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


def _fsbl_photastrom_orbit_param1_pairs(orbit: str) -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param1 with keplerian orbit (jax-eval)."""
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
    )
    no_par = f"FSBL_PhotAstrom_noPar_{orbit}_Param1"
    par = f"FSBL_PhotAstrom_Par_{orbit}_Param1"
    return _fsbl_jax_eval_pairs(
        (no_par, par),
        methods=PSBL_PHOT_METHODS + core_ast,
    )


def fsbl_photastrom_linorbs_param1_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom LinOrbs Param1 phot + core astrometry (jax-eval)."""
    return _fsbl_photastrom_orbit_param1_pairs("LinOrbs")


def fsbl_photastrom_accorbs_param1_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom AccOrbs Param1 phot + core astrometry (jax-eval)."""
    return _fsbl_photastrom_orbit_param1_pairs("AccOrbs")


def fsbl_photastrom_circorbs_param1_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom CircOrbs Param1 phot + core astrometry (jax-eval)."""
    return _fsbl_photastrom_orbit_param1_pairs("CircOrbs")


def fsbl_photastrom_ellorbs_param1_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom EllOrbs Param1 phot + core astrometry (jax-eval)."""
    return _fsbl_photastrom_orbit_param1_pairs("EllOrbs")


def fsbl_photastrom_orbit_param1_extended_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom orbit Param1 extended + resolved astrometry (jax-eval)."""
    classes: list[str] = []
    for orb in ("LinOrbs", "AccOrbs", "CircOrbs", "EllOrbs"):
        classes.extend(
            (
                f"FSBL_PhotAstrom_noPar_{orb}_Param1",
                f"FSBL_PhotAstrom_Par_{orb}_Param1",
            )
        )
    methods = FSPL_PHOTASTROM_EXTENDED_METHODS + (
        "get_resolved_astrometry",
        "get_resolved_lens_astrometry",
    )
    return _fsbl_jax_eval_pairs(tuple(classes), methods=methods)


def fsbl_photastrom_param12_resolved_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param1/2 resolved astrometry (jax-eval)."""
    return _fsbl_jax_eval_pairs(
        (
            "FSBL_PhotAstrom_noPar_Param1",
            "FSBL_PhotAstrom_Par_Param1",
            "FSBL_PhotAstrom_noPar_Param2",
            "FSBL_PhotAstrom_Par_Param2",
        ),
        methods=("get_resolved_astrometry", "get_resolved_lens_astrometry"),
    )


def fsbl_phot_ellorbs_param2_pairs() -> list[tuple[str, str]]:
    """FSBL phot EllOrbs Param2 — no such ModelClassABC in model_jax."""
    return []


def fsbl_photastrom_param2_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param2 phot + core astrometry (jax-eval)."""
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
    )
    return _fsbl_jax_eval_pairs(
        ("FSBL_PhotAstrom_noPar_Param2", "FSBL_PhotAstrom_Par_Param2"),
        methods=PSBL_PHOT_METHODS + core_ast,
    )


def _fsbl_photastrom_param3plus_class_names() -> tuple[str, ...]:
    """FSBL PhotAstrom Param3+ classes (base + orbit variants)."""
    import re

    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = {c for c, _ in applicable_task_pairs(model_jax)}
    out: set[str] = set()
    for c in applicable:
        if not c.startswith("FSBL_PhotAstrom_"):
            continue
        m = re.search(r"_Param(\d+)$", c)
        if m is None or int(m.group(1)) < 3:
            continue
        if hasattr(model_jax, c):
            out.add(c)
    return tuple(sorted(out))


def fsbl_photastrom_param3plus_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param3–8 jax-eval phot + astrom + likelihood parity."""
    return _fsbl_jax_eval_pairs(
        _fsbl_photastrom_param3plus_class_names(),
        methods=(
            PSBL_PHOT_METHODS
            + PSBL_PHOTASTROM_AST_METHODS
            + PSBL_PHOTASTROM_LIKELIHOOD_METHODS
        ),
    )


_BSPL_PHOTASTROM_PARAM34_PREFIXES = (
    "BSPL_PhotAstrom_noPar_Param3",
    "BSPL_PhotAstrom_Par_Param3",
    "BSPL_PhotAstrom_noPar_Param4",
    "BSPL_PhotAstrom_Par_Param4",
    "BSPL_PhotAstrom_noPar_AccOrbs_Param3",
    "BSPL_PhotAstrom_Par_AccOrbs_Param3",
    "BSPL_PhotAstrom_noPar_AccOrbs_Param4",
    "BSPL_PhotAstrom_Par_AccOrbs_Param4",
    "BSPL_PhotAstrom_noPar_CircOrbs_Param3",
    "BSPL_PhotAstrom_Par_CircOrbs_Param3",
    "BSPL_PhotAstrom_noPar_CircOrbs_Param4",
    "BSPL_PhotAstrom_Par_CircOrbs_Param4",
    "BSPL_PhotAstrom_noPar_LinOrbs_Param3",
    "BSPL_PhotAstrom_Par_LinOrbs_Param3",
    "BSPL_PhotAstrom_noPar_LinOrbs_Param4",
    "BSPL_PhotAstrom_Par_LinOrbs_Param4",
    "BSPL_PhotAstrom_noPar_EllOrbs_Param3",
    "BSPL_PhotAstrom_Par_EllOrbs_Param3",
    "BSPL_PhotAstrom_noPar_EllOrbs_Param4",
    "BSPL_PhotAstrom_Par_EllOrbs_Param4",
)


def bspl_photastrom_param34_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom Param3/4 base + orbit phot + core astrometry."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    class_names = {
        c
        for c, _ in applicable
        if any(c.startswith(p) for p in _BSPL_PHOTASTROM_PARAM34_PREFIXES)
    }
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
    )
    methods = PSBL_PHOT_METHODS + core_ast
    return sorted(
        (c, m) for c in class_names for m in methods if (c, m) in applicable
    )


_BSPL_PHOTASTROM_ORBIT_ORBITS = ("AccOrbs", "CircOrbs", "EllOrbs", "LinOrbs")


def _bspl_photastrom_orbit_param12_class_names() -> tuple[str, ...]:
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = {c for c, _ in applicable_task_pairs(model_jax)}
    out: list[str] = []
    for par in ("noPar", "Par"):
        for orb in _BSPL_PHOTASTROM_ORBIT_ORBITS:
            for n in (1, 2):
                c = f"BSPL_PhotAstrom_{par}_{orb}_Param{n}"
                if c in applicable and hasattr(model_jax, c):
                    out.append(c)
    return tuple(sorted(out))


def bspl_photastrom_orbit_param12_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom keplerian orbit Param1/2 phot + astrometry (noPar + Par)."""
    return _psbl_photastrom_pairs_for_classes(_bspl_photastrom_orbit_param12_class_names())


def bspl_photastrom_orbit_param12_extended_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom orbit Param1/2 extended likelihoods + ``get_u``."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    class_names = _bspl_photastrom_orbit_param12_class_names()
    methods = FSPL_PHOTASTROM_EXTENDED_METHODS
    return sorted(
        (c, m) for c in class_names for m in methods if (c, m) in applicable
    )


def bspl_photastrom_orbit_param12_extended_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom orbit Param1/2 extended likelihood grad (host FD)."""
    return sorted(bspl_photastrom_orbit_param12_extended_pairs())


_BSPL_PHOTASTROM_EXTENDED_PREFIXES = (
    "BSPL_PhotAstrom_noPar_Param",
    "BSPL_PhotAstrom_Par_Param",
    "BSPL_PhotAstrom_noPar_AccOrbs_Param3",
    "BSPL_PhotAstrom_Par_AccOrbs_Param3",
    "BSPL_PhotAstrom_noPar_CircOrbs_Param3",
    "BSPL_PhotAstrom_Par_CircOrbs_Param3",
    "BSPL_PhotAstrom_noPar_EllOrbs_Param3",
    "BSPL_PhotAstrom_Par_EllOrbs_Param3",
    "BSPL_PhotAstrom_noPar_EllOrbs_Param4",
    "BSPL_PhotAstrom_Par_EllOrbs_Param4",
    "BSPL_PhotAstrom_noPar_LinOrbs_Param3",
    "BSPL_PhotAstrom_Par_LinOrbs_Param3",
)


def bspl_photastrom_extended_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom Param1–3 extended likelihoods + resolved astrometry."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    class_names = {
        c
        for c, _ in applicable
        if any(c.startswith(p) for p in _BSPL_PHOTASTROM_EXTENDED_PREFIXES)
        and hasattr(model_jax, c)
    }
    methods = FSPL_PHOTASTROM_EXTENDED_METHODS + ("get_resolved_astrometry",)
    return sorted(
        (c, m) for c in class_names for m in methods if (c, m) in applicable
    )


def bspl_phot_extended_pairs() -> list[tuple[str, str]]:
    """BSPL phot-only Param1 extended (chi2, resolved astrometry, ``get_u``)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "BSPL_Phot_noPar_Param1",
        "BSPL_Phot_Par_Param1",
        "BSPL_Phot_noPar_GP_Param1",
        "BSPL_Phot_Par_GP_Param1",
    )
    methods = FSPL_PHOTASTROM_EXTENDED_METHODS + ("get_resolved_astrometry",)
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def bspl_gp_extended_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom GP orbit/base Param extended likelihoods."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    class_names = {
        c
        for c, _ in applicable
        if (
            any(c.startswith(p) for p in _BSPL_GP_ORBIT_CLASS_PREFIXES)
            or c in ("BSPL_PhotAstrom_noPar_GP_Param1", "BSPL_PhotAstrom_Par_GP_Param1")
        )
        and hasattr(model_jax, c)
    }
    methods = FSPL_PHOTASTROM_EXTENDED_METHODS + ("get_resolved_astrometry",)
    return sorted(
        (c, m) for c in class_names for m in methods if (c, m) in applicable
    )


def bsbl_photastrom_param3_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom CircOrbs/EllOrbs Param3 full phot/ast/likelihood parity."""
    return _psbl_photastrom_full_pairs_for_classes(
        (
            "BSBL_PhotAstrom_noPar_CircOrbs_Param3",
            "BSBL_PhotAstrom_Par_CircOrbs_Param3",
            "BSBL_PhotAstrom_noPar_EllOrbs_Param3",
            "BSBL_PhotAstrom_Par_EllOrbs_Param3",
        )
    )


def bsbl_photastrom_ast_likelihood_gaps_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom Param1/2/EllOrbs Param2 ast likelihood gaps."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "BSBL_PhotAstrom_noPar_Param1",
        "BSBL_PhotAstrom_Par_Param1",
        "BSBL_PhotAstrom_noPar_Param2",
        "BSBL_PhotAstrom_Par_Param2",
        "BSBL_PhotAstrom_noPar_EllOrbs_Param2",
        "BSBL_PhotAstrom_Par_EllOrbs_Param2",
    )
    methods = ("get_u", "get_chi2_astrometry", "log_likely_astrometry_each")
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def fsbl_photastrom_extended_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param1/2 extended likelihoods (jax-eval dispatch)."""
    return _fsbl_jax_eval_pairs(
        (
            "FSBL_PhotAstrom_noPar_Param1",
            "FSBL_PhotAstrom_Par_Param1",
            "FSBL_PhotAstrom_noPar_Param2",
            "FSBL_PhotAstrom_Par_Param2",
        ),
        methods=FSPL_PHOTASTROM_EXTENDED_METHODS
        + ("get_resolved_astrometry", "get_resolved_lens_astrometry"),
    )


def fsbl_phot_extended_pairs() -> list[tuple[str, str]]:
    """FSBL phot Param1 orbit variants extended (jax-eval)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = {c for c, _ in applicable_task_pairs(model_jax)}
    classes = tuple(
        sorted(
            c
            for c in applicable
            if c.startswith("FSBL_Phot_")
            and "PhotAstrom" not in c
            and hasattr(model_jax, c)
        )
    )
    methods = (
        "get_resolved_astrometry",
        "get_resolved_lens_astrometry",
        "get_u",
        "get_chi2_photometry",
        "log_likely_photometry_each",
    )
    return _fsbl_jax_eval_pairs(classes, methods=methods)


_FSBL_GRAD_ZERO_AST = frozenset(
    {
        "get_lens_astrometry",
        "get_resolved_lens_astrometry",
    }
)


def _fsbl_grad_pair_ok(class_name: str, method_name: str) -> bool:
    """Filter FSBL grad harness pairs with known FD smoke failures.

    Phot-only ``get_resolved_lens_astrometry`` is no longer excluded here;
    recovered pairs live in ``resolved_ast_grad_recovered_pairs()``.
    """
    if method_name == "get_u" and "PhotAstrom" in class_name:
        return False
    if method_name in _FSBL_GRAD_ZERO_AST:
        for tag in ("_Param4", "_Param5", "_Param7", "_Param8"):
            if tag in class_name:
                return False
        if "AccOrbs" in class_name and "_Param6" in class_name:
            return False
    return True


def _filter_fsbl_grad_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted((c, m) for c, m in pairs if _fsbl_grad_pair_ok(c, m))


def _psbl_orbit_param1_likelihood_grad_pair_ok(
    class_name: str, method_name: str
) -> bool:
    """Filter PSBL orbit Param1 extended grad pairs with NaN/zero FD smoke."""
    if method_name == "get_u" and any(
        tag in class_name for tag in ("CircOrbs", "EllOrbs")
    ):
        return False
    return True


_BSBL_PARAM1_CORE_GRAD_METHODS = frozenset(
    (
        "get_amplification",
        "get_astrometry_unlensed",
        "get_lens_astrometry",
        "get_resolved_lens_astrometry",
    )
)


def _bsbl_param1_core_grad_pair_ok(class_name: str, method_name: str) -> bool:
    """Filter BSBL Param1/orbit Param1 grad pairs with known FD smoke failures."""
    if method_name not in _BSBL_PARAM1_CORE_GRAD_METHODS:
        return False
    if method_name in _FSBL_GRAD_ZERO_AST:
        return False
    return True




def _is_bsbl_param1(class_name: str) -> bool:
    """True for BSBL static/orbit Param1 layouts (not Param2+)."""
    if "_Param2" in class_name or "_Param3" in class_name:
        return False
    return "_Param1" in class_name


def _bsbl_grad_pair_ok(class_name: str, method_name: str) -> bool:
    """Filter BSBL grad harness pairs with known FD smoke failures."""
    if _is_bsbl_param1(class_name) and method_name in _FSBL_GRAD_ZERO_AST:
        return False
    return True


def _filter_bsbl_grad_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted((c, m) for c, m in pairs if _bsbl_grad_pair_ok(c, m))


_BSBL_CORE_GRAD_METHODS = frozenset(
    (
        "get_amplification",
        "get_astrometry_unlensed",
        "get_lens_astrometry",
        "get_resolved_lens_astrometry",
    )
)
_BSBL_BULK_GRAD_METHODS = _BSBL_CORE_GRAD_METHODS | frozenset(("get_u",))


def _bsbl_bulk_grad_from_pairs(
    pairs: list[tuple[str, str]], *, include_get_u: bool = True
) -> list[tuple[str, str]]:
    methods = _BSBL_BULK_GRAD_METHODS if include_get_u else _BSBL_CORE_GRAD_METHODS
    return _filter_bsbl_grad_pairs([(c, m) for c, m in pairs if m in methods])


def _fsbl_photastrom_paramN_grad_pairs(param_n: int) -> list[tuple[str, str]]:
    """FSBL PhotAstrom ParamN grad pairs (jax-eval FD)."""
    suffix = f"_Param{param_n}"
    classes = tuple(
        c for c in _fsbl_photastrom_param3plus_class_names() if c.endswith(suffix)
    )
    return _filter_fsbl_grad_pairs(
        _fsbl_jax_eval_pairs(
            classes,
            methods=(
                PSBL_PHOT_METHODS
                + PSBL_PHOTASTROM_AST_METHODS
                + PSBL_PHOTASTROM_LIKELIHOOD_METHODS
            ),
        )
    )


def fsbl_phot_grad_pairs() -> list[tuple[str, str]]:
    """FSBL phot Param1 + orbit Param1 phot grad (jax-eval FD)."""
    out: list[tuple[str, str]] = []
    for fn in (
        fsbl_phot_param1_pairs,
        fsbl_phot_ellorbs_param1_pairs,
        fsbl_phot_circorbs_param1_pairs,
    ):
        out.extend(fn())
    return _filter_fsbl_grad_pairs(out)


def fsbl_photastrom_param1_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param1 phot + core astrometry grad (jax-eval FD)."""
    return _filter_fsbl_grad_pairs(fsbl_photastrom_param1_pairs())


def fsbl_photastrom_orbit_param1_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom keplerian orbit Param1 grad (jax-eval FD)."""
    out: list[tuple[str, str]] = []
    for fn in (
        fsbl_photastrom_linorbs_param1_pairs,
        fsbl_photastrom_accorbs_param1_pairs,
        fsbl_photastrom_circorbs_param1_pairs,
        fsbl_photastrom_ellorbs_param1_pairs,
    ):
        out.extend(fn())
    return _filter_fsbl_grad_pairs(out)


def fsbl_photastrom_param2_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param2 phot + core astrometry grad (jax-eval FD)."""
    return _filter_fsbl_grad_pairs(fsbl_photastrom_param2_pairs())


def fsbl_photastrom_param12_resolved_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param1/2 resolved astrometry grad (jax-eval FD)."""
    return _filter_fsbl_grad_pairs(fsbl_photastrom_param12_resolved_pairs())


def fsbl_photastrom_param3_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param3 grad (jax-eval FD)."""
    return _fsbl_photastrom_paramN_grad_pairs(3)


def fsbl_photastrom_param4_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param4 grad (jax-eval FD)."""
    return _fsbl_photastrom_paramN_grad_pairs(4)


def fsbl_photastrom_param5_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param5 grad (jax-eval FD)."""
    return _fsbl_photastrom_paramN_grad_pairs(5)


def fsbl_photastrom_param6_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param6 grad (jax-eval FD)."""
    return _fsbl_photastrom_paramN_grad_pairs(6)


def fsbl_photastrom_param7_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param7 grad (jax-eval FD)."""
    return _fsbl_photastrom_paramN_grad_pairs(7)


def fsbl_photastrom_param8_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param8 grad (jax-eval FD)."""
    return _fsbl_photastrom_paramN_grad_pairs(8)


def fsbl_photastrom_param38_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom CircOrbs/EllOrbs Param3/8 grad (jax-eval FD)."""
    classes: list[str] = []
    for orbit in ("CircOrbs", "EllOrbs"):
        for suffix in ("Param3", "Param8"):
            for par in ("noPar", "Par"):
                classes.append(f"FSBL_PhotAstrom_{par}_{orbit}_{suffix}")
    return _filter_fsbl_grad_pairs(
        _fsbl_jax_eval_pairs(
            tuple(classes),
            methods=(
                PSBL_PHOT_METHODS
                + PSBL_PHOTASTROM_AST_METHODS
                + PSBL_PHOTASTROM_LIKELIHOOD_METHODS
            ),
        )
    )


def fsbl_photastrom_orbit_param1_extended_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom orbit Param1 extended + resolved grad (jax-eval FD)."""
    return _filter_fsbl_grad_pairs(fsbl_photastrom_orbit_param1_extended_pairs())


def fsbl_photastrom_extended_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param1/2 extended likelihood grad (jax-eval FD)."""
    return _filter_fsbl_grad_pairs(fsbl_photastrom_extended_pairs())


def fsbl_phot_extended_grad_pairs() -> list[tuple[str, str]]:
    """FSBL phot extended + likelihood grad (jax-eval FD)."""
    return _filter_fsbl_grad_pairs(fsbl_phot_extended_pairs())


def psbl_phot_extended_pairs() -> list[tuple[str, str]]:
    """PSBL phot-only extended (resolved astrometry, chi2, likelihoods)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = sorted(
        {
            c
            for c, _ in applicable
            if c.startswith("PSBL_Phot_")
            and "PhotAstrom" not in c
            and not any(s in c for s in SKIP_CLASS_SUBSTR)
        }
    )
    methods = PSBL_PHOT_METHODS + (
        "get_resolved_astrometry",
        "get_resolved_lens_astrometry",
        "get_u",
        "get_chi2_photometry",
        "log_likely_photometry_each",
    )
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def _psbl_phot_extended_grad_pair_ok(class_name: str, method_name: str) -> bool:
    """Return whether a PSBL phot-only extended pair is grad-smoke eligible.

    Excludes core phot paths (separate harness), ``get_resolved_lens_astrometry``,
    and keplerian ``get_resolved_astrometry`` (covered by recovered-pair tests).
    """
    if method_name in PSBL_PHOT_METHODS:
        return False
    if method_name == "get_resolved_lens_astrometry":
        return False
    if method_name == "get_resolved_astrometry" and any(
        tag in class_name for tag in ("CircOrbs", "EllOrbs")
    ):
        return False
    return True


def psbl_phot_extended_grad_pairs() -> list[tuple[str, str]]:
    """PSBL phot extended + likelihood grad (host FD through roots)."""
    return sorted(
        (c, m)
        for c, m in psbl_phot_extended_pairs()
        if _psbl_phot_extended_grad_pair_ok(c, m)
    )


def psbl_photastrom_param1_ast_likelihood_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param1/4 ast likelihood + ``get_u`` gaps."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "PSBL_PhotAstrom_noPar_Param1",
        "PSBL_PhotAstrom_Par_Param1",
        "PSBL_PhotAstrom_noPar_Param4",
        "PSBL_PhotAstrom_Par_Param4",
    )
    methods = (
        "get_u",
        "get_chi2_photometry",
        "get_chi2_astrometry",
        "log_likely_photometry_each",
        "log_likely_astrometry_each",
    )
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_photastrom_param7_ast_likelihood_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param7 ast chi2 / log-likelihood (all orbit variants)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = sorted(
        {
            c
            for c, _ in applicable
            if c.startswith("PSBL_PhotAstrom_")
            and "Param7" in c
            and "GP" not in c
            and hasattr(model_jax, c)
        }
    )
    methods = ("get_chi2_astrometry", "log_likely_astrometry_each")
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_photastrom_gp_extended_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom/Phot GP Param1–2 extended astrometry + phot likelihoods."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    photastrom = (
        "PSBL_PhotAstrom_noPar_GP_Param1",
        "PSBL_PhotAstrom_Par_GP_Param1",
        "PSBL_PhotAstrom_noPar_GP_Param2",
        "PSBL_PhotAstrom_Par_GP_Param2",
    )
    phot = ("PSBL_Phot_noPar_GP_Param1", "PSBL_Phot_Par_GP_Param1")
    out: list[tuple[str, str]] = []
    for c in photastrom:
        for m in (
            "get_resolved_astrometry",
            "get_resolved_lens_astrometry",
            "get_chi2_astrometry",
            "log_likely_astrometry_each",
        ):
            if (c, m) in applicable:
                out.append((c, m))
    for c in phot:
        for m in PSBL_GP_EXTENDED_METHODS + (
            "get_resolved_astrometry",
            "get_resolved_lens_astrometry",
        ):
            if (c, m) in applicable:
                out.append((c, m))
    return sorted(out)


def fspl_outline_and_resolved_pairs() -> list[tuple[str, str]]:
    """FSPL outline unlensed astrometry (PhotAstrom Param1/2)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    pairs: list[tuple[str, str]] = []
    for c in (
        "FSPL_PhotAstrom_noPar_Param1",
        "FSPL_PhotAstrom_Par_Param1",
        "FSPL_PhotAstrom_noPar_Param2",
        "FSPL_PhotAstrom_Par_Param2",
    ):
        if (c, "get_astrometry_outline_unlensed") in applicable:
            pairs.append((c, "get_astrometry_outline_unlensed"))
    return sorted(pairs)


def fspl_phot_param2_resolved_pairs() -> list[tuple[str, str]]:
    """FSPL phot-only Param2 ``get_resolved_astrometry`` (jax-eval)."""
    return _fsbl_jax_eval_pairs(
        ("FSPL_Phot_noPar_Param2", "FSPL_Phot_Par_Param2"),
        methods=("get_resolved_astrometry",),
    )


def migration_numpy_fallback_pairs() -> list[tuple[str, str]]:
    """Union of harness pairs for applicable tasks still on numpy_fallback."""
    fns = (
        fsbl_photastrom_orbit_param1_extended_pairs,
        fsbl_photastrom_param12_resolved_pairs,
        fsbl_photastrom_extended_pairs,
        fsbl_phot_extended_pairs,
        bspl_photastrom_orbit_param12_extended_pairs,
        bspl_photastrom_extended_pairs,
        fspl_outline_and_resolved_pairs,
        fspl_phot_param2_resolved_pairs,
    )
    out: set[tuple[str, str]] = set()
    for fn in fns:
        out.update(fn())
    return sorted(out)


def bfspl_photastrom_param1_pairs() -> list[tuple[str, str]]:
    """BFSPL PhotAstrom Param1 phot + core astrometry (noPar + Par)."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "BFSPL_PhotAstrom_noPar_Param1",
            "BFSPL_PhotAstrom_Par_Param1",
        )
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


def bspl_gp_grad_pairs() -> list[tuple[str, str]]:
    """BSPL phot-only GP ``get_photometry_with_gp`` grad smoke (noPar + Par)."""
    return [
        (c, m)
        for c, m in bspl_phot_gp_param1_pairs()
        if m == "get_photometry_with_gp"
    ]


def bspl_photastrom_gp_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom GP Param1 ``get_photometry_with_gp`` grad smoke."""
    return [
        (c, m)
        for c, m in bspl_photastrom_gp_param1_pairs()
        if m == "get_photometry_with_gp"
    ]


def psbl_gp_grad_pairs() -> list[tuple[str, str]]:
    """PSBL phot / PhotAstrom GP Param1 ``get_photometry_with_gp`` grad smoke."""
    out: list[tuple[str, str]] = []
    for pairs_fn in (psbl_gp_param1_pairs, psbl_photastrom_gp_param1_pairs):
        out.extend(
            (c, m)
            for c, m in pairs_fn()
            if m == "get_photometry_with_gp"
        )
    return sorted(set(out))


_BSPL_PHOTASTROM_PARAM1_CORE_AST = tuple(
    m
    for m in PSBL_PHOTASTROM_AST_METHODS
    if m not in ("get_resolved_astrometry", "get_resolved_lens_astrometry")
)


def bspl_photastrom_param1_grad_bulk_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom Param1 phot + core astrometry (FD grad)."""
    methods = PSBL_PHOT_METHODS + _BSPL_PHOTASTROM_PARAM1_CORE_AST
    return sorted(
        (c, m)
        for c, m in bspl_photastrom_param1_pairs()
        if m in methods
    )


def psbl_photastrom_param1_grad_bulk_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param1 phot + core astrometry (FD grad; chi2 flat at fixture)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_Param1", "PSBL_PhotAstrom_Par_Param1")
    methods = PSBL_PHOT_METHODS + _BSPL_PHOTASTROM_PARAM1_CORE_AST
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def bspl_phot_grad_pairs() -> list[tuple[str, str]]:
    """BSPL static phot-only Param1 phot/amp grad smoke."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("BSPL_Phot_noPar_Param1", "BSPL_Phot_Par_Param1")
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOT_METHODS
        if (c, m) in applicable
    )


def bspl_orbit_phot_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom keplerian orbit Param1/2 phot grad (host FD)."""
    return [
        (c, m)
        for c, m in bspl_photastrom_orbit_param12_pairs()
        if m in PSBL_PHOT_METHODS
    ]


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


def psbl_photastrom_param4_grad_ast_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param4 astrometry + ast likelihood JAX grad smoke."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_Param4", "PSBL_PhotAstrom_Par_Param4")
    methods = _BSPL_PHOTASTROM_PARAM1_CORE_AST + (
        "get_chi2_astrometry",
        "log_likely_astrometry_each",
    )
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_photastrom_likelihood_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param2/3 phot+ast chi2 / log-likelihood JAX grad smoke."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "PSBL_PhotAstrom_noPar_Param2",
        "PSBL_PhotAstrom_Par_Param2",
        "PSBL_PhotAstrom_noPar_Param3",
        "PSBL_PhotAstrom_Par_Param3",
    )
    param3 = {
        "PSBL_PhotAstrom_noPar_Param3",
        "PSBL_PhotAstrom_Par_Param3",
    }
    phot_lik = PHOT_LIKELIHOOD_METHODS
    methods = (
        "get_chi2_photometry",
        "log_likely_photometry_each",
        "get_chi2_astrometry",
        "log_likely_astrometry_each",
    )
    return sorted(
        (c, m)
        for c in classes
        for m in methods
        if (c, m) in applicable
        and not (c in param3 and m in phot_lik)
        and not (c.endswith("_Par_Param2") and m in phot_lik)
    )


def psbl_gp_param2_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom GP Param2 ``get_photometry_with_gp`` grad smoke."""
    return [
        (c, m)
        for c, m in psbl_gp_photastrom_param2_pairs()
        if m == "get_photometry_with_gp"
    ]


def psbl_gp_param2_extended_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom GP Param2 ``get_u`` + phot chi2 / log-likelihood grad smoke."""
    skip = {
        ("PSBL_PhotAstrom_Par_GP_Param2", "get_chi2_photometry"),
        ("PSBL_PhotAstrom_Par_GP_Param2", "log_likely_photometry_each"),
    }
    return sorted(
        (c, m)
        for c, m in psbl_gp_photastrom_param2_pairs()
        if m in PSBL_GP_EXTENDED_METHODS
        and (c, m) not in skip
    )


def bspl_photastrom_param2_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom Param2 phot + core astrometry grad smoke (noPar + Par)."""
    return bspl_photastrom_param2_pairs()


def bspl_photastrom_param2_likelihood_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom noPar Param2 phot likelihood + ``get_u`` grad smoke."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("BSPL_PhotAstrom_noPar_Param2",)
    methods = PSBL_PHOTASTROM_PHOT_LIKELIHOOD_METHODS
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_photastrom_param5_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Par Param5 phot + core astrometry grad smoke."""
    core_ast = tuple(
        m
        for m in PSBL_PHOTASTROM_AST_METHODS
        if m
        not in (
            "get_resolved_astrometry",
            "get_resolved_lens_astrometry",
        )
    )
    methods = PSBL_PHOT_METHODS + core_ast
    return sorted(
        (c, m)
        for c, m in psbl_photastrom_param5_pairs()
        if m in methods
    )


def psbl_photastrom_param6_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param6 phot + core astrometry + ``get_u`` grad smoke."""
    methods = PSBL_PHOT_METHODS + _BSPL_PHOTASTROM_PARAM1_CORE_AST + ("get_u",)
    return sorted(
        (c, m)
        for c, m in psbl_photastrom_param6_pairs()
        if m in methods
    )


def psbl_photastrom_orbit_param2_grad_bulk_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom keplerian orbit Param2 phot + core astrometry grad smoke."""
    methods = PSBL_PHOT_METHODS + _BSPL_PHOTASTROM_PARAM1_CORE_AST
    out: list[tuple[str, str]] = []
    for orbit in ("CircOrbs", "EllOrbs", "AccOrbs", "LinOrbs"):
        no_par = f"PSBL_PhotAstrom_noPar_{orbit}_Param2"
        par = f"PSBL_PhotAstrom_Par_{orbit}_Param2"
        pairs = _psbl_photastrom_full_pairs_for_classes((no_par, par))
        out.extend((c, m) for c, m in pairs if m in methods)
    return sorted(set(out))


def psbl_photastrom_param2_grad_bulk_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom static Param2 phot + core astrometry grad smoke."""
    methods = PSBL_PHOT_METHODS + _BSPL_PHOTASTROM_PARAM1_CORE_AST
    return sorted(
        (c, m) for c, m in psbl_photastrom_param2_pairs() if m in methods
    )


def psbl_photastrom_param3_grad_bulk_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom static Param3 phot + core astrometry + ``get_u`` grad smoke."""
    methods = PSBL_PHOT_METHODS + _BSPL_PHOTASTROM_PARAM1_CORE_AST + ("get_u",)
    return sorted(
        (c, m) for c, m in psbl_photastrom_param3_pairs() if m in methods
    )


def psbl_photastrom_circorbs_ellorbs_param2_extended_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom CircOrbs/EllOrbs Param2 extended likelihood grad (host FD)."""
    methods = FSPL_PHOTASTROM_EXTENDED_METHODS
    out: list[tuple[str, str]] = []
    for fn in (psbl_photastrom_circorbs_param2_pairs, psbl_photastrom_ellorbs_param2_pairs):
        out.extend((c, m) for c, m in fn() if m in methods)
    return sorted(set(out))


def psbl_gp_param2_bulk_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom GP Param2 phot + core astrometry grad (host FD for GP)."""
    methods = PSBL_PHOT_METHODS + _BSPL_PHOTASTROM_PARAM1_CORE_AST
    return sorted(
        (c, m)
        for c, m in psbl_gp_photastrom_param2_pairs()
        if m in methods
    )


def fspl_photastrom_param1_grad_ast_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param1 core astrometry grad (jax-eval FD)."""
    return sorted(
        (c, m)
        for c, m in fspl_photastrom_param1_pairs()
        if m in _BSPL_PHOTASTROM_PARAM1_CORE_AST
    )


def fspl_photastrom_param2_grad_ast_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param2 core astrometry grad (jax-eval FD)."""
    return sorted(
        (c, m)
        for c, m in fspl_photastrom_param2_pairs()
        if m in _BSPL_PHOTASTROM_PARAM1_CORE_AST
    )


def fspl_photastrom_param1_extended_grad_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param1 extended likelihood grad (jax-eval FD)."""
    return sorted(fspl_photastrom_param1_extended_pairs())


def fspl_photastrom_param2_extended_grad_pairs() -> list[tuple[str, str]]:
    """FSPL PhotAstrom Param2 extended likelihood grad (jax-eval FD)."""
    return list(fspl_photastrom_param2_extended_pairs())


def fspl_phot_param2_grad_pairs() -> list[tuple[str, str]]:
    """FSPL phot-only Param2 phot/amp grad (jax-eval FD)."""
    return list(fspl_phot_param2_pairs())


def fspl_phot_param2_extended_grad_pairs() -> list[tuple[str, str]]:
    """FSPL phot-only Param2 ``get_u`` + phot likelihood grad (jax-eval FD)."""
    return list(fspl_phot_param2_extended_pairs())


def fspl_phot_param2_resolved_grad_pairs() -> list[tuple[str, str]]:
    """FSPL phot-only Param2 ``get_resolved_astrometry`` grad (jax-eval FD)."""
    return list(fspl_phot_param2_resolved_pairs())


def bspl_photastrom_param34_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom Param3/4 phot + core astrometry grad (host FD)."""
    methods = PSBL_PHOT_METHODS + _BSPL_PHOTASTROM_PARAM1_CORE_AST
    return sorted(
        (c, m) for c, m in bspl_photastrom_param34_pairs() if m in methods
    )


def psbl_photastrom_orbit_param1_likelihood_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom orbit Param1 likelihood + ``get_u`` grad (host FD)."""
    methods = PSBL_PHOTASTROM_LIKELIHOOD_METHODS + (
        "get_u",
        "get_resolved_astrometry",
        "get_resolved_lens_astrometry",
    )
    pair_fns = (
        psbl_photastrom_circorbs_param1_pairs,
        psbl_photastrom_ellorbs_param1_pairs,
        psbl_photastrom_accorbs_param1_pairs,
        psbl_photastrom_linorbs_param1_pairs,
    )
    out: list[tuple[str, str]] = []
    for fn in pair_fns:
        out.extend((c, m) for c, m in fn() if m in methods)
    return sorted(
        (c, m)
        for c, m in set(out)
        if _psbl_orbit_param1_likelihood_grad_pair_ok(c, m)
    )


def bspl_photastrom_gp_orbit_ast_grad_pairs() -> list[tuple[str, str]]:
    """BSPL GP orbit Param1-3 core astrometry grad (host FD)."""
    return sorted(
        (c, m)
        for c, m in bspl_photastrom_gp_orbit_and_param23_pairs()
        if m in _BSPL_PHOTASTROM_PARAM1_CORE_AST and "Orbs" in c
    )


def bfspl_photastrom_param1_grad_pairs() -> list[tuple[str, str]]:
    """BFSPL PhotAstrom Param1 phot + core astrometry grad (host FD)."""
    return list(bfspl_photastrom_param1_pairs())


def fspl_outline_and_resolved_grad_pairs() -> list[tuple[str, str]]:
    """FSPL outline unlensed + resolved astrometry grad (jax-eval FD)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    skip = {("FSPL_PhotAstrom_Par_Param1", "get_u")}
    classes = (
        "FSPL_PhotAstrom_noPar_Param1",
        "FSPL_PhotAstrom_Par_Param1",
        "FSPL_PhotAstrom_noPar_Param2",
        "FSPL_PhotAstrom_Par_Param2",
    )
    methods = ("get_astrometry_outline_unlensed", "get_resolved_astrometry")
    return sorted(
        (c, m)
        for c in classes
        for m in methods
        if (c, m) in applicable and (c, m) not in skip
    )


def bsbl_photastrom_param1_core_grad_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom Param1/orbit Param1 core ast + amp grad (jax-eval FD)."""
    pair_fns = (
        bsbl_photastrom_param1_pairs,
        bsbl_photastrom_circorbs_param1_pairs,
        bsbl_photastrom_ellorbs_param1_pairs,
        bsbl_photastrom_linorbs_param1_pairs,
        bsbl_photastrom_accorbs_param1_pairs,
    )
    out: list[tuple[str, str]] = []
    for fn in pair_fns:
        out.extend(fn())
    return sorted((c, m) for c, m in set(out) if _bsbl_param1_core_grad_pair_ok(c, m))


def bsbl_photastrom_orbit_param1_get_u_grad_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom Param1/orbit Param1 ``get_u`` grad (jax-eval FD)."""
    pair_fns = (
        bsbl_photastrom_param1_phot_likelihood_pairs,
        bsbl_photastrom_circorbs_param1_likelihood_pairs,
        bsbl_photastrom_linorbs_param1_likelihood_pairs,
        bsbl_photastrom_accorbs_param1_likelihood_pairs,
    )
    out: list[tuple[str, str]] = []
    for fn in pair_fns:
        out.extend((c, m) for c, m in fn() if m == "get_u")
    return sorted(set(out))


def bsbl_photastrom_param2_grad_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom Param2 core ast + ``get_u`` grad (host FD)."""
    out: list[tuple[str, str]] = []
    out.extend(bsbl_photastrom_param2_pairs())
    out.extend(bsbl_photastrom_param2_phot_likelihood_pairs())
    return _bsbl_bulk_grad_from_pairs(out)


def bsbl_photastrom_param3_grad_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom CircOrbs/EllOrbs Param3 core ast + ``get_u`` grad."""
    return _bsbl_bulk_grad_from_pairs(list(bsbl_photastrom_param3_pairs()))


def bsbl_photastrom_circorbs_param2_grad_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom CircOrbs Param2 core ast + ``get_u`` grad (host FD)."""
    out: list[tuple[str, str]] = []
    for fn in (
        bsbl_photastrom_circorbs_param2_pairs,
        bsbl_photastrom_circorbs_param2_phot_likelihood_pairs,
    ):
        out.extend(fn())
    return _bsbl_bulk_grad_from_pairs(out)


def bsbl_photastrom_ellorbs_param2_grad_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom EllOrbs Param2 core ast grad (host FD; ``get_u`` flat)."""
    out: list[tuple[str, str]] = []
    for fn in (
        bsbl_photastrom_ellorbs_param2_pairs,
        bsbl_photastrom_ellorbs_param2_phot_likelihood_pairs,
    ):
        out.extend(fn())
    return _bsbl_bulk_grad_from_pairs(out, include_get_u=False)


def bspl_photastrom_orbit_param12_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom orbit Param1/2 core astrometry grad (host FD)."""
    skip = frozenset(("get_resolved_lens_astrometry",))
    methods = tuple(m for m in PSBL_PHOTASTROM_AST_METHODS if m not in skip)
    return sorted(
        (c, m) for c, m in bspl_photastrom_orbit_param12_pairs() if m in methods
    )


def fsbl_photastrom_param3plus_remaining_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom Param3+ ``get_u`` + lens astrometry grad (jax-eval FD)."""
    methods = ("get_u", "get_lens_astrometry", "get_resolved_lens_astrometry")
    return _filter_fsbl_grad_pairs(
        [(c, m) for c, m in fsbl_photastrom_param3plus_pairs() if m in methods]
    )


def psbl_photastrom_param1_ast_likelihood_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param1 ast chi2 / log-likelihood grad (host FD)."""
    return list(psbl_photastrom_param1_ast_likelihood_pairs())


def bspl_photastrom_gp_orbit_param23_grad_pairs() -> list[tuple[str, str]]:
    """BSPL GP Param2/3 + orbit GP core astrometry grad (host FD)."""
    skip = frozenset(("get_resolved_lens_astrometry",))
    methods = tuple(m for m in _BSPL_PHOTASTROM_PARAM1_CORE_AST if m not in skip)
    return sorted(
        (c, m)
        for c, m in bspl_photastrom_gp_orbit_and_param23_pairs()
        if m in methods
    )


def psbl_photastrom_param7_remaining_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param7 phot + likelihood grad not yet marked."""
    methods = (
        "get_photometry",
        "get_amplification",
        "get_chi2_photometry",
        "log_likely_photometry_each",
    )
    return sorted((c, m) for c, m in psbl_photastrom_param7_pairs() if m in methods)


def psbl_phot_extended_remaining_grad_pairs() -> list[tuple[str, str]]:
    """PSBL phot-only extended likelihood grad (host FD)."""
    methods = ("get_chi2_photometry", "log_likely_photometry_each", "get_u")
    return sorted((c, m) for c, m in psbl_phot_extended_pairs() if m in methods)


def bspl_photastrom_gp_param1_remaining_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom GP Param1 phot + GP grad (host FD)."""
    methods = GP_PHOT_METHODS
    return sorted(
        (c, m) for c, m in bspl_photastrom_gp_param1_pairs() if m in methods
    )


def psbl_photastrom_gp_param1_remaining_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom GP Param1 phot + GP + likelihood grad (host FD)."""
    methods = GP_PHOT_METHODS + PSBL_PHOTASTROM_LIKELIHOOD_METHODS
    return sorted(
        (c, m) for c, m in psbl_photastrom_gp_param1_pairs() if m in methods
    )


def psbl_phot_remaining_grad_pairs() -> list[tuple[str, str]]:
    """PSBL phot-only phot + likelihood grad (host FD)."""
    methods = PSBL_PHOT_METHODS + (
        "get_chi2_photometry",
        "log_likely_photometry_each",
        "get_u",
    )
    return sorted((c, m) for c, m in psbl_phot_pairs() if m in methods)


def psbl_photastrom_param3_likelihood_remaining_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param3 likelihood grad (host FD)."""
    return list(psbl_photastrom_param3_likelihood_pairs())


def psbl_photastrom_param2_likelihood_remaining_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param2 likelihood grad (host FD)."""
    return list(psbl_photastrom_param2_likelihood_pairs())


def fsbl_photastrom_orbit_param1_extended_remaining_grad_pairs() -> list[
    tuple[str, str]
]:
    """FSBL PhotAstrom orbit Param1 extended likelihood grad (jax-eval FD)."""
    skip = frozenset(("get_resolved_lens_astrometry",))
    return _filter_fsbl_grad_pairs(
        [
            (c, m)
            for c, m in fsbl_photastrom_orbit_param1_extended_pairs()
            if m not in skip
        ]
    )


def _grad_probe_nonresolved_pass_raw() -> list[tuple[str, str]]:
    """Load non-resolved grad probe passes from ``docs/grad_probe_nonresolved.json``."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "docs" / "grad_probe_nonresolved.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(pair) for pair in data["pass"]]


def grad_probe_nonresolved_pass_pairs() -> list[tuple[str, str]]:
    """Grad pairs passing finite-diff smoke (non-resolved probe batch)."""
    return sorted(_grad_probe_nonresolved_pass_raw())


def fsbl_photastrom_get_u_probe_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom ``get_u`` pairs verified by non-resolved grad probe."""
    return sorted(
        (c, m)
        for c, m in _grad_probe_nonresolved_pass_raw()
        if c.startswith("FSBL_") and m == "get_u"
    )


def fsbl_photastrom_lens_ast_probe_grad_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom ``get_lens_astrometry`` pairs from non-resolved probe."""
    return sorted(
        (c, m)
        for c, m in _grad_probe_nonresolved_pass_raw()
        if c.startswith("FSBL_") and m == "get_lens_astrometry"
    )


def bsbl_lens_u_probe_grad_pairs() -> list[tuple[str, str]]:
    """BSBL Param1/2 ``get_lens_astrometry`` / ``get_u`` pairs from probe."""
    return sorted(
        (c, m)
        for c, m in _grad_probe_nonresolved_pass_raw()
        if c.startswith("BSBL_") and m in ("get_lens_astrometry", "get_u")
    )


def bspl_gp_probe_grad_pairs() -> list[tuple[str, str]]:
    """BSPL GP Param1 core ast + phot pairs from non-resolved probe."""
    return sorted(
        (c, m) for c, m in _grad_probe_nonresolved_pass_raw() if c.startswith("BSPL_")
    )


def psbl_gp_param_probe_grad_pairs() -> list[tuple[str, str]]:
    """PSBL GP/Param4/5 grad pairs from non-resolved probe."""
    return sorted(
        (c, m) for c, m in _grad_probe_nonresolved_pass_raw() if c.startswith("PSBL_")
    )


def _grad_probe_resolved_pass_raw() -> list[tuple[str, str]]:
    """Load resolved grad probe passes from ``docs/grad_probe_resolved_*.json``."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "docs"
    passes: list[tuple[str, str]] = []
    for name in ("bsbl", "bspl", "psbl", "fsbl"):
        path = root / f"grad_probe_resolved_{name}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        passes.extend(tuple(pair) for pair in data["pass"])
    return passes


def grad_probe_resolved_pass_pairs() -> list[tuple[str, str]]:
    """Grad pairs passing finite-diff smoke (resolved astrometry probe batches)."""
    return sorted(set(_grad_probe_resolved_pass_raw()))


def bspl_phot_extended_probe_grad_pairs() -> list[tuple[str, str]]:
    """BSPL phot-only extended likelihood + ``get_u`` (host FD)."""
    methods = ("get_u", "get_chi2_photometry", "log_likely_photometry_each")
    return sorted(
        (c, m) for c, m in bspl_phot_extended_pairs() if m in methods
    )


def psbl_phot_orbit_param1_phot_grad_pairs() -> list[tuple[str, str]]:
    """PSBL phot-only CircOrbs/EllOrbs Param1 phot paths (host FD)."""
    classes = (
        "PSBL_Phot_Par_CircOrbs_Param1",
        "PSBL_Phot_noPar_CircOrbs_Param1",
        "PSBL_Phot_Par_EllOrbs_Param1",
        "PSBL_Phot_noPar_EllOrbs_Param1",
    )
    return sorted(
        (c, m)
        for c in classes
        for m in ("get_photometry", "get_amplification")
    )


def _bsbl_param1_phot_ast_likelihood_grad_pair_sources() -> tuple:
    """Pair-list builders for BSBL Param1/2/3 phot + ast + likelihood grad rows."""
    return (
        bsbl_photastrom_param1_pairs,
        bsbl_photastrom_param1_phot_likelihood_pairs,
        bsbl_photastrom_circorbs_param1_pairs,
        bsbl_photastrom_circorbs_param1_likelihood_pairs,
        bsbl_photastrom_ellorbs_param1_pairs,
        bsbl_photastrom_ellorbs_param1_likelihood_pairs,
        bsbl_photastrom_linorbs_param1_pairs,
        bsbl_photastrom_linorbs_param1_likelihood_pairs,
        bsbl_photastrom_accorbs_param1_pairs,
        bsbl_photastrom_accorbs_param1_likelihood_pairs,
        bsbl_photastrom_param2_pairs,
        bsbl_photastrom_param2_phot_likelihood_pairs,
        bsbl_photastrom_param3_pairs,
        bsbl_photastrom_circorbs_param2_pairs,
        bsbl_photastrom_circorbs_param2_phot_likelihood_pairs,
        bsbl_photastrom_circorbs_param2_ast_likelihood_pairs,
        bsbl_photastrom_ellorbs_param2_pairs,
        bsbl_photastrom_ellorbs_param2_phot_likelihood_pairs,
        bsbl_photastrom_ast_likelihood_gaps_pairs,
    )


def bsbl_param1_phot_ast_likelihood_grad_recovered_pairs() -> list[tuple[str, str]]:
    """BSBL Param1/2/3 phot + ast + likelihood grad pairs recovered via ``root_tol`` FD skip.

    Returns
    ----
    list of tuple[str, str]
        One hundred forty ``(class_name, method_name)`` rows formerly marked
        ``grad: skip`` because host FD through ``root_tol`` produced NaN.
    """
    methods = _BSBL_PARAM1_PHOT_AST_LIKELIHOOD_GRAD_METHODS
    out: set[tuple[str, str]] = set()
    for fn in _bsbl_param1_phot_ast_likelihood_grad_pair_sources():
        out.update((c, m) for c, m in fn() if m in methods)
    return sorted(out)


def bsbl_param1_phot_grad_recovered_pairs() -> list[tuple[str, str]]:
    """BSBL Param1/2/3 phot-only subset of ``bsbl_param1_phot_ast_likelihood_grad_recovered_pairs``."""
    phot_methods = frozenset(
        (
            "get_photometry",
            "get_centroid_shift",
            "get_chi2_photometry",
            "log_likely_photometry_each",
        )
    )
    return sorted(
        (c, m)
        for c, m in bsbl_param1_phot_ast_likelihood_grad_recovered_pairs()
        if m in phot_methods
    )


def fsbl_lens_ast_grad_recovered_pairs() -> list[tuple[str, str]]:
    """FSBL PhotAstrom ``get_lens_astrometry`` pairs recovered from zero-FD skip batch.

    Returns
    ----
    list of tuple[str, str]
        Eighteen ``(class_name, method_name)`` rows that pass jax-eval FD after
        derived-geometry refresh (Param4/8 heliocentric COM) and/or squared FD
        objective at the fixture.
    """
    static_param48 = (
        "FSBL_PhotAstrom_Par_Param4",
        "FSBL_PhotAstrom_Par_Param8",
        "FSBL_PhotAstrom_noPar_Param4",
        "FSBL_PhotAstrom_noPar_Param8",
    )
    orbit_param48 = (
        "FSBL_PhotAstrom_Par_CircOrbs_Param4",
        "FSBL_PhotAstrom_Par_CircOrbs_Param8",
        "FSBL_PhotAstrom_Par_EllOrbs_Param4",
        "FSBL_PhotAstrom_Par_EllOrbs_Param8",
        "FSBL_PhotAstrom_noPar_CircOrbs_Param4",
        "FSBL_PhotAstrom_noPar_CircOrbs_Param8",
        "FSBL_PhotAstrom_noPar_EllOrbs_Param4",
        "FSBL_PhotAstrom_noPar_EllOrbs_Param8",
    )
    return sorted(
        [
            ("FSBL_PhotAstrom_Par_AccOrbs_Param6", "get_lens_astrometry"),
            ("FSBL_PhotAstrom_Par_AccOrbs_Param7", "get_lens_astrometry"),
            ("FSBL_PhotAstrom_Par_LinOrbs_Param7", "get_lens_astrometry"),
            ("FSBL_PhotAstrom_Par_Param5", "get_lens_astrometry"),
            ("FSBL_PhotAstrom_Par_Param7", "get_lens_astrometry"),
            ("FSBL_PhotAstrom_noPar_AccOrbs_Param6", "get_lens_astrometry"),
        ]
        + [(c, "get_lens_astrometry") for c in static_param48 + orbit_param48]
    )


def get_u_grad_recovered_pairs() -> list[tuple[str, str]]:
    """Thirty-one ``get_u`` pairs recovered from the skip batch via geometry refresh.

    Probe artifact: ``docs/grad_probe_get_u_reprobe.json``.

    Returns
    ----
    list of tuple[str, str]
        PSBL/FSBL orbit PhotAstrom Param1/7, BSPL phot/photastrom Param1/GP,
        and FSPL PhotAstrom Par Param1 rows that pass host FD grad smoke after
        derived-geometry refresh and squared FD objective on ``get_u``.
    """
    return sorted(
        [
            ("BSPL_PhotAstrom_Par_GP_Param1", "get_u"),
            ("BSPL_PhotAstrom_Par_Param1", "get_u"),
            ("BSPL_Phot_noPar_GP_Param1", "get_u"),
            ("BSPL_Phot_noPar_Param1", "get_u"),
            ("FSBL_PhotAstrom_Par_AccOrbs_Param7", "get_u"),
            ("FSBL_PhotAstrom_Par_CircOrbs_Param1", "get_u"),
            ("FSBL_PhotAstrom_Par_CircOrbs_Param7", "get_u"),
            ("FSBL_PhotAstrom_Par_EllOrbs_Param1", "get_u"),
            ("FSBL_PhotAstrom_Par_EllOrbs_Param7", "get_u"),
            ("FSBL_PhotAstrom_Par_LinOrbs_Param7", "get_u"),
            ("FSBL_PhotAstrom_Par_Param7", "get_u"),
            ("FSBL_PhotAstrom_noPar_AccOrbs_Param7", "get_u"),
            ("FSBL_PhotAstrom_noPar_CircOrbs_Param1", "get_u"),
            ("FSBL_PhotAstrom_noPar_CircOrbs_Param7", "get_u"),
            ("FSBL_PhotAstrom_noPar_EllOrbs_Param1", "get_u"),
            ("FSBL_PhotAstrom_noPar_EllOrbs_Param7", "get_u"),
            ("FSBL_PhotAstrom_noPar_LinOrbs_Param7", "get_u"),
            ("FSPL_PhotAstrom_Par_Param1", "get_u"),
            ("PSBL_PhotAstrom_Par_AccOrbs_Param7", "get_u"),
            ("PSBL_PhotAstrom_Par_CircOrbs_Param1", "get_u"),
            ("PSBL_PhotAstrom_Par_CircOrbs_Param7", "get_u"),
            ("PSBL_PhotAstrom_Par_EllOrbs_Param1", "get_u"),
            ("PSBL_PhotAstrom_Par_EllOrbs_Param7", "get_u"),
            ("PSBL_PhotAstrom_Par_LinOrbs_Param7", "get_u"),
            ("PSBL_PhotAstrom_Par_Param7", "get_u"),
            ("PSBL_PhotAstrom_noPar_AccOrbs_Param7", "get_u"),
            ("PSBL_PhotAstrom_noPar_CircOrbs_Param1", "get_u"),
            ("PSBL_PhotAstrom_noPar_CircOrbs_Param7", "get_u"),
            ("PSBL_PhotAstrom_noPar_EllOrbs_Param1", "get_u"),
            ("PSBL_PhotAstrom_noPar_EllOrbs_Param7", "get_u"),
            ("PSBL_PhotAstrom_noPar_LinOrbs_Param7", "get_u"),
        ]
    )


def resolved_ast_grad_recovered_pairs() -> list[tuple[str, str]]:
    """Eleven resolved-astrometry grad pairs recovered from probe skip batches.

    Returns
    ----
    list of tuple[str, str]
        ``(class_name, method_name)`` for FSBL/PSBL phot paths that pass
        host FD grad smoke after derived-geometry refresh and squared FD
        objective on resolved astrometry outputs.
    """
    return sorted(
        [
            ("FSBL_Phot_Par_Param1", "get_resolved_lens_astrometry"),
            ("FSBL_Phot_noPar_Param1", "get_resolved_lens_astrometry"),
            ("PSBL_PhotAstrom_Par_Param5", "get_resolved_lens_astrometry"),
            ("PSBL_Phot_Par_CircOrbs_Param1", "get_resolved_astrometry"),
            ("PSBL_Phot_Par_EllOrbs_Param1", "get_resolved_astrometry"),
            ("PSBL_Phot_Par_GP_Param1", "get_resolved_lens_astrometry"),
            ("PSBL_Phot_Par_Param1", "get_resolved_lens_astrometry"),
            ("PSBL_Phot_noPar_CircOrbs_Param1", "get_resolved_astrometry"),
            ("PSBL_Phot_noPar_EllOrbs_Param1", "get_resolved_astrometry"),
            ("PSBL_Phot_noPar_GP_Param1", "get_resolved_lens_astrometry"),
            ("PSBL_Phot_noPar_Param1", "get_resolved_lens_astrometry"),
        ]
    )


def psbl_photastrom_circorbs_ellorbs_param38_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom CircOrbs/EllOrbs Param3/8 grad (FD)."""
    methods = (
        PSBL_PHOT_METHODS
        + _BSPL_PHOTASTROM_PARAM1_CORE_AST
        + ("get_u",)
        + PSBL_PHOTASTROM_LIKELIHOOD_METHODS
    )
    return sorted(
        (c, m)
        for c, m in psbl_photastrom_circorbs_ellorbs_param38_pairs()
        if m in methods
    )


def psbl_photastrom_param7_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param7 static + orbit variants grad (FD; ``get_u`` flat)."""
    methods = (
        PSBL_PHOT_METHODS
        + _BSPL_PHOTASTROM_PARAM1_CORE_AST
        + ("get_chi2_photometry", "log_likely_photometry_each")
    )
    return sorted(
        (c, m) for c, m in psbl_photastrom_param7_pairs() if m in methods
    )


def psbl_photastrom_orbit_param4_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom CircOrbs/EllOrbs Param4 grad (FD)."""
    methods = (
        PSBL_PHOT_METHODS
        + _BSPL_PHOTASTROM_PARAM1_CORE_AST
        + ("get_u",)
        + PSBL_PHOTASTROM_LIKELIHOOD_METHODS
    )
    return sorted(
        (c, m) for c, m in psbl_photastrom_orbit_param4_pairs() if m in methods
    )


def bspl_gp_extended_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom GP orbit Param extended grad (host FD for orbit GP)."""
    methods = (
        GP_PHOT_METHODS
        + _BSPL_PHOTASTROM_PARAM1_CORE_AST
        + FSPL_PHOTASTROM_EXTENDED_METHODS
    )
    return sorted(
        (c, m) for c, m in bspl_gp_extended_pairs() if m in methods
    )


def bspl_photastrom_gp_orbit_phot_grad_pairs() -> list[tuple[str, str]]:
    """BSPL GP LinOrbs/AccOrbs phot + GP grad (host FD)."""
    return sorted(
        (c, m)
        for c, m in bspl_photastrom_gp_orbit_and_param23_pairs()
        if m in GP_PHOT_METHODS and "Orbs" in c
    )


def bspl_photastrom_extended_grad_pairs() -> list[tuple[str, str]]:
    """BSPL PhotAstrom extended phot + likelihood grad (host FD)."""
    methods = (
        PSBL_PHOT_METHODS
        + _BSPL_PHOTASTROM_PARAM1_CORE_AST
        + FSPL_PHOTASTROM_EXTENDED_METHODS
    )
    return sorted(
        (c, m) for c, m in bspl_photastrom_extended_pairs() if m in methods
    )


def psbl_photastrom_gp_param1_extended_grad_pairs() -> list[tuple[str, str]]:
    """PSBL GP Param1 ``get_photometry_with_gp`` + extended likelihood grad."""
    methods = GP_PHOT_METHODS + PSBL_GP_EXTENDED_METHODS
    out: list[tuple[str, str]] = []
    for pairs_fn in (psbl_gp_param1_pairs, psbl_photastrom_gp_param1_pairs):
        out.extend((c, m) for c, m in pairs_fn() if m in methods)
    return sorted(set(out))


def psbl_photastrom_gp_extended_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom GP Param1–2 extended ast likelihood grad (host FD)."""
    return sorted(
        (c, m)
        for c, m in psbl_photastrom_gp_extended_pairs()
        if c.startswith("PSBL_PhotAstrom_")
    )


def psbl_photastrom_param6_likelihood_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param6 chi2 / log-likelihood grad smoke."""
    return list(psbl_photastrom_param6_likelihood_pairs())


def psbl_photastrom_param7_ast_likelihood_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param7 ast chi2 / log-likelihood grad smoke."""
    return list(psbl_photastrom_param7_ast_likelihood_pairs())


def bsbl_photastrom_ellorbs_param1_likelihood_grad_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom EllOrbs Param1 ``get_u`` grad smoke (host FD; chi2 flat)."""
    return [
        (c, m)
        for c, m in bsbl_photastrom_ellorbs_param1_likelihood_pairs()
        if m == "get_u"
    ]


def bspl_photastrom_gp_orbit_grad_pairs() -> list[tuple[str, str]]:
    """BSPL static GP Param2/3 phot + GP grad (host FD for orbit GP classes)."""
    methods = GP_PHOT_METHODS
    return sorted(
        (c, m)
        for c, m in bspl_photastrom_gp_orbit_and_param23_pairs()
        if m in methods
        and "Orbs" not in c
        and c.endswith(("GP_Param2", "GP_Param3"))
    )


def psbl_photastrom_orbit_param1_grad_bulk_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom keplerian orbit Param1 bulk grad (FD for orbit params)."""
    methods = PSBL_PHOT_METHODS + _BSPL_PHOTASTROM_PARAM1_CORE_AST
    pair_fns = (
        psbl_photastrom_circorbs_param1_pairs,
        psbl_photastrom_ellorbs_param1_pairs,
        psbl_photastrom_accorbs_param1_pairs,
        psbl_photastrom_linorbs_param1_pairs,
    )
    out: list[tuple[str, str]] = []
    for fn in pair_fns:
        out.extend((c, m) for c, m in fn() if m in methods)
    return sorted(set(out))


def psbl_photastrom_orbit_param1_extended_grad_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom AccOrbs/LinOrbs Param1 extended grad (host FD)."""
    methods = FSPL_PHOTASTROM_EXTENDED_METHODS + (
        "get_resolved_astrometry",
        "get_resolved_lens_astrometry",
    )
    out: list[tuple[str, str]] = []
    for fn in (psbl_photastrom_accorbs_param1_pairs, psbl_photastrom_linorbs_param1_pairs):
        out.extend((c, m) for c, m in fn() if m in methods)
    return sorted(set(out))


def bsbl_photastrom_ellorbs_param1_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom EllOrbs Param1 phot + core astrometry (noPar + Par)."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "BSBL_PhotAstrom_noPar_EllOrbs_Param1",
            "BSBL_PhotAstrom_Par_EllOrbs_Param1",
        )
    )


def psbl_photastrom_circorbs_param2_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom CircOrbs Param2 phot + astrom + likelihoods."""
    return _psbl_photastrom_orbit_param2_pairs("CircOrbs")


def psbl_photastrom_ellorbs_param2_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom EllOrbs Param2 phot + astrom + likelihoods."""
    return _psbl_photastrom_orbit_param2_pairs("EllOrbs")


def _psbl_photastrom_orbit_param2_pairs(orbit: str) -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param2 with keplerian orbit (noPar + Par)."""
    no_par = f"PSBL_PhotAstrom_noPar_{orbit}_Param2"
    par = f"PSBL_PhotAstrom_Par_{orbit}_Param2"
    return _psbl_photastrom_full_pairs_for_classes((no_par, par))


def psbl_photastrom_orbit_param4_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom CircOrbs/EllOrbs Param4 phot + astrom + likelihoods."""
    classes: list[str] = []
    for orbit in ("CircOrbs", "EllOrbs"):
        for par in ("noPar", "Par"):
            classes.append(f"PSBL_PhotAstrom_{par}_{orbit}_Param4")
    return _psbl_photastrom_full_pairs_for_classes(tuple(classes))


def bsbl_photastrom_circorbs_param2_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom CircOrbs Param2 phot + core astrometry (noPar + Par)."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "BSBL_PhotAstrom_noPar_CircOrbs_Param2",
            "BSBL_PhotAstrom_Par_CircOrbs_Param2",
        )
    )


_BSBL_AST_LIKELIHOOD = (
    "get_chi2_astrometry",
    "log_likely_astrometry_each",
)


def bsbl_photastrom_circorbs_param2_phot_likelihood_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom CircOrbs Param2 phot chi2 / log-likelihood / ``get_u``."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "BSBL_PhotAstrom_noPar_CircOrbs_Param2",
        "BSBL_PhotAstrom_Par_CircOrbs_Param2",
    )
    methods = PSBL_PHOTASTROM_PHOT_LIKELIHOOD_METHODS
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def bsbl_photastrom_ellorbs_param2_phot_likelihood_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom EllOrbs Param2 phot chi2 / log-likelihood / ``get_u``."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "BSBL_PhotAstrom_noPar_EllOrbs_Param2",
        "BSBL_PhotAstrom_Par_EllOrbs_Param2",
    )
    methods = PSBL_PHOTASTROM_PHOT_LIKELIHOOD_METHODS
    return sorted(
        (c, m) for c in classes for m in methods if (c, m) in applicable
    )


def psbl_photastrom_param2_likelihood_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param2 ``get_u`` and chi2 / log-likelihood parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_Param2", "PSBL_PhotAstrom_Par_Param2")
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOTASTROM_LIKELIHOOD_METHODS
        if (c, m) in applicable
    )


def bsbl_photastrom_circorbs_param2_ast_likelihood_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom CircOrbs Param2 astrom chi2 / log-likelihood."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "BSBL_PhotAstrom_noPar_CircOrbs_Param2",
        "BSBL_PhotAstrom_Par_CircOrbs_Param2",
    )
    return sorted(
        (c, m)
        for c in classes
        for m in _BSBL_AST_LIKELIHOOD
        if (c, m) in applicable
    )


def psbl_photastrom_circorbs_ellorbs_param38_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom CircOrbs/EllOrbs Param3/8 phot + astrom + likelihoods."""
    classes: list[str] = []
    for orbit in ("CircOrbs", "EllOrbs"):
        for suffix in ("Param3", "Param8"):
            for par in ("noPar", "Par"):
                classes.append(f"PSBL_PhotAstrom_{par}_{orbit}_{suffix}")
    return _psbl_photastrom_full_pairs_for_classes(tuple(classes))


def bsbl_photastrom_linorbs_param1_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom LinOrbs Param1 phot + core astrometry (noPar + Par)."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "BSBL_PhotAstrom_noPar_LinOrbs_Param1",
            "BSBL_PhotAstrom_Par_LinOrbs_Param1",
        )
    )


def bsbl_photastrom_accorbs_param1_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom AccOrbs Param1 phot + core astrometry (noPar + Par)."""
    return _psbl_photastrom_pairs_for_classes(
        (
            "BSBL_PhotAstrom_noPar_AccOrbs_Param1",
            "BSBL_PhotAstrom_Par_AccOrbs_Param1",
        )
    )


def bsbl_photastrom_circorbs_param1_likelihood_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom CircOrbs Param1 likelihood parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "BSBL_PhotAstrom_noPar_CircOrbs_Param1",
        "BSBL_PhotAstrom_Par_CircOrbs_Param1",
    )
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOTASTROM_LIKELIHOOD_METHODS
        if (c, m) in applicable
    )


def bsbl_photastrom_ellorbs_param1_likelihood_pairs() -> list[tuple[str, str]]:
    """BSBL PhotAstrom EllOrbs Param1 likelihood parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = (
        "BSBL_PhotAstrom_noPar_EllOrbs_Param1",
        "BSBL_PhotAstrom_Par_EllOrbs_Param1",
    )
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOTASTROM_LIKELIHOOD_METHODS
        if (c, m) in applicable
    )


def psbl_photastrom_param6_likelihood_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param6 static + orbit likelihood parity."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = sorted(
        {
            c
            for c, _ in applicable
            if c.startswith("PSBL_PhotAstrom_")
            and "Param6" in c
            and "GP" not in c
            and not any(s in c for s in SKIP_CLASS_SUBSTR)
        }
    )
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOTASTROM_LIKELIHOOD_METHODS
        if (c, m) in applicable
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


def psbl_photastrom_param3_likelihood_pairs() -> list[tuple[str, str]]:
    """PSBL PhotAstrom Param3 likelihood parity (log10 thetaE layout)."""
    import bagle.model_jax as model_jax
    from bagle.jax.migration_tasks import applicable_task_pairs

    applicable = set(applicable_task_pairs(model_jax))
    classes = ("PSBL_PhotAstrom_noPar_Param3", "PSBL_PhotAstrom_Par_Param3")
    return sorted(
        (c, m)
        for c in classes
        for m in PSBL_PHOTASTROM_LIKELIHOOD_METHODS
        if (c, m) in applicable
    )


def time_grid_phot(instance) -> np.ndarray:
    t0 = float(instance.t0)
    tE = float(instance.tE)
    return np.linspace(t0 - 3.0 * tE, t0 + 3.0 * tE, 80)


def time_grid_ast(instance, n: int = 60) -> np.ndarray:
    t0 = float(instance.t0)
    tE = float(instance.tE)
    return np.linspace(t0 - 3.0 * tE, t0 + 3.0 * tE, n)


def synthetic_phot_obs(
    instance,
    t: np.ndarray,
    *,
    mag_offset: float = 0.05,
    err: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic photometry obs offset from model at the current init vector."""
    mag = np.asarray(instance.get_photometry(t), dtype=np.float64)
    mag_obs = mag + mag_offset
    mag_err = np.full_like(t, err, dtype=np.float64)
    return mag_obs, mag_err


def synthetic_ast_obs(
    instance,
    t: np.ndarray,
    *,
    pos_offset: tuple[float, float] = (0.001, 0.001),
    err: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic astrometry obs offset from model at the current init vector."""
    pos = np.asarray(instance.get_astrometry(t), dtype=np.float64)
    x_obs = pos[:, 0] + pos_offset[0]
    y_obs = pos[:, 1] + pos_offset[1]
    x_err = np.full_like(t, err, dtype=np.float64)
    y_err = np.full_like(t, err, dtype=np.float64)
    return x_obs, y_obs, x_err, y_err


def call_method(
    instance,
    method_name: str,
    t: np.ndarray,
    *,
    fixed_phot: tuple[np.ndarray, np.ndarray] | None = None,
    fixed_ast: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
):
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
        if fixed_phot is not None:
            mag, err = fixed_phot
        else:
            mag, err = synthetic_phot_obs(instance, t)
        return method(t, mag, err, **kwargs)
    if method_name in AST_LIKELIHOOD_METHODS:
        if fixed_ast is not None:
            x_obs, y_obs, x_err, y_err = fixed_ast
        else:
            x_obs, y_obs, x_err, y_err = synthetic_ast_obs(instance, t)
        return method(t, x_obs, y_obs, x_err, y_err, **kwargs)
    return method(t, **kwargs)


INIT_PARAM_SKIP = frozenset({"self", "raL", "decL", "obsLocation"})
# Host/JAX-eval FD skips ``root_tol``: perturbing it breaks binary-lens root
# finding and yields NaN photometry / astrometry at the fixture point.
GRAD_FD_SKIP_PARAMS = frozenset({"root_tol"})
_BSBL_PARAM1_PHOT_AST_LIKELIHOOD_GRAD_METHODS = frozenset(
    (
        "get_photometry",
        "get_centroid_shift",
        "get_astrometry",
        "get_chi2_photometry",
        "get_chi2_astrometry",
        "log_likely_photometry_each",
        "log_likely_astrometry_each",
    )
)
INT_INIT_PARAMS = frozenset(
    {
        "n_outline",
        "n_outline_pri",
        "n_outline_sec",
    }
)
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


_DAYS_PER_YEAR = 365.25


def _refresh_psbl_physical_base(instance) -> None:
    """Recompute PSBL/FSBL PhotAstrom mass/distance derived fields.

    Parameters
    ----
    instance
        Model instance whose ``mLp``, ``mLs``, ``dL``, ``dS``, and proper motions
        were just updated by ``scatter_init_vector``.
    """
    import astropy.constants as const
    import astropy.units as units

    dL = float(instance.dL)
    dS = float(instance.dS)
    inv_dist_diff = (1.0 / (dL * units.pc)) - (1.0 / (dS * units.pc))
    piRel = units.rad * units.au * inv_dist_diff
    instance.piRel = piRel.to("mas").value

    piS = (1.0 / dS) * (units.rad * units.au / units.pc)
    piL = (1.0 / dL) * (units.rad * units.au / units.pc)
    instance.piS = piS.to("mas").value
    instance.piL = piL.to("mas").value

    instance.muRel = np.asarray(instance.muS, dtype=np.float64) - np.asarray(
        instance.muL, dtype=np.float64
    )
    instance.muRel_amp = float(np.linalg.norm(instance.muRel))
    instance.mL = float(instance.mLp) + float(instance.mLs)

    thetaE = units.rad * np.sqrt(
        (4.0 * const.G * instance.mL * units.M_sun / const.c ** 2) * inv_dist_diff
    )
    instance.thetaE_amp = thetaE.to("mas").value
    instance.thetaE_hat = instance.muRel / instance.muRel_amp
    instance.muRel_hat = instance.thetaE_hat
    instance.thetaE = instance.thetaE_amp * instance.thetaE_hat

    instance.tE = (instance.thetaE_amp / instance.muRel_amp) * _DAYS_PER_YEAR

    m1 = (
        units.rad ** 2
        * (4 * const.G * float(instance.mLp) * units.Msun / const.c ** 2)
        * inv_dist_diff
    )
    m2 = (
        units.rad ** 2
        * (4 * const.G * float(instance.mLs) * units.Msun / const.c ** 2)
        * inv_dist_diff
    )
    instance.m1 = m1.to(units.arcsec ** 2).value
    instance.m2 = m2.to(units.arcsec ** 2).value

    instance.piE_amp = instance.piRel / instance.thetaE_amp
    instance.piE = instance.piE_amp * instance.thetaE_hat
    instance.piE_E, instance.piE_N = instance.piE
    instance.q = float(instance.mLs) / float(instance.mLp)


def _refresh_psbl_keplerian_orbit_alpha(instance) -> None:
    """Update Keplerian orbital geometry (``alpha_rad``, ``sep``, period) at ``tp``.

    Parameters
    ----
    instance
        Keplerian PSBL/FSBL PhotAstrom instance with ``a``, ``tp``, and mass fields.
    """
    import astropy.constants as const
    import astropy.units as units
    import bagle.orbits as orbits

    a = float(instance.a)
    instance.sep = a
    instance.aleph_sec = (float(instance.mLp) / float(instance.mL)) * a
    instance.aleph = a - instance.aleph_sec
    a_AU = float(instance.dL) * (a * 1e-3) * units.AU
    mL = float(instance.mL) * units.Msun
    p = (2 * np.pi * np.sqrt(a_AU ** 3 / (const.G * mL))).to("day")
    instance.p = p.value

    orb = orbits.Orbit()
    orb.w = float(instance.omega_pri)
    orb.o = float(instance.big_omega_sec)
    orb.i = float(instance.i)
    orb.e = float(instance.e)
    orb.tp = float(instance.tp)
    orb.aleph = instance.aleph * 1e-3
    orb.aleph2 = instance.aleph_sec * 1e-3
    orb.p = instance.p
    x, y, x2, y2 = orb.oal2xy(np.array([float(instance.tp)]))
    instance.alpha_rad = np.arctan2(x - x2, y - y2)[0]
    instance.alpha = np.rad2deg(instance.alpha_rad)


def _refresh_psbl_com_u0_geometry(instance) -> None:
    """Re-derive geometric ``t0``/``u0`` from perturbed ``t0_com``/``beta_com``.

    Mirrors ``PSBL_PhotAstrom_*Orbs_Param1.__init__`` after orbital elements are set.
    """
    import bagle.frame_convert as fc
    from bagle.model import u0_hat_from_thetaE_hat

    orbit_flag = getattr(instance, "orbitFlag", False)
    if orbit_flag == "Keplerian" and hasattr(instance, "a"):
        _refresh_psbl_keplerian_orbit_alpha(instance)
    elif hasattr(instance, "alpha"):
        instance.alpha_rad = float(instance.alpha) * np.pi / 180.0

    instance.phi_rad = instance.alpha_rad - np.arctan2(
        instance.piE[0], instance.piE[1]
    )
    instance.u0_hat_com = u0_hat_from_thetaE_hat(
        instance.thetaE_hat, float(instance.beta_com)
    )
    instance.u0_amp_com = float(instance.beta_com) / instance.thetaE_amp
    instance.u0_com = np.abs(instance.u0_amp_com) * instance.u0_hat_com

    u0_x_out, u0_y_out, t0_out = fc.convert_u0_t0_psbl(
        t0_in=float(instance.t0_com),
        u0_x_in=float(instance.u0_com[0]),
        u0_y_in=float(instance.u0_com[1]),
        tE=float(instance.tE),
        theta_E=float(instance.thetaE_amp),
        q=float(instance.q),
        phi=float(instance.phi_rad),
        sep=float(instance.sep),
        mu_rel_x=float(instance.muRel[0]),
        mu_rel_y=float(instance.muRel[1]),
        coords_in="COM",
        coords_out="geom_mid",
    )
    instance.u0 = np.array([u0_x_out, u0_y_out], dtype=np.float64)
    instance.u0_amp = np.sqrt(instance.u0[0] ** 2 + instance.u0[1] ** 2)
    instance.t0 = t0_out
    instance.beta = instance.u0_amp * instance.thetaE_amp
    instance.thetaS0 = instance.u0 * instance.thetaE_amp
    instance.xL0 = instance.xS0 - (instance.thetaS0 * 1e-3)
    instance.xL0_E, instance.xL0_N = instance.xL0
    thetaS0_com = instance.u0_com * instance.thetaE_amp
    instance.xL0_com = instance.xS0 - (thetaS0_com * 1e-3)


def _refresh_psbl_prim_u0_geometry(instance) -> None:
    """Re-derive geometric ``t0``/``u0`` from perturbed ``t0_p``/``beta_p``.

    Covers Param7 static, linear, accelerated, and Keplerian PhotAstrom layouts.
    """
    import bagle.frame_convert as fc
    from bagle.model import u0_hat_from_thetaE_hat

    orbit_flag = getattr(instance, "orbitFlag", False)
    if orbit_flag == "Keplerian" and hasattr(instance, "a"):
        _refresh_psbl_keplerian_orbit_alpha(instance)
        instance.phi_piE_rad = np.arctan2(instance.piE[0], instance.piE[1])
        instance.phi_rad = instance.alpha_rad - instance.phi_piE_rad
    else:
        if hasattr(instance, "alpha"):
            instance.alpha_rad = float(instance.alpha) * np.pi / 180.0
        instance.phi_piE_rad = np.arctan2(instance.piE[0], instance.piE[1])
        instance.phi_rad = instance.alpha_rad - instance.phi_piE_rad

    instance.u0_hat_p = u0_hat_from_thetaE_hat(
        instance.thetaE_hat, float(instance.beta_p)
    )
    instance.u0_amp_p = float(instance.beta_p) / instance.thetaE_amp
    instance.u0_p = np.abs(instance.u0_amp_p) * instance.u0_hat_p

    u0_x_out, u0_y_out, t0_out = fc.convert_u0_t0_psbl(
        t0_in=float(instance.t0_p),
        u0_x_in=float(instance.u0_p[0]),
        u0_y_in=float(instance.u0_p[1]),
        tE=float(instance.tE),
        theta_E=float(instance.thetaE_amp),
        q=float(instance.q),
        phi=float(instance.phi_rad),
        sep=float(instance.sep),
        mu_rel_x=float(instance.muRel[0]),
        mu_rel_y=float(instance.muRel[1]),
        coords_in="prim_center",
        coords_out="geom_mid",
    )
    instance.u0 = np.array([u0_x_out, u0_y_out], dtype=np.float64)
    instance.u0_amp = np.sqrt(instance.u0[0] ** 2 + instance.u0[1] ** 2)
    instance.t0 = t0_out
    instance.beta = instance.u0_amp * instance.thetaE_amp
    instance.thetaS0 = instance.u0 * instance.thetaE_amp
    instance.xL0 = instance.xS0 - (instance.thetaS0 * 1e-3)

    if orbit_flag == "Keplerian" and hasattr(instance, "a"):
        u0_x_com, u0_y_com, t0_com = fc.convert_u0_t0_psbl(
            t0_in=float(instance.t0),
            u0_x_in=float(instance.u0[0]),
            u0_y_in=float(instance.u0[1]),
            tE=float(instance.tE),
            theta_E=float(instance.thetaE_amp),
            q=float(instance.q),
            phi=float(instance.phi_rad),
            sep=float(instance.a),
            mu_rel_x=float(instance.muRel[0]),
            mu_rel_y=float(instance.muRel[1]),
            coords_in="geom_mid",
            coords_out="COM",
        )
        instance.t0_com = t0_com
        instance.u0_com = np.array([u0_x_com, u0_y_com], dtype=np.float64)
        instance.u0_amp_com = np.sqrt(instance.u0_com[0] ** 2 + instance.u0_com[1] ** 2)
        instance.beta_com = instance.u0_amp_com * instance.thetaE_amp
        instance.u0_hat_com = u0_hat_from_thetaE_hat(
            instance.thetaE_hat, instance.beta_com
        )
        instance.u0_com = np.abs(instance.u0_amp_com) * instance.u0_hat_com


def _refresh_psbl_param4_heliocentric_geometry(instance) -> None:
    """Re-derive Param4/8 heliocentric COM init after ``scatter_init_vector``.

    Mirrors ``PSBL_PhotAstromParam4.__init__`` / ``Param8`` derived fields
    (``t0``, ``u0``, ``xL0``, ``muL``) when fitter uses ``t0_com``/``u0_amp_com``
    without ``beta_com`` (static FSBL/PSBL PhotAstrom Param4/8).
    """
    import astropy.constants as const
    import astropy.units as units

    from bagle.model import u0_hat_from_thetaE_hat

    piE_E = float(instance.piE[0])
    piE_N = float(instance.piE[1])
    instance.piE = np.array([piE_E, piE_N], dtype=np.float64)
    instance.alpha_rad = float(instance.alpha) * np.pi / 180.0
    instance.phi_rad = instance.alpha_rad - np.arctan2(piE_E, piE_N)
    q = float(instance.q)
    qeff = (1.0 - q) / (1.0 + q)
    instance.t0 = (
        float(instance.t0_com)
        - 0.5
        * qeff
        * float(instance.tE)
        * float(instance.sep)
        * np.cos(instance.phi_rad)
        / float(instance.thetaE_amp)
    )
    instance.u0_amp = (
        float(instance.u0_amp_com)
        - 0.5
        * qeff
        * float(instance.sep)
        * np.sin(instance.phi_rad)
        / float(instance.thetaE_amp)
    )
    instance.beta = instance.u0_amp * instance.thetaE_amp
    instance.piE_amp = np.linalg.norm(instance.piE)
    instance.piRel = instance.piE_amp * instance.thetaE_amp
    instance.muRel_amp = instance.thetaE_amp / (float(instance.tE) / _DAYS_PER_YEAR)
    instance.piL = instance.piRel + float(instance.piS)
    kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
    kappa = kappa_tmp.to(
        units.mas / units.Msun, equivalencies=units.dimensionless_angles()
    ).value
    instance.mL = instance.thetaE_amp ** 2 / (instance.piRel * kappa)
    instance.mLp = instance.mL / (1.0 + instance.q)
    instance.mLs = instance.mLp * instance.q
    dL = (instance.piL * units.mas).to(
        units.parsec, equivalencies=units.parallax()
    )
    dS = (instance.piS * units.mas).to(
        units.parsec, equivalencies=units.parallax()
    )
    instance.dL = dL.to("pc").value
    instance.dS = dS.to("pc").value
    instance.thetaE_hat = instance.piE / instance.piE_amp
    instance.muRel_hat = instance.thetaE_hat
    instance.thetaE = instance.thetaE_amp * instance.thetaE_hat
    instance.muRel = instance.muRel_amp * instance.thetaE_hat
    instance.muL = np.asarray(instance.muS, dtype=np.float64) - instance.muRel
    instance.u0_hat = u0_hat_from_thetaE_hat(instance.thetaE_hat, instance.beta)
    instance.u0 = np.abs(instance.u0_amp) * instance.u0_hat
    instance.thetaS0 = instance.u0 * instance.thetaE_amp
    instance.xL0 = instance.xS0 - (instance.thetaS0 * 1e-3)


def _refresh_psbl_photastrom_physical(instance) -> None:
    """Full derived-geometry refresh for PSBL/FSBL PhotAstrom physical layouts."""
    _refresh_psbl_physical_base(instance)
    if hasattr(instance, "t0_com") and hasattr(instance, "beta_com"):
        _refresh_psbl_com_u0_geometry(instance)
    elif hasattr(instance, "t0_p") and hasattr(instance, "beta_p"):
        _refresh_psbl_prim_u0_geometry(instance)


def _refresh_bspl_phot_static(instance) -> None:
    """Refresh BSPL phot-only reduced-parameter derived geometry."""
    from bagle.model import u0_hat_from_thetaE_hat

    instance.piE_amp = np.linalg.norm(instance.piE)
    instance.piE_E, instance.piE_N = instance.piE
    instance.thetaE_hat = instance.piE / instance.piE_amp
    instance.muRel_hat = instance.thetaE_hat
    instance.u0_hat = u0_hat_from_thetaE_hat(
        instance.thetaE_hat, float(instance.u0_amp)
    )
    instance.u0 = np.abs(float(instance.u0_amp)) * instance.u0_hat
    instance.phi_rad = float(instance.phi) * np.pi / 180.0
    instance.phi_piE_rad = np.arctan2(instance.piE[0], instance.piE[1])
    instance.phi_rho1_rad = instance.phi_piE_rad + instance.phi_rad
    instance.t0_pri = float(instance.t0)
    instance.u0_amp_pri = float(instance.u0_amp)
    instance.u0_pri = instance.u0
    sep_vec = float(instance.sep) * np.array(
        (np.sin(instance.phi_rho1_rad), np.cos(instance.phi_rho1_rad)),
        dtype=np.float64,
    )
    instance.u0_amp_sec = instance.u0_amp_pri + np.dot(sep_vec, instance.u0_hat)
    instance.u0_sec = instance.u0_amp_sec * instance.u0_hat


def _refresh_bspl_photastrom_param1(instance) -> None:
    """Refresh BSPL PhotAstrom Param1 physical derived geometry."""
    import astropy.constants as const
    import astropy.units as units
    from bagle.model import u0_hat_from_thetaE_hat

    if hasattr(instance, "dL_dS"):
        instance.dS = float(instance.dL) / float(instance.dL_dS)
    inv_dist_diff = (1.0 / (float(instance.dL) * units.pc)) - (
        1.0 / (float(instance.dS) * units.pc)
    )
    piRel = units.rad * units.au * inv_dist_diff
    instance.piRel = piRel.to("mas").value
    piS = (1.0 / float(instance.dS)) * (units.rad * units.au / units.pc)
    piL = (1.0 / float(instance.dL)) * (units.rad * units.au / units.pc)
    instance.piS = piS.to("mas").value
    instance.piL = piL.to("mas").value

    instance.muRel = np.asarray(instance.muS, dtype=np.float64) - np.asarray(
        instance.muL, dtype=np.float64
    )
    instance.muRel_amp = float(np.linalg.norm(instance.muRel))
    instance.muRel_E, instance.muRel_N = instance.muRel

    thetaE = units.rad * np.sqrt(
        (4.0 * const.G * float(instance.mL) * units.M_sun / const.c ** 2)
        * inv_dist_diff
    )
    instance.thetaE_amp = thetaE.to("mas").value
    instance.thetaE_hat = instance.muRel / instance.muRel_amp
    instance.muRel_hat = instance.thetaE_hat
    instance.thetaE = instance.thetaE_amp * instance.thetaE_hat
    instance.thetaE_E, instance.thetaE_N = instance.thetaE

    instance.u0_hat = u0_hat_from_thetaE_hat(
        instance.thetaE_hat, float(instance.beta)
    )
    instance.u0_amp = float(instance.beta) / instance.thetaE_amp
    instance.u0 = np.abs(instance.u0_amp) * instance.u0_hat
    instance.thetaS0 = instance.u0 * instance.thetaE_amp
    instance.xL0 = instance.xS0 - (instance.thetaS0 * 1e-3)

    instance.piE_amp = instance.piRel / instance.thetaE_amp
    instance.piE = instance.piE_amp * instance.thetaE_hat
    instance.piE_E, instance.piE_N = instance.piE
    instance.tE = (instance.thetaE_amp / instance.muRel_amp) * _DAYS_PER_YEAR

    if hasattr(instance, "alpha"):
        instance.alpha_rad = float(instance.alpha) * np.pi / 180.0
    instance.t0_pri = float(instance.t0)
    instance.xS0_pri = np.asarray(instance.xS0, dtype=np.float64)
    instance.u0_amp_pri = instance.u0_amp
    instance.u0_pri = instance.u0
    sep_vec = float(instance.sep) * np.array(
        (np.sin(instance.alpha_rad), np.cos(instance.alpha_rad)),
        dtype=np.float64,
    )
    instance.xS0_sec = instance.xS0_pri + (sep_vec * 1e-3)


def _refresh_fspl_photastrom_param1(instance) -> None:
    """Refresh FSPL PhotAstrom Param1 physical derived geometry."""
    import astropy.constants as const
    import astropy.units as units
    from bagle.model import u0_hat_from_thetaE_hat

    if hasattr(instance, "dL_dS"):
        instance.dS = float(instance.dL) / float(instance.dL_dS)
    inv_dist_diff = (1.0 / (float(instance.dL) * units.pc)) - (
        1.0 / (float(instance.dS) * units.pc)
    )
    piRel = units.rad * units.au * inv_dist_diff
    instance.piRel = piRel.to("mas").value
    piS = (1.0 / float(instance.dS)) * (units.rad * units.au / units.pc)
    piL = (1.0 / float(instance.dL)) * (units.rad * units.au / units.pc)
    instance.piS = piS.to("mas").value
    instance.piL = piL.to("mas").value

    instance.muRel = np.asarray(instance.muS, dtype=np.float64) - np.asarray(
        instance.muL, dtype=np.float64
    )
    instance.muRel_E, instance.muRel_N = instance.muRel
    instance.muRel_amp = float(np.linalg.norm(instance.muRel))

    thetaE = units.rad * np.sqrt(
        (4.0 * const.G * float(instance.mL) * units.M_sun / const.c ** 2)
        * inv_dist_diff
    )
    instance.thetaE_amp = thetaE.to("mas").value
    instance.thetaE_hat = instance.muRel / instance.muRel_amp
    instance.muRel_hat = instance.thetaE_hat
    instance.thetaE = instance.thetaE_amp * instance.thetaE_hat
    instance.thetaE_E, instance.thetaE_N = instance.thetaE

    instance.u0_hat = u0_hat_from_thetaE_hat(
        instance.thetaE_hat, float(instance.beta)
    )
    instance.u0_amp = float(instance.beta) / instance.thetaE_amp
    instance.u0 = np.abs(instance.u0_amp) * instance.u0_hat
    instance.thetaS0 = instance.u0 * instance.thetaE_amp
    instance.xL0 = instance.xS0 - (instance.thetaS0 * 1e-3)

    instance.piE_amp = instance.piRel / instance.thetaE_amp
    instance.piE = instance.piE_amp * instance.thetaE_hat
    instance.piE_E, instance.piE_N = instance.piE
    instance.tE = (instance.thetaE_amp / instance.muRel_amp) * _DAYS_PER_YEAR


def _refresh_derived_geometry(instance) -> None:
    """Recompute derived geometry after ``scatter_init_vector`` for FD smoke.

    Host models cache ``piE``, ``thetaE_hat``, lens positions, and orbital
    elements at construction; perturbing the packed init vector without this
    refresh leaves stale geometry and breaks central-difference grad checks.

    Parameters
    ----
    instance
        NumPy or JAX model instance whose numeric init fields were just updated.
    """
    from bagle.model import u0_hat_from_thetaE_hat

    if hasattr(instance, "piE_E"):
        piE_E = float(instance.piE_E)
        if hasattr(instance, "piEN_piEE"):
            piE_N = float(instance.piEN_piEE) * piE_E
            instance.piE_N = piE_N
        else:
            piE_N = float(getattr(instance, "piE_N", instance.piE[1]))
        instance.piE = np.array([piE_E, piE_N], dtype=np.float64)

    if hasattr(instance, "alpha"):
        instance.alpha_rad = float(instance.alpha) * np.pi / 180.0

    if (
        hasattr(instance, "t0_com")
        and hasattr(instance, "u0_amp_com")
        and not hasattr(instance, "beta_com")
    ):
        _refresh_psbl_param4_heliocentric_geometry(instance)
        return None

    if hasattr(instance, "t0_prim") and hasattr(instance, "thetaE_amp"):
        instance.phi_rad = instance.alpha_rad - np.arctan2(
            instance.piE[0], instance.piE[1]
        )
        instance.t0 = (
            instance.t0_prim
            - 0.5
            * instance.tE
            * instance.sep
            * np.cos(instance.phi_rad)
            / instance.thetaE_amp
        )
        instance.u0_amp = (
            instance.u0_amp_prim
            - 0.5
            * instance.sep
            * np.sin(instance.phi_rad)
            / instance.thetaE_amp
        )
        instance.piE_amp = np.linalg.norm(instance.piE)
        instance.thetaE_hat = instance.piE / instance.piE_amp
        return None

    if hasattr(instance, "mLp") and hasattr(instance, "mLs") and hasattr(instance, "dL"):
        _refresh_psbl_photastrom_physical(instance)
        return None

    if (
        hasattr(instance, "mL")
        and hasattr(instance, "beta")
        and hasattr(instance, "mag_src_pri")
    ):
        _refresh_bspl_photastrom_param1(instance)
        return None

    if (
        hasattr(instance, "mL")
        and hasattr(instance, "beta")
        and hasattr(instance, "radiusS")
        and not hasattr(instance, "mLp")
    ):
        _refresh_fspl_photastrom_param1(instance)
        return None

    phot_only = getattr(instance, "photometryFlag", False) and not getattr(
        instance, "astrometryFlag", False
    )
    orbit = getattr(instance, "orbitFlag", False)

    if phot_only and hasattr(instance, "mag_src_pri") and hasattr(instance, "mag_src_sec"):
        _refresh_bspl_phot_static(instance)
        return None

    if orbit is True and hasattr(instance, "get_me_some_orbital_parameters"):
        if hasattr(instance, "aleph") and hasattr(instance, "aleph_sec"):
            instance.sep = float(instance.aleph) + float(instance.aleph_sec)
        ecc, i, o, w, p, tp = instance.get_me_some_orbital_parameters(
            instance.t0,
            instance.sep,
            instance.r_s,
            instance.a_s,
            instance.v_para,
            instance.v_perp,
            instance.v_rad,
        )
        instance.w = w
        instance.o = o
        instance.i = i
        instance.e = ecc
        instance.tp = tp
        instance.p = p
        instance.piE_amp = np.linalg.norm(instance.piE)
        instance.thetaE_hat = instance.piE / instance.piE_amp
        instance.muRel_hat = instance.thetaE_hat
        instance.u0_hat = u0_hat_from_thetaE_hat(
            instance.thetaE_hat, instance.u0_amp
        )
        instance.u0 = np.abs(instance.u0_amp) * instance.u0_hat
        instance.m1 = 1.0 / (1.0 + instance.q)
        instance.m2 = instance.q / (1.0 + instance.q)
        import bagle.orbits as orbits

        orb = orbits.Orbit()
        orb.w = w
        orb.o = o
        orb.i = i
        orb.e = ecc
        orb.tp = tp
        orb.aleph2 = instance.aleph_sec
        orb.aleph = instance.aleph
        orb.p = p
        x, y, x2, y2 = orb.oal2xy(np.array([tp]))
        instance.alpha_rad = np.arctan2(x - x2, y - y2)[0]
        instance.alpha = np.rad2deg(instance.alpha_rad)
        instance.phi_rho1_rad = instance.alpha
        instance.phi_piE_rad = np.arctan2(instance.piE[0], instance.piE[1])
        instance.phi_rad = instance.phi_rho1_rad - instance.phi_piE_rad
        instance.phi = np.rad2deg(instance.phi_rad)
        return

    if phot_only and hasattr(instance, "xL1_over_theta") and hasattr(instance, "phi"):
        instance.piE_amp = np.linalg.norm(instance.piE)
        instance.phi_rad = float(instance.phi) * np.pi / 180.0
        instance.phi_piE_rad = np.arctan2(instance.piE[0], instance.piE[1])
        instance.phi_rho1_rad = instance.phi_piE_rad + instance.phi_rad
        instance.xL1_over_theta = np.array(
            [
                0.5 * instance.sep * np.sin(instance.phi_rho1_rad),
                0.5 * instance.sep * np.cos(instance.phi_rho1_rad),
            ],
            dtype=np.float64,
        )
        instance.xL2_over_theta = np.array(
            [
                -0.5 * instance.sep * np.sin(instance.phi_rho1_rad),
                -0.5 * instance.sep * np.cos(instance.phi_rho1_rad),
            ],
            dtype=np.float64,
        )
        instance.thetaE_hat = instance.piE / instance.piE_amp
        instance.muRel_hat = instance.thetaE_hat
        instance.u0_hat = u0_hat_from_thetaE_hat(
            instance.thetaE_hat, instance.u0_amp
        )
        instance.u0 = np.abs(instance.u0_amp) * instance.u0_hat
        instance.m1 = 1.0 / (1.0 + instance.q)
        instance.m2 = instance.q / (1.0 + instance.q)


def scatter_init_vector(instance, vec, init_names: tuple[str, ...]) -> None:
    """Write a packed init vector back onto a model instance (filter-0 lists).

    After scattering raw init values, calls ``_refresh_derived_geometry`` so
    host FD grad smoke sees consistent derived fields (``piE``, ``u0``, orbits).

    Parameters
    ----
    instance
        Model instance to mutate in place.
    vec : array-like
        1D init vector aligned with ``init_names``.
    init_names : tuple[str, ...]
        Ordered numeric Param-mixin parameter names from ``pack_init_vector``.
    """
    for i, name in enumerate(init_names):
        val = float(vec[i])
        if name in _ARRAY_COMPONENT:
            attr, idx = _ARRAY_COMPONENT[name]
            arr = np.array(getattr(instance, attr), dtype=np.float64, copy=True)
            arr.reshape(-1)[idx] = val
            setattr(instance, attr, arr)
        elif name in LIST_INIT_PARAMS:
            setattr(instance, name, [val])
        elif name in INT_INIT_PARAMS:
            setattr(instance, name, int(round(val)))
        elif name == "thetaE":
            instance.thetaE_amp = val
        elif name == "log10_thetaE":
            instance.thetaE_amp = 10.0 ** val
        else:
            setattr(instance, name, val)
    _refresh_derived_geometry(instance)


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


_COMPANION_INIT_ALIASES: dict[str, tuple[str, ...]] = {
    "t0": ("t0_prim", "t0_p"),
    "u0_amp": ("u0_amp_prim",),
    "piE_N": ("piEN_piEE",),
}


def _init_value_for_base_name(v, init_names: tuple[str, ...], base_name: str):
    from bagle.jax.geometry import derive_psbl_param4_heliocentric

    if base_name == "t0" and "t0_com" in init_names:
        t0, _ = derive_psbl_param4_heliocentric(
            v[init_names.index("t0_com")],
            v[init_names.index("u0_amp_com")],
            v[init_names.index("tE")],
            v[init_names.index("thetaE")],
            v[init_names.index("piE_E")],
            v[init_names.index("piE_N")],
            v[init_names.index("q")],
            v[init_names.index("sep")],
            v[init_names.index("alpha")],
        )
        return t0
    if base_name == "u0_amp" and "u0_amp_com" in init_names:
        _, u0_amp = derive_psbl_param4_heliocentric(
            v[init_names.index("t0_com")],
            v[init_names.index("u0_amp_com")],
            v[init_names.index("tE")],
            v[init_names.index("thetaE")],
            v[init_names.index("piE_E")],
            v[init_names.index("piE_N")],
            v[init_names.index("q")],
            v[init_names.index("sep")],
            v[init_names.index("alpha")],
        )
        return u0_amp
    if base_name in init_names:
        return v[init_names.index(base_name)]
    if base_name == "sep" and "aleph" in init_names and "aleph_sec" in init_names:
        return v[init_names.index("aleph")] + v[init_names.index("aleph_sec")]
    if base_name == "piE_N" and "piEN_piEE" in init_names:
        return v[init_names.index("piE_E")] * v[init_names.index("piEN_piEE")]
    for alias in _COMPANION_INIT_ALIASES.get(base_name, ()):
        if alias in init_names:
            return v[init_names.index(alias)]
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
    if layout.mag_fitter == "mag_base" or (
        hasattr(jax_inst, "mag_base") and not hasattr(jax_inst, "mag_src")
    ):
        mag_base = float(np.asarray(jax_inst.mag_base).reshape(-1)[0])
        return float(
            mag_src_from_fitter(
                jnp.asarray(mag_base), "mag_base", jnp.asarray(b_sff)
            )
        )
    return float(np.asarray(jax_inst.mag_src).reshape(-1)[0])


def _bspl_mags_from_init(v, init_names: tuple[str, ...], layout, b_sff):
    from bagle.jax.geometry import mag_src_from_fitter

    b_sff_j = jnp.asarray(b_sff, dtype=jnp.float64)
    mag_pri = _init_param(v, init_names, "mag_src_pri", None)
    mag_sec = _init_param(v, init_names, "mag_src_sec", None)
    if mag_pri is not None and mag_sec is not None:
        return mag_pri, mag_sec
    if "fratio_bin" in init_names and "mag_base" in init_names:
        mag_base = _init_param(v, init_names, "mag_base")
        fratio = _init_param(v, init_names, "fratio_bin", 1.0)
        b = jnp.asarray(b_sff, dtype=jnp.float64)
        mag_pri = mag_base - 2.5 * jnp.log10(b) + 2.5 * jnp.log10(1.0 + fratio)
        mag_sec = mag_base - 2.5 * jnp.log10(b) + 2.5 * jnp.log10(
            1.0 + 1.0 / fratio
        )
        return mag_pri, mag_sec
    mag = _mag_from_init(v, init_names, layout, b_sff)
    if mag is not None:
        return mag, mag
    raise ValueError(
        f"cannot resolve BSPL source magnitudes from init parameters {init_names!r}"
    )


_DAYS_PER_YEAR = 365.25


def _psbl_reduced_static_geom(base, base_names: tuple[str, ...]) -> dict:
    """Static PSBL reduced fitter (Param2/4/GP) geometry for JAX grad smoke."""
    from bagle.jax.geometry import unpack_base_params
    from bagle.jax_physics import derive_psbl_static_geometry

    p = unpack_base_params(base_names, base)
    phi_key = "phi" if "phi" in p else "alpha"
    phi = p[phi_key]
    t0 = p["t0"]
    u0_amp = p["u0_amp"]
    tE = p["tE"]
    if "thetaE" in p:
        thetaE_amp = p["thetaE"]
    else:
        thetaE_amp = jnp.power(10.0, p["log10_thetaE"])
    piS = p["piS"]
    piE_E = p["piE_E"]
    piE_N = p["piE_N"]
    xS0 = jnp.stack([p["xS0_E"], p["xS0_N"]])
    muS = jnp.stack([p["muS_E"], p["muS_N"]])
    m1, m2, u0, thetaE_hat, xL1, xL2, piE_amp = derive_psbl_static_geometry(
        u0_amp, piE_E, piE_N, p["q"], p["sep"], phi
    )
    piRel = piE_amp * thetaE_amp
    piL = piRel + piS
    muRel_amp = thetaE_amp / (tE / _DAYS_PER_YEAR)
    muRel = muRel_amp * thetaE_hat
    muL = muS - muRel
    xL0 = xS0 - u0 * thetaE_amp * 1e-3
    return {
        "t0": t0,
        "tE": tE,
        "u0": u0,
        "thetaE_hat": thetaE_hat,
        "m1": m1,
        "m2": m2,
        "xL1": xL1,
        "xL2": xL2,
        "xS0": xS0,
        "xL0": xL0,
        "muS": muS,
        "muL": muL,
        "thetaE_amp": thetaE_amp,
        "piS": piS,
        "piL": piL,
        "piE_E": piE_E,
        "piE_N": piE_N,
    }


def _psbl_param1_geom(base, base_names: tuple[str, ...]) -> dict:
    from bagle.jax.geometry import derive_psbl_photastrom_param1, unpack_base_params

    p = unpack_base_params(base_names, base)
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
        m1,
        m2,
        xL1,
        xL2,
        _mLp,
        _mLs,
    ) = derive_psbl_photastrom_param1(
        p["mLp"],
        p["mLs"],
        p["t0"],
        p["xS0_E"],
        p["xS0_N"],
        p["beta"],
        p["muL_E"],
        p["muL_N"],
        p["muS_E"],
        p["muS_N"],
        p["dL"],
        p["dS"],
        p["sep"],
        p["alpha"],
    )
    return {
        "t0": p["t0"],
        "tE": tE,
        "u0": u0,
        "thetaE_hat": thetaE_hat,
        "m1": m1,
        "m2": m2,
        "xL1": xL1,
        "xL2": xL2,
        "xS0": xS0,
        "xL0": xL0,
        "muS": muS,
        "muL": muL,
        "thetaE_amp": thetaE_amp,
        "piS": piS,
        "piL": piL,
        "piE_E": piE_E,
        "piE_N": piE_N,
    }


def _psbl_geom_from_base(base, base_names: tuple[str, ...]) -> dict:
    if "mLp" in base_names:
        return _psbl_param1_geom(base, base_names)
    return _psbl_reduced_static_geom(base, base_names)


def _bspl_param1_geom_from_base(base, base_names: tuple[str, ...]) -> dict:
    """BSPL PhotAstrom Param1 physical base (mL, beta, dL, sep, alpha)."""
    from bagle.jax.geometry import unpack_base_params
    from bagle.jax_physics import derive_pspl_photastrom_param1_geometry

    p = unpack_base_params(base_names, base)
    (
        u0,
        thetaE_hat,
        tE,
        piE_E,
        piE_N,
        _xS0,
        _xL0,
        _muS,
        _muL,
        thetaE_amp,
        _piS,
        _piL,
    ) = derive_pspl_photastrom_param1_geometry(
        p["mL"],
        p["t0"],
        p["beta"],
        p["dL"],
        p["dL_dS"],
        p["xS0_E"],
        p["xS0_N"],
        p["muL_E"],
        p["muL_N"],
        p["muS_E"],
        p["muS_N"],
    )
    alpha_rad = p["alpha"] * jnp.pi / 180.0
    phi_piE = jnp.arctan2(piE_E, piE_N)
    phi_rho1 = phi_piE + alpha_rad
    sep_th = p["sep"] / thetaE_amp
    u0_sec = u0 + sep_th * jnp.stack(
        [jnp.sin(phi_rho1), jnp.cos(phi_rho1)]
    )
    return {
        "u0_pri": u0,
        "u0_sec": u0_sec,
        "thetaE_hat": thetaE_hat,
        "t0_pri": p["t0"],
        "t0_sec": p["t0"],
        "tE": tE,
        "piE_E": piE_E,
        "piE_N": piE_N,
    }


def _bspl_phot_geom_from_base(base, base_names: tuple[str, ...]) -> dict:
    from bagle.jax.geometry import unpack_base_params
    from bagle.jax_physics import u0_hat_from_thetaE_hat_jax

    p = unpack_base_params(base_names, base)
    piE = jnp.stack([p["piE_E"], p["piE_N"]])
    piE_amp = jnp.linalg.norm(piE)
    thetaE_hat = piE / piE_amp
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, p["u0_amp"])
    u0_pri = jnp.abs(p["u0_amp"]) * u0_hat
    phi_key = "phi" if "phi" in p else "alpha"
    phi_rad = p[phi_key] * jnp.pi / 180.0
    phi_piE = jnp.arctan2(p["piE_E"], p["piE_N"])
    phi_rho1 = phi_piE + phi_rad
    sep_vec = p["sep"] * jnp.stack([jnp.sin(phi_rho1), jnp.cos(phi_rho1)])
    u0_amp_sec = p["u0_amp"] + jnp.dot(sep_vec, u0_hat)
    u0_sec = u0_amp_sec * u0_hat
    out = (
        "bspl_phot",
        u0_pri,
        u0_sec,
        thetaE_hat,
        p["t0"],
        p["t0"],
        p["tE"],
        p["piE_E"],
        p["piE_N"],
    )
    (
        _tag,
        u0_pri,
        u0_sec,
        thetaE_hat,
        t0_pri,
        t0_sec,
        tE,
        piE_E,
        piE_N,
    ) = out
    return {
        "u0_pri": u0_pri,
        "u0_sec": u0_sec,
        "thetaE_hat": thetaE_hat,
        "t0_pri": t0_pri,
        "t0_sec": t0_sec,
        "tE": tE,
        "piE_E": piE_E,
        "piE_N": piE_N,
    }


_RESOLVED_AST_FD_METHODS = frozenset(
    (
        "get_resolved_astrometry",
        "get_resolved_lens_astrometry",
        "get_lens_astrometry",
    )
)

_GET_U_FD_METHODS = frozenset(("get_u",))

_SQUARED_FD_METHODS = _RESOLVED_AST_FD_METHODS | _GET_U_FD_METHODS


def _fd_scalar_from_output(out, method_name: str) -> float:
    """Reduce method output to a scalar for central finite-difference grad.

    Resolved astrometry arrays may contain NaN padding and antisymmetric lens
    components; ``nansum(arr * arr)`` avoids cancellation that zeroes FD steps.
    ``get_u`` uses the same squared objective because ``sum(u)`` can cancel when
    the separation vector rotates under parameter perturbations.

    Parameters
    ----
    out
        Forward output (array-like) from ``call_method``.
    method_name : str
        Method name; squared sum is used for resolved astrometry and ``get_u``.

    Returns
    ----
    float
        Scalar objective summed (or squared-summed) over finite entries.
    """
    arr = np.asarray(out, dtype=np.float64)
    if method_name in _SQUARED_FD_METHODS:
        return float(np.nansum(arr * arr))
    return float(np.sum(arr))


def _fd_grad_host(
    class_name: str,
    init_names: tuple[str, ...],
    vec0,
    t: np.ndarray,
    method_name: str,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central finite-difference grad w.r.t. init vector via host model forward."""
    vec0_np = np.asarray(vec0, dtype=np.float64)
    t_np = np.asarray(t, dtype=np.float64)
    fixed_phot = fixed_ast = None
    if method_name in PHOT_LIKELIHOOD_METHODS or method_name in AST_LIKELIHOOD_METHODS:
        _, inst0 = build_paired_instances(class_name)
        scatter_init_vector(inst0, vec0_np, init_names)
        if method_name in PHOT_LIKELIHOOD_METHODS:
            fixed_phot = synthetic_phot_obs(inst0, t_np)
        else:
            fixed_ast = synthetic_ast_obs(inst0, t_np)

    def _sum(vec_np: np.ndarray) -> float:
        _, inst = build_paired_instances(class_name)
        scatter_init_vector(inst, vec_np, init_names)
        out = call_method(
            inst,
            method_name,
            t_np,
            fixed_phot=fixed_phot,
            fixed_ast=fixed_ast,
        )
        return _fd_scalar_from_output(out, method_name)

    g = np.zeros(len(vec0_np), dtype=np.float64)
    for i in range(len(vec0_np)):
        if init_names[i] in GRAD_FD_SKIP_PARAMS:
            continue
        vp = vec0_np.copy()
        vm = vec0_np.copy()
        vp[i] += eps
        vm[i] -= eps
        g[i] = (_sum(vp) - _sum(vm)) / (2.0 * eps)
    return g


def _fd_grad_jax_eval(
    class_name: str,
    init_names: tuple[str, ...],
    vec0,
    t: np.ndarray,
    method_name: str,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central FD grad via ``jax/evaluate`` dispatch (jax-only FSBL layouts)."""
    vec0_np = np.asarray(vec0, dtype=np.float64)
    t_np = np.asarray(t, dtype=np.float64)
    fixed_phot = fixed_ast = None
    if method_name in PHOT_LIKELIHOOD_METHODS or method_name in AST_LIKELIHOOD_METHODS:
        _, inst0 = build_jax_eval_paired_instances(class_name)
        scatter_init_vector(inst0, vec0_np, init_names)
        if method_name in PHOT_LIKELIHOOD_METHODS:
            fixed_phot = synthetic_phot_obs(inst0, t_np)
        else:
            fixed_ast = synthetic_ast_obs(inst0, t_np)

    def _sum(vec_np: np.ndarray) -> float:
        _, inst = build_jax_eval_paired_instances(class_name)
        scatter_init_vector(inst, vec_np, init_names)
        out = call_method_via_jax_eval(
            inst,
            method_name,
            t_np,
            fixed_phot=fixed_phot,
            fixed_ast=fixed_ast,
        )
        return _fd_scalar_from_output(out, method_name)

    g = np.zeros(len(vec0_np), dtype=np.float64)
    for i in range(len(vec0_np)):
        if init_names[i] in GRAD_FD_SKIP_PARAMS:
            continue
        vp = vec0_np.copy()
        vm = vec0_np.copy()
        vp[i] += eps
        vm[i] -= eps
        g[i] = (_sum(vp) - _sum(vm)) / (2.0 * eps)
    return g


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
    """Pure-JAX grad smoke w.r.t. all numeric Param-mixin ``__init__`` parameters.

    PSBL phot-only resolved astrometry and related paths fall back to host FD
    with ``eps=1e-4`` for orbital layouts and ``1e-5`` otherwise, using the
    squared FD objective from ``_fd_scalar_from_output``.

    Parameters
    ----
    class_name : str
        JAX model class name.
    method_name : str
        Method under test.
    jax_inst
        Constructed JAX model instance.
    t : ndarray
        Evaluation times in days.
    return_names : bool, optional
        If True, return ``(grad, init_names)`` instead of grad alone.

    Returns
    ----
    ndarray or tuple[ndarray, tuple[str, ...]]
        Gradient vector w.r.t. packed init parameters, optionally with names.
    """
    import jax

    from bagle.jax.layout_registry import resolve_layout
    from bagle.jax.geometry import derive_geometry_from_layout
    from bagle.jax.bspl import bspl_photometry_jax
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
        mag, err = synthetic_phot_obs(jax_inst, t)
        return jnp.asarray(mag, dtype=jnp.float64), jnp.asarray(err, dtype=jnp.float64)

    def _ast_obs():
        x_obs, y_obs, x_err, y_err = synthetic_ast_obs(jax_inst, t)
        return (
            jnp.asarray(x_obs, dtype=jnp.float64),
            jnp.asarray(y_obs, dtype=jnp.float64),
            jnp.asarray(x_err, dtype=jnp.float64),
            jnp.asarray(y_err, dtype=jnp.float64),
        )

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

    _psbl_phot_only = ek.startswith("psbl_phot_")
    _static_psbl_photastrom = (
        ek.startswith("psbl_photastrom")
        and "none" in ek
        and "u0_amp" in base_names
    )
    _static_bspl_photastrom = (
        ek.startswith("bspl_photastrom")
        and "none" in ek
        and "u0_amp" in base_names
    )
    _bspl_physical_photastrom = (
        ek.startswith("bspl_photastrom")
        and "none" in ek
        and "mL" in base_names
    )
    _bspl_photastrom_static_jax = _static_bspl_photastrom or _bspl_physical_photastrom
    if method_name == "get_photometry_with_gp" and layout.has_gp and (
        ek.startswith("bspl_photastrom") and not _static_bspl_photastrom
    ):
        g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    if method_name == "get_photometry_with_gp" and layout.has_gp and (
        ek in pspl_phot_kinds
        or ek == "bspl_phot"
        or _psbl_phot_only
        or _static_psbl_photastrom
        or _bspl_photastrom_static_jax
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
                if _psbl_phot_only:
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
                elif _static_psbl_photastrom:
                    base = _base_vec_from_init(v, init_names, base_names)
                    geom = _psbl_geom_from_base(base, base_names)
                    m = psbl_photometry(
                        x1,
                        geom["t0"],
                        geom["tE"],
                        geom["u0"],
                        geom["thetaE_hat"],
                        geom["xL1"],
                        geom["xL2"],
                        geom["m1"],
                        geom["m2"],
                        mag,
                        b_sff=b_sff,
                        parallax_vectors=pvec,
                        piE_E=geom["piE_E"],
                        piE_N=geom["piE_N"],
                        root_tol=root_tol,
                    )
                elif ek == "bspl_phot" or _bspl_photastrom_static_jax:
                    base = _base_vec_from_init(v, init_names, base_names)
                    if _bspl_physical_photastrom:
                        bg = _bspl_param1_geom_from_base(base, base_names)
                    else:
                        bg = _bspl_phot_geom_from_base(base, base_names)
                    mag_pri, mag_sec = _bspl_mags_from_init(
                        v, init_names, layout, b_sff
                    )
                    m = bspl_photometry_jax(
                        x1,
                        bg["t0_pri"],
                        bg["t0_sec"],
                        bg["tE"],
                        bg["u0_pri"],
                        bg["u0_sec"],
                        bg["thetaE_hat"],
                        mag_pri,
                        mag_sec,
                        b_sff=b_sff,
                        pvec=pvec,
                        piE_E=bg["piE_E"],
                        piE_N=bg["piE_N"],
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
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) == 0.0:
            g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    if method_name in ("get_photometry", "get_amplification") and ek == "bspl_phot":

        def forward(v):
            base = _base_vec_from_init(v, init_names, base_names)
            (
                _tag,
                u0_pri,
                u0_sec,
                thetaE_hat,
                t0_pri,
                t0_sec,
                tE,
                piE_E,
                piE_N,
            ) = derive_geometry_from_layout("", ek, base, base_names)
            b_sff = _init_param(v, init_names, "b_sff", 1.0)
            mag_pri, mag_sec = _bspl_mags_from_init(v, init_names, layout, b_sff)
            out = bspl_photometry_jax(
                t_j,
                t0_pri,
                t0_sec,
                tE,
                u0_pri,
                u0_sec,
                thetaE_hat,
                mag_pri,
                mag_sec,
                b_sff=b_sff,
                pvec=pvec,
                piE_E=piE_E,
                piE_N=piE_N,
            )
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) == 0.0:
            g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    if method_name in ("get_photometry", "get_amplification") and _psbl_phot_only:
        if layout.orbit != "none":
            g = _fd_grad_host(
                class_name, init_names, vec0, t, method_name, eps=1e-4
            )
            if return_names:
                return g, init_names
            return g

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
        if not np.all(np.isfinite(g)):
            g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    if _psbl_phot_only and method_name in (
        "get_u",
        "get_chi2_photometry",
        "log_likely_photometry_each",
        "get_resolved_astrometry",
        "get_resolved_lens_astrometry",
    ):
        fd_eps = 1e-4 if layout.orbit != "none" else 1e-5
        g = _fd_grad_host(
            class_name, init_names, vec0, t, method_name, eps=fd_eps
        )
        if return_names:
            return g, init_names
        return g

    if ek == "bspl_phot" and method_name in (
        "get_u",
        "get_chi2_photometry",
        "log_likely_photometry_each",
        "get_resolved_astrometry",
    ):
        g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    _photastrom_ek = ek.startswith(
        ("bspl_photastrom", "psbl_photastrom", "bsbl_photastrom")
    )
    _fd_photastrom_methods = (
        ("get_u",)
        + tuple(
            m
            for m in PSBL_PHOTASTROM_AST_METHODS
            if m not in _BSPL_PHOTASTROM_PARAM1_CORE_AST
        )
    )

    if method_name in PSBL_PHOT_METHODS and _static_psbl_photastrom:
        root_tol = float(getattr(jax_inst, "root_tol", 1e-8))

        def forward(v):
            base = _base_vec_from_init(v, init_names, base_names)
            geom = _psbl_geom_from_base(base, base_names)
            b_sff = _init_param(v, init_names, "b_sff", 1.0)
            mag = _mag_from_init(v, init_names, layout, b_sff)
            if mag is None:
                mag = jnp.asarray(_mag_scalar(jax_inst, layout), dtype=jnp.float64)
            if method_name == "get_photometry":
                out = psbl_photometry(
                    t_j,
                    geom["t0"],
                    geom["tE"],
                    geom["u0"],
                    geom["thetaE_hat"],
                    geom["xL1"],
                    geom["xL2"],
                    geom["m1"],
                    geom["m2"],
                    mag,
                    b_sff=b_sff,
                    parallax_vectors=pvec,
                    piE_E=geom["piE_E"],
                    piE_N=geom["piE_N"],
                    root_tol=root_tol,
                )
            else:
                w, z1, z2 = psbl_complex_pos_static(
                    t_j,
                    geom["t0"],
                    geom["tE"],
                    geom["u0"],
                    geom["thetaE_hat"],
                    geom["xL1"],
                    geom["xL2"],
                    parallax_vectors=pvec,
                    piE_E=geom["piE_E"],
                    piE_N=geom["piE_N"],
                )
                _, amp_arr = psbl_all_arrays(w, z1, z2, geom["m1"], geom["m2"], root_tol)
                out = psbl_total_amplification(amp_arr)
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if not np.all(np.isfinite(g)):
            g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    if method_name in PSBL_PHOT_METHODS and _bspl_photastrom_static_jax:

        def forward(v):
            base = _base_vec_from_init(v, init_names, base_names)
            if _bspl_physical_photastrom:
                bg = _bspl_param1_geom_from_base(base, base_names)
            else:
                bg = _bspl_phot_geom_from_base(base, base_names)
            b_sff = _init_param(v, init_names, "b_sff", 1.0)
            mag_pri, mag_sec = _bspl_mags_from_init(v, init_names, layout, b_sff)
            out = bspl_photometry_jax(
                t_j,
                bg["t0_pri"],
                bg["t0_sec"],
                bg["tE"],
                bg["u0_pri"],
                bg["u0_sec"],
                bg["thetaE_hat"],
                mag_pri,
                mag_sec,
                b_sff=b_sff,
                pvec=pvec,
                piE_E=bg["piE_E"],
                piE_N=bg["piE_N"],
            )
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) == 0.0:
            g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    if (
        method_name in _BSPL_PHOTASTROM_PARAM1_CORE_AST
        and _static_psbl_photastrom
    ):
        default_b = float(np.asarray(getattr(jax_inst, "b_sff", [1.0])).reshape(-1)[0])

        def forward(v):
            base = _base_vec_from_init(v, init_names, base_names)
            geom = _psbl_geom_from_base(base, base_names)
            b_sff = _init_param(v, init_names, "b_sff", default_b)
            out = _pspl_astrom_forward(
                method_name, geom, t_j, geom["t0"], b_sff, pvec
            )
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) == 0.0:
            g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    if method_name in PHOT_LIKELIHOOD_METHODS and (
        _static_psbl_photastrom or _bspl_photastrom_static_jax
    ):
        mag_obs, mag_err = _phot_obs()

        def forward(v):
            base = _base_vec_from_init(v, init_names, base_names)
            b_sff = _init_param(v, init_names, "b_sff", 1.0)
            mag = _mag_from_init(v, init_names, layout, b_sff)
            if mag is None:
                mag = jnp.asarray(_mag_scalar(jax_inst, layout), dtype=jnp.float64)
            root_tol = float(getattr(jax_inst, "root_tol", 1e-8))
            if _static_psbl_photastrom:
                geom = _psbl_geom_from_base(base, base_names)
                mag_model = psbl_photometry(
                    t_j,
                    geom["t0"],
                    geom["tE"],
                    geom["u0"],
                    geom["thetaE_hat"],
                    geom["xL1"],
                    geom["xL2"],
                    geom["m1"],
                    geom["m2"],
                    mag,
                    b_sff=b_sff,
                    parallax_vectors=pvec,
                    piE_E=geom["piE_E"],
                    piE_N=geom["piE_N"],
                    root_tol=root_tol,
                )
            else:
                if _bspl_physical_photastrom:
                    bg = _bspl_param1_geom_from_base(base, base_names)
                else:
                    bg = _bspl_phot_geom_from_base(base, base_names)
                mag_pri, mag_sec = _bspl_mags_from_init(
                    v, init_names, layout, b_sff
                )
                mag_model = bspl_photometry_jax(
                    t_j,
                    bg["t0_pri"],
                    bg["t0_sec"],
                    bg["tE"],
                    bg["u0_pri"],
                    bg["u0_sec"],
                    bg["thetaE_hat"],
                    mag_pri,
                    mag_sec,
                    b_sff=b_sff,
                    pvec=pvec,
                    piE_E=bg["piE_E"],
                    piE_N=bg["piE_N"],
                )
            if method_name == "get_chi2_photometry":
                out = gaussian_chi2_photometry(mag_model, mag_obs, mag_err)
            else:
                out = gaussian_log_likelihood_photometry_each(
                    mag_model, mag_obs, mag_err
                )
            return jnp.sum(out)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) == 0.0:
            g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    if method_name in AST_LIKELIHOOD_METHODS and _static_psbl_photastrom:
        x_obs, y_obs, x_err, y_err = _ast_obs()
        default_b = float(np.asarray(getattr(jax_inst, "b_sff", [1.0])).reshape(-1)[0])

        def forward(v):
            base = _base_vec_from_init(v, init_names, base_names)
            geom = _psbl_geom_from_base(base, base_names)
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
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) == 0.0:
            g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    if _photastrom_ek and (
        method_name in _fd_photastrom_methods
        or (ek.startswith("psbl_photastrom") and not _static_psbl_photastrom)
        or (ek.startswith("bspl_photastrom") and not _bspl_photastrom_static_jax)
        or (
            ek.startswith("bspl_photastrom")
            and method_name in _BSPL_PHOTASTROM_PARAM1_CORE_AST
        )
        or (
            method_name in AST_LIKELIHOOD_METHODS
            and _bspl_photastrom_static_jax
        )
        or ek.startswith("bsbl_photastrom")
    ):
        g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
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
            return jnp.sum(u * u)

        g = np.asarray(jax.grad(forward)(vec0), dtype=np.float64)
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) == 0.0:
            g = _fd_grad_host(class_name, init_names, vec0, t, method_name)
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

    if ek.startswith(("fsbl_phot", "fsbl_photastrom")):
        g = _fd_grad_jax_eval(class_name, init_names, vec0, t, method_name)
        if return_names:
            return g, init_names
        return g

    raise NotImplementedError(f"grad smoke not wired for {class_name}.{method_name}")
