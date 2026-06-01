"""Param-mixin layout registry for Phot / PhotAstrom / Astrom model classes."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

Family = Literal["pspl", "psbl", "bspl", "fsbl", "bsbl"]
LikelihoodMode = Literal["phot", "ast", "joint", "phot_gp", "joint_gp"]
OrbitKind = Literal["none", "linear", "accelerated", "keplerian"]
EvalKind = str

_MODEL_JAX_PY = Path(__file__).resolve().parents[1] / "model_jax.py"
_MODEL_PY = Path(__file__).resolve().parents[1] / "model.py"


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: str
    param_mixin: str
    base_fitter_names: tuple[str, ...]
    family: Family
    likelihood_mode: LikelihoodMode
    orbit: OrbitKind
    has_gp: bool
    eval_kind: EvalKind
    mag_fitter: Literal["mag_src", "mag_base", "none"] = "mag_src"
    geoproj: bool = False


def _list_from_assign_value(node, known: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    if isinstance(node, ast.List):
        elts = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                elts.append(elt.value)
            else:
                return None
        return tuple(elts)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        parent = node.value.id
        if parent in known:
            return known[parent]
    return None


def _parse_fitter_names_from_model_py(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Parse ``fitter_param_names`` from model_jax.py without importing it."""
    text = (path or _MODEL_JAX_PY).read_text(encoding="utf-8")
    tree = ast.parse(text)
    raw: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "fitter_param_names":
                    raw[node.name] = stmt.value
                    break
    out: dict[str, tuple[str, ...]] = {}
    for _ in range(len(raw) + 2):
        for cls_name, value_node in raw.items():
            if cls_name in out:
                continue
            names = _list_from_assign_value(value_node, out)
            if names:
                out[cls_name] = names
    return out


def _orbit_kind(class_name: str) -> OrbitKind:
    if "EllOrbs" in class_name or "CircOrbs" in class_name:
        return "keplerian"
    if "AccOrbs" in class_name:
        return "accelerated"
    if "LinOrbs" in class_name:
        return "linear"
    return "none"


def _family_from_name(name: str) -> Family:
    if name.startswith("BSBL"):
        return "bsbl"
    if name.startswith("FSBL") or name.startswith("FSPL"):
        return "fsbl"
    if name.startswith("BSPL"):
        return "bspl"
    if name.startswith("PSBL"):
        return "psbl"
    return "pspl"


def _eval_kind_for(
    param_mixin: str,
    family: Family,
    likelihood_mode: LikelihoodMode,
    orbit: OrbitKind,
) -> EvalKind:
    if "GP_" in param_mixin or param_mixin.startswith("PSPL_GP") or "GP_Phot" in param_mixin:
        if likelihood_mode in ("phot_gp", "joint_gp"):
            base = param_mixin.replace("PSPL_GP_", "PSPL_").replace("PSBL_GP_", "PSBL_")
            base = base.replace("BSPL_GP_", "BSPL_")
            param_mixin = base

    if family == "fsbl" or "FSBL" in param_mixin or "FSPL" in param_mixin:
        if "PhotAstrom" in param_mixin or likelihood_mode == "joint":
            return "fsbl_photastrom"
        return "fsbl_phot"

    if family == "bsbl":
        if "PhotAstrom" in param_mixin:
            return f"bsbl_photastrom_{orbit}"
        return "bsbl_phot"

    if family == "bspl":
        if "PhotAstrom" in param_mixin:
            return f"bspl_photastrom_{orbit}"
        return "bspl_phot"

    if family == "psbl":
        if "PhotAstrom" in param_mixin:
            return f"psbl_photastrom_{orbit}"
        if "Phot" in param_mixin:
            return f"psbl_phot_{orbit}"
        return "psbl_phot_none"

    # PSPL
    if "AstromParam" in param_mixin and "PhotAstrom" not in param_mixin:
        return "pspl_astrom_reduced"
    if "PhotAstromParam1" in param_mixin:
        return "pspl_photastrom_physical"
    if "PhotAstrom" in param_mixin:
        return "pspl_photastrom_reduced"
    if "PhotParam3" in param_mixin:
        return "pspl_phot_log"
    if "PhotParam2" in param_mixin or "PhotParam1" in param_mixin:
        return "pspl_phot_static"
    return "pspl_phot_static"


def _mag_fitter(param_mixin: str) -> Literal["mag_src", "mag_base", "none"]:
    if "PhotParam2" in param_mixin or "PhotParam3" in param_mixin:
        if "PhotAstromParam4" in param_mixin or "PhotAstromParam6" in param_mixin:
            return "mag_base"
        if "PhotParam2" in param_mixin or "PhotParam3" in param_mixin:
            if "PhotAstrom" not in param_mixin:
                return "mag_base"
    if "PhotAstromParam4" in param_mixin or "PhotAstromParam6" in param_mixin:
        return "mag_base"
    if "Astrom" in param_mixin and "Phot" not in param_mixin:
        return "none"
    return "mag_src"


def _likelihood_mode(param_mixin: str, has_gp: bool) -> LikelihoodMode:
    ast_only = "AstromParam" in param_mixin and "PhotAstrom" not in param_mixin
    phot_ast = "PhotAstrom" in param_mixin
    phot_only = ("PhotParam" in param_mixin or "_Phot_" in param_mixin) and not phot_ast
    if has_gp:
        if phot_ast:
            return "joint_gp"
        return "phot_gp"
    if ast_only:
        return "ast"
    if phot_ast:
        return "joint"
    if phot_only or "Phot" in param_mixin:
        return "phot"
    return "phot"


def _layout_id(
    param_mixin: str,
    family: Family,
    orbit: OrbitKind,
    geoproj: bool,
    eval_kind: EvalKind,
) -> str:
    """Stable layout id from eval kind (not full mixin class name)."""
    base = eval_kind if eval_kind.startswith(f"{family}_") else f"{family}_{eval_kind}"
    parts = [base]
    if orbit != "none":
        parts.append(orbit)
    if geoproj:
        parts.append("geoproj")
    return "_".join(parts)


# Legacy ids used by jax_physics before the registry (tests / callers).
_LEGACY_LAYOUT_IDS: dict[str, str] = {
    "pspl_photastrom_physical": "pspl_photastrom_param1",
}


def legacy_layout_id(layout: LayoutSpec) -> str:
    """Return legacy layout id when one exists, else ``layout.layout_id``."""
    return _LEGACY_LAYOUT_IDS.get(layout.eval_kind, layout.layout_id)


_PARAM_NAMES = _parse_fitter_names_from_model_py()

# Param mixins that participate in Phot / PhotAstrom / Astrom layouts.
_PARAM_MIXIN_PATTERNS = (
    "PhotParam",
    "PhotAstromParam",
    "PhotAstrom_LinOrbs",
    "PhotAstrom_AccOrbs",
    "PhotAstrom_EllOrbs",
    "PhotAstrom_CircOrbs",
    "Phot_EllOrbs",
    "Phot_CircOrbs",
    "AstromParam",
    "GP_Phot",
    "GP_PhotAstrom",
)


def _is_param_mixin(name: str) -> bool:
    return any(p in name for p in _PARAM_MIXIN_PATTERNS)


def check_param_mixin_parity() -> None:
    """Ensure shared Param mixins have matching ``fitter_param_names`` in both models."""
    ref = _parse_fitter_names_from_model_py(_MODEL_PY)
    jax = _parse_fitter_names_from_model_py(_MODEL_JAX_PY)
    ref_mixins = {k for k in ref if _is_param_mixin(k)}
    jax_mixins = {k for k in jax if _is_param_mixin(k)}
    shared = ref_mixins & jax_mixins
    mismatched = [name for name in sorted(shared) if ref[name] != jax[name]]
    if mismatched:
        raise ValueError(
            f"fitter_param_names differ for shared mixins: {mismatched[:10]}"
            + (f" (+{len(mismatched) - 10} more)" if len(mismatched) > 10 else "")
        )


def _build_registry() -> dict[str, LayoutSpec]:
    reg: dict[str, LayoutSpec] = {}
    for param_mixin, names in _PARAM_NAMES.items():
        if not _is_param_mixin(param_mixin):
            continue
        family = _family_from_name(param_mixin)
        orbit = _orbit_kind(param_mixin)
        has_gp = "GP_" in param_mixin or param_mixin.startswith("PSPL_GP")
        geoproj = "geoproj" in param_mixin
        lm = _likelihood_mode(param_mixin, has_gp)
        ek = _eval_kind_for(param_mixin, family, lm, orbit)
        lid = _layout_id(param_mixin, family, orbit, geoproj, ek)
        reg[param_mixin] = LayoutSpec(
            layout_id=lid,
            param_mixin=param_mixin,
            base_fitter_names=names,
            family=family,
            likelihood_mode=lm,
            orbit=orbit,
            has_gp=has_gp,
            eval_kind=ek,
            mag_fitter=_mag_fitter(param_mixin),
            geoproj=geoproj,
        )
    return reg


LAYOUT_BY_PARAM_MIXIN: dict[str, LayoutSpec] = _build_registry()


def _find_param_mixin(model_class) -> str | None:
    for cls in model_class.__mro__:
        name = cls.__name__
        if name in LAYOUT_BY_PARAM_MIXIN:
            return name
    return None


def resolve_layout(model_class) -> LayoutSpec | None:
    """Return :class:`LayoutSpec` for a concrete or abstract model class."""
    mixin = _find_param_mixin(model_class)
    if mixin is None:
        return None
    layout = LAYOUT_BY_PARAM_MIXIN[mixin]
    cls_name = model_class.__name__
    if "_GP_" not in cls_name and "GPnoJitter" not in cls_name:
        return layout
    phot_ast = "PhotAstrom" in cls_name
    lm: LikelihoodMode = "joint_gp" if phot_ast else "phot_gp"
    return replace(layout, has_gp=True, likelihood_mode=lm)


def iter_phot_astrom_ast_concrete_classes(model_module):
    """Yield concrete model classes whose names match Phot / PhotAstrom / Astrom."""
    import inspect

    for name, obj in inspect.getmembers(model_module, inspect.isclass):
        if obj.__module__ != model_module.__name__:
            continue
        if not (
            ("_Phot_" in name or name.endswith("_Phot"))
            and "PhotAstrom" not in name
            or "PhotAstrom" in name
            or ("_Astrom_" in name or name.endswith("_Astrom"))
        ):
            continue
        if "Param" in name and "ModelClassABC" not in str(obj):
            # skip param mixins themselves
            if name.endswith("Param1") or "Param" in name and any(
                name.endswith(f"Param{i}") for i in range(1, 10)
            ):
                if "Phot" not in name and "Astrom" not in name:
                    continue
        if name.startswith(("PSPL_", "PSBL_", "BSPL_", "FSPL_", "FSBL_", "BSBL_")):
            if resolve_layout(obj) is not None:
                yield name, obj


def supports_jax_loglik(fitter) -> str | None:
    """Return layout_id when JAX likelihood is available for this fitter."""
    from bagle.jax.likelihood import supports_jax_loglik_for_fitter

    return supports_jax_loglik_for_fitter(fitter)
