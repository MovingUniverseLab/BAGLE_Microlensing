"""Class×method JAX migration task discovery (shared by generator, dashboard, tests)."""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Iterable, Literal

from bagle.jax.layout_registry import resolve_layout

Step = Literal["jax_forward", "parity", "grad"]
MethodName = str

ALL_FORWARD_METHODS: tuple[MethodName, ...] = (
    "get_amplification",
    "get_photometry",
    "get_astrometry_unlensed",
    "get_lens_astrometry",
    "get_astrometry",
    "get_centroid_shift",
    "get_resolved_astrometry",
    "get_resolved_lens_astrometry",
    "get_astrometry_outline_unlensed",
    "get_photometry_with_gp",
)

FAMILY_ORDER: tuple[str, ...] = (
    "PSPL",
    "PSBL",
    "BSPL",
    "FSPL",
    "FSBL",
    "BSBL",
    "BFSPL",
)


@dataclass(frozen=True)
class MigrationTask:
    class_name: str
    method_name: MethodName
    family: str
    applicable: bool
    task_id: str

    @property
    def atomic_id(self) -> str:
        return f"{self.class_name}::{self.method_name}"


def _family(class_name: str) -> str:
    return class_name.split("_", 1)[0]


def discover_concrete_classes(model_module) -> list[tuple[str, type]]:
    """Yield concrete ModelClassABC Phot / PhotAstrom / Astrom classes."""
    out: list[tuple[str, type]] = []
    for name, cls in inspect.getmembers(model_module, inspect.isclass):
        if cls.__module__ != model_module.__name__:
            continue
        mro_names = [c.__name__ for c in cls.mro()]
        if "ModelClassABC" not in mro_names or cls.__name__ == "ModelClassABC":
            continue
        if not re.search(r"_(Phot|PhotAstrom|Astrom)_", name):
            continue
        out.append((name, cls))
    return sorted(out, key=lambda x: (FAMILY_ORDER.index(_family(x[0])) if _family(x[0]) in FAMILY_ORDER else 99, x[0]))


def _class_has_method(cls: type, method_name: str) -> bool:
    for base in cls.mro():
        if method_name in base.__dict__:
            return True
    return hasattr(cls, method_name)


def applicable_methods(cls: type) -> dict[MethodName, bool]:
    """Return forward methods applicable to ``cls`` (True) or n/a (False)."""
    name = cls.__name__
    phot = bool(getattr(cls, "photometryFlag", False) or getattr(cls, "paramPhotFlag", False))
    ast = bool(getattr(cls, "astrometryFlag", False) or getattr(cls, "paramAstromFlag", False))
    has_gp = "GP" in name or bool(getattr(cls, "gpFlag", False))
    is_fsbl = name.startswith(("FSBL", "FSPL"))
    is_bsbl = name.startswith(("BSBL", "BSPL"))
    is_psbl = name.startswith("PSBL")

    out: dict[MethodName, bool] = {m: False for m in ALL_FORWARD_METHODS}

    if phot and _class_has_method(cls, "get_amplification"):
        out["get_amplification"] = True
    if phot and _class_has_method(cls, "get_photometry"):
        out["get_photometry"] = True

    if ast:
        for m in (
            "get_astrometry_unlensed",
            "get_lens_astrometry",
            "get_astrometry",
        ):
            if _class_has_method(cls, m):
                out[m] = True
        if _class_has_method(cls, "get_centroid_shift"):
            out["get_centroid_shift"] = True

    if is_psbl or is_bsbl or is_fsbl:
        for m in ("get_resolved_astrometry", "get_resolved_lens_astrometry"):
            if _class_has_method(cls, m):
                out[m] = True

    if name.startswith("FSPL") and _class_has_method(cls, "get_astrometry_outline_unlensed"):
        out["get_astrometry_outline_unlensed"] = True

    if has_gp and _class_has_method(cls, "get_photometry_with_gp"):
        out["get_photometry_with_gp"] = True

    return out


def generate_tasks(model_module) -> list[MigrationTask]:
    tasks: list[MigrationTask] = []
    for class_name, cls in discover_concrete_classes(model_module):
        fam = _family(class_name)
        for method_name, ok in applicable_methods(cls).items():
            tasks.append(
                MigrationTask(
                    class_name=class_name,
                    method_name=method_name,
                    family=fam,
                    applicable=ok,
                    task_id=f"{class_name}::{method_name}",
                )
            )
    return tasks


def applicable_task_pairs(model_module) -> list[tuple[str, str]]:
    """(class_name, method_name) pairs with applicable=True."""
    return [
        (t.class_name, t.method_name)
        for t in generate_tasks(model_module)
        if t.applicable
    ]


def family_summary(tasks: Iterable[MigrationTask], done: set[str] | None = None) -> dict[str, tuple[int, int]]:
    """Return {family: (done_count, total_applicable)}."""
    done = done or set()
    totals: dict[str, list[int]] = {}
    for t in tasks:
        if not t.applicable:
            continue
        if t.family not in totals:
            totals[t.family] = [0, 0]
        totals[t.family][1] += 1
        if t.atomic_id in done:
            totals[t.family][0] += 1
    return {k: (v[0], v[1]) for k, v in totals.items()}


def layout_for_class_name(model_module, class_name: str):
    cls = getattr(model_module, class_name)
    return resolve_layout(cls)
