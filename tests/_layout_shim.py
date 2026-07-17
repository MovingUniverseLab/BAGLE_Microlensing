"""Local layout-like shim for parity/grad fixtures (no bagle.jax.layout_registry)."""
from __future__ import annotations

from types import SimpleNamespace


def _family(name: str) -> str:
    for fam in ("pspl", "psbl", "bspl", "fsbl", "bsbl", "fspl", "bfspl"):
        if name.upper().startswith(fam.upper()):
            return fam
    return "pspl"


def _orbit(name: str) -> str:
    low = name.lower()
    if "ellorbs" in low or "circorbs" in low:
        return "keplerian"
    if "accorbs" in low:
        return "accelerated"
    if "linorbs" in low:
        return "linear"
    return "none"


def _eval_kind(name: str, orbit: str) -> str:
    fam = _family(name)
    low = name.lower()
    if "photastrom" in low or ("astrom" in low and "phot" in low):
        mode = "photastrom"
    elif "astrom" in low:
        mode = "astrom"
    else:
        mode = "phot"
    # PSPL phot kinds used by grad_smoke
    if fam == "pspl" and mode == "phot":
        if "log_te" in "".join(
            # hint from param names not available here
            []
        ):
            return "pspl_phot_log"
        return "pspl_phot_static"
    if fam == "pspl" and mode == "photastrom":
        if "Param1" in name and "mL" not in name:
            # physical vs reduced distinguished later via fitter names
            return "pspl_photastrom_physical"
        if "Param3" in name or "AstromParam3" in name:
            return "pspl_photastrom_reduced"
        return "pspl_photastrom_reduced"
    if fam == "pspl" and mode == "astrom":
        return "pspl_astrom_reduced"
    if fam == "bspl" and mode == "phot":
        return "bspl_phot"
    if fam == "bspl" and mode == "photastrom":
        return f"bspl_photastrom_{orbit}"
    if fam == "psbl" and mode == "phot":
        return f"psbl_phot_{orbit}"
    if fam == "psbl" and mode == "photastrom":
        return f"psbl_photastrom_{orbit}"
    if fam in ("fsbl", "fspl", "bfspl"):
        return f"{fam}_{mode}"
    if fam == "bsbl":
        return f"bsbl_{mode}"
    return f"{fam}_{mode}_{orbit}"


def _mag_fitter(cls) -> str:
    phot = tuple(getattr(cls, "phot_param_names", ()) or ())
    if "mag_base" in phot:
        return "mag_base"
    if "mag_src" in phot:
        return "mag_src"
    if "mag_src_pri" in phot:
        return "mag_src"
    return "none"


def _likelihood_mode(cls) -> str:
    phot = getattr(cls, "paramPhotFlag", False)
    ast = getattr(cls, "paramAstromFlag", False)
    has_gp = "GP" in cls.__name__
    if phot and ast:
        return "joint_gp" if has_gp else "joint"
    if phot:
        return "phot_gp" if has_gp else "phot"
    if ast:
        return "ast"
    return "phot"


def _param_mixin_for_fixtures(model_class):
    """Find the Param mixin that owns ``fitter_param_names`` (NumPy or JAX)."""
    from bagle.jax.likelihood import _param_mixin_class

    mixin = _param_mixin_class(model_class)
    if mixin is not None:
        return mixin
    # NumPy reference models lack get_params_for_jax; prefer __dict__ owner.
    for cls in model_class.__mro__:
        if "fitter_param_names" not in cls.__dict__:
            continue
        names = cls.fitter_param_names
        if names:
            return cls
    return None


def resolve_layout(model_class):
    """Return a SimpleNamespace mimicking the old LayoutSpec for fixtures."""
    mixin = _param_mixin_for_fixtures(model_class)
    if mixin is None:
        return None
    name = mixin.__name__
    orbit = _orbit(name)
    # Refine PSPL photastrom physical vs reduced from fitter names
    names = tuple(mixin.fitter_param_names)
    ek = _eval_kind(name, orbit)
    if "mL" in names and "beta" in names:
        ek = "pspl_photastrom_physical"
    elif "log_tE" in names:
        ek = "pspl_phot_log"
    elif ek.startswith("pspl_photastrom") and "thetaE" in names:
        ek = "pspl_photastrom_reduced"
    elif ek.startswith("pspl_photastrom") and "log10_thetaE" in names:
        ek = "pspl_photastrom_reduced"
    has_gp = "GP" in name or "GP" in model_class.__name__
    return SimpleNamespace(
        layout_id=name.lower(),
        param_mixin=name,
        base_fitter_names=names,
        family=_family(name),
        likelihood_mode=_likelihood_mode(mixin),
        orbit=orbit,
        has_gp=has_gp,
        eval_kind=ek,
        mag_fitter=_mag_fitter(mixin),
        geoproj="geoproj" in name.lower(),
    )
