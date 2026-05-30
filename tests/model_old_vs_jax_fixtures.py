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
    "gp_log_sigma": [-1.0],
    "gp_log_rho": [0.5],
    "gp_log_S0": [-2.0],
    "gp_log_omega0": [0.0],
}

PARALLAX_KW = dict(raL=259.5, decL=-29.0, obsLocation="earth")

SKIP_CLASS_SUBSTR = (
    "Param5",
    "Param6",
    "geoproj",
    "RefPar",
    "LumLens",
    "PhotAstrom_Par_Param4",
)


def _value_for(name: str) -> Any:
    if name in CANONICAL:
        val = CANONICAL[name]
    elif name.endswith("1") and name[:-1] in CANONICAL:
        v = CANONICAL[name[:-1]]
        val = [v] if name.startswith(("b_sff", "mag_", "gp_")) else v
    else:
        raise KeyError(f"No canonical value for parameter {name!r}")
    if name in ("b_sff", "mag_src", "mag_base") and not isinstance(val, (list, tuple)):
        return [val]
    return val


def _param_mixin_cls(model_module, class_name: str):
    cls = getattr(model_module, class_name)
    layout = resolve_layout(cls)
    if layout is None:
        raise ValueError(f"No layout for {class_name}")
    return getattr(model_module, layout.param_mixin)


def build_init_args(cls: type, model_module) -> tuple[list[Any], dict[str, Any]]:
    mixin = _param_mixin_cls(model_module, cls.__name__)
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

    ref_cls = getattr(ref_model, class_name)
    jax_cls = getattr(jax_model, class_name)
    ref_args, ref_kw = build_init_args(ref_cls, ref_model)
    jax_args, jax_kw = build_init_args(jax_cls, jax_model)
    ref_inst = ref_cls(*ref_args, **ref_kw)
    jax_inst = jax_cls(*jax_args, **jax_kw)
    _post_init(ref_inst)
    _post_init(jax_inst)
    return ref_inst, jax_inst


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


def psbl_phot_first_pairs() -> list[tuple[str, str]]:
    """Seed PSBL parity harness (photometry Param1, no GP)."""
    return [("PSBL_Phot_noPar_Param1", "get_photometry")]


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
    return method(t, **kwargs)


def pack_fitter_vector(instance) -> tuple[np.ndarray, list[str]]:
    layout = resolve_layout(instance.__class__)
    names = list(layout.base_fitter_names) if layout else list(instance.fitter_param_names)
    vec = np.array([_fitter_scalar(instance, n) for n in names])
    return vec, names


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
        return float(b_sff) * xS + (1.0 - float(b_sff)) * xL
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
    unl = float(b_sff) * xS + (1.0 - float(b_sff)) * xL
    return (ast - unl) * 1e3


def grad_smoke_jax(class_name: str, method_name: str, jax_inst, t: np.ndarray) -> np.ndarray:
    """Pure-JAX grad smoke via jax_physics kernels."""
    import jax

    from bagle.jax.layout_registry import resolve_layout
    from bagle.jax_physics import pspl_amplification, pspl_photometry

    layout = resolve_layout(jax_inst.__class__)
    if layout is None:
        raise NotImplementedError(f"no layout for {class_name}")

    names = tuple(layout.base_fitter_names)
    t_j = jnp.asarray(t, dtype=jnp.float64)
    pvec = _parallax_vectors(jax_inst, t)
    vec0 = jnp.array([_fitter_scalar(jax_inst, n) for n in names], dtype=jnp.float64)
    ek = layout.eval_kind

    if method_name in ("get_photometry", "get_amplification") and ek in (
        "pspl_phot_static",
        "pspl_phot_log",
        "pspl_photastrom_physical",
        "pspl_photastrom_reduced",
    ):
        b_sff = float(np.asarray(jax_inst.b_sff).reshape(-1)[0])
        mag = _mag_scalar(jax_inst, layout)

        def forward(v):
            geom = _unpack_pspl_geom(ek, names, v)
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

        return np.asarray(jax.grad(forward)(vec0), dtype=np.float64)

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
        b_sff = float(np.asarray(getattr(jax_inst, "b_sff", [1.0])).reshape(-1)[0])

        def forward(v):
            geom = _unpack_pspl_geom(ek, names, v)
            out = _pspl_astrom_forward(
                method_name, geom, t_j, geom["t0"], b_sff, pvec
            )
            return jnp.sum(out)

        return np.asarray(jax.grad(forward)(vec0), dtype=np.float64)

    raise NotImplementedError(f"grad smoke not wired for {class_name}.{method_name}")
