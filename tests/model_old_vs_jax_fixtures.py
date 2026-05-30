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
    vec = np.array([float(getattr(instance, n, CANONICAL.get(n, 0.0))) for n in names])
    return vec, names


def _fitter_scalar(instance, name: str) -> float:
    if name == "piE_E":
        return float(instance.piE[0])
    if name == "piE_N":
        return float(instance.piE[1])
    return float(getattr(instance, name))


def grad_smoke_jax(class_name: str, method_name: str, jax_inst, t: np.ndarray) -> np.ndarray:
    """Pure-JAX grad smoke via jax_physics kernels."""
    import jax

    from bagle.jax.layout_registry import resolve_layout
    from bagle.jax_physics import (
        derive_pspl_static_geometry,
        precompute_parallax_vectors,
        pspl_amplification,
        pspl_astrometry_param1,
        pspl_photometry,
        unpack_pspl_phot_param1,
    )

    layout = resolve_layout(jax_inst.__class__)
    if layout is None:
        raise NotImplementedError(f"no layout for {class_name}")

    t_j = jnp.asarray(t, dtype=jnp.float64)
    pvec = None
    if getattr(jax_inst, "parallaxFlag", False):
        pvec = precompute_parallax_vectors(
            float(jax_inst.raL),
            float(jax_inst.decL),
            t,
            obs_location=str(jax_inst.obsLocation),
        )

    if method_name in ("get_photometry", "get_amplification") and layout.eval_kind in (
        "pspl_phot_static",
        "pspl_phot_log",
    ):
        names = list(jax_inst.fitter_param_names)
        vec0 = jnp.array([_fitter_scalar(jax_inst, n) for n in names], dtype=jnp.float64)
        b_sff = float(np.asarray(jax_inst.b_sff).reshape(-1)[0])
        if hasattr(jax_inst, "mag_src"):
            mag = float(np.asarray(jax_inst.mag_src).reshape(-1)[0])
        else:
            mag = float(np.asarray(jax_inst.mag_base).reshape(-1)[0])

        def forward(v):
            if layout.eval_kind == "pspl_phot_log":
                p = {n: v[i] for i, n in enumerate(names)}
                tE = 10.0 ** p["log_tE"]
                piE_amp = 10.0 ** p["log_piE"]
                phi = p["phi_muRel"] * jnp.pi / 180.0
                piE_E = piE_amp * jnp.cos(phi)
                piE_N = piE_amp * jnp.sin(phi)
                u0, thetaE_hat, _ = derive_pspl_static_geometry(
                    p["u0_amp"], piE_E, piE_N
                )
                t0 = p["t0"]
            else:
                p = unpack_pspl_phot_param1(v)
                u0, thetaE_hat, _ = derive_pspl_static_geometry(
                    p["u0_amp"], p["piE_E"], p["piE_N"]
                )
                t0, tE, piE_E, piE_N = p["t0"], p["tE"], p["piE_E"], p["piE_N"]
            fn = pspl_amplification if method_name == "get_amplification" else pspl_photometry
            return jnp.sum(
                fn(
                    t_j,
                    t0,
                    tE,
                    u0,
                    thetaE_hat,
                    mag,
                    b_sff=b_sff,
                    parallax_vectors=pvec,
                    piE_E=piE_E,
                    piE_N=piE_N,
                )
            )

        return np.asarray(jax.grad(forward)(vec0), dtype=np.float64)

    if method_name in ("get_photometry", "get_amplification") and layout.eval_kind in (
        "pspl_photastrom_physical",
        "pspl_photastrom_reduced",
    ):
        from bagle.jax_physics import PSPL_PHOTASTROM_PARAM1_FITTER_NAMES, derive_pspl_photastrom_param1_geometry

        names = PSPL_PHOTASTROM_PARAM1_FITTER_NAMES
        vec0 = jnp.array([_fitter_scalar(jax_inst, n) for n in names], dtype=jnp.float64)
        b_sff = float(np.asarray(jax_inst.b_sff).reshape(-1)[0])
        mag = float(np.asarray(jax_inst.mag_src).reshape(-1)[0])

        def forward(v):
            p = {n: v[i] for i, n in enumerate(names)}
            u0, thetaE_hat, tE, piE_E, piE_N, *_rest = derive_pspl_photastrom_param1_geometry(
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
            fn = pspl_amplification if method_name == "get_amplification" else pspl_photometry
            return jnp.sum(
                fn(
                    t_j,
                    p["t0"],
                    tE,
                    u0,
                    thetaE_hat,
                    mag,
                    b_sff=b_sff,
                    parallax_vectors=pvec,
                    piE_E=piE_E,
                    piE_N=piE_N,
                )
            )

        return np.asarray(jax.grad(forward)(vec0), dtype=np.float64)

    if method_name in (
        "get_astrometry",
        "get_astrometry_unlensed",
        "get_lens_astrometry",
        "get_centroid_shift",
    ) and layout.eval_kind in (
        "pspl_photastrom_physical",
        "pspl_photastrom_reduced",
        "pspl_astrom_reduced",
    ):
        from bagle.jax_physics import PSPL_PHOTASTROM_PARAM1_FITTER_NAMES, derive_pspl_photastrom_param1_geometry

        b_sff = float(np.asarray(getattr(jax_inst, "b_sff", [1.0])).reshape(-1)[0])

        if layout.eval_kind == "pspl_photastrom_physical":
            names = PSPL_PHOTASTROM_PARAM1_FITTER_NAMES
            vec0 = jnp.array([_fitter_scalar(jax_inst, n) for n in names], dtype=jnp.float64)

            def forward(v):
                p = {n: v[i] for i, n in enumerate(names)}
                *geom, thetaE_amp, piS, piL = derive_pspl_photastrom_param1_geometry(
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
                _u0, _thetaE_hat, _tE, _piE_E, _piE_N, xS0, xL0, muS, muL = geom
                out = pspl_astrometry_param1(
                    t_j,
                    p["t0"],
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
                return jnp.sum(out)

            return np.asarray(jax.grad(forward)(vec0), dtype=np.float64)

        names = list(jax_inst.fitter_param_names)
        vec0 = jnp.array([_fitter_scalar(jax_inst, n) for n in names], dtype=jnp.float64)

        def forward(v):
            for i, n in enumerate(names):
                setattr(jax_inst, n, float(v[i]))
            if "log10_thetaE" in names:
                jax_inst.thetaE_amp = 10.0 ** float(v[names.index("log10_thetaE")])
            out = call_method(jax_inst, method_name, t)
            return jnp.sum(jnp.asarray(out, dtype=jnp.float64))

        return np.asarray(jax.grad(forward)(vec0), dtype=np.float64)

    raise NotImplementedError(f"grad smoke not wired for {class_name}.{method_name}")
