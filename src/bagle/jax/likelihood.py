"""Analytic JAX likelihood construction from model Param mixins."""
from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax.bspl import bspl_photometry_jax
from bagle.jax_physics import (
    AstFilterLikelihoodData,
    JaxJointLikelihoodContext,
    PhotAstromFilterLikelihoodData,
    PhotFilterLikelihoodData,
    _fitter_has_blocked_params,
    _fitter_weight,
    _param_index,
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    precompute_parallax_vectors,
    psbl_photometry,
    pspl_astrometry_param1,
    pspl_photometry,
)


def _param_mixin_class(model_class):
    """Return the Param mixin that owns ``fitter_param_names`` in the MRO."""
    # Prefer the class that defines fitter_param_names (true Param mixin).
    for cls in model_class.__mro__:
        if cls.__name__ == "PSPL_Param":
            continue
        if "fitter_param_names" not in cls.__dict__:
            continue
        names = cls.fitter_param_names
        if not names:
            continue
        if hasattr(cls, "get_params_for_jax"):
            return cls
    # Fallback: first MRO entry with non-empty names + packing.
    for cls in model_class.__mro__:
        names = getattr(cls, "fitter_param_names", None)
        if hasattr(cls, "get_params_for_jax") and names:
            if cls.__name__ == "PSPL_Param":
                continue
            return cls
    return None


def _infer_loglik_mode(model_class) -> str | None:
    """Infer JAX likelihood mode from parameterization flags."""
    phot = getattr(model_class, "paramPhotFlag", False)
    ast = getattr(model_class, "paramAstromFlag", False)
    if phot and ast:
        return "joint"
    if phot:
        return "phot"
    if ast:
        return "ast"
    return None


def _mag_fitter_from_phot_params(model_class) -> str:
    """Return declared fitter magnitude convention."""
    phot_names = tuple(getattr(model_class, "phot_param_names", ()))
    if "mag_base" in phot_names:
        return "mag_base"
    if "mag_src" in phot_names:
        return "mag_src"
    if "mag_src_pri" in phot_names:
        return "mag_src_pri"
    return "none"


def _supports_jax_loglik(fitter, param_cls) -> bool:
    """Check whether fitter data and parameters support analytic JAX."""
    if param_cls is None or _fitter_has_blocked_params(fitter):
        return False
    base_names = tuple(param_cls.fitter_param_names)
    if tuple(fitter.fitter_param_names)[:len(base_names)] != base_names:
        return False
    model_class = fitter.model_class
    mode = _infer_loglik_mode(model_class)
    if mode == "phot":
        return (getattr(model_class, "photometryFlag", False)
                and fitter.n_phot_sets > 0
                and not (getattr(model_class, "astrometryFlag", False)
                         and fitter.n_ast_sets > 0))
    if mode == "ast":
        return getattr(model_class, "astrometryFlag", False) and fitter.n_ast_sets > 0
    if mode == "joint":
        return (getattr(model_class, "photometryFlag", False)
                and getattr(model_class, "astrometryFlag", False)
                and fitter.n_phot_sets > 0 and fitter.n_ast_sets > 0)
    return False


def supports_jax_loglik_for_fitter(fitter) -> str | None:
    """Return the supported Param mixin class name, if any."""
    param_cls = _param_mixin_class(fitter.model_class)
    if param_cls is None:
        return None
    if _supports_jax_loglik(fitter, param_cls):
        return param_cls.__name__
    return None


def _mag_index(names, mag_fitter, filt_1):
    """Find per-filter magnitude parameter index."""
    return _param_index(names, mag_fitter,
                        filt_1 if f"{mag_fitter}{filt_1}" in names else None)


def build_joint_context_for_param(fitter, param_cls):
    """Build immutable joint data context for a Param mixin."""
    names = tuple(fitter.fitter_param_names)
    base_names = tuple(param_cls.fitter_param_names)
    if names[:len(base_names)] != base_names:
        return None
    mode = _infer_loglik_mode(fitter.model_class)
    mag_fitter = _mag_fitter_from_phot_params(fitter.model_class)
    use_parallax = "raL" in fitter.data and "decL" in fitter.data
    ra_l = float(fitter.data["raL"]) if use_parallax else 0.0
    dec_l = float(fitter.data["decL"]) if use_parallax else 0.0
    mapping = getattr(fitter, "map_phot_idx_to_ast_idx", [])
    filters = []

    for ast_idx in range(fitter.n_ast_sets):
        ast_filt = ast_idx + 1
        phot_idx = mapping[ast_idx] if len(mapping) > ast_idx else ast_idx
        t_ast = np.asarray(fitter.data[f"t_ast{ast_filt}"], dtype=np.float64)
        pvec_ast = (precompute_parallax_vectors(ra_l, dec_l, t_ast)
                    if use_parallax else None)
        idx_b = 0 if mode == "ast" else _param_index(
            names, "b_sff",
            phot_idx + 1 if f"b_sff{phot_idx + 1}" in names else None,
        )
        ast = AstFilterLikelihoodData(
            t=t_ast,
            x_obs=np.asarray(fitter.data[f"xpos{ast_filt}"], dtype=np.float64),
            y_obs=np.asarray(fitter.data[f"ypos{ast_filt}"], dtype=np.float64),
            x_err=np.asarray(fitter.data[f"xpos_err{ast_filt}"], dtype=np.float64),
            y_err=np.asarray(fitter.data[f"ypos_err{ast_filt}"], dtype=np.float64),
            weight=_fitter_weight(fitter, fitter.n_phot_sets + ast_idx),
            parallax_vectors=pvec_ast,
            idx_b_sff=idx_b,
            phot_filt_idx=phot_idx,
        )
        phot = None
        if mode != "ast" and phot_idx < fitter.n_phot_sets:
            filt_1 = phot_idx + 1
            t_phot = np.asarray(fitter.data[f"t_phot{filt_1}"], dtype=np.float64)
            phot = PhotFilterLikelihoodData(
                t=t_phot,
                mag_obs=np.asarray(fitter.data[f"mag{filt_1}"], dtype=np.float64),
                mag_err=np.asarray(fitter.data[f"mag_err{filt_1}"], dtype=np.float64),
                weight=_fitter_weight(fitter, phot_idx),
                parallax_vectors=(precompute_parallax_vectors(ra_l, dec_l, t_phot)
                                  if use_parallax else None),
                idx_b_sff=idx_b,
                idx_mag_src=_mag_index(names, mag_fitter, filt_1),
            )
        filters.append(PhotAstromFilterLikelihoodData(phot=phot, ast=ast))

    if mode != "ast":
        mapped = set(mapping) if mapping else set(range(fitter.n_ast_sets))
        for phot_idx in range(fitter.n_phot_sets):
            if phot_idx in mapped:
                continue
            filt_1 = phot_idx + 1
            t_phot = np.asarray(fitter.data[f"t_phot{filt_1}"], dtype=np.float64)
            phot = PhotFilterLikelihoodData(
                t=t_phot,
                mag_obs=np.asarray(fitter.data[f"mag{filt_1}"], dtype=np.float64),
                mag_err=np.asarray(fitter.data[f"mag_err{filt_1}"], dtype=np.float64),
                weight=_fitter_weight(fitter, phot_idx),
                parallax_vectors=(precompute_parallax_vectors(ra_l, dec_l, t_phot)
                                  if use_parallax else None),
                idx_b_sff=_param_index(
                    names, "b_sff", filt_1 if f"b_sff{filt_1}" in names else None),
                idx_mag_src=_mag_index(names, mag_fitter, filt_1),
            )
            filters.append(PhotAstromFilterLikelihoodData(
                phot=phot, ast=cast(AstFilterLikelihoodData, None)))

    return JaxJointLikelihoodContext(
        layout=param_cls.__name__,
        use_parallax=use_parallax,
        fitter_param_names=names,
        base_indices=tuple(range(len(base_names))),
        filters=tuple(filters),
    )


def _phot_mean(param_vec, phot, packed, mag_fitter):
    """Compute analytic model magnitudes for one photometric block."""
    b_sff = param_vec[phot.idx_b_sff]
    mag_primary = param_vec[phot.idx_mag_src]
    if mag_fitter == "mag_base":
        mag_primary = mag_primary - 2.5 * jnp.log10(b_sff)
    pvec = (None if phot.parallax_vectors is None
            else jnp.asarray(phot.parallax_vectors, dtype=jnp.float64))
    t = jnp.asarray(phot.t, dtype=jnp.float64)
    if {"t0", "tE", "u0", "thetaE_hat", "m1", "m2", "xL1", "xL2"} <= packed.keys():
        return psbl_photometry(
            t, packed["t0"], packed["tE"], packed["u0"], packed["thetaE_hat"],
            packed["xL1"], packed["xL2"], packed["m1"], packed["m2"], mag_primary,
            b_sff=b_sff, parallax_vectors=pvec, piE_E=packed.get("piE_E", 0.0),
            piE_N=packed.get("piE_N", 0.0),
        )
    if {"t0_pri", "t0_sec", "tE", "u0_pri", "u0_sec", "thetaE_hat"} <= packed.keys():
        return bspl_photometry_jax(
            t, packed["t0_pri"], packed["t0_sec"], packed["tE"],
            packed["u0_pri"], packed["u0_sec"], packed["thetaE_hat"], mag_primary,
            param_vec[phot.idx_mag_src + 1], b_sff, pvec=pvec,
            piE_E=packed.get("piE_E", 0.0), piE_N=packed.get("piE_N", 0.0),
        )
    if {"t0", "tE", "u0", "thetaE_hat"} <= packed.keys():
        return pspl_photometry(
            t, packed["t0"], packed["tE"], packed["u0"], packed["thetaE_hat"],
            mag_primary, b_sff=b_sff, parallax_vectors=pvec,
            piE_E=packed.get("piE_E", 0.0), piE_N=packed.get("piE_N", 0.0),
        )
    raise NotImplementedError("Param mixin has no supported analytic photometry")


def _ast_loglik(param_vec, block, packed, mode):
    """Evaluate one analytic astrometry block."""
    if block is None:
        return 0.0
    b_sff = 1.0 if mode == "ast" else param_vec[block.idx_b_sff]
    pvec = (None if block.parallax_vectors is None
            else jnp.asarray(block.parallax_vectors, dtype=jnp.float64))
    pos = pspl_astrometry_param1(
        jnp.asarray(block.t, dtype=jnp.float64), packed["t0"], packed["xS0"],
        packed["xL0"], packed["muS"], packed["muL"], packed["thetaE_amp"], b_sff,
        parallax_vectors=pvec, piS=packed.get("piS", 0.0),
        piL=packed.get("piL", 0.0),
    )
    return block.weight * gaussian_astrometry_log_likelihood_sum(
        pos, block.x_obs, block.y_obs, block.x_err, block.y_err)


def _joint_loglik_reduced(param_vec, ctx, param_cls, mag_fitter, mode):
    """Evaluate analytic JAX likelihood from Param-mixin packed geometry."""
    param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
    base = param_vec[jnp.array(ctx.base_indices, dtype=jnp.int32)]
    packed = param_cls.get_params_for_jax(base)
    ln_likelihood = 0.0
    for block in ctx.filters:
        ln_likelihood = ln_likelihood + _ast_loglik(
            param_vec, block.ast, packed, mode)
        if block.phot is not None:
            mean = _phot_mean(param_vec, block.phot, packed, mag_fitter)
            ln_likelihood = ln_likelihood + block.phot.weight * gaussian_log_likelihood_sum(
                mean, block.phot.mag_obs, block.phot.mag_err)
    return ln_likelihood


def _build_phot_loglik(fitter, param_cls):
    """Build analytic photometry-only JAX likelihood."""
    names = tuple(fitter.fitter_param_names)
    mag_fitter = _mag_fitter_from_phot_params(fitter.model_class)
    use_parallax = "raL" in fitter.data and "decL" in fitter.data
    ra_l = float(fitter.data["raL"]) if use_parallax else 0.0
    dec_l = float(fitter.data["decL"]) if use_parallax else 0.0
    filters = []
    for phot_idx in range(fitter.n_phot_sets):
        filt_1 = phot_idx + 1
        t = np.asarray(fitter.data[f"t_phot{filt_1}"], dtype=np.float64)
        filters.append(PhotFilterLikelihoodData(
            t=t, mag_obs=np.asarray(fitter.data[f"mag{filt_1}"], dtype=np.float64),
            mag_err=np.asarray(fitter.data[f"mag_err{filt_1}"], dtype=np.float64),
            weight=_fitter_weight(fitter, phot_idx),
            parallax_vectors=(precompute_parallax_vectors(ra_l, dec_l, t)
                              if use_parallax else None),
            idx_b_sff=_param_index(
                names, "b_sff", filt_1 if f"b_sff{filt_1}" in names else None),
            idx_mag_src=_mag_index(names, mag_fitter, filt_1),
        ))
    ctx = (param_cls, tuple(filters), tuple(range(len(param_cls.fitter_param_names))),
           mag_fitter)
    return jax.jit(lambda vec: _phot_loglik_vec(vec, ctx)), ctx


def _phot_loglik_vec(param_vec, ctx):
    """Evaluate photometric Gaussian likelihood using packed Param geometry."""
    param_cls, filters, base_indices, mag_fitter = ctx
    param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
    packed = param_cls.get_params_for_jax(
        param_vec[jnp.array(base_indices, dtype=jnp.int32)])
    ln_likelihood = 0.0
    for phot in filters:
        mean = _phot_mean(param_vec, phot, packed, mag_fitter)
        ln_likelihood = ln_likelihood + phot.weight * gaussian_log_likelihood_sum(
            mean, phot.mag_obs, phot.mag_err)
    return ln_likelihood


def _build_joint_loglik(fitter, param_cls):
    """Build analytic joint or astrometry-only JAX likelihood."""
    ctx = build_joint_context_for_param(fitter, param_cls)
    if ctx is None:
        return None, None
    mag_fitter = _mag_fitter_from_phot_params(fitter.model_class)
    mode = _infer_loglik_mode(fitter.model_class)
    fn = lambda vec: _joint_loglik_reduced(vec, ctx, param_cls, mag_fitter, mode)
    return jax.jit(fn), (ctx, param_cls)


def _gp_param_index(names, key, filt_1):
    """Look up optional per-filter GP parameter without raising."""
    candidate = f"{key}{filt_1}" if f"{key}{filt_1}" in names else key
    return names.index(candidate) if candidate in names else None


def build_analytic_gp_loglik_fn(fitter, param_cls=None):
    """Build analytic GP photometry likelihood plus optional astrometry."""
    try:
        from bagle.jax.gp import build_gp_kernel_from_params, gp_log_probability
    except ImportError:
        return None, None
    param_cls = param_cls or _param_mixin_class(fitter.model_class)
    if param_cls is None:
        return None, None
    if not _supports_jax_loglik(fitter, param_cls):
        return None, None
    mode = _infer_loglik_mode(fitter.model_class)
    if mode == "phot":
        _fn, phot_ctx = _build_phot_loglik(fitter, param_cls)
        filters = phot_ctx[1]
        base_indices = phot_ctx[2]
        mag_fitter = phot_ctx[3]
        joint_ctx = None
    else:
        joint_ctx = build_joint_context_for_param(fitter, param_cls)
        if joint_ctx is None:
            return None, None
        filters = tuple(block.phot for block in joint_ctx.filters if block.phot)
        base_indices = joint_ctx.base_indices
        mag_fitter = _mag_fitter_from_phot_params(fitter.model_class)
    names = tuple(fitter.fitter_param_names)
    fixed_jitter = "GPnoJitter" not in fitter.model_class.__name__

    def _loglik(param_vec):
        param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
        packed = param_cls.get_params_for_jax(
            param_vec[jnp.array(base_indices, dtype=jnp.int32)])
        ln_likelihood = 0.0
        for filt_1, phot in enumerate(filters, start=1):
            gp_params = {}
            for key in ("gp_log_sigma", "gp_log_rho", "gp_rho", "gp_log_S0",
                        "gp_log_omega0", "gp_log_jit_sigma"):
                idx = _gp_param_index(names, key, filt_1)
                if idx is not None:
                    gp_params[key] = param_vec[idx]
            kernel, jitter = build_gp_kernel_from_params(
                gp_params, phot.mag_err, fixed_jitter=fixed_jitter)
            mean = _phot_mean(param_vec, phot, packed, mag_fitter)
            ln_likelihood = ln_likelihood + phot.weight * gp_log_probability(
                kernel, phot.t, phot.mag_obs, phot.mag_err, mean, jitter)
        if joint_ctx is not None:
            for block in joint_ctx.filters:
                ln_likelihood = ln_likelihood + _ast_loglik(
                    param_vec, block.ast, packed, "ast")
        return ln_likelihood

    return jax.jit(_loglik), (joint_ctx, param_cls)


def build_jax_loglik_fn(fitter):
    """Return all-analytic JAX likelihood and context, when supported."""
    param_cls = _param_mixin_class(fitter.model_class)
    if not _supports_jax_loglik(fitter, param_cls):
        return None, None
    from bagle.jax.gp import supports_gp_class
    if supports_gp_class(fitter.model_class):
        return build_analytic_gp_loglik_fn(fitter, param_cls)
    if _infer_loglik_mode(fitter.model_class) == "phot":
        return _build_phot_loglik(fitter, param_cls)
    return _build_joint_loglik(fitter, param_cls)
