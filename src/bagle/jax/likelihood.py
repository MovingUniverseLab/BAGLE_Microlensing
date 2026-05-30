"""Unified JAX log-likelihood construction via layout registry."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax import geometry as geom
from bagle.jax.layout_registry import LayoutSpec, legacy_layout_id, resolve_layout
from bagle.jax_physics import (
    AstFilterLikelihoodData,
    JaxJointLikelihoodContext,
    PhotAstromFilterLikelihoodData,
    PhotFilterLikelihoodData,
    PSPL_PHOT_PARAM1_FITTER_NAMES,
    _fitter_has_blocked_params,
    _fitter_weight,
    _param_index,
    build_jax_joint_likelihood_context,
    build_jax_phot_likelihood_context,
    build_jax_phot_loglik_fn,
    build_jax_joint_loglik_fn,
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    precompute_parallax_vectors,
    psbl_photometry,
    pspl_astrometry_param1,
    pspl_photometry,
)
from bagle.jax.geometry import derive_geometry_from_layout, mag_src_from_fitter, unpack_base_params
from bagle.jax.bspl import bspl_photometry_jax


def supports_jax_loglik_for_fitter(fitter) -> str | None:
    """Return ``layout_id`` when this fitter can use JAX autodiff."""
    layout = resolve_layout(fitter.model_class)
    if layout is None:
        return None
    if layout.has_gp:
        return None
    if _fitter_has_blocked_params(fitter):
        return None
    names = tuple(fitter.fitter_param_names)
    if names[: len(layout.base_fitter_names)] != layout.base_fitter_names:
        return None
    mc = fitter.model_class
    if layout.likelihood_mode == "phot":
        if not getattr(mc, "photometryFlag", False) or fitter.n_phot_sets == 0:
            return None
        if getattr(mc, "astrometryFlag", False) and fitter.n_ast_sets > 0:
            return None
    elif layout.likelihood_mode == "ast":
        if not getattr(mc, "astrometryFlag", False) or fitter.n_ast_sets == 0:
            return None
    elif layout.likelihood_mode in ("joint", "joint_gp"):
        if not getattr(mc, "photometryFlag", False) or fitter.n_phot_sets == 0:
            return None
        if not getattr(mc, "astrometryFlag", False) or fitter.n_ast_sets == 0:
            return None
    elif layout.likelihood_mode == "phot_gp":
        if not getattr(mc, "photometryFlag", False) or fitter.n_phot_sets == 0:
            return None
    else:
        return None
    return legacy_layout_id(layout)


def build_joint_context_for_layout(fitter, layout: LayoutSpec):
    """Build joint likelihood context for any registered PhotAstrom / Astrom layout."""
    names = tuple(fitter.fitter_param_names)
    if names[: len(layout.base_fitter_names)] != layout.base_fitter_names:
        return None
    base_indices = tuple(range(len(layout.base_fitter_names)))
    use_parallax = "raL" in fitter.data and "decL" in fitter.data
    ra_l = float(fitter.data["raL"]) if use_parallax else None
    dec_l = float(fitter.data["decL"]) if use_parallax else None
    map_phot = getattr(fitter, "map_phot_idx_to_ast_idx", [])
    joint_filters: list[PhotAstromFilterLikelihoodData] = []
    ast_only = layout.likelihood_mode == "ast"

    if not ast_only:
        for i in range(fitter.n_ast_sets):
            ast_filt = i + 1
            phot_idx = map_phot[i] if len(map_phot) > i else i
            phot_filt = phot_idx + 1
            t_ast = np.asarray(fitter.data[f"t_ast{ast_filt}"], dtype=np.float64)
            x_obs = np.asarray(fitter.data[f"xpos{ast_filt}"], dtype=np.float64)
            y_obs = np.asarray(fitter.data[f"ypos{ast_filt}"], dtype=np.float64)
            x_err = np.asarray(fitter.data[f"xpos_err{ast_filt}"], dtype=np.float64)
            y_err = np.asarray(fitter.data[f"ypos_err{ast_filt}"], dtype=np.float64)
            ast_weight = _fitter_weight(fitter, fitter.n_phot_sets + i)
            pvec_ast = (
                precompute_parallax_vectors(ra_l, dec_l, t_ast) if use_parallax else None
            )
            idx_b_ast = _param_index(
                names, "b_sff", phot_filt if f"b_sff{phot_filt}" in names else None
            )
            phot_block = None
            if phot_idx < fitter.n_phot_sets:
                t_phot = np.asarray(fitter.data[f"t_phot{phot_filt}"], dtype=np.float64)
                mag_obs = np.asarray(fitter.data[f"mag{phot_filt}"], dtype=np.float64)
                mag_err = np.asarray(fitter.data[f"mag_err{phot_filt}"], dtype=np.float64)
                phot_weight = _fitter_weight(fitter, phot_idx)
                pvec_phot = (
                    precompute_parallax_vectors(ra_l, dec_l, t_phot)
                    if use_parallax
                    else None
                )
                mag_key = "mag_src" if layout.mag_fitter == "mag_src" else "mag_base"
                idx_m = _param_index(
                    names,
                    mag_key,
                    phot_filt if f"{mag_key}{phot_filt}" in names else None,
                )
                phot_block = PhotFilterLikelihoodData(
                    t=t_phot,
                    mag_obs=mag_obs,
                    mag_err=mag_err,
                    weight=phot_weight,
                    parallax_vectors=pvec_phot,
                    idx_b_sff=idx_b_ast,
                    idx_mag_src=idx_m,
                )
            ast_block = AstFilterLikelihoodData(
                t=t_ast,
                x_obs=x_obs,
                y_obs=y_obs,
                x_err=x_err,
                y_err=y_err,
                weight=ast_weight,
                parallax_vectors=pvec_ast,
                idx_b_sff=idx_b_ast,
                phot_filt_idx=phot_idx,
            )
            joint_filters.append(
                PhotAstromFilterLikelihoodData(phot=phot_block, ast=ast_block)
            )
        mapped_phot = set(map_phot) if map_phot else set(range(fitter.n_ast_sets))
        for phot_idx in range(fitter.n_phot_sets):
            if phot_idx in mapped_phot:
                continue
            phot_filt = phot_idx + 1
            t_phot = np.asarray(fitter.data[f"t_phot{phot_filt}"], dtype=np.float64)
            mag_obs = np.asarray(fitter.data[f"mag{phot_filt}"], dtype=np.float64)
            mag_err = np.asarray(fitter.data[f"mag_err{phot_filt}"], dtype=np.float64)
            phot_weight = _fitter_weight(fitter, phot_idx)
            pvec_phot = (
                precompute_parallax_vectors(ra_l, dec_l, t_phot) if use_parallax else None
            )
            idx_b = _param_index(
                names, "b_sff", phot_filt if f"b_sff{phot_filt}" in names else None
            )
            mag_key = "mag_src" if layout.mag_fitter == "mag_src" else "mag_base"
            idx_m = _param_index(
                names,
                mag_key,
                phot_filt if f"{mag_key}{phot_filt}" in names else None,
            )
            phot_block = PhotFilterLikelihoodData(
                t=t_phot,
                mag_obs=mag_obs,
                mag_err=mag_err,
                weight=phot_weight,
                parallax_vectors=pvec_phot,
                idx_b_sff=idx_b,
                idx_mag_src=idx_m,
            )
            joint_filters.append(PhotAstromFilterLikelihoodData(phot=phot_block, ast=None))
    else:
        for i in range(fitter.n_ast_sets):
            ast_filt = i + 1
            t_ast = np.asarray(fitter.data[f"t_ast{ast_filt}"], dtype=np.float64)
            x_obs = np.asarray(fitter.data[f"xpos{ast_filt}"], dtype=np.float64)
            y_obs = np.asarray(fitter.data[f"ypos{ast_filt}"], dtype=np.float64)
            x_err = np.asarray(fitter.data[f"xpos_err{ast_filt}"], dtype=np.float64)
            y_err = np.asarray(fitter.data[f"ypos_err{ast_filt}"], dtype=np.float64)
            ast_weight = _fitter_weight(fitter, fitter.n_phot_sets + i)
            pvec_ast = (
                precompute_parallax_vectors(ra_l, dec_l, t_ast) if use_parallax else None
            )
            idx_b = 0
            ast_block = AstFilterLikelihoodData(
                t=t_ast,
                x_obs=x_obs,
                y_obs=y_obs,
                x_err=x_err,
                y_err=y_err,
                weight=ast_weight,
                parallax_vectors=pvec_ast,
                idx_b_sff=idx_b,
                phot_filt_idx=0,
            )
            joint_filters.append(PhotAstromFilterLikelihoodData(phot=None, ast=ast_block))

    return JaxJointLikelihoodContext(
        layout=layout.layout_id,
        use_parallax=use_parallax,
        fitter_param_names=names,
        base_indices=base_indices,
        filters=tuple(joint_filters),
    )


def _joint_loglik_reduced(param_vec, ctx, layout: LayoutSpec):
    """Joint likelihood for PSPL/PSBL reduced PhotAstrom / Astrom layouts."""
    param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
    base = param_vec[jnp.array(ctx.base_indices, dtype=jnp.int32)]
    geom_out = derive_geometry_from_layout(
        layout.layout_id, layout.eval_kind, base, layout.base_fitter_names
    )
    if isinstance(geom_out, tuple) and geom_out[0] == "pspl_phot":
        raise ValueError("phot-only geometry in joint likelihood")
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
    ) = geom_out
    t0 = unpack_base_params(layout.base_fitter_names, base)["t0"]

    lnL = 0.0
    for block in ctx.filters:
        b_sff_ast = (
            1.0
            if layout.likelihood_mode == "ast"
            else param_vec[block.ast.idx_b_sff]
        )
        if block.ast is not None:
            pvec_ast = None
            if block.ast.parallax_vectors is not None:
                pvec_ast = jnp.asarray(block.ast.parallax_vectors, dtype=jnp.float64)
            pos = pspl_astrometry_param1(
                jnp.asarray(block.ast.t, dtype=jnp.float64),
                t0,
                xS0,
                xL0,
                muS,
                muL,
                thetaE_amp,
                b_sff_ast,
                parallax_vectors=pvec_ast,
                piS=piS,
                piL=piL,
            )
            lnL = lnL + block.ast.weight * gaussian_astrometry_log_likelihood_sum(
                pos,
                block.ast.x_obs,
                block.ast.y_obs,
                block.ast.x_err,
                block.ast.y_err,
            )
        if block.phot is not None:
            b_sff = param_vec[block.phot.idx_b_sff]
            mag_src = param_vec[block.phot.idx_mag_src]
            if layout.mag_fitter == "mag_base":
                mag_src = geom.mag_src_from_base(mag_src, b_sff)
            pvec_phot = None
            if block.phot.parallax_vectors is not None:
                pvec_phot = jnp.asarray(block.phot.parallax_vectors, dtype=jnp.float64)
            mag_model = pspl_photometry(
                jnp.asarray(block.phot.t, dtype=jnp.float64),
                t0,
                tE,
                u0,
                thetaE_hat,
                mag_src,
                b_sff=b_sff,
                parallax_vectors=pvec_phot,
                piE_E=piE_E,
                piE_N=piE_N,
            )
            lnL = lnL + block.phot.weight * gaussian_log_likelihood_sum(
                mag_model, block.phot.mag_obs, block.phot.mag_err
            )
    return lnL


def build_jax_loglik_fn(fitter):
    """Return ``(jit_loglik, ctx)`` using the layout registry."""
    layout = resolve_layout(fitter.model_class)
    if layout is None:
        return None, None
    if supports_jax_loglik_for_fitter(fitter) is None:
        return None, None

    if layout.has_gp:
        from bagle.jax.gp import build_gp_loglik_fn

        return build_gp_loglik_fn(fitter, layout)

    if layout.likelihood_mode == "phot":
        if (
            layout.eval_kind == "pspl_phot_static"
            and layout.param_mixin == "PSPL_PhotParam1"
            and layout.base_fitter_names == PSPL_PHOT_PARAM1_FITTER_NAMES
        ):
            return build_jax_phot_loglik_fn(fitter)
        return _build_registry_phot_loglik(fitter, layout)

    if layout.likelihood_mode == "joint":
        if layout.eval_kind == "pspl_photastrom_physical":
            return build_jax_joint_loglik_fn(fitter)
        if layout.eval_kind.startswith("psbl_photastrom") and layout.orbit == "none":
            # Analytic PSBL joint path is not yet validated; use host lnL + numeric VJP.
            return _build_host_loglik_vjp(fitter, layout)
        if layout.eval_kind in ("pspl_photastrom_reduced",):
            return _build_reduced_joint_loglik(fitter, layout)
        if layout.eval_kind.startswith("bspl_photastrom"):
            from bagle.jax.bspl import build_bspl_joint_loglik

            return build_bspl_joint_loglik(fitter, layout)
        if layout.eval_kind.startswith("fsbl"):
            from bagle.jax.fspl import build_fsbl_joint_loglik

            return build_fsbl_joint_loglik(fitter, layout)
        if layout.eval_kind.startswith("bsbl"):
            from bagle.jax.bsbl import build_bsbl_joint_loglik

            return build_bsbl_joint_loglik(fitter, layout)
        return _build_reduced_joint_loglik(fitter, layout)

    if layout.likelihood_mode == "ast":
        fn, ctx = _build_ast_loglik(fitter, layout)
        if fn is not None:
            return fn, ctx

    return _build_host_loglik_vjp(fitter, layout)


def _build_host_loglik_vjp(fitter, layout: LayoutSpec):
    """Fallback: host ``log_likely`` with numeric gradient for registered layouts."""

    names = tuple(fitter.fitter_param_names)

    def _lnL_host(vec_np):
        cube = {names[i]: float(vec_np[i]) for i in range(len(names))}
        return float(fitter.log_likely(cube))

    @jax.custom_vjp
    def _loglik(param_vec):
        param_vec = jnp.asarray(param_vec, dtype=jnp.float64)
        return jax.pure_callback(
            _lnL_host,
            jax.ShapeDtypeStruct((), jnp.float64),
            param_vec,
        )

    def _fwd(param_vec):
        return _loglik(param_vec), (param_vec,)

    def _bwd(res, g):
        param_vec, = res
        p0 = np.asarray(param_vec, dtype=np.float64)
        eps = 1e-5
        grad = np.zeros_like(p0)
        f0 = _lnL_host(p0)
        for i in range(len(grad)):
            p = p0.copy()
            p[i] += eps
            grad[i] = float(g) * (_lnL_host(p) - f0) / eps
        return (jnp.asarray(grad, dtype=jnp.float64),)

    _loglik.defvjp(_fwd, _bwd)
    return jax.jit(_loglik), (fitter, layout)


def _build_registry_phot_loglik(fitter, layout: LayoutSpec):
    if (
        layout.eval_kind == "pspl_phot_static"
        and layout.param_mixin == "PSPL_PhotParam1"
        and layout.base_fitter_names == PSPL_PHOT_PARAM1_FITTER_NAMES
    ):
        return build_jax_phot_loglik_fn(fitter)

    names = tuple(fitter.fitter_param_names)
    base_n = len(layout.base_fitter_names)
    base_idx = tuple(range(base_n))
    use_parallax = "raL" in fitter.data
    ra_l = float(fitter.data["raL"]) if use_parallax else None
    dec_l = float(fitter.data["decL"]) if use_parallax else None
    filters = []
    for i in range(fitter.n_phot_sets):
        filt_1 = i + 1
        t = np.asarray(fitter.data[f"t_phot{filt_1}"], dtype=np.float64)
        mag_obs = np.asarray(fitter.data[f"mag{filt_1}"], dtype=np.float64)
        mag_err = np.asarray(fitter.data[f"mag_err{filt_1}"], dtype=np.float64)
        weight = _fitter_weight(fitter, i)
        pvec = precompute_parallax_vectors(ra_l, dec_l, t) if use_parallax else None
        idx_b = _param_index(names, "b_sff", filt_1 if f"b_sff{filt_1}" in names else None)
        if layout.eval_kind == "bspl_phot":
            idx_m = _param_index(
                names,
                "mag_src_pri",
                filt_1 if f"mag_src_pri{filt_1}" in names else None,
            )
        else:
            mag_key = "mag_src" if layout.mag_fitter == "mag_src" else "mag_base"
            idx_m = _param_index(
                names, mag_key, filt_1 if f"{mag_key}{filt_1}" in names else None
            )
        from bagle.jax_physics import PhotFilterLikelihoodData

        filters.append(
            PhotFilterLikelihoodData(
                t=t,
                mag_obs=mag_obs,
                mag_err=mag_err,
                weight=weight,
                parallax_vectors=pvec,
                idx_b_sff=idx_b,
                idx_mag_src=idx_m,
            )
        )

    host_ctx = (layout, tuple(filters), base_idx, use_parallax)

    def _loglik(param_vec):
        return _phot_loglik_vec(param_vec, host_ctx)

    return jax.jit(_loglik), host_ctx


def _phot_loglik_vec(param_vec, host_ctx):
    layout, filters, base_idx, _ = host_ctx
    param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
    base = param_vec[jnp.array(base_idx, dtype=jnp.int32)]
    geom_out = derive_geometry_from_layout(
        layout.layout_id, layout.eval_kind, base, layout.base_fitter_names
    )
    lnL = 0.0
    if geom_out[0] == "pspl_phot":
        _, u0, thetaE_hat, tE, piE_E, piE_N = geom_out
        p = unpack_base_params(layout.base_fitter_names, base)
        t0 = p["t0"]
        for phot in filters:
            b_sff = param_vec[phot.idx_b_sff]
            mag_v = param_vec[phot.idx_mag_src]
            mag_src = mag_src_from_fitter(mag_v, layout.mag_fitter, b_sff)
            pvec = (
                None
                if phot.parallax_vectors is None
                else jnp.asarray(phot.parallax_vectors, dtype=jnp.float64)
            )
            mag_model = pspl_photometry(
                jnp.asarray(phot.t, dtype=jnp.float64),
                t0,
                tE,
                u0,
                thetaE_hat,
                mag_src,
                b_sff=b_sff,
                parallax_vectors=pvec,
                piE_E=piE_E,
                piE_N=piE_N,
            )
            lnL = lnL + phot.weight * gaussian_log_likelihood_sum(
                mag_model, phot.mag_obs, phot.mag_err
            )
        return lnL
    if geom_out[0] == "psbl_phot":
        _, u0, thetaE_hat, t0, tE, m1, m2, xL1, xL2, piE_E, piE_N = geom_out
        for phot in filters:
            b_sff = param_vec[phot.idx_b_sff]
            mag_src = param_vec[phot.idx_mag_src]
            pvec = (
                None
                if phot.parallax_vectors is None
                else jnp.asarray(phot.parallax_vectors, dtype=jnp.float64)
            )
            mag_model = psbl_photometry(
                jnp.asarray(phot.t, dtype=jnp.float64),
                t0,
                tE,
                u0,
                thetaE_hat,
                xL1,
                xL2,
                m1,
                m2,
                mag_src,
                b_sff=b_sff,
                parallax_vectors=pvec,
                piE_E=piE_E,
                piE_N=piE_N,
            )
            lnL = lnL + phot.weight * gaussian_log_likelihood_sum(
                mag_model, phot.mag_obs, phot.mag_err
            )
        return lnL
    if geom_out[0] == "bspl_phot":
        _, u0_pri, u0_sec, thetaE_hat, t0_pri, t0_sec, tE, piE_E, piE_N = geom_out
        names_full = layout.base_fitter_names + ("mag_src_pri", "mag_src_sec", "b_sff")
        for phot in filters:
            b_sff = param_vec[phot.idx_b_sff]
            # phot filter stores idx_mag_src as primary index
            idx_pri = phot.idx_mag_src
            idx_sec = idx_pri + 1
            mag_pri = param_vec[idx_pri]
            mag_sec = param_vec[idx_sec]
            pvec = (
                None
                if phot.parallax_vectors is None
                else jnp.asarray(phot.parallax_vectors, dtype=jnp.float64)
            )
            mag_model = bspl_photometry_jax(
                jnp.asarray(phot.t, dtype=jnp.float64),
                t0_pri,
                t0_sec,
                tE,
                u0_pri,
                u0_sec,
                thetaE_hat,
                mag_pri,
                mag_sec,
                b_sff,
                pvec=pvec,
                piE_E=piE_E,
                piE_N=piE_N,
            )
            lnL = lnL + phot.weight * gaussian_log_likelihood_sum(
                mag_model, phot.mag_obs, phot.mag_err
            )
        return lnL
    raise NotImplementedError(layout.eval_kind)


def _build_reduced_joint_loglik(fitter, layout: LayoutSpec):
    ctx = build_joint_context_for_layout(fitter, layout)
    if ctx is None:
        ctx = build_jax_joint_likelihood_context(fitter)
    if ctx is None:
        return None, None

    def _loglik(param_vec):
        if layout.eval_kind == "pspl_photastrom_physical":
            from bagle.jax_physics import _joint_loglik_pspl_param1

            return _joint_loglik_pspl_param1(param_vec, ctx)
        return _joint_loglik_reduced(param_vec, ctx, layout)

    return jax.jit(_loglik), (ctx, layout)


def _build_psbl_joint_param1_loglik(fitter, layout: LayoutSpec):
    ctx = build_joint_context_for_layout(fitter, layout)
    if ctx is None:
        return None, None

    def _loglik(param_vec):
        from bagle.jax.psbl_ast import joint_loglik_psbl_param1

        return joint_loglik_psbl_param1(param_vec, ctx, layout)

    return jax.jit(_loglik), (ctx, layout)


def _build_ast_loglik(fitter, layout: LayoutSpec):
    ctx = build_joint_context_for_layout(fitter, layout)
    if ctx is None:
        return None, None

    def _loglik(param_vec):
        return _joint_loglik_reduced(param_vec, ctx, layout)

    return jax.jit(_loglik), (ctx, layout)
