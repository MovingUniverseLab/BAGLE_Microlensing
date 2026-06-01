"""Forward model evaluation (photometry / astrometry) via JAX."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from bagle.jax import geometry as geom
from bagle.jax.layout_registry import LayoutSpec, resolve_layout
from bagle.jax_physics import (
    gaussian_chi2_astrometry,
    gaussian_chi2_photometry,
    gaussian_log_likelihood_astrometry_each,
    gaussian_log_likelihood_photometry_each,
    precompute_parallax_vectors,
    psbl_all_arrays,
    psbl_complex_pos_keplerian,
    psbl_complex_pos_static,
    psbl_photometry,
    psbl_photometry_from_amp,
    psbl_total_amplification,
    pspl_astrometry_param1,
    pspl_amplification,
    pspl_phot_astrometry,
    pspl_phot_astrometry_unlensed,
    pspl_photometry,
    pspl_resolved_amplification,
    pspl_resolved_astrometry,
    pspl_source_astrometry_unlensed,
    pspl_u,
)
from bagle.jax.geometry import mag_src_from_fitter


def _parallax_table(model, t, filt_idx: int):
    if not getattr(model, "parallaxFlag", False):
        return None
    ra_l = float(model.raL)
    dec_l = float(model.decL)
    obs = model.obsLocation
    if isinstance(obs, (list, tuple, np.ndarray)):
        obs_loc = obs[filt_idx]
    else:
        obs_loc = obs
    return precompute_parallax_vectors(ra_l, dec_l, t, obs_location=str(obs_loc))


def _phot_attr(model, filt_idx: int, base: str):
    val = getattr(model, base, None)
    if val is None:
        return None
    arr = np.asarray(val).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[filt_idx]) if arr.size > filt_idx else float(arr[0])


def _evaluate_psbl_phot(
    layout: LayoutSpec,
    model,
    t_j,
    pvec,
    mag_src: float,
    b_sff: float,
):
    """Shared PSBL phot-only forward (static or Keplerian orbit)."""
    m1 = float(model.m1)
    m2 = float(model.m2)
    root_tol = float(getattr(model, "root_tol", 1e-8))
    piE_E = float(model.piE[0])
    piE_N = float(model.piE[1])
    t0 = float(model.t0)
    tE = float(model.tE)
    u0 = jnp.asarray(model.u0, dtype=jnp.float64)
    thetaE_hat = jnp.asarray(model.thetaE_hat, dtype=jnp.float64)

    if layout.orbit == "keplerian":
        w, z1, z2 = psbl_complex_pos_keplerian(
            t_j,
            t0,
            tE,
            u0,
            thetaE_hat,
            float(model.w),
            float(model.o),
            float(model.i),
            float(model.e),
            float(model.p),
            float(model.tp),
            float(model.aleph),
            float(model.aleph_sec),
            parallax_vectors=pvec,
            piE_E=piE_E,
            piE_N=piE_N,
        )
    else:
        xL1 = jnp.asarray(model.xL1_over_theta, dtype=jnp.float64)
        xL2 = jnp.asarray(model.xL2_over_theta, dtype=jnp.float64)
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
    amp = psbl_total_amplification(amp_arr)
    return amp, psbl_photometry_from_amp(amp, mag_src, b_sff=b_sff)


def evaluate_astrometry_unlensed_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    pos = evaluate_astrometry_jax(layout, model, t, filt_idx)
    if pos is not None and layout.eval_kind.startswith("pspl"):
        return pos
    return None


def evaluate_lens_astrometry_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    return evaluate_astrometry_jax(layout, model, t, filt_idx)


def evaluate_centroid_shift_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    return None


def evaluate_photometry_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    """Compute model magnitudes; return None if layout is not JAX-supported."""
    t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    pvec = _parallax_table(model, t, filt_idx)
    b_sff = _phot_attr(model, filt_idx, "b_sff")
    mag_raw = _phot_attr(model, filt_idx, "mag_src")
    if mag_raw is None:
        mag_raw = _phot_attr(model, filt_idx, "mag_base")
        mag_fitter = "mag_base"
    else:
        mag_fitter = "mag_src"
    if b_sff is None or mag_raw is None:
        return None
    mag_src = float(
        mag_src_from_fitter(jnp.asarray(mag_raw), mag_fitter, jnp.asarray(b_sff))
    )

    ek = layout.eval_kind
    try:
        if ek == "pspl_phot_static":
            u0 = jnp.asarray(model.u0, dtype=jnp.float64)
            thetaE_hat = jnp.asarray(model.thetaE_hat, dtype=jnp.float64)
            mag = pspl_photometry(
                t_j,
                float(model.t0),
                float(model.tE),
                u0,
                thetaE_hat,
                mag_src,
                b_sff=b_sff,
                parallax_vectors=pvec,
                piE_E=float(model.piE[0]),
                piE_N=float(model.piE[1]),
            )
            return np.asarray(mag, dtype=np.float64)

        if ek in ("pspl_photastrom_physical", "pspl_photastrom_reduced"):
            mag = pspl_photometry(
                t_j,
                float(model.t0),
                float(model.tE),
                jnp.asarray(model.u0, dtype=jnp.float64),
                jnp.asarray(model.thetaE_hat, dtype=jnp.float64),
                mag_src,
                b_sff=b_sff,
                parallax_vectors=pvec,
                piE_E=float(model.piE[0]),
                piE_N=float(model.piE[1]),
            )
            return np.asarray(mag, dtype=np.float64)

        if ek == "pspl_astrom_reduced":
            return None  # no photometry

        if ek.startswith("psbl_phot"):
            _, mag = _evaluate_psbl_phot(layout, model, t_j, pvec, mag_src, b_sff)
            return np.asarray(mag, dtype=np.float64)

        if ek.startswith("bspl_phot"):
            from bagle.jax.bspl import bspl_photometry_from_model

            return bspl_photometry_from_model(model, t, filt_idx, pvec)

        if ek.startswith(("fsbl_phot", "fsbl_photastrom")):
            from bagle.jax.fspl import fspl_photometry_from_model

            return fspl_photometry_from_model(model, t, filt_idx, pvec)

        if ek.startswith("psbl_photastrom"):
            mtot = float(model.mLp) + float(model.mLs)
            m1 = float(model.mLp) / mtot
            m2 = float(model.mLs) / mtot
            sep_th = float(model.sep) / float(model.thetaE_amp)
            ar = float(model.alpha_rad)
            xL1 = jnp.array([0.5 * sep_th * jnp.sin(ar), 0.5 * sep_th * jnp.cos(ar)])
            xL2 = -xL1
            mag = psbl_photometry(
                t_j,
                float(model.t0),
                float(model.tE),
                jnp.asarray(model.u0, dtype=jnp.float64),
                jnp.asarray(model.thetaE_hat, dtype=jnp.float64),
                xL1,
                xL2,
                m1,
                m2,
                mag_src,
                b_sff=b_sff,
                parallax_vectors=pvec,
                piE_E=float(model.piE[0]),
                piE_N=float(model.piE[1]),
                root_tol=float(getattr(model, "root_tol", 1e-8)),
            )
            return np.asarray(mag, dtype=np.float64)

        if ek.startswith("bsbl"):
            from bagle.jax.bsbl import bsbl_photometry_from_model

            return bsbl_photometry_from_model(model, t, filt_idx, pvec)

    except (AttributeError, NotImplementedError, TypeError):
        return None
    return None


def _pspl_phot_mag_src(model, filt_idx: int, b_sff: float) -> float | None:
    mag_raw = _phot_attr(model, filt_idx, "mag_src")
    if mag_raw is None:
        mag_raw = _phot_attr(model, filt_idx, "mag_base")
        mag_fitter = "mag_base"
    else:
        mag_fitter = "mag_src"
    if mag_raw is None:
        return None
    return float(
        mag_src_from_fitter(jnp.asarray(mag_raw), mag_fitter, jnp.asarray(b_sff))
    )


def _pspl_phot_astrometry_inputs(model, t, filt_idx: int):
    t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    pvec = _parallax_table(model, t, filt_idx)
    b_sff = _phot_attr(model, filt_idx, "b_sff")
    if b_sff is None:
        b_sff = 1.0
    mag_src = _pspl_phot_mag_src(model, filt_idx, b_sff)
    if mag_src is None:
        return None
    return t_j, pvec, b_sff, mag_src


def evaluate_astrometry_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    """Compute model astrometry; PSPL_Phot returns Einstein radii, else arcsec."""
    ek = layout.eval_kind
    phot_only_pspl = ek in ("pspl_phot_static", "pspl_phot_log")
    if not phot_only_pspl and layout.likelihood_mode not in (
        "ast",
        "joint",
        "joint_gp",
    ):
        return None
    t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    pvec = _parallax_table(model, t, filt_idx)
    b_sff = _phot_attr(model, filt_idx, "b_sff")
    if b_sff is None:
        b_sff = 1.0

    try:
        if phot_only_pspl:
            inputs = _pspl_phot_astrometry_inputs(model, t, filt_idx)
            if inputs is None:
                return None
            t_j, pvec, b_sff, mag_src = inputs
            pos = pspl_phot_astrometry(
                t_j,
                float(model.t0),
                float(model.tE),
                jnp.asarray(model.u0, dtype=jnp.float64),
                jnp.asarray(model.thetaE_hat, dtype=jnp.float64),
                mag_src,
                b_sff,
                parallax_vectors=pvec,
                piE_E=float(model.piE[0]),
                piE_N=float(model.piE[1]),
            )
            return np.asarray(pos, dtype=np.float64)

        if ek in (
            "pspl_photastrom_physical",
            "pspl_photastrom_reduced",
            "pspl_astrom_reduced",
        ):
            pos = pspl_astrometry_param1(
                t_j,
                float(model.t0),
                jnp.asarray(model.xS0, dtype=jnp.float64),
                jnp.asarray(model.xL0, dtype=jnp.float64),
                jnp.asarray(model.muS, dtype=jnp.float64),
                jnp.asarray(model.muL, dtype=jnp.float64),
                float(model.thetaE_amp),
                b_sff,
                parallax_vectors=pvec,
                piS=float(model.piS),
                piL=float(model.piL),
            )
            return np.asarray(pos, dtype=np.float64)

        if ek.startswith("psbl_photastrom"):
            from bagle.jax.psbl_ast import psbl_astrometry_from_model

            return psbl_astrometry_from_model(model, t, filt_idx, pvec)

        if ek.startswith("bspl_photastrom"):
            from bagle.jax.bspl import bspl_astrometry_from_model

            return bspl_astrometry_from_model(model, t, filt_idx, pvec)

        if ek.startswith("fsbl_photastrom"):
            from bagle.jax.fspl import fspl_astrometry_from_model

            return fspl_astrometry_from_model(model, t, filt_idx, pvec)
    except (AttributeError, NotImplementedError, TypeError):
        return None
    return None


def try_get_photometry(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_photometry_jax(layout, model, t, filt_idx)


def try_get_astrometry(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_astrometry_jax(layout, model, t, filt_idx)


def try_get_amplification(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_amplification_jax(layout, model, t, filt_idx)


def try_get_lens_astrometry(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_lens_astrometry_jax(layout, model, t, filt_idx)


def try_get_astrometry_unlensed(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_astrometry_unlensed_jax(layout, model, t, filt_idx)


def try_get_centroid_shift(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_centroid_shift_jax(layout, model, t, filt_idx)


def evaluate_method_jax(model, method_name: str, t, filt_idx: int = 0):
    """Dispatch forward evaluation by method name."""
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    dispatch = {
        "get_amplification": evaluate_amplification_jax,
        "get_photometry": evaluate_photometry_jax,
        "get_astrometry": evaluate_astrometry_jax,
        "get_astrometry_unlensed": evaluate_astrometry_unlensed_jax,
        "get_lens_astrometry": evaluate_lens_astrometry_jax,
        "get_centroid_shift": evaluate_centroid_shift_jax,
    }
    fn = dispatch.get(method_name)
    if fn is None:
        return None
    return fn(layout, model, t, filt_idx)


def _linear_astrometry_jax(t_j, t0, x0, mu, parallax_vectors, pi):
    """Source or lens linear motion in arcsec."""
    dt = ((t_j - t0) / 365.25).reshape(-1, 1)
    pos = x0.reshape(1, 2) + dt * mu.reshape(1, 2) * 1e-3
    if parallax_vectors is not None and pi is not None:
        pos = pos + float(pi) * jnp.asarray(parallax_vectors, dtype=jnp.float64) * 1e-3
    return pos


def evaluate_amplification_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    pvec = _parallax_table(model, t, filt_idx)
    ek = layout.eval_kind
    try:
        if ek in (
            "pspl_phot_static",
            "pspl_phot_log",
            "pspl_photastrom_physical",
            "pspl_photastrom_reduced",
        ):
            amp = pspl_amplification(
                t_j,
                float(model.t0),
                float(model.tE),
                jnp.asarray(model.u0, dtype=jnp.float64),
                jnp.asarray(model.thetaE_hat, dtype=jnp.float64),
                parallax_vectors=pvec,
                piE_E=float(model.piE[0]),
                piE_N=float(model.piE[1]),
            )
            return np.asarray(amp, dtype=np.float64)
        if ek.startswith("psbl_phot"):
            amp, _mag = _evaluate_psbl_phot(layout, model, t_j, pvec, 0.0, 1.0)
            return np.asarray(amp, dtype=np.float64)
        if ek.startswith("psbl_photastrom"):
            amp, _mag = _evaluate_psbl_phot(layout, model, t_j, pvec, 0.0, 1.0)
            return np.asarray(amp, dtype=np.float64)
        if ek.startswith("bspl_phot"):
            from bagle.jax.bspl import bspl_amplification_from_model

            return bspl_amplification_from_model(model, t, filt_idx, pvec)
    except (AttributeError, NotImplementedError, TypeError):
        return None
    return None


def evaluate_lens_astrometry_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    ek = layout.eval_kind
    if ek in ("pspl_phot_static", "pspl_phot_log"):
        t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
        return np.zeros((int(t_j.shape[0]), 2), dtype=np.float64)
    if layout.likelihood_mode not in ("ast", "joint", "joint_gp"):
        return None
    t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    pvec = _parallax_table(model, t, filt_idx)
    try:
        if _joint_astrom_kind(layout.eval_kind):
            pos = _linear_astrometry_jax(
                t_j,
                float(model.t0),
                jnp.asarray(model.xL0, dtype=jnp.float64),
                jnp.asarray(model.muL, dtype=jnp.float64),
                pvec,
                float(model.piL),
            )
            return np.asarray(pos, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None
    return None


def evaluate_astrometry_unlensed_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    ek = layout.eval_kind
    if ek in ("pspl_phot_static", "pspl_phot_log"):
        inputs = _pspl_phot_astrometry_inputs(model, t, filt_idx)
        if inputs is None:
            return None
        t_j, pvec, b_sff, _mag_src = inputs
        pos = pspl_phot_astrometry_unlensed(
            t_j,
            float(model.t0),
            float(model.tE),
            jnp.asarray(model.u0, dtype=jnp.float64),
            jnp.asarray(model.thetaE_hat, dtype=jnp.float64),
            b_sff,
            parallax_vectors=pvec,
            piE_E=float(model.piE[0]),
            piE_N=float(model.piE[1]),
        )
        return np.asarray(pos, dtype=np.float64)
    if layout.likelihood_mode not in ("ast", "joint", "joint_gp"):
        return None
    xL = evaluate_lens_astrometry_jax(layout, model, t, filt_idx)
    if xL is None:
        return None
    t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    pvec = _parallax_table(model, t, filt_idx)
    b_sff = _phot_attr(model, filt_idx, "b_sff") or 1.0
    try:
        if _joint_astrom_kind(layout.eval_kind):
            xS = _linear_astrometry_jax(
                t_j,
                float(model.t0),
                jnp.asarray(model.xS0, dtype=jnp.float64),
                jnp.asarray(model.muS, dtype=jnp.float64),
                pvec,
                float(model.piS),
            )
            pos = float(b_sff) * xS + (1.0 - float(b_sff)) * jnp.asarray(xL)
            return np.asarray(pos, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None
    return None


def evaluate_centroid_shift_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    ast = evaluate_astrometry_jax(layout, model, t, filt_idx)
    unl = evaluate_astrometry_unlensed_jax(layout, model, t, filt_idx)
    if ast is None or unl is None:
        return None
    shift = (jnp.asarray(ast) - jnp.asarray(unl)) * 1e3
    return np.asarray(shift, dtype=np.float64)


_PSPL_U_KINDS = (
    "pspl_phot_static",
    "pspl_phot_log",
    "pspl_photastrom_physical",
    "pspl_photastrom_reduced",
    "pspl_astrom_reduced",
)
_PSPL_ASTROM_KINDS = (
    "pspl_photastrom_physical",
    "pspl_photastrom_reduced",
    "pspl_astrom_reduced",
)


def _joint_astrom_kind(eval_kind: str) -> bool:
    """Return True for PSPL/PSBL phot+astrom layouts with linear source/lens motion."""
    return eval_kind in _PSPL_ASTROM_KINDS or eval_kind.startswith("psbl_photastrom")


def _pspl_u_jax(model, t, filt_idx: int, pvec):
    t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    return pspl_u(
        t_j,
        float(model.t0),
        float(model.tE),
        jnp.asarray(model.u0, dtype=jnp.float64),
        jnp.asarray(model.thetaE_hat, dtype=jnp.float64),
        parallax_vectors=pvec,
        piE_E=float(model.piE[0]),
        piE_N=float(model.piE[1]),
    )


def evaluate_u_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    if layout.eval_kind not in _PSPL_U_KINDS:
        return None
    try:
        pvec = _parallax_table(model, t, filt_idx)
        u = _pspl_u_jax(model, t, filt_idx, pvec)
        return np.asarray(u, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None


def evaluate_resolved_amplification_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    if layout.eval_kind not in _PSPL_U_KINDS:
        return None
    try:
        pvec = _parallax_table(model, t, filt_idx)
        amp = pspl_resolved_amplification(
            jnp.asarray(t, dtype=jnp.float64).reshape(-1),
            float(model.t0),
            float(model.tE),
            jnp.asarray(model.u0, dtype=jnp.float64),
            jnp.asarray(model.thetaE_hat, dtype=jnp.float64),
            parallax_vectors=pvec,
            piE_E=float(model.piE[0]),
            piE_N=float(model.piE[1]),
        )
        return np.asarray(amp, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None


def evaluate_source_astrometry_unlensed_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    if layout.eval_kind not in _PSPL_ASTROM_KINDS:
        return None
    try:
        t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
        pvec = _parallax_table(model, t, filt_idx)
        pos = pspl_source_astrometry_unlensed(
            t_j,
            float(model.t0),
            jnp.asarray(model.xS0, dtype=jnp.float64),
            jnp.asarray(model.muS, dtype=jnp.float64),
            parallax_vectors=pvec,
            piS=float(model.piS),
        )
        return np.asarray(pos, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None


def evaluate_resolved_astrometry_jax(
    layout: LayoutSpec,
    model,
    t,
    filt_idx: int = 0,
) -> np.ndarray | None:
    if layout.eval_kind not in _PSPL_ASTROM_KINDS:
        return None
    try:
        t_j = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
        pvec = _parallax_table(model, t, filt_idx)
        pos = pspl_resolved_astrometry(
            t_j,
            float(model.t0),
            float(model.tE),
            jnp.asarray(model.u0, dtype=jnp.float64),
            jnp.asarray(model.thetaE_hat, dtype=jnp.float64),
            jnp.asarray(model.xL0, dtype=jnp.float64),
            jnp.asarray(model.muL, dtype=jnp.float64),
            float(model.thetaE_amp),
            parallax_vectors=pvec,
            piE_E=float(model.piE[0]),
            piE_N=float(model.piE[1]),
            piL=float(model.piL),
        )
        return np.asarray(pos, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None


def evaluate_chi2_photometry_jax(
    layout: LayoutSpec,
    model,
    t,
    mag_obs,
    mag_err_obs,
    filt_idx: int = 0,
) -> np.ndarray | None:
    mag_model = evaluate_photometry_jax(layout, model, t, filt_idx)
    if mag_model is None:
        return None
    try:
        chi2 = gaussian_chi2_photometry(
            jnp.asarray(mag_model, dtype=jnp.float64),
            jnp.asarray(mag_obs, dtype=jnp.float64),
            jnp.asarray(mag_err_obs, dtype=jnp.float64),
        )
        return np.asarray(chi2, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None


def evaluate_log_likely_photometry_each_jax(
    layout: LayoutSpec,
    model,
    t,
    mag_obs,
    mag_err_obs,
    filt_idx: int = 0,
) -> np.ndarray | None:
    mag_model = evaluate_photometry_jax(layout, model, t, filt_idx)
    if mag_model is None:
        return None
    try:
        lnL = gaussian_log_likelihood_photometry_each(
            jnp.asarray(mag_model, dtype=jnp.float64),
            jnp.asarray(mag_obs, dtype=jnp.float64),
            jnp.asarray(mag_err_obs, dtype=jnp.float64),
        )
        return np.asarray(lnL, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None


def evaluate_chi2_astrometry_jax(
    layout: LayoutSpec,
    model,
    t,
    x_obs,
    y_obs,
    x_err_obs,
    y_err_obs,
    filt_idx: int = 0,
) -> np.ndarray | None:
    pos_model = evaluate_astrometry_jax(layout, model, t, filt_idx)
    if pos_model is None:
        return None
    try:
        chi2 = gaussian_chi2_astrometry(
            jnp.asarray(pos_model, dtype=jnp.float64),
            jnp.asarray(x_obs, dtype=jnp.float64),
            jnp.asarray(y_obs, dtype=jnp.float64),
            jnp.asarray(x_err_obs, dtype=jnp.float64),
            jnp.asarray(y_err_obs, dtype=jnp.float64),
        )
        return np.asarray(chi2, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None


def evaluate_log_likely_astrometry_each_jax(
    layout: LayoutSpec,
    model,
    t,
    x_obs,
    y_obs,
    x_err_obs,
    y_err_obs,
    filt_idx: int = 0,
) -> np.ndarray | None:
    pos_model = evaluate_astrometry_jax(layout, model, t, filt_idx)
    if pos_model is None:
        return None
    try:
        lnL = gaussian_log_likelihood_astrometry_each(
            jnp.asarray(pos_model, dtype=jnp.float64),
            jnp.asarray(x_obs, dtype=jnp.float64),
            jnp.asarray(y_obs, dtype=jnp.float64),
            jnp.asarray(x_err_obs, dtype=jnp.float64),
            jnp.asarray(y_err_obs, dtype=jnp.float64),
        )
        return np.asarray(lnL, dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError):
        return None


def try_get_u(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_u_jax(layout, model, t, filt_idx)


def try_get_resolved_amplification(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_resolved_amplification_jax(layout, model, t, filt_idx)


def try_get_source_astrometry_unlensed(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_source_astrometry_unlensed_jax(layout, model, t, filt_idx)


def try_get_resolved_astrometry(model, t, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_resolved_astrometry_jax(layout, model, t, filt_idx)


def try_get_chi2_photometry(model, t, mag_obs, mag_err_obs, filt_idx: int = 0):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_chi2_photometry_jax(
        layout, model, t, mag_obs, mag_err_obs, filt_idx
    )


def try_get_log_likely_photometry_each(
    model, t, mag_obs, mag_err_obs, filt_idx: int = 0
):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_log_likely_photometry_each_jax(
        layout, model, t, mag_obs, mag_err_obs, filt_idx
    )


def try_get_chi2_astrometry(
    model, t, x_obs, y_obs, x_err_obs, y_err_obs, filt_idx: int = 0
):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_chi2_astrometry_jax(
        layout, model, t, x_obs, y_obs, x_err_obs, y_err_obs, filt_idx
    )


def try_get_log_likely_astrometry_each(
    model, t, x_obs, y_obs, x_err_obs, y_err_obs, filt_idx: int = 0
):
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_log_likely_astrometry_each_jax(
        layout, model, t, x_obs, y_obs, x_err_obs, y_err_obs, filt_idx
    )


def evaluate_photometry_with_gp_jax(
    layout: LayoutSpec,
    model,
    t,
    mag_obs,
    mag_err_obs,
    filt_idx: int = 0,
    t_pred=None,
):
    """GP predictive photometry; returns ``(mean, std)`` or ``None``."""
    from bagle.jax.gp import photometry_with_gp_jax, supports_gp_layout

    if not supports_gp_layout(layout):
        return None
    return photometry_with_gp_jax(
        layout, model, t, mag_obs, mag_err_obs, filt_idx, t_pred=t_pred
    )


def try_get_photometry_with_gp(
    model,
    t,
    mag_obs,
    mag_err_obs,
    filt_idx: int = 0,
    t_pred=None,
):
    """JAX GP photometry when supported; else ``None`` (NumPy/celerite fallback)."""
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    return evaluate_photometry_with_gp_jax(
        layout, model, t, mag_obs, mag_err_obs, filt_idx, t_pred=t_pred
    )


def evaluate_forward_jax(
    model,
    method_name: str,
    t,
    filt_idx: int = 0,
    **kwargs,
) -> np.ndarray | None:
    """Dispatch forward evaluation by method name."""
    layout = resolve_layout(model.__class__)
    if layout is None:
        return None
    dispatch = {
        "get_photometry": lambda: evaluate_photometry_jax(layout, model, t, filt_idx),
        "get_amplification": lambda: evaluate_amplification_jax(layout, model, t, filt_idx),
        "get_astrometry": lambda: evaluate_astrometry_jax(layout, model, t, filt_idx),
        "get_astrometry_unlensed": lambda: evaluate_astrometry_unlensed_jax(
            layout, model, t, filt_idx
        ),
        "get_lens_astrometry": lambda: evaluate_lens_astrometry_jax(
            layout, model, t, filt_idx
        ),
        "get_centroid_shift": lambda: evaluate_centroid_shift_jax(
            layout, model, t, filt_idx
        ),
    }
    fn = dispatch.get(method_name)
    if fn is None:
        return None
    return fn()


evaluate_method_jax = evaluate_forward_jax

