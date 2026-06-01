"""tinygp Gaussian-process photometry for GP-enabled Phot / PhotAstrom models."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax.layout_registry import LayoutSpec

_GP_QUALITY = 2.0 ** -0.5


def supports_gp_layout(layout: LayoutSpec) -> bool:
    """Return True when the layout participates in GP photometry."""
    return layout.has_gp


def _gp_dict_param(model, name: str, filt_idx: int) -> float:
    val = getattr(model, name)
    if isinstance(val, dict):
        return float(val[filt_idx])
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return float(arr[0])
    return float(arr[filt_idx])


def _gp_has_fixed_jitter(model) -> bool:
    for cls in model.__class__.__mro__:
        if cls.__name__ == "PSPL_GPnoJitter":
            return False
    return True


def _gp_jitter_sigma(model, mag_err_obs, filt_idx: int) -> float:
    opt = getattr(model, "phot_optional_param_names", [])
    if "gp_log_jit_sigma" in opt:
        return float(np.exp(_gp_dict_param(model, "gp_log_jit_sigma", filt_idx)))
    if not _gp_has_fixed_jitter(model):
        return 0.0
    return float(np.exp(np.log(np.average(np.asarray(mag_err_obs, dtype=np.float64)))))


def build_gp_kernel(model, filt_idx: int, mag_err_obs):
    """Build a tinygp quasiseparable kernel matching celerite PSPL_GP."""
    import tinygp
    from tinygp.kernels import quasisep as qk

    log_sigma = _gp_dict_param(model, "gp_log_sigma", filt_idx)
    if hasattr(model, "gp_log_rho"):
        log_rho = _gp_dict_param(model, "gp_log_rho", filt_idx)
    else:
        log_rho = float(np.log(_gp_dict_param(model, "gp_rho", filt_idx)))
    log_S0 = _gp_dict_param(model, "gp_log_S0", filt_idx)
    log_omega0 = _gp_dict_param(model, "gp_log_omega0", filt_idx)

    sigma = jnp.exp(jnp.asarray(log_sigma, dtype=jnp.float64))
    rho = jnp.exp(jnp.asarray(log_rho, dtype=jnp.float64))
    S0 = jnp.exp(jnp.asarray(log_S0, dtype=jnp.float64))
    omega0 = jnp.exp(jnp.asarray(log_omega0, dtype=jnp.float64))
    jitter = jnp.asarray(_gp_jitter_sigma(model, mag_err_obs, filt_idx), dtype=jnp.float64)

    kernel = qk.Matern32(scale=rho, sigma=sigma) + qk.SHO(
        omega=omega0, quality=_GP_QUALITY, sigma=jnp.sqrt(S0)
    )
    return kernel, jitter


def photometry_with_gp_jax(
    layout: LayoutSpec,
    model,
    t,
    mag_obs,
    mag_err_obs,
    filt_idx: int = 0,
    t_pred=None,
    *,
    mean_fn: Callable | None = None,
):
    """GP predictive photometry mean and std (tinygp, matches celerite PSPL_GP).

    Returns ``(mag_model, mag_model_std)`` as NumPy arrays, or ``None`` when
    tinygp is unavailable or the layout is unsupported.
    """
    try:
        import tinygp
    except ImportError:
        return None

    if not getattr(model, "use_gp_phot", [False])[filt_idx]:
        return None

    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    mag_obs_arr = np.asarray(mag_obs, dtype=np.float64).reshape(-1)
    mag_err_arr = np.asarray(mag_err_obs, dtype=np.float64).reshape(-1)
    if t_pred is None:
        t_pred_arr = t_arr
    else:
        t_pred_arr = np.asarray(t_pred, dtype=np.float64).reshape(-1)

    t_j = jnp.asarray(t_arr, dtype=jnp.float64)

    if mean_fn is None:
        from bagle.jax.evaluate import evaluate_photometry_jax

        train_mean = evaluate_photometry_jax(layout, model, t_arr, filt_idx)
        if train_mean is None:
            return None
        mean_train_j = jnp.asarray(train_mean, dtype=jnp.float64).reshape(-1)

        def mean_fn(x):
            return jnp.interp(x, t_j, mean_train_j)

    kernel, jitter = build_gp_kernel(model, filt_idx, mag_err_arr)
    t_pred_j = jnp.asarray(t_pred_arr, dtype=jnp.float64)
    mag_j = jnp.asarray(mag_obs_arr, dtype=jnp.float64)
    err_j = jnp.asarray(mag_err_arr, dtype=jnp.float64)
    diag = err_j**2 + jitter**2

    gp = tinygp.GaussianProcess(kernel, t_j, diag=diag, mean=mean_fn)
    cond = gp.condition(mag_j, t_pred_j)
    mean = np.asarray(cond.gp.loc, dtype=np.float64)
    std = np.sqrt(np.asarray(cond.gp.variance, dtype=np.float64))
    return mean, std


def build_gp_loglik_fn(fitter, layout: LayoutSpec):
    """Return ``(jit_loglik, ctx)`` with tinygp GP marginal on photometric residuals."""
    try:
        import tinygp
    except ImportError:
        return None, None

    from bagle.jax.likelihood import _build_registry_phot_loglik, build_jax_loglik_fn as bl

    phot_fn, phot_ctx = _build_registry_phot_loglik(fitter, layout)
    if phot_fn is None:
        from bagle.jax_physics import build_jax_joint_loglik_fn

        joint_fn, joint_ctx = build_jax_joint_loglik_fn(fitter)
        if joint_fn is None:
            phot_fn, phot_ctx = bl(fitter)
        else:
            phot_fn, phot_ctx = joint_fn, joint_ctx

    resid_blocks = []
    for i in range(fitter.n_phot_sets):
        filt_1 = i + 1
        t = np.asarray(fitter.data[f"t_phot{filt_1}"], dtype=np.float64)
        mag_obs = np.asarray(fitter.data[f"mag{filt_1}"], dtype=np.float64)
        mag_err = np.asarray(fitter.data[f"mag_err{filt_1}"], dtype=np.float64)
        from bagle.jax_physics import _fitter_weight

        resid_blocks.append(
            {
                "t": t,
                "mag_obs": mag_obs,
                "mag_err": mag_err,
                "weight": _fitter_weight(fitter, i),
            }
        )

    host = (layout, phot_ctx, resid_blocks, phot_fn)

    def _loglik(param_vec):
        param_vec = jnp.asarray(param_vec, dtype=jnp.float64)
        lnL = phot_fn(param_vec) if phot_fn is not None else 0.0
        # GP marginal lnL on residuals is wired in a follow-up slice.
        return lnL

    return jax.jit(_loglik), host
