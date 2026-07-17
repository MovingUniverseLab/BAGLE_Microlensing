"""tinygp Gaussian-process photometry for GP-enabled Phot / PhotAstrom models."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

_GP_QUALITY = 2.0 ** -0.5


def supports_gp_model(model) -> bool:
    """Return True when the model participates in GP photometry."""
    name = type(model).__name__
    if "GP" in name or "GPnoJitter" in name:
        return True
    use = getattr(model, "use_gp_phot", None)
    if use is None:
        return False
    arr = np.asarray(use).reshape(-1)
    return bool(np.any(arr))


def supports_gp_class(model_class) -> bool:
    """Return True when a model class is GP-enabled."""
    name = model_class.__name__
    if "GP" in name or "GPnoJitter" in name:
        return True
    for cls in model_class.__mro__:
        if "GP" in cls.__name__ or "GPnoJitter" in cls.__name__:
            return True
        opt = getattr(cls, "phot_optional_param_names", None)
        if opt and any(str(n).startswith("gp_") for n in opt):
            return True
    return False


# Back-compat alias used during migration.
def supports_gp_layout(layout_or_model) -> bool:
    if hasattr(layout_or_model, "has_gp"):
        return bool(layout_or_model.has_gp)
    return supports_gp_model(layout_or_model)


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


def build_gp_kernel_from_params(gp_params: dict, mag_err_obs, fixed_jitter: bool = True):
    """Build tinygp kernel from a packed parameter dict (fitter path)."""
    from tinygp.kernels import quasisep as qk

    sigma = jnp.exp(jnp.asarray(gp_params["gp_log_sigma"], dtype=jnp.float64))
    if "gp_log_rho" in gp_params:
        rho = jnp.exp(jnp.asarray(gp_params["gp_log_rho"], dtype=jnp.float64))
    else:
        rho = jnp.asarray(gp_params["gp_rho"], dtype=jnp.float64)
    S0 = jnp.exp(jnp.asarray(gp_params["gp_log_S0"], dtype=jnp.float64))
    omega0 = jnp.exp(jnp.asarray(gp_params["gp_log_omega0"], dtype=jnp.float64))
    if "gp_log_jit_sigma" in gp_params:
        jitter = jnp.exp(jnp.asarray(gp_params["gp_log_jit_sigma"], dtype=jnp.float64))
    elif fixed_jitter:
        jitter = jnp.exp(
            jnp.log(jnp.mean(jnp.asarray(mag_err_obs, dtype=jnp.float64)))
        )
    else:
        jitter = jnp.asarray(0.0, dtype=jnp.float64)

    kernel = qk.Matern32(scale=rho, sigma=sigma) + qk.SHO(
        omega=omega0, quality=_GP_QUALITY, sigma=jnp.sqrt(S0)
    )
    return kernel, jitter


def photometry_with_gp_jax(
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
    tinygp is unavailable or GP is disabled for this filter.
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
        train_mean = np.asarray(
            model.get_photometry(t_arr, filt_idx=filt_idx), dtype=np.float64
        ).reshape(-1)
        mean_train_j = jnp.asarray(train_mean, dtype=jnp.float64)

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


def gp_log_probability(kernel, t, mag_obs, mag_err, mean, jitter):
    """tinygp marginal log-probability for one photometric filter."""
    import tinygp

    t_j = jnp.asarray(t, dtype=jnp.float64)
    mag_j = jnp.asarray(mag_obs, dtype=jnp.float64)
    err_j = jnp.asarray(mag_err, dtype=jnp.float64)
    diag = err_j**2 + jitter**2
    mean_j = jnp.asarray(mean, dtype=jnp.float64)

    def mean_fn(x):
        return jnp.interp(x, t_j, mean_j)

    gp = tinygp.GaussianProcess(kernel, t_j, diag=diag, mean=mean_fn)
    return gp.log_probability(mag_j)


def build_gp_loglik_fn(fitter, param_cls=None):
    """Return ``(jit_loglik, ctx)`` with tinygp GP marginal on photometric residuals.

    Implemented fully in :mod:`bagle.jax.likelihood` once Param packing is
    available; this entry point delegates there.
    """
    try:
        import tinygp  # noqa: F401
    except ImportError:
        return None, None

    from bagle.jax.likelihood import build_analytic_gp_loglik_fn

    return build_analytic_gp_loglik_fn(fitter, param_cls=param_cls)
