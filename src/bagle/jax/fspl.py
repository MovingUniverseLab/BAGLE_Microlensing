"""FSPL/FSBL finite-source JAX evaluation helpers."""
from __future__ import annotations

import numpy as np


def fspl_photometry_from_model(model, t, filt_idx, pvec):
    """Host numpy photometry (finite-source); used when JAX contour not wired."""
    return np.asarray(model.get_photometry(t, filt_idx=filt_idx), dtype=np.float64)


def fspl_amplification_from_model(model, t, filt_idx, pvec):
    """Host numpy amplification (finite-source); AMG path until JAX contour wired."""
    return np.asarray(model.get_amplification(t, filt_idx=filt_idx), dtype=np.float64)


def fspl_astrometry_from_model(model, t, filt_idx, pvec):
    """Host numpy astrometry when the model provides it."""
    if not getattr(model, "astrometryFlag", False):
        return None
    return np.asarray(model.get_astrometry(t, filt_idx=filt_idx), dtype=np.float64)


def finite_source_image_positions(image_arr) -> np.ndarray:
    """Convert AMG complex image positions to ``(N_t, N_img, 2)`` float."""
    img = np.asarray(image_arr)
    if np.iscomplexobj(img):
        return np.stack(
            [np.real(img), np.imag(img)], axis=-1
        ).astype(np.float64, copy=False)
    if getattr(img.dtype, "names", None) is not None:
        return img.view("(2,)float")
    return np.asarray(img, dtype=np.float64)


def fspl_resolved_astrometry_from_model(model, t, filt_idx, pvec):
    """Finite-source resolved image positions via host ``get_all_arrays``."""
    out = model.get_all_arrays(t, filt_idx=filt_idx)
    if isinstance(out, tuple) and len(out) == 4:
        img_arr = out[0]
    else:
        img_arr, _amp_arr = out
    return finite_source_image_positions(img_arr)


def fspl_resolved_amplification_from_model(model, t, filt_idx, pvec):
    """Finite-source resolved amplifications via host AMG ``get_all_arrays``.

    FSPL_PhotAstrom and BFSPL_PhotAstrom use different ``swapaxes`` conventions
    in :meth:`bagle.model.FSPL_PhotAstrom.get_resolved_amplification`.
    """
    img_arr, amp_arr = model.get_all_arrays(t, filt_idx=filt_idx)
    amp_arr = np.asarray(amp_arr, dtype=np.float64)
    if type(model).__name__.startswith("BFSPL"):
        return np.swapaxes(amp_arr, 1, 2)
    return np.swapaxes(amp_arr, 0, 1)


def fspl_centroid_shift_from_model(model, t, filt_idx, pvec):
    """Finite-source centroid shift (mas) from AMG astrometry minus unlensed."""
    if not getattr(model, "astrometryFlag", False):
        return None
    ast = fspl_astrometry_from_model(model, t, filt_idx, pvec)
    unl = np.asarray(
        model.get_astrometry_unlensed(t, filt_idx=filt_idx), dtype=np.float64
    )
    return (ast - unl) * 1e3
