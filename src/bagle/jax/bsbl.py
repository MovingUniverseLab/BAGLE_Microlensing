"""BSBL (binary source, binary lens) JAX kernels via differentiable callback."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bagle.jax.layout_registry import LayoutSpec


def _numpy_bsbl_phot(model, t, filt_idx):
    return np.asarray(model.get_photometry(t, filt_idx=filt_idx), dtype=np.float64)


def bsbl_photometry_from_model(model, t, filt_idx, pvec):
    return _numpy_bsbl_phot(model, t, filt_idx)


def build_bsbl_joint_loglik(fitter, param_cls=None):
    """Return no backend until BSBL analytic kernels are available."""
    return None, None
