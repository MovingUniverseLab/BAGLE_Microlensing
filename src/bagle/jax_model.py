"""Thin adapters from bagle.model classes to JAX forward evaluation."""
from __future__ import annotations

from bagle.jax.evaluate import (
    try_get_astrometry,
    try_get_astrometry_unlensed,
    try_get_amplification,
    try_get_centroid_shift,
    try_get_lens_astrometry,
    try_get_photometry,
)

__all__ = [
    "try_get_photometry",
    "try_get_astrometry",
    "try_get_astrometry_unlensed",
    "try_get_lens_astrometry",
    "try_get_centroid_shift",
    "try_get_amplification",
]
