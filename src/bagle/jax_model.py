"""Thin adapters from bagle.model classes to JAX forward evaluation."""
from __future__ import annotations

from bagle.jax.evaluate import (
    try_get_astrometry,
    try_get_astrometry_unlensed,
    try_get_amplification,
    try_get_centroid_shift,
    try_get_chi2_astrometry,
    try_get_chi2_photometry,
    try_get_lens_astrometry,
    try_get_log_likely_astrometry_each,
    try_get_log_likely_photometry_each,
    try_get_photometry,
    try_get_photometry_with_gp,
    try_get_resolved_amplification,
    try_get_resolved_astrometry,
    try_get_source_astrometry_unlensed,
    try_get_u,
)

__all__ = [
    "try_get_photometry",
    "try_get_photometry_with_gp",
    "try_get_astrometry",
    "try_get_astrometry_unlensed",
    "try_get_lens_astrometry",
    "try_get_centroid_shift",
    "try_get_amplification",
    "try_get_u",
    "try_get_resolved_amplification",
    "try_get_source_astrometry_unlensed",
    "try_get_resolved_astrometry",
    "try_get_chi2_photometry",
    "try_get_chi2_astrometry",
    "try_get_log_likely_photometry_each",
    "try_get_log_likely_astrometry_each",
]
