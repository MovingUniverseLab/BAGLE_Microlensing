"""Pytest configuration for BAGLE tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Local helpers (sampler_agreement, fixtures) live in this directory.
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


def _resolve_data_path(default_path):
    """
    Return a data path, falling back to /data when the default is missing.

    Parameters
    ----------
    default_path : str
        Preferred path to a local data file or directory.

    Returns
    -------
    path : str
        ``default_path`` when it exists, otherwise the same basename under
        ``/data``.
    """
    if os.path.exists(default_path):
        return default_path

    fallback_path = os.path.join('/data', os.path.basename(default_path))
    if os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(
        f"File not found at default path '{default_path}' "
        f"or fallback path '{fallback_path}'."
    )
