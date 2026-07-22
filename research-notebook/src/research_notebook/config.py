"""Load and save research-notebook configuration."""

from __future__ import annotations

import os
import socket
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(
    os.environ.get(
        "RESEARCH_NOTEBOOK_CONFIG_DIR",
        Path.home() / ".config" / "research_notebook",
    )
).expanduser()

CONFIG_PATH = CONFIG_DIR / "config.yaml"
CLIENT_SECRET_PATH = CONFIG_DIR / "client_secret.json"
TOKEN_PATH = CONFIG_DIR / "token.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "timezone": "America/Los_Angeles",
    "machine_label": socket.gethostname().split(".")[0],
    "notebook_root_name": "Research Notebook",
    "folder_ids": {
        "root": None,
        "templates": None,
        "index": None,
        "daily": None,
        "weekly": None,
        "monthly": None,
        "yearly": None,
        "projects": None,
        "shared": None,
        "assets": None,
    },
    "template_ids": {
        "daily": None,
        "weekly": None,
        "monthly": None,
    },
    "index_sheet_id": None,
    "transcripts_root": str(Path.home() / ".cursor" / "projects"),
    "project_allowlist": [],
    "project_blocklist": [
        "Graduate-Admissions",
        "empty-window",
    ],
    "reconcile_skip_subagents": True,
    "github": {
        "owners": ["jluastro", "MovingUniverseLab"],
        "authors": ["jluastro"],
        "repo_blocklist": [],
        "require_gh": True,
        "active_repos": [],
    },
}


def ensure_config_dir() -> Path:
    """Create the config directory if needed.

    Returns
    -------
    pathlib.Path
        Path to ``~/.config/research_notebook``.
    """

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config() -> dict[str, Any]:
    """Load config.yaml, merging with defaults.

    Returns
    -------
    dict
        Configuration dictionary.
    """

    ensure_config_dir()
    cfg = deepcopy(DEFAULT_CONFIG)

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        _deep_update(cfg, loaded)

    # Allow env override for machine label / notebook root path hints.
    if os.environ.get("RESEARCH_NOTEBOOK_MACHINE_LABEL"):
        cfg["machine_label"] = os.environ["RESEARCH_NOTEBOOK_MACHINE_LABEL"]

    return cfg


def save_config(cfg: dict[str, Any]) -> Path:
    """Write configuration to disk.

    Parameters
    ----------
    cfg : dict
        Configuration to save.

    Returns
    -------
    pathlib.Path
        Path written.
    """

    ensure_config_dir()
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
    return CONFIG_PATH


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Recursively update ``base`` with ``overlay`` in place."""

    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value

    return None
