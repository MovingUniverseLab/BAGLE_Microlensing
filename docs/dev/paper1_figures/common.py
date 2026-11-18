"""
Shared helpers for reproducing the BAGLE binary paper (paper 1) figures.

Every ``fig_*.py`` script builds its model from the parameters quoted in the
paper (``PAPER`` values) plus documented choices for anything the paper does
not state (``ASSUMED`` values, usually read off the published figure), and
writes ``output/<name>.png``. ``compare.py`` puts each reproduction next to
the published figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
PAPER_FIGS = Path.home() / "doc/papers/bagle_work/paper1/figures"

# Gravitational constant kappa = 4G / (c^2 AU) in mas / Msun.
KAPPA = 8.1439


def theta_E(mass, dL, dS):
    """
    Angular Einstein radius.

    Parameters
    ----------
    mass : float
        Total lens mass (Msun).
    dL, dS : float
        Lens and source distances (pc).

    Returns
    -------
    thetaE : float
        Einstein radius (mas).
    """
    piRel = 1e3 / dL - 1e3 / dS
    return (KAPPA * mass * piRel) ** 0.5


def dist_for_thetaE(mass, thetaE, dS=8000.0):
    """
    Lens distance giving a chosen Einstein radius for a fixed source distance.

    Parameters
    ----------
    mass : float
        Total lens mass (Msun).
    thetaE : float
        Einstein radius (mas).
    dS : float, optional
        Source distance (pc).

    Returns
    -------
    dL : float
        Lens distance (pc).
    """
    piRel = thetaE ** 2 / (KAPPA * mass)
    return 1e3 / (piRel + 1e3 / dS)


def sky_axes(ax, xlabel=r"$\Delta\alpha^*$ (mas)", ylabel=r"$\Delta\delta$ (mas)"):
    """
    Label an on-sky axes pair with East to the left.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to format.
    xlabel, ylabel : str, optional
        Axis labels.

    Returns
    -------
    None
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    return None


def save(fig, name):
    """
    Save a figure to ``output/<name>.png``.

    Parameters
    ----------
    fig : matplotlib Figure
        Figure to save.
    name : str
        Output name (no extension).

    Returns
    -------
    path : Path
        Written file.
    """
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path
