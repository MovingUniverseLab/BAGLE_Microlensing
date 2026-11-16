"""
Run every reproduction script and build published-vs-BAGLE side-by-side PNGs.

Usage::

    PYTHONPATH=../../src:. python compare.py          # run scripts + compare
    PYTHONPATH=../../src:. python compare.py --no-run # compare only
"""
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from common import OUT, PAPER_FIGS, save

import fig_bspl_linacc
import fig_caustics
import fig_com_motion
import fig_keplerian
import fig_maps
import fig_parallax_comparison
import fig_psbl_linacc

# (published figure, reproduction) pairs.
PAIRS = [
    ("com_motion", "com_motion"),
    ("linorbs", "linorbs"),
    ("linorbs", "linorbs_match"),
    ("lin_lenses", "lin_lenses"),
    ("bspl_keplerian", "bspl_keplerian"),
    ("psbl_keplerian", "psbl_keplerian"),
    ("bsbl_keplerian", "bsbl_keplerian"),
    ("magmap", "magmap"),
    ("csmap", "csmap"),
    ("magmaps_varyq", "magmaps_varyq"),
    ("magmaps_varysep", "magmaps_varysep"),
    ("parallax_comparison", "parallax_comparison"),
    ("parallax_comparison_bspl", "parallax_comparison_bspl"),
]


def run_all():
    """
    Regenerate every reproduced figure.

    Returns
    -------
    None
    """
    fig_com_motion.main()
    fig_bspl_linacc.main()
    fig_psbl_linacc.main()
    fig_keplerian.psbl_keplerian()
    fig_keplerian.bspl_keplerian()
    fig_keplerian.bsbl_keplerian()
    fig_maps.magmap()
    fig_maps.csmap()
    fig_caustics.main()
    fig_parallax_comparison.psbl()
    fig_parallax_comparison.bspl()
    return None


def side_by_side(pub, rep):
    """
    Write ``output/compare_<rep>.png`` with the published figure on the left.

    Parameters
    ----------
    pub : str
        Published figure name in ``paper1/figures``.
    rep : str
        Reproduced figure name in ``output``.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    for ax, path, title in [(axs[0], PAPER_FIGS / f"{pub}.png", "Published"),
                            (axs[1], OUT / f"{rep}.png", "BAGLE (this repo)")]:
        ax.imshow(mpimg.imread(path), origin="upper")
        ax.set_title(f"{title}: {path.name}")
        ax.axis("off")
    save(fig, f"compare_{rep}")
    return None


if __name__ == "__main__":
    if "--no-run" not in sys.argv:
        run_all()
    for pub, rep in PAIRS:
        side_by_side(pub, rep)
