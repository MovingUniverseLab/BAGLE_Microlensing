import os
import tempfile

import matplotlib
matplotlib.use('Agg')

import numpy as np
from bagle import model
from bagle import plot_models


def _make_psbl_photastrom():
    """
    Build a simple PSBL photometry+astrometry test model.

    Returns
    -------
    psbl : bagle.model.PSBL_PhotAstrom_noPar_Param2
        A caustic-crossing PSBL event used by plotter smoke tests.
    """
    # Geometric and photometric parameters, matching other PSBL tests.
    raL = 259.5
    decL = -29.0
    t0 = 57000.0
    u0 = 0.3
    tE = 200.0
    piE_E = 0.01
    piE_N = -0.01
    b_sff = np.array([1.0])
    mag_src = np.array([18.0])
    thetaE = 3.0
    xS0_E = 0.0
    xS0_N = 0.01
    muS_E = 3.0
    muS_N = 0.0
    piS = (1.0 / 8000.0) * 1e3
    q = 0.8
    sep = 3.0
    alpha = 135.0
    dmag_L1_L2 = np.array([0.0])

    psbl = model.PSBL_PhotAstrom_noPar_Param2(
        t0, u0, tE, thetaE, piS, piE_E, piE_N,
        xS0_E, xS0_N, muS_E, muS_N, q, sep, alpha,
        b_sff, mag_src, dmag_L1_L2,
        raL=raL, decL=decL, root_tol=1e-4,
    )

    return psbl


def test_plot_PSBL_static_writes_png():
    """
    Smoke-test the static PSBL geometry plotter.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Uses a short time window and few samples so the image-array
    calculation stays cheap. The Agg backend avoids a GUI display.
    """
    psbl = _make_psbl_photastrom()

    # Write into a temp dir so the test does not litter the repo.
    tmpdir = tempfile.mkdtemp()
    outfile = os.path.join(tmpdir, 'psbl_geometry_static.png')

    result = plot_models.plot_PSBL_static(
        psbl, duration=2, time_steps=40, outfile=outfile,
    )

    # Explicit None return is part of the plotter contract.
    assert result is None
    assert os.path.isfile(outfile)
    assert os.path.getsize(outfile) > 0

    return None


def test_plot_PSBL_still_callable():
    """
    Confirm plot_PSBL keeps its public signature and still runs.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    psbl = _make_psbl_photastrom()

    tmpdir = tempfile.mkdtemp()
    outfile = os.path.join(tmpdir, 'psbl_geometry.png')

    result = plot_models.plot_PSBL(
        psbl, duration=2, time_steps=40, outfile=outfile,
    )

    # Original plotter returns None implicitly (bare return).
    assert result is None
    assert os.path.isfile(outfile)
    assert os.path.getsize(outfile) > 0

    return None
