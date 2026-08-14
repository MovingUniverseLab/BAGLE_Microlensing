import os
import inspect
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


def test_plot_PSBL_signature_unchanged():
    """
    Confirm plot_PSBL keeps its public signature.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This only checks the call signature so ``plot_PSBL`` stays frozen.
    """
    sig = inspect.signature(plot_models.plot_PSBL)
    params = list(sig.parameters)

    assert params == ['psbl', 'duration', 'time_steps', 'outfile']
    assert sig.parameters['duration'].default == 10
    assert sig.parameters['time_steps'].default == 300
    assert sig.parameters['outfile'].default == 'psbl_geometry.png'

    return None


def _make_bsbl_photastrom():
    """
    Build a simple BSBL photometry+astrometry test model.

    Returns
    -------
    bsbl : bagle.model.BSBL_PhotAstrom_noPar_Param1
        A static-binary BSBL event used by plotter smoke tests.
        Parameters match ``test_BSBL_PhotAstrom_Par_Param1`` in
        ``tests/test_model.py``.
    """
    # Physical parameters from the existing BSBL unit tests.
    raL = 259.5
    decL = -28.5
    mLp = 10.0
    mLs = 3.0
    t0 = 57000.0
    xS0_E = 0.001
    xS0_N = 0.0
    beta = 1.0
    muL_E = 0.0
    muL_N = 0.0
    muS_E = -3.0
    muS_N = 0.0
    dL = 4000.0
    dS = 8000.0
    sepL = 3.0
    alphaL = -35.0
    sepS = 0.5
    alphaS = 0.0
    mag_src_pri = np.array([18.0])
    mag_src_sec = np.array([19.0])
    b_sff = np.array([1.0])
    dmag_Lp_Ls = np.array([20.0])

    bsbl = model.BSBL_PhotAstrom_noPar_Param1(
        mLp, mLs, t0, xS0_E, xS0_N,
        beta, muL_E, muL_N, muS_E, muS_N,
        dL, dS, sepL, alphaL, sepS, alphaS,
        mag_src_pri, mag_src_sec, b_sff, dmag_Lp_Ls,
        raL=raL, decL=decL, root_tol=1e-4,
    )

    return bsbl


def test_plot_BSBL_static_writes_png():
    """
    Smoke-test the static BSBL geometry plotter.

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
    bsbl = _make_bsbl_photastrom()

    tmpdir = tempfile.mkdtemp()
    outfile = os.path.join(tmpdir, 'bsbl_geometry_static.png')

    result = plot_models.plot_BSBL_static(
        bsbl, duration=2, time_steps=40, outfile=outfile,
    )

    assert result is None
    assert os.path.isfile(outfile)
    assert os.path.getsize(outfile) > 0

    return None


def test_plot_BSBL_signature():
    """
    Confirm plot_BSBL has a stable public signature.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Mirrors ``test_plot_PSBL_signature_unchanged``. The animated
    plotter is not executed here because it writes an mp4.
    """
    sig = inspect.signature(plot_models.plot_BSBL)
    params = list(sig.parameters)

    assert params == ['bsbl', 'duration', 'time_steps', 'outfile']
    assert sig.parameters['duration'].default == 10
    assert sig.parameters['time_steps'].default == 300
    assert sig.parameters['outfile'].default == 'bsbl_movie'

    return None
