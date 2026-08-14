import os
import inspect
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def _fig_text_blob():
    """
    Concatenate figure-level text, restoring wrapped class names.

    Returns
    -------
    blob : str
        All ``figtext`` strings joined with newlines replaced by
        underscores so ``PhotAstrom`` wraps reconstruct the class name.
    """
    fig = plt.gcf()
    parts = [t.get_text().replace('\n', '_') for t in fig.texts]
    blob = ' '.join(parts)

    return blob


def test_plot_PSBL_static_shows_class_name():
    """
    The static PSBL panel title is the concrete model class name.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    psbl = _make_psbl_photastrom()
    tmpdir = tempfile.mkdtemp()
    outfile = os.path.join(tmpdir, 'psbl_class.png')

    plot_models.plot_PSBL_static(
        psbl, duration=2, time_steps=40, outfile=outfile,
    )

    blob = _fig_text_blob()
    assert type(psbl).__name__ in blob

    return None


def _make_psbl_par_ellorbs():
    """
    Build a PSBL model with parallax and Keplerian lens orbits.

    Returns
    -------
    psbl : bagle.model.PSBL_PhotAstrom_Par_EllOrbs_Param1
        Nearby-lens event used to show curved orbits and parallax.
    """
    # Nearby lens + bulge source so parallax wobble is visible.
    # Short-period Keplerian orbit (a = 5 mas at 300 pc ~ 1.5 AU).
    mLp = 0.8
    mLs = 0.5
    t0_com = 57000.0
    xS0_E = 0.0
    xS0_N = 0.0
    beta_com = 1.5
    muL_E = 5.0
    muL_N = 2.0
    muS_E = -1.0
    muS_N = 3.0
    omega_pri = 90.0
    big_omega_sec = 20.0
    i = 55.0
    e = 0.35
    tp = 57000.0
    a = 5.0
    dL = 300.0
    dS = 8000.0
    b_sff = np.array([1.0])
    mag_src = np.array([18.0])
    dmag_Lp_Ls = np.array([0.0])
    raL = 259.5
    decL = -29.0

    psbl = model.PSBL_PhotAstrom_Par_EllOrbs_Param1(
        mLp, mLs, t0_com, xS0_E, xS0_N,
        beta_com, muL_E, muL_N,
        omega_pri, big_omega_sec, i, e, tp, a,
        muS_E, muS_N, dL, dS,
        b_sff, mag_src, dmag_Lp_Ls,
        raL=raL, decL=decL, root_tol=1e-8,
    )

    return psbl


def _make_bsbl_par_ellorbs():
    """
    Build a BSBL model with parallax and Keplerian orbits.

    Returns
    -------
    bsbl : bagle.model.BSBL_PhotAstrom_Par_EllOrbs_Param1
        Nearby event with lens and source orbital motion plus parallax.
    """
    # Nearby lens + bulge source for a large parallax wobble.
    # Lens and source both have Keplerian orbits.
    mLp = 0.8
    mLs = 0.4
    t0_com = 57000.0
    xS0_E = 0.0
    xS0_N = 0.0
    beta = 1.5
    muL_E = 4.0
    muL_N = 1.0
    muS_E = -2.0
    muS_N = 3.0
    dL = 300.0
    dS = 8000.0
    omegaL_pri = 90.0
    big_omegaL_sec = 20.0
    iL = 55.0
    eL = 0.35
    tpL = 57000.0
    aL = 5.0
    omegaS_pri = 0.0
    big_omegaS_sec = 90.0
    iS = 50.0
    eS = 0.25
    pS = 500.0
    tpS = 57000.0
    alephS = 1.5
    aleph_secS = 2.0
    mag_src_pri = 18.0
    mag_src_sec = 19.0
    b_sff = np.array([1.0])
    dmag_Lp_Ls = np.array([20.0])
    raL = 259.5
    decL = -29.0

    bsbl = model.BSBL_PhotAstrom_Par_EllOrbs_Param1(
        mLp, mLs, t0_com, xS0_E, xS0_N,
        beta, muL_E, muL_N, muS_E, muS_N, dL, dS,
        omegaL_pri, big_omegaL_sec, iL, eL, tpL, aL,
        omegaS_pri, big_omegaS_sec, iS, eS, pS, tpS,
        alephS, aleph_secS,
        mag_src_pri, mag_src_sec, b_sff, dmag_Lp_Ls,
        raL=raL, decL=decL, root_tol=1e-8,
    )

    return bsbl


def test_plot_PSBL_par_ellorbs_static_writes_png():
    """
    Smoke-test plot_PSBL_static on a Par + EllOrbs model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    psbl = _make_psbl_par_ellorbs()
    tmpdir = tempfile.mkdtemp()
    outfile = os.path.join(tmpdir, 'psbl_par_ellorbs.png')

    result = plot_models.plot_PSBL_static(
        psbl, duration=4, time_steps=40, outfile=outfile,
    )

    assert result is None
    assert os.path.isfile(outfile)
    assert os.path.getsize(outfile) > 0
    assert type(psbl).__name__ in _fig_text_blob()

    return None


def test_plot_BSBL_par_ellorbs_static_writes_png():
    """
    Smoke-test plot_BSBL_static on a Par + EllOrbs model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    bsbl = _make_bsbl_par_ellorbs()
    tmpdir = tempfile.mkdtemp()
    outfile = os.path.join(tmpdir, 'bsbl_par_ellorbs.png')

    result = plot_models.plot_BSBL_static(
        bsbl, duration=4, time_steps=40, outfile=outfile,
    )

    assert result is None
    assert os.path.isfile(outfile)
    assert os.path.getsize(outfile) > 0
    assert type(bsbl).__name__ in _fig_text_blob()

    return None
