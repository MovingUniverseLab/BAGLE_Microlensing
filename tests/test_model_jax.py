import numpy as np
import pylab as plt
from scipy.signal import find_peaks
from astropy.io import fits
from astropy import constants as c
from astropy import units as u
from astropy.coordinates import SkyCoord, GCRS
from astropy.time import Time
from astropy.table import Table
import os
from bagle import model_jax
from bagle import model
#from bagle import model_fitter
#from bagle.fake_data import *
import time
import pdb
import pytest
import inspect, sys

from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation, spherical_to_cartesian, cartesian_to_spherical
from astropy.coordinates import get_body_barycentric, get_body, get_moon, get_body_barycentric_posvel

# Always generate the same fake data.
np.random.seed(0)

def test_PSPL_vs_JAX():
    mL = 10.0  # msun
    t0 = 57000.00
    xS0 = np.array([0.000, 0.000])
    # beta = -0.4 # mas
    beta = 1.4  # mas
    muS = np.array([8.0, 0.0])
    # muL = np.array([-7.0, 0.00])
    muL = np.array([0.00, 0.00])
    dL = 4000.0
    dS = 8000.0
    b_sff = 1.0
    mag_src = 19.0
    
    pspl = model.PSPL_PhotAstrom_noPar_Param1(mL,
                                              t0,
                                              beta,
                                              dL,
                                              dL / dS,
                                              xS0[0],
                                              xS0[1],
                                              muL[0],
                                              muL[1],
                                              muS[0],
                                              muS[1],
                                              [b_sff],
                                              [mag_src])
    pspl_jax = model_jax.PSPL_PhotAstrom_noPar_Param1(mL,
                                              t0,
                                              beta,
                                              dL,
                                              dL / dS,
                                              xS0[0],
                                              xS0[1],
                                              muL[0],
                                              muL[1],
                                              muS[0],
                                              muS[1],
                                              [b_sff],
                                              [mag_src])
    assert pspl.t0 == pspl_jax.t0
    assert pspl.beta == pspl_jax.beta
    assert pspl.dL == pspl_jax.dL
    assert pspl.xS0[0] == pspl_jax.xS0[0]
    assert pspl.xS0[1] == pspl_jax.xS0[1]
    assert pspl.muL[0] == pspl_jax.muL[0]
    assert pspl.muL[1] == pspl_jax.muL[1]
    assert pspl.muS[0] == pspl_jax.muS[0]
    assert pspl.muS[1] == pspl_jax.muS[1]
    assert pspl.b_sff[0] == pspl_jax.b_sff[0]
    assert pspl.mag_src[0] == pspl_jax.mag_src[0]
    
    #print(pspl.thetaE_amp, pspl_jax.thetaE_amp)
    np.testing.assert_almost_equal(pspl.thetaE_amp, pspl_jax.thetaE_amp)
    np.testing.assert_almost_equal(pspl.tE, pspl_jax.tE)
    
    t = np.arange(t0 - 3000, t0 + 3000, 1)
    dt = t - pspl.t0
    
    A = pspl.get_amplification(t)
    A_jax = pspl_jax.get_amplification(t)
    print(A[::10])
    print(A_jax[::10])
    np.testing.assert_almost_equal(A,A_jax)
    
                                              

def test_PSPL_other():
    mL = 10.0  # msun
    t0 = 57000.00
    xS0 = np.array([0.000, 0.000])
    # beta = -0.4 # mas
    beta = 1.4  # mas
    muS = np.array([8.0, 0.0])
    # muL = np.array([-7.0, 0.00])
    muL = np.array([0.00, 0.00])
    dL = 4000.0
    dS = 8000.0
    b_sff = 1.0
    mag_src = 19.0

    run_test_PSPL(mL, t0, xS0, beta, muS, muL, dL, dS, b_sff, mag_src,
                  outdir='tests/test_pspl_other/')
                  # 10.0, 57000.00, np.array([0.0,0.0]), 1.4,np.array([8.0, 0.0]),np.array([0.0,0.0]), 4000.0,8000.0,1.0,19.0
    run_test_PSPL_jax(mL, t0, xS0, beta, muS, muL, dL, dS, b_sff, mag_src,
                  outdir='tests/test_pspl_other_jax/')

    return


def run_test_PSPL_jax(mL, t0, xS0, beta, muS, muL, dL, dS, b_sff, mag_src,
                  outdir=''):
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    pspl = model_jax.PSPL_PhotAstrom_noPar_Param1(mL,
                                              t0,
                                              beta,
                                              dL,
                                              dL / dS,
                                              xS0[0],
                                              xS0[1],
                                              muL[0],
                                              muL[1],
                                              muS[0],
                                              muS[1],
                                              [b_sff],
                                              [mag_src])

    assert pspl.t0 == t0
    assert pspl.beta == beta
    assert pspl.dL == dL
    assert pspl.xS0[0] == xS0[0]
    assert pspl.xS0[1] == xS0[1]
    assert pspl.muL[0] == muL[0]
    assert pspl.muL[1] == muL[1]
    assert pspl.muS[0] == muS[0]
    assert pspl.muS[1] == muS[1]
    assert pspl.b_sff[0] == b_sff
    assert pspl.mag_src[0] == mag_src

    t = np.arange(t0 - 3000, t0 + 3000, 1)
    dt = t - pspl.t0

    A = pspl.get_amplification(t)
    shift = pspl.get_centroid_shift(t)
    shift_amp = np.linalg.norm(shift, axis=1)

    # Plot the amplification
    plt.figure(1)
    plt.clf()
    plt.plot(dt, 2.5 * np.log10(A), 'k.')
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('2.5 * log(A)')
    plt.savefig(outdir + 'amp_v_time.png')

    # Plot the positions of everything
    lens_pos = pspl.xL0 + np.outer(dt / model.days_per_year, pspl.muL) * 1e-3
    srce_pos = pspl.xS0 + np.outer(dt / model.days_per_year, pspl.muS) * 1e-3
    imag_pos = srce_pos + (shift * 1e-3)

    plt.figure(2)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1], 'r--', mfc='none', mec='red')
    plt.plot(srce_pos[:, 0], srce_pos[:, 1], 'b--', mfc='none', mec='blue')
    plt.plot(imag_pos[:, 0], imag_pos[:, 1], 'b-')
    lim = 0.005
    plt.xlim(lim, -lim)  # arcsec
    plt.ylim(-lim, lim)
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.title('Zoomed-in')
    plt.savefig(outdir + 'on_sky_zoomed.png')

    plt.figure(3)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1], 'r--', mfc='none', mec='red')
    plt.plot(srce_pos[:, 0], srce_pos[:, 1], 'b--', mfc='none', mec='blue')
    plt.plot(imag_pos[:, 0], imag_pos[:, 1], 'b-')
    lim = 0.05
    plt.xlim(lim, -lim)  # arcsec
    plt.ylim(-lim, lim)
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.title('Zoomed-out')
    plt.savefig(outdir + 'on_sky.png')

    plt.figure(4)
    plt.clf()
    plt.plot(dt, shift_amp)
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('Astrometric Shift (mas)')
    plt.savefig(outdir + 'shift_amp_v_t.png')

    plt.figure(5)
    plt.clf()
    plt.plot(shift[:, 0], shift[:, 1])
    plt.gca().invert_xaxis()
    plt.xlabel('RA Shift (mas)')
    plt.ylabel('Dec Shift (mas)')
    plt.xlim(1.5, -1.5)
    plt.ylim(-0.5, 2.5)
    plt.savefig(outdir + 'shift_on_sky.png')

    plt.close(6)
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    f.subplots_adjust(hspace=0)
    ax1.plot(dt / pspl.tE, shift[:, 0] / pspl.thetaE_amp, 'k-')
    ax2.plot(dt / pspl.tE, shift[:, 1] / pspl.thetaE_amp, 'k-')
    ax3.plot(dt / pspl.tE, shift_amp / pspl.thetaE_amp, 'k-')
    ax3.set_xlabel('(t - t0) / tE)')
    ax1.set_ylabel(r'dX / $\theta_E$')
    ax2.set_ylabel(r'dY / $\theta_E$')
    ax3.set_ylabel(r'dT / $\theta_E$')
    ax1.set_ylim(-0.4, 0.4)
    ax2.set_ylim(-0.4, 0.4)
    ax3.set_ylim(0, 0.4)
    plt.savefig(outdir + 'shift_v_t.png')

    print('Einstein radius: ', pspl.thetaE_amp)
    print('Einstein crossing time: ', pspl.tE)

    return pspl

def run_test_PSPL(mL, t0, xS0, beta, muS, muL, dL, dS, b_sff, mag_src,
                  outdir=''):
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    pspl = model.PSPL_PhotAstrom_noPar_Param1(mL,
                                              t0,
                                              beta,
                                              dL,
                                              dL / dS,
                                              xS0[0],
                                              xS0[1],
                                              muL[0],
                                              muL[1],
                                              muS[0],
                                              muS[1],
                                              [b_sff],
                                              [mag_src])

    assert pspl.t0 == t0
    assert pspl.beta == beta
    assert pspl.dL == dL
    assert pspl.xS0[0] == xS0[0]
    assert pspl.xS0[1] == xS0[1]
    assert pspl.muL[0] == muL[0]
    assert pspl.muL[1] == muL[1]
    assert pspl.muS[0] == muS[0]
    assert pspl.muS[1] == muS[1]
    assert pspl.b_sff[0] == b_sff
    assert pspl.mag_src[0] == mag_src

    t = np.arange(t0 - 3000, t0 + 3000, 1)
    dt = t - pspl.t0

    A = pspl.get_amplification(t)
    shift = pspl.get_centroid_shift(t)
    shift_amp = np.linalg.norm(shift, axis=1)

    # Plot the amplification
    plt.figure(1)
    plt.clf()
    plt.plot(dt, 2.5 * np.log10(A), 'k.')
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('2.5 * log(A)')
    plt.savefig(outdir + 'amp_v_time.png')

    # Plot the positions of everything
    lens_pos = pspl.xL0 + np.outer(dt / model.days_per_year, pspl.muL) * 1e-3
    srce_pos = pspl.xS0 + np.outer(dt / model.days_per_year, pspl.muS) * 1e-3
    imag_pos = srce_pos + (shift * 1e-3)

    plt.figure(2)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1], 'r--', mfc='none', mec='red')
    plt.plot(srce_pos[:, 0], srce_pos[:, 1], 'b--', mfc='none', mec='blue')
    plt.plot(imag_pos[:, 0], imag_pos[:, 1], 'b-')
    lim = 0.005
    plt.xlim(lim, -lim)  # arcsec
    plt.ylim(-lim, lim)
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.title('Zoomed-in')
    plt.savefig(outdir + 'on_sky_zoomed.png')

    plt.figure(3)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1], 'r--', mfc='none', mec='red')
    plt.plot(srce_pos[:, 0], srce_pos[:, 1], 'b--', mfc='none', mec='blue')
    plt.plot(imag_pos[:, 0], imag_pos[:, 1], 'b-')
    lim = 0.05
    plt.xlim(lim, -lim)  # arcsec
    plt.ylim(-lim, lim)
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.title('Zoomed-out')
    plt.savefig(outdir + 'on_sky.png')

    plt.figure(4)
    plt.clf()
    plt.plot(dt, shift_amp)
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('Astrometric Shift (mas)')
    plt.savefig(outdir + 'shift_amp_v_t.png')

    plt.figure(5)
    plt.clf()
    plt.plot(shift[:, 0], shift[:, 1])
    plt.gca().invert_xaxis()
    plt.xlabel('RA Shift (mas)')
    plt.ylabel('Dec Shift (mas)')
    plt.xlim(1.5, -1.5)
    plt.ylim(-0.5, 2.5)
    plt.savefig(outdir + 'shift_on_sky.png')

    plt.close(6)
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    f.subplots_adjust(hspace=0)
    ax1.plot(dt / pspl.tE, shift[:, 0] / pspl.thetaE_amp, 'k-')
    ax2.plot(dt / pspl.tE, shift[:, 1] / pspl.thetaE_amp, 'k-')
    ax3.plot(dt / pspl.tE, shift_amp / pspl.thetaE_amp, 'k-')
    ax3.set_xlabel('(t - t0) / tE)')
    ax1.set_ylabel(r'dX / $\theta_E$')
    ax2.set_ylabel(r'dY / $\theta_E$')
    ax3.set_ylabel(r'dT / $\theta_E$')
    ax1.set_ylim(-0.4, 0.4)
    ax2.set_ylim(-0.4, 0.4)
    ax3.set_ylim(0, 0.4)
    plt.savefig(outdir + 'shift_v_t.png')

    print('Einstein radius: ', pspl.thetaE_amp)
    print('Einstein crossing time: ', pspl.tE)

    return pspl

# test_PSPL_vs_JAX()
test_PSPL_other()