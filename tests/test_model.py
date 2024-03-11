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
from bagle import model
from bagle import model_fitter
from bagle import frame_convert as fc
from bagle.fake_data import *
from bagle import frame_convert
import time
import pdb
import pytest
import inspect, sys

from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation, spherical_to_cartesian, cartesian_to_spherical
from astropy.coordinates import get_body_barycentric, get_body, get_moon, get_body_barycentric_posvel

# Always generate the same fake data.
np.random.seed(0)


def test_PSPL_other(plot=False):
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
                  outdir='tests/test_pspl_other/', plot=plot)

    return


def test_PSPL_belokurov(plot=False):
    # Scenario from Belokurov and Evans 2002 (Figure 1)
    # Note that this isn't a direct comparison; because we don't have parallax.
    mL = 0.5  # msun
    t0 = 57160.00
    xS0 = np.array([0.000, 0.000])
    beta = -7.41  # mas
    muS = np.array([-2.0, 7.0])
    muL = np.array([90.00, -24.71])
    dL = 150.0
    dS = 1500.0
    b_sff = 1.0
    mag_src = 19.0

    run_test_PSPL(mL, t0, xS0, beta, muS, muL, dL, dS, b_sff, mag_src,
                  outdir='tests/test_pspl_belokurov/', plot=plot)

    return


def run_test_PSPL(mL, t0, xS0, beta, muS, muL, dL, dS, b_sff, mag_src,
                  outdir='', plot=False):
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

    if plot:
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


def compare_pspl_parallax_belokurov():
    outdir = 'tests/test_pspl_parallax_belokurov/'
    dim_ang = u.dimensionless_angles()

    # Scenario from Belokurov and Evans 2002 (Figure 1)
    # Parameters specified in the paper: 
    mL = 0.5 * u.M_sun
    dL = 150.0 * u.pc
    dS = 1500.0 * u.pc
    vL = 70 * u.km / u.s  # assuming this is \tilde{v}
    u0 = 0.303  # Einstein radii

    # Parameters we calculate from above or guess.
    # raL = 17.5 * 15.0  # in decimal degrees
    # decL = -30.0
    raL = 10.5 * 15.0  # in decimal degrees
    decL = 20.0
    imag = 19.0
    muS = np.array([-1.75, 6.0])  # Measured from Figure 1
    # muL = (vL / dL).to(u.mas/u.yr, equivalencies=dim_ang)  # mas/yr
    muRelAmp = (vL / dL).to(u.mas / u.yr, equivalencies=dim_ang)
    # Note this is in the literature convention of muRel = muL - muS.
    # We typically use the opposite.
    muL = np.array(
        [((muRelAmp.value ** 2 - muS[1] ** 2) ** 0.5 + muS[0]), 0.0])
    # muL = np.array([-muL.value, 0.0])
    # muL = np.array([-muL.value, 0.0])

    thetaE = ((4.0 * c.G * mL / c.c ** 2) * ((1. / dL) - (1. / dS))) ** 0.5
    thetaE = thetaE.to(u.mas, equivalencies=dim_ang)  # mas
    xS0amp = u0 * thetaE  # in mas
    xS0 = (muL / np.linalg.norm(muL))[::-1] * xS0amp
    xS0 = np.array([0.0, 0.0]) * u.mas
    # xS0_E = -0.5
    # xS0 = np.array([xS0_E, -(1.0**2 - xS0_E**2)**0.5])  * xS0amp

    piRel = (u.AU / dL) - (u.AU / dS)
    piRel = piRel.to(u.mas, equivalencies=dim_ang)

    # vtilde = 1 * u.AU / (piE * tE)
    # vtilde = 1 * u.AU / (piE * (thetaE / muRel))
    # vtilde = 1 * u.AU * muRel / (piE * thetaE)
    # vtilde = 1 * u.AU * muRel / ((piRel / thetaE) * thetaE)
    # vtilde = 1 * u.AU * muRel / piRel
    # muRelAmp = vtilde * piRel / (1 * u.AU)
    # muRel = muL - muS
    # muRelAmp = vL * piRel / u.AU
    muRelAmp = muRelAmp.to(u.mas / u.yr)
    muRel = muL - muS  # opposite sign to our convention

    print('mu_rel = [{0:4.2f}, {1:4.2f}] (opposite to our convention)'.format(muRel[0], muRel[1]))
    print('mu_rel_amp = {0:4.2f}'.format(muRelAmp))
    print('mu_rel_amp = {0:4.2f}'.format(np.linalg.norm(muRel)))
    print('v_tilde =  {0:4.2f}'.format(
        (muRelAmp * dL).to(u.km / u.s, equivalencies=dim_ang)))
    print('mu_L =  [{0:4.2f}, {1:4.2f}], '.format(muL[0], muL[1]))
    print('mu_S =  [{0:4.2f}, {1:4.2f}], '.format(muS[0], muS[1]))
    print('thetaE = {0:4.2f}'.format(thetaE))
    print('piRel = {0:4.2f}'.format(piRel))
    print('xS0amp = {0:4.2f}'.format(xS0amp))
    print('xS0 =   [{0:4.2f}, {1:4.2f}], '.format(xS0[0], xS0[1]))

    beta = -xS0amp  # mas
    # t0 = 57160.00  # MJD
    t0 = 57290.00  # MJD
    # muS = np.array([-2.0, 7.0])
    # muL = np.array([90.00, -24.71])

    # Convert out of astropy units
    mL = mL.value
    xS0 = xS0.value / 1e3
    beta = beta.value
    dL = dL.value
    dS = dS.value
    # muL = np.array([0, 0])
    b_sff = 1.0

    run_test_pspl_parallax(raL, decL, mL, t0, xS0, beta, muS, muL, dL, dS,
                           b_sff, imag, outdir='tests/test_pspl_parallax_belokurov/')

    # Modify some axis limits to match the published figure.
    plt.figure(2)
    plt.gca().set_aspect('auto')
    plt.arrow(1, 10, muL[0] / 50.0, muL[1] / 50.0, head_width=0.8,
              head_length=0.5, color='black')
    plt.arrow(1, 10, muS[0] / 3.0, muS[1] / 3.0, head_width=0.3, head_length=1,
              color='blue')
    plt.text(3.5, 7, r'$\mu_L$ = {0:.1f} mas/yr'.format(np.linalg.norm(muL)),
             color='black', fontsize=12)
    plt.text(0, 12, r'$\mu_S$ = {0:.1f} mas/yr'.format(np.linalg.norm(muS)),
             color='blue', fontsize=12)

    plt.gcf().set_size_inches(8, 5)
    plt.subplots_adjust(bottom=0.2)
    plt.ylim(-16, 16)
    plt.xlim(4, -4)
    plt.xlabel(r'$\Delta \alpha^*$ (mas)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig(outdir + 'pspl_parallax_belokurov.png')

    return


def compare_pspl_parallax_han2000():
    # Scenario from Han+ 2000 (Figure 1)
    raL = 80.89375  # LMC R.A.
    decL = -69.75611  # LMC Dec.
    mL = 0.5  # msun
    dL = 10000.0
    dS = 50000.0
    xS0 = np.array([0.000, 0.0001])  # arcsec?

    tE = 100.0  # days
    u0amp = 0.2
    inv_dist_diff = (1.0 / (dL * u.pc)) - (1.0 / (dS * u.pc))
    thetaE = u.rad * np.sqrt(
        (4.0 * c.G * mL * u.M_sun / c.c ** 2) * inv_dist_diff)
    thetaE_amp = thetaE.to('mas').value  # mas

    muRelAmp = thetaE_amp / (tE / 365.25)

    print(thetaE_amp, muRelAmp)

    beta = -u0amp * thetaE_amp  # mas
    muS = np.array([muRelAmp / 2 ** 0.5, -muRelAmp / 2 ** 0.5])
    muL = np.array([0.0, 0.0])
    t0 = 57190.00  # ??
    b_sff = 1.0
    imag = 19.0

    run_test_pspl_parallax(raL, decL, mL, t0, xS0, beta, muS, muL, dL, dS,
                           b_sff, imag, outdir='tests/test_pspl_parallax_han2000/')

    return


def compare_pspl_parallax_bulge1():
    # Scenario from Belokurov and Evans 2002 (Figure 1)
    raL = 17.5 * 15.0  # in degrees
    decL = -30.0
    mL = 10.0  # msun
    t0 = 57650.0
    xS0 = np.array([0.000, 0.000])
    beta = 3.0  # mas
    muS = np.array([-4.0, -4.0])
    muL = np.array([-6.0, -10.0])
    dL = 3000.0
    dS = 6000.0
    b_sff = 1.0
    imag = 19.0

    run_test_pspl_parallax(raL, decL, mL, t0, xS0, beta, muS, muL, dL, dS,
                           b_sff, imag, outdir='tests/test_pspl_par_bulge1/')

    return


def run_test_pspl_parallax(raL, decL, mL, t0, xS0, beta, muS, muL, dL, dS,
                           b_sff, mag_src, outdir=''):
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    # No parallax
    pspl_n = model.PSPL_PhotAstrom_noPar_Param1(mL,
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
    print('pspl_n.u0', pspl_n.u0)
    print('pspl_n.muS', pspl_n.muS)
    print('pspl_n.u0_hat', pspl_n.u0_hat)
    print('pspl_n.thetaE_hat', pspl_n.thetaE_hat)

    # With parallax
    pspl_p = model.PSPL_PhotAstrom_Par_Param1(mL,
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
                                              [mag_src],
                                              raL=raL,
                                              decL=decL)

    t = np.arange(t0 - 1000, t0 + 1000, 1)
    dt = t - pspl_n.t0

    A_n = pspl_n.get_amplification(t)
    A_p = pspl_p.get_amplification(t)

    xS_n = pspl_n.get_astrometry(t)
    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)
    xL_p = pspl_p.get_lens_astrometry(t)

    # Plot the amplification
    fig1 = plt.figure(1)
    plt.clf()
    f1_1 = fig1.add_axes((0.20, 0.3, 0.75, 0.6))
    plt.plot(dt, 2.5 * np.log10(A_n), 'b-', label='No parallax')
    plt.plot(dt, 2.5 * np.log10(A_p), 'r-', label='Parallax')
    plt.legend(fontsize=10)
    plt.ylabel('2.5 * log(A)')
    f1_1.set_xticklabels([])

    f2_1 = fig1.add_axes((0.20, 0.1, 0.75, 0.2))
    plt.plot(dt, 2.5 * (np.log10(A_p) - np.log10(A_n)), 'k-',
             label='Par - No par')
    plt.axhline(0, linestyle='--', color='k')
    plt.legend(fontsize=10)
    plt.ylabel('Diff')
    plt.xlabel('t - t0 (MJD)')

    plt.savefig(outdir + 'amp_v_time.png')
    print("save to " + outdir)

    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_n[:, 0] * 1e3, xS_n[:, 1] * 1e3, 'r--',
             mfc='none', mec='red', label='Src, No parallax model')
    plt.plot(xS_p_unlens[:, 0] * 1e3, xS_p_unlens[:, 1] * 1e3, 'b--',
             mfc='none', mec='blue',
             label='Src, Parallax model, unlensed')
    plt.plot(xL_p[:, 0] * 1e3, xL_p[:, 1] * 1e3, 'k--',
             mfc='none', mec='grey', label='Lens')
    plt.plot(xS_p_lensed[:, 0] * 1e3, xS_p_lensed[:, 1] * 1e3, 'b-',
             label='Src, Parallax model, lensed')
    plt.legend(fontsize=10)
    plt.gca().invert_xaxis()
    plt.xlabel('R.A. (mas)')
    plt.ylabel('Dec. (mas)')
    plt.axis('equal')
    lim = 20
    print('LIM = ', lim)
    # plt.xlim(lim, -lim) # arcsec
    # plt.ylim(-lim, lim)
    # plt.axis('tight')
    # plt.xlim(0.7, -0.7)
    # plt.ylim(-0.7, 0.7)
    plt.savefig(outdir + 'on_sky.png')

    # Check just the astrometric shift part.
    shift_n = pspl_n.get_centroid_shift(t)  # mas
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3  # mas
    shift_n_amp = np.linalg.norm(shift_n, axis=1)
    shift_p_amp = np.linalg.norm(shift_p, axis=1)

    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.20, 0.3, 0.75, 0.6))
    plt.plot(dt, shift_n_amp, 'r--', label='No parallax model')
    plt.plot(dt, shift_p_amp, 'b--', label='Parallax model')
    plt.ylabel('Astrometric Shift (mas)')
    plt.legend(fontsize=10)
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.20, 0.1, 0.75, 0.2))
    plt.plot(dt, shift_p_amp - shift_n_amp, 'k-', label='Par - No par')
    plt.legend(fontsize=10)
    plt.axhline(0, linestyle='--', color='k')
    plt.ylabel('Diff (mas)')
    plt.xlabel('t - t0 (MJD)')

    plt.savefig(outdir + 'shift_amp_v_t.png')

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_n[:, 0], shift_n[:, 1], 'r-', label='No parallax')
    plt.plot(shift_p[:, 0], shift_p[:, 1], 'b-', label='Parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend(fontsize=10)
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')
    plt.savefig(outdir + 'shift_on_sky.png')

    print('Einstein radius: ', pspl_n.thetaE_amp, pspl_p.thetaE_amp)
    print('Einstein crossing time: ', pspl_n.tE, pspl_n.tE)

    return


def compare_pspl_parallax_paczynski1998(t0=57000):
    """
    I can't quite get this one to match!!! Why not? Maybe they kept in the parallax of the source?
    i.e. just removed proper motions. 
    """

    outdir = 'tests/test_pspl_parallax_paczynski1998/'
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    # Scenarios from Paczynski 1998
    raL = 80.89375  # LMC R.A.
    raL = 240.0  # LMC R.A.
    # decL = -69.75611 # LMC Dec.
    decL = -71.74  # LMC Dec. This is the sin \beta = -0.99 where \beta =
    mL = 0.3  # msun
    # t0 = 57000.00
    xS0 = np.array([0.000, 0.088e-3])  # arcsec
    beta = 0.088  # mas
    # muS = np.array([-3.18, -0.28])
    # muL = np.array([0.0, 0.0])
    muS = np.array([-4.18, -0.28])
    muL = np.array([0.0, 0.0])
    # muS = np.array([-2.4, -0.00000001])
    # muL = np.array([0.0, 0.0])
    dL = 10e3  # 10 kpc
    dS = 50e3  # 50 kpc in LMC
    b_sff = 1.0
    mag_src = 19.0

    # No parallax
    pspl_n = model.PSPL_PhotAstrom_noPar_Param1(mL,
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
    print('pspl_n.u0', pspl_n.u0)
    print('pspl_n.muS', pspl_n.muS)
    print('pspl_n.u0_hat', pspl_n.u0_hat)
    print('pspl_n.thetaE_hat', pspl_n.thetaE_hat)

    # With parallax
    pspl_p = model.PSPL_PhotAstrom_Par_Param1(mL,
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
                                              [mag_src],
                                              raL=raL,
                                              decL=decL)
    print('pspl_p.u0', pspl_p.u0)
    print('pspl_p.muS', pspl_p.muS)
    print('pspl_p.u0_hat', pspl_p.u0_hat)
    print('pspl_p.thetaE_hat', pspl_p.thetaE_hat)

    # t = np.arange(56000, 58000, 1)
    t = np.arange(t0 - 500, t0 + 500, 1)
    dt = t - pspl_n.t0

    A_n = pspl_n.get_amplification(t)
    A_p = pspl_p.get_amplification(t)

    xS_n = pspl_n.get_astrometry(t)
    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)
    xL_p_unlens = pspl_p.get_lens_astrometry(t)

    thetaS = (xS_p_unlens - xL_p_unlens) * 1e3  # mas
    u = thetaS / pspl_p.tE
    thetaS_lensed = (xS_p_lensed - xL_p_unlens) * 1e3  # mas

    shift_n = pspl_n.get_centroid_shift(t)  # mas
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3  # mas
    shift_n_amp = np.linalg.norm(shift_n, axis=1)
    shift_p_amp = np.linalg.norm(shift_p, axis=1)

    # Plot the amplification
    fig1 = plt.figure(1)
    plt.clf()
    f1_1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    plt.plot(dt, 2.5 * np.log10(A_n), 'b-', label='No parallax')
    plt.plot(dt, 2.5 * np.log10(A_p), 'r-', label='Parallax')
    plt.legend()
    plt.ylabel('2.5 * log(A)')
    f1_1.set_xticklabels([])

    f2_1 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    plt.plot(dt, 2.5 * (np.log10(A_p) - np.log10(A_n)), 'k-',
             label='Par - No par')
    plt.axhline(0, linestyle='--', color='k')
    plt.legend()
    plt.ylabel('Diff')
    plt.xlabel('t - t0 (MJD)')

    idx = np.argmin(np.abs(t - t0))

    plt.savefig(outdir + 'fig1.png')

    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_n[:, 0], xS_n[:, 1], 'r--', mfc='none', mec='red',
             label='No parallax model')
    plt.plot(xS_p_unlens[:, 0], xS_p_unlens[:, 1], 'b--', mfc='blue',
             mec='blue',
             label='Parallax model, unlensed')
    plt.plot(xS_p_lensed[:, 0], xS_p_lensed[:, 1], 'b-',
             label='Parallax model, lensed')
    plt.plot(xL_p_unlens[:, 0], xL_p_unlens[:, 1], 'g--', mfc='none',
             mec='green',
             label='Parallax model, Lens')
    plt.plot(xS_n[idx, 0], xS_n[idx, 1], 'rx')
    plt.plot(xS_p_unlens[idx, 0], xS_p_unlens[idx, 1], 'bx')
    plt.plot(xS_p_lensed[idx, 0], xS_p_lensed[idx, 1], 'bx')
    plt.plot(xL_p_unlens[idx, 0], xL_p_unlens[idx, 1], 'gx')
    plt.legend()
    plt.gca().invert_xaxis()
    # lim = 0.05
    # plt.xlim(lim, -lim) # arcsec
    # plt.ylim(-lim, lim)
    # plt.xlim(0.006, -0.006) # arcsec
    # plt.ylim(-0.02, 0.02)
    plt.xlabel('R.A. (")')
    plt.ylabel('Dec. (")')

    plt.savefig(outdir + 'fig2.png')

    # Check just the astrometric shift part.
    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.2, 0.3, 0.7, 0.6))
    plt.plot(dt, shift_n_amp, 'r--', label='No parallax model')
    plt.plot(dt, shift_p_amp, 'b--', label='Parallax model')
    plt.legend(fontsize=10)
    plt.ylabel('Astrometric Shift (mas)')
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.2, 0.1, 0.7, 0.2))
    plt.plot(dt, shift_p_amp - shift_n_amp, 'k-', label='Par - No par')
    plt.legend()
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('Res.')

    plt.savefig(outdir + 'fig3.png')

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_n[:, 0], shift_n[:, 1], 'r-', label='No parallax')
    plt.plot(shift_p[:, 0], shift_p[:, 1], 'b-', label='Parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')

    plt.savefig(outdir + 'fig4.png')

    plt.figure(5)
    plt.clf()
    plt.plot(thetaS[:, 0], shift_p[:, 0], 'r-', label='RA')
    plt.plot(thetaS[:, 1], shift_p[:, 1], 'b-', label='Dec')
    plt.xlabel('thetaS (")')
    plt.ylabel('Shift (mas)')

    plt.savefig(outdir + 'fig5.png')

    plt.figure(6)
    plt.clf()
    plt.plot(thetaS[:, 0], thetaS[:, 1], 'r-', label='Unlensed')
    plt.plot(thetaS_lensed[:, 0], thetaS_lensed[:, 1], 'b-', label='Lensed')
    plt.axvline(0, linestyle='--', color='k')
    plt.legend()
    plt.xlabel('thetaS_E (")')
    plt.ylabel('thetaS_N (")')

    plt.savefig(outdir + 'fig6.png')

    print('Einstein radius: ', pspl_n.thetaE_amp, pspl_p.thetaE_amp)
    print('Einstein crossing time: ', pspl_n.tE, pspl_n.tE)

    return


def compare_pspl_parallax_boden1998(t0=57000):
    """
    I can get this one to match Figure 6 of Boden et al. 1998.
    """

    # Scenarios from Paczynski 1998
    raL = 80.89375  # LMC R.A.
    decL = -71.74  # LMC Dec. This is the sin \beta = -0.99 where \beta =
    mL = 0.1  # msun
    xS0 = np.array([0.000, 0.088e-3])  # arcsec
    beta = -0.16  # mas  same as p=0.4
    muS = np.array([-2.0, 1.5])
    muL = np.array([0.0, 0.0])
    dL = 8e3  # 10 kpc
    dS = 50e3  # 50 kpc in LMC
    b_sff = 1.0
    mag_src = 19.0

    # No parallax
    pspl_n = model.PSPL_PhotAstrom_noPar_Param1(mL,
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
    print('pspl_n.u0', pspl_n.u0)
    print('pspl_n.muS', pspl_n.muS)
    print('pspl_n.u0_hat', pspl_n.u0_hat)
    print('pspl_n.thetaE_hat', pspl_n.thetaE_hat)

    # With parallax
    pspl_p = model.PSPL_PhotAstrom_Par_Param1(mL,
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
                                              [mag_src],
                                              raL=raL,
                                              decL=decL)
    print('pspl_p.u0', pspl_p.u0)
    print('pspl_p.muS', pspl_p.muS)
    print('pspl_p.u0_hat', pspl_p.u0_hat)
    print('pspl_p.thetaE_hat', pspl_p.thetaE_hat)

    # t = np.arange(56000, 58000, 1)
    t = np.arange(t0 - 500, t0 + 500, 1)
    dt = t - pspl_n.t0

    A_n = pspl_n.get_amplification(t)
    A_p = pspl_p.get_amplification(t)

    xS_n = pspl_n.get_astrometry(t)
    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)
    xL_p_unlens = pspl_p.get_lens_astrometry(t)

    thetaS = (xS_p_unlens - xL_p_unlens) * 1e3  # mas
    u = thetaS / pspl_p.tE
    thetaS_lensed = (xS_p_lensed - xL_p_unlens) * 1e3  # mas

    shift_n = pspl_n.get_centroid_shift(t)  # mas
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3  # mas
    shift_n_amp = np.linalg.norm(shift_n, axis=1)
    shift_p_amp = np.linalg.norm(shift_p, axis=1)

    # Plot the amplification
    fig1 = plt.figure(1)
    plt.clf()
    f1_1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    plt.plot(dt, 2.5 * np.log10(A_n), 'b-', label='No parallax')
    plt.plot(dt, 2.5 * np.log10(A_p), 'r-', label='Parallax')
    plt.legend()
    plt.ylabel('2.5 * log(A)')
    f1_1.set_xticklabels([])

    f2_1 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    plt.plot(dt, 2.5 * (np.log10(A_p) - np.log10(A_n)), 'k-',
             label='Par - No par')
    plt.axhline(0, linestyle='--', color='k')
    plt.legend()
    plt.ylabel('Diff')
    plt.xlabel('t - t0 (MJD)')

    idx = np.argmin(np.abs(t - t0))

    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_n[:, 0], xS_n[:, 1], 'r--', mfc='none', mec='red',
             label='No parallax model')
    plt.plot(xS_p_unlens[:, 0], xS_p_unlens[:, 1], 'b--', mfc='blue',
             mec='blue',
             label='Parallax model, unlensed')
    plt.plot(xS_p_lensed[:, 0], xS_p_lensed[:, 1], 'b-',
             label='Parallax model, lensed')
    plt.plot(xL_p_unlens[:, 0], xL_p_unlens[:, 1], 'g--', mfc='none',
             mec='green',
             label='Parallax model, Lens')
    plt.plot(xS_n[idx, 0], xS_n[idx, 1], 'rx')
    plt.plot(xS_p_unlens[idx, 0], xS_p_unlens[idx, 1], 'bx')
    plt.plot(xS_p_lensed[idx, 0], xS_p_lensed[idx, 1], 'bx')
    plt.plot(xL_p_unlens[idx, 0], xL_p_unlens[idx, 1], 'gx')
    plt.legend()
    plt.gca().invert_xaxis()
    # lim = 0.05
    # plt.xlim(lim, -lim) # arcsec
    # plt.ylim(-lim, lim)
    # plt.xlim(0.006, -0.006) # arcsec
    # plt.ylim(-0.02, 0.02)
    plt.xlabel('R.A. (")')
    plt.ylabel('Dec. (")')

    # Check just the astrometric shift part.
    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.2, 0.3, 0.7, 0.6))
    plt.plot(dt, shift_n_amp, 'r--', label='No parallax model')
    plt.plot(dt, shift_p_amp, 'b--', label='Parallax model')
    plt.legend(fontsize=10)
    plt.ylabel('Astrometric Shift (mas)')
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.2, 0.1, 0.7, 0.2))
    plt.plot(dt, shift_p_amp - shift_n_amp, 'k-', label='Par - No par')
    plt.legend()
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('Res.')

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_n[:, 0], shift_n[:, 1], 'r-', label='No parallax')
    plt.plot(shift_p[:, 0], shift_p[:, 1], 'b-', label='Parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')

    plt.figure(5)
    plt.clf()
    plt.plot(thetaS[:, 0], shift_p[:, 0], 'r-', label='RA')
    plt.plot(thetaS[:, 1], shift_p[:, 1], 'b-', label='Dec')
    plt.xlabel('thetaS (")')
    plt.ylabel('Shift (mas)')

    plt.figure(6)
    plt.clf()
    plt.plot(thetaS[:, 0], thetaS[:, 1], 'r-', label='Unlensed')
    plt.plot(thetaS_lensed[:, 0], thetaS_lensed[:, 1], 'b-', label='Lensed')
    plt.axvline(0, linestyle='--', color='k')
    plt.legend()
    plt.xlabel('thetaS_E (")')
    plt.ylabel('thetaS_N (")')

    print('Einstein radius: ', pspl_n.thetaE_amp, pspl_p.thetaE_amp)
    print('Einstein crossing time: ', pspl_n.tE, pspl_n.tE)

    return


def compare_PSPL_phot_Lu2016():
    """
    Compare observed photometry to model for
    OB120169 as listed in Table 6 (photometry
    solution #1.
    """
    raL = (17.0 + (49.0 / 60.) + (51.38 / 3600.0)) * 15.0  # degrees
    decL = -35 + (22.0 / 60.0) + (28.0 / 3600.0)
    t0 = 56026.03
    u0_amp = -0.222
    tE = 135.0
    piE_E = -0.058
    piE_N = 0.11
    b_sff = [1.1]
    mag_src = [19.266]

    # Read in the OGLE I-band photometry.
    tests_dir = os.path.dirname(os.path.realpath(__file__))
    dat = Table.read(tests_dir + '/data/OB120169_phot.dat', format='ascii')
    dat['col1'] -= 2400000.5
    dat.rename_column('col1', 'mjd')
    dat.rename_column('col2', 'I')
    dat.rename_column('col3', 'Ierr')

    t_mod = np.arange(dat['mjd'].min(), dat['mjd'].max(), 10)

    def plot_data_model(dat, t_mod, I_mod, I_mod_at_tobs, fig_num=1, title=''):
        plt.clf()
        f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, num=fig_num,
                                     gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0)
        ax1.errorbar(dat['mjd'], dat['I'], yerr=dat['Ierr'],
                     fmt='.', alpha=0.5, color='red')
        ax1.plot(t_mod, I_mod, color='black')

        ax2.errorbar(dat['mjd'], dat['I'] - I_mod_at_tobs, yerr=dat['Ierr'],
                     fmt='.', alpha=0.5, color='red')
        ax2.axhline(0, color='black')

        ax1.invert_yaxis()
        ax1.set_ylabel('I-band')
        ax2.set_ylabel('Resid.')
        ax2.set_xlabel('Time (MJD)')
        ax1.set_title(title)

        return

    ##########
    # Test #0: PSPL_phot - no blending
    ##########
    mod = model.PSPL_Phot_noPar_Param1(t0,
                                       u0_amp,
                                       tE,
                                       piE_E,
                                       piE_N,
                                       [1.0],
                                       mag_src)
    I_mod = mod.get_photometry(t_mod)
    I_mod_at_tobs = mod.get_photometry(dat['mjd'])

    plt.figure(1)
    plot_data_model(dat, t_mod, I_mod, I_mod_at_tobs,
                    fig_num=1, title='PSPL_phot b_sff=0')

    ##########
    # Test #1: PSPL_phot
    ##########
    mod = model.PSPL_Phot_noPar_Param1(t0,
                                       u0_amp,
                                       tE,
                                       piE_E,
                                       piE_N,
                                       b_sff,
                                       mag_src)

    I_mod = mod.get_photometry(t_mod)
    I_mod_at_tobs = mod.get_photometry(dat['mjd'])

    plt.figure(2)
    plot_data_model(dat, t_mod, I_mod, I_mod_at_tobs,
                    fig_num=2, title='PSPL_phot')

    ##########
    # Test #1: PSPL_phot_parallax
    ##########
    mod = model.PSPL_Phot_Par_Param1(t0,
                                     u0_amp,
                                     tE,
                                     piE_E,
                                     piE_N,
                                     b_sff,
                                     mag_src,
                                     raL=raL,
                                     decL=decL)
    I_mod = mod.get_photometry(t_mod)
    I_mod_at_tobs = mod.get_photometry(dat['mjd'])

    plt.figure(3)
    plot_data_model(dat, t_mod, I_mod, I_mod_at_tobs,
                    fig_num=3, title='PSPL_phot_parallax')

    return


def test_pspl_parallax2_bulge():
    outdir = 'tests/test_pspl_par2_bulge1/'

    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    # Scenario from Belokurov and Evans 2002 (Figure 1)
    raL = 17.5 * 15.0  # in degrees
    decL = -30.0
    mL = 10.0  # msun
    t0 = 57650.0
    xS0 = np.array([0.000, 0.000])
    beta = 3.0  # mas
    muS = np.array([-4.0, -4.0])
    muL = np.array([-6.0, -10.0])
    dL = 3000.0
    dS = 6000.0
    b_sff = 1.0
    mag_src = 19.0

    pspl_par1 = model.PSPL_PhotAstrom_Par_Param1(mL,
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
                                                 [mag_src],
                                                 raL=raL,
                                                 decL=decL)

    pspl_par2 = model.PSPL_PhotAstrom_Par_Param2(pspl_par1.t0,
                                                 pspl_par1.u0_amp,
                                                 pspl_par1.tE,
                                                 pspl_par1.thetaE_amp,
                                                 pspl_par1.piS,
                                                 pspl_par1.piE_E,
                                                 pspl_par1.piE_N,
                                                 pspl_par1.xS0[0],
                                                 pspl_par1.xS0[1],
                                                 pspl_par1.muS[0],
                                                 pspl_par1.muS[1],
                                                 [b_sff],
                                                 [mag_src],
                                                 raL=raL,
                                                 decL=decL)

    members1 = vars(pspl_par1)
    members2 = vars(pspl_par2)

    # Check results with assertions
    for kk in members1.keys():
        if kk in members2.keys():
            print('{0:13s}  {1:25s}  {2:25s}'.format(kk, str(members1[kk]),
                                                     str(members2[kk])))

            if isinstance(members1[kk], str):
                assert members1[kk] == members2[kk]
            else:
                np.testing.assert_almost_equal(members1[kk], members2[kk], 3)

    t = np.arange(t0 - 1000, t0 + 1000, 1)
    dt = t - pspl_par1.t0

    mag_out1 = pspl_par1.get_photometry(t)
    mag_out2 = pspl_par2.get_photometry(t)
    plt.figure(1)
    plt.clf()
    plt.plot(t, mag_out1, 'k-', label='mod 1')
    plt.plot(t, mag_out2, 'r-', label='mod 2')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.xlabel('Time (days)')
    plt.ylabel('Magnitude')
    plt.savefig(outdir + 'comp_mod_phot.png')

    upos_out1 = pspl_par1.get_astrometry_unlensed(t)
    upos_out2 = pspl_par2.get_astrometry_unlensed(t)
    pos_out1 = pspl_par1.get_astrometry(t)
    pos_out2 = pspl_par2.get_astrometry(t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, upos_out1[:, 0] * 1e3, 'k--', label='mod 1 unlens')
    plt.plot(t, upos_out2[:, 0] * 1e3, 'r--', label='mod 2 unlens')
    plt.plot(t, pos_out1[:, 0] * 1e3, 'k-', label='mod 1')
    plt.plot(t, pos_out2[:, 0] * 1e3, 'r-', label='mod 2')
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel(r'$\alpha^*$ (mas)')
    plt.savefig(outdir + 'comp_mod_posX.png')

    plt.figure(3)
    plt.clf()
    plt.plot(t, upos_out1[:, 1] * 1e3, 'k--', label='mod 1 unlens')
    plt.plot(t, upos_out2[:, 1] * 1e3, 'r--', label='mod 2 unlens')
    plt.plot(t, pos_out1[:, 1] * 1e3, 'k-', label='mod 1')
    plt.plot(t, pos_out2[:, 1] * 1e3, 'r-', label='mod 2')
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel(r'$\delta$ (mas)')
    plt.savefig(outdir + 'comp_mod_posX.png')

    plt.figure(4)
    plt.clf()
    plt.plot(upos_out1[:, 0] * 1e3, upos_out1[:, 1] * 1e3, 'k--',
             label='mod 1 unlens')
    plt.plot(upos_out2[:, 0] * 1e3, upos_out2[:, 1] * 1e3, 'r--',
             label='mod 2 unlens')
    plt.plot(pos_out1[:, 0] * 1e3, pos_out1[:, 1] * 1e3, 'k-', label='mod 1')
    plt.plot(pos_out2[:, 0] * 1e3, pos_out2[:, 1] * 1e3, 'r-', label='mod 2')
    plt.xlabel(r'$\alpha^*$ (mas)')
    plt.ylabel(r'$\delta$ (mas)')
    plt.legend()
    plt.savefig(outdir + 'comp_mod_posX.png')

    t = np.arange(t0 - 1000, t0 + 1000, 10)
    dt = t - pspl_par1.t0

    # Compare that we get some the same things out of the two models.
    np.testing.assert_almost_equal(pspl_par1.mL, pspl_par2.mL, 3)
    np.testing.assert_almost_equal(pspl_par1.dL, pspl_par2.dL)
    np.testing.assert_almost_equal(pspl_par1.dS, pspl_par2.dS)
    np.testing.assert_almost_equal(pspl_par1.piS, pspl_par2.piS)
    np.testing.assert_almost_equal(pspl_par1.piL, pspl_par2.piL)
    np.testing.assert_almost_equal(pspl_par1.muS, pspl_par2.muS)
    np.testing.assert_almost_equal(pspl_par1.muL, pspl_par2.muL)
    np.testing.assert_almost_equal(pspl_par1.muRel, pspl_par2.muRel)

    A_1 = pspl_par1.get_amplification(t)
    A_2 = pspl_par2.get_amplification(t)
    np.testing.assert_almost_equal(A_1, A_2)

    xS_1 = pspl_par1.get_astrometry(t)
    xS_2 = pspl_par2.get_astrometry(t)
    np.testing.assert_almost_equal(xS_1, xS_2)

    xS_unlens_1 = pspl_par1.get_astrometry_unlensed(t)
    xS_unlens_2 = pspl_par2.get_astrometry_unlensed(t)
    np.testing.assert_almost_equal(xS_unlens_1, xS_unlens_2)

    xL_1 = pspl_par1.get_lens_astrometry(t)
    xL_2 = pspl_par2.get_lens_astrometry(t)
    np.testing.assert_almost_equal(xL_1, xL_2)

    # Check just the astrometric shift part.
    shift_1 = pspl_par1.get_centroid_shift(t)  # mas
    shift_2 = pspl_par2.get_centroid_shift(t)  # mas
    np.testing.assert_almost_equal(shift_1, shift_2)

    return


def compare_lumlens_parallax_bulge():
    raL_in = 17.30 * 15.  # Bulge R.A.
    decL_in = -29.0
    mL_in = 10.0  # msun
    t0_in = 57000.0
    xS0_in = np.array([0.000, 0.088e-3])  # arcsec
    beta_in = 2.0  # mas  same as p=0.4
    muS_in = np.array([-5.0, 0.0])
    muL_in = np.array([0.0, 0.0])
    dL_in = 4000.0  # pc
    dS_in = 8000.0  # pc
    b_sff_in = 0.5
    mag_src_in = 19.0

    pspl_ll = model.PSPL_PhotAstrom_LumLens_Par_Param1(mL=mL_in,
                                                       t0=t0_in,
                                                       beta=beta_in,
                                                       dL=dL_in,
                                                       dL_dS=dL_in / dS_in,
                                                       xS0_E=xS0_in[0],
                                                       xS0_N=xS0_in[1],
                                                       muL_E=muL_in[0],
                                                       muL_N=muL_in[1],
                                                       muS_E=muS_in[0],
                                                       muS_N=muS_in[1],
                                                       raL=raL_in,
                                                       decL=decL_in,
                                                       b_sff=[b_sff_in],
                                                       mag_src=[mag_src_in])

    pspl = model.PSPL_PhotAstrom_Par_Param1(mL=mL_in,
                                            t0=t0_in,
                                            beta=beta_in,
                                            dL=dL_in,
                                            dL_dS=dL_in / dS_in,
                                            xS0_E=xS0_in[0],
                                            xS0_N=xS0_in[1],
                                            muL_E=muL_in[0],
                                            muL_N=muL_in[1],
                                            muS_E=muS_in[0],
                                            muS_N=muS_in[1],
                                            raL=raL_in,
                                            decL=decL_in,
                                            b_sff=[b_sff_in],
                                            mag_src=[mag_src_in])

    t = np.linspace(t0_in - 1500, t0_in + 1500, 1000)

    mag = pspl.get_photometry(t)
    pos = pspl.get_astrometry(t)
    pos_src = pspl.get_astrometry_unlensed(t)

    mag_ll = pspl_ll.get_photometry(t)
    pos_ll = pspl_ll.get_astrometry(t)
    pos_src_ll = pspl_ll.get_astrometry_unlensed(t)

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(left=0.2)
    #    plt.plot(pos[:, 0], pos[:, 1], label='Dark Lens')
    #    plt.plot(pos_ll[:, 0], pos_ll[:, 1], label='Lum Lens')
    #    plt.plot(pos_src[:, 0], pos_src[:, 1], label='Dark Unlens')
    #    plt.plot(pos_src_ll[:, 0], pos_src_ll[:, 1], label='Lum Unlens')
    plt.plot((pos[:, 0] - pos_src[:, 0]) * 1E3, (pos[:, 1] - pos_src[:, 1]) * 1E3, label='Dark Lens')
    plt.plot((pos_ll[:, 0] - pos_src_ll[:, 0]) * 1E3, (pos_ll[:, 1] - pos_src_ll[:, 1]) * 1E3, label='Lum Lens')
    plt.legend()
    plt.axis('equal')
    plt.xlabel('$\delta_{c,x}$ (mas)')
    plt.ylabel('$\delta_{c,y}$ (mas)')
    plt.show()
    #
    plt.figure(2)
    plt.clf()
    plt.plot(t, mag, label='Dark Lens')
    plt.plot(t, mag_ll, label='Lum Lens')
    plt.gca().invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Mag')
    plt.legend()
    plt.show()


def test_parallax(plot=False, verbose=False):
    """
    Compare our parallax vector and motion equations with
    Astropy (which now has it implemented and is well tested
    with Gaia analysis. 
    """
    # Make a parallax model to use our code directly.
    raL_in = 17.30 * 15.  # Bulge R.A.
    decL_in = -29.0
    mL_in = 10.0  # msun
    t0_in = 57000.0
    xS0_in = np.array([0.000, 0.000])  # arcsec
    beta_in = 2.0  # mas  same as p=0.4
    muS_in = np.array([0.0, 0.0])
    muL_in = np.array([-5.0, 0.0])
    dL_in = 4000.0  # pc
    dS_in = 8000.0  # pc
    b_sff_in = 1.0
    mag_src_in = 19.0

    ##########
    # BAGLE
    ##########
    pspl = model.PSPL_PhotAstrom_Par_Param1(mL_in,
                                            t0_in,
                                            beta_in,
                                            dL_in,
                                            dL_in / dS_in,
                                            xS0_in[0],
                                            xS0_in[1],
                                            muL_in[0],
                                            muL_in[1],
                                            muS_in[0],
                                            muS_in[1],
                                            [b_sff_in],
                                            [mag_src_in],
                                            raL=raL_in,
                                            decL=decL_in)

    # Fetch the astrometry for the unlensed source and lens.
    # Note these are the positions in the observers geocentric frame; but
    # fixed to stars reference. 
    t = np.linspace(-2 * pspl.tE, 2 * pspl.tE, 300)  # in days (MJD)
    t += t0_in

    xS_bagle = pspl.get_astrometry_unlensed(t) * 1e3 * u.mas  # in mas
    xL_bagle = pspl.get_lens_astrometry(t) * 1e3 * u.mas  # in mas

    ##########
    # Astropy
    ##########
    # Now make an Astropy coordinate for the same source object
    # with the same position, proper motion, distance and calculate
    # the source trajectory vs. time.
    t_obj = Time(t, format='mjd')

    #### REF
    # First we need a reference point at the RA and Dec that will
    # serve as the origin point for our relative coordinate system.
    if verbose: print(f'c0 coords = {raL_in:.10f}, {decL_in:.10f}')
    c0 = SkyCoord(raL_in * u.deg, decL_in * u.deg,
                  pm_ra_cosdec=0.0 * u.mas / u.yr,
                  pm_dec=0.0 * u.mas / u.yr,
                  distance=1e6 * u.Mpc,
                  obstime=Time(t0_in, format='mjd'))

    # Propogate the motion (velocity and parallax) to the correct times).
    x0_astropy_icrs = c0.apply_space_motion(new_obstime=t_obj)
    x0_astropy_gcrs = x0_astropy_icrs.transform_to('gcrs')
    x0_apy_ra = x0_astropy_gcrs.ra
    x0_apy_dec = x0_astropy_gcrs.dec
    cosd = np.cos(x0_apy_dec.to('radian'))

    #### SOURCE
    # Define the source coordinates (with the correct proper motion and distance).
    raS0 = (raL_in * u.deg) + (pspl.xS0[0] * u.deg / 3600)
    decS0 = (decL_in * u.deg) + (pspl.xS0[1] * u.deg / 3600)
    if verbose: print(f'S0 coords = {raS0:.10f}, {decS0:.10f}')
    cS = SkyCoord(raS0, decS0,
                  pm_ra_cosdec=muS_in[0] * u.mas / u.yr,
                  pm_dec=muS_in[1] * u.mas / u.yr,
                  distance=dS_in * u.pc,
                  obstime=Time(t0_in, format='mjd'))

    # Propogate the motion (velocity and parallax) to the correct times).
    xS_astropy_icrs = cS.apply_space_motion(new_obstime=t_obj)
    xS_astropy_gcrs = xS_astropy_icrs.transform_to('gcrs')
    xS_apy_ra = xS_astropy_gcrs.ra
    xS_apy_dec = xS_astropy_gcrs.dec

    dxS_apy = (xS_apy_ra - x0_apy_ra).to('mas') * cosd
    dyS_apy = (xS_apy_dec - x0_apy_dec).to('mas')

    xS_astropy = np.vstack([dxS_apy.value, dyS_apy.value]).T * u.mas

    #### LENS
    raL0 = (raL_in * u.deg) + (pspl.xL0[0] * u.deg / 3600)
    decL0 = (decL_in * u.deg) + (pspl.xL0[1] * u.deg / 3600)
    if verbose: print(f'L0 coords = {raL0:.10f}, {decL0:.10f}')
    
    cL = SkyCoord(raL0, decL0,
                  pm_ra_cosdec=muL_in[0] * u.mas / u.yr,
                  pm_dec=muL_in[1] * u.mas / u.yr,
                  distance=dL_in * u.pc,
                  obstime=Time(t0_in, format='mjd'))

    xL_astropy_icrs = cL.apply_space_motion(new_obstime=t_obj)
    xL_astropy_gcrs = xL_astropy_icrs.transform_to('gcrs')
    xL_apy_ra = xL_astropy_gcrs.ra
    xL_apy_dec = xL_astropy_gcrs.dec
    if verbose:
        print(f'xL_apy_ra  = {xL_apy_ra[0]}, {xL_apy_ra[-1]}')
        print(f'xL_apy_dec = {xL_apy_dec[0]}, {xL_apy_dec[-1]}')
        print(f'x0_apy_ra  = {x0_apy_ra[0]}, {x0_apy_ra[-1]}')
        print(f'x0_apy_dec = {x0_apy_dec[0]}, {x0_apy_dec[-1]}')

    assert xL_apy_ra[0].value   == pytest.approx(259.50146543, rel=1e-7)
    assert xL_apy_ra[-1].value  == pytest.approx(259.50084555, rel=1e-7)
    assert xL_apy_dec[0].value  == pytest.approx(-29.00065871, rel=1e-7)
    assert xL_apy_dec[-1].value == pytest.approx(-28.99947064, rel=1e-7)
    assert x0_apy_ra[0].value   == pytest.approx(259.50146348, rel=1e-7)
    assert x0_apy_ra[-1].value  == pytest.approx(259.50084750, rel=1e-7)
    assert x0_apy_dec[0].value  == pytest.approx(-29.00065816, rel=1e-7)
    assert x0_apy_dec[-1].value == pytest.approx(-28.99947008, rel=1e-7)

    dxL_apy = (xL_apy_ra - x0_apy_ra).to('mas') * cosd
    dyL_apy = (xL_apy_dec - x0_apy_dec).to('mas')

    xL_astropy = np.vstack([dxL_apy.value, dyL_apy.value]).T * u.mas

    ##########
    # Casey's conversion.
    ##########
    if verbose: print('!!! frame_convert.py conversions')
    foo = frame_convert.convert_helio_geo_phot(raL_in, decL_in, t0_in, pspl.u0_amp, pspl.tE,
                                               pspl.piE[0], pspl.piE[1], t0_in, plot=plot)
    t0_geo_casey = foo[0]
    u0_geo_casey = foo[1]
    tE_geo_casey = foo[2]
    piEE_geo_casey = foo[3]
    piEN_geo_casey = foo[4]

    ##########
    # Gould 2004 version:
    ##########
    #
    # Set the ephemeris
    t0_obj = Time(t0_in, format='mjd')

    solar_system_ephemeris.set('builtin')

    # Get the position and velocity of the Earth in the barycentric frame.
    Earth_t0 = get_body_barycentric_posvel('Earth', t0_obj)
    Earth_t = get_body_barycentric_posvel('Earth', t_obj)

    Earth_pos_t0 = Earth_t0[0].get_xyz()
    Earth_vel_t0 = Earth_t0[1].get_xyz()
    Earth_pos_t = Earth_t[0].get_xyz()
    Earth_vel_t = Earth_t[1].get_xyz()

    # This is the position of the Sun w.r.t. to the Earth at time t0 (in geocentric frame)
    Sun_pos_t0 = -Earth_pos_t0
    Sun_vel_t0 = -Earth_vel_t0
    Sun_pos_t = -Earth_pos_t
    Sun_vel_t = -Earth_vel_t

    # Calculate the 3D delta-s(t) vector as defined in Gould 2004 (in IRCS rectilinear):
    #
    #    ds(t) = s(t) - v_earth(t0) * (t - t0) - s(t0)
    #
    ds_gould_3d = Sun_pos_t - (Sun_vel_t0 * (t_obj - t0_obj)[:, None]).T
    ds_gould_3d -= Sun_pos_t0[:, None]

    # Project onto East and North -- identical code to PyLIMA
    target_angles_in_the_sky = [raL_in * np.pi / 180, decL_in * np.pi / 180]
    Target = np.array(
        [np.cos(target_angles_in_the_sky[1]) * np.cos(target_angles_in_the_sky[0]),
         np.cos(target_angles_in_the_sky[1]) * np.sin(target_angles_in_the_sky[0]),
         np.sin(target_angles_in_the_sky[1])])

    East = np.array([-np.sin(target_angles_in_the_sky[0]),
                     np.cos(target_angles_in_the_sky[0]),
                     0.0])
    North = np.cross(Target, East)

    Sun_pos_t0_EN = np.array([np.dot(Sun_pos_t0.value, East),
                              np.dot(Sun_pos_t0.value, North)]) * u.AU
    Sun_vel_t0_EN = np.array([np.dot(Sun_vel_t0.value, East),
                              np.dot(Sun_vel_t0.value, North)]) * u.AU / u.day
    Sun_pos_t_EN = np.zeros([len(t), 2], dtype=float)
    Sun_vel_t_EN = np.zeros([len(t), 2], dtype=float)
    ds_gould_2d = np.zeros([len(t), 2], dtype=float)
    for tt in range(len(t)):
        # Note, positions still in AU, velocities in AU/day.
        Sun_pos_t_EN[tt] = np.array([np.dot(Sun_pos_t.value[:, tt], East),
                                     np.dot(Sun_pos_t.value[:, tt], North)])
        Sun_vel_t_EN[tt] = np.array([np.dot(Sun_vel_t.value[:, tt], East),
                                     np.dot(Sun_vel_t.value[:, tt], North)])
        ds_gould_2d[tt] = np.array([np.dot(ds_gould_3d.value[:, tt], East),
                                    np.dot(ds_gould_3d.value[:, tt], North)])

    Sun_pos_t_EN *= u.AU
    Sun_vel_t_EN *= u.AU / u.day
    ds_gould_2d *= u.AU

    # ds_gould_2d  = Sun_pos_t_EN - (Sun_vel_t0_EN *(t_obj.value - t0_obj.value)[:, None])
    # ds_gould_2d -= Sun_pos_t0_EN[:, None].T

    # Calculate d-tau and d-beta
    dtau = ds_gould_2d[:, 0] / u.AU * pspl.piE[0] + ds_gould_2d[:, 1] / u.AU * pspl.piE[1]
    dbeta = ds_gould_2d[:, 0] / u.AU * pspl.piE[1] - ds_gould_2d[:, 1] / u.AU * pspl.piE[0]

    # dtau = np.dot(pspl.piE, ds_gould_2d)
    # dbeta = np.cross(pspl.piE, ds_gould_2d.T)

    # Need to convert to t0_geo at tr... pick a reference time (arbitrarily choose t0):
    A = (pspl.muRel[0] / pspl.thetaE_amp) / u.yr - (pspl.piE_amp * Sun_vel_t0_EN[0] / u.AU)
    B = (pspl.muRel[1] / pspl.thetaE_amp) / u.yr - (pspl.piE_amp * Sun_vel_t0_EN[1] / u.AU)
    t0_obj_geotr = (-1 / (A ** 2 + B ** 2)) * (A * pspl.u0[0]
                                               + B * pspl.u0[1]
                                               - A * pspl.piE_amp * Sun_pos_t0_EN[0] / u.AU
                                               - B * pspl.piE_amp * Sun_pos_t0_EN[1] / u.AU)
    t0_obj_geotr += t0_obj

    # Need to convert to u0_geo at tr
    # u0_geotr = pspl.u0
    u0_geotr = (((pspl.muRel / pspl.thetaE_amp) / u.yr) * (t0_obj_geotr.value - t0_obj.value) * u.day).to('')
    u0_geotr -= (pspl.piE_amp * Sun_pos_t0_EN / u.AU)
    u0_geotr += pspl.u0

    # Need to convert u0 amplitude.
    u0_amp_geotr = np.linalg.norm(u0_geotr)

    # Need to convert to tE_geo at tr
    tE_geotr = (pspl.tE * u.day * np.linalg.norm(pspl.muRel) * u.mas / u.yr) / (
                np.hypot(A, B) * pspl.thetaE_amp * u.mas)

    # Just for comparison, lets also calculate piE and muRel in both frames.
    muRel_geotr = (pspl.muRel * u.mas / u.yr) - (pspl.piE_amp * pspl.thetaE_amp * u.mas * Sun_vel_t0_EN / u.AU)
    piE_geotr = pspl.piE_amp * muRel_geotr / np.linalg.norm(muRel_geotr)

    if verbose:
        print(f'pspl.muRel = {pspl.muRel}')
        print(f'pspl.thetaE_amp = {pspl.thetaE_amp}')
        print(f'A = {A}')
        print(f'B = {B}')
        print(f'Sun vel at t0 (East) = {Sun_vel_t0_EN[0]}')
    
        print(f'u0E:    pspl = {pspl.u0[0]:.3f},  geotr = {u0_geotr[0]:.3f}')
        print(f'u0N:    pspl = {pspl.u0[1]:.3f},  geotr = {u0_geotr[1]:.3f}')
        print(f'u0_amp: pspl = {pspl.u0_amp:.3f},  geotr = {u0_amp_geotr:.3f}, geotr_c = {u0_geo_casey:.3f}')
        print(f't0:     pspl = {t0_obj.value:.2f},  geotr = {t0_obj_geotr.value:.2f},  geotr_c = {t0_geo_casey:.2f}')
        print(f'tE:     pspl = {pspl.tE:.3f},  geotr = {tE_geotr:.3f},  geotr_c = {tE_geo_casey:.3f}')
        print(f'muRelE: pspl = {pspl.muRel[0]:.3f},  geotr = {muRel_geotr[0]:.3f}')
        print(f'muRelN: pspl = {pspl.muRel[1]:.3f},  geotr = {muRel_geotr[1]:.3f}')
        print(f'piEE:   pspl = {pspl.piE[0]:.4f},  geotr = {piE_geotr[0]:.4f},  geotr_c = {piEE_geo_casey:.4f}')
        print(f'piEN:   pspl = {pspl.piE[1]:.4f},  geotr = {piE_geotr[1]:.4f},  geotr_c = {piEN_geo_casey:.4f}')

    assert pspl.u0[0]  == pytest.approx(0.000, rel=1e-3)
    assert pspl.u0[1]  == pytest.approx(0.627, rel=1e-3)
    assert t0_obj.value == pytest.approx(57000.00, rel=1e-1)
    assert t0_geo_casey == pytest.approx(57001.49, rel=1e-1)

    # Calculate tau (in relative proper motion direction) and beta (in u0 direction)
    tau = ((t_obj.value - t0_obj_geotr.value) * u.day / tE_geotr) + dtau
    beta = u0_amp_geotr + dbeta
    
    tau_vec = np.outer(tau, muRel_geotr.T) / np.linalg.norm(muRel_geotr)
    beta_vec = np.outer(beta, u0_geotr.T) / np.linalg.norm(u0_geotr)

    if verbose:
        print('t      = ', t[0:500:80])
        print('tau    = ', tau[0:500:80])
        print('dtau   = ', dtau[0:500:80])
        print('beta   = ', beta[0:500:80])
        print('dbeta  = ', dbeta[0:500:80])

    u_bagel = (xS_bagle - xL_bagle) / (pspl.thetaE_amp * u.mas)
    u_astropy = (xS_astropy - xL_astropy) / (pspl.thetaE_amp * u.mas)
    u_gould = tau_vec + beta_vec

    xL_gould = xL_bagle
    xS_gould = (u_gould * pspl.thetaE_amp * u.mas) + xL_gould

    # Position of source w.r.t. lens in Gould frame.
    if verbose:
        print('t = ', t[0:500:80])
        print('xL_bagle   = ', xL_bagle[0:500:80])
        print('xL_astropy = ', xL_astropy[0:500:80])
        print('xL_gould   = ', xL_gould[0:500:80])
        print('xS_bagle   = ', xS_bagle[0:500:80])
        print('xS_astropy = ', xS_astropy[0:500:80])
        print('xS_gould   = ', xS_gould[0:500:80])
        print('lens pos (mas), vel (mas/yr) = ', pspl.xL0 * 1e3, pspl.muL)
        print('sorc pos (mas), vel (mas/yr) = ', pspl.xS0 * 1e3, pspl.muS)

    assert xL_bagle[0, 0].value == pytest.approx(xL_astropy[0, 0].value, 1e-2)
    assert xL_bagle[0, 1].value == pytest.approx(xL_astropy[0, 1].value, 1e-2)
    assert xS_bagle[0, 0].value == pytest.approx(xS_astropy[0, 0].value, 1e-2)
    assert xS_bagle[0, 1].value == pytest.approx(xS_astropy[0, 1].value, 1e-2)

    # Calculate the residuals.
    resid = xS_astropy - xS_bagle  # mas

    if plot:
        plt.figure(1, figsize=(10, 3))
        plt.subplots_adjust(wspace=0.7)
        plt.subplot(1, 3, 1)
        plt.plot(t, xS_bagle[:, 0], color='red', label='Our code')
        plt.plot(t, xS_astropy[:, 0], linestyle='--', color='black', label='Astropy')
        plt.plot(t, xS_gould[:, 0], linestyle='-.', color='blue', label='Gould')
        plt.xlabel('Time (MJD)')
        plt.ylabel(r'$\Delta\alpha^*$ (mas)')
        plt.legend(fontsize=10)

        plt.subplot(1, 3, 2)
        plt.plot(t, xS_bagle[:, 1], color='red')
        plt.plot(t, xS_astropy[:, 1], linestyle='--', color='black')
        plt.plot(t, xS_gould[:, 1], linestyle='-.', color='blue')
        plt.xlabel('Time (MJD)')
        plt.ylabel(r'$\Delta\delta$ (mas)')

        plt.subplot(1, 3, 3)
        plt.plot(xS_bagle[:, 0], xS_bagle[:, 1], color='red', label='Our code')
        plt.plot(xS_astropy[:, 0], xS_astropy[:, 1], linestyle='-.',
                 color='black', label='Astropy')
        plt.xlabel(r'$\Delta\alpha^*$ (mas)')
        plt.ylabel(r'$\Delta\delta$ (mas)')
        plt.legend(fontsize=10)
        plt.axis('equal')

        plt.figure(2, figsize=(10, 3))
        plt.subplots_adjust(wspace=0.7)
        plt.subplot(1, 3, 1)
        plt.plot(t, u_bagel[:, 0], color='red')
        plt.plot(t, u_astropy[:, 0], linestyle='--', color='black')
        plt.plot(t, u_gould[:, 0], linestyle='-.', color='blue')
        plt.xlabel('Time (MJD)')
        plt.ylabel(r'$\Delta\alpha^*$ ($\theta_E$)')

        plt.subplot(1, 3, 2)
        plt.plot(t, u_bagel[:, 1], color='red')
        plt.plot(t, u_astropy[:, 1], linestyle='--', color='black')
        plt.plot(t, u_gould[:, 1], linestyle='-.', color='blue')
        plt.xlabel('Time (MJD)')
        plt.ylabel(r'$\Delta\delta$ ($\theta_E$)')

        plt.subplot(1, 3, 3)
        plt.plot(u_bagel[:, 0], u_bagel[:, 1], color='red', label='Our code')
        plt.plot(u_astropy[:, 0], u_astropy[:, 1], linestyle='-.',
                 color='black', label='Astropy')
        plt.xlabel(r'$\Delta\alpha^*$ ($\theta_E$)')
        plt.ylabel(r'$\Delta\delta$ ($\theta_E$)')
        plt.legend(fontsize=10)
        plt.axis('equal')

        plt.figure()
        plt.plot(t, resid[:, 0], 'b--', label=r'$\Delta\alpha^*$ diff')
        plt.plot(t, resid[:, 1], 'r--', label=r'$\Delta\delta$ diff')
        plt.xlabel('Time (MJD)')
        plt.ylabel('Residuals (mas)')
        plt.legend(fontsize=10)

    return


def example_astropy_parallax():
    from astropy.coordinates import SkyCoord, GCRS
    from astropy.time import Time
    import astropy.units as u

    ra = 17.5 * 15.0 * u.deg
    dec = -29 * u.deg
    dist = 8000.0 * u.pc
    sc = SkyCoord(ra, dec,
                  pm_ra_cosdec=0.0 * u.mas / u.yr,
                  pm_dec=0.0 * u.mas / u.yr,
                  distance=dist,
                  obstime=Time(57000.0, format='mjd'))

    sc0 = SkyCoord(ra, dec,
                   pm_ra_cosdec=0.0 * u.mas / u.yr,
                   pm_dec=0.0 * u.mas / u.yr,
                   distance=1e6 * u.Mpc,
                   obstime=Time(57000.0, format='mjd'))

    t = np.arange(56000, 58000)
    t_obj = Time(t, format='mjd')

    sc_t_icrs = sc.apply_space_motion(new_obstime=t_obj)
    sc_t_gcrs = sc_t_icrs.transform_to('gcrs')

    sc_t_icrs0 = sc0.apply_space_motion(new_obstime=t_obj)
    sc_t_gcrs0 = sc_t_icrs0.transform_to('gcrs')

    ra_t = sc_t_gcrs.ra
    dec_t = sc_t_gcrs.dec
    cosd_t = np.cos(dec_t.to('radian'))

    ra0_t = sc_t_gcrs0.ra
    dec0_t = sc_t_gcrs0.dec

    #     dra = ((ra_t - ra) * cosd_t).to('arcsec')  # in arcsec
    #     ddec = (dec_t - dec).to('arcsec')          # in arcsec
    dra = ((ra_t - ra0_t) * cosd_t).to('arcsec')  # in arcsec
    ddec = (dec_t - dec0_t).to('arcsec')  # in arcsec

    plt.figure(1, figsize=(10, 3))
    plt.subplots_adjust(wspace=0.7)

    parallax_pred = 1.0 / dist.value
    parallax_meas = np.max(np.hypot(dra.value, ddec.value))
    print('Predicted parallax from manual calculation:')
    print('    {0:.2f} mas'.format(parallax_pred * 1e3))
    print('Total parallax from astrpy calculation:')
    print('    {0:.2f} mas'.format(parallax_meas * 1e3))

    plt.subplot(1, 3, 1)
    plt.plot(t, dra.to('mas'), color='black')
    plt.xlabel('Time (MJD)')
    plt.ylabel(r'$\Delta \alpha^*$ (mas)')
    plt.axhline(parallax_pred * 1e3, color='red', linestyle='--')

    plt.subplot(1, 3, 2)
    plt.plot(t, ddec.to('mas'), color='black')
    plt.xlabel('Time (MJD)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.title('Star Distance = ' + str(dist))

    plt.subplot(1, 3, 3)
    plt.plot(dra.to('mas'), ddec.to('mas'), color='black')
    plt.xlabel(r'$\Delta \alpha^*$ (mas)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.axis('equal')

    return


def plot_PSBL(psbl, t_obs):
    """
    Make some standard plots for PSBL.
    """
    images, amps = psbl.get_all_arrays(t_obs)

    ##########
    # Photometry
    ##########
    phot = psbl.get_photometry(t_obs, amp_arr=amps)

    # Plot the photometry
    plt.figure(1)
    plt.clf()
    plt.plot(t_obs, phot, 'r-')
    plt.ylabel('Photometry (mag)')
    plt.xlabel('Time (MJD)')
    plt.gca().invert_yaxis()

    ##########
    # Astrometry
    ##########
    if psbl.astrometryFlag:
        # Find the points closest to t0
        t0idx = np.argmin(np.abs(t_obs - psbl.t0))

        xL1, xL2 = psbl.get_resolved_lens_astrometry(t_obs)
        xL1 *= 1e3
        xL2 *= 1e3
        xS_unlens = psbl.get_astrometry_unlensed(t_obs) * 1e3
        xS_lensed = psbl.get_astrometry(t_obs, image_arr=images, amp_arr=amps) * 1e3

        dxS = (xS_lensed - xS_unlens)

        # Plot the positions of everything
        plt.figure(2)
        plt.clf()
        plt.plot(xS_unlens[:, 0], xS_unlens[:, 1], 'b--', mfc='blue',
                 mec='blue')
        plt.plot(xS_lensed[:, 0], xS_lensed[:, 1], 'b-')
        plt.plot(xL1[:, 0], xL1[:, 1], 'g--', mfc='none',
                 mec='green')
        plt.plot(xL2[:, 0], xL2[:, 1], 'g--', mfc='none',
                 mec='dark green')

        plt.plot(xS_unlens[t0idx, 0], xS_unlens[t0idx, 1], 'bx', mfc='blue',
                 mec='blue',
                 label='xS, unlensed')
        plt.plot(xS_lensed[t0idx, 0], xS_lensed[t0idx, 1], 'bo',
                 label='xS, lensed')
        plt.plot(xL1[t0idx, 0], xL1[t0idx, 1], 'gs', mfc='green',
                 mec='green',
                 label='Primary lens')
        plt.plot(xL2[t0idx, 0], xL2[t0idx, 1], 'gs', mfc='none',
                 mec='green',
                 label='Secondary lens')

        plt.legend()
        plt.gca().invert_xaxis()
        plt.xlabel('R.A. (mas)')
        plt.ylabel('Dec. (mas)')

        # Check just the astrometric shift part.
        plt.figure(3)
        plt.clf()
        plt.plot(t_obs, dxS[:, 0], 'r--', label='R.A.')
        plt.plot(t_obs, dxS[:, 1], 'b--', label='Dec.')
        plt.legend(fontsize=10)
        plt.ylabel('Astrometric Shift (mas)')
        plt.xlabel('Time (MJD)')

        plt.figure(4)
        plt.clf()
        plt.plot(dxS[:, 0], dxS[:, 1], 'r-')
        plt.axhline(0, linestyle='--')
        plt.axvline(0, linestyle='--')
        plt.gca().invert_xaxis()
        plt.xlabel('Shift RA (mas)')
        plt.ylabel('Shift Dec (mas)')
        plt.axis('equal')

        print('Einstein radius: ', psbl.thetaE_amp)
        print('Einstein crossing time: ', psbl.tE)

    return


def plot_PSBL_compare(psbl1, label1, psbl2, label2, t_obs):
    """
    Make some standard plots for PSBL.
    """
    images1, amps1 = psbl1.get_all_arrays(t_obs)
    images2, amps2 = psbl2.get_all_arrays(t_obs)

    ##########
    # Photometry
    ##########
    phot1 = psbl1.get_photometry(t_obs, amp_arr=amps1)
    phot2 = psbl2.get_photometry(t_obs, amp_arr=amps2)

    # Plot the photometry
    plt.figure(1)
    plt.clf()
    plt.plot(t_obs, phot1, 'r-', label=label1)
    plt.plot(t_obs, phot2, 'b-', label=label2)
    plt.legend()
    plt.ylabel('Photometry (mag)')
    plt.xlabel('Time (MJD)')
    plt.gca().invert_yaxis()

    ##########
    # Astrometry
    ##########
    if psbl1.astrometryFlag:
        # Find the points closest to t0
        t0idx1 = np.argmin(np.abs(t_obs - psbl1.t0))
        t0idx2 = np.argmin(np.abs(t_obs - psbl2.t0))

        xL1_1, xL2_1 = psbl1.get_resolved_lens_astrometry(t_obs)
        xL1_1 *= 1e3
        xL2_1 *= 1e3
        xL1_2, xL2_2 = psbl2.get_resolved_lens_astrometry(t_obs)
        xL1_2 *= 1e3
        xL2_2 *= 1e3

        xS_unlens1 = psbl1.get_astrometry_unlensed(t_obs) * 1e3
        xS_unlens2 = psbl2.get_astrometry_unlensed(t_obs) * 1e3
        xS_lensed1 = psbl1.get_astrometry(t_obs, image_arr=images1, amp_arr=amps1) * 1e3
        xS_lensed2 = psbl2.get_astrometry(t_obs, image_arr=images2, amp_arr=amps2) * 1e3

        dxS1 = (xS_lensed1 - xS_unlens1)
        dxS2 = (xS_lensed2 - xS_unlens2)

        # Plot the positions of everything
        plt.figure(2)
        plt.clf()
        plt.plot(xS_unlens1[:, 0], xS_unlens1[:, 1],
                 'b--', mfc='blue', mec='blue')
        plt.plot(xS_unlens2[:, 0], xS_unlens2[:, 1],
                 'b--', mfc='blue', mec='blue', alpha=0.2)

        plt.plot(xS_lensed1[:, 0], xS_lensed1[:, 1], 'b-')
        plt.plot(xS_lensed2[:, 0], xS_lensed2[:, 1], 'b-', alpha=0.2)

        plt.plot(xL1_1[:, 0], xL1_1[:, 1], 'g--', mfc='none', mec='green')
        plt.plot(xL1_2[:, 0], xL1_2[:, 1], 'g--', mfc='none', mec='green', alpha=0.2)
        plt.plot(xL2_1[:, 0], xL2_1[:, 1], 'g--', mfc='none', mec='dark green')
        plt.plot(xL2_2[:, 0], xL2_2[:, 1], 'g--', mfc='none', mec='dark green', alpha=0.2)

        # Plot closest approach points.
        plt.plot(xS_unlens1[t0idx1, 0], xS_unlens1[t0idx1, 1],
                 'bx', mfc='blue', mec='blue',
                 label='xS, unlensed, ' + label1)
        plt.plot(xS_unlens2[t0idx2, 0], xS_unlens2[t0idx2, 1],
                 'bx', mfc='blue', mec='blue', alpha=0.2,
                 label='xS, unlensed, ' + label2)
        plt.plot(xS_lensed1[t0idx1, 0], xS_lensed1[t0idx1, 1],
                 'bo',
                 label='xS, lensed, ' + label1)
        plt.plot(xS_lensed2[t0idx2, 0], xS_lensed2[t0idx2, 1],
                 'bo', alpha=0.2,
                 label='xS, lensed, ' + label2)
        plt.plot(xL1_1[t0idx1, 0], xL1_1[t0idx1, 1],
                 'gs', mfc='green', mec='green',
                 label='Primary lens, ' + label1)
        plt.plot(xL1_2[t0idx2, 0], xL1_2[t0idx2, 1],
                 'gs', mfc='green', mec='green', alpha=0.2,
                 label='Primary lens, ' + label2)
        plt.plot(xL2_1[t0idx1, 0], xL2_1[t0idx1, 1],
                 'gs', mfc='none', mec='green',
                 label='Secondary lens, ' + label1)
        plt.plot(xL2_2[t0idx2, 0], xL2_2[t0idx2, 1],
                 'gs', mfc='none', mec='green', alpha=0.2,
                 label='Secondary lens, ' + label2)

        plt.legend(fontsize=10)
        plt.gca().invert_xaxis()
        plt.xlabel('R.A. (mas)')
        plt.ylabel('Dec. (mas)')

        # Check just the astrometric shift part.
        plt.figure(3)
        plt.clf()
        plt.plot(t_obs, dxS1[:, 0], 'r--', label='R.A., ' + label1)
        plt.plot(t_obs, dxS1[:, 1], 'r-.', label='Dec., ' + label1)
        plt.plot(t_obs, dxS2[:, 0], 'b--', label='R.A., ' + label2, alpha=0.2)
        plt.plot(t_obs, dxS2[:, 1], 'b-.', label='Dec., ' + label2, alpha=0.2)
        plt.legend(fontsize=10)
        plt.ylabel('Astrometric Shift (mas)')
        plt.xlabel('Time (MJD)')

        plt.figure(4)
        plt.clf()
        plt.plot(dxS1[:, 0], dxS1[:, 1], 'r-', label=label1)
        plt.plot(dxS2[:, 0], dxS2[:, 1], 'b-', label=label2, alpha=0.2)
        plt.axhline(0, linestyle='--')
        plt.axvline(0, linestyle='--')
        plt.legend()
        plt.gca().invert_xaxis()
        plt.xlabel('Shift RA (mas)')
        plt.ylabel('Shift Dec (mas)')
        plt.axis('equal')

        print('Einstein radius: ', psbl1.thetaE_amp, psbl2.thetaE_amp)

    # Print some common stuff
    print('tE: ', psbl1.tE, psbl2.tE)
    print('u0_amp: ', psbl1.u0_amp, psbl2.u0_amp)

    return


def test_PSBL_PhotAstrom_noPar_Param2(plot=False):
    """
    General testing of PSBL... caustic crossings.
    """

    raL = 259.5
    decL = -29.0
    t0 = 57000
    u0 = 0.3  # in units of Einstein radii
    tE = 200.0
    piE_E = 0.01
    piE_N = -0.01
    b_sff = np.array([1.0])
    mag_src = np.array([18])
    thetaE = 3.0  # in mas
    xS0_E = 0.0
    xS0_N = 0.01
    muS_E = 3.0
    muS_N = 0.0
    piS = (1.0 / 8000.0) * 1e3  # mas
    q = 0.8  # M2 / M1
    sep = 3.0  # mas
    alpha = 135.0

    psbl = model.PSBL_PhotAstrom_noPar_Param2(t0, u0, tE,
                                              thetaE, piS,
                                              piE_E, piE_N,
                                              xS0_E, xS0_N,
                                              muS_E, muS_N,
                                              q, sep, alpha,
                                              b_sff, mag_src,
                                              raL=raL, decL=decL,
                                              root_tol=1e-4)

    t_obs = np.arange(56000.0, 58000.0, 3)

    if plot:
        plot_PSBL(psbl, t_obs)

    # Check that we have some extreme magnifications since this
    # is caustic crossing.
    phot = psbl.get_photometry(t_obs)

    assert phot.min() < 16

    return


def test_PSBL_Phot_noPar_Param1(plot=False):
    """
    General testing of PSBL... caustic crossings.
    """

    # NOTE this gives the same model as in test_PSBL_Phot_noPar_Param1()
    raL = 259.5
    decL = -29.0
    t0 = 57000
    u0 = 0.3  # in units of Einstein radii
    tE = 200.0
    piE_E = 0.01
    piE_N = -0.01
    b_sff = np.array([1.0])
    mag_src = np.array([18])
    q = 0.8  # M2 / M1
    # sep = 3e-3 # in arcsec
    sep = 1.0  # in Einstein radii
    alpha = 135.0  # PA of binary on the sky
    phi_piE = np.degrees(np.arctan2(piE_N, piE_E))  # PA of muRel on the sky
    phi = alpha - phi_piE  # relative angle between binary and muRel.
    print('alpha = ', alpha, ' deg')
    print('phi_piE = ', phi_piE, ' deg')
    print('phi = ', phi, ' deg')

    psbl = model.PSBL_Phot_noPar_Param1(t0, u0, tE,
                                        piE_E, piE_N,
                                        q, sep, phi,
                                        b_sff, mag_src,
                                        raL=raL, decL=decL,
                                        root_tol=1e-4)

    t_obs = np.arange(56000.0, 58000.0, 3)

    if plot:
        plot_PSBL(psbl, t_obs)

    # Check that we have some extreme magnifications since this
    # is caustic crossing.
    phot = psbl.get_photometry(t_obs)

    assert phot.max() > 15

    return


def test_PSBL_PhotAstrom_Par_Param2(plot=False, verbose=False):
    """
    General testing of PSBL... caustic crossings.
    """

    raL = 259.5
    decL = -29.0
    t0 = 57000
    u0 = 0.3  # in units of Einstein radii
    tE = 200.0
    piE_E = 0.01
    piE_N = -0.01
    b_sff = np.array([1.0])
    mag_src = np.array([18])
    thetaE = 3.0  # in mas
    xS0_E = 0.0
    xS0_N = 0.01
    muS_E = 3.0
    muS_N = 0.0
    piS = (1.0 / 8000.0) * 1e3  # mas
    q = 0.8  # M2 / M1
    sep = 3.0  # mas
    alpha = 135.0

    psbl_n = model.PSBL_PhotAstrom_noPar_Param2(t0, u0, tE,
                                                thetaE, piS,
                                                piE_E, piE_N,
                                                xS0_E, xS0_N,
                                                muS_E, muS_N,
                                                q, sep, alpha,
                                                b_sff, mag_src,
                                                raL=raL, decL=decL,
                                                root_tol=1e-4)

    psbl_p = model.PSBL_PhotAstrom_Par_Param2(t0, u0, tE,
                                              thetaE, piS,
                                              piE_E, piE_N,
                                              xS0_E, xS0_N,
                                              muS_E, muS_N,
                                              q, sep, alpha,
                                              b_sff, mag_src,
                                              raL=raL, decL=decL,
                                              root_tol=1e-4)

    t_obs = np.arange(56000.0, 58000.0, 3)

    if plot:
        plot_PSBL_compare(psbl_n, 'No Parallax', psbl_p, 'Parallax', t_obs)

    # Check that we have some extreme magnifications since this
    # is caustic crossing.
    phot1 = psbl_n.get_photometry(t_obs)
    phot2 = psbl_p.get_photometry(t_obs)

    assert phot1.min() < 16
    assert phot2.min() < 16

    if verbose:
        print('Sep (in thetaE, no par): ', psbl_n.sep / psbl_n.thetaE_amp)
        print('Sep (in thetaE, with par): ', psbl_p.sep / psbl_p.thetaE_amp)
        print('m1 (in thetaE**2, not mass): ', psbl_n.m1 / (psbl_n.thetaE_amp * 1e-3) ** 2,
              psbl_p.m1 / (psbl_p.thetaE_amp * 1e-3) ** 2)
        print('m2 (in thetaE**2, not mass): ', psbl_n.m2 / (psbl_n.thetaE_amp * 1e-3) ** 2,
              psbl_p.m2 / (psbl_p.thetaE_amp * 1e-3) ** 2)

    assert psbl_n.m1 == pytest.approx(psbl_p.m1)

    ##########
    # Recalculate u calculation from complex_pos() to debug.
    ##########
    # Calculate the position of the source w.r.t. lens (in Einstein radii)
    # Distance along muRel direction
    # tau = (t_obs - psbl_p.t0) / psbl_p.tE
    # tau = tau.reshape(len(tau), 1)

    # # Distance along u0 direction -- always constant with time.
    # u0 = psbl_p.u0.reshape(1, len(psbl_p.u0))
    # thetaE_hat = psbl_p.thetaE_hat.reshape(1, len(psbl_p.thetaE_hat))

    # # Total distance
    # u = u0 + tau * thetaE_hat

    # # Incorporate parallax
    # parallax_vec = model.parallax_in_direction(psbl_p.raL, psbl_p.decL, t_obs)
    # u -= psbl_p.piE_amp * parallax_vec

    # t0dx = np.argmin(np.abs(tau))
    # print('u = ')
    # print(u[t0dx - 5:t0dx + 5, :])

    # w, z1, z2 = psbl_p.get_complex_pos(t_obs)
    # comp = psbl_p.get_complex_pos(t_obs)
    # images_p, amps_p = psbl_p.get_all_arrays(t_obs)
    # amp_arr_msk = np.ma.masked_invalid(amps_p)
    # amp = np.sum(amp_arr_msk, axis=1)

    # # print(images_p[t0dx-5:t0dx+5])
    # # print(amps_p[t0dx-5:t0dx+5])

    # # Get the astrometry in the lens rest frame in units of thetaE
    # xL = psbl_p.get_lens_astrometry(t_obs)  # in arcsec
    # xL1, xL2 = psbl_p.get_resolved_lens_astrometry(t_obs)  # in arcsec
    # xS_u = psbl_p.get_astrometry_unlensed(t_obs)  # in arcsec
    # u2 = (xS_u - xL) / (psbl_p.thetaE_amp * 1e-3)  # -- this should basically be u

    # w_new = u2
    # z1_new = (xL1 - xL) / (psbl_p.thetaE_amp * 1e-3)
    # z2_new = (xL2 - xL) / (psbl_p.thetaE_amp * 1e-3)

    # print('w: ')
    # print(w[t0dx - 5:t0dx + 5] / (psbl_p.thetaE_amp * 1e-3))
    # print(w_new[t0dx - 5:t0dx + 5])
    # print('z1: ')
    # print(z1[t0dx - 5:t0dx + 5] / (psbl_p.thetaE_amp * 1e-3))
    # print(z1_new[t0dx - 5:t0dx + 5])
    # print('z12 ')
    # print(z2[t0dx - 5:t0dx + 5] / (psbl_p.thetaE_amp * 1e-3))
    # print(z2_new[t0dx - 5:t0dx + 5])

    # print('u2 = ')
    # print(u2[t0dx - 5:t0dx + 5])

    return


def test_PSBL_Phot_Par_Param1(plot=False, verbose=False):
    """
    General testing of PSBL... caustic crossings.
    """

    # NOTE this gives the same model as in test_PSBL_Phot_noPar_Param1()
    raL = 259.5
    decL = -29.0
    t0 = 57000
    u0 = 0.3  # in units of Einstein radii
    tE = 200.0
    piE_E = 0.01
    piE_N = -0.01
    b_sff = np.array([1.0])
    mag_src = np.array([18])
    q = 0.8  # M2 / M1
    sep = 1.0  # in Einstein radii
    alpha = 135.0  # PA of binary on the sky
    phi_piE = np.degrees(np.arctan2(piE_N, piE_E))  # PA of muRel on the sky
    phi = alpha - phi_piE  # relative angle between binary and muRel.
    print('alpha = ', alpha, ' deg')
    print('phi_piE = ', phi_piE, ' deg')
    print('phi = ', phi, ' deg')

    psbl_n = model.PSBL_Phot_noPar_Param1(t0, u0, tE,
                                          piE_E, piE_N,
                                          q, sep, phi,
                                          b_sff, mag_src,
                                          raL=raL, decL=decL,
                                          root_tol=1e-4)
    psbl_p = model.PSBL_Phot_Par_Param1(t0, u0, tE,
                                        piE_E, piE_N,
                                        q, sep, phi,
                                        b_sff, mag_src,
                                        raL=raL, decL=decL,
                                        root_tol=1e-4)

    t_obs = np.arange(56000.0, 58000.0, 3)

    if plot:
        plot_PSBL_compare(psbl_n, 'No Parallax', psbl_p, 'Parallax', t_obs)

    # Check that we have some extreme magnifications since this
    # is caustic crossing.
    phot1 = psbl_n.get_photometry(t_obs)
    phot2 = psbl_p.get_photometry(t_obs)

    assert phot1.min() < 16
    assert phot2.min() < 16

    if verbose:
        print('Sep (in thetaE, no par): ', psbl_n.sep)
        print('Sep (in thetaE, with par): ', psbl_p.sep)
        print('m1 (in thetaE**2, not mass): ', psbl_n.m1, psbl_p.m1)
        print('m2 (in thetaE**2, not mass): ', psbl_n.m2, psbl_p.m2)

    # ##########
    # # Recalculate u calculation from complex_pos() to debug.
    # ##########
    # # Calculate the position of the source w.r.t. lens (in Einstein radii)
    # # Distance along muRel direction
    # tau = (t_obs - psbl_p.t0) / psbl_p.tE
    # tau = tau.reshape(len(tau), 1)

    # # Distance along u0 direction -- always constant with time.
    # u0 = psbl_p.u0.reshape(1, len(psbl_p.u0))
    # thetaE_hat = psbl_p.thetaE_hat.reshape(1, len(psbl_p.thetaE_hat))

    # # Total distance
    # u = u0 + tau * thetaE_hat

    # # Incorporate parallax
    # parallax_vec = model.parallax_in_direction(psbl_p.raL, psbl_p.decL, t_obs)
    # u -= psbl_p.piE_amp * parallax_vec

    # t0dx = np.argmin(np.abs(tau))
    # print('u = ')
    # print(u[t0dx - 5:t0dx + 5, :])

    # w, z1, z2 = psbl_p.get_complex_pos(t_obs)
    # images_p, amps_p = psbl_p.get_all_arrays(t_obs)
    # amp_arr_msk = np.ma.masked_invalid(amps_p)
    # amp = np.sum(amp_arr_msk, axis=1)

    # print('w: ')
    # print(w[t0dx - 5:t0dx + 5])
    # print('z1: ')
    # print(z1[t0dx - 5:t0dx + 5])
    # print('z2: ')
    # print(z2[t0dx - 5:t0dx + 5])

    return


def test_PSBL_phot_vs_pyLIMA(plot=False):
    # Parameters -- common to ours and pyLIMA
    t0 = 55775.0
    u0_amp = 0.5
    tE = 60.0
    mag_src = 16
    b_sff = 0.5
    q = 1.0
    sep = 0.6   # mas
    phi = 125.0 # deg

    tol = 5e-4

    res = plot_compare_vs_pylima(t0, u0_amp, tE, mag_src, b_sff, q, sep, phi,
                                 tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < tol

    # Change t0:
    t0_new = 56000.0
    res = plot_compare_vs_pylima(t0_new, u0_amp, tE, mag_src, b_sff, q, sep, phi,
                                 tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < tol

    # Change u0_amp:
    u0_amp_new = 0.4
    res = plot_compare_vs_pylima(t0_new, u0_amp_new, tE, mag_src, b_sff, q, sep, phi,
                                 tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < tol

    # Change tE:
    tE_new = 120.0
    res = plot_compare_vs_pylima(t0, u0_amp, tE_new, mag_src, b_sff, q, sep, phi,
                                 tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < tol

    # Change sep:
    sep_new = 0.3
    res = plot_compare_vs_pylima(t0, u0_amp, tE, mag_src, b_sff, q, sep_new, phi,
                                 tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < tol

    # Change phi:
    phi_new = 0.3
    res = plot_compare_vs_pylima(t0, u0_amp, tE, mag_src, b_sff, q, sep, phi_new,
                                 tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < tol

    # Change mag_src:
    mag_src_new = 18
    res = plot_compare_vs_pylima(t0, u0_amp, tE, mag_src_new, b_sff, q, sep, phi,
                                 tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < tol

    # Change b_sff:
    b_sff_new = 0.5
    res = plot_compare_vs_pylima(t0, u0_amp, tE, mag_src, b_sff_new, q, sep, phi,
                                 tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < tol

    # Change q
    q_new = 0.8
    res = plot_compare_vs_pylima(t0, u0_amp, tE, mag_src, b_sff, q_new, sep, phi,
                                 tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < tol

    return


def plot_compare_vs_pylima(t0, u0_amp, tE, mag_src, b_sff, q, sep, phi, piEE=0.1, piEN=0.1,
                           tol = 1e-6, plot=False):
    """All input values are our model definitions.
    
    Note piEE and piEN should be arbitrary. But might want to check just in case. 
    """
    from pyLIMA import models as microlmodels
    from pyLIMA import event
    from pyLIMA import telescopes
    from pyLIMA import toolbox as microltoolbox
    from pyLIMA.models import generate_model

    phi_rad = np.radians(phi)

    # These are arbitrary in the Phot-noParallax model.
    piEE = 0.1
    piEN = 0.1

    # Our --> PyLIMA  conversions
    pylima_q = q

    q_prime = (1.0 - q) / (2.0 * (1 + q))

    pylima_u0 = u0_amp + q_prime * sep * np.sin(phi_rad)
    pylima_t0 = t0 + q_prime * sep * tE * np.cos(phi_rad)

    # Note that pylima_phi = phi

    log_q = np.log10(pylima_q)
    log_s = np.log10(sep)

    # Load up some artificial data for pyLIMA... need this for time array definition.
    tests_dir = os.path.dirname(os.path.realpath(__file__))
    pylima_data = np.loadtxt(tests_dir + '/OB120169_phot.dat')
    #pylima_data[:, 1] = 1e5
    time_jd = pylima_data[:, 0]
    time_mjd = time_jd - 2400000.5

    pylima_tel = telescopes.Telescope(name='OGLE', camera_filter='I',
                                      light_curve=pylima_data,
                                      light_curve_names = ['time', 'mag', 'err_mag'],
                                      light_curve_units = ['JD', 'mag', 'mag'])

    pylima_ev = event.Event()
    pylima_ev.name = 'Fubar'
    pylima_ev.telescopes.append(pylima_tel)

    pylima_mod = generate_model.create_model('PSBL', pylima_ev)
    pylima_mod.define_model_parameters()
    pylima_mod.blend_flux_ratio = False

    tmp_params = [pylima_t0 + 2400000.5, pylima_u0, tE, sep, pylima_q, phi_rad]
    pylima_par = pylima_mod.compute_pyLIMA_parameters(tmp_params)
    pylima_par.fsource_OGLE = microltoolbox.brightness_transformation.magnitude_to_flux(mag_src)
    pylima_par.fblend_OGLE = pylima_par.fsource_OGLE * (1.0 - b_sff) / b_sff
    pylima_amp = pylima_mod.model_magnification(pylima_tel, pylima_par)

    pylima_mod_out = pylima_mod.compute_the_microlensing_model(pylima_tel, pylima_par)
    pylima_lcurve = pylima_mod_out['photometry']
    pylima_lcurve_mag = microltoolbox.brightness_transformation.flux_to_magnitude(pylima_lcurve)

    # Compute our model
    psbl = model.PSBL_Phot_noPar_Param1(t0, u0_amp, tE, piEE, piEN, q, sep, phi,
                                        [b_sff], [mag_src], root_tol=1e-6)

    our_mag = psbl.get_photometry(time_mjd)

    max_delta = np.max(np.abs(pylima_lcurve_mag - our_mag))

    if plot:
        plt.figure(1, figsize=(11, 6))
        plt.clf()
        f1 = plt.gcf().add_axes([0.4, 0.35, 0.57, 0.6])
        f2 = plt.gcf().add_axes([0.4, 0.15, 0.57, 0.2])
        f1.get_shared_x_axes().join(f1, f2)
        f1.set_xticklabels([])

        f1.plot(time_mjd, pylima_lcurve_mag, 'ko', label='pyLIMA')
        f1.plot(time_mjd, our_mag, 'r.', label='Ours')
        f1.invert_yaxis()
        f1.set_xlabel('MJD (day)')
        f1.set_ylabel('I (mag)')
        f1.legend()

        f2.plot(time_mjd, pylima_lcurve_mag - our_mag, 'k.')
        f2.set_xlabel('MJD (day)')
        f2.set_ylabel('PyL-Ours')

        tleft = 0.03
        ttop = 0.7
        ttstep = 0.05
        fig = plt.gcf()
        fig.text(tleft, ttop - 0 * ttstep, 't0 = {0:.1f} (MJD)'.format(t0), fontsize=12)
        fig.text(tleft, ttop - 1 * ttstep, 't0_pyL = {0:.1f} (MJD)'.format(pylima_t0), fontsize=12)
        fig.text(tleft, ttop - 2 * ttstep, 'u0 = {0:.3f}'.format(u0_amp), fontsize=12)
        fig.text(tleft, ttop - 3 * ttstep, 'u0_pyL = {0:.3f}'.format(pylima_u0), fontsize=12)
        fig.text(tleft, ttop - 4 * ttstep, 'tE = {0:.1f} (day)'.format(tE), fontsize=12)
        fig.text(tleft, ttop - 5 * ttstep, 'q  = {0:.5f}'.format(q), fontsize=12)
        fig.text(tleft, ttop - 6 * ttstep, 'q_pyL  = {0:.5f}'.format(pylima_q), fontsize=12)
        fig.text(tleft, ttop - 7 * ttstep, 'sep  = {0:.5f}'.format(sep), fontsize=12)
        fig.text(tleft, ttop - 8 * ttstep, 'phi  = {0:.1f}'.format(phi), fontsize=12)
        fig.text(tleft, ttop - 9 * ttstep, 'b_sff  = {0:.2f}'.format(b_sff), fontsize=12)
        fig.text(tleft, ttop - 10 * ttstep, 'mag_src  = {0:.1f}'.format(mag_src), fontsize=12)

        if max_delta > tol:
            fig.text(tleft, 0.05, '!!BAD!!', fontsize=16, color='red')

            plt.savefig('PSBL_phot_vs_pyLIMA.png')

    return (time_mjd, pylima_lcurve_mag, our_mag, max_delta)


@pytest.mark.skip(reason="broken: pyLIMA parallax differs slightly (erfa vs. jpl?)")
def test_PSPL_phot_vs_pyLIMA_parallax(plot=False):
    # Parameters: BAGLE style (conversion down later)
    ra = 267.4640833333333
    dec = -34.62555555555556
    t0 = 55775.0
    u0_amp = 0.5
    tE = 200.0
    piEE = 0.5
    piEN = -0.1
    mag_src = 16
    b_sff = 1.0

    tol = 1e-6

    # res = plot_compare_vs_pylima_pspl(ra, dec, t0, u0_amp, tE, piEE, piEN, mag_src, b_sff, parallax=False)
    # t_mjd, mag_pyl, mag_our, max_delta = res
    # assert max_delta < 1e-6

    res = plot_compare_vs_pylima_pspl(ra, dec, t0, u0_amp, tE, piEE, piEN, mag_src, b_sff,
                                      parallax=True, tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < 1e-6

    return

def test_PSPL_phot_vs_pyLIMA_noparallax(plot=False):
    # Parameters: BAGLE style (conversion down later)
    ra = 267.4640833333333
    dec = -34.62555555555556
    t0 = 55775.0
    u0_amp = 0.5
    tE = 200.0
    piEE = 0.5
    piEN = -0.1
    mag_src = 16
    b_sff = 1.0

    tol = 1e-6

    # res = plot_compare_vs_pylima_pspl(ra, dec, t0, u0_amp, tE, piEE, piEN, mag_src, b_sff, parallax=False)
    # t_mjd, mag_pyl, mag_our, max_delta = res
    # assert max_delta < 1e-6

    res = plot_compare_vs_pylima_pspl(ra, dec, t0, u0_amp, tE, piEE, piEN, mag_src, b_sff,
                                      parallax=False, tol=tol, plot=plot)
    t_mjd, mag_pyl, mag_our, max_delta = res
    assert max_delta < 1e-6

    return


def plot_compare_vs_pylima_pspl(ra, dec, t0, u0_amp, tE, piEE, piEN, mag_src, b_sff,
                                parallax=True, tol=1e-6, plot=True):
    """
    Compare pyLIMA to our models with some plots. 

    Input parameters are in our conventions and heliocentric coordinate system.
    """
    from pyLIMA import models as microlmodels
    from pyLIMA import event
    from pyLIMA import telescopes
    from pyLIMA import toolbox as microltoolbox
    from pyLIMA.models import generate_model

    # To convert between pyLIMA (geocentric) and Ours (heliocentric), we will
    # need some info about the Sun's position and velocity.
    # First define the reference time: We will use our t0 as the reference time.
    t0_par = t0

    #####
    # Load up some artificial data for pyLIMA... need this for time array definition.
    #####
    tests_dir = os.path.dirname(os.path.realpath(__file__))
    pylima_data = np.loadtxt(tests_dir + '/OB120169_phot.dat')
    time_jd = pylima_data[:, 0]
    time_mjd = time_jd - 2400000.5

    #####
    # Compute our model
    #####
    if parallax:
        pspl = model.PSPL_Phot_Par_Param1(t0, u0_amp, tE, piEE, piEN,
                                          [b_sff], [mag_src], raL=ra, decL=dec)
    else:
        pspl = model.PSPL_Phot_noPar_Param1(t0, u0_amp, tE, piEE, piEN,
                                            [b_sff], [mag_src], raL=ra, decL=dec)

    our_mag = pspl.get_photometry(time_mjd)
    # our_xL = pspl.get_lens_astrometry(time_mjd)
    # our_xS = pspl.get_astrometry_unlensed(time_mjd)
    # our_u = our_xS - our_xL

    #####
    # Compute pyLIMA model
    #####


    foo = frame_convert.convert_helio_geo_phot(ra, dec, t0, pspl.u0_amp, pspl.tE,
                                               pspl.piE[0], pspl.piE[1], t0_par,
                                               murel_in='SL', murel_out='LS',
                                               coord_in='EN', coord_out='tb',
                                               plot=False)
    t0_geo = foo[0]
    u0_geo = foo[1]
    tE_geo = foo[2]
    piEE_geo = foo[3]
    piEN_geo = foo[4]

    # Hmmm... I can only get PSPL without parallax to work with helio coords.
    # Hmmm... I can only get PSPL with    parallax to work with geotr coords.
    if parallax:
        pylima_u0 = u0_geo
        pylima_t0 = t0_geo + 2400000.5
        pylima_tE = tE_geo
        pylima_piEE = piEE_geo
        pylima_piEN = piEN_geo
    else:
        pylima_u0 = u0_amp
        pylima_t0 = t0 + 2400000.5
        pylima_tE = tE
        pylima_piEE = piEE
        pylima_piEN = piEN
        
    pylima_t0_par = t0_par + 2400000.5

    pylima_tel = telescopes.Telescope(name='OGLE', camera_filter='I',
                                      light_curve=pylima_data,
                                      light_curve_names = ['time', 'mag', 'err_mag'],
                                      light_curve_units = ['JD', 'mag', 'mag'])
    pylima_ev = event.Event()
    pylima_ev.name = 'Fubar'
    pylima_ev.telescopes.append(pylima_tel)
    pylima_ev.ra = ra
    pylima_ev.dec = dec

    if parallax:
        pylima_fancy_params = {}
        pylima_mod = generate_model.create_model('PSPL', pylima_ev, parallax=['Annual', pylima_t0_par])
        tmp_params = [pylima_t0, pylima_u0, pylima_tE, pylima_piEN, pylima_piEE]
    else:
        pylima_mod = generate_model.create_model('PSPL', pylima_ev)
        tmp_params = [pylima_t0, pylima_u0, pylima_tE]

    pylima_mod.define_model_parameters()
    pylima_mod.blend_flux_ratio = False

    pylima_par = pylima_mod.compute_pyLIMA_parameters(tmp_params)
    pylima_par.fsource_OGLE = microltoolbox.brightness_transformation.magnitude_to_flux(mag_src)
    pylima_par.fblend_OGLE = pylima_par.fsource_OGLE * (1.0 - b_sff) / b_sff

    pylima_amp = pylima_mod.model_magnification(pylima_tel, pylima_par)
    
    pylima_mod_out = pylima_mod.compute_the_microlensing_model(pylima_tel, pylima_par)
    pylima_lcurve = pylima_mod_out['photometry']
    pylima_lcurve_mag = microltoolbox.brightness_transformation.flux_to_magnitude(pylima_lcurve)
    # pylima_x, pylima_y, pylima_s, pylima_ang = pylima_mod.source_trajectory(pylima_tel, pylima_par,
    #                                                                         data_type='photometry')

    max_delta = np.max(np.abs(pylima_lcurve_mag - our_mag))

    if plot:
        plt.figure(1, figsize=(11, 6))
        plt.clf()
        f1 = plt.gcf().add_axes([0.4, 0.35, 0.57, 0.6])
        f2 = plt.gcf().add_axes([0.4, 0.15, 0.57, 0.2])
        f1.get_shared_x_axes().join(f1, f2)
        f1.set_xticklabels([])

        f1.plot(time_mjd, pylima_lcurve_mag, 'ko', label='pyLIMA')
        f1.plot(time_mjd, our_mag, 'r.', label='Ours')
        f1.invert_yaxis()
        f1.set_xlabel('MJD (day)')
        f1.set_ylabel('I (mag)')
        f1.legend()

        f2.plot(time_mjd, pylima_lcurve_mag - our_mag, 'k.')
        f2.set_xlabel('MJD (day)')
        f2.set_ylabel('PyL-Ours')

        tleft = 0.03
        ttop = 0.8
        ttstep = 0.05
        fig = plt.gcf()
        fig.text(tleft, ttop - 0 * ttstep, 'raL = {0:.2f} (deg)'.format(ra), fontsize=12)
        fig.text(tleft, ttop - 1 * ttstep, 'decL = {0:.2f} (deg)'.format(dec), fontsize=12)
        fig.text(tleft, ttop - 2 * ttstep, 't0 = {0:.1f} (MJD)'.format(t0), fontsize=12)
        fig.text(tleft, ttop - 3 * ttstep, 'u0     = {0:.3f}'.format(u0_amp), fontsize=12)
        fig.text(tleft, ttop - 4 * ttstep, 'u0_pyL = {0:.3f}'.format(pylima_u0), fontsize=12)
        fig.text(tleft, ttop - 5 * ttstep, 'tE = {0:.1f} (day)'.format(tE), fontsize=12)
        fig.text(tleft, ttop - 6 * ttstep, 'tE_pyL = {0:.1f} (day)'.format(pylima_tE), fontsize=12)
        fig.text(tleft, ttop - 7 * ttstep, 'piEE (S-L) = {0:.4f}'.format(piEE), fontsize=12)
        fig.text(tleft, ttop - 8 * ttstep, 'piEE_pyL (L-S) = {0:.4f}'.format(pylima_piEE), fontsize=12)
        fig.text(tleft, ttop - 9 * ttstep, 'piEN (S-L) = {0:.4f}'.format(piEN), fontsize=12)
        fig.text(tleft, ttop - 10 * ttstep, 'piEN_pyL (L-S) = {0:.4f}'.format(pylima_piEN), fontsize=12)
        fig.text(tleft, ttop - 11 * ttstep, 'b_sff = {0:.2f}'.format(b_sff), fontsize=12)
        fig.text(tleft, ttop - 12 * ttstep, 'mag_src = {0:.1f}'.format(mag_src), fontsize=12)

        if max_delta > 1e-6:
            fig.text(tleft, 0.05, '!!BAD!!', fontsize=16, color='red')

        plt.savefig('PSPL_phot_vs_pyLIMA.png')

        # plt.figure(2, figsize=(11, 6))
        # plt.clf()
        # f3 = plt.gcf().add_axes([0.4, 0.60, 0.57, 0.3])
        # f4 = plt.gcf().add_axes([0.4, 0.15, 0.57, 0.3])
        # f3.get_shared_x_axes().join(f3, f4)

        # f3.plot(time_mjd, pylima_x, 'ko', label='pyLIMA')
        # f3.plot(time_mjd, our_u[:, 0], 'r.', label='Ours')
        # f4.plot(time_mjd, pylima_y, 'ko', label='pyLIMA')
        # f4.plot(time_mjd, our_u[:, 1], 'r.', label='Ours')
        # f3.legend()

    return (time_mjd, pylima_lcurve_mag, our_mag, max_delta)


def test_bagle_self_conversion(plot=False):
    """
    Test whether our helio --> geotr converesions are
    self-consistent.
    """
    model.solar_system_ephemeris.set('builtin')

    raL = 260.0
    decL = -30.0

    t0 = 56000.0
    u0_amp = 0.2
    tE = 135.0
    piE_E = 0.1
    piE_N = 0.1
    b_sff = 1
    mag_src = 18
    t0par = 56050.0

    t_obs = np.arange(t0 - 3 * tE, t0 + 3 * tE, 0.1)

    ########## 
    # Test Setup: (1) Helio model --> (2) convert to geotr params --> (3) geotr model
    ##########
    pspl_h = model.PSPL_Phot_Par_Param1(t0, u0_amp, tE,
                                        piE_E, piE_N, b_sff, mag_src,
                                        raL, decL)

    t0_g, u0_amp_g, tE_g, piE_E_g, piE_N_g = pspl_h.get_geoproj_params(t0par, plot=False)

    pspl_g = model.PSPL_Phot_Par_Param1_geoproj(t0_g, u0_amp_g, tE_g,
                                                piE_E_g, piE_N_g, b_sff, mag_src,
                                                t0par,
                                                raL, decL)

    mag_h = pspl_h.get_photometry(t_obs)
    mag_g = pspl_g.get_photometry(t_obs)

    amp_h = pspl_h.get_amplification(t_obs)
    amp_g = pspl_g.get_amplification(t_obs)

    if plot:
        # Plot the amplification
        plt.figure(8)
        plt.clf()
        fig1, ax1 = plt.subplots(2, 1, num=8, sharex=True)
        ax1[0].plot(t_obs - t0, mag_h, 'ko', label='helio', alpha=0.5)
        ax1[0].plot(t_obs - t0, mag_g, 'r.', label='geotr', alpha=0.5)
        ax1[1].plot(t_obs - t0, mag_h - mag_g, 'b.')
        ax1[1].set_xlabel('t - t0 (MJD)')
        ax1[0].set_ylabel('Mag')
        ax1[1].set_ylabel('helio - geotr')
        ax1[0].legend()
        ax1[0].invert_yaxis()
        ax1[0].set_title('helio --> geotr')
        

    # Loose tolerance because this is a JPL vs. erfa.epv00 ephemeris problem.
    assert np.abs(mag_h - mag_g).max() < 1e-2

    ##########
    # Test Setup: (1) Geotr model --> (2) convert to helio params --> (3) helio model
    ##########
    t0_h, u0_amp_h, tE_h, piE_E_h, piE_N_h = pspl_g.get_helio_params(plot=False)

    pspl_h_from_g = model.PSPL_Phot_Par_Param1(t0_h, u0_amp_h, tE_h,
                                               piE_E_h, piE_N_h, b_sff, mag_src,
                                               raL, decL)

    mag_h_from_g = pspl_h_from_g.get_photometry(t_obs)

    amp_h_from_g = pspl_h_from_g.get_amplification(t_obs)

    if plot:
        # Plot the magnitudes of helio vs. geo
        plt.figure(9)
        plt.clf()
        fig1, ax1 = plt.subplots(2, 1, num=9, sharex=True)
        ax1[0].plot(t_obs - t0, mag_h_from_g, 'ko', label='helio_convert', alpha=0.5)
        ax1[0].plot(t_obs - t0, mag_g, 'r.', label='geotr', alpha=0.5)
        ax1[1].plot(t_obs - t0, mag_h_from_g - mag_g, 'b.')
        ax1[1].set_xlabel('t - t0 (MJD)')
        ax1[0].set_ylabel('Mag')
        ax1[1].set_ylabel('helio_convert - geotr')
        ax1[0].legend()
        ax1[0].invert_yaxis()
        ax1[0].set_title('geotr --> helio')

        # Compare helio vs. helio after conversion
        plt.figure(10)
        plt.clf()
        fig1, ax1 = plt.subplots(2, 1, num=10, sharex=True)
        ax1[0].plot(t_obs - t0, mag_h, 'ko', label='helio_orig', alpha=0.5)
        ax1[0].plot(t_obs - t0, mag_h, 'r.', label='helio_convert', alpha=0.5)
        ax1[1].plot(t_obs - t0, mag_h_from_g - mag_h, 'b.')
        ax1[1].set_xlabel('t - t0 (MJD)')
        ax1[0].set_ylabel('Mag')
        ax1[1].set_ylabel('helio_convert - helio')
        ax1[0].legend()
        ax1[0].invert_yaxis()
        ax1[0].set_title('helio --> geotr --> helio')
        

    assert np.abs(mag_h - mag_h_from_g).max() < 1e-9

    model.solar_system_ephemeris.set('jpl')

    return



def test_PSPL_Phot_Par_Param1_geoproj():
    t0par = 56000
    t0 = 56026.03
    u0_amp = -0.222
    tE = 135.0
    piE_E = -0.058
    piE_N = 0.11
    b_sff = 1.0
    mag_src = 19.266

    raL = (17.0 + (49.0 / 60.) + (51.38 / 3600.0)) * 15.0  # degrees
    decL = -35 + (22.0 / 60.0) + (28.0 / 3600.0)

    pspl = run_test_PSPL_Phot_Par_Param1_geoproj(t0, u0_amp, tE,
                                                 piE_E, piE_N, b_sff, mag_src,
                                                 t0par,
                                                 raL, decL, outdir='')

    return

def test_PSPL_PhotAstrom_Par_Param4_geoproj():
    t0par = 56000
    t0 = 56026.03
    u0_amp = -0.222
    tE = 135.0
    piE_E = -0.058
    piE_N = 0.11
    b_sff = 1.0
    mag_base = 19.266

    thetaE = 3
    piS = 0.125
    xS0_E = 0.1
    xS0_N = -0.1
    muS_E = 2
    muS_N = 1

    raL = (17.0 + (49.0 / 60.) + (51.38 / 3600.0)) * 15.0  # degrees
    decL = -35 + (22.0 / 60.0) + (28.0 / 3600.0)

    run_test_PSPL_PhotAstrom_Par_Param4_geoproj(t0, u0_amp, tE, thetaE, piS,
                                                piE_E, piE_N,
                                                xS0_E, xS0_N,
                                                muS_E, muS_N,
                                                b_sff, mag_base,
                                                t0par,
                                                raL, decL,
                                                outdir='')
    return


def run_test_PSPL_PhotAstrom_Par_Param4_geoproj(t0, u0_amp, tE, thetaE, piS,
                                                piE_E, piE_N,
                                                xS0_E, xS0_N,
                                                muS_E, muS_N,
                                                b_sff, mag_base,
                                                t0par,
                                                raL, decL,
                                                plot=False,
                                                outdir=''):
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    pspl = model.PSPL_PhotAstrom_Par_Param4_geoproj(t0, u0_amp, tE, thetaE, piS,
                                                    piE_E, piE_N,
                                                    xS0_E, xS0_N,
                                                    muS_E, muS_N,
                                                    b_sff, mag_base,
                                                    t0par,
                                                    raL, decL)

    assert pspl.t0 == pytest.approx(t0)
    assert pspl.u0_amp == pytest.approx(u0_amp)

    t = np.arange(t0 - 1000, t0 + 1000, 1)
    dt = t - pspl.t0

    A = pspl.get_amplification(t)
    pos = pspl.get_astrometry(t)
    shift = pspl.get_centroid_shift(t)

    if plot:
        # Plot the amplification
        plt.figure(1)
        plt.clf()
        plt.plot(dt, 2.5 * np.log10(A), 'k.')
        plt.xlabel('t - t0 (MJD)')
        plt.ylabel('2.5 * log(A)')
        plt.savefig(outdir + 'amp_v_time.png')
        plt.show()

        # Plot the position
        plt.figure(2)
        plt.clf()
        plt.plot(pos[:, 0], pos[:, 1], 'k.')
        plt.xlabel('RA (arcsec)')
        plt.ylabel('Dec (arcsec)')
        plt.savefig(outdir + 'pos.png')
        plt.show()

        # Plot the shift
        plt.figure(3)
        plt.clf()
        plt.plot(shift[:, 0], shift[:, 1], 'k.')
        plt.xlabel('RA (mas)')
        plt.ylabel('Dec (mas)')
        plt.savefig(outdir + 'shift.png')
        plt.show()
        
    return pspl


def run_test_PSPL_Phot_Par_Param1_geoproj(t0, u0_amp, tE,
                                          piE_E, piE_N, b_sff, mag_src,
                                          t0par,
                                          raL, decL,
                                          plot=False, outdir=''):
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    pspl = model.PSPL_Phot_Par_Param1_geoproj(t0, u0_amp, tE,
                                              piE_E, piE_N, b_sff, mag_src,
                                              t0par,
                                              raL, decL)

    assert pspl.t0 == pytest.approx(t0)
    assert pspl.u0_amp == pytest.approx(u0_amp)

    t = np.arange(t0 - 500, t0 + 500, 1)
    dt = t - pspl.t0

    A = pspl.get_amplification(t)

    if plot:
        # Plot the amplification
        plt.figure(1)
        plt.clf()
        plt.plot(dt, 2.5 * np.log10(A), 'k.')
        plt.xlabel('t - t0 (MJD)')
        plt.ylabel('2.5 * log(A)')
        plt.savefig(outdir + 'amp_v_time.png')
        plt.show()

    return pspl

def test_geoproj_mm(plot=False):
    """
    Test making sure BAGLE's geocentric projected model gives the same
    thing as MulensModel.
    Note: diff of less than 1e-10 is pretty strict. Maybe loosen if needed.
    """
    diff = compare_geoproj_mm(260, -30, 56000, 56050, -0.2, 135, -0.058, 0.11, plot=plot)
    assert np.abs(diff).max() < 1e-10

    diff = compare_geoproj_mm(260, -30, 56000, 56050, 0.02, 135, 0.058, 0.01, plot=True)
    assert np.abs(diff).max() < 1e-10


def compare_geoproj_mm(raL, decL, t0par, t0, u0_amp, tE, piE_E, piE_N, plot=False):
    """
    Compare BAGLE's geocentric projected formalism implementation against
    that of MulensModel.
    All conventions are the same, except MulensModel takes times in HJD and 
    BAGLE takes times in MJD.
    Note: 
        raL and decL should be in degrees.
        t0par and t0 should be in MJD.
    """
    import MulensModel as mm

    coords = SkyCoord(raL, decL, unit=(u.deg, u.deg))

    t_mjd = np.arange(t0 - 5 * tE, t0 + 5 * tE, 0.1)
    t_hjd = t_mjd + 2400000.5

    params = {}
    params['t_0'] = t0 + 2400000.5
    params['t_0_par'] = t0par + 2400000.5
    params['u_0'] = u0_amp
    params['t_E'] = tE
    params['pi_E_N'] = piE_N
    params['pi_E_E'] = piE_E

    my_model = mm.Model(params, coords=coords)

    A_mm = my_model.get_magnification(t_hjd)

    b_sff = 1
    mag_src = 18

    pspl = model.PSPL_Phot_Par_Param1_geoproj(t0, u0_amp, tE,
                                              piE_E, piE_N, b_sff, mag_src,
                                              t0par,
                                              raL, decL)
    A_b = pspl.get_amplification(t_mjd)

    if plot:
        # Plot the amplification
        fig, ax = plt.subplots(2, 1, num=1, sharex=True)
        plt.clf()
        fig, ax = plt.subplots(2, 1, num=1, sharex=True)
        ax[0].plot(t_mjd - t0, A_b, 'ko', label='BAGLE', alpha=0.5)
        ax[0].plot(t_mjd - t0, A_mm, 'r.', label='MuLens', alpha=0.5)
        ax[1].plot(t_mjd - t0, A_b - A_mm, 'b.')
        ax[1].set_xlabel('t - t0 (MJD)')
        ax[0].set_ylabel('A')
        ax[1].set_ylabel('BAGLE - MuLens')
        ax[0].legend()
        plt.show()

    return (A_b - A_mm) / A_mm


def test_u0_hat_thetaE_hat():
    """
    Tests for:
        u0_hat_from_thetaE_hat()
        thetaE_hat_from_u0_hat()

    """
    # Tests the current code implementation:
    # Defines beta = u0_amp sign convention
    # opposite of how Andy Gould (2004) does. Ours has:
    #    beta > 0 means u0_E > 0
    #    u0_amp > 0 mean u0_E > 0
    #

    E_hat = np.array([1.0, 0.0])
    N_hat = np.array([0.0, 1.0])

    ##########
    # Test 1:
    #   u0 sign:     +, +
    #   muRel sign:  +, -
    ##########
    u0_hatE_in = 0.3
    u0_hatN_in = (1.0 - u0_hatE_in ** 2) ** 0.5
    u0_hat_in = np.array([u0_hatE_in, u0_hatN_in])

    # direction of relative proper motion vector
    # Same as thetaE_hat
    muRel_hatE_in = (1.0 - u0_hatE_in ** 2) ** 0.5
    muRel_hatN_in = -0.3
    muRel_hat_in = np.array([muRel_hatE_in, muRel_hatN_in])

    # Should be positive.
    # in units of thetaE, opposite sign as beta???? NOT SURE ANYMORE.
    u0amp_in = np.hypot(u0_hatE_in, u0_hatN_in) * np.cross(u0_hat_in, N_hat) * 1.0

    # Test
    u0_hat = model.u0_hat_from_thetaE_hat(muRel_hat_in, u0amp_in)

    assert u0_hat[0] == u0_hat_in[0]
    assert u0_hat[1] == u0_hat_in[1]
    assert np.sign(u0_hat[0]) == np.sign(u0amp_in)

    ##########
    # Test 2
    #   u0 sign:     -, +
    #   muRel sign:  +, +
    ##########
    u0_hatE_in = -0.3
    u0_hatN_in = (1.0 - u0_hatE_in ** 2) ** 0.5
    u0_hat_in = np.array([u0_hatE_in, u0_hatN_in])

    # direction of relative proper motion vector
    # Same as thetaE_hat
    muRel_hatE_in = (1.0 - u0_hatE_in ** 2) ** 0.5
    muRel_hatN_in = 0.3
    muRel_hat_in = np.array([muRel_hatE_in, muRel_hatN_in])

    # Should be negative.
    u0amp_in = np.hypot(u0_hatE_in, u0_hatN_in) * np.cross(u0_hat_in, N_hat) * 1.0

    # Test
    u0_hat = model.u0_hat_from_thetaE_hat(muRel_hat_in, u0amp_in)

    assert u0_hat[0] == u0_hat_in[0]
    assert u0_hat[1] == u0_hat_in[1]
    assert np.sign(u0_hat[0]) == np.sign(u0amp_in)

    ##########
    # Test 3
    #   u0 sign:     -, -
    #   muRel sign:  -, +
    ##########
    u0_hatE_in = -0.3
    u0_hatN_in = -(1.0 - u0_hatE_in ** 2) ** 0.5
    u0_hat_in = np.array([u0_hatE_in, u0_hatN_in])

    # direction of relative proper motion vector
    # Same as thetaE_hat
    muRel_hatE_in = -(1.0 - u0_hatE_in ** 2) ** 0.5
    muRel_hatN_in = 0.3
    muRel_hat_in = np.array([muRel_hatE_in, muRel_hatN_in])

    # Should be negative.
    u0amp_in = np.hypot(u0_hatE_in, u0_hatN_in) * np.cross(u0_hat_in, N_hat) * 1.0

    # Test
    u0_hat = model.u0_hat_from_thetaE_hat(muRel_hat_in, u0amp_in)

    assert u0_hat[0] == u0_hat_in[0]
    assert u0_hat[1] == u0_hat_in[1]
    assert np.sign(u0_hat[0]) == np.sign(u0amp_in)

    ##########
    # Test 4
    #   u0 sign:     +, -
    #   muRel sign:  +, +
    ##########
    u0_hatE_in = 0.3
    u0_hatN_in = -(1.0 - u0_hatE_in ** 2) ** 0.5
    u0_hat_in = np.array([u0_hatE_in, u0_hatN_in])

    # direction of relative proper motion vector
    # Same as thetaE_hat
    muRel_hatE_in = (1.0 - u0_hatE_in ** 2) ** 0.5
    muRel_hatN_in = 0.3
    muRel_hat_in = np.array([muRel_hatE_in, muRel_hatN_in])

    # Should be negative.
    u0amp_in = np.hypot(u0_hatE_in, u0_hatN_in) * np.cross(u0_hat_in, N_hat) * 1.0

    # Test
    u0_hat = model.u0_hat_from_thetaE_hat(muRel_hat_in, u0amp_in)

    assert u0_hat[0] == u0_hat_in[0]
    assert u0_hat[1] == u0_hat_in[1]
    assert np.sign(u0_hat[0]) == np.sign(u0amp_in)

    return


def test_PSBL_get_photometry_nans():
    # This set of parameters reproduces the problem of lots of NaNs
    # in the amplification array.
    raL = 255.9785152922
    decL = -26.7699679331
    mL1 = 0.5449857890
    mL2 = 0.2513479648
    t0 = -445.5166414077
    xS0 = [0, 0]
    beta = 63.1888232379
    muL = [-5.64504014, -5.63716286]
    muS = [-4.60154964, -5.37112324]
    dL = 6971.8480854741
    dS = 13330.5805517047
    sep = 205.1250722516
    alpha = -496.2173351517
    mag_src = 10.6950225830
    b_sff = 0.0005696291

    psbl = model.PSBL_PhotAstrom_Par_Param1(mL1, mL2, t0, xS0[0], xS0[1],
                                            beta, muL[0], muL[1], muS[0], muS[1], dL, dS,
                                            sep, alpha, [b_sff], [mag_src],
                                            raL=raL, decL=decL, root_tol=1e-10)

    # print(f't0 = {psbl.t0:.1f} MJD')
    # print(f'tE = {psbl.tE:.1f} days')

    duration = 100  # tE
    time_steps = 5000
    tmin = psbl.t0 - ((duration / 2.0) * psbl.tE)
    tmax = psbl.t0 + ((duration / 2.0) * psbl.tE)
    dt = np.linspace(tmin, tmax, time_steps)

    dt = dt[1780:1788]

    img, amp = psbl.get_all_arrays(dt)
    phot = psbl.get_photometry(dt, amp_arr=amp)

    # print('dt = ', dt)
    # print('poht = ', phot)
    # print('amp = ')
    # print(amp)

    # Check that we have both NaNs and valid values
    # in our amplifications for testing.
    idx_nan = np.where(np.isnan(amp).sum(axis=1) == 5)[0]
    # print(amp.shape)
    if len(idx_nan) > 0:
        assert len(idx_nan) != amp.shape[0]

        # Check that the amp=nans are returned as masked.
        assert np.sum(phot.mask[idx_nan]) == len(idx_nan)

        # Check that the data itself has nan (not junk values)
        # print(phot)
        # print(phot.data)
        # print(phot.mask)
        # print(phot.data[idx_nan[0]])
        assert np.isnan(phot.data[idx_nan]).sum() == len(idx_nan)

    return


def old_test_PSBL_too_many_peaks3():
    """
    What are we testing here? 
    """
    raL = 267.9828892936
    decL = -26.4253612405
    mL1 = 1.1694705685
    mL2 = 6.8748978010
    t0 = 272.1230025420
    xS0 = [0, 0]
    beta = 56.9742058606
    muL = [-3.19572701, -5.71742749]
    muS = [-3.50599981, -6.20537068]
    dL = 3693.8211092591
    dS = 8293.3433805508
    sep = 0.9251665444
    alpha = 147.6
    mag_src = 5.1452097893
    b_sff = 0.2092014345

    psbl = model.PSBL_PhotAstrom_Par_Param1(mL1, mL2, t0, xS0[0], xS0[1],
                                            beta, muL[0], muL[1], muS[0], muS[1], dL, dS,
                                            sep, alpha, [b_sff], [mag_src],
                                            raL=raL, decL=decL, root_tol=1e-8)

    print(f't0 = {psbl.t0:.1f} MJD')
    print(f'tE = {psbl.tE:.1f} days')

    # Print out some angles to see when things might be a problem.
    phi_muRel = np.degrees(np.arctan2(muS[1] - muL[1], muS[0] - muL[0]) - np.arctan2(1, 0))
    phi = alpha - phi_muRel
    print(f'phi_muRel = {phi_muRel} deg')
    print(f'phi = {phi} deg')
    print(f'alpha = {alpha} deg')

    duration = 100  # tE
    time_steps = 5000
    tmin = psbl.t0 - 5000
    tmax = psbl.t0 + 5000
    dt = np.linspace(tmin, tmax, time_steps)

    img_arr, amp_arr = psbl.get_all_arrays(dt)
    phot = psbl.get_photometry(dt, amp_arr=amp_arr)

    plt.figure(1)
    plt.clf()
    plt.plot(dt, phot)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.ticklabel_format(useOffset=False)

    return


def old_test_PSBL_too_many_peaks_621():
    raL = 255.9785152922
    decL = -26.7699679331
    mL1 = 0.5449857890
    mL2 = 0.2513479648
    t0 = -445.5166414077
    xS0 = [0, 0]
    beta = 63.1888232379
    muL = [-5.64504014, -5.63716286]
    muS = [-4.60154964, -5.37112324]
    dL = 6971.8480854741
    dS = 13330.5805517047
    sep = 205.1250722516
    alpha = -496.2173351517
    mag_src = 10.6950225830
    b_sff = 0.0005696291

    psbl = model.PSBL_PhotAstrom_Par_Param1(mL1, mL2, t0, xS0[0], xS0[1],
                                            beta, muL[0], muL[1], muS[0], muS[1], dL, dS,
                                            sep, alpha, [b_sff], [mag_src],
                                            raL=raL, decL=decL, root_tol=1e-8)

    print(f't0 = {psbl.t0:.1f} MJD')
    print(f'tE = {psbl.tE:.1f} days')

    # Print out some angles to see when things might be a problem.
    phi_muRel = np.degrees(np.arctan2(muS[1] - muL[1], muS[0] - muL[0]) - np.arctan2(1, 0))
    phi = alpha - phi_muRel
    print(f'phi_muRel = {phi_muRel} deg')
    print(f'phi = {phi} deg')
    print(f'alpha = {alpha} deg')

    duration = 100  # tE
    time_steps = 100
    tmin = psbl.t0 - 2
    tmax = psbl.t0 + 2

    # time_steps = 10000
    # tmin = psbl.t0 - 1000
    # tmax = psbl.t0 + 1000
    dt = np.linspace(tmin, tmax, time_steps)

    img_arr, amp_arr = psbl.get_all_arrays(dt)
    phot = psbl.get_photometry(dt, amp_arr=amp_arr)

    print(amp_arr[0:5], img_arr[0:5])

    plt.figure(1)
    plt.clf()
    plt.plot(dt, phot, 'k.-')
    ax = plt.gca()
    ax.invert_yaxis()
    ax.ticklabel_format(useOffset=False)

    return


# Testing GP classes.
def test_GP_classes(plot=False):
    """
    Make sure can instantiate.
    """
    t0 = 57000
    u0_amp = 0.5
    tE = 150
    piE_E = 0.05
    piE_N = 0.05
    b_sff = 0.9
    mag_src = 17.0
    gp_log_sigma = 0.1
    gp_log_rho = 1.0
    gp_log_So = 0.1
    gp_log_omegao = 0.1
    raL = 17.30 * 15.
    decL = -29.0

    gp_rho = 1.0
    gp_log_omegaofour_So = 0.1

    np.random.seed(42)
    data_stuff = fake_correlated_data(t0=t0, u0_amp=u0_amp, tE=tE,
                                      piE_E=piE_E, piE_N=piE_N,
                                      b_sff=b_sff, mag_src=mag_src,
                                      gp_log_sigma=gp_log_sigma, gp_log_rho=gp_log_rho,
                                      gp_log_So=gp_log_So, gp_log_omegao=gp_log_omegao,
                                      raL=raL, decL=decL)
    pspl_model_in = data_stuff[0]
    data23_uncorr = data_stuff[1]
    data23 = data_stuff[2]
    params = data_stuff[3]

    model2 = model.PSPL_Phot_Par_GP_Param1(t0, u0_amp, tE,
                                           piE_E, piE_N, b_sff, mag_src,
                                           gp_log_sigma, gp_log_rho,
                                           gp_log_So, gp_log_omegao,
                                           raL=raL, decL=decL)

    model3 = model.PSPL_Phot_noPar_GP_Param1(t0, u0_amp, tE,
                                             piE_E, piE_N, b_sff, mag_src,
                                             gp_log_sigma, gp_log_rho,
                                             gp_log_So, gp_log_omegao,
                                             raL=raL, decL=decL)

    # Put in some assert statements to make sure things don't break in the future.
    times = np.arange(56000, 58000, 100)

    ##########
    # Note: Model 2 should match data (both with parallax) within errors.
    ##########
    # Test that model 2 photometry is sensible (with no GP).
    mod2_phot_good = np.array([16.88473035, 16.88425742, 16.88347207, 16.88227126, 16.87984510, 16.87418014,
                               16.86227960, 16.83498515, 16.74673373, 16.48463579, 16.11297016, 16.43088797,
                               16.75054887, 16.83643682, 16.86184533, 16.87410198, 16.87991725, 16.88227589,
                               16.88345499, 16.88426179])
    mod2_phot_out = model2.get_photometry(times)
    np.testing.assert_allclose(mod2_phot_out, mod2_phot_good, rtol=1e-4)

    # Test the model 2 GP photometry... seed is fixed so this should remain identical. But I loosened tolerance
    mod2_gpphot_good = np.array([16.88473035, 17.08646751, 16.74805916, 16.68253705, 16.87984509, 16.96989977,
                                 16.6386355, 16.83498515, 16.85998419, 16.51997825, 16.12006682, 16.43088797,
                                 16.61076884, 16.70502475, 16.93342688, 16.87410206, 16.87479723, 16.94838136,
                                 16.88345499, 17.01714564])
    mod2_gpphot_out, mod2_gpphot_std_out = model2.get_photometry_with_gp(data23['t_phot1'],
                                                                         data23['mag1'], data23['mag_err1'],
                                                                         t_pred=times)
    np.testing.assert_allclose(mod2_phot_out, mod2_phot_good, rtol=1e-2)

    # Test that we get the PSPL model out that we put in. (no GP)
    mod2_phot_out_at_tobs = model2.get_photometry(data23_uncorr['t_phot1'])
    np.testing.assert_allclose(mod2_phot_out_at_tobs, data23_uncorr['mag1'], rtol=0.3)

    if plot:
        plt.figure(1)
        plt.clf()
        plt.plot(times, mod2_gpphot_out, 'k-', label='With GP')
        plt.plot(times, mod2_phot_out, 'r-', label='No GP')
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title('test_GP: model2')

    ##########
    # Note: Model 3 should NOT match data (model without parallax, data with parallax)
    ##########
    # Test that model 2 photometry is sensible (with no GP).
    mod3_phot_good = np.array([16.88470908, 16.88426858, 16.88352731, 16.88220882, 16.87970123, 16.87452809,
                               16.86275283, 16.83268396, 16.74634087, 16.48738257, 16.09854878, 16.48738257,
                               16.74634087, 16.83268396, 16.86275283, 16.87452809, 16.87970123, 16.88220882,
                               16.88352731, 16.88426858])
    mod3_phot_out = model3.get_photometry(times)
    np.testing.assert_allclose(mod3_phot_out, mod3_phot_good, rtol=1e-5)

    # Test the model 2 GP photometry... seed is fixed so this should remain identical. But I loosened tolerance
    mod3_gpphot_good = np.array([16.88470908, 17.08646752, 16.74805921, 16.68253668, 16.87970122, 16.96989985,
                                 16.63863561, 16.83268396, 16.85998404, 16.51997894, 16.12006319, 16.48738257,
                                 16.61076444, 16.70502078, 16.93406828, 16.87452817, 16.874797, 16.94838128,
                                 16.88352731, 17.01714565])
    mod3_gpphot_out, mod3_gpphot_std_out = model3.get_photometry_with_gp(data23['t_phot1'],
                                                                         data23['mag1'], data23['mag_err1'],
                                                                         t_pred=times)
    np.testing.assert_allclose(mod3_phot_out, mod3_phot_good, rtol=1e-5)

    return


def test_FSPL_source_astrometry(plot=False):
    """
    Make sure we can instantiate.
    """
    mL = 10 # Msun
    t0 = 57000 # MJD
    beta = 2  # mas
    dL = 4000 # pc
    dS = 8000 # pc
    xS0_E = 0.000 # arcsec
    xS0_N = 0.002 # arcsec
    muS_E = 4.0 # mas/yr
    muS_N = 0.0
    muL_E = 0.0
    muL_N = 0.0
    radius = 1.0 # mas
    b_sff = 0.9
    mag_src = 19.0
    raL = 17.30 * 15.
    decL = -29.0

    # Array of times we will sample on.
    time_arr = np.linspace(t0 - 1000, t0 + 1000, 1000)

    ##########
    # Test 1: Compare FSPL and PSPL lens astrometry.
    ##########
    n_outline = 100
    fspl = model.FSPL_PhotAstrom_Par_Param1(mL, t0, beta, dL, dL / dS,
                                            xS0_E, xS0_N,
                                            muL_E, muL_N,
                                            muS_E, muS_N,
                                            radius,
                                            b_sff, mag_src, n_outline,
                                            raL=raL, decL=decL)

    pspl = model.PSPL_PhotAstrom_Par_Param1(mL, t0, beta, dL, dL / dS,
                                            xS0_E, xS0_N,
                                            muL_E, muL_N,
                                            muS_E, muS_N,
                                            b_sff, mag_src,
                                            raL=raL, decL=decL)

    #####
    # Compare the unlensed source astrometry.
    #####
    fspl_src_ast = fspl.get_astrometry_unlensed(time_arr)
    pspl_src_ast = pspl.get_astrometry_unlensed(time_arr)

    np.testing.assert_allclose(fspl_src_ast[::100], pspl_src_ast[::100], rtol=1e-6)

    if plot:
        plt.figure(4)
        plt.clf()
        fig, ax = plt.subplots(2, 1, sharex=True, num=4)
        ax[0].set_title('Unlensed, unresolved, source')
        ax[0].plot(time_arr, pspl_src_ast[:, 0], 'ko', label='pspl')
        ax[1].plot(time_arr, pspl_src_ast[:, 1], 'ko')
        ax[0].plot(time_arr, fspl_src_ast[:, 0], 'r.', label='fspl')
        ax[1].plot(time_arr, fspl_src_ast[:, 1], 'r.')
        ax[0].legend()
        ax[1].set_xlabel('Time (MJD)')
        ax[0].set_ylabel('X (")')
        ax[1].set_ylabel('Y (")')
        fig.show()

    #####
    # Check the unlensed, resolved source astrometry.
    #####
    time3 = np.linspace(t0 - 200, t0 + 200+1, 3)

    # Shape [N_times, N_outline, [E, N]]
    fspl_xyS_unlensed_res = fspl.get_astrometry_outline_unlensed(time3)

    # Check that all the points in the first time step are < second time step.
    np.testing.assert_array_less(fspl_xyS_unlensed_res[0, :, 0],
                                 fspl_xyS_unlensed_res[1, :, 0])
    # Check that the points at the first time step are in a circle of radius of 1 mas.
    x0_mean = fspl_xyS_unlensed_res[0, :, 0].mean()
    y0_mean = fspl_xyS_unlensed_res[0, :, 1].mean()
    dr_all0 = np.hypot(fspl_xyS_unlensed_res[0, :, 0] - x0_mean,
                       fspl_xyS_unlensed_res[0, :, 1] - y0_mean)
    np.testing.assert_almost_equal(dr_all0, 1.0, 1e-3)

    if plot:
        plt.figure(3)
        plt.clf()
        plt.title('Unlensed, resolved, source')
        colors = ['red', 'orange', 'green', 'cyan', 'blue']

        for tt in range(len(time3)):
            plt.plot(fspl_xyS_unlensed_res[tt, :, 0], fspl_xyS_unlensed_res[tt, :, 1],
                     'k.', color=colors[tt], label=f'{time3[tt]:.0f}')

        plt.axis('equal')
        plt.xlabel('X (")')
        plt.ylabel('Y (")')
        plt.legend()
        fig.show()

    #####
    # Check the lensed, resolved source astrometry.
    #####
    # Shape [N_times, N_outline, [+,-], [E, N]]
    fspl_xyS_lensed_res = fspl.get_resolved_astrometry_outline(time3) * 1e3 # mas
    # Shape [N_times, [E, N]]
    fspl_xyL_unlens = fspl.get_lens_astrometry(time3) * 1e3 # mas

    # all positive images should have greater Y than X
    np.testing.assert_array_less(fspl_xyS_lensed_res[:, :, 1, 1], fspl_xyS_lensed_res[:, :, 0, 1])

    # all positive images should be above the unlensed image in Y.
    np.testing.assert_array_less(fspl_xyS_unlensed_res[:, :, 1], fspl_xyS_lensed_res[:, :, 0, 1])

    if plot:
        plt.figure(2)
        plt.clf()
        plt.title('Time = red --> green')
        for tt in range(len(time3)):
            if tt == 1:
                lab_lens = 'Lens'
                lab_src = 'Src, unlensed'
                lab_src_p = 'Src, lensed, plus'
                lab_src_m = 'Src, lensed, minus'
            else:
                lab_lens = None
                lab_src = None
                lab_src_p = None
                lab_src_m = None

            plt.plot(fspl_xyL_unlens[tt, 0], fspl_xyL_unlens[tt, 1],
                     ls='None', marker='*', mec=colors[tt], mfc='none', label=lab_lens)

            plt.plot(fspl_xyS_unlensed_res[tt, :, 0], fspl_xyS_unlensed_res[tt, :, 1],
                     ls='None', marker='.', color=colors[tt], alpha=0.2, ms=2, label=lab_src)
            # plus image
            plt.plot(fspl_xyS_lensed_res[tt, :, 0, 0], fspl_xyS_lensed_res[tt, :, 0, 1],
                     ls='None', marker='x', color=colors[tt], ms=3, label=lab_src_p)
            # minus image
            plt.plot(fspl_xyS_lensed_res[tt, :, 1, 0], fspl_xyS_lensed_res[tt, :, 1, 1],
                     ls='None', marker='+', color=colors[tt], ms=3, label=lab_src_m)

        plt.axis('equal')
        plt.xlabel('X (mas)')
        plt.ylabel('Y (mas)')
        plt.legend(fontsize=12, loc='lower right')
        fig.show()

    #####
    # Check the lensed, unresolved source astrometry.
    #####
    fspl_xyS_unlens = fspl.get_astrometry_unlensed(time3) * 1e3 # mas
    fspl_xyS_lensed = fspl.get_astrometry(time3) * 1e3 # mas
    fspl_xyS_lensed_res = fspl.get_resolved_astrometry(time3) * 1e3 # mas

    if plot:
        plt.figure(1)
        plt.clf()
        plt.title('Time = red --> green')
        for tt in range(len(time3)):
        #for tt in range(1,2):
            if tt == 1:
                lab_lens = 'Lens'
                lab_src = 'Src, unlensed'
                lab_src_lensed = 'Src, lensed'
                lab_src_p = 'Src, lensed, p'
                lab_src_m = 'Src, lensed, m'
            else:
                lab_lens = None
                lab_src = None
                lab_src_lensed = None
                lab_src_p = None
                lab_src_m = None

            # lens position
            plt.plot(fspl_xyL_unlens[tt, 0], fspl_xyL_unlens[tt, 1],
                     ls='None', marker='*', mec=colors[tt], mfc='none', ms=15, label=lab_lens)

            # source position, unlensed
            plt.plot(fspl_xyS_unlens[tt, 0], fspl_xyS_unlens[tt, 1],
                     ls='None', marker='.', color=colors[tt], alpha=0.2, ms=10, label=lab_src)

            # lensed image, unresolved
            plt.plot(fspl_xyS_lensed[tt, 0], fspl_xyS_lensed[tt, 1],
                     ls='None', marker='o', color=colors[tt], ms=10, label=lab_src_lensed)

            # lensed image, resolved plus
            plt.plot(fspl_xyS_lensed_res[tt, 0, 0], fspl_xyS_lensed_res[tt, 0, 1],
                     ls='None', marker='x', color=colors[tt], ms=10, label=lab_src_p)

            # lensed image, resolved minus
            plt.plot(fspl_xyS_lensed_res[tt, 1, 0], fspl_xyS_lensed_res[tt, 1, 1],
                     ls='None', marker='+', color=colors[tt], ms=10, label=lab_src_m)


        plt.axis('equal')
        plt.xlabel('X (mas)')
        plt.ylabel('Y (mas)')
        plt.legend(fontsize=12)
        fig.show()

    return


def test_FSPL_PhotAstrom_classes(plot=False):
    """
    Make sure we can instantiate.
    """
    mL = 10
    t0 = 57000
    beta = 2
    dL = 4000
    dS = 8000
    xS0_E = 0.0
    xS0_N = 0.08E-3
    muS_E = -4.0
    muS_N = 0.0
    muL_E = 0.0
    muL_N = 0.0
    radius = 1e-3
    b_sff = 0.9
    mag_src = 19.0
    raL = 17.30 * 15.
    decL = -29.0

    # Array of times we will sample on.
    time_arr = np.linspace(t0 - 1000, t0 + 1000, 1000)

    ##########
    # Test 1: Compare FSPL and PSPL lens astrometry.
    ##########
    n_outline = 100
    fspl = model.FSPL_PhotAstrom_Par_Param1(mL, t0, beta, dL, dL / dS,
                                            xS0_E, xS0_N,
                                            muL_E, muL_N,
                                            muS_E, muS_N,
                                            radius,
                                            b_sff, mag_src, n_outline,
                                            raL=raL, decL=decL)

    pspl = model.PSPL_PhotAstrom_Par_Param1(mL, t0, beta, dL, dL / dS,
                                            xS0_E, xS0_N,
                                            muL_E, muL_N,
                                            muS_E, muS_N,
                                            b_sff, mag_src,
                                            raL=raL, decL=decL)

    # Compare the lens astrometry.
    fspl_lens_ast = fspl.get_lens_astrometry(time_arr)
    pspl_lens_ast = pspl.get_lens_astrometry(time_arr)

    np.testing.assert_allclose(fspl_lens_ast[::100], pspl_lens_ast[::100], rtol=1e-6)

    if plot:
        plt.figure(4)
        plt.clf()
        fig, ax = plt.subplots(2, 1, sharex=True, num=4)
        ax[0].plot(time_arr, pspl_lens_ast[:, 0], 'ko', label='pspl')
        ax[1].plot(time_arr, pspl_lens_ast[:, 1], 'ko')
        ax[0].plot(time_arr, fspl_lens_ast[:, 0], 'r.', label='fspl')
        ax[1].plot(time_arr, fspl_lens_ast[:, 1], 'r.')
        ax[0].legend()
        ax[1].set_xlabel('Time (MJD)')
        ax[0].set_ylabel('X (")')
        ax[1].set_ylabel('X (")')
        fig.show()
@pytest.mark.skip(reason="broken")
def test_FSPL_PhotAstrom_classes(plot=False):
    """
    Make sure we can instantiate.
    """
    mL = 10
    t0 = 57000
    beta = 2
    dL = 4000
    dS = 8000
    xS0_E = 0.0
    xS0_N = 0.08E-3
    muS_E = -4.0
    muS_N = 0.0
    muL_E = 0.0
    muL_N = 0.0
    radius = 1e-3
    b_sff = 0.9
    mag_src = 19.0
    raL = 17.30 * 15.
    decL = -29.0

    # Array of times we will sample on.
    time_arr = np.linspace(t0 - 1000, t0 + 1000, 1000)

    def test_fspl_once(r_in, n_in, phot_arr_good, mod='FSPL', tol=1e-3):
        # Make the model
        tstart = time.time()

        if mod == 'FSPL':
            tmp_mod = model.FSPL_PhotAstrom_Par_Param1(mL, t0, beta, dL, dL / dS,
                                                       xS0_E, xS0_N,
                                                       muL_E, muL_N,
                                                       muS_E, muS_N,
                                                       r_in,
                                                       b_sff, mag_src, n_in,
                                                       raL=raL, decL=decL)
        else:
            tmp_mod = model.PSPL_PhotAstrom_Par_Param1(mL, t0, beta, dL, dL / dS,
                                                       xS0_E, xS0_N,
                                                       muL_E, muL_N,
                                                       muS_E, muS_N,
                                                       b_sff, mag_src,
                                                       raL=raL, decL=decL)

        # Generate photometry
        phot_arr = tmp_mod.get_photometry(time_arr)

        # When working, fetch 10 points, evenly distributed and
        # save as the "right answer" in the individual tests below.
        # print(repr(phot_arr[::100]))  # GET GOOD

        # Check against good data
        # np.testing.assert_allclose(phot_arr[::100], phot_arr_good, rtol=tol)

        tdelta = time.time() - tstart

        print(f'Eval time for {mod:s}: {tdelta:.4f} sec, n = {n_in:d}, r = {r_in:.3f}')

        return phot_arr

    ##########
    # FSPL n=10, r=0.001
    ##########
    fspl_arr_n10_good = np.array([18.95006166, 18.94949701, 18.9472021, 18.93811389, 18.85723805,
                                  18.3424005, 18.85941388, 18.93826845, 18.9472443, 18.94950649])
    fspl_arr_n10 = test_fspl_once(radius, 10, fspl_arr_n10_good, mod='FSPL')

    ##########
    # FSPL n=50, r=0.001
    ##########
    fspl_arr_n50_good = np.array([18.88768975, 18.88712136, 18.88481127, 18.87566324, 18.79427668,
                                  18.27694883, 18.79646575, 18.87581881, 18.88485374, 18.8871309])
    fspl_arr_n50 = test_fspl_once(radius, 50, fspl_arr_n50_good, mod='FSPL')

    ##########
    # FSPL n=100, r=0.001
    ##########
    fspl_arr_n100_good = np.array([18.88575986, 18.88519136, 18.88288081, 18.87373098, 18.79232908,
                                   18.27492657, 18.79451856, 18.87388658, 18.88292329, 18.8852009])
    fspl_arr_n100 = test_fspl_once(radius, 100, fspl_arr_n100_good, mod='FSPL')

    ##########
    # FSPL n=10, r=1
    ##########
    fspl_arr_n10_r1_good = np.array([18.950062, 18.949497, 18.947202, 18.938114, 18.857238, 18.3424,
                                     18.859414, 18.938268, 18.947244, 18.949506])
    fspl_arr_n10_r1 = test_fspl_once(1, 10, fspl_arr_n10_r1_good, mod='FSPL')

    ##########
    # PSPL for run-time comparisons.
    ##########
    pspl_arr_good = np.array([18.8851265, 18.88452664, 18.88231504, 18.87282729, 18.79254205,
                              18.28992589, 18.79486318, 18.87299118, 18.88235892, 18.88453582])
    pspl_arr = test_fspl_once(-1, -1, pspl_arr_good, mod='PSPL')

    if plot:
        plt.figure(1)
        plt.clf()
        plt.plot(time_arr, fspl_arr_n100, '-', label='fspl, n=100')
        plt.plot(time_arr, fspl_arr_n50, '-', label='fspl, n=50')
        plt.plot(time_arr, fspl_arr_n10, '-', label='fspl, n=10')
        plt.plot(time_arr, pspl_arr, '-', label='pspl')
        plt.gca().invert_yaxis()
        plt.xlabel('Time (HJD)')
        plt.ylabel('Mag')
        plt.legend()

        plt.figure(2)
        plt.clf()
        plt.plot(time_arr, fspl_arr_n100 - pspl_arr, '-', label='fspl, n=100')
        plt.plot(time_arr, fspl_arr_n50 - pspl_arr, '-', label='fspl, n=50')
        plt.plot(time_arr, fspl_arr_n10 - pspl_arr, '-', label='fspl, n=10')
        plt.gca().invert_yaxis()
        plt.xlabel('Time (HJD)')
        plt.ylabel('FSPL-PSPL (Mag)')
        plt.legend()

        plt.figure(3)
        plt.clf()
        plt.plot(time_arr, fspl_arr_n10_r1 - fspl_arr_n10, '.')
        plt.title('Different radii')



    return


def test_FSPL_Phot_classes(plot=False):
    """
    Make sure can instantiate.
    """
    raL = 17.3 * 15.
    decL = -29.0
    t0 = 57000.0
    u0_amp = -0.2
    tE = 135.0
    piE_E = -0.1
    piE_N = 0.1
    radius = 1e-3
    b_sff = [1.1]
    mag_base = [19.0]
    
    # Array of times we will sample on.
    time_arr = np.linspace(t0 - 1000, t0 + 1000, 1000)

    def test_fspl_phot_once(r_in, n_in, phot_arr_good, mod='FSPL', tol=1e-3):
        # Make the model
        tstart = time.time()

        if mod == 'FSPL':
            tmp_mod = model.FSPL_Phot_Par_Param2(t0,
                                                 u0_amp,
                                                 tE,
                                                 piE_E,
                                                 piE_N,
                                                 b_sff,
                                                 mag_base,
                                                 radius,
                                                 raL=raL, decL=decL)
        else:
            tmp_mod = model.PSPL_Phot_Par_Param2(t0,
                                                 u0_amp,
                                                 tE,
                                                 piE_E,
                                                 piE_N,
                                                 b_sff,
                                                 mag_base,
                                                 raL=raL, decL=decL)

        # Generate photometry
        phot_arr = tmp_mod.get_photometry(time_arr)

        # When working, fetch 10 points, evenly distributed and
        # save as the "right answer" in the individual tests below.
        # print(repr(phot_arr[::100]))  # GET GOOD

        # Check against good data
        np.testing.assert_allclose(phot_arr[::100], phot_arr_good, rtol=tol)

        tdelta = time.time() - tstart

        print(f'Eval time for {mod:s}: {tdelta:.4f} sec, n = {n_in:d}, r = {r_in:.3f}')

        return phot_arr

    ##########
    # FSPL n=10, r=0.001
    ##########
    fspl_arr_n10_good = np.array([18.95006166, 18.94949701, 18.9472021, 18.93811389, 18.85723805,
                                  18.3424005, 18.85941388, 18.93826845, 18.9472443, 18.94950649])
    fspl_arr_n10 = test_fspl_phot_once(radius, 10, fspl_arr_n10_good, mod='FSPL')

    ##########
    # FSPL n=50, r=0.001
    ##########
    fspl_arr_n50_good = np.array([18.88768975, 18.88712136, 18.88481127, 18.87566324, 18.79427668,
                                  18.27694883, 18.79646575, 18.87581881, 18.88485374, 18.8871309])
    fspl_arr_n50 = test_fspl_phot_once(radius, 50, fspl_arr_n50_good, mod='FSPL')

    ##########
    # FSPL n=100, r=0.001
    ##########
    fspl_arr_n100_good = np.array([18.88575986, 18.88519136, 18.88288081, 18.87373098, 18.79232908,
                                   18.27492657, 18.79451856, 18.87388658, 18.88292329, 18.8852009])
    fspl_arr_n100 = test_fspl_phot_once(radius, 100, fspl_arr_n100_good, mod='FSPL')

    ##########
    # FSPL n=10, r=1
    ##########
    fspl_arr_n10_r1_good = np.array([18.950062, 18.949497, 18.947202, 18.938114, 18.857238, 18.3424,
                                     18.859414, 18.938268, 18.947244, 18.949506])
    fspl_arr_n10_r1 = test_fspl_phot_once(1, 10, fspl_arr_n10_r1_good, mod='FSPL')

    ##########
    # PSPL for run-time comparisons.
    ##########
    pspl_arr_good = np.array([18.8851265, 18.88452664, 18.88231504, 18.87282729, 18.79254205,
                              18.28992589, 18.79486318, 18.87299118, 18.88235892, 18.88453582])
    pspl_arr = test_fspl_phot_once(-1, -1, pspl_arr_good, mod='PSPL')

    if plot:
        plt.figure(1)
        plt.clf()
        plt.plot(time_arr, fspl_arr_n100, '-', label='fspl, n=100')
        plt.plot(time_arr, fspl_arr_n50, '-', label='fspl, n=50')
        plt.plot(time_arr, fspl_arr_n10, '-', label='fspl, n=10')
        plt.plot(time_arr, pspl_arr, '-', label='pspl')
        plt.gca().invert_yaxis()
        plt.xlabel('Time (HJD)')
        plt.ylabel('Mag')
        plt.legend()

        plt.figure(2)
        plt.clf()
        plt.plot(time_arr, fspl_arr_n100 - pspl_arr, '-', label='fspl, n=100')
        plt.plot(time_arr, fspl_arr_n50 - pspl_arr, '-', label='fspl, n=50')
        plt.plot(time_arr, fspl_arr_n10 - pspl_arr, '-', label='fspl, n=10')
        plt.gca().invert_yaxis()
        plt.xlabel('Time (HJD)')
        plt.ylabel('FSPL-PSPL (Mag)')
        plt.legend()

        plt.figure(3)
        plt.clf()
        plt.plot(time_arr, fspl_arr_n10_r1 - fspl_arr_n10, '.')
        plt.title('Different radii')

    return


def test_PSPL_PhotAstrom_LumLens_Par_Param1():
    """
    Make sure can instantiate
    """
    raL_in = 17.30 * 15.  # Bulge R.A.
    decL_in = -29.0
    mL_in = 1.0  # msun
    t0_in = 57100.0
    xS0_in = np.array([0.000, 0.088e-3])  # arcsec
    beta_in = 2.0  # mas  same as p=0.4
    muS_in = np.array([-5.0, 0.0])
    muL_in = np.array([5.0, 0.0])
    dL_in = 4000.0  # pc
    dS_in = 8000.0  # pc
    b_sff_in = 1.0
    mag_src_in = 19.0

    mod = model.PSPL_PhotAstrom_LumLens_Par_Param1(mL=mL_in,
                                                   t0=t0_in,
                                                   beta=beta_in,
                                                   dL=dL_in,
                                                   dL_dS=dL_in / dS_in,
                                                   xS0_E=xS0_in[0],
                                                   xS0_N=xS0_in[1],
                                                   muL_E=muL_in[0],
                                                   muL_N=muL_in[1],
                                                   muS_E=muS_in[0],
                                                   muS_N=muS_in[1],
                                                   raL=raL_in,
                                                   decL=decL_in,
                                                   b_sff=[b_sff_in],
                                                   mag_src=[mag_src_in])
    return


def test_PSPL_GP_MRO():
    """
    For the GP class, the PSPL_GP likelihood should be the one that's used.
    This test makes sure that it is the 2nd (index 1) list in the MRO.
    """
    pspl = [model.PSPL_Phot_Par_GP_Param1,
            model.PSPL_Phot_Par_GP_Param2,
            model.PSPL_Phot_Par_GP_Param1_2,
            model.PSPL_Phot_noPar_GP_Param1,
            model.PSPL_Phot_noPar_GP_Param2,
            model.PSPL_PhotAstrom_Par_GP_Param1,
            model.PSPL_PhotAstrom_Par_GP_Param2,
            model.PSPL_PhotAstrom_Par_LumLens_GP_Param1,
            model.PSPL_PhotAstrom_Par_LumLens_GP_Param2,
            model.PSPL_PhotAstrom_noPar_GP_Param1,
            model.PSPL_PhotAstrom_noPar_GP_Param2]

    for pp in pspl:
        print(pp)
        mro_list = pp.mro()
        pspl_gp_idx = [mro.__name__ for mro in mro_list].index('PSPL_GP')
        np.testing.assert_array_equal(pspl_gp_idx, 2)

    return


def test_PSBL_GP_MRO():
    """
    For the GP class, the PSPL_GP likelihood should be the one that's used.
    This test makes sure that it is the 2nd (index 1) list in the MRO.
    """
    psbl = [model.PSBL_PhotAstrom_noPar_GP_Param1,
            model.PSBL_PhotAstrom_Par_GP_Param1,
            model.PSBL_PhotAstrom_noPar_GP_Param2,
            model.PSBL_PhotAstrom_Par_GP_Param2,
            model.PSBL_Phot_noPar_GP_Param1,
            model.PSBL_Phot_Par_GP_Param1]

    for pp in psbl:
        print(pp)
        mro_list = pp.mro()
        pspl_gp_idx = [mro.__name__ for mro in mro_list].index('PSPL_GP')
        np.testing.assert_array_equal(pspl_gp_idx, 2)

    return


def test_ABC_MRO():
    classes_list = []
    data_classes = []
    parallax_classes = []
    gp_classes = []
    param_classes = []
    model_classes = []

    for name, obj in inspect.getmembers(model):
        if inspect.isclass(obj) and obj.__module__ == 'bagle.model':
            classes_list.append(obj)

            # First, figure out if it is a ModelClass
            mro_list = obj.mro()
            mro_names = [foo.__name__ for foo in mro_list]

            # Loop through the inherited objects and check for 
            if 'ModelClassABC' in mro_names and 'ModelClassABC' != obj.__name__:
                model_classes.append(obj.__name__)
            else:
                if ('PSPL_Parallax' in mro_names) or ('PSPL_noParallax' in mro_names):
                    parallax_classes.append(obj.__name__)

                if ('PSPL_Param' in mro_names):
                    param_classes.append(obj.__name__)

                if ('PSPL_GP' in mro_names):
                    gp_classes.append(obj.__name__)

                if ('PSPL' in mro_names):
                    data_classes.append(obj.__name__)

    # All ModelClass objects should probably have "_Param" in their name.
    for pcls in model_classes:
        assert '_Param' in pcls

    # All data classes should be hard-coded into this test.
    # But we will test such that new ones won't make this test fail. 
    data_classes_good = ['BSPL', 'BSPL_Phot', 'BSPL_PhotAstrom',
                         'FSPL', 'FSPL_Limb', 'FSPL_Phot', 'FSPL_PhotAstrom',
                         'PSBL', 'PSBL_Phot', 'PSBL_PhotAstrom', 'PSPL',
                         'PSPL_Astrom', 'PSPL_Phot', 'PSPL_PhotAstrom']
    for dcls in data_classes_good:
        assert dcls in data_classes

    assert gp_classes[0] == 'PSPL_GP'

    parallax_classes_good = ['BSPL_Parallax', 'BSPL_noParallax', 'FSPL_Limb_Parallax',
                             'FSPL_Limb_noParallax', 'FSPL_Parallax', 'FSPL_noParallax',
                             'PSBL_Parallax', 'PSBL_noParallax', 'PSPL_Parallax',
                             'PSPL_Parallax_LumLens', 'PSPL_Parallax_geoproj',
                             'PSPL_noParallax', 'PSPL_noParallax_LumLens']
    for pcls in parallax_classes_good:
        assert pcls in parallax_classes

    model_classes.sort()
    model_classes = model_classes[::-1]
    print(model_classes)

    return

def plot_BSBL(bsbl, t_obs):
    """
    Make some standard plots for PSBL.
    """
    images, amps = bsbl.get_all_arrays(t_obs, rescale=True)

    ##########
    # Photometry
    ##########
    phot = bsbl.get_photometry(t_obs, amp_arr=amps)

    # Plot the photometry
    plt.figure(1)
    plt.clf()
    plt.plot(t_obs, phot, 'r-')
    plt.ylabel('Photometry (mag)')
    plt.xlabel('Time (MJD)')
    plt.gca().invert_yaxis()

    ##########
    # Astrometry
    ##########
    if bsbl.astrometryFlag:
        # Find the points closest to t0
        t0idx = np.argmin(np.abs(t_obs - bsbl.t0))

        # Resolved astrometry - unlensed
        xL1, xL2 = bsbl.get_resolved_lens_astrometry(t_obs)
        xL1 *= 1e3
        xL2 *= 1e3
        xS_res_unlens = bsbl.get_resolved_astrometry_unlensed(t_obs) * 1e3
        xS1 = xS_res_unlens[:, 0, :]
        xS2 = xS_res_unlens[:, 1, :]

        # Resolved astrometry - lensed
        xS_res_lensed = bsbl.get_resolved_astrometry(t_obs,
                                                     image_arr=images,
                                                     amp_arr=amps) * 1e3

        # Unresolved astrometry - lensed and unlensed
        xS_unlens = bsbl.get_astrometry_unlensed(t_obs) * 1e3
        xS_lensed = bsbl.get_astrometry(t_obs, image_arr=images, amp_arr=amps) * 1e3

        dxS = (xS_lensed - xS_unlens)

        # Resolved: Plot the positions of everything
        plt.figure(2)
        plt.clf()
        plt.plot(xS1[:, 0], xS1[:, 1], ls='--', color='blue')
        plt.plot(xS2[:, 0], xS2[:, 1], ls='--', color='darkblue')
        plt.plot(xL1[:, 0], xL1[:, 1], ls='--', color='green')
        plt.plot(xL2[:, 0], xL2[:, 1], ls='--', color='darkgreen')

        plt.plot(xS1[t0idx, 0], xS1[t0idx, 1], 'b+', mfc='blue',
                 mec='blue',
                 label='S1')
        plt.plot(xS2[t0idx, 0], xS2[t0idx, 1], 'bx', mfc='darkblue',
                 mec='darkblue',
                 label='S2')
        plt.plot(xL1[t0idx, 0], xL1[t0idx, 1], 'gs', mfc='green',
                 mec='green',
                 label='L1')
        plt.plot(xL2[t0idx, 0], xL2[t0idx, 1], 'gs', mfc='none',
                 mec='green',
                 label='L2')

        colors = ['mediumvioletred', 'purple', 'indigo', 'deeppink', 'darkviolet']
        for ii in range(xS_res_lensed.shape[2]):
            label_S1  = ''
            label_S2 = ''
            if ii == 0:
                label_S1 = 'S1_lensed images'
                label_S2 = 'S2_lensed images'
            plt.plot(xS_res_lensed[:, 0, ii, 0], xS_res_lensed[:, 0, ii, 1],
                     ls='none', marker='.', ms=1, color=colors[0], alpha=0.7)
            plt.plot(xS_res_lensed[:, 1, ii, 0], xS_res_lensed[:, 1, ii, 1],
                     ls='none', marker='.', ms=1, color=colors[1], alpha=0.7)
            plt.plot(xS_res_lensed[t0idx, 0, ii, 0], xS_res_lensed[t0idx, 0, ii, 1],
                     ls='none', marker='+', color=colors[0], alpha=0.7,
                     label=label_S1)
            plt.plot(xS_res_lensed[t0idx, 1, ii, 0], xS_res_lensed[t0idx, 1, ii, 1],
                     ls='none', marker='x', color=colors[1], alpha=0.7,
                     label=label_S2)

        plt.legend()
        plt.gca().invert_xaxis()
        plt.xlabel('R.A. (mas)')
        plt.ylabel('Dec. (mas)')
        plt.title('Resolved Astrometry')

        # Unresolved: Plot the unresolved astrometry, both unlensed and lensed.
        plt.figure(3)
        plt.clf()
        plt.plot(xS_unlens[:, 0], xS_unlens[:, 1], ls='--', color='blue',
                 label='xS, unlensed')
        plt.plot(xS_lensed[:, 0], xS_lensed[:, 1], ls='-', color='blue',
                 label='xS, lensed')
        plt.plot(xS_unlens[t0idx, 0], xS_unlens[t0idx, 1], 'bx',
                 label='xS, unlensed')
        plt.plot(xS_lensed[t0idx, 0], xS_lensed[t0idx, 1], 'bo',
                 label='xS, lensed')

        plt.legend()
        plt.gca().invert_xaxis()
        plt.xlabel('R.A. (mas)')
        plt.ylabel('Dec. (mas)')
        plt.title('Resolved Astrometry')

        # Check just the astrometric shift part.
        plt.figure(4)
        plt.clf()
        plt.plot(t_obs, dxS[:, 0], 'r--', label='R.A.')
        plt.plot(t_obs, dxS[:, 1], 'b--', label='Dec.')
        plt.legend(fontsize=10)
        plt.ylabel('Astrometric Shift (mas)')
        plt.xlabel('Time (MJD)')

        plt.figure(5)
        plt.clf()
        plt.plot(dxS[:, 0], dxS[:, 1], 'r-')
        plt.axhline(0, linestyle='--')
        plt.axvline(0, linestyle='--')
        plt.gca().invert_xaxis()
        plt.xlabel('Shift RA (mas)')
        plt.ylabel('Shift Dec (mas)')
        plt.axis('equal')

        print('Einstein radius: ', bsbl.thetaE_amp)
        print('Einstein crossing time: ', bsbl.tE)

    return

def test_BSBL_PhotAstrom_Par_Param1():
    """
    General testing of BSBL... caustic crossings.
    """

    # NOTE this gives the same model as in test_BSBL_Phot_noPar_Param1()
    raL = 259.5
    decL = -28.5
    mLp = 10.0
    mLs = 3.0
    t0 = 57000
    xS0_E = 0.001 # mas
    xS0_N = 0.0
    beta = 1.0 # mas
    muL_E = 0.0  # mas/yr
    muL_N = 0.0  # mas/yr
    muS_E = -3.0  # mas/yr
    muS_N = 0.0  # mas/yr
    dL = 4000 # pc
    dS = 8000 # pc
    sepL = 3.0  # in mas
    alphaL = -35.0  # PA of binary on the sky
    sepS = 0.5 # mas
    alphaS = 0.0 # PA of source binary on the sky
    mag_src_pri = np.array([18.0])
    mag_src_sec = np.array([19.0])
    b_sff = np.array([1.0])
    
    # phi_piE = np.degrees(np.arctan2(piE_N, piE_E))  # PA of muRel on the sky
    # phi = alpha - phi_piE  # relative angle between binary and muRel.
    # print('alpha = ', alpha, ' deg')
    # print('phi_piE = ', phi_piE, ' deg')
    # print('phi = ', phi, ' deg')
    # 
    bsbl_n = model.BSBL_PhotAstrom_noPar_Param1(mLp, mLs, t0, xS0_E, xS0_N,
                                                beta, muL_E, muL_N, muS_E, muS_N,
                                                dL, dS, sepL, alphaL, sepS, alphaS,
                                                mag_src_pri, mag_src_sec, b_sff,
                                                raL=raL, decL=decL,
                                                root_tol=1e-4)
    bsbl_p = model.BSBL_PhotAstrom_Par_Param1(mLp, mLs, t0, xS0_E, xS0_N,
                                              beta, muL_E, muL_N, muS_E, muS_N,
                                              dL, dS, sepL, alphaL, sepS, alphaS,
                                              mag_src_pri, mag_src_sec, b_sff,
                                              raL=raL, decL=decL,
                                              root_tol=1e-4)
    t_obs = np.arange(56000.0, 58000.0, 1)

    plot_BSBL(bsbl_p, t_obs)

    # Check that we have some extreme magnifications since this
    # is caustic crossing.
    phot1 = bsbl_n.get_photometry(t_obs)
    phot2 = bsbl_p.get_photometry(t_obs)

    assert phot1.min() < 18.1
    assert phot2.min() < 18.1

    ##########
    # Recalculate u calculation from complex_pos() to debug.
    ##########
    # Calculate the position of the source w.r.t. lens (in Einstein radii)
    # Distance along muRel direction
    tau = (t_obs - bsbl_p.t0) / bsbl_p.tE
    tau = tau.reshape(len(tau), 1)

    # Distance along u0 direction -- always constant with time.
    u0 = bsbl_p.u0.reshape(1, len(bsbl_p.u0))
    thetaE_hat = bsbl_p.thetaE_hat.reshape(1, len(bsbl_p.thetaE_hat))

    # Total distance
    u = u0 + tau * thetaE_hat

    # Incorporate parallax
    parallax_vec = model.parallax_in_direction(bsbl_p.raL, bsbl_p.decL, t_obs)
    u -= bsbl_p.piE_amp * parallax_vec

    t0dx = np.argmin(np.abs(tau))
    print('u = ')
    print(u[t0dx - 5:t0dx + 5, :])

    w, z1, z2 = bsbl_p.get_complex_pos(t_obs)
    images_p, amps_p = bsbl_p.get_all_arrays(t_obs)
    amp_arr_msk = np.ma.masked_invalid(amps_p)
    amp = np.sum(amp_arr_msk, axis=1)

    print('w: ')
    print(w[t0dx - 5:t0dx + 5])
    print('z1: ')
    print(z1[t0dx - 5:t0dx + 5])
    print('z2: ')
    print(z2[t0dx - 5:t0dx + 5])

    return


def test_PSPL_Phot_Param2_vs_Param3():
    """
    Compare models with parameters vs. log parameters. 
    """
    raL = (17.0 + (49.0 / 60.) + (51.38 / 3600.0)) * 15.0  # degrees
    decL = -35 + (22.0 / 60.0) + (28.0 / 3600.0)
    t0 = 56026.03
    u0_amp = -0.222
    tE = 135.0
    log_tE = np.log10(tE)
    piE_E = -0.058
    piE_N = 0.11
    piE_amp = np.linalg.norm([piE_E, piE_N])
    log_piE = np.log10(piE_amp)
    phi_muRel = np.rad2deg(np.arctan2(piE_E, piE_N))
    b_sff = [1.1]
    mag_base = [19.0]

    t_mod = np.arange(t0 - 1000, t0 + 1000, 2)

    ##########
    # Compare two models with same properties
    # recast in different parameters.
    ##########
    mod2 = model.PSPL_Phot_Par_Param2(t0,
                                      u0_amp,
                                      tE,
                                      piE_E,
                                      piE_N,
                                      [1.0],
                                      mag_base,
                                      raL=raL, decL=decL)
    mod3 = model.PSPL_Phot_Par_Param3(t0,
                                      u0_amp,
                                      log_tE,
                                      log_piE,
                                      phi_muRel,
                                      [1.0],
                                      mag_base,
                                      raL=raL, decL=decL)
    
    I_mod2 = mod2.get_photometry(t_mod)
    I_mod3 = mod3.get_photometry(t_mod)

    plt.figure(1)
    plt.clf()
    plt.plot(t_mod, I_mod2, 'r-', label='Param2')
    plt.plot(t_mod, I_mod3, 'b-', label='Param3')
    plt.xlabel('Time (MJD)')
    plt.ylabel('Mag')
    plt.gca().invert_yaxis()
    plt.legend()

    return

def test_jwst_parallax_bulge1():
    # Scenario from Belokurov and Evans 2002 (Figure 1)
    raL = 17.5 * 15.0  # in degrees
    decL = -30.0
    obsLocation = 'jwst'
    mL = 10.0  # msun
    t0 = 60218.0
    xS0 = np.array([0.000, 0.000])
    beta = 3.0  # mas
    muS = np.array([-4.0, -4.0])
    muL = np.array([-6.0, -10.0])
    dL = 3000.0
    dS = 6000.0
    b_sff = 1.0
    imag = 19.0

    run_test_pspl_satellite_parallax(raL, decL, obsLocation,
                                     mL, t0, xS0, beta, muS, muL, dL, dS,
                                     b_sff, imag, outdir='tests/test_pspl_par_jwst_bulge1/')

    return

def test_spitzer_parallax_bulge1():
    # Scenario from Belokurov and Evans 2002 (Figure 1)
    raL = 17.5 * 15.0  # in degrees
    decL = -30.0
    obsLocation = 'spitzer'
    mL = 10.0  # msun
    t0 = 60218.0
    xS0 = np.array([0.000, 0.000])
    beta = 3.0  # mas
    muS = np.array([-4.0, -4.0])
    muL = np.array([-6.0, -10.0])
    dL = 3000.0
    dS = 6000.0
    b_sff = 1.0
    imag = 19.0

    run_test_pspl_satellite_parallax(raL, decL, obsLocation,
                                     mL, t0, xS0, beta, muS, muL, dL, dS,
                                     b_sff, imag, outdir='tests/test_pspl_par_spitzer_bulge1/')

    return

def run_test_pspl_satellite_parallax(raL, decL, obsLocation,
                                     mL, t0, xS0, beta, muS, muL, dL, dS,
                                     b_sff, mag_src, outdir=''):
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    # Earth parallax
    pspl_e = model.PSPL_PhotAstrom_Par_Param1(mL,
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
                                              [mag_src],
                                              raL=raL,
                                              decL=decL)
    
    print('pspl_e.u0', pspl_e.u0)
    print('pspl_e.muS', pspl_e.muS)
    print('pspl_e.u0_hat', pspl_e.u0_hat)
    print('pspl_e.thetaE_hat', pspl_e.thetaE_hat)

    # Satellite parallax
    pspl_s = model.PSPL_PhotAstrom_Par_Param1(mL,
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
                                              [mag_src],
                                              raL=raL,
                                              decL=decL,
                                              obsLocation=obsLocation)

    t = np.arange(t0 - 600, t0 + 600, 1)
    dt = t - pspl_e.t0

    A_e = pspl_e.get_amplification(t)
    A_s = pspl_s.get_amplification(t)

    xS_e = pspl_e.get_astrometry(t)
    xS_s_unlens = pspl_s.get_astrometry_unlensed(t)
    xS_s_lensed = pspl_s.get_astrometry(t)
    xL_s = pspl_s.get_lens_astrometry(t)

    # make sure the two light curves and trajectories are different.
    # assert np.abs(A_e - A_s).max() > 0.1  # mag
    # assert np.abs(xS_e - xS_s_lensed).max() > 0.001  # arcsec

    # Plot the amplification
    fig1 = plt.figure(1)
    plt.clf()
    f1_1 = fig1.add_axes((0.20, 0.3, 0.75, 0.6))
    plt.plot(dt, 2.5 * np.log10(A_e), 'b-', label='Earth parallax')
    plt.plot(dt, 2.5 * np.log10(A_s), 'r-', label=f'{obsLocation} parallax')
    plt.legend(fontsize=10)
    plt.ylabel('2.5 * log(A)')
    f1_1.set_xticklabels([])

    f2_1 = fig1.add_axes((0.20, 0.1, 0.75, 0.2))
    plt.plot(dt, 2.5 * (np.log10(A_s) - np.log10(A_e)), 'k-',
             label='Par - No par')
    plt.axhline(0, linestyle='--', color='k')
    plt.legend(fontsize=10)
    plt.ylabel('Diff')
    plt.xlabel('t - t0 (MJD)')

    plt.savefig(outdir + 'amp_v_time.png')
    print("save to " + outdir)

    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_e[:, 0] * 1e3, xS_e[:, 1] * 1e3, 'r--',
             mfc='none', mec='red', label='Src, Earth parallax model')
    plt.plot(xS_s_unlens[:, 0] * 1e3, xS_s_unlens[:, 1] * 1e3, 'b--',
             mfc='none', mec='blue',
             label=f'Src, {obsLocation} parallax model, unlensed')
    plt.plot(xL_s[:, 0] * 1e3, xL_s[:, 1] * 1e3, 'k--',
             mfc='none', mec='grey', label='Lens')
    plt.plot(xS_s_lensed[:, 0] * 1e3, xS_s_lensed[:, 1] * 1e3, 'b-',
             label=f'Src, {obsLocation} parallax model, lensed')
    plt.legend(fontsize=10)
    plt.gca().invert_xaxis()
    plt.xlabel('R.A. (mas)')
    plt.ylabel('Dec. (mas)')
    plt.axis('equal')
    lim = 20
    print('LIM = ', lim)
    # plt.xlim(lim, -lim) # arcsec
    # plt.ylim(-lim, lim)
    # plt.axis('tight')
    # plt.xlim(0.7, -0.7)
    # plt.ylim(-0.7, 0.7)
    plt.savefig(outdir + 'on_sky.png')

    # Check just the astrometric shift part.
    shift_e = pspl_e.get_centroid_shift(t)  # mas
    shift_s = (xS_s_lensed - xS_s_unlens) * 1e3  # mas
    shift_e_amp = np.linalg.norm(shift_e, axis=1)
    shift_s_amp = np.linalg.norm(shift_s, axis=1)

    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.20, 0.3, 0.75, 0.6))
    plt.plot(dt, shift_e_amp, 'r--', label='Earth parallax model')
    plt.plot(dt, shift_s_amp, 'b--', label=f'{obsLocation} parallax model')
    plt.ylabel('Astrometric Shift (mas)')
    plt.legend(fontsize=10)
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.20, 0.1, 0.75, 0.2))
    plt.plot(dt, shift_s_amp - shift_e_amp, 'k-', label='Par - No par')
    plt.legend(fontsize=10)
    plt.axhline(0, linestyle='--', color='k')
    plt.ylabel('Diff (mas)')
    plt.xlabel('t - t0 (MJD)')

    plt.savefig(outdir + 'shift_amp_v_t.png')

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_e[:, 0], shift_e[:, 1], 'r-', label='Earth parallax')
    plt.plot(shift_s[:, 0], shift_s[:, 1], 'b-', label=f'{obsLocation} parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend(fontsize=10)
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')
    plt.savefig(outdir + 'shift_on_sky.png')

    print('Einstein radius: ', pspl_e.thetaE_amp, pspl_s.thetaE_amp)
    print('Einstein crossing time: ', pspl_e.tE, pspl_e.tE)

    return

def test_spitzer_zang2020(plot=False):
    # Scenario from Zang et al. 2020
    # Target: OB171254: 0,+ solution (see Table 1)
    raL = (17. + 57./60.)  * 15.0  # in degrees
    decL = -(27. + 13./60.)
    obsLocation = 'spitzer'
    t0_geotr = 57952.2519
    u0_geotr = 0.0003
    tE_geotr = 15.43
    piEN_geotr = 0.0203
    piEE_geotr = 0.0368
    t0par = t0_geotr
    
    mag_src = 18.53
    mag_blend = 21.32

    f_src = 10**(mag_src / -2.5)
    f_blend = 10**(mag_blend / -2.5)
    f_base = f_src + f_blend
    
    b_sff = f_src / f_base
    mag_base = -2.5 * np.log10(f_base)

    out = fc.convert_helio_geo_phot(raL, decL,
                                    t0_geotr, u0_geotr, tE_geotr,
                                    piEE_geotr, piEN_geotr, t0par,
                                    in_frame='geo',
                                    murel_in='LS', murel_out='SL',
                                    coord_in='tb', coord_out='EN',
                                    plot=False)

    t0_helio = out[0]
    u0_helio = out[1]
    tE_helio = out[2]
    piEE_helio = out[3]
    piEN_helio = out[4]

    outdir = 'tests/test_pspl_par_spitzer_zang2020/'

    # Make Earth and Spitzer observations and make plots.
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    # Earth parallax
    pspl_e = model.PSPL_Phot_Par_Param1(t0_helio,
                                        u0_helio,
                                        tE_helio,
                                        piEE_helio,
                                        piEN_helio,
                                        [b_sff],
                                        [mag_src],
                                        raL=raL,
                                        decL=decL,
                                        obsLocation='earth')
    # Satellite parallax
    pspl_s = model.PSPL_Phot_Par_Param1(t0_helio,
                                        u0_helio,
                                        tE_helio,
                                        piEE_helio,
                                        piEN_helio,
                                        [b_sff],
                                        [mag_src],
                                        raL=raL,
                                        decL=decL,
                                        obsLocation=obsLocation)

    t = np.arange(t0_geotr - 30, t0_geotr + 30, 0.5)
    dt = t - pspl_e.t0

    A_e = pspl_e.get_amplification(t)
    A_s = pspl_s.get_amplification(t)

    m_e = pspl_e.get_photometry(t)
    m_s = pspl_s.get_photometry(t)

    # make sure the two light curves and trajectories are different.
    assert np.abs(A_e - A_s).max() > 0.1  # mag
    # assert np.abs(xS_e - xS_s_lensed).max() > 0.001  # arcsec

    if plot:
        # Plot the amplification
        fig1 = plt.figure(1)
        plt.clf()
        f1_1 = fig1.add_axes((0.20, 0.3, 0.75, 0.6))
        plt.plot(dt, 2.5 * np.log10(A_e), 'b-', label='Earth parallax')
        plt.plot(dt, 2.5 * np.log10(A_s), 'r-', label=f'{obsLocation} parallax')
        plt.legend(fontsize=10)
        plt.ylabel('2.5 * log(A)')
        f1_1.set_xticklabels([])

        f2_1 = fig1.add_axes((0.20, 0.1, 0.75, 0.2))
        plt.plot(dt, 2.5 * (np.log10(A_s) - np.log10(A_e)), 'k-',
                 label='Par - No par')
        plt.axhline(0, linestyle='--', color='k')
        plt.legend(fontsize=10)
        plt.ylabel('Diff')
        plt.xlabel('t - t0 (MJD)')

        plt.savefig(outdir + 'amp_v_time.png')

        # Plot the magnitude
        fig2 = plt.figure(2)
        plt.clf()
        f2_1 = fig2.add_axes((0.20, 0.3, 0.75, 0.6))
        plt.plot(dt, m_e, 'b-', label='Earth parallax')
        plt.plot(dt, m_s, 'r-', label=f'{obsLocation} parallax')
        plt.legend(fontsize=10)
        plt.ylabel('mag')
        plt.gca().invert_yaxis()
        f2_1.set_xticklabels([])

        f2_2 = fig2.add_axes((0.20, 0.1, 0.75, 0.2))
        plt.plot(dt, m_s - m_e, 'k-',
                 label=f'Earth - {obsLocation}')
        plt.axhline(0, linestyle='--', color='k')
        plt.legend(fontsize=10)
        plt.ylabel('Diff')
        plt.xlabel('t - t0 (MJD)')
        plt.gca().invert_yaxis()

        plt.savefig(outdir + 'mag_v_time.png')

    #print("save to " + outdir)

    return


def test_spitzer_shvartzvald2019(plot=False):
    # Scenario from Shvartzvald et al. 2019
    # Target: OB170896: +,+ solution (see Table 1)
    raL = (17. + 57./60.)  * 15.0  # in degrees
    decL = -(27. + 13./60.)
    obsLocation = 'spitzer'
    t0_geotr = 57911.05582
    u0_geotr = 0.0039
    tE_geotr = 14.883
    piEN_geotr = -0.779
    piEE_geotr = -0.615
    t0par = t0_geotr

    # Guessed these values -- not reported in table.
    mag_src = 17.9
    mag_blend = 21.0

    # Compare to Figure 1
    f_src = 10**(mag_src / -2.5)
    f_blend = 10**(mag_blend / -2.5)
    f_base = f_src + f_blend
    
    b_sff = f_src / f_base
    mag_base = -2.5 * np.log10(f_base)

    out = fc.convert_helio_geo_phot(raL, decL,
                                    t0_geotr, u0_geotr, tE_geotr,
                                    piEE_geotr, piEN_geotr, t0par,
                                    in_frame='geo',
                                    murel_in='LS', murel_out='SL',
                                    coord_in='tb', coord_out='EN',
                                    plot=False)

    t0_helio = out[0]
    u0_helio = out[1]
    tE_helio = out[2]
    piEE_helio = out[3]
    piEN_helio = out[4]

    outdir = 'tests/test_pspl_par_spitzer_shvartzvald2019/'

    # Make Earth and Spitzer observations and make plots.
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    # Earth parallax
    pspl_e = model.PSPL_Phot_Par_Param1(t0_helio,
                                        u0_helio,
                                        tE_helio,
                                        piEE_helio,
                                        piEN_helio,
                                        [b_sff],
                                        [mag_src],
                                        raL=raL,
                                        decL=decL,
                                        obsLocation='earth')
    # Satellite parallax
    pspl_s = model.PSPL_Phot_Par_Param1(t0_helio,
                                        u0_helio,
                                        tE_helio,
                                        piEE_helio,
                                        piEN_helio,
                                        [b_sff],
                                        [mag_src],
                                        raL=raL,
                                        decL=decL,
                                        obsLocation=obsLocation)

    t = np.arange(t0_geotr - 30, t0_geotr + 30, 0.5)
    dt = t - pspl_e.t0

    A_e = pspl_e.get_amplification(t)
    A_s = pspl_s.get_amplification(t)

    m_e = pspl_e.get_photometry(t)
    m_s = pspl_s.get_photometry(t)

    # make sure the two light curves and trajectories are different.
    assert np.abs(A_e - A_s).max() > 0.1  # mag
    # assert np.abs(xS_e - xS_s_lensed).max() > 0.001  # arcsec

    if plot:

        # Plot the amplification
        fig1 = plt.figure(1)
        plt.clf()
        f1_1 = fig1.add_axes((0.20, 0.3, 0.75, 0.6))
        plt.plot(dt, 2.5 * np.log10(A_e), 'b-', label='Earth1 parallax')
        plt.plot(dt, 2.5 * np.log10(A_s), 'r-', label=f'{obsLocation} parallax')
        plt.legend(fontsize=10)
        plt.ylabel('2.5 * log(A)')
        f1_1.set_xticklabels([])

        f2_1 = fig1.add_axes((0.20, 0.1, 0.75, 0.2))
        plt.plot(dt, 2.5 * (np.log10(A_s) - np.log10(A_e)), 'k-',
                 label='Par - No par')
        plt.axhline(0, linestyle='--', color='k')
        plt.legend(fontsize=10)
        plt.ylabel('Diff')
        plt.xlabel('t - t0 (MJD)')

        plt.savefig(outdir + 'amp_v_time.png')

        # Plot the magnitude
        fig2 = plt.figure(2)
        plt.clf()
        f2_1 = fig2.add_axes((0.20, 0.3, 0.75, 0.6))
        plt.plot(dt, m_e, 'b-', label='Earth1 parallax')
        plt.plot(dt, m_s, 'r-', label=f'{obsLocation} parallax')
        plt.legend(fontsize=10)
        plt.ylabel('mag')
        plt.gca().invert_yaxis()
        f2_1.set_xticklabels([])

        f2_2 = fig2.add_axes((0.20, 0.1, 0.75, 0.2))
        plt.plot(dt, m_s - m_e, 'k-',
                 label=f'Earth - {obsLocation}')
        plt.axhline(0, linestyle='--', color='k')
        plt.legend(fontsize=10)
        plt.ylabel('Diff')
        plt.xlabel('t - t0 (MJD)')
        plt.gca().invert_yaxis()

        plt.savefig(outdir + 'mag_v_time.png')

    return

