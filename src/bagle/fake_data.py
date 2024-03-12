from scipy.signal import find_peaks
from astropy.io import fits
from astropy import constants as c
from astropy import units as u
import numpy as np
import pylab as plt
from astropy.table import Table
import os
from bagle import model
from bagle import model_fitter
from bagle import plot_models
import time

from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation, spherical_to_cartesian, cartesian_to_spherical
from astropy.coordinates import get_body_barycentric, get_body, get_moon, get_body_barycentric_posvel

# Always generate the same fake data.
np.random.seed(0)


def fake_lightcurve_parallax_bulge(outdir='./casey_testing_stuff/'):
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

    fake_lightcurve_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in, beta_in,
                             muS_in, muL_in, dL_in, dS_in, b_sff_in,
                             mag_src_in,
                             outdir=outdir)

    return


def fake_lightcurve_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in, beta_in,
                             muS_in, muL_in, dL_in, dS_in, b_sff_in,
                             mag_src_in,
                             outdir=''):
    if (outdir != '') and (outdir != None):
        os.makedirs(outdir, exist_ok=True)

    pspl_par_in = model.PSPL_PhotAstrom_Par_Param1(mL_in,
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

    pspl_in = model.PSPL_PhotAstrom_noPar_Param1(mL_in,
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
                                                 [mag_src_in])

    # Simulate    
    t = np.linspace(55000, 59000, 2000)

    imag_obs_par = pspl_par_in.get_photometry(t)
    pos_obs_par = pspl_par_in.get_astrometry(t)

    imag_obs = pspl_in.get_photometry(t)
    pos_obs = pspl_in.get_astrometry(t)

    fig = plt.figure(1)
    plt.clf()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.plot(t, imag_obs_par)
    ax1.plot(t, imag_obs)
    ax2.plot(t, imag_obs_par - imag_obs)
    plt.xlabel('time')
    plt.ylabel('mag')
    plt.show()

    plt.savefig(outdir + 'fig1.png')

    fig = plt.figure(2)
    plt.clf()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.plot(t, pos_obs_par[:, 0])
    ax1.plot(t, pos_obs[:, 0])
    ax2.plot(t, pos_obs_par[:, 0] - pos_obs[:, 0])
    plt.xlabel('time')
    plt.ylabel('pos, 0')
    plt.show()

    plt.savefig(outdir + 'fig2.png')

    fig = plt.figure(3)
    plt.clf()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.plot(t, pos_obs_par[:, 1])
    ax1.plot(t, pos_obs[:, 1])
    ax2.plot(t, pos_obs_par[:, 1] - pos_obs[:, 1])
    plt.xlabel('time')
    plt.ylabel('pos, 1')
    plt.show()

    plt.savefig(outdir + 'fig3.png')

    return

def fake_data_parallax_bulge(outdir='test_mnest_bulge/'):
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
    b_sff = 1.0
    imag_in = 19.0

    data, params = fake_data_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in,
                                      beta_in,
                                      muS_in, muL_in, dL_in, dS_in, b_sff,
                                      imag_in, outdir=outdir, target='Bulge', noise=False)

    return data, params


def fake_data_parallax_lmc(outdir='test_mnest_lmc/'):
    raL_in = 80.89375  # LMC R.A.
    decL_in = -29.0  # LMC Dec. This is the sin \beta = -0.99 where \beta = ecliptic lat
    mL_in = 10.0  # msun
    t0_in = 57000.0
    xS0_in = np.array([0.000, 0.088e-3])  # arcsec
    beta_in = 2.0  # mas  same as p=0.4
    # muS_in = np.array([-2.0, 1.5])
    # muL_in = np.array([0.0, 0.0])
    muS_in = np.array([-5.0, 0.0])
    muL_in = np.array([0.0, 0.0])
    dL_in = 4000.0  # pc
    dS_in = 8000.0  # pc
    b_sff = 1.0
    imag_in = 19.0

    data, params = fake_data_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in,
                                      beta_in,
                                      muS_in, muL_in, dL_in, dS_in, b_sff,
                                      imag_in, outdir=outdir, target='LMC')

    return data, params


def fake_data_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in, beta_in,
                       muS_in, muL_in, dL_in, dS_in, b_sff_in, mag_src_in,
                       outdir='', target='Unknown', noise=True,
                       obsLocation='earth'):
    pspl_par_in = model.PSPL_PhotAstrom_Par_Param1(mL_in,
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
                                                   b_sff=[b_sff_in],
                                                   mag_src=[mag_src_in],
                                                   raL=raL_in,
                                                   decL=decL_in)

    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(56000, 58000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

        ast_win = 120.0
        ast_start = (365.25 - ast_win) / 2.0
        t_ast_new = np.arange(year_start + ast_start,
                              year_start + ast_start + ast_win, 14)
        t_ast = np.concatenate([t_ast, t_ast_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = pspl_par_in.get_photometry(t_phot)
    imag_obs_err = np.zeros(len(t_phot))
    if noise:
        flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
        flux_obs_err = flux_obs ** 0.5
        flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
        imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
        imag_obs_err = 1.087 / flux_obs_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs. 
    if noise:
        pos_obs_tmp = pspl_par_in.get_astrometry(t_ast)
        pos_obs_err = np.ones((len(t_ast), 2), dtype=float) * 0.01 * 1e-3
        pos_obs = pos_obs_tmp + pos_obs_err * np.random.randn(len(t_ast), 2)
    else: 
        pos_obs = pspl_par_in.get_astrometry(t_ast)
        pos_obs_err = np.zeros((len(t_ast), 2))
    

    data = {}
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs
    data['mag_err1'] = imag_obs_err

    data['phot_files'] = ['fake_data_parallax_phot1']
    data['ast_files'] = ['fake_data_parallax_ast1']

    data['t_ast1'] = t_ast
    data['xpos1'] = pos_obs[:, 0]
    data['ypos1'] = pos_obs[:, 1]
    data['xpos_err1'] = pos_obs_err[:, 0]
    data['ypos_err1'] = pos_obs_err[:, 1]
    
    data['raL'] = raL_in
    data['decL'] = decL_in
    data['obsLocation'] = obsLocation
    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'

    params = {}
    params['raL'] = raL_in
    params['decL'] = decL_in
    params['obsLocation'] = obsLocation
    params['mL'] = mL_in
    params['t0'] = t0_in
    params['xS0_E'] = xS0_in[0]
    params['xS0_N'] = xS0_in[1]
    params['beta'] = beta_in
    params['muS_E'] = muS_in[0]
    params['muS_N'] = muS_in[1]
    params['muL_E'] = muL_in[0]
    params['muL_N'] = muL_in[1]
    params['dL'] = dL_in
    params['dS'] = dS_in
    params['b_sff'] = b_sff_in
    params['mag_src'] = mag_src_in

    # Extra parameters
    params['dL_dS'] = params['dL'] / params['dS']
    params['tE'] = pspl_par_in.tE
    params['thetaE'] = pspl_par_in.thetaE_amp
    params['piE_E'] = pspl_par_in.piE[0]
    params['piE_N'] = pspl_par_in.piE[1]
    params['u0_amp'] = pspl_par_in.u0_amp
    params['muRel_E'] = pspl_par_in.muRel[0]
    params['muRel_N'] = pspl_par_in.muRel[1]

#    model_fitter.plot_photometry(data, pspl_par_in, dense_time=True)
#    plt.figure(1)
#    plt.title('Input Data and Model')
#    plt.savefig(outdir + 'fake_data_phot.png')
#
#    model_fitter.plot_astrometry(data, pspl_par_in, dense_time=True)
#    plt.figure(2)
#    plt.title('Input Data and Model')
#    plt.savefig(outdir + 'fake_data_ast.png')
#
#    plt.figure(3)
#    plt.title('Input Data and Model')
#    plt.savefig(outdir + 'fake_data_t_vs_E.png')
#
#    plt.figure(4)
#    plt.title('Input Data and Model')
#    plt.savefig(outdir + 'fake_data_t_vs_N.png')

    return data, params


def fake_data1(beta_sign=-1, plot=False, verbose=False):
    # Input parameters
    mL_in = 10.0  # msun
    t0_in = 57000.00
    xS0_in = np.array([0.000, 0.000])
    beta_in = beta_sign * 0.4  # Einstein radii
    muL_in = np.array([0.0, -7.0])  # Strong
    # muL_in = np.array([-7.0, 0.0])  # Weak
    muS_in = np.array([1.5, -0.5])  # mas/yr
    dL_in = 4000.0
    dS_in = 8000.0
    b_sff_in = 1.0
    mag_src_in = 19.0

    pspl_in = model.PSPL_PhotAstrom_noPar_Param1(mL_in,
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
                                                 [mag_src_in])

    if verbose:
        print('Photometry Parameters: ')
        print('t0 = ', pspl_in.t0)
        print('u0 = ', pspl_in.u0_amp)
        print('tE = ', pspl_in.tE)
        print('piE_E = ', pspl_in.piE[0])
        print('piE_N = ', pspl_in.piE[1])
        print('b_sff = ', pspl_in.b_sff)
        print('mag_src = ', pspl_in.mag_src)

        print('Astrometry Parameters: ')
        print('mL = ', pspl_in.t0)
        print('beta = ', pspl_in.u0_amp)
        print('dL = ', pspl_in.tE)
        print('dS = ', pspl_in.piE[0])
        print('xS0_E = ', pspl_in.xS0[0])
        print('xS0_N = ', pspl_in.xS0[1])
        print('muL_E = ', pspl_in.muL[0])
        print('muL_N = ', pspl_in.muL[1])
        print('muS_E = ', pspl_in.muS[0])
        print('muS_N = ', pspl_in.muS[1])
        print('muRel_E = ', pspl_in.muRel[0])
        print('muRel_N = ', pspl_in.muRel[1])
        
    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(54000, 60000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

        ast_win = 120.0
        ast_start = (365.25 - ast_win) / 2.0
        t_ast_new = np.arange(year_start + ast_start,
                              year_start + ast_start + ast_win, 14)
        t_ast = np.concatenate([t_ast, t_ast_new])

    t_mod = np.arange(t_phot.min(), t_phot.max(), 1)

    A = pspl_in.get_amplification(t_phot)
    shift = pspl_in.get_centroid_shift(t_ast)

    dt_phot = t_phot - pspl_in.t0
    dt_ast = t_ast - pspl_in.t0

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 400.0
    imag0 = 19.0
    # flux_in = flux0 * 10**((imag_in - imag0) / -2.5)
    # flux_obs = flux_in * A
    # flux_obs_err = flux_obs**0.5
    # flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    # imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
    # imag_obs_err = 1.087 / flux_obs_err

    imag_mod = pspl_in.get_photometry(t_mod)
    imag_obs = pspl_in.get_photometry(t_phot)
    flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs ** 0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs. 
    lens_pos_in = pspl_in.get_lens_astrometry(t_mod)
    srce_pos_in = pspl_in.get_astrometry_unlensed(t_mod)
    pos_mod = pspl_in.get_astrometry(t_mod)
    pos_obs_tmp = pspl_in.get_astrometry(t_ast)  # srce_pos_in + (shift * 1e-3)
    pos_obs_err = np.ones((len(t_ast), 2), dtype=float) * 0.15 * 1e-3
    pos_obs = pos_obs_tmp + pos_obs_err * np.random.randn(len(t_ast), 2)

    data = {}
    data['target'] = 'NoPar'
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'
    data['phot_files'] = ['fake_data_parallax_phot1']
    data['ast_files'] = ['fake_data_parallax_ast1']
    
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs
    data['mag_err1'] = imag_obs_err

    data['t_ast1'] = t_ast
    data['xpos1'] = pos_obs[:, 0]
    data['ypos1'] = pos_obs[:, 1]
    data['xpos_err1'] = pos_obs_err[:, 0]
    data['ypos_err1'] = pos_obs_err[:, 1]

    params = {}
    params['mL'] = mL_in
    params['t0'] = t0_in
    params['xS0_E'] = xS0_in[0]
    params['xS0_N'] = xS0_in[1]
    params['beta'] = beta_in
    params['muS_E'] = muS_in[0]
    params['muS_N'] = muS_in[1]
    params['muL_E'] = muL_in[0]
    params['muL_N'] = muL_in[1]
    params['dL'] = dL_in
    params['dS'] = dS_in
    params['dL_dS'] = dL_in / dS_in
    params['b_sff1'] = b_sff_in
    params['mag_src1'] = mag_src_in

    if plot:
        model_fitter.plot_photometry(data, pspl_in)
        plt.figure(1)
        plt.title('Input Data and Model')

        model_fitter.plot_astrometry(data, pspl_in)
        plt.figure(2)
        plt.title('Input Data and Model')
        plt.figure(3)
        plt.title('Input Data and Model')
        plt.figure(4)
        plt.title('Input Data and Model')

    return data, params

def fake_data_PSBL(outdir='', outroot='psbl',
                   raL=259.5, decL=-29.0,
                   mLp=10, mLs=10, t0=57000,
                   xS0_E=0, xS0_N=0, beta=5.0,
                   muL_E=0, muL_N=0, muS_E=1, muS_N=1,
                   dL=4000, dS=8000, sep=5e-3, alpha=90,
                   mag_src=18, b_sff=1, parallax=True,
                   target='PSBL', animate=False):
    """
    Optional Inputs
    ---------------
    outdir : str
        The output directory where figures and data are saved.
    outroot : str
        The output file name root for a saved figure.
    raL : float (deg)
        The right ascension in degrees. Needed if parallax=True.
    decL : float (deg)
        The declination in degrees. Needed if parallax=False.
    mL1 : float (Msun)
        Mass of the primary lens.
    mL2 : float (Msun)
        Mass of the secondary lens.
    t0 : float (mjd)
        The time of closest projected approach between the source
        and the geometric center of the lens system in heliocentric
        coordinates.
    xS0_E : float (arcsec)
        Position of the source in RA relative to the
        geometric center of the lens system at time t0.
    xS0_N : float (arcsec)
        Position of the source in Dec relative to the
        geometric center of the lens system at time t0.
    beta : float (mas)
        The closest projected approach between the source
        and the geometric center of the lens system in heliocentric
        coordinates.
    muL_E : float (mas/yr)
        Proper motion of the lens system in RA direction
    muL_N : float (mas/yr)
        Proper motion of the lens system in the Dec direction
    muS_E : float (mas/yr)
        Proper motion of the source in the RA direction
    muS_N : float (mas/yr)
        Proper motion of the source in the Dec direction
    dL : float (pc)
        Distance to the lens system
    dS : float (pc)
        Distance to the source
    sep : float (mas)
        Separation between the binary lens stars,
        projected onto the sky.
    alpha : float (degrees)
        Angle of the project binary separation vector on the
        sky. The separation vector points from the secondary
        to the primary and the angle alpha is measured in
        degrees East of North.
    mag_src : float (mag)
        Brightness of the source.
    b_sff : float
        Source flux fraction = fluxS / (fluxS + fluxL1 + fluxL2 + fluxN)

    """

    start = time.time()
    if parallax:
        psbl = model.PSBL_PhotAstrom_Par_Param1(mLp, mLs, t0, xS0_E, xS0_N,
                                   beta, muL_E, muL_N, muS_E, muS_N, dL, dS,
                                   sep, alpha, [b_sff], [mag_src],
                                   raL=raL, decL=decL, root_tol=1e-8)
    else:
        psbl = model.PSBL_PhotAstrom_noPar_Param1(mLp, mLs, t0, xS0_E, xS0_N,
                          beta, muL_E, muL_N, muS_E, muS_N, dL, dS,
                          sep, alpha, [b_sff], [mag_src],
                          root_tol=1e-8)

    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_pho = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(54000, 60000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_pho_new = np.arange(year_start + phot_start,
                              year_start + phot_start + phot_win, 1)
        t_pho = np.concatenate([t_pho, t_pho_new])

        ast_win = 120.0
        ast_start = (365.25 - ast_win) / 2.0
        t_ast_new = np.arange(year_start + ast_start,
                              year_start + ast_start + ast_win, 28)
        t_ast = np.concatenate([t_ast, t_ast_new])

    t_mod = np.arange(t_pho.min(), t_pho.max(), 1)

    i_pho, A_pho = psbl.get_all_arrays(t_pho)
    i_ast, A_ast = psbl.get_all_arrays(t_ast)
    i_mod, A_mod = psbl.get_all_arrays(t_mod)

    imag_pho = psbl.get_photometry(t_pho, amp_arr=A_pho)
    imag_mod = psbl.get_photometry(t_mod, amp_arr=A_mod)

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 400.0
    imag0 = 19.0

    flux_pho = flux0 * 10 ** ((imag_pho - imag0) / -2.5)
    flux_pho_err = flux_pho ** 0.5
    flux_pho += np.random.randn(len(t_pho)) * flux_pho_err
    imag_pho = -2.5 * np.log10(flux_pho / flux0) + imag0
    imag_pho_err = 1.087 / flux_pho_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs.
    lens_pos = psbl.get_lens_astrometry(t_mod)
    lens1_pos, lens2_pos = psbl.get_resolved_lens_astrometry(t_mod)
    srce_pos = psbl.get_astrometry_unlensed(t_mod)
    srce_pos_lensed_res = psbl.get_resolved_astrometry(t_mod)
    srce_pos_lensed_unres = psbl.get_astrometry(t_mod)

    srce_pos_lensed_res = np.ma.masked_invalid(srce_pos_lensed_res)

    stop = time.time()

    fmt = 'It took {0:.2f} seconds to evaluate the model at {1:d} time steps'
    print(fmt.format(stop - start, len(t_mod) + len(t_ast) + len(t_pho)))

    ##########
    # Plot photometry
    ##########
    plt.figure(1)
    plt.clf()
    plt.errorbar(t_pho, imag_pho, yerr=imag_pho_err, fmt='k.', label='Sim Obs',
                 alpha=0.2)
    plt.plot(t_mod, imag_mod, color='red', label='Model')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (MJD)')
    plt.ylabel('I (mag)')
    plt.legend()

    ##########
    # Plot astrometry
    ##########
    plt.figure(2)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1],
             c='gray', marker='.', linestyle='none', alpha=0.2,
             label='lens system')
    plt.plot(lens1_pos[:, 0], lens1_pos[:, 1],
             c='black', linestyle='none', marker='o',
             label='lens primary')
    plt.plot(lens2_pos[:, 0], lens2_pos[:, 1],
             c='black', linestyle='none', marker='o', mfc='none',
             label='lens secondary')
    plt.scatter(srce_pos[:, 0], srce_pos[:, 1],
                c=t_mod, marker='.', s=2, alpha=0.2,
                label='src unlensed')

    colors = ['navy', 'blue', 'slateblue', 'darkslateblue', 'indigo']
    for ii in range(srce_pos_lensed_res.shape[1]):
        plt.plot(srce_pos_lensed_res[:, ii, 0], srce_pos_lensed_res[:, ii, 1],
                 c=colors[ii], linestyle='none', marker='.', markersize=1,
                 alpha=0.5,
                 label='src lensed img{0:d}'.format(ii + 1))

    plt.plot(srce_pos_lensed_unres[:, 0], srce_pos_lensed_unres[:, 1],
             c='red', linestyle='-',
             label='src lensed unres')

    pos_ast_tmp = psbl.get_astrometry(t_ast, image_arr=i_ast, amp_arr=A_ast)
    pos_ast_err = np.ones((len(t_ast), 2), dtype=float) * 0.15 * 1e-3
    pos_ast = pos_ast_tmp + pos_ast_err * np.random.randn(len(t_ast), 2)

    plt.errorbar(pos_ast[:, 0], pos_ast[:, 1],
                 xerr=pos_ast_err[:, 0], yerr=pos_ast_err[:, 0],
                 fmt='k.', color='black', alpha=0.2)

    plt.gca().invert_xaxis()
    plt.xlabel(r'$\Delta \alpha^*$ (mas)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.legend(fontsize=8)
    plt.subplots_adjust(left=0.25, top=0.8)

    p2 = plt.gca().get_position().get_points().flatten()
    ax_cbar = plt.gcf().add_axes([p2[0], 0.82, p2[2] - p2[0], 0.05])
    plt.colorbar(cax=ax_cbar, orientation='horizontal', label='Time (MJD)',
                 ticklocation='top')

    data = {}
    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'
    data['phot_files'] = ['fake_data_parallax_phot1']
    data['ast_files'] = ['fake_data_parallax_ast1']
    
    data['t_phot1'] = t_pho
    data['mag1'] = imag_pho
    data['mag_err1'] = imag_pho_err

    data['t_ast1'] = t_ast
    data['xpos1'] = pos_ast[:, 0]
    data['ypos1'] = pos_ast[:, 1]
    data['xpos_err1'] = pos_ast_err[:, 0]
    data['ypos_err1'] = pos_ast_err[:, 1]

    data['raL'] = raL
    data['decL'] = decL

    params = {}
    params['mLp'] = mLp
    params['mLs'] = mLs
    params['sep'] = sep
    params['alpha'] = alpha
    params['t0'] = t0
    params['xS0_E'] = xS0_E
    params['xS0_N'] = xS0_N
    params['beta'] = beta
    params['muS_E'] = muS_E
    params['muS_N'] = muS_N
    params['muL_E'] = muL_E
    params['muL_N'] = muL_N
    params['dL'] = dL
    params['dS'] = dS
    params['b_sff'] = b_sff
    params['mag_src'] = mag_src
    params['thetaE_amp'] = psbl.thetaE_amp
    params['u0_amp'] = psbl.u0_amp
    params['tE'] = psbl.tE
    params['piS'] = psbl.piS
    params['piE_E'] = psbl.piE[0]
    params['piE_N'] = psbl.piE[1]
    params['q'] = mLs/mLp
    
    out_name = outdir + outroot + '_movie.gif'
    if animate:
        ani = plot_models.animate_PSBL(psbl, outfile=out_name)
    else:
        ani = None

#    np.savetxt('fake_data_PSBL_phot.dat', (data['t_phot1'], data['mag1'], data['mag_err1']))

    return data, params, psbl, ani

def fake_data_continuous_tiny_err_PSBL(outdir='', outroot='psbl',
                                       raL=259.5, decL=-29.0,
                                       mL1=10, mL2=10, t0=57000,
                                       xS0_E=0, xS0_N=0, beta=5.0,
                                       muL_E=0, muL_N=0, muS_E=1, muS_N=1,
                                       dL=4000, dS=8000, sep=5e-3, alpha=90,
                                       mag_src=18, b_sff=1, parallax=True,
                                       target='PSBL', animate=False):
    """
    Optional Inputs
    ---------------
    outdir : str
        The output directory where figures and data are saved.
    outroot : str
        The output file name root for a saved figure.
    raL : float (deg)
        The right ascension in degrees. Needed if parallax=True.
    decL : float (deg)
        The declination in degrees. Needed if parallax=False.
    mL1 : float (Msun)
        Mass of the primary lens.
    mL2 : float (Msun)
        Mass of the secondary lens.
    t0 : float (mjd)
        The time of closest projected approach between the source
        and the geometric center of the lens system in heliocentric
        coordinates.
    xS0_E : float (arcsec)
        Position of the source in RA relative to the
        geometric center of the lens system at time t0.
    xS0_N : float (arcsec)
        Position of the source in Dec relative to the
        geometric center of the lens system at time t0.
    beta : float (mas)
        The closest projected approach between the source
        and the geometric center of the lens system in heliocentric
        coordinates.
    muL_E : float (mas/yr)
        Proper motion of the lens system in RA direction
    muL_N : float (mas/yr)
        Proper motion of the lens system in the Dec direction
    muS_E : float (mas/yr)
        Proper motion of the source in the RA direction
    muS_N : float (mas/yr)
        Proper motion of the source in the Dec direction
    dL : float (pc)
        Distance to the lens system
    dS : float (pc)
        Distance to the source
    sep : float (arcsec)
        Separation between the binary lens stars,
        projected onto the sky.
    alpha : float (degrees)
        Angle of the project binary separation vector on the
        sky. The separation vector points from the secondary
        to the primary and the angle alpha is measured in
        degrees East of North.
    mag_src : float (mag)
        Brightness of the source.
    b_sff : float
        Source flux fraction = fluxS / (fluxS + fluxL1 + fluxL2 + fluxN)

    """

    start = time.time()
    if parallax:
        psbl = model.PSBL_PhotAstrom_Par_Param1(mL1, mL2, t0, xS0_E, xS0_N,
                                   beta, muL_E, muL_N, muS_E, muS_N, dL, dS,
                                   sep, alpha, [b_sff], [mag_src],
                                   raL=raL, decL=decL, root_tol=1e-8)
    else:
        psbl = model.PSBL_PhotAstrom_noPar_Param1(mL1, mL2, t0, xS0_E, xS0_N,
                          beta, muL_E, muL_N, muS_E, muS_N, dL, dS,
                          sep, alpha, [b_sff], [mag_src],
                          root_tol=1e-8)

    # Simulate photometric and astrometric observations every day.
    t_pho = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    t_pho =  np.arange(54000, 60000, 1)
    t_ast =  np.arange(54000, 60000, 1)

    t_mod = np.arange(t_pho.min(), t_pho.max(), 1)

    i_pho, A_pho = psbl.get_all_arrays(t_pho)
    i_ast, A_ast = psbl.get_all_arrays(t_ast)
    i_mod, A_mod = psbl.get_all_arrays(t_mod)

    imag_pho = psbl.get_photometry(t_pho, amp_arr=A_pho)
    imag_mod = psbl.get_photometry(t_mod, amp_arr=A_mod)

    # Make the photometric observations.
    # Assume 0.005 mag photoemtric errors at I=19.
    # This means Signal = 40000 e- at I=19.
    flux0 = 40000.0
    imag0 = 19.0

    flux_pho = flux0 * 10 ** ((imag_pho - imag0) / -2.5)
    flux_pho_err = flux_pho ** 0.5
    flux_pho += np.random.randn(len(t_pho)) * flux_pho_err
    imag_pho = -2.5 * np.log10(flux_pho / flux0) + imag0
    imag_pho_err = 1.087 / flux_pho_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs.
    # Q: Where does the 0.15 milliarcsec error comes from?
    lens_pos = psbl.get_lens_astrometry(t_mod)
    lens1_pos, lens2_pos = psbl.get_resolved_lens_astrometry(t_mod)
    srce_pos = psbl.get_astrometry_unlensed(t_mod)
    srce_pos_lensed_res = psbl.get_resolved_astrometry(t_mod)
    srce_pos_lensed_unres = psbl.get_astrometry(t_mod)

    srce_pos_lensed_res = np.ma.masked_invalid(srce_pos_lensed_res)

    stop = time.time()

    fmt = 'It took {0:.2f} seconds to evaluate the model at {1:d} time steps'
    print(fmt.format(stop - start, len(t_mod) + len(t_ast) + len(t_pho)))

    ##########
    # Plot photometry
    ##########
    plt.figure(1)
    plt.clf()
    plt.errorbar(t_pho, imag_pho, yerr=imag_pho_err, fmt='k.', label='Sim Obs',
                 alpha=0.2)
    plt.plot(t_mod, imag_mod, color='red', label='Model')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (MJD)')
    plt.ylabel('I (mag)')
    plt.legend()

    ##########
    # Plot astrometry
    ##########
    plt.figure(2)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1],
             c='gray', marker='.', linestyle='none', alpha=0.2,
             label='lens system')
    plt.plot(lens1_pos[:, 0], lens1_pos[:, 1],
             c='black', linestyle='none', marker='o',
             label='lens primary')
    plt.plot(lens2_pos[:, 0], lens2_pos[:, 1],
             c='black', linestyle='none', marker='o', mfc='none',
             label='lens secondary')
    plt.scatter(srce_pos[:, 0], srce_pos[:, 1],
                c=t_mod, marker='.', s=2, alpha=0.2,
                label='src unlensed')

    colors = ['navy', 'blue', 'slateblue', 'darkslateblue', 'indigo']
    for ii in range(srce_pos_lensed_res.shape[1]):
        plt.plot(srce_pos_lensed_res[:, ii, 0], srce_pos_lensed_res[:, ii, 1],
                 c=colors[ii], linestyle='none', marker='.', markersize=1,
                 alpha=0.5,
                 label='src lensed img{0:d}'.format(ii + 1))

    plt.plot(srce_pos_lensed_unres[:, 0], srce_pos_lensed_unres[:, 1],
             c='red', linestyle='-',
             label='src lensed unres')

    pos_ast_tmp = psbl.get_astrometry(t_ast, image_arr=i_ast, amp_arr=A_ast)
    pos_ast_err = np.ones((len(t_ast), 2), dtype=float) * 0.15 * 1e-3
    pos_ast = pos_ast_tmp + pos_ast_err * np.random.randn(len(t_ast), 2)

    plt.errorbar(pos_ast[:, 0], pos_ast[:, 1],
                 xerr=pos_ast_err[:, 0], yerr=pos_ast_err[:, 0],
                 fmt='k.', color='black', alpha=0.2)

    plt.gca().invert_xaxis()
    plt.xlabel(r'$\Delta \alpha^*$ (mas)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.legend(fontsize=8)
    plt.subplots_adjust(left=0.25, top=0.8)

    p2 = plt.gca().get_position().get_points().flatten()
    ax_cbar = plt.gcf().add_axes([p2[0], 0.82, p2[2] - p2[0], 0.05])
    plt.colorbar(cax=ax_cbar, orientation='horizontal', label='Time (MJD)',
                 ticklocation='top')

    data = {}
    data['t_phot1'] = t_pho
    data['mag1'] = imag_pho
    data['mag_err1'] = imag_pho_err
    data['phot_files'] = ['fake_data_parallax_phot1']
    data['ast_files'] = ['fake_data_parallax_ast1']

    data['t_ast1'] = t_ast
    data['xpos1'] = pos_ast[:, 0]
    data['ypos1'] = pos_ast[:, 1]
    data['xpos_err1'] = pos_ast_err[:, 0]
    data['ypos_err1'] = pos_ast_err[:, 1]

    data['raL'] = raL
    data['decL'] = decL
    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'

    params = {}
    params['mL1'] = mL1
    params['mL2'] = mL2
    params['sep'] = sep
    params['alpha'] = alpha
    params['t0'] = t0
    params['xS0_E'] = xS0_E
    params['xS0_N'] = xS0_N
    params['beta'] = beta
    params['muS_E'] = muS_E
    params['muS_N'] = muS_N
    params['muL_E'] = muL_E
    params['muL_N'] = muL_N
    params['dL'] = dL
    params['dS'] = dS
    params['b_sff'] = b_sff
    params['mag_src'] = mag_src

    out_name = outdir + outroot + '_movie.gif'
    if animate:
        ani = plot_models.animate_PSBL(psbl, outfile=out_name)
    else:
        ani = None

#    np.savetxt('fake_data_continuous_tiny_err_PSBL_phot.dat', (data['t_phot1'], data['mag1'], data['mag_err1']))

    return data, params, psbl, ani

def fake_data_PSBL_phot(outdir='', outroot='psbl',
                        raL=259.5, decL=-29.0,
                        t0=57000.0, u0_amp=0.8, tE=500.0,
                        piE_E=0.02, piE_N=0.02,
                        q=0.5, sep=5.0, phi=75.0, b_sff1=0.5, mag_src1=16.0,
                        parallax=True, target='Unknown', animate=False):
    """
    Optional Inputs
    ---------------
    outdir : str
        The output directory where figures and data are saved.
    outroot : str
        The output file name root for a saved figure.
    raL : float (deg)
        The right ascension in degrees. Needed if parallax=True.
    decL : float (deg)
        The declination in degrees. Needed if parallax=False.
    t0: float
        Time of photometric peak, as seen from Earth [MJD]
    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. It can be
        positive (u0_hat cross thetaE_hat pointing away from us) or
        negative (u0_hat cross thetaE_hat pointing towards us).
    tE: float
        Einstein crossing time. [MJD]
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    q: float
        Mass ratio (low-mass / high-mass)
    sep: float
        Angular separation of the two lenses in units of thetaE where
        thetaE is defined with the total binary mass.
    phi: float
        Angle made between the binary axis and the relative proper motion vector,
        measured in degrees.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        b_sff = f_S / (f_S + f_L + f_N). This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_src:  array or list
        Photometric magnitude of the source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    """

    start = time.time()
    if parallax:
        psbl = model.PSBL_Phot_Par_Param1(t0, u0_amp, tE, piE_E, piE_N, q, sep, phi,
                                          [b_sff1], [mag_src1],
                                          raL=raL, decL=decL, root_tol=1e-8)
    else:
        psbl = model.PSBL_Phot_noPar_Param1(t0, u0_amp, tE, piE_E, piE_N, q, sep, phi,
                                            [b_sff1], [mag_src1],
                                            root_tol=1e-8)

    # Simulate
    # photometric observations every 1 day and
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry.
    t_pho = np.array([], dtype=float)
    for year_start in np.arange(54000, 60000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_pho_new = np.arange(year_start + phot_start,
                              year_start + phot_start + phot_win, 1)
        t_pho = np.concatenate([t_pho, t_pho_new])

    t_mod = np.arange(t_pho.min(), t_pho.max(), 1)

    i_pho, A_pho = psbl.get_all_arrays(t_pho)
    i_mod, A_mod = psbl.get_all_arrays(t_mod)

    imag_pho = psbl.get_photometry(t_pho, amp_arr=A_pho)
    imag_mod = psbl.get_photometry(t_mod, amp_arr=A_mod)

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 400.0
    imag0 = 19.0

    flux_pho = flux0 * 10 ** ((imag_pho - imag0) / -2.5)
    flux_pho_err = flux_pho ** 0.5
    flux_pho += np.random.randn(len(t_pho)) * flux_pho_err
    imag_pho = -2.5 * np.log10(flux_pho / flux0) + imag0
    imag_pho_err = 1.087 / flux_pho_err

    stop = time.time()

    fmt = 'It took {0:.2f} seconds to evaluate the model at {1:d} time steps'
    print(fmt.format(stop - start, len(t_mod) + len(t_pho)))

    ##########
    # Plot photometry
    ##########
    plt.figure(1)
    plt.clf()
    plt.errorbar(t_pho, imag_pho, yerr=imag_pho_err, fmt='k.', label='Sim Obs',
                 alpha=0.2)
    plt.plot(t_mod, imag_mod, color='red', label='Model')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (MJD)')
    plt.ylabel('I (mag)')
    plt.legend()

    data = {}
    data['t_phot1'] = t_pho
    data['mag1'] = imag_pho
    data['mag_err1'] = imag_pho_err
    data['phot_files'] = ['fake_data_parallax_phot1']
    data['ast_files'] = ['fake_data_parallax_ast1']

    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'
    data['raL'] = raL
    data['decL'] = decL

    params = {}
    params['t0'] = t0
    params['u0_amp'] = u0_amp
    params['tE'] = tE
    params['piE_E'] = piE_E
    params['piE_N'] = piE_N
    params['q'] = q
    params['sep'] = sep
    params['phi'] = phi
    params['b_sff'] = b_sff1
    params['mag_src'] = mag_src1

    out_name = outdir + outroot + '_movie.gif'
    if animate:
        ani = plot_models.animate_PSBL(psbl, outfile=out_name)
    else:
        ani = None

    return data, params, psbl, ani

def fake_data_multiphot_parallax(raL_in, decL_in, t0_in, u0_amp_in, tE_in, piE_E_in, piE_N_in,
                                 b_sff_in1, mag_src_in1, b_sff_in2, mag_src_in2,
                                 target='Unknown',
                                 outdir=''):
    
    pspl_par_in = model.PSPL_Phot_Par_Param1(t0_in, u0_amp_in, tE_in,
                                             piE_E_in, piE_N_in,
                                             np.array([b_sff_in1, b_sff_in2]),
                                             np.array([mag_src_in1, mag_src_in2]),
                                             raL=raL_in, decL=decL_in)

    # Simulate
    # OGLE-like photometric observations every 1 day and
    # HST-like photometric observations every 30 days
    # for the bulge observing window.
    # Observations missed for 125 days out of 365 days
    t_phot1 = np.array([], dtype=float)
    t_phot2 = np.array([], dtype=float)

    # Start on a Jan 1
    jan1_2020 = Time('2020-01-01').mjd
    end_time = jan1_2020 + 7.0 * 365.25
    
    for year_start in np.arange(jan1_2020, end_time, 365.25):
        phot1_win = 240.0
        phot1_start = (365.25 - phot1_win) / 2.0
        t_phot1_new = np.arange(year_start + phot1_start,
                                year_start + phot1_start + phot1_win, 1)
        t_phot1 = np.concatenate([t_phot1, t_phot1_new])

        phot2_win = 180.0
        phot2_start = (365.25 - phot2_win) / 2.0
        t_phot2_new = np.arange(year_start + phot2_start,
                                year_start + phot2_start + phot2_win, 30)
        t_phot2 = np.concatenate([t_phot2, t_phot2_new])

    # Only keep HST/AO photometry after peak.
    idx = np.where(t_phot2 > t0_in)[0]
    t_phot2 = t_phot2[idx]
        
    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    iflux0 = 4000.0
    imag0 = 19.0
    imag_obs1 = pspl_par_in.get_photometry(t_phot1, filt_idx=0)
    flux_obs1 = iflux0 * 10 ** ((imag_obs1 - imag0) / -2.5)
    flux_obs1_err = flux_obs1 ** 0.5
    flux_obs1 += np.random.randn(len(t_phot1)) * flux_obs1_err
    imag_obs1 = -2.5 * np.log10(flux_obs1 / iflux0) + imag0
    imag_obs1_err = 1.087 / flux_obs1_err

    kflux0 = 4000.0
    kmag0 = 18.0
    kmag_obs2 = pspl_par_in.get_photometry(t_phot2, filt_idx=1)
    flux_obs2 = kflux0 * 10 ** ((kmag_obs2 - kmag0) / -2.5)
    flux_obs2_err = flux_obs2 ** 0.5
    flux_obs2 += np.random.randn(len(t_phot2)) * flux_obs2_err
    imag_obs2 = -2.5 * np.log10(flux_obs2 / kflux0) + kmag0
    imag_obs2_err = 1.087 / flux_obs2_err
    

    data = {}
    data['t_phot1'] = t_phot1
    data['mag1'] = imag_obs1
    data['mag_err1'] = imag_obs1_err
    data['t_phot2'] = t_phot2
    data['mag2'] = imag_obs2
    data['mag_err2'] = imag_obs2_err

    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'
    data['raL'] = raL_in
    data['decL'] = decL_in

    data['phot_files'] = ['fake_data_phot1', 'fake_data_phot2']
    data['ast_files'] = ['fake_data_ast1']
    
    params = {}
    params['raL'] = raL_in
    params['decL'] = decL_in
    params['t0'] = t0_in
    params['u0_amp'] = u0_amp_in
    params['tE'] = tE_in
    params['piE_E'] = piE_E_in
    params['piE_N'] = piE_N_in
    params['b_sff1'] = b_sff_in1
    params['mag_src1'] = mag_src_in1
    params['b_sff2'] = b_sff_in2
    params['mag_src2'] = mag_src_in2

    model_fitter.plot_photometry(data, pspl_par_in, filt_index=0, dense_time=True)
    plt.figure(1)
    plt.title('Input Data and Model')
    plt.savefig(outdir + 'fake_data_multiphot_par1.png')

    model_fitter.plot_photometry(data, pspl_par_in, filt_index=1, dense_time=True)
    plt.figure(2)
    plt.title('Input Data and Model')
    plt.savefig(outdir + 'fake_data_multiphot_par2.png')


    return data, params, pspl_par_in

def fake_correlated_data_with_astrom():
    """
    Only correlations in the photometry, not astrometry
    """
    t0 = 57000
    u0_amp = 0.1
    tE = 150
    thetaE = 1
    piS = 0.125
    piE_E = 0.05
    piE_N = 0.05
    xS0_E = 0.0
    xS0_N = 0.08E-3
    muS_E = -4.18
    muS_N = -0.28
    b_sff = 0.9
    mag_src = 19.0
    raL = 17.30 * 15.
    decL = -29.0

    our_model = model.PSPL_PhotAstrom_Par_Param2(t0=t0, u0_amp=u0_amp, tE=tE, 
                                                 thetaE=thetaE, piS=piS,
                                                 piE_E=piE_E, piE_N=piE_N,
                                                 xS0_E=xS0_E, xS0_N=xS0_N,
                                                 muS_E=muS_E, muS_N=muS_N,
                                                 b_sff=b_sff, mag_src=mag_src,
                                                 raL=raL, decL=decL)
    
    cel_model = model.Celerite_GP_Model(our_model, 0)

    # Simuate the data
    # Simulate photometric observations every 1 day and
    # astrometric observations every 30 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_mod = np.linspace(56000, 58000, 2000)
    t_phot = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(56000, 58000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

    for year_start in np.arange(56000, 58000, 365.25):
        ast_win = 120.0
        ast_start = (365.25 - ast_win) / 2.0
        t_ast_new = np.arange(year_start + ast_start,
                              year_start + ast_start + ast_win, 30)
        t_ast = np.concatenate([t_ast, t_ast_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = our_model.get_photometry(t_phot)
    flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs ** 0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs_uncorr = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    K = 0.001*np.exp(-0.5*(t_phot[:, None] - t_phot[None, :])**2/1.5)
    K[np.diag_indices(len(t_phot))] += imag_obs_err**2
    imag_obs_corr = np.random.multivariate_normal(cel_model.get_value(t_phot), K)

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs. 
    pos_obs_tmp = our_model.get_astrometry(t_ast)
    pos_obs_err = np.ones((len(t_ast), 2), dtype=float) * 0.01 * 1e-3
    pos_obs = pos_obs_tmp + pos_obs_err * np.random.randn(len(t_ast), 2)

    imag_mod = our_model.get_photometry(t_mod)
    pos_mod = our_model.get_astrometry(t_mod)
    
    # Plot the data
    plt.figure(1)
    plt.plot(t_mod, imag_mod, label='Model')
    plt.errorbar(t_phot, imag_obs_corr, yerr=imag_obs_err, fmt=".r", label='Corr')
    plt.errorbar(t_phot, imag_obs_uncorr, yerr=imag_obs_err, fmt=".k", label='No corr')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

    plt.figure(2)
    plt.plot(pos_mod[:,0], pos_mod[:,1], label='Model')
    plt.errorbar(pos_obs[:,0], pos_obs[:,1], 
                 xerr=pos_obs_err[:,0], yerr=pos_obs_err[:,1], fmt=".k")
    plt.show()
    data = {}
    target = 'fake'
    data['phot_files'] = ['fake_data_phot1']
    data['ast_files'] = ['fake_data_ast1']
    
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs_uncorr
    data['mag_err1'] = imag_obs_err

    data['t_ast1'] = t_ast
    data['xpos1'] = pos_obs[:, 0]
    data['ypos1'] = pos_obs[:, 1]
    data['xpos_err1'] = pos_obs_err[:, 0]
    data['ypos_err1'] = pos_obs_err[:, 1]

    data_corr = {}
    data_corr['t_phot1'] = t_phot
    data_corr['mag1'] = imag_obs_corr
    data_corr['mag_err1'] = imag_obs_err
    data_corr['t_ast1'] = t_ast
    data_corr['xpos1'] = pos_obs[:, 0]
    data_corr['ypos1'] = pos_obs[:, 1]
    data_corr['xpos_err1'] = pos_obs_err[:, 0]
    data_corr['ypos_err1'] = pos_obs_err[:, 1]

    data['raL'] = raL
    data['decL'] = decL
    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'

    data_corr['raL'] = raL
    data_corr['decL'] = decL
    data_corr['target'] = target
    data_corr['phot_data'] = 'sim'
    data_corr['ast_data'] = 'sim'

    params = {}
    params['t0'] = t0
    params['u0_amp'] = u0_amp
    params['tE'] = tE
    params['thetaE'] = thetaE
    params['piS'] = piS
    params['piE_E'] = piE_E
    params['piE_N'] = piE_N
    params['xS0_E'] = xS0_E
    params['xS0_N'] = xS0_N
    params['muS_E'] = muS_E
    params['muS_N'] = muS_N
    params['b_sff1'] = b_sff
    params['mag_src1'] = mag_src
    params['raL'] = raL
    params['decL'] = decL

    return our_model, data, data_corr, params


def fake_correlated_data_multiphot(t0 = 57000, u0_amp = 0.1, tE = 150,
                                       piE_E = 0.05, piE_N = 0.05,
                                       b_sff1 = 0.9, mag_src1 = 19.0,
                                       b_sff2 = 0.9, mag_src2 = 19.0,
                                       gp_log_sigma1 = 1, gp_log_rho1 = 0.1,
                                       gp_log_So1 = 1, gp_log_omegao1 = 1,
                                       raL = 17.30 * 15., decL = -29.0):

    our_model = model.PSPL_Phot_Par_Param1(t0, u0_amp, tE, 
                                           piE_E, piE_N, 
                                           np.array([b_sff1, b_sff2]), 
                                           np.array([mag_src1, mag_src2]),
                                           gp_log_sigma, gp_log_rho, 
                                           gp_log_So, gp_log_omegao, 
                                           raL=raL, decL=decL)
    
    cel_model = model.Celerite_GP_Model(our_model, 0)

    # Simuate the data
    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(56000, 58000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = our_model.get_photometry(t_phot)
    flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs ** 0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs_uncorr = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    K = 0.01*np.exp(-0.5*(t_phot[:, None] - t_phot[None, :])**2/1.5)
    K[np.diag_indices(len(t_phot))] += imag_obs_err**2
    imag_obs_corr = np.random.multivariate_normal(cel_model.get_value(t_phot), K)
    
    # Plot the data
    plt.errorbar(t_phot, imag_obs_uncorr, yerr=imag_obs_err, fmt=".k", label='No corr')
    plt.errorbar(t_phot, imag_obs_corr, yerr=imag_obs_err, fmt=".r", label='Corr')
    plt.legend()
    plt.gca().invert_yaxis()

    data = {}
    target = 'fake'
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs_uncorr
    data['mag_err1'] = imag_obs_err

    data_corr = {}
    data_corr['t_phot1'] = t_phot
    data_corr['mag1'] = imag_obs_corr
    data_corr['mag_err1'] = imag_obs_err

    data['raL'] = raL
    data['decL'] = decL
    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = ''
    data['phot_files'] = ['fake_data_phot1']
    data['ast_files'] = []

    data_corr['raL'] = raL
    data_corr['decL'] = decL
    data_corr['target'] = target
    data_corr['phot_data'] = 'sim'
    data_corr['ast_data'] = ''
    data_corr['phot_files'] = ['fake_data_phot1']
    data_corr['ast_files'] = []

    params = {}
    params['t0'] = 57000
    params['u0_amp'] = 0.1
    params['tE'] = 150
    params['piE_E'] = 0.05
    params['piE_N'] = 0.05
    params['b_sff1'] = 0.9
    params['mag_src1'] = 19.0
    params['gp_log_sigma'] = 1
    params['gp_log_rho'] = 0.1
    params['gp_log_So'] = 1
    params['gp_log_omegao'] = 1
    params['raL'] = 17.30 * 15.
    params['decL'] = -29.0

#    params['gp_rho'] = np.exp(0.1)
#    params['gp_log_omegaofour_So'] = 1 + 4*1

    return our_model, data, data_corr, params


def fake_correlated_data(t0 = 57000, u0_amp = 0.1, tE = 150,
                             piE_E = 0.05, piE_N = 0.05,
                             b_sff = 0.9, mag_src = 19.0,
                             gp_log_sigma = 1, gp_log_rho = 0.1,
                             gp_log_So = 1, gp_log_omegao = 1,
                             raL = 17.30 * 15., decL = -29.0):

    # Does it make sense to "set" the GP params here?
    our_model = model.PSPL_Phot_Par_GP_Param1(t0, u0_amp, tE, 
                                              piE_E, piE_N, b_sff, mag_src,
                                              gp_log_sigma, gp_log_rho, 
                                              gp_log_So, gp_log_omegao, 
                                              raL=raL, decL=decL)
    
    cel_model = model.Celerite_GP_Model(our_model, 0)

    # Simuate the data
    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(56000, 58000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = our_model.get_photometry(t_phot)
    flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs ** 0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs_uncorr = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    K = 0.01*np.exp(-0.5*(t_phot[:, None] - t_phot[None, :])**2/1.5)
    K[np.diag_indices(len(t_phot))] += imag_obs_err**2
    imag_obs_corr = np.random.multivariate_normal(cel_model.get_value(t_phot), K)
    
    # Plot the data
    plt.errorbar(t_phot, imag_obs_uncorr, yerr=imag_obs_err, fmt=".k", label='No corr')
    plt.errorbar(t_phot, imag_obs_corr, yerr=imag_obs_err, fmt=".r", label='Corr')
    plt.legend()
    plt.gca().invert_yaxis()

    data = {}
    target = 'fake'
    data['phot_files'] = ['fake_data_phot1']
    data['ast_files'] = []
    
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs_uncorr
    data['mag_err1'] = imag_obs_err

    data_corr = {}
    data_corr['t_phot1'] = t_phot
    data_corr['mag1'] = imag_obs_corr
    data_corr['mag_err1'] = imag_obs_err

    data['raL'] = raL
    data['decL'] = decL
    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = ''

    data_corr['raL'] = raL
    data_corr['decL'] = decL
    data_corr['target'] = target
    data_corr['phot_data'] = 'sim'
    data_corr['ast_data'] = ''

    params = {}
    params['t0'] = 57000
    params['u0_amp'] = 0.1
    params['tE'] = 150
    params['piE_E'] = 0.05
    params['piE_N'] = 0.05
    params['b_sff1'] = 0.9
    params['mag_src1'] = 19.0
    params['gp_log_sigma1'] = 1
    params['gp_log_rho1'] = 0.1
    params['gp_log_So1'] = 1
    params['gp_log_omegao1'] = 1
    params['raL'] = 17.30 * 15.
    params['decL'] = -29.0

#    params['gp_rho'] = np.exp(0.1)
#    params['gp_log_omegaofour_So'] = 1 + 4*1

    return our_model, data, data_corr, params


def fake_correlated_data_lunch_talk():
    t0 = 57000
    u0_amp = 0.1
    tE = 150
    piE_E = 0.20
    piE_N = 0.05
    b_sff = 0.9
    mag_src = 19.0
    raL = 17.30 * 15.
    decL = -29.0

    # Does it make sense to "set" the GP params here?

    our_model = model.PSPL_Phot_Par_Param1(t0, u0_amp, tE, 
                                              piE_E, piE_N, b_sff, mag_src,
                                              raL=raL, decL=decL)
    
    cel_model = model.Celerite_GP_Model(our_model, 0)

    # Simuate the data
    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(56000, 58000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = our_model.get_photometry(t_phot)
    flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs ** 0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs_uncorr = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    diff = t_phot[:, None] - t_phot[None, :]
    l1 = 1
    A1 = 0.001
    l2 = 180
    A2 = 0.05
    p2 = 365.25
    l3 = 30
    A3 = 1
    p3 = 180
    K = A1*np.exp(-0.5*diff**2/(2*l1**2))
    K *= A2*np.exp(2*np.sin(np.pi*np.abs(diff)/p2)**2/l2**2)
    K *= A3*np.exp(2*np.sin(np.pi*np.abs(diff)/p3)**2/l3**2)
    K[np.diag_indices(len(t_phot))] += imag_obs_err**2
    imag_obs_corr = np.random.multivariate_normal(cel_model.get_value(t_phot), K)

#    K = 0.01*np.exp(-0.5*(t_phot[:, None] - t_phot[None, :])**2/1.5)
#    K[np.diag_indices(len(t_phot))] += imag_obs_err**2
#    imag_obs_corr = np.random.multivariate_normal(cel_model.get_value(t_phot), K)
    
    # Plot the data
    plt.errorbar(t_phot, imag_obs_uncorr, yerr=imag_obs_err, fmt=".k", label='No corr')
    plt.errorbar(t_phot, imag_obs_corr, yerr=imag_obs_err, fmt=".r", label='Corr')
    plt.legend()
    plt.gca().invert_yaxis()

    data = {}
    target = 'fake'
    data['phot_files'] = ['fake_data_phot1']
    data['ast_files'] = []
    
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs_uncorr
    data['mag_err1'] = imag_obs_err

    data_corr = {}
    data_corr['phot_files'] = ['fake_data_phot1']
    data_corr['ast_files'] = []
    data_corr['t_phot1'] = t_phot
    data_corr['mag1'] = imag_obs_corr
    data_corr['mag_err1'] = imag_obs_err

    data['raL'] = raL
    data['decL'] = decL
    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = ''

    data_corr['raL'] = raL
    data_corr['decL'] = decL
    data_corr['target'] = target
    data_corr['phot_data'] = 'sim'
    data_corr['ast_data'] = ''

    params = {}
    params['t0'] = t0
    params['u0_amp'] = u0_amp
    params['tE'] = tE
    params['piE_E'] = piE_E
    params['piE_N'] = piE_N
    params['b_sff1'] = b_sff
    params['mag_src1'] = mag_src
    params['raL'] = raL
    params['decL'] = decL

#    params['gp_rho'] = np.exp(0.1)
#    params['gp_log_omegaofour_So'] = 1 + 4*1

    return our_model, data, data_corr, params


# def fake_data_FSBL():
#     #####
#     # initial test
#     # can't get to work...
#     # trying edward's numbers instead.
#     #####
#     #    lens_mass1_in = 5.0 # Msun
#     #    lens_mass2_in = 10.0 # Msun
#     #    t0_in = 57000.00
#     #    xS0_in = np.array([0.000, 0.000])
#     #    beta_in = 1.4 # mas
#     #    muL_in = np.array([0.00, 0.00])
#     #    muS_in = np.array([8.00, 0.00])
#     #    dL_in = 4000.0 # pc
#     #    dS_in = 8000.0 # pc
#     #    n_in = 1
#     #    radius_in = 20 # Rsun
#     #    separation_in = 5 * 10**-3 # arcsec (corresponds to 20 AU at 4 kpc)
#     #    angle_in = 45
#     #    utilde_in = 0.9
#     #    mag_src_in = 19.0
#     #    b_sff_in = 0.9

#     #####
#     # edward's numbers
#     #####
#     lens_mass1_in = 4  # Msun
#     lens_mass2_in = 7  # Msun
#     t0_in = 0
#     xS0_in = np.array([0, 0])
#     beta_in = -0.55  # mas
#     muL_in = np.array([0, 0])
#     muS_in = np.array([-2, 0])
#     dL_in = 4000  # pc
#     dS_in = 8000  # pc
#     n_in = 40
#     radius_in = 1  # Rsun
#     separation_in = 0.004  # arcsec (corresponds to 20 AU at 4 kpc)
#     angle_in = 0
#     utilde_in = 0
#     mag_src_in = 10  # originally 10
#     b_sff_in = 1

#     fsbl_in = model.FSBL(lens_mass1_in, lens_mass2_in, t0_in, xS0_in,
#                          beta_in, muL_in, muS_in, dL_in, dS_in, n_in,
#                          radius_in,
#                          separation_in, angle_in, utilde_in, mag_src_in,
#                          b_sff_in)

#     # Simulate
#     # photometric observations every 1 day and
#     # astrometric observations every 14 days
#     # for the bulge observing window. Observations missed
#     # for 125 days out of 365 days for photometry and missed
#     # for 245 days out of 365 days for astrometry.
#     t_phot = np.array([], dtype=float)
#     t_ast = np.array([], dtype=float)
#     for year_start in np.arange(t0_in - 1000, t0_in + 1000, 365.25):
#         phot_win = 240.0
#         phot_start = (365.25 - phot_win) / 2.0
#         t_phot_new = np.arange(year_start + phot_start,
#                                year_start + phot_start + phot_win, 1)
#         t_phot = np.concatenate([t_phot, t_phot_new])

#         ast_win = 120.0
#         ast_start = (365.25 - ast_win) / 2.0
#         t_ast_new = np.arange(year_start + ast_start,
#                               year_start + ast_start + ast_win, 14)
#         t_ast = np.concatenate([t_ast, t_ast_new])

#     t_mod = np.arange(t_phot.min(), t_phot.max(), 1)

#     # Make the photometric observations.
#     # Assume 0.05 mag photometric errors at I=19.
#     # This means Signal = 1000 e- at I=19.
#     flux0 = 400.0
#     imag0 = 19.0

#     imag_mod = fsbl_in.get_photometry(t_mod)
#     imag_obs = fsbl_in.get_photometry(t_phot)
#     flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
#     flux_obs_err = flux_obs ** 0.5
#     flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
#     imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
#     imag_obs_err = 1.087 / flux_obs_err

#     # Make the astrometric observations.
#     # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs.
#     pos_mod = fsbl_in.get_astrometry(t_mod)
#     pos_obs_tmp = fsbl_in.get_astrometry(t_ast)  # srce_pos_in + (shift * 1e-3)
#     pos_obs_err = np.ones((len(t_ast), 2), dtype=float) * 0.15 * 1e-3
#     pos_obs = pos_obs_tmp + pos_obs_err * np.random.randn(len(t_ast), 2)

#     data = {}
#     data['t_phot1'] = t_phot
#     data['mag1'] = imag_obs
#     data['mag_err1'] = imag_obs_err

#     data['t_ast1'] = t_ast
#     data['xpos1'] = pos_obs[:, 0]
#     data['ypos1'] = pos_obs[:, 1]
#     data['xpos_err1'] = pos_obs_err[:, 0]
#     data['ypos_err1'] = pos_obs_err[:, 1]

#     params = {}
#     params['lens_mass1'] = lens_mass1_in
#     params['lens_mass2'] = lens_mass2_in
#     params['t0'] = t0_in
#     params['xS0_E'] = xS0_in[0]
#     params['xS0_N'] = xS0_in[1]
#     params['beta'] = beta_in
#     params['muL_E'] = muL_in[0]
#     params['muL_N'] = muL_in[1]
#     params['muS_E'] = muS_in[0]
#     params['muS_N'] = muS_in[1]
#     params['dL'] = dL_in
#     params['dS'] = dS_in
#     params['n'] = n_in
#     params['radius'] = radius_in
#     params['separation'] = separation_in
#     params['angle'] = angle_in
#     params['utilde'] = utilde_in
#     params['mag_src'] = mag_src_in
#     params['b_sff'] = b_sff_in

#     model_fitter_reorg.plot_photometry(data, fsbl_in)
#     plt.figure(1)
#     plt.title('Input Data and Model')

#     model_fitter_reorg.plot_astrometry(data, fsbl_in)
#     plt.figure(2)
#     plt.title('Input Data and Model')
#     plt.figure(3)
#     plt.title('Input Data and Model')
#     plt.figure(4)
#     plt.title('Input Data and Model')

#     return data, params

def fake_data_lumlens_parallax_bulge(outdir='./test_mnest_lumlens_bulge/'):
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
    b_sff = 1.0
    imag_in = 19.0

    data, params = fake_data_lumlens_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in,
                                      beta_in,
                                      muS_in, muL_in, dL_in, dS_in, b_sff,
                                      imag_in, outdir=outdir, target='Bulge')

    return data, params


def fake_data_lumlens_parallax_bulge2(outdir='./test_mnest_lumlens_bulge/'):
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
    b_sff1 = 0.5
    b_sff2 = 1.0
    imag_in1 = 17.0
    imag_in2 = 19.0

    data1, params1 = fake_data_lumlens_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in,
                                      beta_in,
                                      muS_in, muL_in, dL_in, dS_in, b_sff1,
                                      imag_in1, outdir=outdir, target='Bulge')

    data2, params2 = fake_data_lumlens_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in,
                                      beta_in,
                                      muS_in, muL_in, dL_in, dS_in, b_sff2,
                                      imag_in2, outdir=outdir, target='Bulge')

    return data1, data2, params1, params2 

def fake_data_lumlens_parallax_bulge4(outdir='./test_mnest_lumlens_bulge4_DEBUG/'):
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
    b_sff1 = 0.5
    b_sff2 = 1.0
    b_sff3 = 0.88
    b_sff4 = 0.4
    imag_in1 = 17.0
    imag_in2 = 19.0
    imag_in3 = 18.0
    imag_in4 = 16.0

    data1, params1 = fake_data_lumlens_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in,
                                      beta_in,
                                      muS_in, muL_in, dL_in, dS_in, b_sff1,
                                      imag_in1, outdir=outdir, target='sim1')

    data2, params2 = fake_data_lumlens_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in,
                                      beta_in,
                                      muS_in, muL_in, dL_in, dS_in, b_sff2,
                                      imag_in2, outdir=outdir, target='sim2')

    data3, params3 = fake_data_lumlens_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in,
                                      beta_in,
                                      muS_in, muL_in, dL_in, dS_in, b_sff3,
                                      imag_in3, outdir=outdir, target='sim3')

    data4, params4 = fake_data_lumlens_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in,
                                      beta_in,
                                      muS_in, muL_in, dL_in, dS_in, b_sff4,
                                      imag_in4, outdir=outdir, target='sim4')

    return data1, data2, data3, data4, params1, params2, params3, params4 


def fake_data_lumlens_parallax(raL_in, decL_in, mL_in, t0_in, xS0_in, beta_in,
                               muS_in, muL_in, dL_in, dS_in, b_sff_in, mag_src_in,
                               outdir='', target='Unknwon'):
    pspl_par_in = model.PSPL_PhotAstrom_LumLens_Par_Param1(mL=mL_in,
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

    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(56000, 58000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

        ast_win = 120.0
        ast_start = (365.25 - ast_win) / 2.0
        t_ast_new = np.arange(year_start + ast_start,
                              year_start + ast_start + ast_win, 14)
        t_ast = np.concatenate([t_ast, t_ast_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = pspl_par_in.get_photometry(t_phot)
    flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs ** 0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs. 
    pos_obs_tmp = pspl_par_in.get_astrometry(t_ast)
    pos_obs_err = np.ones((len(t_ast), 2), dtype=float) * 0.01 * 1e-3
    pos_obs = pos_obs_tmp + pos_obs_err * np.random.randn(len(t_ast), 2)

    data = {}
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs
    data['mag_err1'] = imag_obs_err

    data['t_ast1'] = t_ast
    data['xpos1'] = pos_obs[:, 0]
    data['ypos1'] = pos_obs[:, 1]
    data['xpos_err1'] = pos_obs_err[:, 0]
    data['ypos_err1'] = pos_obs_err[:, 1]
    
    data['raL'] = raL_in
    data['decL'] = decL_in
    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'

    params = {}
    params['raL'] = raL_in
    params['decL'] = decL_in
    params['mL'] = mL_in
    params['t0'] = t0_in
    params['xS0_E'] = xS0_in[0]
    params['xS0_N'] = xS0_in[1]
    params['beta'] = beta_in
    params['muS_E'] = muS_in[0]
    params['muS_N'] = muS_in[1]
    params['muL_E'] = muL_in[0]
    params['muL_N'] = muL_in[1]
    params['dL'] = dL_in
    params['dS'] = dS_in
    params['b_sff1'] = b_sff_in
    params['mag_src1'] = mag_src_in

    # Extra parameters
    params['dL_dS'] = params['dL'] / params['dS']
    params['tE'] = pspl_par_in.tE
    params['thetaE'] = pspl_par_in.thetaE_amp
    params['piE_E'] = pspl_par_in.piE[0]
    params['piE_N'] = pspl_par_in.piE[1]
    params['u0_amp'] = pspl_par_in.u0_amp
    params['muRel_E'] = pspl_par_in.muRel[0]
    params['muRel_N'] = pspl_par_in.muRel[1]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model_fitter.plot_photometry(data, pspl_par_in, dense_time=True)
    plt.figure(1)
    plt.title('Input Data and Model')
    plt.savefig(outdir + target + '_fake_data_phot.png')

    model_fitter.plot_astrometry(data, pspl_par_in, dense_time=True)
    plt.figure(2)
    plt.title('Input Data and Model')
    plt.savefig(outdir + target +  '_fake_data_ast.png')

    plt.figure(3)
    plt.title('Input Data and Model')
    plt.savefig(outdir + target + '_fake_data_t_vs_E.png')

    plt.figure(4)
    plt.title('Input Data and Model')
    plt.savefig(outdir + target + '_fake_data_t_vs_N.png')

    return data, params


def fake_data_BSPL(outdir='', outroot='bspl',
                   raL=259.5, decL=-29.0,
                   mL=10, t0=57100, beta=1.0,
                   xS0_E=0, xS0_N=0,
                   muL_E=0, muL_N=0,
                   muS_E=3, muS_N=0,
                   dL=4000, dS=8000,
                   sep=3, alpha=70,
                   mag_src_pri=16, mag_src_sec=17,
                   b_sff=1, parallax=True,
                   target='BSPL', animate=False):
    """
    Optional Inputs
    ---------------
    outdir : str
        The output directory where figures and data are saved.
    outroot : str
        The output file name root for a saved figure.
    raL : float (deg)
        The right ascension in degrees. Needed if parallax=True.
    decL : float (deg)
        The declination in degrees. Needed if parallax=False.
    mL : float (Msun)
        Mass of the  lens.
    t0 : float (mjd)
        The time of closest projected approach between the source
        and the geometric center of the lens system in heliocentric
        coordinates.
    beta : float (mas)
        The closest projected approach between the source
        and the geometric center of the lens system in heliocentric
        coordinates.
    xS0_E : float (arcsec)
        Position of the source in RA relative to the
        geometric center of the lens system at time t0.
    xS0_N : float (arcsec)
        Position of the source in Dec relative to the
        geometric center of the lens system at time t0.
    muL_E : float (mas/yr)
        Proper motion of the lens system in RA direction
    muL_N : float (mas/yr)
        Proper motion of the lens system in the Dec direction
    muS_E : float (mas/yr)
        Proper motion of the source in the RA direction
    muS_N : float (mas/yr)
        Proper motion of the source in the Dec direction
    dL : float (pc)
        Distance to the lens system
    dS : float (pc)
        Distance to the source
    sep : float (mas)
        Separation between the binary source stars,
        projected onto the sky.
    alpha : float (degrees)
        Angle of the project binary separation vector on the
        sky. The separation vector points from the primary
        to the secondary and the angle alpha is measured in
        degrees East of North.
    mag_src_pri : float (mag)
        Brightness of the primary source star.
    mag_src_sec : float (mag)
        Brightness of the secondary source star.
    b_sff : float
        Source flux fraction = fluxS / (fluxS + fluxL1 + fluxL2 + fluxN)

    """

    start = time.time()
    if parallax:
        bspl = model.BSPL_PhotAstrom_Par_Param1(mL, t0, beta,
                                                dL, dL/dS,
                                                xS0_E, xS0_N,
                                                muL_E, muL_N, muS_E, muS_N,
                                                sep, alpha,
                                                np.array([mag_src_pri]),
                                                np.array([mag_src_sec]),
                                                np.array([b_sff]),
                                                raL=raL, decL=decL)
    else:
        bspl = model.BSPL_PhotAstrom_noPar_Param1(mL, t0, beta,
                                                dL, dL/dS,
                                                xS0_E, xS0_N,
                                                muL_E, muL_N, muS_E, muS_N,
                                                sep, alpha,
                                                np.array([mag_src_pri]),
                                                np.array([mag_src_sec]),
                                                np.array([b_sff]))

    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_pho = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(54000, 60000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_pho_new = np.arange(year_start + phot_start,
                              year_start + phot_start + phot_win, 1)
        t_pho = np.concatenate([t_pho, t_pho_new])

        ast_win = 120.0
        ast_start = (365.25 - ast_win) / 2.0
        t_ast_new = np.arange(year_start + ast_start,
                              year_start + ast_start + ast_win, 28)
        t_ast = np.concatenate([t_ast, t_ast_new])

    t_mod = np.arange(t_pho.min(), t_pho.max(), 1)

    imag_pho = bspl.get_photometry(t_pho)
    imag_mod = bspl.get_photometry(t_mod)

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 400.0
    imag0 = 19.0

    flux_pho = flux0 * 10 ** ((imag_pho - imag0) / -2.5)
    flux_pho_err = flux_pho ** 0.5
    flux_pho += np.random.randn(len(t_pho)) * flux_pho_err
    imag_pho = -2.5 * np.log10(flux_pho / flux0) + imag0
    imag_pho_err = 1.087 / flux_pho_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs.
    lens_pos = bspl.get_lens_astrometry(t_mod)
    srce_pos = bspl.get_astrometry_unlensed(t_mod)
    srce_pos_lensed_res = bspl.get_resolved_astrometry(t_mod)
    srce_pos_lensed_unres = bspl.get_astrometry(t_mod)

    srce_pos_lensed_res = np.ma.masked_invalid(srce_pos_lensed_res)

    stop = time.time()

    fmt = 'It took {0:.2f} seconds to evaluate the model at {1:d} time steps'
    print(fmt.format(stop - start, len(t_mod) + len(t_ast) + len(t_pho)))

    ##########
    # Plot photometry
    ##########
    plt.figure(1)
    plt.clf()
    plt.errorbar(t_pho, imag_pho, yerr=imag_pho_err, fmt='k.', label='Sim Obs',
                 alpha=0.2)
    plt.plot(t_mod, imag_mod, color='red', label='Model')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (MJD)')
    plt.ylabel('I (mag)')
    plt.legend()

    ##########
    # Plot astrometry
    ##########
    plt.figure(2)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1],
             c='gray', marker='.', linestyle='none', alpha=0.2,
             label='lens system')
    plt.scatter(srce_pos[:, 0], srce_pos[:, 1],
                c=t_mod, marker='.', s=2, alpha=0.2,
                label='src unlensed')

    colors = ['navy', 'blue', 'slateblue', 'darkslateblue', 'indigo']
    for ii in range(srce_pos_lensed_res.shape[1]):
        plt.plot(srce_pos_lensed_res[:, ii, 0], srce_pos_lensed_res[:, ii, 1],
                 c=colors[ii], linestyle='none', marker='.', markersize=1,
                 alpha=0.5,
                 label='src lensed img{0:d}'.format(ii + 1))

    plt.plot(srce_pos_lensed_unres[:, 0], srce_pos_lensed_unres[:, 1],
             c='red', linestyle='-',
             label='src lensed unres')

    pos_ast_tmp = bspl.get_astrometry(t_ast)
    pos_ast_err = np.ones((len(t_ast), 2), dtype=float) * 0.15 * 1e-3
    pos_ast = pos_ast_tmp + pos_ast_err *  np.random.randn(len(t_ast), 2)

    plt.errorbar(pos_ast[:, 0], pos_ast[:, 1],
                 xerr=pos_ast_err[:, 0], yerr=pos_ast_err[:, 0],
                 marker='.', color='black', alpha=0.2)

    plt.gca().invert_xaxis()
    plt.xlabel(r'$\Delta \alpha^*$ (mas)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.legend(fontsize=8)
    plt.subplots_adjust(left=0.25, top=0.8)

    p2 = plt.gca().get_position().get_points().flatten()
    ax_cbar = plt.gcf().add_axes([p2[0], 0.82, p2[2] - p2[0], 0.05])
    plt.colorbar(cax=ax_cbar, orientation='horizontal', label='Time (MJD)',
                 ticklocation='top')

    data = {}
    data['target'] = target
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'
    data['phot_files'] = ['fake_data_parallax_phot1']
    data['ast_files'] = ['fake_data_parallax_ast1']

    data['t_phot1'] = t_pho
    data['mag1'] = imag_pho
    data['mag_err1'] = imag_pho_err

    data['t_ast1'] = t_ast
    data['xpos1'] = pos_ast[:, 0]
    data['ypos1'] = pos_ast[:, 1]
    data['xpos_err1'] = pos_ast_err[:, 0]
    data['ypos_err1'] = pos_ast_err[:, 1]

    data['raL'] = raL
    data['decL'] = decL

    params = {}
    params['mL'] = mL
    params['beta'] = beta
    params['sep'] = sep
    params['alpha'] = alpha
    params['t0'] = t0
    params['xS0_E'] = xS0_E
    params['xS0_N'] = xS0_N
    params['muS_E'] = muS_E
    params['muS_N'] = muS_N
    params['muL_E'] = muL_E
    params['muL_N'] = muL_N
    params['dL'] = dL
    params['dS'] = dS
    params['b_sff'] = np.array([b_sff])
    params['mag_src_pri'] = np.array([mag_src_pri])
    params['mag_src_sec'] = np.array([mag_src_sec])

    out_name = outdir + outroot + '_movie.gif'
    if animate:
        ani = plot_models.animate_PSBL(bspl, outfile=out_name)
    else:
        ani = None

    #    np.savetxt('fake_data_PSBL_phot.dat', (data['t_phot1'], data['mag1'], data['mag_err1']))

    return data, params, bspl, ani
