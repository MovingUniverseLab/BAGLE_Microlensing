import pdb
from bagle import model
import numpy as np
import matplotlib.pyplot as plt
from bagle import frame_convert as fc
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table

from matplotlib import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def blah():
    t = Table.read('/u/jlu/work/microlens/OB110462/a_2022_03_01/model_fits/pspl/ogle_hst_phot_gp/base_b/b0_post_equal_weights.dat', format='ascii')
    t = Table.read('/u/jlu/work/microlens/OB110462/a_2021_12_20/model_fits/ogle_hst_phot/base_a/a0_post_equal_weights.dat', format='ascii')
    ra = '17:51:40.19'
    dec = '-29:53:26.3'

    t0_h = t['col1']
    u0_h = t['col2']
    tE_h = t['col3']
    piEE_h = t['col4']
    piEN_h = t['col5']
    t0par = 55763.327

    output_arr = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                           tE_h, piEE_h, piEN_h,
                                           t0par, in_frame='helio',
                                           murel_in='SL', murel_out='LS',
                                           coord_in='EN', coord_out='tb',
                                           plot=False)

    t0_g, u0_g, tE_g, piEE_g, piEN_g = output_arr

    fig, ax = plt.subplots(2, 5, figsize=(10, 6))
    ax[0,0].hist(t0_h)
    ax[0,1].hist(u0_h)
    ax[0,2].hist(tE_h)
    ax[0,3].hist(piEE_h)
    ax[0,4].hist(piEN_h)

    ax[1,0].hist(t0_g)
    ax[1,1].hist(u0_g)
    ax[1,2].hist(tE_g)
    ax[1,3].hist(piEE_g)
    ax[1,4].hist(piEN_g)
    plt.show()

def test_array_input():
    ra = 259.0
    dec = -29.0
    t0_h = 57000 * np.ones(8)
    u0_h = [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]
    tE_h = 300 #* np.ones(8)
    piEE_h = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
    piEN_h = np.array([0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1])
    t0par = 57100

    output_arr = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                       tE_h, piEE_h, piEN_h,
                                       t0par, in_frame='helio',
                                       murel_in='SL', murel_out='LS',
                                       coord_in='EN', coord_out='tb',
                                       plot=False)

    t0_g_arr, u0_g_arr, tE_g_arr, piEE_g_arr, piEN_g_arr = output_arr

    for ii in np.arange(8):
        print(ii)
        output = fc.convert_helio_geo_phot(ra, dec, t0_h[ii], u0_h[ii], 
                                           tE_h,#[ii], 
                                           piEE_h[ii], piEN_h[ii],
                                           t0par, in_frame='helio',
                                           murel_in='SL', murel_out='LS',
                                           coord_in='EN', coord_out='tb',
                                           plot=False)

        t0_g, u0_g, tE_g, piEE_g, piEN_g = output
        
        assert t0_g == t0_g_arr[ii]
        assert u0_g == u0_g_arr[ii]
        assert tE_g == tE_g_arr[ii]
        assert piEE_g == piEE_g_arr[ii]
        assert piEN_g == piEN_g_arr[ii]
        
def compare_mulens_to_bagle_psbl_phot(ra, dec, t_mjd,
                                      t0_m, u0_m, tE_m,
                                      piEE_m, piEN_m, t0par,
                                      q_m, alpha_m, sep_m,
                                      return_mag=False,
                                      plot_lc=False,
                                      plot_conv=False,
                                      verbose=False):
    """
    return_mag : bool
        True : lightcurve in  magnitudes assuming an unblended 22nd mag source.
        False : lightcurve in magnification units.
    """

    output = fc.convert_bagle_mulens_psbl_phot(ra, dec,
                                               t0_m, u0_m, tE_m,
                                               piEE_m, piEN_m, t0par,
                                               q_m, alpha_m, sep_m,
                                               mod_in='mulens', plot=plot_conv)
    t0_b, u0_b, tE_b, piEE_b, piEN_b, q_b, alpha_b, sep_b = output

    # Get HJD from MJD (since MulensModel uses HJD).
    t_hjd = t_mjd + 2400000.5

    # Turn RA and Dec into SkyCoord object (for MulensModel).
    coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))

    # Get the lightcurve from the geoprojected parameters.
    mag_mulens = get_phot_mulens_psbl(coords, t0_m, u0_m, tE_m,
                                      piEE_m, piEN_m, t0par,
                                      alpha_m, q_m, sep_m, t_hjd,
                                      return_mag=return_mag)

    mag_pylima = get_phot_pylima_psbl(ra, dec, t0_m, u0_m, tE_m,
                                      piEE_m, piEN_m, t0par,
                                      alpha_m, q_m, sep_m, t_hjd,
                                      return_mag=return_mag)

    # Get the lightcurve from the heliocentric parameters.
    mag_bagle, mod_bagle = get_phot_bagle_psbl(ra, dec, t0_b, u0_b,
                                               tE_b, piEE_b, piEN_b,
                                               alpha_b, q_b, sep_b, t_mjd,
                                               return_mag=return_mag)

    if verbose:
        print('Mulens Params:')
        print(t0_m, u0_m, tE_m,
              piEE_m, piEN_m, t0par,
              q_m, alpha_m, sep_m)

        print('PyLIMA Params:')
        print(t0_m, u0_m, tE_m,
              piEE_m, piEN_m, t0par,
              q_m, alpha_m, sep_m)

        print('BAGLE Params:')
        print(t0_b, u0_b, tE_b,
              piEE_b, piEN_b,
              q_b, alpha_b, sep_b)

    diff_mulens = mag_mulens - mag_bagle
    diff_pylima = mag_pylima - mag_bagle

    if plot_lc:
        fig, ax = plt.subplots(3, 1, figsize=(8,8), sharex=True, num=3)
        plt.clf()
        fig, ax = plt.subplots(3, 1, figsize=(8,8), sharex=True, num=3)
        ax[0].plot(t_mjd, mag_mulens, '-', label='M')
        ax[0].plot(t_mjd, mag_pylima, '-', label='P')
        ax[0].plot(t_mjd, mag_bagle, '-', label='B')
        ax[0].legend()
        ax[1].plot(t_mjd, diff_mulens, '.')
        ax[2].plot(t_mjd, diff_pylima, '.')
        if return_mag:
            ax[0].set_ylabel('Mag')
            ax[1].set_ylabel('Mag Mulens - BAGLE', fontsize=12)
            ax[2].set_ylabel('Mag pyLIMA - BAGLE', fontsize=12)
            ax[0].invert_yaxis()
        else:
            ax[0].set_ylabel('Amp')
            ax[1].set_ylabel('Amp Mulens - BAGLE', fontsize=12)
            ax[2].set_ylabel('Amp pyLIMA - BAGLE', fontsize=12)
            ax[0].set_ylim(0.95, 10)
            ax[1].set_ylim(-0.4, 0.4)
            ax[2].set_ylim(-0.4, 0.4)
        ax[2].set_xlabel('MJD')

        # Print mean and max difference
        ax[1].set_title(f'Diff mean={diff_mulens.mean():.3e}, max={np.abs(diff_mulens).max():.3e}',
                        fontsize=12)
        ax[2].set_title(f'Diff mean={diff_pylima.mean():.3e}, max={np.abs(diff_pylima).max():.3e}',
                        fontsize=12)


    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-1 magnitudes on average.
    diff = (mag_mulens - mag_bagle)/mag_mulens
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-4

    return


def compare_bagle_to_mulens_psbl_phot(ra, dec, t_mjd,
                                      t0_b, u0_b, tE_b,
                                      piEE_b, piEN_b, t0par,
                                      q_b, alpha_b, sep_b,
                                      return_mag=False,
                                      plot_lc=False,
                                      plot_conv=False,
                                      verbose=False, fignum=3):
    
    # Get the BAGLE lightcurve from the heliocentric parameters.
    mag_bagle, mod_bagle = get_phot_bagle_psbl(ra, dec, t0_b, u0_b,
                                               tE_b, piEE_b, piEN_b,
                                               alpha_b, q_b, sep_b, t_mjd,
                                               return_mag=return_mag)

    output = fc.convert_bagle_mulens_psbl_phot(ra, dec,
                                               t0_b, u0_b, tE_b,
                                               piEE_b, piEN_b, t0par,
                                               q_b, alpha_b, sep_b,
                                               mod_in='bagle', plot=plot_conv)
    t0_m, u0_m, tE_m, piEE_m, piEN_m, q_m, alpha_m, sep_m = output

    output = fc.convert_bagle_pylima_psbl_phot(ra, dec,
                                               t0_b, u0_b, tE_b,
                                               piEE_b, piEN_b, t0par,
                                               q_b, alpha_b, sep_b,
                                               mod_in='bagle', plot=plot_conv)
    t0_p, u0_p, tE_p, piEE_p, piEN_p, q_p, alpha_p, sep_p = output

    # Get HJD from MJD (since MulensModel uses HJD).
    t_hjd = t_mjd + 2400000.5

    # Turn RA and Dec into SkyCoord object (for MulensModel).
    coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))

    # Get the lightcurve from the geoprojected parameters.
    mag_mulens = get_phot_mulens_psbl(coords, t0_m, u0_m, tE_m,
                                      piEE_m, piEN_m, t0par,
                                      alpha_m, q_m, sep_m, t_hjd,
                                      return_mag=return_mag)

    mag_pylima = get_phot_pylima_psbl(ra, dec, t0_p, u0_p, tE_p,
                                      piEE_p, piEN_p, t0par,
                                      alpha_p, q_p, sep_p, t_hjd,
                                      return_mag=return_mag)

    if verbose:
        print('Mulens Params:')
        print(t0_m, u0_m, tE_m,
              piEE_m, piEN_m, t0par,
              q_m, alpha_m, sep_m)

        print('PyLIMA Params:')
        print(t0_p, u0_p, tE_p,
              piEE_p, piEN_p, t0par,
              q_p, alpha_p, sep_p)

        print('BAGLE Params:')
        print(t0_b, u0_b, tE_b,
              piEE_b, piEN_b,
              q_b, alpha_b, sep_b)

    diff_mulens = mag_mulens - mag_bagle
    diff_pylima = mag_pylima - mag_bagle

    if plot_lc:
        fig, ax = plt.subplots(3, 1, figsize=(8,8), sharex=True, num=fignum)
        plt.clf()
        fig, ax = plt.subplots(3, 1, figsize=(8,8), sharex=True, num=fignum)
        ax[0].plot(t_mjd, mag_mulens, '-', label='M')
        ax[0].plot(t_mjd, mag_pylima, '-', label='P')
        ax[0].plot(t_mjd, mag_bagle, '-', label='B')
        ax[0].legend()
        ax[1].plot(t_mjd, diff_mulens, '-')
        ax[2].plot(t_mjd, diff_pylima, '-')
        if return_mag:
            ax[0].set_ylabel('Mag')
            ax[1].set_ylabel('Mag Mulens - Bagle', fontsize=12)
            ax[2].set_ylabel('Mag pyLIMA - BAGLE', fontsize=12)
            ax[0].invert_yaxis()
        else:
            ax[0].set_ylabel('Amp')
            ax[1].set_ylabel('Amp Mulens - BAGLE', fontsize=12)
            ax[2].set_ylabel('Amp pyLIMA - BAGLE', fontsize=12)
            ax[0].set_ylim(0.95, 10)
            ax[1].set_ylim(-0.4, 0.4)
            ax[2].set_ylim(-0.4, 0.4)
        ax[2].set_xlabel('MJD')

        # Print mean and max difference
        ax[1].set_title(f'Diff mean={diff_mulens.mean():.3e}, max={np.abs(diff_mulens).max():.3e}',
                        fontsize=12)
        ax[2].set_title(f'Diff mean={diff_pylima.mean():.3e}, max={np.abs(diff_pylima).max():.3e}',
                        fontsize=12)



#        fig, ax = plt.subplots(2, 1, sharex=True, num=3)
#        plt.clf()
#        fig, ax = plt.subplots(2, 1, sharex=True, num=3)
#        ax[0].plot(t_mjd, mag_pylima, '.', ms=10)
#        ax[0].plot(t_mjd, mag_bagle, '.', ms=5)
#        ax[0].plot(t_mjd, mag_mulens, '.', ms=3)
#        ax[0].invert_yaxis()
#        ax[1].plot(t_mjd, mag_mulens - mag_pylima, '.', ms=10, label='M-P')
#        ax[1].plot(t_mjd, mag_mulens - mag_bagle, '.', ms=5, label='M-B')
#        ax[0].set_ylabel('Mag')
#        ax[1].set_ylabel('Mag Diff')
#        ax[1].set_xlabel('MJD')
#        ax[1].legend()
#        plt.show()
#        plt.pause(0.5)

    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-1 magnitudes on average.
    diff = (diff_mulens) / mag_mulens
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-4


def compare_bagle_geo_to_helio_phot(ra, dec, t_mjd,
                                    t0_g, u0_g, tE_g,
                                    piEE_g, piEN_g,
                                    t0par,
                                    plot_conv=False,
                                    plot_lc=False,
                                    verbose=False):

    """
    Test conversion from geocentric projected frame
    (using lens-source tau-beta convention) to
    heliocentric frame (using source-lens East-North 
    convention) photometry parameters in BAGLE.
    
    Parameters
    ----------
    ra, dec : int or float (degrees)

    t_mjd, t0_g, t0par need to be in MJD.

    t0_g, u0_g, tE_g, piEE_g, piEN_g need to be
    in geocentric projected frame using lens-source
    and tau-beta conventions.
    """
    # Convert the geoprojected parameters (in lens-source tau-beta)
    # to heliocentric parameters (in source-lens East-North).
    output = fc.convert_helio_geo_phot(ra, dec, t0_g, u0_g, 
                                       tE_g, piEE_g, piEN_g,
                                       t0par, in_frame='geo',
                                       murel_in='LS', murel_out='SL',
                                       coord_in='tb', coord_out='EN',
                                       plot=plot_conv)

    t0_h, u0_h, tE_h, piEE_h, piEN_h = output

    # Get the lightcurve from the geoprojected parameters.
    mag_bagle_geoproj = get_phot_bagle_geoproj(ra, dec, t0_g, u0_g, 
                                               tE_g, piEE_g, piEN_g, 
                                               t_mjd, t0par)

    # Get the lightcurve from the heliocentric parameters.
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, 
                               tE_h, piEE_h, piEN_h, t_mjd)

    if verbose:
        label = ['t0 (MJD)', 'u0', 'tE (days)', 'piEE', 'piEN', 't0par']
        geoproj = [t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par]
        helio = [t0_h, u0_h, tE_h, piEE_h, piEN_h]

        print('*******************************************')
        print('Geo proj parameters, tau-beta, L-S frame')
        for gg, _ in enumerate(geoproj):
            print('{0} : {1:.2f}'.format(label[gg], geoproj[gg]))
        print('-------------------------------------------')
        print('Helio parameters, East-North, S-L frame')
        for hh, _ in enumerate(helio):
            print('{0} : {1:.2f}'.format(label[hh], helio[hh]))
        print('*******************************************')

    if plot_lc:
        fig, ax = plt.subplots(2, 1, sharex=True, num=3)
        plt.clf()
        fig, ax = plt.subplots(2, 1, sharex=True, num=3)
        ax[0].plot(t_mjd, mag_bagle_geoproj, 'o', label='Geo proj')
        ax[0].plot(t_mjd, mag_bagle, '.', label='Helio')
        ax[0].legend()
        ax[0].invert_yaxis()
        ax[1].plot(t_mjd, mag_bagle_geoproj - mag_bagle, '.')
        ax[0].set_ylabel('Mag')
        ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[1].set_ylabel('Geo proj - Helio')
        ax[1].set_xlabel('MJD')
        plt.show()
        plt.pause(0.5)

    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-4 magnitudes on average.
    # Note: current test fails if require < 1e-5.
    diff = (mag_bagle_geoproj - mag_bagle)/mag_bagle_geoproj
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-4

def compare_bagle_helio_to_geo_phot(ra, dec, t_mjd,
                                    t0_h, u0_h, tE_h,
                                    piEE_h, piEN_h,
                                    t0par,
                                    plot_conv=False,
                                    plot_lc=False,
                                    verbose=False,
                                    fignum=3):
    """
    Test conversion from heliocentric frame (using 
    source-lens East-North convention) to geocentric 
    projected frame (using lens-source tau-beta convention)
    photometry parameters in BAGLE.

    Parameters
    ----------
    ra, dec : int or float (degrees)

    t_mjd, t0_h, t0par need to be in MJD.
    
    t0_h, u0_h, tE_h, piEE_h, piEN_h need to be
    in heliocentric frame using source-lens
    and East-North conventions.
    """
    # Convert the heliocentric parameters (in source-lens East-North)
    # to geoprojected parameters (in lens-source tau-beta).
    output = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                       tE_h, piEE_h, piEN_h,
                                       t0par, in_frame='helio',
                                       murel_in='SL', murel_out='LS',
                                       coord_in='EN', coord_out='tb',
                                       plot=plot_conv)

    t0_g, u0_g, tE_g, piEE_g, piEN_g = output

    # Get the lightcurve from the geoprojected parameters.
    mag_bagle_geoproj = get_phot_bagle_geoproj(ra, dec, t0_g, u0_g, 
                                               tE_g, piEE_g, piEN_g, 
                                               t_mjd, t0par)

    # Get the lightcurve from the heliocentric parameters.
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, 
                               tE_h, piEE_h, piEN_h, t_mjd)

    if verbose:
        label = ['t0 (MJD)', 'u0', 'tE (days)', 'piEE', 'piEN', 't0par']
        geoproj = [t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par]
        helio = [t0_h, u0_h, tE_h, piEE_h, piEN_h]

        print('*******************************************')
        print('Geo proj parameters, tau-beta, L-S frame')
        for gg, _ in enumerate(geoproj):
            print('{0} : {1:.2f}'.format(label[gg], geoproj[gg]))
        print('-------------------------------------------')
        print('Helio parameters, East-North, S-L frame')
        for hh, _ in enumerate(helio):
            print('{0} : {1:.2f}'.format(label[hh], helio[hh]))
        print('*******************************************')

    if plot_lc:
        fig, ax = plt.subplots(2, 1, sharex=True, num=fignum)
        plt.clf()
        fig, ax = plt.subplots(2, 1, sharex=True, num=fignum)
        ax[0].plot(t_mjd, mag_bagle_geoproj, 'o', label='Geo proj')
        ax[0].plot(t_mjd, mag_bagle, '.', label='Helio')
        ax[0].invert_yaxis()
        ax[0].legend()
        ax[1].plot(t_mjd, mag_bagle_geoproj - mag_bagle, '.')
        ax[0].set_ylabel('Mag')
        ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[1].set_ylabel('Geo proj - Helio')
        ax[1].set_xlabel('MJD')
        plt.show()
        plt.pause(1)

    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-5 magnitudes on average.
    # Note: current test fails if require < 1e-6.
    diff = (mag_bagle - mag_bagle_geoproj)/mag_bagle
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-5
    
def test_convert_u0_t0_psbl():
    """
    Test converting u0 and t0 from primary center coordinate system
    to geometric center coordinate frame. This is done via plugging parameters
    into PSBL_PhotAstrom_Par_Param7, converting them mathematically, then
    plugging them into PSBL_PhotAstrom_Par_Param1, then checking photometry
    is equal.
    """
    mLp = 10
    mLs = 10
    xS0_E = 0 
    xS0_N = 0 
    muL_E = 0
    muL_N = 0
    muS_E = 0
    muS_N = 2
    dL = 4000
    dS = 8000
    sep = 0.1
    alpha = 40
    b_sff = 0.5
    mag_src = 18
    raL = 270
    decL = -30
    dmag_Lp_Ls = 20.0
    
    t0_p = 0
    beta_p = 0.5
    psbl_model = model.PSBL_PhotAstrom_Par_Param7(mLp, mLs, t0_p, xS0_E, xS0_N, beta_p, 
                                                  muL_E, muL_N, muS_E, muS_N, dL, dS,
                                                  sep, alpha, [b_sff], [mag_src], [dmag_Lp_Ls],
                                                  raL=raL, decL=decL, 
                                                  root_tol = 0.00000001)
    
    # Conversions determined by specific geometry (can be checked in u0/t0 conversion overleaf)
    # https://www.overleaf.com/read/qzvfpvkpxdqm
    t0_geom = psbl_model.t0_p - psbl_model.tE*np.cos(np.deg2rad(psbl_model.phi))*(psbl_model.sep/(2*psbl_model.thetaE_amp))
    beta_geom = (psbl_model.u0_amp_p - (-1)*(psbl_model.sep/(2*psbl_model.thetaE_amp))*np.sin(np.deg2rad(psbl_model.phi)))*psbl_model.thetaE_amp
    
    psbl_model_geom = model.PSBL_PhotAstrom_Par_Param1(mLp, mLs, t0_geom, xS0_E, xS0_N, beta_geom, 
                                                       muL_E, muL_N, muS_E, muS_N, dL, dS,
                                                       sep, alpha, [b_sff], [mag_src], [dmag_Lp_Ls],
                                                       raL=raL, decL=decL, 
                                                       root_tol = 0.00000001)
    
    
    dt = np.linspace(-500, 500, 200)
    img, amp = psbl_model.get_all_arrays(dt)
    phot = psbl_model.get_photometry(dt, amp_arr=amp)
    
    img_geom, amp_geom = psbl_model_geom.get_all_arrays(dt)
    phot_geom = psbl_model_geom.get_photometry(dt, amp_arr=amp_geom)
    
    assert(np.allclose(phot, phot_geom, rtol=1e-15))


def compare_bagle_to_mulens(ra, dec, t_mjd,
                            t0_h, u0_h, tE_h,
                            piEE_h, piEN_h,
                            t0par,
                            plot_conv=False,
                            plot_lc=False,
                            verbose=False):
    """
    Test conversion from heliocentric frame (using 
    source-lens East-North convention) in BAGLE to geocentric 
    projected frame (using lens-source tau-beta convention)
    in MulensModel for photometry parameters.

    Parameters
    ----------
    ra, dec : int or float (degrees)

    t_mjd, t0_h, t0par need to be in MJD.

    t0_h, u0_h, tE_h, piEE_h, piEN_h need to be
    in heliocentric frame using source-lens
    and East-North conventions.
    """
    # Convert the heliocentric parameters (in source-lens East-North)
    # to geoprojected parameters (in lens-source tau-beta).
    output = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                       tE_h, piEE_h, piEN_h,
                                       t0par, in_frame='helio',
                                       murel_in='SL', murel_out='LS',
                                       coord_in='EN', coord_out='tb',
                                       plot=plot_conv)

    t0_g, u0_g, tE_g, piEE_g, piEN_g = output

    # Get HJD from MJD (since MulensModel uses HJD).
    t_hjd = t_mjd + 2400000.5
    
    # Turn RA and Dec into SkyCoord object (for MulensModel).
    coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))

    # Get the lightcurve from the geoprojected parameters.
    mag_mulens = get_phot_mulens(coords, t0_g, u0_g, 
                                 tE_g, piEE_g, piEN_g, 
                                 t0par, t_hjd)

    # Get the lightcurve from the heliocentric parameters.
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, 
                               tE_h, piEE_h, piEN_h, t_mjd)
    
    if verbose:
        label = ['t0 (MJD)', 'u0', 'tE (days)', 'piEE', 'piEN', 't0par']
        geoproj = [t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par]
        helio = [t0_h, u0_h, tE_h, piEE_h, piEN_h]

        print('*******************************************')
        print('Geo proj parameters, tau-beta, L-S frame')
        for gg, _ in enumerate(geoproj):
            print('{0} : {1:.2f}'.format(label[gg], geoproj[gg]))
        print('-------------------------------------------')
        print('Helio parameters, East-North, S-L frame')
        for hh, _ in enumerate(helio):
            print('{0} : {1:.2f}'.format(label[hh], helio[hh]))
        print('*******************************************')

    if plot_lc:
        fig, ax = plt.subplots(2, 1, sharex=True, num=3)
        plt.clf()
        fig, ax = plt.subplots(2, 1, sharex=True, num=3)
        ax[0].plot(t_mjd, mag_mulens, 'o', label='Mulens')
        ax[0].plot(t_mjd, mag_bagle, '.', label='BAGLE')
        ax[0].invert_yaxis()
        ax[0].legend()
        ax[1].plot(t_mjd, mag_mulens - mag_bagle, '.')
        ax[0].set_ylabel('Mag')
        ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[1].set_ylabel('Mulens - BAGLE')
        ax[1].set_xlabel('MJD')
        plt.show()
        plt.pause(0.5)

    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-5 magnitudes on average.
    # Note: current test fails if require < 1e-6.
    diff = (mag_bagle - mag_mulens)/mag_bagle
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-5

def compare_mulens_to_bagle(ra, dec, t_mjd,
                            t0_g, u0_g, tE_g,
                            piEE_g, piEN_g,
                            t0par,
                            plot_conv=False,
                            plot_lc=False,
                            verbose=False):
    """
    Test conversion from geocentric projected frame (using 
    lens-source tau-beta convention) in MulensModel to 
    heliocentric frame (using source-lens East-North 
    convention) in BAGLE for photometry parameters.

    Parameters
    ----------
    ra, dec : int or float (degrees)

    t_mjd, t0_g, t0par need to be in MJD.

    t0_g, u0_g, tE_g, piEE_g, piEN_g need to be
    in geocentric projected frame using lens-source
    and tau-beta conventions.
    """
    # Convert the geoprojected parameters (in lens-source tau-beta)
    # to heliocentric parameters (in source-lens East-North).
    output = fc.convert_helio_geo_phot(ra, dec, t0_g, u0_g, 
                                       tE_g, piEE_g, piEN_g,
                                       t0par, in_frame='geo',
                                       murel_in='LS', murel_out='SL',
                                       coord_in='tb', coord_out='EN',
                                       plot=plot_conv)

    t0_h, u0_h, tE_h, piEE_h, piEN_h = output

    # Get HJD from MJD (since MulensModel uses HJD).
    t_hjd = t_mjd + 2400000.5

    # Turn RA and Dec into SkyCoord object (for MulensModel).
    coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))

    # Get the lightcurve from the heliocentric parameters.
    mag_mulens = get_phot_mulens(coords, t0_g, u0_g, 
                                 tE_g, piEE_g, piEN_g, 
                                 t0par, t_hjd)

    # Get the lightcurve from the geoprojected parameters.
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, 
                               tE_h, piEE_h, piEN_h, t_mjd)

    if verbose:
        label = ['t0 (MJD)', 'u0', 'tE (days)', 'piEE', 'piEN', 't0par']
        geoproj = [t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par]
        helio = [t0_h, u0_h, tE_h, piEE_h, piEN_h]

        print('*******************************************')
        print('Geo proj parameters, tau-beta, L-S frame')
        for gg, _ in enumerate(geoproj):
            print('{0} : {1:.2f}'.format(label[gg], geoproj[gg]))
        print('-------------------------------------------')
        print('Helio parameters, East-North, S-L frame')
        for hh, _ in enumerate(helio):
            print('{0} : {1:.2f}'.format(label[hh], helio[hh]))
        print('*******************************************')

    if plot_lc:
        fig, ax = plt.subplots(2, 1, sharex=True, num=3)
        plt.clf()
        fig, ax = plt.subplots(2, 1, sharex=True, num=3)
        ax[0].plot(t_mjd, mag_mulens, 'o', label='Mulens')
        ax[0].plot(t_mjd, mag_bagle, '.', label='BAGLE')
        ax[0].invert_yaxis()
        ax[0].legend()
        ax[1].plot(t_mjd, mag_mulens - mag_bagle, '.')
        ax[0].set_ylabel('Mag')
        ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[1].set_ylabel('Mulens - BAGLE')
        ax[1].set_xlabel('MJD')
        plt.show()
        plt.pause(0.5)

    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-4 magnitudes on average.
    # Note: current test fails if require < 1e-5.
    diff = (mag_mulens - mag_bagle)/mag_mulens
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-4


def get_phot_pylima_psbl(ra, dec, t0_g, u0_g, tE_g,
                         piEE_g, piEN_g, t0par, 
                         alpha_g, q, sep, t_hjd,
                         return_mag=True):
    """
    Compare pyLIMA to our models with some plots. 

    Input parameters are in our conventions and heliocentric coordinate system.
    """
    from pyLIMA import event
    from pyLIMA import telescopes
    from pyLIMA import toolbox as microltoolbox
    from pyLIMA.models import generate_model

    #####
    # PyLIMA setup stuff.
    #####
    # Make fake data for pyLIMA... need this for time array definition.
    pylima_data = np.zeros((len(t_hjd), 3))
    pylima_data[:,0] = t_hjd
    pylima_data[:,1] = np.ones(len(t_hjd))
    pylima_data[:,2] = np.ones(len(t_hjd))

    # Make a telescope object
    pylima_tel = telescopes.Telescope(name='OGLE', camera_filter='I',
                                      lightcurve=pylima_data,
                                      lightcurve_names = ['time', 'mag', 'err_mag'],
                                      lightcurve_units = ['JD', 'mag', 'mag'],
                                      location='Earth',
                                      altitude=1000, longitude=-109.285399, latitude=-27.130)
    pylima_ev = event.Event(ra=ra, dec=dec)
    pylima_ev.name = 'Fubar'
    pylima_ev.telescopes.append(pylima_tel)

    # Set up parameters.
    pylima_t0 = t0_g + 2400000.5
    pylima_u0 = u0_g
    pylima_tE = tE_g
    pylima_piEE = piEE_g 
    pylima_piEN = piEN_g
    pylima_t0_par = t0par + 2400000.5
    pylima_phi = np.deg2rad(alpha_g) # phi in radians
    pylima_q = q
    pylima_s = sep

    # Now finally... create the model.
    pylima_mod = generate_model.create_model('PSBL', pylima_ev,
                                             parallax=['Full', pylima_t0_par],
                                             fancy_parameters=None)

    tmp_params = [pylima_t0, pylima_u0, pylima_tE, 
                  pylima_s, pylima_q, pylima_phi,
                  pylima_piEN, pylima_piEE]
       
    pylima_mod.define_model_parameters()
    pylima_mod.blend_flux_ratio = False
    pylima_mod.astrometry = False
    pylima_mod.orbital_motion_model = ['None']

    mag_src=22
    b_sff=1
    pylima_par = pylima_mod.compute_pyLIMA_parameters(tmp_params)
    pylima_par.fs_OGLE = microltoolbox.brightness_transformation.magnitude_to_flux(mag_src)
    pylima_par.fb_OGLE = pylima_par.fs_OGLE * (1.0 - b_sff) / b_sff

    if return_mag:    
        pylima_lcurve, sf, bf = pylima_mod.compute_the_microlensing_model(pylima_tel, pylima_par)
        pylima_lcurve_mag = microltoolbox.brightness_transformation.flux_to_magnitude(pylima_lcurve)
    else:
        pylima_amp = pylima_mod.model_magnification(pylima_tel, pylima_par)
        pylima_lcurve_mag = pylima_amp

    return pylima_lcurve_mag

def get_phot_mulens(coords, t0_g, u0_g, tE_g, 
                    piEE_g, piEN_g, t0par, t_hjd,
                    return_mag=True):
    """
    Get lightcurve from MulensModel, assuming an 
    unblended 22nd mag source. MulensModel uses the 
    geocentric projected frame, with Gould (tau-beta) 
    and lens-source conventions.
    """
    import MulensModel as mm

    # Set up parameter dictionary. 
    # t_0 and t_0_par need to be in HJD.
    params = {}
    params['t_0'] = t0_g + 2400000.5
    params['t_0_par'] = t0par + 2400000.5
    params['u_0'] = u0_g
    params['t_E'] = tE_g
    params['pi_E_N'] = piEN_g
    params['pi_E_E'] = piEE_g

    # Then instantiate model and get lightcurve.
    my_model = mm.Model(params, coords=coords)

    if return_mag:
        mag_obs = my_model.get_lc(times=t_hjd, source_flux=1, blend_flux=0)
    else:
        mag_obs = my_model.get_magnification(t_hjd)

    return mag_obs


def get_phot_mulens_psbl(coords, t0_g, u0_g, tE_g, 
                         piEE_g, piEN_g, t0par, 
                         alpha_g, q, sep, t_hjd,
                         return_mag=True):
    """
    Get PSBL lightcurve from MulensModel, assuming an 
    unblended 22nd mag source. MulensModel uses the 
    geocentric projected frame, with Gould (tau-beta) 
    and lens-source conventions.
    """
    import MulensModel as mm

    # Set up parameter dictionary. 
    # t_0 and t_0_par need to be in HJD.
    params = {}
    params['t_0'] = t0_g + 2400000.5
    params['t_0_par'] = t0par + 2400000.5
    params['u_0'] = u0_g
    params['t_E'] = tE_g
    params['pi_E_N'] = piEN_g
    params['pi_E_E'] = piEE_g
    params['q'] = q
    params['s'] = sep
    params['alpha'] = alpha_g

    # Then instantiate model and get lightcurve.
    my_model = mm.Model(params, coords=coords)

    if return_mag:
        mag_obs = my_model.get_lc(times=t_hjd, source_flux=1, blend_flux=0)
    else:
        mag_obs = my_model.get_magnification(t_hjd)
    return mag_obs


def get_phot_bagle(ra, dec, t0_h, u0_h, 
                   tE_h, piEE_h, piEN_h, t_mjd,
                   return_mag=True):
    """
    Get lightcurve from BAGLE, assuming an unblended 
    22nd mag source. This BAGLE model uses the 
    heliocentric frame, with Lu (East-North) and
    source-lens conventions.
    """
    # Set up parameter dictionary.
    # t0 needs to be in MJD.
    params = {}
    params['raL'] = ra
    params['decL'] = dec
    params['t0'] = t0_h
    params['u0_amp'] = u0_h
    params['tE'] = tE_h
    params['piE_E'] = piEE_h
    params['piE_N'] = piEN_h
    params['b_sff'] = 1
    params['mag_src'] = 22
    
    # Then instantiate model and get lightcurve.
    mod = model.PSPL_Phot_Par_Param1(params['t0'], params['u0_amp'], 
                                     params['tE'],
                                     params['piE_E'], params['piE_N'],
                                     params['b_sff'], params['mag_src'],
                                     raL=params['raL'], decL=params['decL'])

    if return_mag:
        mag_obs = mod.get_photometry(t_mjd)
    else:
        mag_obs = mod.get_amplification(t_mjd)

    return mag_obs


def get_phot_bagle_psbl(ra, dec, t0_h, u0_h,
                        tE_h, piEE_h, piEN_h, 
                        alpha_h, q, sep, t_mjd,
                        return_mag=True):
    """
    Get PSBL lightcurve from BAGLE, assuming an unblended 
    22nd mag source. This BAGLE model uses the 
    heliocentric frame, with Lu (East-North) and
    source-lens conventions.
    """
    # Set up parameter dictionary.
    # t0 needs to be in MJD.
    params = {}
    params['raL'] = ra
    params['decL'] = dec
    params['t0'] = t0_h
    params['u0_amp'] = u0_h
    params['tE'] = tE_h
    params['piE_E'] = piEE_h
    params['piE_N'] = piEN_h
    params['q'] = q
    params['alpha'] = alpha_h
    params['sep'] = sep
    params['b_sff'] = 1
    params['mag_src'] = 22
    
    # Then instantiate model and get lightcurve.
    # FIXME notational consistency: alpha or phi? (or are they the same?)
    mod = model.PSBL_Phot_Par_Param1(params['t0'], params['u0_amp'], 
                                     params['tE'],
                                     params['piE_E'], params['piE_N'],
                                     params['q'], params['sep'], 
                                     params['alpha'],
                                     params['b_sff'], params['mag_src'],
                                     raL=params['raL'], decL=params['decL'])

    if return_mag:
        mag_obs = mod.get_photometry(t_mjd)
    else:
        mag_obs = mod.get_amplification(t_mjd)

    return mag_obs, mod


def get_phot_bagle_geoproj(ra, dec, t0_h, u0_h, tE_h, 
                           piEE_h, piEN_h, t_mjd, t0par,
                           return_mag=True):
    """
    Get lightcurve from BAGLE, assuming an unblended 
    22nd mag source. This BAGLE model uses the    
    geocentric projected frame, with Gould (tau-beta) 
    and lens-source conventions.
    """
    # Set up parameter dictionary.
    # t0 and t0par need to be in MJD.
    params = {}
    params['raL'] = ra
    params['decL'] = dec
    params['t0'] = t0_h
    params['u0_amp'] = u0_h
    params['tE'] = tE_h
    params['piE_E'] = piEE_h
    params['piE_N'] = piEN_h
    params['b_sff'] = 1
    params['mag_src'] = 22
    params['t0par'] = t0par

    # Then instantiate model and get lightcurve.
    mod = model.PSPL_Phot_Par_Param1_geoproj(params['t0'], params['u0_amp'],
                                             params['tE'],
                                             params['piE_E'], params['piE_N'],
                                             params['b_sff'], params['mag_src'],
                                             params['t0par'],
                                             raL=params['raL'], decL=params['decL'])
    if return_mag:
        mag_obs = mod.get_photometry(t_mjd)
    else:
        mag_obs = mod.get_amplification(t_mjd)

    return mag_obs


def test_bagle_mulens_psbl_phot_set(plot_lc=False, plot_conv=False, verbose=False):

    kwargs = {'plot_lc': plot_lc, 'plot_conv': plot_conv, 'verbose': verbose}

    t_mjd = np.arange(60500 - 500, 60500 + 500, 0.5)

    print('set 1')
    compare_bagle_to_mulens_psbl_phot(259.0, -29.0, t_mjd, 60478, 0.5, 100, 0.2, 0.1, 60500, 0.5, 125, 1.5, **kwargs)
    compare_bagle_to_mulens_psbl_phot(259.0, -29.0, t_mjd, 60478, 0.5, 300, 0.2, 0.1, 60478, 0.5, 90, 1.5, **kwargs)
    compare_bagle_to_mulens_psbl_phot(259.0, -29.0, t_mjd, 60478, 0.3, 50, 0.2, 0.1, 60500, 2, 339, 1.5, **kwargs)
    compare_bagle_to_mulens_psbl_phot(259.0, -29.0, t_mjd, 60478, 0.1, 300, 0.2, 0.1, 60478, 0.5, 90, 1.5, **kwargs)
    compare_bagle_to_mulens_psbl_phot(259.0, -29.0, t_mjd, 60478, 0.1, 50, 0.2, 0.1, 60500, 2, 339, 1.5, **kwargs)

    print('set 2')
    compare_mulens_to_bagle_psbl_phot(259, -29, t_mjd,
                                      60461.11916524347, 0.2580016909695294, 77.22238744939838,
                                      -0.21245185338062486, -0.06974388858629471,
                                60500, 0.5, -116.60896987280785, 1.5, **kwargs)
    compare_mulens_to_bagle_psbl_phot(259.0, -29.0, t_mjd, 60478, 0.5, 300, 0.2, 0.1, 60200, 0.5, 90, 1.5, **kwargs)
    compare_mulens_to_bagle_psbl_phot(259.0, -29.0, t_mjd, 60478, 0.3, 50, 0.2, 0.1, 60300, 2, 339, 1.5, **kwargs)

    return

def test_bagle_mulens_set(plot_lc=False, plot_conv=False, verbose=False):
    """
    Test BAGLE --> Mulens as well as Mulens --> BAGLE 
    conversions for all combos of u0/piEE/piEN sign,
    as well as three sets of t0par (<, =, and > t0).
    """
    t_mjd = np.arange(57000 - 500, 57000 + 500, 1)

    kwargs = {'plot_lc' : plot_lc, 'plot_conv' : plot_lc, 'verbose' : verbose}

    print('set 1')
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57100, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57100, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57100, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57100, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57100, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57100, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57100, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57100, **kwargs)
    
    print('set 2')
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57100, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57100, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57100, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57100, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57100, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57100, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57100, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57100, **kwargs)
    
    print('set 3')
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57000, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57000, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57000, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57000, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57000, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57000, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57000, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57000, **kwargs)
    
    print('set 4')
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57000, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57000, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57000, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57000, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57000, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57000, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57000, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57000, **kwargs)
    
    print('set 5')
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 69900, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 69900, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 69900, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 69900, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 69900, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 69900, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 69900, **kwargs)
    compare_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 69900, **kwargs)
    
    print('set 6')
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 69900, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 69900, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 69900, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 69900, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 69900, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 69900, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 69900, **kwargs)
    compare_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 69900, **kwargs)


def test_bagle_helio_geo_set(plot_lc=False, plot_conv=False, verbose=False):
    """
    Test BAGLE's helio --> geo as well as geo --> helio
    conversions for all combos of u0/piEE/piEN sign,
    as well as three sets of t0par (<, =, and > t0).
    """
    t_mjd = np.arange(57000 - 500, 57000 + 500, 1)

    kwargs = {'plot_lc' : plot_lc, 'plot_conv' : plot_conv, 'verbose' : verbose}

#    one example to test (make figure for notes).
#    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57000, **kwargs)
#    pdb.set_trace()

    print('set 1') 

    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57100, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57100, **kwargs)
    #compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57100, **kwargs)
    #compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57100, **kwargs)
    #compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57100, **kwargs)
    #compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57100, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57100, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57100, **kwargs)
    
    print('set 2')
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57100, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57100, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57100, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57100, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57100, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57100, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57100, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57100, **kwargs)
    
    print('set 3')
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57000, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57000, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57000, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57000, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57000, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57000, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57000, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57000, **kwargs)
    
    print('set 4')
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57000, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57000, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57000, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57000, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57000, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57000, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57000, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57000, **kwargs)
    
    print('set 5')
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 69900, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 69900, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 69900, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 69900, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 69900, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 69900, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 69900, **kwargs)
    compare_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 69900, **kwargs)
    
    print('set 6')
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 69900, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 69900, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 69900, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 69900, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 69900, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 69900, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 69900, **kwargs)
    compare_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 69900, **kwargs)



################################################################################
###########  Stuff below is incomplete/wrong/not yet functional...  ############
################################################################################
# def compare_geo_geo():
#     t0_h, u0_h, tE_h, piEE_h, piEN_h = 57000, 0.5, 300, 0.2, 0.1
#
#     t0par1 = 57050
#     t0par2 = 56950
#
#     t0_g1, u0_g1, tE_g1, piEE_g1, piEN_g1 = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h,
#                                                                       tE_h, piEE_h, piEN_h,
#                                                                       t0par1, in_frame='helio',
#                                                                       murel_in='SL', murel_out='LS',
#                                                                       coord_in='EN', coord_out='tb')
#
#     t0_g2, u0_g2, tE_g2, piEE_g2, piEN_g2 = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h,
#                                                                       tE_h, piEE_h, piEN_h,
#                                                                       t0par2, in_frame='helio',
#                                                                       murel_in='SL', murel_out='LS',
#                                                                       coord_in='EN', coord_out='tb')
#
#     mag_mulens1 = get_phot_mulens(coords, t0_g1, u0_g1, tE_g1, piEE_g1, piEN_g1, t0par1, t_hjd)
#     mag_mulens2 = get_phot_mulens(coords, t0_g2, u0_g2, tE_g2, piEE_g2, piEN_g2, t0par2, t_hjd)
#     mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd)
#
#     print(t0_g1, u0_g1, tE_g1, piEE_g1, piEN_g1, t0par1)
#     print(t0_g2, u0_g2, tE_g2, piEE_g2, piEN_g2, t0par2)
#
#     # For a formal test, require this to be smaller than some number.
#     # x = np.sum(np.abs(mag_mulens - mag_bagle))
#
#     fig, ax = plt.subplots(2, 1, sharex=True, num=3)
#     plt.clf()
#     fig, ax = plt.subplots(2, 1, sharex=True, num=3)
#     ax[0].plot(t_mjd, mag_mulens1, 'o')
#     ax[0].plot(t_mjd, mag_mulens2, 'o')
#     ax[0].plot(t_mjd, mag_bagle, '.')
#     ax[0].invert_yaxis()
#     ax[1].plot(t_mjd, mag_mulens1 - mag_bagle, '.')
#     ax[1].plot(t_mjd, mag_mulens2 - mag_bagle, '.')
#     ax[0].set_ylabel('Mag')
#     ax[1].set_ylabel('MM - Bagle')
#     ax[1].set_xlabel('MJD')
#     plt.show()
#     plt.pause(1)
# #    plt.close()
#
# def compare_geo():
#     t0_h, u0_h, tE_h, piEE_h, piEN_h = 57000, 0.5, 300, 0.2, 0.1
#
#     t0par_arr = np.arange(57000 - 720, 57000 + 720, 1)
#     t0_arr = np.zeros(len(t0par_arr))
#     u0_arr = np.zeros(len(t0par_arr))
#     tE_arr = np.zeros(len(t0par_arr))
#     piEE_arr = np.zeros(len(t0par_arr))
#     piEN_arr = np.zeros(len(t0par_arr))
#
#     mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd)
#
#     fig, ax = plt.subplots(5, 1, sharex=True, num=3, figsize=(6,10))
#     plt.clf()
#     fig, ax = plt.subplots(5, 1, sharex=True, num=3, figsize=(6,10))
#     plt.subplots_adjust(left=0.25, top=0.98, bottom=0.1)
#
#     for ii, t0par in enumerate(t0par_arr):
#         t0_g, u0_g, tE_g, piEE_g, piEN_g = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h,
#                                                                           tE_h, piEE_h, piEN_h,
#                                                                           t0par, in_frame='helio',
#                                                                           murel_in='SL', murel_out='LS',
#                                                                           coord_in='EN', coord_out='tb')
#
#         t0_arr[ii] = t0_g
#         u0_arr[ii] = u0_g
#         tE_arr[ii] = tE_g
#         piEE_arr[ii] = piEE_g
#         piEN_arr[ii] = piEN_g
#
#         mag_mulens = get_phot_mulens(coords, t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par, t_hjd)
#
# #        print(np.sum(np.abs(mag_mulens - mag_bagle)))
#
#     ax[0].plot(t0par_arr, t0_arr, 'k-', label='Geo projected')
#     ax[1].plot(t0par_arr, u0_arr, 'k-')
#     ax[2].plot(t0par_arr, tE_arr, 'k-')
#     ax[3].plot(t0par_arr, piEE_arr, 'k-')
#     ax[4].plot(t0par_arr, piEN_arr, 'k-')
#
#     ax[0].axhline(y=t0_h, ls=':', color='b', label='Helio')
#     ax[1].axhline(y=u0_h, ls=':', color='b')
#     ax[2].axhline(y=tE_h, ls=':', color='b')
#     ax[3].axhline(y=-1*piEE_h, ls=':', color='b')
#     ax[4].axhline(y=-1*piEN_h, ls=':', color='b')
#
#     ax[0].set_ylabel('$t_0$ (MJD)')
#     ax[1].set_ylabel('$u_0$')
#     ax[2].set_ylabel('$t_E$ (days)')
#     ax[3].set_ylabel('$\pi_{E,E}$ (LS)')
#     ax[4].set_ylabel('$\pi_{E,N}$ (LS)')
#     ax[4].set_xlabel('$t_{0,par}$ (MJD)')
#
#     ax[0].legend()
#     plt.show()
