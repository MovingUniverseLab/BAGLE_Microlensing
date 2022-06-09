import pdb
from microlens.jlu import model
import numpy as np
import matplotlib.pyplot as plt
import MulensModel as mm
from microlens.jlu import frame_convert as fc
from astropy.coordinates import SkyCoord
from astropy import units as u

ra = 259.0
dec = -29.0
coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))
t_mjd = np.arange(57000 - 500, 57000 + 500, 1)
t_hjd = t_mjd + 2400000.5

def test_bagle_geoproj_to_bagle(t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par):
    t0_h, u0_h, tE_h, piEE_h, piEN_h = fc.convert_helio_geo_phot(ra, dec, t0_g, u0_g, 
                                                                 tE_g, piEE_g, piEN_g,
                                                                 t0par, in_frame='geo',
                                                                 murel_in='LS', murel_out='SL',
                                                                 coord_in='tb', coord_out='EN')


    mag_bagle_geoproj = get_phot_bagle_geoproj(ra, dec, t0_g, u0_g, tE_g, piEE_g, piEN_g, t_mjd, t0par)
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd)

    # For a formal test, require this to be smaller than some number.
    # x = np.sum(np.abs(mag_mulens - mag_bagle))

    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    plt.clf()
    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    ax[0].plot(t_mjd, mag_bagle_geoproj, 'o')
    ax[0].plot(t_mjd, mag_bagle, '.')
    ax[0].invert_yaxis()
    ax[1].plot(t_mjd, mag_bagle_geoproj - mag_bagle, '.')
    ax[0].set_ylabel('Mag')
    ax[1].set_ylabel('Bagle Geoproj - Bagle')
    ax[1].set_xlabel('MJD')
    plt.show()
    plt.pause(0.5)
#    plt.pause(1)
#    plt.close()


def test_bagle_to_bagle_geoproj(t0_h, u0_h, tE_h, piEE_h, piEN_h, t0par):
    t0_g, u0_g, tE_g, piEE_g, piEN_g = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                                                 tE_h, piEE_h, piEN_h,
                                                                 t0par, in_frame='helio',
                                                                 murel_in='SL', murel_out='LS',
                                                                 coord_in='EN', coord_out='tb')

    mag_bagle_geoproj = get_phot_bagle_geoproj(ra, dec, t0_g, u0_g, tE_g, piEE_g, piEN_g, t_mjd, t0par)
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd)
    
    # For a formal test, require this to be smaller than some number.
    # x = np.sum(np.abs(mag_mulens - mag_bagle))

    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    plt.clf()
    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    ax[0].plot(t_mjd, mag_bagle_geoproj, 'o')
    ax[0].plot(t_mjd, mag_bagle, '.')
    ax[0].invert_yaxis()
    ax[1].plot(t_mjd, mag_bagle_geoproj - mag_bagle, '.')
    ax[0].set_ylabel('Mag')
    ax[1].set_ylabel('Bagle Geoproj - Bagle')
    ax[1].set_xlabel('MJD')
    plt.show()
    plt.pause(0.5)
#    plt.pause(1)
#    plt.close()


def test_bagle_to_mulens(t0_h, u0_h, tE_h, piEE_h, piEN_h, t0par):
    t0_g, u0_g, tE_g, piEE_g, piEN_g = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                                                 tE_h, piEE_h, piEN_h,
                                                                 t0par, in_frame='helio',
                                                                 murel_in='SL', murel_out='LS',
                                                                 coord_in='EN', coord_out='tb')

    mag_mulens = get_phot_mulens(coords, t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par, t_hjd)
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd)
    
    # For a formal test, require this to be smaller than some number.
    # x = np.sum(np.abs(mag_mulens - mag_bagle))

    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    plt.clf()
    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    ax[0].plot(t_mjd, mag_mulens, 'o')
    ax[0].plot(t_mjd, mag_bagle, '.')
    ax[0].invert_yaxis()
    ax[1].plot(t_mjd, mag_mulens - mag_bagle, '.')
    ax[0].set_ylabel('Mag')
    ax[1].set_ylabel('MM - Bagle')
    ax[1].set_xlabel('MJD')
    plt.show()
    plt.pause(0.5)
#    plt.pause(1)
#    plt.close()

def test_mulens_to_bagle(t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par):
    t0_h, u0_h, tE_h, piEE_h, piEN_h = fc.convert_helio_geo_phot(ra, dec, t0_g, u0_g, 
                                                                 tE_g, piEE_g, piEN_g,
                                                                 t0par, in_frame='geo',
                                                                 murel_in='LS', murel_out='SL',
                                                                 coord_in='tb', coord_out='EN')


    mag_mulens = get_phot_mulens(coords, t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par, t_hjd)
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd)

    # For a formal test, require this to be smaller than some number.
    # x = np.sum(np.abs(mag_mulens - mag_bagle))

    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    plt.clf()
    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    ax[0].plot(t_mjd, mag_mulens, 'o')
    ax[0].plot(t_mjd, mag_bagle, '.')
    ax[0].invert_yaxis()
    ax[1].plot(t_mjd, mag_mulens - mag_bagle, '.')
    ax[0].set_ylabel('Mag')
    ax[1].set_ylabel('MM - Bagle')
    ax[1].set_xlabel('MJD')
    plt.show()
    plt.pause(0.5)
#    plt.pause(1)
#    plt.close()

    
def get_phot_mulens(coords, t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par, t_hjd):
    """
    Gould coordinate system, geocentric projected, murel=LS.
    """
    params = {}
    params['t_0'] = t0_g + 2400000.5
    params['t_0_par'] = t0par + 2400000.5
    params['u_0'] = u0_g
    params['t_E'] = tE_g
    params['pi_E_N'] = piEN_g
    params['pi_E_E'] = piEE_g

    my_model = mm.Model(params, coords=coords)

    mag_obs = my_model.get_lc(times=t_hjd, source_flux=1, blend_flux=0)

    return mag_obs

def get_phot_bagle(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd):
    """
    Lu coordinate system, heliocentric, murel=SL.
    """
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

    mod = model.PSPL_Phot_Par_Param1(params['t0'], params['u0_amp'], params['tE'],
                                     params['piE_E'], params['piE_N'],
                                     params['b_sff'], params['mag_src'],
                                     raL=params['raL'], decL=params['decL'])
    mag_obs = mod.get_photometry(t_mjd)

    return mag_obs


def get_phot_bagle_geoproj(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd, t0par):
    """
    Lu coordinate system, heliocentric, murel=SL.
    """
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

    mod = model.PSPL_Phot_Par_Param1_geoproj(params['t0'], params['u0_amp'], params['tE'],
                                     params['piE_E'], params['piE_N'],
                                     params['b_sff'], params['mag_src'],
                                     params['t0par'],
                                     raL=params['raL'], decL=params['decL'])
    mag_obs = mod.get_photometry(t_mjd)

    return mag_obs

def test():
    print('set 1')
    test_bagle_to_mulens(57000, 0.5, 300, 0.2, 0.1, 57100)
    test_bagle_to_mulens(57000, 0.5, 300, -0.2, 0.1, 57100)
    test_bagle_to_mulens(57000, 0.5, 300, 0.2, -0.1, 57100)
    test_bagle_to_mulens(57000, 0.5, 300, -0.2, -0.1, 57100)
    test_bagle_to_mulens(57000, -0.5, 300, 0.2, 0.1, 57100)
    test_bagle_to_mulens(57000, -0.5, 300, -0.2, 0.1, 57100)
    test_bagle_to_mulens(57000, -0.5, 300, 0.2, -0.1, 57100)
    test_bagle_to_mulens(57000, -0.5, 300, -0.2, -0.1, 57100)
    
    print('set 2')
    test_mulens_to_bagle(57000, 0.5, 300, 0.2, 0.1, 57100)
    test_mulens_to_bagle(57000, 0.5, 300, -0.2, 0.1, 57100)
    test_mulens_to_bagle(57000, 0.5, 300, 0.2, -0.1, 57100)
    test_mulens_to_bagle(57000, 0.5, 300, -0.2, -0.1, 57100)
    test_mulens_to_bagle(57000, -0.5, 300, 0.2, 0.1, 57100)
    test_mulens_to_bagle(57000, -0.5, 300, -0.2, 0.1, 57100)
    test_mulens_to_bagle(57000, -0.5, 300, 0.2, -0.1, 57100)
    test_mulens_to_bagle(57000, -0.5, 300, -0.2, -0.1, 57100)
    
    print('set 3')
    test_bagle_to_mulens(57000, 0.5, 300, 0.2, 0.1, 57000)
    test_bagle_to_mulens(57000, 0.5, 300, -0.2, 0.1, 57000)
    test_bagle_to_mulens(57000, 0.5, 300, 0.2, -0.1, 57000)
    test_bagle_to_mulens(57000, 0.5, 300, -0.2, -0.1, 57000)
    test_bagle_to_mulens(57000, -0.5, 300, 0.2, 0.1, 57000)
    test_bagle_to_mulens(57000, -0.5, 300, -0.2, 0.1, 57000)
    test_bagle_to_mulens(57000, -0.5, 300, 0.2, -0.1, 57000)
    test_bagle_to_mulens(57000, -0.5, 300, -0.2, -0.1, 57000)
    
    print('set 4')
    test_mulens_to_bagle(57000, 0.5, 300, 0.2, 0.1, 57000)
    test_mulens_to_bagle(57000, 0.5, 300, -0.2, 0.1, 57000)
    test_mulens_to_bagle(57000, 0.5, 300, 0.2, -0.1, 57000)
    test_mulens_to_bagle(57000, 0.5, 300, -0.2, -0.1, 57000)
    test_mulens_to_bagle(57000, -0.5, 300, 0.2, 0.1, 57000)
    test_mulens_to_bagle(57000, -0.5, 300, -0.2, 0.1, 57000)
    test_mulens_to_bagle(57000, -0.5, 300, 0.2, -0.1, 57000)
    test_mulens_to_bagle(57000, -0.5, 300, -0.2, -0.1, 57000)
    
    print('set 5')
    test_bagle_to_mulens(57000, 0.5, 300, 0.2, 0.1, 69900)
    test_bagle_to_mulens(57000, 0.5, 300, -0.2, 0.1, 69900)
    test_bagle_to_mulens(57000, 0.5, 300, 0.2, -0.1, 69900)
    test_bagle_to_mulens(57000, 0.5, 300, -0.2, -0.1, 69900)
    test_bagle_to_mulens(57000, -0.5, 300, 0.2, 0.1, 69900)
    test_bagle_to_mulens(57000, -0.5, 300, -0.2, 0.1, 69900)
    test_bagle_to_mulens(57000, -0.5, 300, 0.2, -0.1, 69900)
    test_bagle_to_mulens(57000, -0.5, 300, -0.2, -0.1, 69900)
    
    print('set 6')
    test_mulens_to_bagle(57000, 0.5, 300, 0.2, 0.1, 69900)
    test_mulens_to_bagle(57000, 0.5, 300, -0.2, 0.1, 69900)
    test_mulens_to_bagle(57000, 0.5, 300, 0.2, -0.1, 69900)
    test_mulens_to_bagle(57000, 0.5, 300, -0.2, -0.1, 69900)
    test_mulens_to_bagle(57000, -0.5, 300, 0.2, 0.1, 69900)
    test_mulens_to_bagle(57000, -0.5, 300, -0.2, 0.1, 69900)
    test_mulens_to_bagle(57000, -0.5, 300, 0.2, -0.1, 69900)
    test_mulens_to_bagle(57000, -0.5, 300, -0.2, -0.1, 69900)

def test_2():
    print('set 1')
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, 0.2, 0.1, 57100)
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, -0.2, 0.1, 57100)
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, 0.2, -0.1, 57100)
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, -0.2, -0.1, 57100)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, 0.2, 0.1, 57100)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, -0.2, 0.1, 57100)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, 0.2, -0.1, 57100)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, -0.2, -0.1, 57100)
    
    print('set 2')
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, 0.2, 0.1, 57100)
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, -0.2, 0.1, 57100)
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, 0.2, -0.1, 57100)
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, -0.2, -0.1, 57100)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, 0.2, 0.1, 57100)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, -0.2, 0.1, 57100)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, 0.2, -0.1, 57100)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, -0.2, -0.1, 57100)
    
    print('set 3')
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, 0.2, 0.1, 57000)
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, -0.2, 0.1, 57000)
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, 0.2, -0.1, 57000)
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, -0.2, -0.1, 57000)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, 0.2, 0.1, 57000)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, -0.2, 0.1, 57000)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, 0.2, -0.1, 57000)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, -0.2, -0.1, 57000)
    
    print('set 4')
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, 0.2, 0.1, 57000)
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, -0.2, 0.1, 57000)
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, 0.2, -0.1, 57000)
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, -0.2, -0.1, 57000)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, 0.2, 0.1, 57000)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, -0.2, 0.1, 57000)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, 0.2, -0.1, 57000)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, -0.2, -0.1, 57000)
    
    print('set 5')
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, 0.2, 0.1, 69900)
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, -0.2, 0.1, 69900)
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, 0.2, -0.1, 69900)
    test_bagle_to_bagle_geoproj(57000, 0.5, 300, -0.2, -0.1, 69900)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, 0.2, 0.1, 69900)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, -0.2, 0.1, 69900)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, 0.2, -0.1, 69900)
    test_bagle_to_bagle_geoproj(57000, -0.5, 300, -0.2, -0.1, 69900)
    
    print('set 6')
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, 0.2, 0.1, 69900)
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, -0.2, 0.1, 69900)
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, 0.2, -0.1, 69900)
    test_bagle_geoproj_to_bagle(57000, 0.5, 300, -0.2, -0.1, 69900)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, 0.2, 0.1, 69900)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, -0.2, 0.1, 69900)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, 0.2, -0.1, 69900)
    test_bagle_geoproj_to_bagle(57000, -0.5, 300, -0.2, -0.1, 69900)

def compare_geo_geo():
    t0_h, u0_h, tE_h, piEE_h, piEN_h = 57000, 0.5, 300, 0.2, 0.1

    t0par1 = 57050
    t0par2 = 56950

    t0_g1, u0_g1, tE_g1, piEE_g1, piEN_g1 = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                                                      tE_h, piEE_h, piEN_h,
                                                                      t0par1, in_frame='helio',
                                                                      murel_in='SL', murel_out='LS',
                                                                      coord_in='EN', coord_out='tb')

    t0_g2, u0_g2, tE_g2, piEE_g2, piEN_g2 = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                                                      tE_h, piEE_h, piEN_h,
                                                                      t0par2, in_frame='helio',
                                                                      murel_in='SL', murel_out='LS',
                                                                      coord_in='EN', coord_out='tb')

    mag_mulens1 = get_phot_mulens(coords, t0_g1, u0_g1, tE_g1, piEE_g1, piEN_g1, t0par1, t_hjd)
    mag_mulens2 = get_phot_mulens(coords, t0_g2, u0_g2, tE_g2, piEE_g2, piEN_g2, t0par2, t_hjd)
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd)

    print(t0_g1, u0_g1, tE_g1, piEE_g1, piEN_g1, t0par1)
    print(t0_g2, u0_g2, tE_g2, piEE_g2, piEN_g2, t0par2)
    
    # For a formal test, require this to be smaller than some number.
    # x = np.sum(np.abs(mag_mulens - mag_bagle))

    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    plt.clf()
    fig, ax = plt.subplots(2, 1, sharex=True, num=3)
    ax[0].plot(t_mjd, mag_mulens1, 'o')
    ax[0].plot(t_mjd, mag_mulens2, 'o')
    ax[0].plot(t_mjd, mag_bagle, '.')
    ax[0].invert_yaxis()
    ax[1].plot(t_mjd, mag_mulens1 - mag_bagle, '.')
    ax[1].plot(t_mjd, mag_mulens2 - mag_bagle, '.')
    ax[0].set_ylabel('Mag')
    ax[1].set_ylabel('MM - Bagle')
    ax[1].set_xlabel('MJD')
    plt.show()
    plt.pause(1)
#    plt.close()

def compare_geo():
    t0_h, u0_h, tE_h, piEE_h, piEN_h = 57000, 0.5, 300, 0.2, 0.1

    t0par_arr = np.arange(57000 - 720, 57000 + 720, 1)
    t0_arr = np.zeros(len(t0par_arr))
    u0_arr = np.zeros(len(t0par_arr))
    tE_arr = np.zeros(len(t0par_arr))
    piEE_arr = np.zeros(len(t0par_arr))
    piEN_arr = np.zeros(len(t0par_arr))

    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, tE_h, piEE_h, piEN_h, t_mjd)

    fig, ax = plt.subplots(5, 1, sharex=True, num=3, figsize=(6,10))
    plt.clf()
    fig, ax = plt.subplots(5, 1, sharex=True, num=3, figsize=(6,10))
    plt.subplots_adjust(left=0.25, top=0.98, bottom=0.1)

    for ii, t0par in enumerate(t0par_arr):
        t0_g, u0_g, tE_g, piEE_g, piEN_g = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                                                          tE_h, piEE_h, piEN_h,
                                                                          t0par, in_frame='helio',
                                                                          murel_in='SL', murel_out='LS',
                                                                          coord_in='EN', coord_out='tb')

        t0_arr[ii] = t0_g
        u0_arr[ii] = u0_g
        tE_arr[ii] = tE_g
        piEE_arr[ii] = piEE_g
        piEN_arr[ii] = piEN_g

        mag_mulens = get_phot_mulens(coords, t0_g, u0_g, tE_g, piEE_g, piEN_g, t0par, t_hjd)

#        print(np.sum(np.abs(mag_mulens - mag_bagle)))

    ax[0].plot(t0par_arr, t0_arr, 'k-', label='Geo projected')
    ax[1].plot(t0par_arr, u0_arr, 'k-')
    ax[2].plot(t0par_arr, tE_arr, 'k-')
    ax[3].plot(t0par_arr, piEE_arr, 'k-')
    ax[4].plot(t0par_arr, piEN_arr, 'k-')
    
    ax[0].axhline(y=t0_h, ls=':', color='b', label='Helio')
    ax[1].axhline(y=u0_h, ls=':', color='b')
    ax[2].axhline(y=tE_h, ls=':', color='b')
    ax[3].axhline(y=-1*piEE_h, ls=':', color='b')
    ax[4].axhline(y=-1*piEN_h, ls=':', color='b')
    
    ax[0].set_ylabel('$t_0$ (MJD)')
    ax[1].set_ylabel('$u_0$')
    ax[2].set_ylabel('$t_E$ (days)')
    ax[3].set_ylabel('$\pi_{E,E}$ (LS)')
    ax[4].set_ylabel('$\pi_{E,N}$ (LS)')
    ax[4].set_xlabel('$t_{0,par}$ (MJD)')

    ax[0].legend()
    plt.show()


#test_bagle_to_mulens(57000, 0.5, 300, 0.2, 0.1, 57100)
#test_mulens_to_bagle(57065.66690589838,
#                     0.38796926905650914,
#                     255.12120946554256,
#                     -0.19705897762298288,
#                     -0.10567761985484313,
#                     57100)
#
#test_bagle_to_mulens(57000, 0.5, 300, -0.2, 0.1, 57100)
#test_mulens_to_bagle(56906.702595128954,
#                     0.3207780388719065,
#                     320.67958147298145,
#                     0.17987564889211743,
#                     -0.13283354597254304,
#                     57100)
#
#test_bagle_to_mulens(57000, 0.5, 300, 0.2, -0.1, 57100)
#test_mulens_to_bagle(57048.20398878524,
#                     -0.39604868589963554,
#                     275.16823976951736,
#                     -0.21254356749441725,
#                     0.06946388930045672,
#                     57100)
#
#test_bagle_to_mulens(57000, 0.5, 300, -0.2, -0.1, 57100)
#test_mulens_to_bagle(56916.42854716019,
#                     -0.40487258961887973,
#                     363.5242968871542,
#                     0.2039081144183697,
#                     0.09176862685223672,
#                     57100)
#
#test_bagle_to_mulens(57000, -0.5, 300, 0.2, 0.1, 57100)
#test_mulens_to_bagle(57058.372346549586,
#                     -0.6116218811451084,
#                     255.12120946554256,
#                     -0.19705897762298288,
#                     -0.10567761985484313,
#                     57100)
#
#test_bagle_to_mulens(57000, -0.5, 300, -0.2, 0.1, 57100)
#test_mulens_to_bagle(56961.72572323333,
#                     -0.664391648641649,
#                     320.67958147298145,
#                     0.17987564889211743,
#                     -0.13283354597254304,
#                     57100)
#
#test_bagle_to_mulens(57000, -0.5, 300, 0.2, -0.1, 57100)
#test_mulens_to_bagle(57088.717442883375,
#                     0.5930533626789467,
#                     275.16823976951736,
#                     -0.21254356749441725,
#                     0.06946388930045672,
#                     57100)
#
#test_bagle_to_mulens(57000, -0.5, 300, -0.2, -0.1, 57100)
#test_mulens_to_bagle(56901.61794152418,
#                     0.5942971217590725,
#                     363.5242968871542,
#                     0.2039081144183697,
#                     0.09176862685223672,
#                     57100)
