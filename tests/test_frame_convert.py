import pdb
from src.BAGLE import model
import numpy as np
import matplotlib.pyplot as plt
from src.BAGLE import frame_convert as fc
from astropy.coordinates import SkyCoord
from astropy import units as u

def test_bagle_geo_to_helio_phot(ra, dec, t_mjd,
                                 t0_g, u0_g, tE_g, 
                                 piEE_g, piEN_g, 
                                 t0par, plot=False):
    """
    Test conversion from geocentric projected frame
    (using lens-source tau-beta convention) to
    heliocentric frame (using source-lens East-North 
    convention) photometry parameters in BAGLE.
    """
    # Convert the geoprojected parameters (in lens-source tau-beta)
    # to heliocentric parameters (in source-lens East-North).
    output = fc.convert_helio_geo_phot(ra, dec, t0_g, u0_g, 
                                       tE_g, piEE_g, piEN_g,
                                       t0par, in_frame='geo',
                                       murel_in='LS', murel_out='SL',
                                       coord_in='tb', coord_out='EN')

    t0_h, u0_h, tE_h, piEE_h, piEN_h = output

    # Get the lightcurve from the geoprojected parameters.
    mag_bagle_geoproj = get_phot_bagle_geoproj(ra, dec, t0_g, u0_g, 
                                               tE_g, piEE_g, piEN_g, 
                                               t_mjd, t0par)

    # Get the lightcurve from the heliocentric parameters.
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, 
                               tE_h, piEE_h, piEN_h, t_mjd)

    if plot:
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
#        plt.pause(0.5)

    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-4 magnitudes on average.
    # Note: current test fails if require < 1e-5.
    diff = (mag_bagle_geoproj - mag_bagle)/mag_bagle_geoproj
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-4

def test_bagle_helio_to_geo_phot(ra, dec, t_mjd,
                                 t0_h, u0_h, tE_h, 
                                 piEE_h, piEN_h, 
                                 t0par, plot=False):
    """
    Test conversion from heliocentric frame (using 
    source-lens East-North convention) to geocentric 
    projected frame (using lens-source tau-beta convention)
    photometry parameters in BAGLE.
    """
    # Convert the heliocentric parameters (in source-lens East-North)
    # to geoprojected parameters (in lens-source tau-beta).
    output = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                       tE_h, piEE_h, piEN_h,
                                       t0par, in_frame='helio',
                                       murel_in='SL', murel_out='LS',
                                       coord_in='EN', coord_out='tb')

    t0_g, u0_g, tE_g, piEE_g, piEN_g = output

    # Get the lightcurve from the geoprojected parameters.
    mag_bagle_geoproj = get_phot_bagle_geoproj(ra, dec, t0_g, u0_g, 
                                               tE_g, piEE_g, piEN_g, 
                                               t_mjd, t0par)

    # Get the lightcurve from the heliocentric parameters.
    mag_bagle = get_phot_bagle(ra, dec, t0_h, u0_h, 
                               tE_h, piEE_h, piEN_h, t_mjd)

    if plot:
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
#        plt.pause(0.5)

    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-5 magnitudes on average.
    # Note: current test fails if require < 1e-6.
    diff = (mag_bagle - mag_bagle_geoproj)/mag_bagle
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-5


def test_bagle_to_mulens(ra, dec, t_mjd,
                         t0_h, u0_h, tE_h, 
                         piEE_h, piEN_h, 
                         t0par, plot=False):
    """
    Test conversion from heliocentric frame (using 
    source-lens East-North convention) in BAGLE to geocentric 
    projected frame (using lens-source tau-beta convention)
    in MulensModel for photometry parameters .
    """
    # Convert the heliocentric parameters (in source-lens East-North)
    # to geoprojected parameters (in lens-source tau-beta).
    output = fc.convert_helio_geo_phot(ra, dec, t0_h, u0_h, 
                                       tE_h, piEE_h, piEN_h,
                                       t0par, in_frame='helio',
                                       murel_in='SL', murel_out='LS',
                                       coord_in='EN', coord_out='tb')

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
    
    if plot:
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
#        plt.pause(0.5)

    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-5 magnitudes on average.
    # Note: current test fails if require < 1e-6.
    diff = (mag_bagle - mag_mulens)/mag_bagle
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-5

def test_mulens_to_bagle(ra, dec, t_mjd,
                         t0_g, u0_g, tE_g, 
                         piEE_g, piEN_g, 
                         t0par, plot=False):
    """
    Test conversion from geocentric projected frame (using 
    lens-source tau-beta convention) in MulensModel to 
    heliocentric frame (using source-lens East-North 
    convention) in BAGLE for photometry parameters .
    """
    # Convert the geoprojected parameters (in lens-source tau-beta)
    # to heliocentric parameters (in source-lens East-North).
    output = fc.convert_helio_geo_phot(ra, dec, t0_g, u0_g, 
                                       tE_g, piEE_g, piEN_g,
                                       t0par, in_frame='geo',
                                       murel_in='LS', murel_out='SL',
                                       coord_in='tb', coord_out='EN')

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

    if plot:
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

    # Make sure that the conversion works by asserting
    # that the lightcurves are no more different than
    # 1e-4 magnitudes on average.
    # Note: current test fails if require < 1e-5.
    diff = (mag_mulens - mag_bagle)/mag_mulens
    total_diff = np.sum(np.abs(diff))
    assert total_diff/len(t_mjd) < 1e-4
    
    
def get_phot_mulens(coords, t0_g, u0_g, tE_g, 
                    piEE_g, piEN_g, t0par, t_hjd):
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

    mag_obs = my_model.get_lc(times=t_hjd, source_flux=1, blend_flux=0)

    return mag_obs

def get_phot_bagle(ra, dec, t0_h, u0_h, 
                   tE_h, piEE_h, piEN_h, t_mjd):
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

    mag_obs = mod.get_photometry(t_mjd)

    return mag_obs


def get_phot_bagle_geoproj(ra, dec, t0_h, u0_h, tE_h, 
                           piEE_h, piEN_h, t_mjd, t0par):
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
    mag_obs = mod.get_photometry(t_mjd)

    return mag_obs

def test():
    t_mjd = np.arange(57000 - 500, 57000 + 500, 1)

    print('set 1')
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57100)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57100)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57100)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57100)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57100)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57100)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57100)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57100)
    
    print('set 2')
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57100)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57100)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57100)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57100)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57100)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57100)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57100)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57100)
    
    print('set 3')
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57000)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57000)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57000)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57000)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57000)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57000)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57000)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57000)
    
    print('set 4')
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57000)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57000)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57000)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57000)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57000)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57000)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57000)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57000)
    
    print('set 5')
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 69900)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 69900)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 69900)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 69900)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 69900)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 69900)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 69900)
    test_bagle_to_mulens(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 69900)
    
    print('set 6')
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 69900)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 69900)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 69900)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 69900)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 69900)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 69900)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 69900)
    test_mulens_to_bagle(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 69900)

def test_2():
    t_mjd = np.arange(57000 - 500, 57000 + 500, 1)

    print('set 1') 
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57100)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57100)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57100)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57100)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57100)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57100)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57100)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57100)
    
    print('set 2')
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57100)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57100)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57100)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57100)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57100)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57100)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57100)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57100)
    
    print('set 3')
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57000)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57000)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57000)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57000)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57000)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57000)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57000)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57000)
    
    print('set 4')
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 57000)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 57000)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 57000)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 57000)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 57000)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 57000)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 57000)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 57000)
    
    print('set 5')
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 69900)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 69900)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 69900)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 69900)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 69900)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 69900)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 69900)
    test_bagle_helio_to_geo_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 69900)
    
    print('set 6')
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, 0.1, 69900)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, 0.1, 69900)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, 0.2, -0.1, 69900)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, 0.5, 300, -0.2, -0.1, 69900)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, 0.1, 69900)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, 0.1, 69900)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, 0.2, -0.1, 69900)
    test_bagle_geo_to_helio_phot(259.0, -29.0, t_mjd, 57000, -0.5, 300, -0.2, -0.1, 69900)

################################################################################
###########  Stuff below is incomplete/wrong/not yet functional...  ############
################################################################################
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
