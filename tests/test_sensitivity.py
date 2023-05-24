import numpy as np
import pylab as plt
from bagle import model
from bagle import sensitivity

def test_fisher_matrix(plot=False):
    """Test PSPL_Phot_Par_Param1 and that
    adding data reduces errors. 
    """
    params = {}
    params['t0'] = 60000.0 # MJD
    params['u0_amp'] = 0.1 # thetaE
    params['tE'] = 150.0   # days
    params['piEE'] = 0.04
    params['piEN'] = 0.03
    params['b_sff'] = 0.1
    params['mag_src'] = 19.0
    raL = 17.30 * 15.  # Bulge R.A.
    decL = -29.0

    # Make some fake data. Only sample up until tE.

    pspl_par_in = model.PSPL_Phot_Par_Param1(params['t0'],
                                             params['u0_amp'],
                                             params['tE'],
                                             params['piEE'],
                                             params['piEN'],
                                             params['b_sff'],
                                             params['mag_src'],
                                             raL=raL, decL=decL)

    # Simulate
    # photometric observations every 1 day and
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    
    for year_start in np.arange(params['t0'] - 1000, params['t0'] + 1000, 365.25):
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
    imag_obs = pspl_par_in.get_photometry(t_phot)
    flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs ** 0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    # Plot the data.
    if plot:
        plt.close(1)
        plt.figure(1)
        plt.clf()
        plt.errorbar(t_phot, imag_obs, yerr=imag_obs_err, fmt='k.')
        plt.gca().invert_yaxis()

    params_fixed = {'raL': raL, 'decL': decL}

    # Check that the errors go down as we add data after the peak.
    # Data only up to tE:
    idx1 = np.where(t_phot < params['t0'])[0]
    cov_mat1 = sensitivity.fisher_matrix(t_phot[idx1], imag_obs_err[idx1],
                                         model.PSPL_Phot_Par_Param1,
                                         params, params_fixed)
    err_on_param1 = np.sqrt(np.diagonal(cov_mat1))
 
    # Data all the way through.
    cov_mat2 = sensitivity.fisher_matrix(t_phot, imag_obs_err,
                                         model.PSPL_Phot_Par_Param1,
                                         params, params_fixed)
    err_on_param2 = np.sqrt(np.diagonal(cov_mat2))


    param_names = list(params.keys())

    for pp in range(len(param_names)):
        assert err_on_param2[pp] < err_on_param1[pp]
        # print(f'Error1 on {param_names[pp]:10s} = {err_on_param1[pp]:.3f} at value {params[param_names[pp]]:.3f}')
        
    return
    

def test_fisher_matrix2(plot=False):
    """Test PSPL_Phot_Par_Param3 and that
    adding data reduces errors. 
    """

    model_class = model.PSPL_Phot_Par_Param3
    params = {}
    params['t0'] = 60000.0 # MJD
    params['u0_amp'] = 0.1 # thetaE
    params['log_tE'] = np.log10(150.0)   # log(days)
    params['log_piE'] = np.log10(0.04)   
    params['phi_muRel'] = 30.0           # degrees
    params['b_sff'] = 0.1
    params['mag_base'] = 16.5
    
    params_fixed = {}
    params_fixed['raL'] = 17.30 * 15.  # Bulge R.A.
    params_fixed['decL'] = -29.0

    # Make some fake data. Only sample up until tE.
    pspl_par_in = model.get_model(model_class, params, params_fixed)

    # Simulate
    # photometric observations every 1 day and
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    
    for year_start in np.arange(params['t0'] - 1000, params['t0'] + 1000, 365.25):
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
    imag_obs = pspl_par_in.get_photometry(t_phot)
    flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs ** 0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    # Plot the data.
    if plot:
        plt.close(2)
        plt.figure(2)
        plt.clf()
        plt.errorbar(t_phot, imag_obs, yerr=imag_obs_err, fmt='r.')
        plt.gca().invert_yaxis()


    # Check that the errors go down as we add data after the peak.
    # Data only up to tE:
    idx1 = np.where(t_phot < params['t0'])[0]
    cov_mat1 = sensitivity.fisher_matrix(t_phot[idx1], imag_obs_err[idx1],
                                         model_class,
                                         params, params_fixed)
    err_on_param1 = np.sqrt(np.diagonal(cov_mat1))
 
    # Data all the way through.
    cov_mat2 = sensitivity.fisher_matrix(t_phot, imag_obs_err,
                                         model_class,
                                         params, params_fixed)
    err_on_param2 = np.sqrt(np.diagonal(cov_mat2))


    param_names = list(params.keys())

    for pp in range(len(param_names)):
        assert err_on_param2[pp] < err_on_param1[pp]
        
    return
    
def test_compare_err_phot_param2_vs_param3():
    """
    Compare errors from models with parameters vs. log parameters. 
    """
    model_class2 = model.PSPL_Phot_Par_Param2
    params2 = {}
    params2['t0'] = 60000.0 # MJD
    params2['u0_amp'] = 0.1 # thetaE
    params2['tE'] = 150.0   # days)
    params2['piE_E'] = -0.05
    params2['piE_N'] = 0.11
    params2['b_sff'] = 0.1
    params2['mag_base'] = 16.5
    
    model_class3 = model.PSPL_Phot_Par_Param3
    params3 = {}
    params3['t0'] = 60000.0 # MJD
    params3['u0_amp'] = 0.1 # thetaE
    params3['log_tE'] = np.log10(150.0)   # log(days)
    params3['log_piE'] = np.log10(np.linalg.norm([params2['piE_E'],
                                                  params2['piE_N']]))
    params3['phi_muRel'] = np.rad2deg(np.arctan2(params2['piE_E'],
                                                 params2['piE_N']))
    params3['b_sff'] = 0.1
    params3['mag_base'] = 16.5

    
    params_fixed = {}
    params_fixed['raL'] = (17.0 + (49.0 / 60.) + (51.38 / 3600.0)) * 15.0  # degrees
    params_fixed['decL'] = -35 + (22.0 / 60.0) + (28.0 / 3600.0)
    

    t_mod = np.arange(params2['t0'] - 1000, params2['t0'] + 1000, 2)

    ##########
    # Compare two models with same properties
    # recast in different parameters.
    ##########
    mod2 = model.get_model(model_class2, params2, params_fixed)
    mod3 = model.get_model(model_class3, params3, params_fixed)

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = mod2.get_photometry(t_mod)
    flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs ** 0.5
    imag_obs_err = 1.087 / flux_obs_err

    # Assume same errors for both models. 
    fish2 = sensitivity.fisher_matrix(t_mod, imag_obs_err,
                                      model_class2, params2, params_fixed)
    fish3 = sensitivity.fisher_matrix(t_mod, imag_obs_err,
                                      model_class3, params3, params_fixed)
    
    err_on_param2 = np.sqrt(np.diagonal(fish2))
    err_on_param3 = np.sqrt(np.diagonal(fish3))

    for pp, key in enumerate(params2):
        print(f'Error2 on {key:10s} = {err_on_param2[pp]:.3f}')

        if key == 'tE':
            print(f'Error2 on {"log_tE":10s} = {np.log10(err_on_param2[pp]):.3f}')

        if key == 'piE_N':
            err_piE = np.hypot(err_on_param2[pp], err_on_param2[pp-1])
            print(f'Error2 on {"piE":10s} = {err_piE:.3f}')
            print(f'Error2 on {"log_piE":10s} = {np.log10(err_piE):.3f}')
            

    print('---')
    for pp, key in enumerate(params3):
        print(f'Error3 on {key:10s} = {err_on_param3[pp]:.3f}')

        if key == 'log_tE':
            print(f'Error3 on {"tE":10s} = {10**err_on_param3[pp]:.3f}')

        if key == 'log_piE':
            err_piE = err_on_param3[pp]
            print(f'Error3 on {"piE":10s} = {10**err_on_param3[pp]:.3f}')
            
            

    return

    
