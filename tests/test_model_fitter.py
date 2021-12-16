from src.micromodel import model
from src.micromodel import model_fitter
from src.micromodel import multinest_utils, munge
from src.micromodel.model_fitter import PSPL_Solver
import test_model
import numpy as np
import pylab as plt
import os
import pickle
import time
from multiprocessing import Pool
from dynesty import utils as dyutil
from dynesty import plotting as dyplot
import dynesty
import pdb
import matplotlib
from astropy.table import Table
import yaml
import shutil

# Always generate the same fake data.
np.random.seed(0)

def test_pspl_dy_fit():
    # No parallax
    data, p_in = test_model.fake_data1()

    fitter = model_fitter.PSPL_Solver(data, model.PSPL_PhotAstrom_noPar_Param1,
                                      custom_additional_param_names = [])
    # Lets adjust some priors for faster solving.
    fitter.priors['mL'] = model_fitter.make_gen(9.9, 10.1)
    fitter.priors['t0'] = model_fitter.make_gen(56990, 57010)
    fitter.priors['beta'] = model_fitter.make_gen(-0.45, -0.35)
    fitter.priors['muL_E'] = model_fitter.make_gen(-0.1, 0.1)
    fitter.priors['muL_N'] = model_fitter.make_gen(-7.1, -6.9)
    fitter.priors['muS_E'] = model_fitter.make_gen(1.3, 1.7)
    fitter.priors['muS_N'] = model_fitter.make_gen(-0.7, -0.3)
    fitter.priors['dL'] = model_fitter.make_gen(3900, 4100)
    fitter.priors['dL_dS'] = model_fitter.make_gen(0.45, 0.55)
    fitter.priors['b_sff1'] = model_fitter.make_gen(0.95, 1.01)
    fitter.priors['mag_src1'] = model_fitter.make_gen(18.9, 19.1)
    fitter.priors['xS0_E'] = model_fitter.make_gen(-1e-4, 1e-4)
    fitter.priors['xS0_N'] = model_fitter.make_gen(-1e-4, 1e-4)

    # n_cpu = 4
    # pool = Pool(n_cpu)

    sampler = dynesty.DynamicNestedSampler(fitter.LogLikelihood, fitter.Prior, 
                                           ndim=fitter.n_dims, bound='multi',
                                           sample='unif')#, pool = pool,
                                           #queue_size = n_cpu)

    t0 = time.time()

    sampler.run_nested(nlive_init=300, print_progress=True, maxiter=1000)

    t1 = time.time()

    print('Runtime: ', t1 - t0)

    results = sampler.results
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    best_avg, best_cov = dyutil.mean_and_cov(samples, weights)

    best = {}
    for i, param_name in enumerate(fitter.fitter_param_names):
        best[param_name] = best_avg[i]

    pspl_out = fitter.get_model(best)
    pspl_in = fitter.get_model(p_in)

    # TEST that output params match input within 20%
    for key in best:
        if p_in[key] == 0:
            assert np.abs(best[key] - p_in[key]) < 0.2
        else:
            assert np.abs( (best[key] - p_in[key]) / p_in[key] ) < 0.2

    # TEST that the generated photometry closely matches the input.
    filt_idx = 0
    mod_m_at_dat_out = pspl_out.get_photometry(data['t_phot' + str(filt_idx + 1)], filt_idx)
    mod_m_at_dat_in = pspl_in.get_photometry(data['t_phot' + str(filt_idx + 1)], filt_idx)
    np.testing.assert_allclose(mod_m_at_dat_out, mod_m_at_dat_in, rtol=1e-2)
    
    model_fitter.plot_photometry(data, pspl_out, input_model = pspl_in)
    model_fitter.plot_astrometry(data, pspl_out, input_model = pspl_in, n_phot_sets=1)

    ######################
    # Dynesty plotting w/ some modifications
    # Still hacking around with the exact values we want...
    # And mebbe they shouldn't be hardcoded...
    ######################

    # CHECK: are these names right?
    param_names = fitter.fitter_param_names

    # Traceplot
    bigfig, bigax = plt.subplots(13, 2, figsize=(10, 16))
    bigfig.subplots_adjust(hspace=1, left = 0.05, top = 0.95, bottom = 0.05)

    for i in range(13):
        for j in range(2):
            bigax[i,j].tick_params(axis = 'both', which = 'major', labelsize = 10)

    matplotlib.rc('font', size=10)

    dyplot.traceplot(results, fig = (bigfig, bigax),
                     labels = param_names, label_kwargs = {'fontsize' : 10},
                     use_math_text = True, show_titles = True,
                     title_kwargs = {'fontsize' : 10},
                     max_n_ticks = 2)

    # # Cornerplot
    # bigfig, bigax = plt.subplots(13, 13, figsize=(10, 10))
    # for i in range(13):
    #     for j in range(13):
    #         bigax[i,j].tick_params(axis = 'both', which = 'major', labelsize = 10)
    # matplotlib.rc('font', size=10)

    # dyplot.cornerplot(results, fig = (bigfig, bigax),
    #                       labels = param_names, label_kwargs = {'fontsize' : 10},
    #                       use_math_text = True, show_titles = True,
    #                       title_kwargs = {'fontsize' : 10})

    # fitter.LogLikelihood(best_avg)

    return

def test_pspl_ultranest_fit():
    import ultranest
    
    # No parallax
    data, p_in = test_model.fake_data1()

    fitter = model_fitter.PSPL_Solver(data, model.PSPL_PhotAstrom_noPar_Param1,
                                      custom_additional_param_names = [], 
                                      outputfiles_basename='./test_fit_ultranest_pspl/a1_')

    # Lets adjust some priors for faster solving.
    fitter.priors['mL'] = model_fitter.make_gen(9.9, 10.1)
    fitter.priors['t0'] = model_fitter.make_gen(56990, 57010)
    fitter.priors['beta'] = model_fitter.make_gen(-0.45, -0.35)
    fitter.priors['muL_E'] = model_fitter.make_gen(-0.1, 0.1)
    fitter.priors['muL_N'] = model_fitter.make_gen(-7.1, -6.9)
    fitter.priors['muS_E'] = model_fitter.make_gen(1.3, 1.7)
    fitter.priors['muS_N'] = model_fitter.make_gen(-0.7, -0.3)
    fitter.priors['dL'] = model_fitter.make_gen(3900, 4100)
    fitter.priors['dL_dS'] = model_fitter.make_gen(0.45, 0.55)
    fitter.priors['b_sff1'] = model_fitter.make_gen(0.95, 1.01)
    fitter.priors['mag_src1'] = model_fitter.make_gen(18.9, 19.1)
    fitter.priors['xS0_E'] = model_fitter.make_gen(-1e-4, 1e-4)
    fitter.priors['xS0_N'] = model_fitter.make_gen(-1e-4, 1e-4)

    # n_cpu = 4
    # pool = Pool(n_cpu)

    sampler = ultranest.ReactiveNestedSampler(fitter.fitter_param_names,
                                              fitter.LogLikelihood, fitter.Prior_copy,
                                              log_dir='test_fit_ultranest_pspl/',
                                              resume='overwrite')


    t0_un = time.time()

    sampler.run()

    t1_un = time.time()

    
    t0_mn = time.time()

    fitter.solve()

    t1_mn = time.time()
    

    print('Ultranest Runtime: ', t1_un - t0_un)
    print('Multinest Runtime: ', t1_mn - t0_mn)

    sampler.print_results()

    sampler.plot_run()
    sampler.plot_trace()
    sampler.plot_corner()

    fitter.summarize_results()
    fitter.plot_dynesty_style(sim_vals=p_in)
    
    return sampler

    # results = sampler.results
    # samples = results.samples
    # weights = np.exp(results.logwt - results.logz[-1])
    # best_avg, best_cov = dyutil.mean_and_cov(samples, weights)

    # best = {}
    # for i, param_name in enumerate(fitter.fitter_param_names):
    #     best[param_name] = best_avg[i]

    # pspl_out = fitter.get_model(best)
    # pspl_in = fitter.get_model(p_in)

    # # TEST that output params match input within 20%
    # for key in best:
    #     if p_in[key] == 0:
    #         assert np.abs(best[key] - p_in[key]) < 0.2
    #     else:
    #         assert np.abs( (best[key] - p_in[key]) / p_in[key] ) < 0.2

    # # TEST that the generated photometry closely matches the input.
    # filt_idx = 0
    # mod_m_at_dat_out = pspl_out.get_photometry(data['t_phot' + str(filt_idx + 1)], filt_idx)
    # mod_m_at_dat_in = pspl_in.get_photometry(data['t_phot' + str(filt_idx + 1)], filt_idx)
    # np.testing.assert_allclose(mod_m_at_dat_out, mod_m_at_dat_in, rtol=1e-2)
    
    # model_fitter.plot_photometry(data, pspl_out, input_model = pspl_in)
    # model_fitter.plot_astrometry(data, pspl_out, input_model = pspl_in, n_phot_sets=1)

    # ######################
    # # Dynesty plotting w/ some modifications
    # # Still hacking around with the exact values we want...
    # # And mebbe they shouldn't be hardcoded...
    # ######################

    # # CHECK: are these names right?
    # param_names = fitter.fitter_param_names

    # # Traceplot
    # bigfig, bigax = plt.subplots(13, 2, figsize=(10, 16))
    # bigfig.subplots_adjust(hspace=1, left = 0.05, top = 0.95, bottom = 0.05)

    # for i in range(13):
    #     for j in range(2):
    #         bigax[i,j].tick_params(axis = 'both', which = 'major', labelsize = 10)

    # matplotlib.rc('font', size=10)

    # dyplot.traceplot(results, fig = (bigfig, bigax),
    #                  labels = param_names, label_kwargs = {'fontsize' : 10},
    #                  use_math_text = True, show_titles = True,
    #                  title_kwargs = {'fontsize' : 10},
    #                  max_n_ticks = 2)

    # # Cornerplot
    # bigfig, bigax = plt.subplots(13, 13, figsize=(10, 10))
    # for i in range(13):
    #     for j in range(13):
    #         bigax[i,j].tick_params(axis = 'both', which = 'major', labelsize = 10)
    # matplotlib.rc('font', size=10)

    # dyplot.cornerplot(results, fig = (bigfig, bigax),
    #                       labels = param_names, label_kwargs = {'fontsize' : 10},
    #                       use_math_text = True, show_titles = True,
    #                       title_kwargs = {'fontsize' : 10})

    # fitter.LogLikelihood(best_avg)

    return


def test_lumlens_parallax_fit():
    outdir = './test_mnest_lumlens_bulge/'
    os.makedirs(outdir, exist_ok=True)

    data, p_in = test_model.fake_data_lumlens_parallax_bulge()

    fitter = PSPL_Solver(data, 
                         model.PSPL_PhotAstrom_LumLens_Par_Param1,
#                         model.PSPL_PhotAstrom_Par_Param1,
                         n_live_points=300,
                         outputfiles_basename=outdir + '/aa_')
#                         outputfiles_basename=outdir + '/bb_')

    fitter.priors['t0'] = model_fitter.make_gen(p_in['t0']-5, p_in['t0']+5)
    fitter.priors['xS0_E'] = model_fitter.make_gen(p_in['xS0_E']-1e-3, p_in['xS0_E']+1e-3)
    fitter.priors['xS0_N'] = model_fitter.make_gen(p_in['xS0_N']-1e-3, p_in['xS0_N']+1e-3)
    fitter.priors['beta'] = model_fitter.make_gen(p_in['beta']-0.1, p_in['beta']+0.1)
    fitter.priors['muL_E'] = model_fitter.make_gen(p_in['muL_E']-0.1, p_in['muL_E']+0.1)
    fitter.priors['muL_N'] = model_fitter.make_gen(p_in['muL_N']-0.1, p_in['muL_N']+0.1)
    fitter.priors['muS_E'] = model_fitter.make_gen(p_in['muS_E']-0.1, p_in['muS_E']+0.1)
    fitter.priors['muS_N'] = model_fitter.make_gen(p_in['muS_N']-0.1, p_in['muS_N']+0.1)
    fitter.priors['dL'] = model_fitter.make_gen(p_in['dL']-100, p_in['dL']+100)
    fitter.priors['dL_dS'] = model_fitter.make_gen((p_in['dL']/p_in['dS'])-0.1, (p_in['dL']/p_in['dS'])+0.1)
    fitter.priors['b_sff1'] = model_fitter.make_gen(p_in['b_sff1']-0.1, p_in['b_sff1']+0.1)
    fitter.priors['mag_src1'] = model_fitter.make_gen(p_in['mag_src1']-0.1, p_in['mag_src1']+0.1)

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(left=0.2)
    plt.plot(data['xpos'], data['ypos'], '.')
    plt.show()

    fitter.solve()

    best = fitter.get_best_fit()

    pspl_out = model.PSPL_PhotAstrom_LumLens_Par_Param1(best['mL'],
                                                best['t0'],
                                                best['beta'],
                                                best['dL'],
                                                best['dL_dS'],
                                                best['xS0_E'],
                                                best['xS0_N'],
                                                best['muL_E'],
                                                best['muL_N'],
                                                best['muS_E'],
                                                best['muS_N'],
                                                [best['b_sff1']],
                                                [best['mag_src1']],
                                                raL=p_in['raL'],
                                                decL=p_in['decL'])

    pspl_in = model.PSPL_PhotAstrom_LumLens_Par_Param1(p_in['mL'],
                                               p_in['t0'],
                                               p_in['beta'],
                                               p_in['dL'],
                                               p_in['dL'] / p_in['dS'],
                                               p_in['xS0_E'],
                                               p_in['xS0_N'],
                                               p_in['muL_E'],
                                               p_in['muL_N'],
                                               p_in['muS_E'],
                                               p_in['muS_N'],
                                               p_in['b_sff1'],
                                               p_in['mag_src1'],
                                               raL=p_in['raL'],
                                               decL=p_in['decL'])


    p_in['dL_dS'] = p_in['dL'] / p_in['dS']
    p_in['tE'] = pspl_in.tE
    p_in['thetaE'] = pspl_in.thetaE_amp
    p_in['piE_E'] = pspl_in.piE[0]
    p_in['piE_N'] = pspl_in.piE[1]
    p_in['u0_amp'] = pspl_in.u0_amp
    p_in['muRel_E'] = pspl_in.muRel[0]
    p_in['muRel_N'] = pspl_in.muRel[1]

    fitter.summarize_results()
    fitter.plot_dynesty_style(sim_vals=p_in)
    fitter.plot_model_and_data(pspl_out, input_model=pspl_in)

    imag_out = pspl_out.get_photometry(data['t_phot1'])
    pos_out = pspl_out.get_astrometry(data['t_ast'])

    imag_in = pspl_in.get_photometry(data['t_phot1'])
    pos_in = pspl_in.get_astrometry(data['t_ast'])

    np.testing.assert_array_almost_equal(imag_out, imag_in, 1)
    np.testing.assert_array_almost_equal(pos_out, pos_in, 4)

    print("OUTPUT:")
    lnL_out = fitter.log_likely(best, verbose=True)
    print("INPUT:")
    lnL_in = fitter.log_likely(p_in, verbose=True)

    assert np.abs(lnL_out - lnL_in) < 10

    return


def test_lumlens_parallax_fit_2p1a():
    outdir = './test_mnest_lumlens_bulge_DEBUG/'
    os.makedirs(outdir, exist_ok=True)

    data1, data2, params1, params2 = test_model.fake_data_lumlens_parallax_bulge2()

    data = data1
    data['t_phot2'] = data2['t_phot1']
    data['mag2'] = data2['mag1']
    data['mag_err2'] = data2['mag_err1']
    data['phot_data'] = ['sim1', 'sim2']
    data['ast_data'] = ['sim1']
    data['phot_files'] = ['pfile1', 'pfile2']
    data['ast_files'] = ['afile1']
    

    p_in = params1
    p_in['b_sff2'] = params2['b_sff1']
    p_in['mag_src2'] = params2['mag_src1']

    fitter = PSPL_Solver(data, 
                         model.PSPL_PhotAstrom_LumLens_Par_Param1,
#                         model.PSPL_PhotAstrom_Par_Param1,
                         n_live_points=300,
                         outputfiles_basename=outdir + '/aa_',
                         evidence_tolerance=2.0, 
                         sampling_efficiency=3.0)
#                         outputfiles_basename=outdir + '/bb_')

    fitter.priors['t0'] = model_fitter.make_gen(p_in['t0']-1, p_in['t0']+1)
    fitter.priors['xS0_E'] = model_fitter.make_gen(p_in['xS0_E']-1e-4, p_in['xS0_E']+1e-4)
    fitter.priors['xS0_N'] = model_fitter.make_gen(p_in['xS0_N']-1e-4, p_in['xS0_N']+1e-4)
    fitter.priors['beta'] = model_fitter.make_gen(p_in['beta']-0.01, p_in['beta']+0.01)
    fitter.priors['muL_E'] = model_fitter.make_gen(p_in['muL_E']-0.01, p_in['muL_E']+0.01)
    fitter.priors['muL_N'] = model_fitter.make_gen(p_in['muL_N']-0.01, p_in['muL_N']+0.01)
    fitter.priors['muS_E'] = model_fitter.make_gen(p_in['muS_E']-0.01, p_in['muS_E']+0.01)
    fitter.priors['muS_N'] = model_fitter.make_gen(p_in['muS_N']-0.01, p_in['muS_N']+0.01)
    fitter.priors['dL'] = model_fitter.make_gen(p_in['dL']-10, p_in['dL']+10)
    fitter.priors['dL_dS'] = model_fitter.make_gen((p_in['dL']/p_in['dS'])-0.01, (p_in['dL']/p_in['dS'])+0.01)
    fitter.priors['b_sff1'] = model_fitter.make_gen(p_in['b_sff1']-0.01, p_in['b_sff1']+0.01)
    fitter.priors['mag_src1'] = model_fitter.make_gen(p_in['mag_src1']-0.01, p_in['mag_src1']+0.01)
    fitter.priors['b_sff2'] = model_fitter.make_gen(p_in['b_sff2']-0.01, p_in['b_sff2']+0.01)
    fitter.priors['mag_src2'] = model_fitter.make_gen(p_in['mag_src2']-0.01, p_in['mag_src2']+0.01)

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(left=0.2)
    plt.plot(data['xpos1'], data['ypos1'], '.')
    plt.show()

#    fitter.solve()

    best = fitter.get_best_fit()

    pspl_out = model.PSPL_PhotAstrom_LumLens_Par_Param1(best['mL'],
                                                best['t0'],
                                                best['beta'],
                                                best['dL'],
                                                best['dL_dS'],
                                                best['xS0_E'],
                                                best['xS0_N'],
                                                best['muL_E'],
                                                best['muL_N'],
                                                best['muS_E'],
                                                best['muS_N'],
                                                [best['b_sff1'], best['b_sff2']],
                                                [best['mag_src1'], best['mag_src2']],
                                                raL=p_in['raL'],
                                                decL=p_in['decL'])

    pspl_in = model.PSPL_PhotAstrom_LumLens_Par_Param1(p_in['mL'],
                                               p_in['t0'],
                                               p_in['beta'],
                                               p_in['dL'],
                                               p_in['dL'] / p_in['dS'],
                                               p_in['xS0_E'],
                                               p_in['xS0_N'],
                                               p_in['muL_E'],
                                               p_in['muL_N'],
                                               p_in['muS_E'],
                                               p_in['muS_N'],
                                               [p_in['b_sff1'], p_in['b_sff2']],
                                               [p_in['mag_src1'], p_in['mag_src2']],
                                               raL=p_in['raL'],
                                               decL=p_in['decL'])


    p_in['dL_dS'] = p_in['dL'] / p_in['dS']
    p_in['tE'] = pspl_in.tE
    p_in['thetaE'] = pspl_in.thetaE_amp
    p_in['piE_E'] = pspl_in.piE[0]
    p_in['piE_N'] = pspl_in.piE[1]
    p_in['u0_amp'] = pspl_in.u0_amp
    p_in['muRel_E'] = pspl_in.muRel[0]
    p_in['muRel_N'] = pspl_in.muRel[1]

    fitter.summarize_results()
    fitter.plot_dynesty_style(sim_vals=p_in)
    fitter.plot_model_and_data(pspl_out, input_model=pspl_in)

    imag_out = pspl_out.get_photometry(data['t_phot1'])
    pos_out = pspl_out.get_astrometry(data['t_ast1'])

    imag_in = pspl_in.get_photometry(data['t_phot1'])
    pos_in = pspl_in.get_astrometry(data['t_ast1'])

    np.testing.assert_array_almost_equal(imag_out, imag_in, 1)
    np.testing.assert_array_almost_equal(pos_out, pos_in, 4)

    print("OUTPUT:")
    lnL_out = fitter.log_likely(best, verbose=True)
    print("INPUT:")
    lnL_in = fitter.log_likely(p_in, verbose=True)

    assert np.abs(lnL_out - lnL_in) < 10

    return


def test_lumlens_parallax_fit_4p2a():
    outdir = './test_mnest_lumlens_bulge4_DEBUG/'
    os.makedirs(outdir, exist_ok=True)

    data1, data2, data3, data4, params1, params2, params3, params4 = test_model.fake_data_lumlens_parallax_bulge4()
    data = data1
    data['t_phot2'] = data2['t_phot1']
    data['mag2'] = data2['mag1']
    data['mag_err2'] = data2['mag_err1']

    data['t_phot3'] = data3['t_phot1']
    data['mag3'] = data3['mag1']
    data['mag_err3'] = data3['mag_err1']

    data['t_phot4'] = data4['t_phot1']
    data['mag4'] = data4['mag1']
    data['mag_err4'] = data4['mag_err1']

    data['t_ast2'] = data3['t_ast1']
    data['xpos2'] = data3['xpos1']
    data['ypos2'] = data3['ypos1']
    data['xpos_err2'] = data3['xpos_err1']
    data['ypos_err2'] = data3['ypos_err1']

    data['phot_data'] = ['sim1', 'sim2', 'sim3', 'sim4']
    data['ast_data'] = ['sim1', 'sim3']
    data['phot_files'] = ['pfile1', 'pfile2', 'pfile3', 'pfile4']
    data['ast_files'] = ['afile1', 'afile3']
    
    p_in = params1
    p_in['b_sff2'] = params2['b_sff1']
    p_in['mag_src2'] = params2['mag_src1']

    p_in['b_sff3'] = params3['b_sff1']
    p_in['mag_src3'] = params3['mag_src1']

    p_in['b_sff4'] = params4['b_sff1']
    p_in['mag_src4'] = params4['mag_src1']

    fitter = PSPL_Solver(data, 
                         model.PSPL_PhotAstrom_LumLens_Par_Param1,
#                         model.PSPL_PhotAstrom_Par_Param1,
                         n_live_points=300,
                         outputfiles_basename=outdir + '/aa_',
                         evidence_tolerance=2.0, 
                         sampling_efficiency=3.0,
                         verbose=True)

    fitter.priors['t0'] = model_fitter.make_gen(p_in['t0']-1, p_in['t0']+1)
    fitter.priors['xS0_E'] = model_fitter.make_gen(p_in['xS0_E']-1e-4, p_in['xS0_E']+1e-4)
    fitter.priors['xS0_N'] = model_fitter.make_gen(p_in['xS0_N']-1e-4, p_in['xS0_N']+1e-4)
    fitter.priors['beta'] = model_fitter.make_gen(p_in['beta']-0.01, p_in['beta']+0.01)
    fitter.priors['muL_E'] = model_fitter.make_gen(p_in['muL_E']-0.01, p_in['muL_E']+0.01)
    fitter.priors['muL_N'] = model_fitter.make_gen(p_in['muL_N']-0.01, p_in['muL_N']+0.01)
    fitter.priors['muS_E'] = model_fitter.make_gen(p_in['muS_E']-0.01, p_in['muS_E']+0.01)
    fitter.priors['muS_N'] = model_fitter.make_gen(p_in['muS_N']-0.01, p_in['muS_N']+0.01)
    fitter.priors['dL'] = model_fitter.make_gen(p_in['dL']-10, p_in['dL']+10)
    fitter.priors['dL_dS'] = model_fitter.make_gen((p_in['dL']/p_in['dS'])-0.01, (p_in['dL']/p_in['dS'])+0.01)
    fitter.priors['b_sff1'] = model_fitter.make_gen(p_in['b_sff1']-0.01, p_in['b_sff1']+0.01)
    fitter.priors['mag_src1'] = model_fitter.make_gen(p_in['mag_src1']-0.01, p_in['mag_src1']+0.01)
    fitter.priors['b_sff2'] = model_fitter.make_gen(p_in['b_sff2']-0.01, p_in['b_sff2']+0.01)
    fitter.priors['mag_src2'] = model_fitter.make_gen(p_in['mag_src2']-0.01, p_in['mag_src2']+0.01)
    fitter.priors['b_sff3'] = model_fitter.make_gen(p_in['b_sff3']-0.01, p_in['b_sff3']+0.01)
    fitter.priors['mag_src3'] = model_fitter.make_gen(p_in['mag_src3']-0.01, p_in['mag_src3']+0.01)
    fitter.priors['b_sff4'] = model_fitter.make_gen(p_in['b_sff4']-0.01, p_in['b_sff4']+0.01)
    fitter.priors['mag_src4'] = model_fitter.make_gen(p_in['mag_src4']-0.01, p_in['mag_src4']+0.01)

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(left=0.2)
    plt.plot(data['xpos1'], data['ypos1'], '.')
    plt.show()

#    fitter.solve()

    best = fitter.get_best_fit()

    pspl_out = model.PSPL_PhotAstrom_LumLens_Par_Param1(best['mL'],
                                                best['t0'],
                                                best['beta'],
                                                best['dL'],
                                                best['dL_dS'],
                                                best['xS0_E'],
                                                best['xS0_N'],
                                                best['muL_E'],
                                                best['muL_N'],
                                                best['muS_E'],
                                                best['muS_N'],
                                                [best['b_sff1'], best['b_sff2'], best['b_sff3'], best['b_sff4']],
                                                [best['mag_src1'], best['mag_src2'], best['mag_src3'], best['mag_src4']],
                                                raL=p_in['raL'],
                                                decL=p_in['decL'])

    pspl_in = model.PSPL_PhotAstrom_LumLens_Par_Param1(p_in['mL'],
                                               p_in['t0'],
                                               p_in['beta'],
                                               p_in['dL'],
                                               p_in['dL'] / p_in['dS'],
                                               p_in['xS0_E'],
                                               p_in['xS0_N'],
                                               p_in['muL_E'],
                                               p_in['muL_N'],
                                               p_in['muS_E'],
                                               p_in['muS_N'],
                                               [p_in['b_sff1'], p_in['b_sff2'], p_in['b_sff3'], p_in['b_sff4']],
                                               [p_in['mag_src1'], p_in['mag_src2'], p_in['mag_src3'], p_in['mag_src4']],
                                               raL=p_in['raL'],
                                               decL=p_in['decL'])


    p_in['dL_dS'] = p_in['dL'] / p_in['dS']
    p_in['tE'] = pspl_in.tE
    p_in['thetaE'] = pspl_in.thetaE_amp
    p_in['piE_E'] = pspl_in.piE[0]
    p_in['piE_N'] = pspl_in.piE[1]
    p_in['u0_amp'] = pspl_in.u0_amp
    p_in['muRel_E'] = pspl_in.muRel[0]
    p_in['muRel_N'] = pspl_in.muRel[1]

    fitter.summarize_results()
    fitter.plot_dynesty_style(sim_vals=p_in)
    fitter.plot_model_and_data(pspl_out, input_model=pspl_in)

    imag_out = pspl_out.get_photometry(data['t_phot1'])
    pos_out = pspl_out.get_astrometry(data['t_ast1'])

    imag_in = pspl_in.get_photometry(data['t_phot1'])
    pos_in = pspl_in.get_astrometry(data['t_ast1'])

    np.testing.assert_array_almost_equal(imag_out, imag_in, 1)
    np.testing.assert_array_almost_equal(pos_out, pos_in, 4)

    print("OUTPUT:")
    lnL_out = fitter.log_likely(best, verbose=True)
    print("INPUT:")
    lnL_in = fitter.log_likely(p_in, verbose=True)

    assert np.abs(lnL_out - lnL_in) < 10

    return

def test_pspl_parallax_fit():
    outdir = './test_mnest_lmc/'
    os.makedirs(outdir, exist_ok=True)

    data, p_in = test_model.fake_data_parallax_lmc()

    fitter = PSPL_Solver(data,
                         model.PSPL_PhotAstrom_Par_Param1,
                         n_live_points=300,
                         outputfiles_basename=outdir + '/aa_',
                         resume=False)

    # Lets adjust some priors for faster solving.
    fitter.priors['t0'] = model_fitter.make_gen(p_in['t0']-5, p_in['t0']+5)
    fitter.priors['xS0_E'] = model_fitter.make_gen(p_in['xS0_E']-1e-3, p_in['xS0_E']+1e-3)
    fitter.priors['xS0_N'] = model_fitter.make_gen(p_in['xS0_N']-1e-3, p_in['xS0_N']+1e-3)
    fitter.priors['beta'] = model_fitter.make_gen(p_in['beta']-0.1, p_in['beta']+0.1)
    fitter.priors['muL_E'] = model_fitter.make_gen(p_in['muL_E']-0.1, p_in['muL_E']+0.1)
    fitter.priors['muL_N'] = model_fitter.make_gen(p_in['muL_N']-0.1, p_in['muL_N']+0.1)
    fitter.priors['muS_E'] = model_fitter.make_gen(p_in['muS_E']-0.1, p_in['muS_E']+0.1)
    fitter.priors['muS_N'] = model_fitter.make_gen(p_in['muS_N']-0.1, p_in['muS_N']+0.1)
    fitter.priors['dL'] = model_fitter.make_gen(p_in['dL']-100, p_in['dL']+100)
    fitter.priors['dL_dS'] = model_fitter.make_gen((p_in['dL']/p_in['dS'])-0.1, (p_in['dL']/p_in['dS'])+0.1)
    fitter.priors['b_sff1'] = model_fitter.make_gen(p_in['b_sff']-0.1, p_in['b_sff']+0.1)
    fitter.priors['mag_src1'] = model_fitter.make_gen(p_in['mag_src']-0.1, p_in['mag_src']+0.1)
    
    fitter.solve()

    best = fitter.get_best_fit()

    pspl_out = model.PSPL_PhotAstrom_Par_Param1(best['mL'],
                                                best['t0'],
                                                best['beta'],
                                                best['dL'],
                                                best['dL_dS'],
                                                best['xS0_E'],
                                                best['xS0_N'],
                                                best['muL_E'],
                                                best['muL_N'],
                                                best['muS_E'],
                                                best['muS_N'],
                                                [best['b_sff1']],
                                                [best['mag_src1']],
                                                raL=p_in['raL'],
                                                decL=p_in['decL'])

    pspl_in = model.PSPL_PhotAstrom_Par_Param1(p_in['mL'],
                                               p_in['t0'],
                                               p_in['beta'],
                                               p_in['dL'],
                                               p_in['dL'] / p_in['dS'],
                                               p_in['xS0_E'],
                                               p_in['xS0_N'],
                                               p_in['muL_E'],
                                               p_in['muL_N'],
                                               p_in['muS_E'],
                                               p_in['muS_N'],
                                               p_in['b_sff'],
                                               p_in['mag_src'],
                                               raL=p_in['raL'],
                                               decL=p_in['decL'])


    p_in['dL_dS'] = p_in['dL'] / p_in['dS']
    p_in['tE'] = pspl_in.tE
    p_in['thetaE'] = pspl_in.thetaE_amp
    p_in['piE_E'] = pspl_in.piE[0]
    p_in['piE_N'] = pspl_in.piE[1]
    p_in['u0_amp'] = pspl_in.u0_amp
    p_in['muRel_E'] = pspl_in.muRel[0]
    p_in['muRel_N'] = pspl_in.muRel[1]
    p_in['b_sff1'] = p_in['b_sff']
    p_in['mag_src1'] = p_in['mag_src']

    fitter.summarize_results()
    fitter.plot_dynesty_style(sim_vals=p_in)
    fitter.plot_model_and_data(pspl_out, input_model=pspl_in)

    imag_out = pspl_out.get_photometry(data['t_phot1'])
    pos_out = pspl_out.get_astrometry(data['t_ast1'])

    imag_in = pspl_in.get_photometry(data['t_phot1'])
    pos_in = pspl_in.get_astrometry(data['t_ast1'])

    np.testing.assert_array_almost_equal(imag_out, imag_in, 1)
    np.testing.assert_array_almost_equal(pos_out, pos_in, 4)

    # print("OUTPUT:")
    lnL_out = fitter.log_likely(best) # , verbose=True)
    # print("INPUT:")
    lnL_in = fitter.log_likely(p_in) # , verbose=True)
 
    assert np.abs(lnL_out - lnL_in) < 50

    return


def plot_mnest_test(data, imag_in, imag_out, pos_in, pos_out, outroot):
    plt.figure(1)
    plt.clf()
    plt.errorbar(data['t_phot1'], data['mag1'], yerr=data['mag_err1'], fmt='k.')
    plt.plot(data['t_phot1'], imag_out, 'r-')
    plt.plot(data['t_phot1'], imag_in, 'g-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('I (mag)')
    plt.title('Input Data and Output Model')
    plt.savefig(outroot + '_phot.png')

    plt.figure(2)
    plt.clf()
    plt.errorbar(data['xpos'], data['ypos'], xerr=data['xpos_err'],
                 yerr=data['ypos_err'], fmt='k.')
    plt.plot(pos_out[:, 0], pos_out[:, 1], 'r-')
    plt.plot(pos_in[:, 0], pos_in[:, 1], 'g-')
    plt.gca().invert_xaxis()
    plt.xlabel('X Pos (")')
    plt.ylabel('Y Pos (")')
    plt.title('Input Data and Output Model')
    plt.savefig(outroot + '_ast.png')

    plt.figure(3)
    plt.clf()
    plt.errorbar(data['t_ast'], data['xpos'], yerr=data['xpos_err'], fmt='k.')
    plt.plot(data['t_ast'], pos_out[:, 0], 'r-')
    plt.plot(data['t_ast'], pos_in[:, 0], 'g-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('X Pos (")')
    plt.title('Input Data and Output Model')
    plt.savefig(outroot + '_t_vs_E.png')

    plt.figure(4)
    plt.clf()
    plt.errorbar(data['t_ast'], data['ypos'], yerr=data['ypos_err'], fmt='k.')
    plt.plot(data['t_ast'], pos_out[:, 1], 'r-')
    plt.plot(data['t_ast'], pos_in[:, 1], 'g-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('Y Pos (")')
    plt.title('Input Data and Output Model')
    plt.savefig(outroot + '_t_vs_N.png')

    return


def test_make_t0_gen():
    data, p_in = test_model.fake_data1()

    t0_gen = model_fitter.make_t0_gen(data['t_phot1'], data['mag1'])

    t0_rand = t0_gen.rvs(size=100)

    plt.clf()
    plt.plot(data['t_phot1'], data['mag1'], 'k.')
    plt.axvline(t0_rand.min())
    plt.axvline(t0_rand.max())
    print('t0 between: ', t0_rand.min(), t0_rand.max())

    assert t0_rand.min() < 56990
    assert t0_rand.max() > 57000

    return


def test_PSPL_Solver(plot=False):
    # This is an old run from when the 0 sign was wrong.
    outdir = './test_pspl_solver/'
    base = outdir + 'aa'

    # Make directory if it doesn't exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data, p_in = test_model.fake_data1()

    fitter = PSPL_Solver(data,
                         model.PSPL_PhotAstrom_noPar_Param1,
                         n_live_points=100,
                         outputfiles_basename=base,
                         resume=False, verbose=False)

    # Lets adjust some priors for faster solving.
    fitter.priors['mL'] = model_fitter.make_gen(5.0, 15.0)
    fitter.priors['t0'] = model_fitter.make_gen(56990, 57010)
    fitter.priors['beta'] = model_fitter.make_gen(-0.5, -0.3)
    fitter.priors['muL_E'] = model_fitter.make_gen(-1, 1)
    fitter.priors['muL_N'] = model_fitter.make_gen(-8, -6)
    fitter.priors['muS_E'] = model_fitter.make_gen(0, 3)
    fitter.priors['muS_N'] = model_fitter.make_gen(-2, 1)
    fitter.priors['dL'] = model_fitter.make_gen(3000, 5000)
    fitter.priors['dL_dS'] = model_fitter.make_gen(0.45, 0.55)
    fitter.priors['b_sff1'] = model_fitter.make_gen(0.5, 1.1)
    fitter.priors['mag_src1'] = model_fitter.make_gen(18.9, 19.1)
    fitter.priors['xS0_E'] = model_fitter.make_gen(-10 ** -3, 10 ** -3)
    fitter.priors['xS0_N'] = model_fitter.make_gen(-10 ** -3, 10 ** -3)

    # fitter.solve()

    best = fitter.get_best_fit()

    pspl_out = model.PSPL_PhotAstrom_noPar_Param1(mL=best['mL'],
                                                t0=best['t0'],
                                                beta=best['beta'],
                                                dL=best['dL'],
                                                dL_dS=best['dL_dS'],
                                                xS0_E=best['xS0_E'],
                                                xS0_N=best['xS0_N'],
                                                muL_E=best['muL_E'],
                                                muL_N=best['muL_N'],
                                                muS_E=best['muS_E'],
                                                muS_N=best['muS_N'],
                                                mag_src=[best['mag_src1']],
                                                b_sff=[best['b_sff1']])

    pspl_in = model.PSPL_PhotAstrom_noPar_Param1(mL=p_in['mL'],
                                               t0=p_in['t0'],
                                               beta=p_in['beta'],
                                               dL=p_in['dL'],
                                               dL_dS=p_in['dL'] / p_in['dS'],
                                               xS0_E=p_in['xS0_E'],
                                               xS0_N=p_in['xS0_N'],
                                               muL_E=p_in['muL_E'],
                                               muL_N=p_in['muL_N'],
                                               muS_E=p_in['muS_E'],
                                               muS_N=p_in['muS_N'],
                                               b_sff=[p_in['b_sff1']],
                                               mag_src=[p_in['mag_src1']])


    p_in['dL_dS'] = p_in['dL'] / p_in['dS']
    p_in['tE'] = pspl_in.tE
    p_in['thetaE'] = pspl_in.thetaE_amp
    p_in['piE_E'] = pspl_in.piE[0]
    p_in['piE_N'] = pspl_in.piE[1]
    p_in['u0_amp'] = pspl_in.u0_amp
    p_in['muRel_E'] = pspl_in.muRel[0]
    p_in['muRel_N'] = pspl_in.muRel[1]

    # Save the data for future plotting.
    pickle_data(base, data, p_in)

    # Compare input and output model paramters
    for param in best.colnames:
        try:
            frac_diff = np.abs(p_in[param] - best[param])
            
            if p_in[param] != 0:
                frac_diff /= p_in[param]
                
            assert frac_diff < 0.2
        except KeyError:
            pass

    if plot:
        fitter.plot_dynesty_style(sim_vals=p_in)
        fitter.plot_model_and_data(pspl_out, input_model=pspl_in)

    lnL_out = fitter.log_likely(best, verbose=False)
    lnL_in = fitter.log_likely(p_in, verbose=False)

    assert np.abs(lnL_out - lnL_in) < 100

    return


# NOTE: some of plotting stuff is not functioning... 
def test_correlated_data2():
    true_model, data, data_corr, params = test_model.fake_correlated_data()
    
    data['phot_files'] = 'fake'
    data_corr['phot_files'] = 'fake'
    data['ast_files'] = 'fake'
    data_corr['ast_files'] = 'fake'

    #####
    # Fit correlated data with GP
    #####
    base = './test_correlated_data/aa_'

    fitter_corr = PSPL_Solver(data_corr,
                              model.PSPL_Phot_Par_GP_Param2,
                              n_live_points=100,
                              outputfiles_basename=base,
                              resume=False, verbose=False)

    # Lets adjust some priors for faster solving.
    fitter_corr.priors['t0'] = model_fitter.make_gen(57000 - 1, 57000 + 1)
    fitter_corr.priors['u0_amp'] = model_fitter.make_gen(0.1 - 0.05, 0.1 + 0.05)
    fitter_corr.priors['tE'] = model_fitter.make_gen(150 - 5, 150 + 5)
    fitter_corr.priors['piE_E'] = model_fitter.make_gen(0.05 - 0.01, 0.05 + 0.01)
    fitter_corr.priors['piE_N'] = model_fitter.make_gen(0.05 - 0.01, 0.05 + 0.01)
    fitter_corr.priors['b_sff1'] = model_fitter.make_gen(0.9 - 0.1, 0.9 + 0.1)
    fitter_corr.priors['mag_src1'] = model_fitter.make_gen(19.0 - 0.1, 19.0 + 0.1)
    fitter_corr.priors['gp_log_sigma'] = model_fitter.make_norm_gen(0.5, 1.5)
    fitter_corr.priors['gp_rho'] = model_fitter.make_invgamma_gen(data_corr['t_phot1'])
    fitter_corr.priors['gp_log_So'] = model_fitter.make_norm_gen(0.5, 1.5)
    fitter_corr.priors['gp_log_omegao'] = model_fitter.make_norm_gen(0.5, 1.5)

    # fitter_corr.solve()
    fitter_corr.plot_dynesty_style(sim_vals=params, remake_fits=True)
    best_mod = fitter_corr.get_best_fit_model(def_best='maxl')
    mnest_results = fitter_corr.load_mnest_results(remake_fits=True)
    fitter_corr.plot_model_and_data(best_mod, input_model=true_model, gp=True,
                                    zoomx=[[56800, 57200]])

    return


def test_correlated_data_astrom():
    true_model, data, data_corr, params = test_model.fake_correlated_data_with_astrom()

    data['phot_files'] = 'fake'
    data_corr['phot_files'] = 'fake'
    data['ast_files'] = 'fake'
    data_corr['ast_files'] = 'fake'

    #####
    # Fit correlated data with GP
    #####
    base = './test_correlated_data_astrom/corr_gp_'

    fitter_corr = PSPL_Solver(data_corr,
                              model.PSPL_PhotAstrom_Par_GP_Param2,
                              n_live_points=100,
                              outputfiles_basename=base,
                              resume=False, verbose=False)

    fitter_corr.priors['t0'] = model_fitter.make_gen(params['t0']-5, params['t0']+5)
    fitter_corr.priors['u0_amp'] = model_fitter.make_gen(params['u0_amp']-0.01, params['u0_amp']+0.01)
    fitter_corr.priors['tE'] = model_fitter.make_gen(params['tE']-5, params['tE']+5)
    fitter_corr.priors['thetaE'] = model_fitter.make_gen(params['thetaE']-0.2, params['thetaE']+0.2)
    fitter_corr.priors['piS'] = model_fitter.make_gen(params['piS']-0.05, params['piS']+0.05)
    fitter_corr.priors['piE_E'] = model_fitter.make_gen(params['piE_E']-0.01, params['piE_E']+0.01)
    fitter_corr.priors['piE_N'] = model_fitter.make_gen(params['piE_N']-0.01, params['piE_N']+0.01)
    fitter_corr.priors['xS0_E'] = model_fitter.make_gen(params['xS0_E']-1E-3, params['xS0_E']+1E-3)
    fitter_corr.priors['xS0_N'] = model_fitter.make_gen(params['xS0_N']-1E-3, params['xS0_N']+1E-3)
    fitter_corr.priors['muS_E'] = model_fitter.make_gen(params['muS_E']-0.01, params['muS_E']+0.01)
    fitter_corr.priors['muS_N'] = model_fitter.make_gen(params['muS_N']-0.01, params['muS_N']+0.01)
    fitter_corr.priors['b_sff1'] = model_fitter.make_gen(params['b_sff1']-0.1, params['b_sff1']+0.1)
    fitter_corr.priors['mag_src1'] = model_fitter.make_gen(params['mag_src1']-0.5, params['mag_src1']+0.5)

    fitter_corr.priors['gp_log_sigma'] = model_fitter.make_norm_gen(0.5,1.5)
    fitter_corr.priors['gp_rho'] = model_fitter.make_invgamma_gen(data_corr['t_phot1'])
    fitter_corr.priors['gp_log_omegaofour_So'] = model_fitter.make_norm_gen(np.median(data['mag_err1'])**2, 5)
    fitter_corr.priors['gp_log_omegao'] = model_fitter.make_norm_gen(0.5,1.5)

    fitter_corr.solve()

    fitter_corr.plot_dynesty_style(sim_vals=params, remake_fits=True)
#    best_mod = fitter_corr.get_best_fit_model(def_best='maxl')
#    fitter_corr.plot_model_and_data(best_mod, input_model=true_model, gp=True,
#                                    mnest_results=mnest_results,
#                                    zoomx=[[56800, 57200]])


def test_PSBL_parallax_Solver():
    base = './test_psbl_parallax_solver/aa'

    data, p_in, psbl, ani = test_model.fake_data_PSBL(parallax=True)

    fitter = PSPL_Solver(data, n_live_points=100,
                         outputfiles_basename=base,
                         resume=False)

    # Lets adjust some priors for faster solving.
    fitter.priors['mL1'] = model_fitter.make_gen(0.1, 100.0)
    fitter.priors['mL2'] = model_fitter.make_gen(0.1, 100.0)
    fitter.priors['t0'] = model_fitter.make_gen(56990, 57010)
    fitter.priors['beta'] = model_fitter.make_gen(-0.5, -0.3)
    fitter.priors['muL_E'] = model_fitter.make_gen(-1, 1)
    fitter.priors['muL_N'] = model_fitter.make_gen(-8, -6)
    fitter.priors['muS_E'] = model_fitter.make_gen(0, 3)
    fitter.priors['muS_N'] = model_fitter.make_gen(-2, 1)
    fitter.priors['dL'] = model_fitter.make_gen(3000, 5000)
    fitter.priors['dL_dS'] = model_fitter.make_gen(0.45, 0.55)
    fitter.priors['b_sff'] = model_fitter.make_gen(0.5, 1.1)
    fitter.priors['mag_src'] = model_fitter.make_gen(18.9, 19.1)
    fitter.priors['xS0_E'] = model_fitter.make_gen(-10**-3, 10**-3)
    fitter.priors['xS0_N'] = model_fitter.make_gen(-10**-3, 10**-3)
    fitter.priors['sep'] = model_fitter.make_gen(1e-3, 1e-2)
    fitter.priors['alpha'] = model_fitter.make_gen(0, 210)

    fitter.solve()

    best = fitter.get_best_fit()

    psbl_out = model.PSBL_PhotAstrom_Par_Param1(best['mL'], best['t0'], np.array([best['xS0_E'], best['xS0_N']]), best['beta'],
                                       np.array([best['muL_E'], best['muL_N']]), np.array([best['muS_E'], best['muS_N']]),
                                       best['dL'], best['dS'], best['sep'], best['alpha'], best['b_sff'], best['mag_src'])

    psbl_in = model.PSBL_PhotAstrom_Par_Param1(p_in['mL'], p_in['t0'], np.array([p_in['xS0_E'], p_in['xS0_N']]), p_in['beta'],
                                      np.array([p_in['muL_E'], p_in['muL_N']]), np.array([p_in['muS_E'], p_in['muS_N']]),
                                      p_in['dL'], p_in['dS'], p_in['sep'], p_in['alpha'], p_in['b_sff'], p_in['mag_src'])

    p_in['dL_dS'] = p_in['dL'] / p_in['dS']
    p_in['tE'] = psbl_in.tE
    p_in['thetaE'] = psbl_in.thetaE_amp
    p_in['piE_E'] = psbl_in.piE[0]
    p_in['piE_N'] = psbl_in.piE[1]
    p_in['u0_amp'] = psbl_in.u0_amp
    p_in['muRel_E'] = psbl_in.muRel[0]
    p_in['muRel_N'] = psbl_in.muRel[1]

    # Save the data for future plotting.
    pickle_data(base, data, p_in)

    fitter.summarize_results()
    fitter.plot_dynesty_style(sim_vals=p_in)
    fitter.plot_model_and_data(psbl_out, input_model=psbl_in)

    print("OUTPUT:")
    lnL_out = fitter.log_likely(best, verbose=True)
    print("INPUT:")
    lnL_in = fitter.log_likely(p_in, verbose=True)

    return

def test_PSBL_parallax2_Solver():
#    base = './test_psbl_parallax2_solver/aa'
    base = './test_psbl_solver/bb' # Casey

    data, p_in, psbl, ani = test_model.fake_data_PSBL(parallax=True)

    fitter = PSPL_Solver(data, n_live_points=100,
                         outputfiles_basename=base,
                         resume=False)

    # Lets adjust some priors for faster solving.
    fitter.priors['t0'] = model_fitter.make_gen(56990, 57010)
    fitter.priors['u0_amp'] = model_fitter.make_gen(1.0, 1.2)
    fitter.priors['tE'] = model_fitter.make_gen(800, 900)
    fitter.priors['piE_E'] = model_fitter.make_gen(0.0, 0.04)
    fitter.priors['piE_N'] = model_fitter.make_gen(-0.01, 0.01)
    fitter.priors['b_sff'] = model_fitter.make_gen(0.5, 1.1)
    fitter.priors['mag_src'] = model_fitter.make_gen(17.9, 18.1)
    fitter.priors['thetaE'] = model_fitter.make_gen(4.0, 5.0)

    fitter.priors['muS_E'] = model_fitter.make_gen(0, 3)
    fitter.priors['muS_N'] = model_fitter.make_gen(-2, 1)
    fitter.priors['piS'] = model_fitter.make_gen(0.1, 0.15)
    fitter.priors['xS0_E'] = model_fitter.make_gen(-10**-3, 10**-3)
    fitter.priors['xS0_N'] = model_fitter.make_gen(-10**-3, 10**-3)

    fitter.priors['q'] = model_fitter.make_gen(0.9, 1.1)
    fitter.priors['sep'] = model_fitter.make_gen(1e-3, 1e-2)
    fitter.priors['alpha'] = model_fitter.make_gen(80, 100)

    fitter.solve()

    best = fitter.get_best_fit()

    psbl_out = model.PSBL_PhotAstrom_Par_Param2(best['t0'], best['u0_amp'], best['tE'], best['piE_E'], best['piE_N'],
                                        best['b_sff'], best['mag_src'], best['thetaE'],
                                        best['xS0_E'], best['xS0_N'], best['muS_E'], best['muS_N'], best['piS'],
                                        best['q'], best['sep'], best['alpha'])

    psbl_in = psbl

    p_in['mL1'] = psbl_in.mL1
    p_in['mL2'] = psbl_in.mL2
    p_in['muL_E'] = psbl_in.muL[0]
    p_in['muL_N'] = psbl_in.muL[1]
    p_in['muRel_E'] = psbl_in.muRel[0]
    p_in['muRel_N'] = psbl_in.muRel[1]
    p_in['piL'] = psbl_in.piL
    p_in['piRel'] = psbl_in.piRel
    p_in['piE'] = psbl_in.piE

    # Save the data for future plotting.
    pickle_data(base, data, p_in)

    fitter.summarize_results()
    fitter.plot_dynesty_style(sim_vals=p_in)
    fitter.plot_model_and_data(psbl_out, input_model=psbl_in)

    print("OUTPUT:")
    lnL_out = fitter.log_likely(best, verbose=True)
    print("INPUT:")
    lnL_in = fitter.log_likely(p_in, verbose=True)

    return


def test_PSBL_phot_nopar_fit(regen=False, fit=False, summarize=False, suffix=''):
    # Choose which base you want to use.
    # Comment out the one you don't want.
    base = './test_psbl_phot_solver/nopar' + suffix + '_'

    # Make directory if it doesn't exist.
    if not os.path.exists('./test_psbl_phot_solver/'):
        os.makedirs('./test_psbl_phot_solver/')

    pkl_file = base + 'data' + suffix + '.pkl'

    if os.path.exists(pkl_file) and regen == False:
        # Load previously generated data from pickle file. 
        _data_pkl = open(pkl_file, 'rb')
        data = pickle.load(_data_pkl)
        p_in_tmp = pickle.load(_data_pkl)
        psbl_in = pickle.load(_data_pkl)
        _data_pkl.close()
    else:
        # Generate random data
        # {'t0': 57000, 'u0_amp': 0.7677249386310591, 'tE': 504.61579173573404,
        # 'piE_E': 0.022619312924845883, 'piE_N': 0.022619312924845883,
        # 'q': 0.5, 'sep': 5.0, 'phi': 285.0, 'b_sff1': 0.5, 'mag_src1': 16}
        data, p_in_tmp, psbl_in, ani = test_model.fake_data_PSBL(mL1 = 10, mL2 = 5,
                                                                 beta = 3.0, sep = 5.0,
                                                                 muL_E = -1.0, muL_N = -1.0,
                                                                 alpha = -30,
                                                                 mag_src = 16, b_sff=1.0,
                                                                 parallax=False, 
                                                                 animate=False)

        # Save to pickle file.
        _data_pkl = open(pkl_file, 'wb')
        pickle.dump(data, _data_pkl)
        pickle.dump(p_in_tmp, _data_pkl)
        pickle.dump(psbl_in, _data_pkl)
        _data_pkl.close()

    # Get astrometry for plotting
    xL1, xL2 = psbl_in.get_resolved_lens_astrometry(np.array([psbl_in.t0]))
    muRel = psbl_in.muRel
    xL1 = xL1[0]

    muRel_angle = np.degrees(np.arctan2(psbl_in.muRel[0], psbl_in.muRel[1]))
    phi = psbl_in.alpha - muRel_angle
    phi = phi % 360 # angle wrapping

    # Make a p_in suitable for photometry.
    p_in = {}
    p_in['t0'] = psbl_in.t0
    p_in['u0_amp'] = psbl_in.u0_amp
    p_in['tE'] = psbl_in.tE
    p_in['piE_E'] = psbl_in.piE[0]
    p_in['piE_N'] = psbl_in.piE[1]
    p_in['q'] = psbl_in.mL2 / psbl_in.mL1
    p_in['sep'] = psbl_in.sep / psbl_in.thetaE_amp    
    p_in['phi'] = phi
    p_in['b_sff1'] = p_in_tmp['b_sff']    
    p_in['mag_src1'] = p_in_tmp['mag_src']
    print('INPUT:')
    print(p_in)

    # Fit the data.
    fitter = PSPL_Solver(data, model.PSBL_Phot_noPar_Param1,
                         n_live_points=100,
                         outputfiles_basename=base,
                         resume=False, verbose=False)
    
    # Lets adjust some priors for faster solving.
    # fitter.priors['t0'] = model_fitter.make_gen(56999, 57001)
    # fitter.priors['u0_amp'] = model_fitter.make_gen(0.76, 0.78)
    # fitter.priors['tE'] = model_fitter.make_gen(500, 510)
    # fitter.priors['piE_E'] = model_fitter.make_gen(0.020, 0.025)
    # fitter.priors['piE_N'] = model_fitter.make_gen(0.020, 0.025)
    # fitter.priors['q'] = model_fitter.make_gen(0.49, 0.51)
    # fitter.priors['sep'] = model_fitter.make_gen(1.27, 1.29)
    # fitter.priors['phi'] = model_fitter.make_gen(284.0, 286.0)
    # fitter.priors['b_sff1'] = model_fitter.make_gen(0.999, 1.001)
    # fitter.priors['mag_src1'] = model_fitter.make_gen(15.9, 16.1)

    fitter.priors['t0'] = model_fitter.make_gen(56900, 57100)
    fitter.priors['u0_amp'] = model_fitter.make_gen(0.5, 1.0)
    fitter.priors['tE'] = model_fitter.make_gen(450, 550)
    fitter.priors['piE_E'] = model_fitter.make_gen(0.0, 0.04)
    fitter.priors['piE_N'] = model_fitter.make_gen(0.0, 0.04)
    fitter.priors['q'] = model_fitter.make_gen(0.4, 0.6)
    fitter.priors['sep'] = model_fitter.make_gen(0.5, 3)
    fitter.priors['phi'] = model_fitter.make_gen(0, 360)
    fitter.priors['b_sff1'] = model_fitter.make_gen(0.90, 1.1)
    fitter.priors['mag_src1'] = model_fitter.make_gen(15.9, 16.1)
    
    
    # Sampler = dynesty.DynamicNestedSampler(fitter.LogLikelihood, fitter.Prior, 
    #                                        ndim=fitter.n_dims, bound='multi',
    #                                        sample='unif')#, pool = pool,
    #                                        #queue_size = n_cpu)

    # sampler.run_nested(nlive_init=100, print_progress=True, maxiter=2000)
    
    if fit == True:
        fitter.solve()

    fitter.load_mnest_results(remake_fits=True)
    fitter.load_mnest_summary(remake_fits=True)
    
    if summarize == True:
        fitter.summarize_results()
        
        fitter.plot_model_and_data(fitter.get_best_fit_model(), input_model=psbl_in)    
        plt.close('all')
        
        print("OUTPUT:")
        lnL_out = fitter.log_likely(fitter.get_best_fit(), verbose=True)
        
        print("")
        print("INPUT:")
        lnL_in = fitter.log_likely(p_in, verbose=True)

        fitter.plot_dynesty_style(sim_vals=p_in)
        plt.close('all')
        
    
    return


def test_PSBL_phot_par_fit(regen=False, fit=False, summarize=False, suffix=''):
    # Choose which base you want to use.
    # Comment out the one you don't want.
    base = './test_psbl_phot_solver/par' + suffix + '_'

    # Make directory if it doesn't exist.
    if not os.path.exists('./test_psbl_phot_solver/'):
        os.makedirs('./test_psbl_phot_solver/')

    pkl_file = base + 'data' + suffix + '.pkl'

    if os.path.exists(pkl_file) and regen == False:
        # Load previously generated data from pickle file. 
        _data_pkl = open(pkl_file, 'rb')
        data = pickle.load(_data_pkl)
        p_in_tmp = pickle.load(_data_pkl)
        psbl_in = pickle.load(_data_pkl)
        _data_pkl.close()
    else:
        # Generate random data
        data, p_in_tmp, psbl_in, ani = test_model.fake_data_PSBL(mL1 = 10, mL2 = 5,
                                                                 beta = 3.0, sep = 5.0,
                                                                 muL_E = -1.0, muL_N = -1.0,
                                                                 alpha = -30,
                                                                 mag_src = 16, b_sff=1.0,
                                                                 parallax=True, 
                                                                 animate=False)

        data['target'] = 'test'
        # Save to pickle file.
        _data_pkl = open(pkl_file, 'wb')
        pickle.dump(data, _data_pkl)
        pickle.dump(p_in_tmp, _data_pkl)
        pickle.dump(psbl_in, _data_pkl)
        _data_pkl.close()

    # Get astrometry for plotting
    xL1, xL2 = psbl_in.get_resolved_lens_astrometry(np.array([psbl_in.t0]))
    muRel = psbl_in.muRel
    xL1 = xL1[0]

    muRel_angle = np.degrees(np.arctan2(psbl_in.muRel[0], psbl_in.muRel[1]))
    phi = psbl_in.alpha - muRel_angle
    phi = phi % 360 # angle wrapping
    
    # Make a p_in suitable for photometry.
    p_in = {}
    p_in['t0'] = psbl_in.t0
    p_in['u0_amp'] = psbl_in.u0_amp
    p_in['tE'] = psbl_in.tE
    p_in['piE_E'] = psbl_in.piE[0]
    p_in['piE_N'] = psbl_in.piE[1]
    p_in['q'] = psbl_in.mL2 / psbl_in.mL1
    p_in['sep'] = psbl_in.sep / psbl_in.thetaE_amp    
    p_in['phi'] = phi
    p_in['b_sff1'] = p_in_tmp['b_sff']    
    p_in['mag_src1'] = p_in_tmp['mag_src']
    print('INPUT:')
    print(p_in)

    fitter = PSPL_Solver(data, model.PSBL_Phot_Par_Param1,
                         n_live_points=100,
                         outputfiles_basename=base,
                         resume=False)

    # Lets adjust some priors for faster solving.
    # fitter.priors['t0'] = model_fitter.make_gen(56999, 57001)
    # fitter.priors['u0_amp'] = model_fitter.make_gen(0.76, 0.78)
    # fitter.priors['tE'] = model_fitter.make_gen(500, 510)
    # fitter.priors['piE_E'] = model_fitter.make_gen(0.020, 0.025)
    # fitter.priors['piE_N'] = model_fitter.make_gen(0.020, 0.025)
    # fitter.priors['q'] = model_fitter.make_gen(0.49, 0.51)
    # fitter.priors['sep'] = model_fitter.make_gen(1.27, 1.29)
    # fitter.priors['phi'] = model_fitter.make_gen(284.0, 286.0)
    # fitter.priors['b_sff1'] = model_fitter.make_gen(0.999, 1.001)
    # fitter.priors['mag_src1'] = model_fitter.make_gen(15.9, 16.1)
    
    fitter.priors['t0'] = model_fitter.make_gen(56900, 57100)
    fitter.priors['u0_amp'] = model_fitter.make_gen(0.5, 1.0)
    fitter.priors['tE'] = model_fitter.make_gen(450, 550)
    fitter.priors['piE_E'] = model_fitter.make_gen(0.0, 0.04)
    fitter.priors['piE_N'] = model_fitter.make_gen(0.0, 0.04)
    fitter.priors['q'] = model_fitter.make_gen(0.4, 0.6)
    fitter.priors['sep'] = model_fitter.make_gen(0.5, 3)
    fitter.priors['phi'] = model_fitter.make_gen(0, 360)
    fitter.priors['b_sff1'] = model_fitter.make_gen(0.90, 1.1)
    fitter.priors['mag_src1'] = model_fitter.make_gen(15.9, 16.1)
    
    # sampler = dynesty.DynamicNestedSampler(fitter.LogLikelihood, fitter.Prior, 
    #                                        ndim=fitter.n_dims, bound='multi',
    #                                        sample='unif')#, pool = pool,
    #                                        #queue_size = n_cpu)
    
    if fit == True:
        fitter.solve()

    fitter.load_mnest_results(remake_fits=True)
    fitter.load_mnest_summary(remake_fits=True)
        
    if summarize == True:
        fitter.summarize_results()
        
        fitter.plot_model_and_data(fitter.get_best_fit_model(), input_model=psbl_in)    
        plt.close('all')
        
        print("OUTPUT:")
        lnL_out = fitter.log_likely(fitter.get_best_fit(), verbose=True)
        
        print("")
        print("INPUT:")
        lnL_in = fitter.log_likely(p_in, verbose=True)

        fitter.plot_dynesty_style(sim_vals=p_in)
        plt.close('all')
        
    return

def pickle_data(basename, data, p_in):
    # Save the data for future plotting.
    _data_pkl = open(basename + '_data.pkl', 'wb')
    pickle.dump(data, _data_pkl)
    pickle.dump(p_in, _data_pkl)
    _data_pkl.close()

    return


def load_pickled_data(basename):
    # Save the data for future plotting.
    _data_pkl = open(basename + '_data.pkl', 'rb')
    data = pickle.load(_data_pkl)
    p_in = pickle.load(_data_pkl)
    _data_pkl.close()

    return data, p_in

def test_generate_params_dict():
    ##########
    # Test 1: input list same length as fitter_param_names
    ##########
    vals1 = np.arange(10)
    names1 = ['p{0:d}'.format(ii) for ii in range(10)]
    results1 = model_fitter.generate_params_dict(vals1, names1)
    assert len(results1) == 10
    assert type(results1).__name__ == 'dict'
    assert results1[names1[0]] == vals1[0]
    assert results1[names1[-1]] == vals1[-1]

    ##########
    # Test 2: input list larger than fitter_param_names
    ##########
    vals2 = np.arange(10)
    names2 = ['p{0:d}'.format(ii) for ii in range(8)]
    results2 = model_fitter.generate_params_dict(vals2, names2)
    assert len(results2) == 8
    assert type(results2).__name__ == 'dict'
    assert results2[names2[0]] == vals2[0]
    assert results2[names2[7]] == vals2[7]

    ##########
    # Test 3: input list contains mag_src and b_sff
    ##########
    vals3 = np.arange(6)
    names3 = ['p0', 'p1', 'mag_src1', 'b_sff1', 'mag_src2', 'b_sff2']
    results3 = model_fitter.generate_params_dict(vals3, names3)
    assert len(results3) == 4
    assert 'p0' in results3
    assert 'p1' in results3
    assert 'mag_src' in results3
    assert 'b_sff' in results3
    assert type(results3).__name__ == 'dict'
    assert results3['p0'] == vals3[0]
    assert results3['mag_src'][0] == vals3[2]

    ##########
    # Test 4: input dictionary contains mag_src and b_sff
    ##########
    vals4 = {'p0': 0, 'p1': 1, 'mag_src1': 2, 'b_sff1': 3, 'mag_src2': 4, 'b_sff2': 5}
    names4 = vals4.keys()
    results4 = model_fitter.generate_params_dict(vals4, names4)
    assert len(results4) == 4
    assert 'p0' in results4
    assert 'p1' in results4
    assert 'mag_src' in results4
    assert 'b_sff' in results4
    assert type(results4).__name__ == 'dict'
    assert results4['p0'] == vals4['p0']
    assert results4['mag_src'][0] == vals4['mag_src1']
    assert results4['b_sff'][0] == vals4['b_sff1']
    assert results4['mag_src'][1] == vals4['mag_src2']
    assert results4['b_sff'][1] == vals4['b_sff2']
    
    return

def test_u0_sign_change():
    new_u0_sign = True
    
    # This is an old run from when the 0 sign was wrong.
    outdir = './test_pspl_solver/'
    base = outdir + 'aa'

    # Make directory if it doesn't exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data, p_in = test_model.fake_data1()

    fitter = PSPL_Solver(data,
                         model.PSPL_PhotAstrom_noPar_Param1,
                         outputfiles_basename=base,
                         resume=False, verbose=False)

    # Modify the parameters into the new parameterization.
    if new_u0_sign:
        # Do some conversions.

        # Plot output suffix
        suffix = '_u0sign_new'
    else:
        suffix = '_u0sign_old'


    # Load them up from an old fit.
    best_model = fitter.get_best_fit()
    fitter.plot_model_and_data(pspl_out, input_model=pspl_in, suffix=suffix)

    # Compare to an old version of the plot made with
    # the git tag wrong_u0_sign_before_here (on master branch).
    
    
    
    
def test_u0_sign_change(new_u0_sign=False):
    # This is an old run from when the 0 sign was wrong. 
    old_base = './test_pspl_solver/aa_old_u0_'
    new_base = './test_pspl_solver/aa_new_u0_'

    if new_u0_sign:
        # Load up the data for the fitter object. 
        data, p_in = test_model.fake_data1(beta_sign = 1.0)
        
        multinest_utils.convert_pre_2020apr_u0_sign(old_base, new_base)

        base = new_base
        suffix = '_u0sign_new'
        
    else:
        # Load up the data for the fitter object. 
        data, p_in = test_model.fake_data1(beta_sign = -1.0)
        
        base = old_base
        suffix = '_u0sign_old'
        
    fitter = PSPL_Solver(data,
                         model.PSPL_PhotAstrom_noPar_Param1,
                         n_live_points=100,
                         outputfiles_basename=base,
                         resume=False, verbose=False)


    # Load them up from an old fit.
    best_model = fitter.get_best_fit_model()
    fitter.plot_model_and_data(best_model, suffix=suffix)

    # Compare to an old version of the plot made with
    # the git tag wrong_u0_sign_before_here (on master branch).
    # Visual inspection!
    
    return

def test_make_invgamma_gen():
    ob140613 = munge.getdata2('ob140613', phot_data=['I_OGLE'], ast_data=[])
    mb10364 = munge.getdata2('mb10364', phot_data=['MOA'], ast_data=[])
    model_fitter.make_invgamma_gen(ob140613['t_phot1'])
    model_fitter.make_invgamma_gen(mb10364['t_phot1'])

    return


def test_pspl_solver_gp_params():
    """
    Instantiate PSPL_Solver and make sure the optional GP 
    parameters are working fine. 
    """
    raL_in = 17.30 * 15.  # Bulge R.A.
    decL_in = -29.0
    t0_in = 57100.0
    u0_amp_in = 0.5
    tE_in = 100.0
    piE_E_in = 0.1
    piE_N_in = -0.1
    b_sff_in1 = 0.8
    mag_src_in1 = 19.0
    b_sff_in2 = 1.0
    mag_src_in2 = 17.0

    outdir = 'test_multiphot_data/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    data, params, true_model = test_model.fake_data_multiphot_parallax(raL_in, decL_in, t0_in,
                                                                       u0_amp_in, tE_in,
                                                                       piE_E_in, piE_N_in,
                                                                       b_sff_in1, mag_src_in1,
                                                                       b_sff_in2, mag_src_in2,
                                                                       target='Unknown',
                                                                       outdir=outdir)    
    
    
    data['phot_files'] = ['fake', 'fake2']
    data['ast_files'] = ['fake']

    #####
    # Instantiate solver with GP parameters set to defaults.
    #####
    base = outdir + 'aa_'

    fitter = PSPL_Solver(data,
                         model.PSPL_Phot_Par_GP_Param2,
                         n_live_points=100,
                         outputfiles_basename=base,
                         resume=False, verbose=False)

    # Check that the GP parameters will be used.
    # Their presence in the fitter_param_names instance variable means
    # they will be fit. 
    assert 'gp_log_sigma1' in fitter.fitter_param_names
    assert 'gp_log_rho1' in fitter.fitter_param_names
    assert 'gp_log_S01' in fitter.fitter_param_names
    assert 'gp_log_omega01' in fitter.fitter_param_names
    assert 'gp_log_sigma2' in fitter.fitter_param_names
    assert 'gp_log_rho2' in fitter.fitter_param_names
    assert 'gp_log_S02' in fitter.fitter_param_names
    assert 'gp_log_omega02' in fitter.fitter_param_names
    
    #####
    # Instantiate solver with GP parameters set on for the first
    # data set and off for the second.
    #####
    base = outdir + 'aa_'

    fitter = PSPL_Solver(data,
                         model.PSPL_Phot_Par_GP_Param2,
                         use_phot_optional_params = [True, False],
                         n_live_points=100,
                         outputfiles_basename=base,
                         resume=False, verbose=False)

    # Check that the GP parameters will be used.
    # Their presence in the fitter_param_names instance variable means
    # they will be fit. 
    assert 'gp_log_sigma1' in fitter.fitter_param_names
    assert 'gp_log_rho1' in fitter.fitter_param_names
    assert 'gp_log_S01' in fitter.fitter_param_names
    assert 'gp_log_omega01' in fitter.fitter_param_names
    assert 'gp_log_sigma2' not in fitter.fitter_param_names
    assert 'gp_log_rho2' not in fitter.fitter_param_names
    assert 'gp_log_S02' not in fitter.fitter_param_names
    assert 'gp_log_omega02' not in fitter.fitter_param_names
    
    #####
    # Instantiate solver with GP parameters set to off for all.
    #####
    base = outdir + 'aa_'

    fitter = PSPL_Solver(data,
                         model.PSPL_Phot_Par_GP_Param2,
                         use_phot_optional_params = False,
                         n_live_points=100,
                         outputfiles_basename=base,
                         resume=False, verbose=False)

    # Check that the GP parameters will be used.
    # Their presence in the fitter_param_names instance variable means
    # they will be fit. 
    assert 'gp_log_sigma1' not in fitter.fitter_param_names
    assert 'gp_log_rho1' not in fitter.fitter_param_names
    assert 'gp_log_S01' not in fitter.fitter_param_names
    assert 'gp_log_omega01' not in fitter.fitter_param_names
    assert 'gp_log_sigma2' not in fitter.fitter_param_names
    assert 'gp_log_rho2' not in fitter.fitter_param_names
    assert 'gp_log_S02' not in fitter.fitter_param_names
    assert 'gp_log_omega02' not in fitter.fitter_param_names
    
    return

def test_plot_model_and_data_GP_err():
    data = munge.getdata2('ob120169',
                          phot_data=['I_OGLE'],
                          ast_data=[])

    outdir = '/u/jlu/work/microlens/OB120169/a_2020_08_18/model_fits/201_phot_ogle_gp/base_a/'
    outbase = 'a1_'

    fitter = model_fitter.PSPL_Solver(data,
                                  model.PSPL_Phot_Par_GP_Param2,
                                  use_phot_optional_params=True,
                                  add_error_on_photometry=False,
                                  multiply_error_on_photometry=False,
                                  importance_nested_sampling = False,
                                  n_live_points = 1000,
                                  evidence_tolerance = 0.1,
                                  sampling_efficiency = 0.8,
                                  outputfiles_basename=outdir + outbase)

    print('testing')
    best_mod = fitter.get_best_fit_model(def_best='maxl')

    # Tweak the model
    best_mod.gp_log_rho[0] = -5.0
    
    # Get the data out.
    filt_index = 0
    print('gp_log_sigma1 = ', best_mod.gp_log_sigma[filt_index])
    print('gp_log_rho    = ', best_mod.gp_log_rho[filt_index])
    print('gp_log_S0     = ', best_mod.gp_log_S0[filt_index])
    print('gp_log_omega0 = ', best_mod.gp_log_omega0[filt_index])

    print('gp_sigma1 = ', 10**best_mod.gp_log_sigma[filt_index])
    print('gp_rho    = ', 10**best_mod.gp_log_rho[filt_index])
    print('gp_S0     = ', 10**best_mod.gp_log_S0[filt_index])
    print('gp_omega0 = ', 10**best_mod.gp_log_omega0[filt_index])
    
    dat_t = data['t_phot' + str(filt_index + 1)]
    dat_m = data['mag' + str(filt_index + 1)]
    dat_me = data['mag_err' + str(filt_index + 1)]
    mod_t = dat_t


    # Fix logQ following Golovich+20
    import celerite
    gp_log_Q = np.log(2**-0.5)
    
    matern = celerite.terms.Matern32Term(best_mod.gp_log_sigma[filt_index], best_mod.gp_log_rho[filt_index])
    sho = celerite.terms.SHOTerm(best_mod.gp_log_S0[filt_index], gp_log_Q, best_mod.gp_log_omega0[filt_index]) 
    kernel = matern + sho
    
    my_model = model.Celerite_GP_Model(best_mod, filt_index)  
    
    gp = celerite.GP(kernel, mean=my_model, fit_mean=True)
    gp.compute(dat_t, dat_me)
    mag_model, mag_model_var = gp.predict(dat_m, mod_t, return_var=True)
    print('dat_t = ', dat_t[0:5])
    print('dat_m = ')
    print(dat_m[0:5])
    print('dat_me = ')
    print(dat_me[0:5])
    
    print(mag_model[0:5], mag_model_var[0:5])
    pdb.set_trace()
    
    # Make models.
    # Decide if we sample the models at a denser time, or just the
    # same times as the measurements.
    mod_m_out, mod_m_out_std = best_mod.get_photometry_with_gp(dat_t, dat_m, dat_me, filt_index, mod_t)

    print(mod_m_out[0:5], mod_m_out_std[0:5])
    
    # fitter.plot_model_and_data(best_mod, suffix='_maxl')

    return

def test_make_default_priors_mag_base():
    """
    Test the new priors for mag_base calculated from the data.
    """

    # Test a single data set. 
    data = munge.getdata2('ob150211',
                          phot_data=['I_OGLE'],
                          ast_data = [])

    mag_base_mean_in = data['mag1'][0:100].mean()
    mag_base_std_in = data['mag1'][0:100].std()

    fitter = model_fitter.PSPL_Solver(data,
                                  model.PSPL_Phot_Par_GP_Param2_2,
                                  use_phot_optional_params=True,
                                  add_error_on_photometry=False,
                                  multiply_error_on_photometry=False,
                                  importance_nested_sampling = False,
                                  n_live_points = 1000,
                                  evidence_tolerance = 0.1,
                                  sampling_efficiency = 0.8, 
                                  outputfiles_basename='junk')

    # Generate a random sample from the prior and
    # check mean/std.
    mag_base_samp = fitter.priors['mag_base1'].rvs(size=100)

    mag_base_mean_out = mag_base_samp.mean()
    mag_base_std_out = mag_base_samp.std()

    assert mag_base_mean_in > (mag_base_mean_out - mag_base_std_out)
    assert mag_base_mean_in < (mag_base_mean_out + mag_base_std_out)

    # Test a multiple photometric data sets. 
    data = munge.getdata2('ob150211',
                          phot_data=['I_OGLE', 'Kp_Keck'],
                          ast_data = [])

    mag_base_mean_in1 = data['mag1'][0:100].mean()
    mag_base_std_in1 = data['mag1'][0:100].std()

    mag_base_mean_in2 = data['mag2'][-4:].mean()
    mag_base_std_in2 = data['mag2'][-4:].std()
    
    fitter = model_fitter.PSPL_Solver(data,
                                  model.PSPL_Phot_Par_GP_Param2_2,
                                  use_phot_optional_params=True,
                                  add_error_on_photometry=False,
                                  multiply_error_on_photometry=False,
                                  importance_nested_sampling = False,
                                  n_live_points = 1000,
                                  evidence_tolerance = 0.1,
                                  sampling_efficiency = 0.8, 
                                  outputfiles_basename='junk')

    # Generate a random sample from the prior and
    # check mean/std.
    mag_base_samp1 = fitter.priors['mag_base1'].rvs(size=100)
    mag_base_samp2 = fitter.priors['mag_base2'].rvs(size=100)

    mag_base_mean_out1 = mag_base_samp1.mean()
    mag_base_std_out1 = mag_base_samp1.std()

    mag_base_mean_out2 = mag_base_samp2.mean()
    mag_base_std_out2 = mag_base_samp2.std()
    
    assert mag_base_mean_in1 > (mag_base_mean_out1 - mag_base_std_out1)
    assert mag_base_mean_in1 < (mag_base_mean_out1 + mag_base_std_out1)

    assert mag_base_mean_in2 > (mag_base_mean_out2 - mag_base_std_out2)
    assert mag_base_mean_in2 < (mag_base_mean_out2 + mag_base_std_out2)

    
    return
    
