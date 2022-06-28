======================
Fitting Models to Data
======================

``bagle.model_fitter`` is a module that provides the infrastructure
needed to fit a model to data using Bayesian inference.
We utilize nested sampling to perform the inference (PyMultiNest);
however, much of the infrastructure can be used with other
inference packages or methods (e.g. Dynesty, emcee).

The key model fitting object is ``PSPL_Solver``, which is a
python object used to organize the data, define the likelihoods and
prior distributions, and call solvers.

First, to show an example of model fitting, we need to load up some
data. BAGLE provides some useful routines for generating fake data.
The routine below provide fake data and the input parameters used
to generate it::

  from bagle import fake_data
  data, p_in = fake_data.fake_data_parallax_lmc()

Now we can setup our ``PSPL_Solver`` and make a directory to save our
fit results. We will be using a ``PSPL_PhotAstrom_Par_Param1`` model
that will jointly fit both photometric and astrometric data with a
PSPL model with parallax, using the `Param1` parameterization::

  import os
  from bagle import model
  from bagle import model_fitter
  
  outdir = './test_mnest_lmc/'
  os.makedirs(outdir, exist_ok=True)


  fitter = PSPL_Solver(data,
                       model.PSPL_PhotAstrom_Par_Param1,
                       n_live_points=300,
                       outputfiles_basename=outdir + '/aa_',
                       resume=False)

We have specified that PyMultiNest will use 300 live points for
sampling posterior. See the :ref:`PyMultiNest documentation
<https://johannesbuchner.github.io/PyMultiNest/>` for details.

Next we need to define prior distributions for each of our input
parameters. All of them have default priors that are extremely wide;
however, it is best to tune these explicitly for your specific
problem. Below, we provide a very small prior range for each parameter
to speed up the fitting process just for a quick-running example::
                       
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

Now we can run the solver. This is likely to take more than 5-10 minutes
depending on your machine. Note, this will run
single-threaded. Examples of mutli-threaded fitting are further
down. You should see a stream of output from PyMultiNest that allows
you to track how the fit is converging.::
  
  fitter.solve()

Once the fit is complete, we can get the list of best-fit parameters
and generate the best-fit output model::
  
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

For comparison, we will also make a model with the input parameters::
  
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

For convenience in plotting, we will save some derived values to our
input parameter array::                                              

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

  
The ``PSPL_Solver`` has some convenience functions for summarizing the
results::
  
  fitter.summarize_results()

We can plot the resulting model posteriors, trace plots, corner plots,
etc. using the Dynesty package's very nice plotting facility (note you
will need to install Dynesty to get this to work)::
  
  fitter.plot_dynesty_style(sim_vals=p_in)

We can plot the model and data against each other. We can also
optionally overplot the input model as well.::
  
  fitter.plot_model_and_data(pspl_out, input_model=pspl_in)

For testing, we can double check that the output model agrees with the
input model within some tolerance.::

  imag_out = pspl_out.get_photometry(data['t_phot1'])
  pos_out = pspl_out.get_astrometry(data['t_ast1'])

  imag_in = pspl_in.get_photometry(data['t_phot1'])
  pos_in = pspl_in.get_astrometry(data['t_ast1'])

  np.testing.assert_array_almost_equal(imag_out, imag_in, 1)
  np.testing.assert_array_almost_equal(pos_out, pos_in, 4)



.. toctree::
   :maxdepth: 3
  
   PSPL_Solver.rst
   Prior_Generator.rst
   Gen_Use_model_fitter.rst
