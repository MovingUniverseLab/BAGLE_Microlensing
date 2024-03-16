import numpy as np
import pylab as plt
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from bagle import model, model_fitter, data
import dynesty
from dynesty import utils as dyutil
from dynesty import plotting as dyplot
from astropy.table import Table
from astropy.time import Time
from astropy import units 
from astropy.coordinates import SkyCoord
from multiprocessing import Pool, cpu_count
import time
import pickle
import pdb
import os, shutil
import yaml
from scipy import spatial
from scipy.ndimage import gaussian_filter as norm_kde
from scipy.stats import gaussian_kde
import glob
import astropy.constants as const

def check_priors(fitter, target, posterior=True):
    # This is specifically for photometry + astrometry.

    names = fitter.fitter_param_names
    all_names = fitter.all_param_names

    #####
    # Priors
    #####
    # Number of draws from posterior.
    Nsamp = 5000

    # Store the prior distributions for each parameter.
    priors_dict = {}

    # First, insert all the independent priors.
    for i in range(len(names)):
        priors_dict[names[i]] = fitter.priors[names[i]].rvs(Nsamp)

    # Next, calculate all the derived priors from the independent priors.
    if 'log10_thetaE' in names:
        priors_dict['thetaE_amp'] = 10**priors_dict['log10_thetaE']
    else:
        priors_dict['thetaE_amp'] = priors_dict['thetaE']
    priors_dict['piE'] = np.hypot(priors_dict['piE_E'], priors_dict['piE_N'])
    priors_dict['piRel'] = priors_dict['piE'] * priors_dict['thetaE_amp']
    kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
    kappa = kappa_tmp.to(units.mas / units.Msun,
                         equivalencies=units.dimensionless_angles()).value
    priors_dict['mL'] = priors_dict['thetaE_amp']**2 / (priors_dict['piRel'] * kappa)

    priors_dict['piL'] = priors_dict['piRel'] + priors_dict['piS']

    priors_dict['muRel'] = priors_dict['thetaE_amp']/(priors_dict['tE']/365.25)
    priors_dict['muRel_E'] = priors_dict['muRel'] * (priors_dict['piE_E']/priors_dict['piE'])
    priors_dict['muRel_N'] = priors_dict['muRel'] * (priors_dict['piE_N']/priors_dict['piE'])
    priors_dict['muL_E'] = priors_dict['muS_E'] - priors_dict['muRel_E']
    priors_dict['muL_N'] = priors_dict['muS_N'] - priors_dict['muRel_N']

    #####
    # Posteriors
    #####
    res = fitter.load_mnest_results(remake_fits=False)

    #####
    # Plot posteriors and priors together for comparison.
    #####
    fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(16, 16))
    fax = ax.flatten()
    fig.subplots_adjust(top = 0.95, bottom = 0.05, left=0.05, right=0.95, hspace = 0.5, wspace = 0.3)
    plt.suptitle(target)
    for ii, name in enumerate(all_names):
        # Get the 3-sigma upper and lower bounds for the priors and the posteriors,
        # so we don't have crazy tails that make the x-range of the plot unreadable.
        sig3_hi = 0.9973
        sig3_lo = 1.0 - sig3_hi
        
        quantiles = [sig3_lo, sig3_hi]
        
        quants_prior = model_fitter.weighted_quantile(priors_dict[name], quantiles)
        quants_post = model_fitter.weighted_quantile(res[name], quantiles,
                                                     sample_weight=res['weights'])

        _min = min([quants_prior[0], quants_post[0]])
        _max = max([quants_prior[1], quants_post[1]])

        bins = np.linspace(_min, _max, 30)

        fax[ii].hist(priors_dict[name], bins=bins,
                    density=True, histtype='step', lw=2, color='blue',
                    label='Prior')       
        fax[ii].hist(res[name], weights=res['weights'], bins=bins,
                     density=True, histtype='step', lw=2, color='red',
                     label='Posterior')
        fax[ii].yaxis.set_ticks([])
        fax[ii].yaxis.set_ticklabels([])
        fax[ii].set_ylabel(name)

    fax[0].legend()
        
    if posterior:
        plt.savefig(target + '_posteriors_priors.png')
    else:
        plt.savefig(target + '_priors.png')


def check_priors_phot(fitter, target, posterior=True):
    # This is specifically for photometry + astrometry.

    names = fitter.fitter_param_names
    all_names = fitter.all_param_names

    #####
    # Priors
    #####
    # Number of draws from posterior.
    Nsamp = 5000

    # Store the prior distributions for each parameter.
    priors_dict = {}

    # First, insert all the independent priors.
    for i in range(len(names)):
        priors_dict[names[i]] = fitter.priors[names[i]].rvs(Nsamp)

    #####
    # Posteriors
    #####
    res = fitter.load_mnest_results(remake_fits=False)

    #####
    # Plot posteriors and priors together for comparison.
    #####
    fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(16, 16))
    fax = ax.flatten()
    fig.subplots_adjust(top = 0.95, bottom = 0.05, left=0.05, right=0.95, hspace = 0.5, wspace = 0.3)
    plt.suptitle(target)
    for ii, name in enumerate(names):
        # Get the 3-sigma upper and lower bounds for the priors and the posteriors,
        # so we don't have crazy tails that make the x-range of the plot unreadable.
        sig3_hi = 0.9973
        sig3_lo = 1.0 - sig3_hi
        
        quantiles = [sig3_lo, sig3_hi]
        
        quants_prior = model_fitter.weighted_quantile(priors_dict[name], quantiles)
        quants_post = model_fitter.weighted_quantile(res[name], quantiles,
                                                     sample_weight=res['weights'])

        _min = min([quants_prior[0], quants_post[0]])
        _max = max([quants_prior[1], quants_post[1]])

        bins = np.linspace(_min, _max, 30)

        fax[ii].hist(priors_dict[name], bins=bins,
                    density=True, histtype='step', lw=2, color='blue',
                    label='Prior')       
        fax[ii].hist(res[name], weights=res['weights'], bins=bins,
                     density=True, histtype='step', lw=2, color='red',
                     label='Posterior')
        fax[ii].yaxis.set_ticks([])
        fax[ii].yaxis.set_ticklabels([])
        fax[ii].set_ylabel(name)

    fax[0].legend()
        
    if posterior:
        plt.savefig(target + '_posteriors_priors.png')
    else:
        plt.savefig(target + '_priors.png')

    return


def check_priors_PSPL_Astrom_Par_Param4(fitter, params_from_post, target, posterior=False):
    """
    Param2, Param4
    """
    Nsamp = 5000
    param_dict = {}
        
    names = fitter.fitter_param_names
    all_names = fitter.all_param_names
    fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(12, 12))
    fax = ax.flatten()
    fig.subplots_adjust(top = 0.95, bottom = 0.05, left=0.05, right=0.95, hspace = 0.5, wspace = 0.3)
    plt.suptitle(target)
    for i in range(len(names)):
        if names[i] in params_from_post:
            continue
        else:
            param_dict[names[i]] = fitter.priors[names[i]].rvs(Nsamp)
            fax[i].hist(fitter.priors[names[i]].rvs(Nsamp), bins=30,
                       density=True, histtype='step', lw=2, color='blue')
            fax[i].set_ylabel(names[i])
            
    # Posterior --> Prior
    binmids = []
    for bb in np.arange(len(fitter.post_param_bins)):
        binmids.append((fitter.post_param_bins[bb][:-1] + fitter.post_param_bins[bb][1:])/2)

    draws = np.zeros((len(params_from_post), Nsamp))
    for ii in np.arange(Nsamp):
        draws[:,ii] = fitter.sample_post(binmids, fitter.post_param_cdf, fitter.post_param_bininds)
        
    for i in np.arange(len(params_from_post)):
        param_dict[params_from_post[i]] = draws[i]
        idx = names.index(params_from_post[i])
        fax[idx].hist(draws[i], bins=30,
                     density=True, histtype='step', lw=2, color='blue',
                     label='Prior')
        fax[idx].set_ylabel(params_from_post[i])

    # Calculate piL, muS, muRel also
    piE = np.hypot(param_dict['piE_E'], param_dict['piE_N'])
    piRel = piE * param_dict['thetaE']
    kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
    kappa = kappa_tmp.to(units.mas / units.Msun,
                         equivalencies=units.dimensionless_angles()).value
    mL = param_dict['thetaE'] ** 2 / (piRel * kappa)

    piL = piRel + param_dict['piS']

    muRel = param_dict['thetaE']/(param_dict['tE']/365.25)
    muRel_E = muRel * (param_dict['piE_E']/piE)
    muRel_N = muRel * (param_dict['piE_N']/piE)
    muL_E = param_dict['muS_E'] - muRel_E
    muL_N = param_dict['muS_N'] - muRel_N
    
    fax[len(names)].hist(mL, bins=30,
                        density=True, histtype='step', lw=2, color='blue')
    fax[len(names)].set_ylabel('mL')
        
    fax[len(names)+1].hist(piL, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+1].set_ylabel('piL')

    fax[len(names)+2].hist(piRel, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+2].set_ylabel('piRel')

    fax[len(names)+3].hist(muL_E, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+3].set_ylabel('muL_E')

    fax[len(names)+4].hist(muL_N, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+4].set_ylabel('muL_N')
    
    fax[len(names)+5].hist(muRel_E, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+5].set_ylabel('muRel_E')

    fax[len(names)+6].hist(muRel_N, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+6].set_ylabel('muRel_N')

    if posterior:
        res = fitter.load_mnest_results(remake_fits=False)
        for ii, name in enumerate(all_names):
            fax[ii].hist(res[name], weights=res['weights'], bins=30,
                        density=True, histtype='step', lw=2, color='red',
                        label='Posterior')
            fax[ii].yaxis.set_ticks([])
            fax[ii].yaxis.set_ticklabels([])
    fax[0].legend()
        
#    ax[len(names)].hist(piE, bins=30)
#    ax[len(names)].set_ylabel('piE')

    if posterior:
        plt.savefig(target + '_posteriors_priors.png')
    else:
        plt.savefig(target + '_priors.png')


def check_priors_PSPL_Astrom_Par_Param5(fitter, params_from_post, target, posterior=False):
    """
    Param2, Param4
    """
    Nsamp = 5000
    param_dict = {}
        
    names = fitter.fitter_param_names
    all_names = fitter.all_param_names
    fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(12, 12))
    fax = ax.flatten()
    fig.subplots_adjust(top = 0.95, bottom = 0.05, left=0.05, right=0.95, hspace = 0.5, wspace = 0.3)
    plt.suptitle(target)
    for i in range(len(names)):
        if names[i] in params_from_post:
            continue
        else:
            param_dict[names[i]] = fitter.priors[names[i]].rvs(Nsamp)
            fax[i].hist(fitter.priors[names[i]].rvs(Nsamp), bins=30,
                       density=True, histtype='step', lw=2, color='blue')
            fax[i].set_ylabel(names[i])
            
    # Posterior --> Prior
    binmids = []
    for bb in np.arange(len(fitter.post_param_bins)):
        binmids.append((fitter.post_param_bins[bb][:-1] + fitter.post_param_bins[bb][1:])/2)

    draws = np.zeros((len(params_from_post), Nsamp))
    for ii in np.arange(Nsamp):
        draws[:,ii] = fitter.sample_post(binmids, fitter.post_param_cdf, fitter.post_param_bininds)
        
    for i in np.arange(len(params_from_post)):
        param_dict[params_from_post[i]] = draws[i]
        idx = names.index(params_from_post[i])
        fax[idx].hist(draws[i], bins=30,
                     density=True, histtype='step', lw=2, color='blue',
                     label='Prior')
        fax[idx].set_ylabel(params_from_post[i])

    # Calculate piL, muS, muRel also
    piE = np.hypot(param_dict['piE_E'], param_dict['piE_N'])
    piRel = piE * 10**param_dict['log10_thetaE']
    kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
    kappa = kappa_tmp.to(units.mas / units.Msun,
                         equivalencies=units.dimensionless_angles()).value
    mL = (10**param_dict['log10_thetaE']) ** 2 / (piRel * kappa)

    piL = piRel + param_dict['piS']

    muRel = 10**param_dict['log10_thetaE']/(param_dict['tE']/365.25)
    muRel_E = muRel * (param_dict['piE_E']/piE)
    muRel_N = muRel * (param_dict['piE_N']/piE)
    muL_E = param_dict['muS_E'] - muRel_E
    muL_N = param_dict['muS_N'] - muRel_N
    
    fax[len(names)].hist(mL, bins=30,
                        density=True, histtype='step', lw=2, color='blue')
    fax[len(names)].set_ylabel('mL')
        
    fax[len(names)+1].hist(piL, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+1].set_ylabel('piL')

    fax[len(names)+2].hist(piRel, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+2].set_ylabel('piRel')

    fax[len(names)+3].hist(muL_E, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+3].set_ylabel('muL_E')

    fax[len(names)+4].hist(muL_N, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+4].set_ylabel('muL_N')
    
    fax[len(names)+5].hist(muRel_E, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+5].set_ylabel('muRel_E')

    fax[len(names)+6].hist(muRel_N, bins=30,
                          density=True, histtype='step', lw=2, color='blue')
    fax[len(names)+6].set_ylabel('muRel_N')

    if posterior:
        res = fitter.load_mnest_results(remake_fits=False)
        for ii, name in enumerate(all_names):
            fax[ii].hist(res[name], weights=res['weights'], bins=30,
                        density=True, histtype='step', lw=2, color='red',
                        label='Posterior')
            fax[ii].yaxis.set_ticks([])
            fax[ii].yaxis.set_ticklabels([])
    fax[0].legend()
        
#    ax[len(names)].hist(piE, bins=30)
#    ax[len(names)].set_ylabel('piE')

    if posterior:
        plt.savefig(target + '_posteriors_priors.png')
    else:
        plt.savefig(target + '_priors.png')

def MultiDimPrior_Gen(tab, pnames, Nbins, binlims=None, plot_1D=False):
    """
    Takes multinest output and histograms them. 
    Returns the histogram bins, along with an array of the 
    PDF, CDF, and indices of the non-zero histogram bins.

    Note: the number of bins Nbins needs to be balanced 
    against the number of parameters, otherwise python
    will crash. Because the number of bins goes as 
    (number of bins)^(number of parameters), small changes to
    either number can cause the program to fail where it didn't 
    before. For example, 40 bins with 5 parameters works, but
    40 bins with 7 parameters will fail.

    Parameters
    ----------
    tab : multinest results 
        output of model_fitter.load_mnest_results

    pnames : list of strings
        names of the parameters

    Nbins : int
        number of bins

    binlims : list of length pnames, each entry is list of length 2
        Limits of  bins. The nth entry corresponds to 
        [lower limit, upper limit] for the nth parameter in pnames.
        If None, limits will be inferred from multinest.

    Return
    ------
    bins : list of lists
        bin edges

    savearr : array of shape (N nonzero histogram bins, N parameters + 2)
        columns are PDF, CDF, parameter1 index, ..., parameterN index
        where indices are multi-dimensional bin indices

    """
    Nrows = len(tab)
    Ncols = len(pnames)
    print(Nrows)

    # Organize the multinest output into a table.
    # Shape of intable is (Nsamples, Nparams)
    intable = np.zeros((Nrows, Ncols))
    for ii, pname in enumerate(pnames):
        intable[:,ii] = tab[pname].data

    # Set bin-edges for each column in the table.
    # Bins is a list of length Nparams, and each entry in the list is an array of length Nbins
    bins = []
    if binlims is None:
        for i in range(Ncols):
            bins.append(np.linspace(intable[:,i].min(), intable[:,i].max(), Nbins))
    else:
        for i in range(Ncols):
            lo = binlims[i][0]
            hi = binlims[i][1]
            bins.append(np.linspace(lo, hi, Nbins))

    # Setup the multidimensional histogram
    # H shape is an array (Nbins-1, Nbins-1, ... , Nbins-1) where the length is the number of dimensions
    H, edges = np.histogramdd(intable, bins=bins, weights=tab['weights'])
    Hnorm = H/np.sum(H)
    print(np.sum(H))
    # Indices where histogram entries != 0
    indpdf = np.transpose(np.array((Hnorm).nonzero()))

    # Checking binning weirdness:
    # 1. Rounding for .nonzero() not problem. Using density=False, i.e.
    # H, edges = np.histogramdd(intable, bins=bins, density=False) 
    # gives the same results for number of nonzero bins.
    # 2. Turning KDE off in dynesty plotting and comparing. Not sure
    # why, plots looked exactly the same...
    # 3. Turns out, you have to be very precise in selecting bin edges.
    # If they are too wide, you get the weird random gaps. IDK why...
    # Use plot_1D=True to visualize everything first and make sure
    # it is all okay.
    Nrows = len(indpdf[:,0])
    cdf = np.zeros(Nrows, dtype='float')
    pdf = np.zeros(Nrows, dtype='float')

    # Calculate 1D cdf and pdf using only non-zero bins.
    for j in range(Nrows):
        cdf[j] = cdf[j-1] + Hnorm[tuple(indpdf[j,:])]
        pdf[j] = Hnorm[tuple(indpdf[j,:])]

    # Stack PDF, CDF, and non-zero histogram indices into a single table.
    savearr = indpdf[:,0].astype(int)
    for i in range(Ncols-1):savearr = np.column_stack((savearr.astype(int), indpdf[:,i+1]))
    savearr = np.column_stack((pdf, cdf, savearr))  

    # Plot 1-D histograms compared to binned to make sure it all works out.
    if plot_1D:
        for ii in np.arange(Ncols):
            plt.figure(ii+1)
            hist, binedges = np.histogram(savearr[:,ii+2], weights=savearr[:,0], bins=Nbins-1, density=True)
            f = ((Nbins-1)/Nbins)/(bins[ii][1] - bins[ii][0])
            plt.step(bins[ii][:-1], hist * f, where='post', lw=2, ls='-', label='N-D hist', zorder=1)
            plt.hist(intable[:, ii], weights=tab['weights'], bins=bins[ii], lw=2, ls=':', 
                     label='1-D hist', histtype='step', density=True, zorder=2)
            plt.axvline(binlims[ii][0])
            plt.axvline(binlims[ii][1])
            plt.title(pnames[ii])
            plt.legend()
            plt.savefig(pnames[ii] + '.png')

    return bins, savearr


def onedee_posterior(tab, colname, log = False, nbin = 50):
    # Make a histogram of the mass using the weights. This creates the    
    # marginalized 1D posteriors.                                                                      

    fontsize1 = 18
    fontsize2 = 14

    xmax = tab[colname].max() 
    xmin = tab[colname].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(left=0.15)
    if log is True:
        xliml = 0.9 * xmin
        xlimu = 1.1 * xmax
        bins = np.logspace(np.log10(xliml), np.log10(xlimu), nbin)

    if log is False:
        xliml = xmin - 0.1 * np.abs(xmin)
        xlimu = xmax + 0.1 * np.abs(xmax)
        bins = np.linspace(xliml, xlimu, nbin)

        
    n, foo, patch = plt.hist(tab[colname], normed=True,
                             histtype='step', weights=tab['weights'],
                             bins=bins, linewidth=1.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.xlabel(colname)
    plt.xlabel(xtitle, fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, xlimu)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

    if log is True:
        plt.xscale('log')

    ##########                                                                                                              
    # Calculate 3-sigma boundaries for mass limits.                                                                         
    ##########                                                                                                              
    sig1_hi = 0.682689
    sig1_lo = 1.0 - sig1_hi
    sig_med = 0.5
    sig2_hi = 0.9545
    sig2_lo = 1.0 - sig2_hi
    sig3_hi = 0.9973
    sig3_lo = 1.0 - sig3_hi

    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi]

    mass_quants = model_fitter.weighted_quantile(tab[colname], quantiles,
                                                 sample_weight=tab['weights'])

    for qq in range(len(quantiles)):
        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))

    ax = plt.axis()
    plt.axvline(mass_quants[0], color='red', linestyle='--')
#     plt.text(mass_quants[0] + 0.05, 0.9*ax[3],
#              colname + '>{0:5.2f} with 99.7% confidence'.format(mass_quants[0]), fontsize=12)
    plt.axvline(mass_quants[3], color='black', linestyle='--')
#     plt.text(mass_quants[3] + 0.05, 0.8*ax[3],
#              colname + 'median = {0:5.2f}'.format(mass_quants[0]), fontsize=12)
    plt.axvline(mass_quants[-1], color='green', linestyle='--')
#     plt.text(mass_quants[-1] + 0.05, 0.7*ax[3],
#              colname + '<{0:5.2f} with 99.7% confidence'.format(mass_quants[-1]), fontsize=12)
# 
    plt.show()

    ##########                                                                                                           
    # Save figure                                                                                                     
    ##########                                                                                                            
    
#    outfile =  outdir + outfile
#    fileUtil.mkdir(outdir)
#    print( 'writing plot to file ' + outfile)
#
#    plt.savefig(outfile)

    return

def onedee_posterior_smooth(tab, colname, log = False, nbin = 500):
    # Make a histogram of the mass using the weights. This creates the    
    # marginalized 1D posteriors.                                                                      

    fontsize1 = 18
    fontsize2 = 14

    xmax = tab[colname].max() 
    xmin = tab[colname].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(left=0.23)
    if log is True:
        xliml = 0.9 * xmin
        xlimu = 1.1 * xmax
        bins = np.logspace(np.log10(xliml), np.log10(xlimu), nbin)

    if log is False:
        xliml = xmin - 0.1 * np.abs(xmin)
        xlimu = xmax + 0.1 * np.abs(xmax)
        bins = np.linspace(xliml, xlimu, nbin)

    n, b = np.histogram(tab[colname], bins = bins, 
                        weights = tab['weights'], normed = True)

    n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    n, b, _ = plt.hist(b0, bins = b, weights = n)

    plt.xlabel(colname)
    xtitle = colname 
    plt.xlabel(xtitle, fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, xlimu)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

    if log is True:
        plt.xscale('log')

    ##########                                                                                                              
    # Calculate 3-sigma boundaries for mass limits.                                                                         
    ##########                                                                                                              
    sig1_hi = 0.682689
    sig1_lo = 1.0 - sig1_hi
    sig_med = 0.5
    sig2_hi = 0.9545
    sig2_lo = 1.0 - sig2_hi
    sig3_hi = 0.9973
    sig3_lo = 1.0 - sig3_hi

    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi] 

    mass_quants = model_fitter.weighted_quantile(tab[colname], quantiles,
                                                 sample_weight=tab['weights'])

    for qq in range(len(quantiles)):
        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))

    ax = plt.axis()
    # plot median and +/- 3 sigma
    plt.axvline(mass_quants[0], color='k', linestyle='--')
    plt.axvline(mass_quants[3], color='k', linestyle='--')
    plt.axvline(mass_quants[-1], color='k', linestyle='--')
    plt.show()

    ##########                                                                                                           
    # Save figure                                                                                                     
    ##########                                                                                                            
    
#    outfile =  outdir + outfile
#    fileUtil.mkdir(outdir)
#    print( 'writing plot to file ' + outfile)
#
#    plt.savefig(outfile)

    return

def twodee_posterior_points(tab, colname1, colname2, logx = False, logy = False):
    """
    Color of points = weight
    """
    fig = plt.figure(1, figsize=(8,6))
    plt.subplots_adjust(left = 0.15)
    plt.clf()
    plt.scatter(tab[colname1], tab[colname2], c = tab['weights'],
                alpha = 0.3, s = 1, cmap = 'viridis')
    plt.xlabel(colname1)
    plt.ylabel(colname2)
    plt.colorbar()
    
    if logx == True:
        plt.xscale('log')
    
    if logy == True:
        plt.yscale('log')

    plt.show()
    
def twodee_posterior_smooth(tab, colname1, colname2, logx = False, logy = False,
                            xnbin = 500, ynbin = 500,
                            xsval = 2, ysval = 2,
                            span=None, weights=None, levels=None,
                            color='gray', plot_datapoints=False, plot_density=True,
                            plot_contours=True, no_fill_contours=False, fill_contours=True,
                            contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
                            **kwargs):
    try:
        str_type = types.StringTypes
        float_type = types.FloatType
        int_type = types.IntType
    except:
        str_type = str
        float_type = float
        int_type = int
        
    """
    Basically the _hist2d function from dynesty, but with a few mods I made.
    https://github.com/joshspeagle/dynesty/blob/master/dynesty/plotting.py

    Parameters
    ----------
    tab : astropy table
        Read in output from load_mnest_results_to_tab

    colname1, 2 : str
        The stuff you wanna plot. Column names from the tab

    x, ysval : int (float?)
        smoothing value. Kinda inverse of xnbin and ynbin I think

    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval).

    weights : iterable with shape (nsamps,)
        Weights associated with the samples. Default is `None` (no weights).

    levels : iterable, optional
        The contour levels to draw. Default are `[0.5, 1, 1.5, 2]`-sigma.

    ax : `~matplotlib.axes.Axes`, optional
        An `~matplotlib.axes.axes` instance on which to add the 2-D histogram.
        If not provided, a figure will be generated.

    color : str, optional
        The `~matplotlib`-style color used to draw lines and color cells
        and contours. Default is `'gray'`.

    plot_datapoints : bool, optional
        Whether to plot the individual data points. Default is `False`.

    plot_density : bool, optional
        Whether to draw the density colormap. Default is `True`.

    plot_contours : bool, optional
        Whether to draw the contours. Default is `True`.

    no_fill_contours : bool, optional
        Whether to add absolutely no filling to the contours. This differs
        from `fill_contours=False`, which still adds a white fill at the
        densest points. Default is `False`.

    fill_contours : bool, optional
        Whether to fill the contours. Default is `True`.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    """

    x = tab[colname1]
    y = tab[colname2]
    weights = tab['weights']

    fig = plt.figure(1)
    plt.clf()
    ax = plt.gca()

    # Determine plotting bounds.
    data = [x, y]
    if span is None:
        span = [0.999999426697 for i in range(2)]
    span = list(span)
    if len(span) != 2:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = dyutil.quantile(data[i], q, weights=weights)

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # Color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # Color map used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)
 
    xmax = tab[colname1].max() 
    xmin = tab[colname1].min()

    ymax = tab[colname2].max() 
    ymin = tab[colname2].min()

    if logx is True:
        xliml = 0.9 * xmin
        xlimu = 1.1 * xmax
        xbins = np.logspace(np.log10(xliml), np.log10(xlimu), xnbin)

    if logx is False:
        xliml = xmin - 0.1 * np.abs(xmin)
        xlimu = xmax + 0.1 * np.abs(xmax)
        xbins = np.linspace(xliml, xlimu, xnbin)

    if logy is True:
        yliml = 0.9 * ymin
        ylimu = 1.1 * ymax
        ybins = np.logspace(np.log10(yliml), np.log10(ylimu), ynbin)

    if logy is False:
        yliml = ymin - 0.1 * np.abs(ymin)
        ylimu = ymax + 0.1 * np.abs(ymax)
        ybins = np.linspace(yliml, ylimu, ynbin)


    # We'll make the 2D histogram to directly estimate the density.
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=[xbins, ybins],
                             range=list(map(np.sort, span)),
                             weights=weights)
    # Smooth the results.
    H = norm_kde(H, [xsval, ysval])
 
    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        logging.warning("Too few points to create valid contours.")
        logging.warning("Make xnbin or ynbin bigger!!!!!!!")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    # Plot the data points.
    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlabel(colname1)
    ax.set_ylabel(colname2)

    ax.set_xlim(span[0])
    ax.set_ylim(span[1])

    if logx is True:
        ax.set_xscale('log')

    if logy is True:
        ax.set_yscale('log')

    plt.show()

    return

def separate_mode_files(mnest_dir = './', mnest_root = 'aa_'):
    # FIXME: Is there a more intelligent way to deal with all the indices???
    # But it seems to work for now...
    mode_file = mnest_dir + mnest_root + 'post_separate.dat'
    
    empty_lines = []
    with open(mode_file, 'r') as orig_file:
        for num, line in enumerate(orig_file, start=0):
            if line == '\n':
                empty_lines.append(num)

    # Error checking
    if len(empty_lines) % 2 != 0:
        print('SOMETHING BAD HAPPENED!')

    idx_range = int(len(empty_lines)/2)
    
    orig_tab = np.loadtxt(mode_file)
    for idx in np.arange(idx_range):
        start_idx = empty_lines[idx * 2 + 1] + 1 - 2 * (idx + 1)
        if idx != np.arange(idx_range)[-1]:
            end_idx = empty_lines[idx * 2 + 2] - 2 * (idx + 1)
            np.savetxt(mnest_dir + mnest_root + 'mode' + str(idx) + '.dat', orig_tab[start_idx:end_idx])  
        else:
            np.savetxt(mnest_dir + mnest_root + 'mode' + str(idx) + '.dat', orig_tab[start_idx:])  

    return

def convert_gp_names(base):
    """
    Fix pre-April 1 2021 gp parameter names to match new names (commit de12879)
    Read in old fits file, and rename columns
    """
    # Gather all the .fits files that need to have columns renamed
    fits_files = glob.glob(base + '*.fits')

    fits_files = [x for x in fits_files if 'old' not in x]

    # Loop through and fix.
    for ff in fits_files:
        tab = Table.read(ff)
        for colname in tab.colnames:
            if 'omegaofour_So' in colname:
                tab.rename_column(colname, colname.replace('omegaofour_So', 'omega04_S0'))
            elif 'So' in colname:
                tab.rename_column(colname, colname.replace('So', 'S0'))
            elif 'omegaofour' in colname:
                tab.rename_column(colname, colname.replace('omegaofour', 'omega04'))
            elif 'omegao' in colname:
                tab.rename_column(colname, colname.replace('omegao', 'omega0'))

        # Move the original file by renaming it "old".
        os.system('mv ' + ff + ' ' + ff[:-5] + '_old.fits')
        
        try:
            tab.write(ff, overwrite=False)
        except:
            print('Exists already, skipping')

def convert_pre_2020apr_u0_sign(old_base, new_base):
    """
    Read in Multinest output from a pre 2020 April u0 sign
    convention and convert everything into a new convention
    (same as Gould+ 2004). 
    """
    # Fetch the model class out of the YAML file.
    yaml_file = open(old_base + 'params.yaml', 'r')
    yaml_params = yaml.full_load(yaml_file)
    with open(new_base + 'params.yaml', 'w') as f:
        foo = yaml.dump(yaml_params, f)

    # Get the parameter names off the model class.
    model_class = getattr(model, yaml_params['model'])
    params = model_class.all_param_names
        
    # Find the u0_amp or beta parameters that need to be changed.
    # There could be multiples (some fit paras, others not). 
    u0_cols = []
    for pp in range(len(params)):
        if ((params[pp] == 'u0_amp') or
            (params[pp] == 'beta')):
            u0_cols.append(pp)
    
    # # Fix the <base>.txt file
    # tab = Table.read(old_base + '.txt', format='ascii')
    # for cc in u0_cols:
    #     tab['col{0:d}'.format(cc+1+2)] *= -1.0
    # tab.write(new_base + '.txt', format='ascii.no_header')

    # # Fix the <base>summary.txt file
    # tab_a = Table.read(old_base + 'summary.txt', format='ascii')
    # for cc in u0_cols:
    #     mean_col = 0 * len(params) + 1 + cc
    #     std_col = 1 * len(params) + 1 + cc
    #     maxL_col = 2 * len(params) + 1 + cc
    #     maxAP_col = 3 * len(params) + 1 + cc

    #     tab_a['col{0:d}'.format(mean_col)] *= -1.0
    #     tab_a['col{0:d}'.format(maxL_col)] *= -1.0
    #     tab_a['col{0:d}'.format(maxAP_col)] *= -1.0
    # tab_a.write(new_base + 'summary.txt', format='ascii.no_header')

    # # Fix the *ev.dat file.
    # tab_e = Table.read(old_base + 'ev.dat', format='ascii')
    # for cc in u0_cols:
    #     tab_e['col{0:d}'.format(cc+1)] *= -1.0
    # tab_e.write(new_base + 'ev.dat', format='ascii.no_header')

    # # Fix the *phys_live.points file
    # tab_l = Table.read(old_base + 'phys_live.points', format='ascii')
    # for cc in u0_cols:
    #     tab_l['col{0:d}'.format(cc+1)] *= -1.0
    # tab_l.write(new_base + 'phys_live.points', format='ascii.no_header')

    # # Fix *post_equal_weights.dat file.
    # tab_pe = Table.read(old_base + 'post_equal_weights.dat', format='ascii')
    # for cc in u0_cols:
    #     tab_pe['col{0:d}'.format(cc+1)] *= -1.0
    # tab_pe.write(new_base + 'post_equal_weights.dat', format='ascii.no_header')

    # Fix *post_separate.dat
    _psep_old = open(old_base + 'post_separate.dat', 'r')
    _psep_new = open(new_base + 'post_separate.dat', 'w')
    for line in _psep_old:
        sline = line.strip()
        if sline == '':
            _psep_new.write(line)
        else:
            # Found a sample row.
            fields = sline.split()

            # Loop through u0 cols
            for cc in u0_cols:
                if fields[cc+2].startswith('-'):
                    fields[cc+2] = fields[cc+2][1:]
                else:
                    fields[cc+2] = '-' + fields[cc+2]
                    
            _psep_new.write('  '.join(fields) + '\n')
            
    _psep_old.close()
    _psep_new.close()

    
    # # Fix *stats.dat
    # _psta_old = open(old_base + 'stats.dat', 'r')
    # _psta_new = open(new_base + 'stats.dat', 'w')
    # for line in _psta_old:
    #     sline = line.strip()

    #     for cc in u0_cols:
    #         # Looking for a match like '3 ' or '13 ':
    #         if sline.startswith(str(cc + 1) + ' '):
    #             fields = sline.split()
    #             u0_val_str = fields[1]

    #             # Determine new "sign" string
    #             if u0_val_str.startswith('-'):
    #                 new_sign_str = ' '
    #                 u0_abs_str = u0_val_str[1:]
    #             else:
    #                 new_sign_str = '-'
    #                 u0_abs_str = u0_val_str

    #             # Replace the sign character.
    #             ldx = line.find(u0_abs_str)
    #             line = line[:ldx-1] + new_sign_str + line[ldx:]
            
    #     _psta_new.write(line)
        
    # _psta_old.close()
    # _psta_new.close()
    

    # # Fix *resume.dat
    # shutil.copyfile(old_base + 'resume.dat', new_base + 'resume.dat')

    # Fix *live.points
    # PASS... I don't know what is in this file.
    
    return

