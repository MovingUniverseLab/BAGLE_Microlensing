import numpy as np
import pylab as plt
from bagle import model
import copy

def fisher_matrix(t, merr,
                  model_class, params, params_fixed,
                  num_deriv_frac=0.01):
    """
    Calculate the fisher matrix for an arbitrary BAGLE microlens model.
    The order of the parameters in the fisher matrix will be that
    of params.keys().

    Parameters
    ----------
    t : array-like
        Array of times at which observations are sampled.
    merr : array-like
        Array of magnitude uncertainties at each observation time.
    model_class : BAGLE model object
        A BAGLE model object with parameters as listed in the params variable.
    params : dict
        Dictionary of model parameters. These are
        the parameters where the uncertainties are evaluated.
    params_fixed : dict
        Dictionary of fixed model parameters. Usually this includes
        raL and decL for the R.A. and Dec of the lens. 

    Optional
    --------
    num_deriv_frac : float
        Fisher matrix is calculated with a numerical derivative. This
        sets the step size used to calculate the numerical derivative.
        This must be carefully tuned for high-magnification events or for
        binary events. 
    """
    mod_par = model.get_model(model_class, params, params_fixed)
    
    # Calculate the derivatives numerically.
    n_params = len(params)

    derivs = {}
    
    for i in params.keys():
        # Estimate the grid step for calculating the derivative.
        dp = params[i] * num_deriv_frac   # Step size for differentiation will be 1%
        
        params_lo = copy.deepcopy(params)
        params_hi = copy.deepcopy(params)

        params_lo[i] = params[i] - dp
        params_hi[i] = params[i] + dp

        mod_lo = model.get_model(model_class, params_lo, params_fixed)

        mod_hi = model.get_model(model_class, params_hi, params_fixed)

        m_lo = mod_lo.get_photometry(t)
        m_hi = mod_hi.get_photometry(t)

        derivs[i] = (m_hi - m_lo) / (2.0 * dp)

    # Make the Fisher matrix.
    fish_mat = np.zeros((n_params, n_params), dtype=float)

    param_names = list(params.keys())
    
    for i in range(n_params):
        for j in range(n_params):
            ikey = param_names[i]
            jkey = param_names[j]
            fish_mat[i, j] = np.sum(derivs[ikey] * derivs[jkey] / merr**2)

    cov_mat = np.linalg.inv(fish_mat)

    return cov_mat
        
