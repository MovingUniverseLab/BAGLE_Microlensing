import numpy as np
import pylab as plt
from bagle import model
import copy

def fisher_cov_matrix_phot_astrom(t, mag_err, ast_err,
                  model_class, params, params_fixed,
                  num_deriv_frac=0.01, param_delta=None):
    """
    Calculate the covariance matrix from the fisher matrix for an
    arbitrary photometric + astrometric BAGLE microlens model.
    The order of the parameters in the fisher matrix will be that
    of params.keys().

    Parameters
    ----------
    t : array-like
        Array of times at which observations are sampled.
    mag_err : array-like
        Array of photometric uncertainties at each observation time.
    ast_err: array-like, shape=(N_times, 2)
        Array of astrometric uncertainties at each observation time.
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
    param_delta : dict
        Dictionary of perturbations to make for each parameter when calculating
        the numerical derivative. If param_delta is set, then it overrides
        the num_deriv_frac. This is paritcularly useful to use as many of
        the parameters in PhotAstrom methods have very different scales.
        Setting param_delta helps to have a properly conditioned fisher
        matrix for inversion.
    """
    mod_par = model.get_model(model_class, params, params_fixed)
    
    # Calculate the derivatives numerically.
    n_params = len(params)

    derivs_phot = {}
    derivs_astr = {}

    for i in params.keys():
        if param_delta is not None:
            dp = param_delta[i]
        else:
            dp = params[i] * num_deriv_frac  # Step size for differentiation will be 1%

        params_lo = copy.deepcopy(params)
        params_hi = copy.deepcopy(params)

        params_lo[i] = params[i] - dp
        params_hi[i] = params[i] + dp

        mod_lo = model.get_model(model_class, params_lo, params_fixed)
        mod_hi = model.get_model(model_class, params_hi, params_fixed)

        m_lo = mod_lo.get_photometry(t)
        m_hi = mod_hi.get_photometry(t)

        p_lo = mod_lo.get_astrometry(t)
        p_hi = mod_hi.get_astrometry(t)

        derivs_phot[i] = (m_hi - m_lo) / (2.0 * dp)
        derivs_astr[i] = (p_hi - p_lo) / (2.0 * dp)

        mid = int(len(m_hi) / 2.0)
        with np.printoptions(precision=5):
            print(f'{i} {params_lo[i]:.3f} {params_hi[i]:.3f} {dp:.3f} {m_hi[mid]:.5f} {m_lo[mid]:.5f}')
            print(f'{i} {params_lo[i]:.3f} {params_hi[i]:.3f} {dp:.3f} {p_hi[mid]} {p_lo[mid]}')

    # Make the Fisher matrix.
    fish_mat = np.zeros((n_params, n_params), dtype=float)

    param_names = list(params.keys())
    
    for i in range(n_params):
        for j in range(n_params):
            ikey = param_names[i]
            jkey = param_names[j]
            fish_mat[i, j] =  np.sum(derivs_phot[ikey] * derivs_phot[jkey] / mag_err**2)
            fish_mat[i, j] += np.sum(derivs_astr[ikey] * derivs_astr[jkey] / ast_err**2)
            #if fish_mat[i, j] == 0 and i == j:
            # with np.printoptions(precision=2):
            #     print(i, j, 'phot', fish_mat[i, j], ikey,  '=', derivs_phot[ikey][0:1], jkey, '=', derivs_phot[jkey][0:1])
            #     print(i, j, 'astr', fish_mat[i, j], ikey,  '=', derivs_astr[ikey][0:1], jkey, '=', derivs_astr[jkey][0:1])

    # with np.printoptions(precision=2):
    #     print(fish_mat)
    cov_mat = np.linalg.inv(fish_mat)
    # with np.printoptions(precision=3):
    #     print(fish_mat)
    #     print(cov_mat)

    return cov_mat


def fisher_matrix(t, merr,
                  model_class, params, params_fixed,
                  num_deriv_frac=0.01, param_delta=None,
                  verbose=False):
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
    param_delta : dict
        Dictionary of perturbations to make for each parameter when calculating
        the numerical derivative. If param_delta is set, then it overrides
        the num_deriv_frac. This is paritcularly useful to use as many of
        the parameters in PhotAstrom methods have very different scales.
        Setting param_delta helps to have a properly conditioned fisher
        matrix for inversion.
    """
    mod_par = model.get_model(model_class, params, params_fixed)

    # Calculate the derivatives numerically.
    n_params = len(params)

    derivs = {}

    for i in params.keys():
        if param_delta is not None:
            dp = param_delta[i]
        else:
            dp = params[i] * num_deriv_frac  # Step size for differentiation will be 1%

        params_lo = copy.deepcopy(params)
        params_hi = copy.deepcopy(params)

        params_lo[i] = params[i] - dp
        params_hi[i] = params[i] + dp

        mod_lo = model.get_model(model_class, params_lo, params_fixed)
        mod_hi = model.get_model(model_class, params_hi, params_fixed)

        m_lo = mod_lo.get_photometry(t)
        m_hi = mod_hi.get_photometry(t)

        derivs[i] = (m_hi - m_lo) / (2.0 * dp)

        if verbose:
            mid = int(len(m_hi) / 2.0)
            with np.printoptions(precision=5):
                print(f'{i} {params_lo[i]:.3f} {params_hi[i]:.3f} {dp:.3f} {m_hi[mid]:.5f} {m_lo[mid]:.5f} {derivs[i][mid]:.3e}')

    # Make the Fisher matrix.
    fish_mat = np.zeros((n_params, n_params), dtype=float)

    param_names = list(params.keys())

    for i in range(n_params):
        for j in range(n_params):
            ikey = param_names[i]
            jkey = param_names[j]
            fish_mat[i, j] = np.sum(derivs[ikey] * derivs[jkey] / merr ** 2)

            if fish_mat[i, j] == 0 and verbose:
                print(f'{i} {j} fish = {fish_mat[i, j]:.2e} ' +
                      f'{ikey} = {derivs[ikey][mid]:.3e} ' +
                      f'{jkey} = {derivs[jkey][mid]:.3e}')
    try:
        cov_mat = np.linalg.inv(fish_mat)
    except np.linalg.LinAlgError:
        cov_mat = np.diag(np.ones(fish_mat.shape[0])) * np.inf

    return cov_mat

