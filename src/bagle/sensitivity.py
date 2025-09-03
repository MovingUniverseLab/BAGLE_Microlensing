import numpy as np
import pylab as plt
from bagle import model
import copy
import pdb
from multiprocessing import Pool

def fisher_cov_matrix_phot_astrom(t, mag_err, ast_err,
                  model_class, params, params_fixed,
                  num_deriv_frac=0.01, param_delta=None, verbose=0):
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

    Optional Parameters
    -------------------
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

        if verbose > 1:
            with np.printoptions(precision=5):
                # print(f'{i} {params_lo[i]:.3f} {params_hi[i]:.3f} {dp:.3f} {m_hi[mid]:.5f} {m_lo[mid]:.5f}')
                # print(f'{i} {params_lo[i]:.3f} {params_hi[i]:.3f} {dp:.3f} {p_hi[mid]} {p_lo[mid]}')
                print(f'{i} low = {params_lo[i]:.3f} hi = {params_hi[i]:.3f} dp = {dp:.3e} m_hi={m_hi.max():.2e} m_lo={m_lo.min():.2e}')
                print(f'{i} low = {params_lo[i]:.3f} hi = {params_hi[i]:.3f} dp = {dp:.3e} p_hi={p_hi.max():.2e} p_lo={p_lo.min():.2e}')
                print(f'{i} derivs_phot min = {derivs_phot[i].min():.3e} max = {derivs_phot[i].max():.3e}')
                print(f'{i} derivs_astr min = {derivs_astr[i].min():.3e} max = {derivs_astr[i].max():.3e}')

    # Make the Fisher matrix.
    fish_mat = np.zeros((n_params, n_params), dtype=float)

    param_names = list(params.keys())
    
    for i in range(n_params):
        for j in range(n_params):
            ikey = param_names[i]
            jkey = param_names[j]
            fish_mat[i, j] =  np.sum(derivs_phot[ikey] * derivs_phot[jkey] / mag_err**2)
            fish_mat[i, j] += np.sum(derivs_astr[ikey] * derivs_astr[jkey] / ast_err**2)
            if verbose > 2:
                # if fish_mat[i, j] == 0 and i == j:
                with np.printoptions(precision=2):
                    print(f'{i}, {j}, phot {fish_mat[i, j]:.2e} {ikey} = {derivs_phot[ikey].max():.2e}, {jkey} = {derivs_phot[jkey][0].max():.2e}')
                    print(f'{i}, {j}, astr {fish_mat[i, j]:.2e} {ikey} = {derivs_astr[ikey].max():.2e}, {jkey} = {derivs_astr[jkey][0].max():.2e}')

    # with np.printoptions(precision=2):
    #     print(fish_mat)
    # with np.printoptions(precision=3):
    #     print(fish_mat)
    #     print(cov_mat)

    try:
        cov_mat = np.linalg.inv(fish_mat)
    except np.linalg.LinAlgError:
        cov_mat = np.diag(np.ones(fish_mat.shape[0])) * np.inf

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
        if param_delta is not None and i in param_delta:
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


def fisher_cov_matrix_multi_cadence(t, mag_err, ast_err,
                                    model_class, params, params_fixed,
                                    params_in_matrix,
                                    list_of_cadence_indices,
                                    num_deriv_frac=0.01, param_delta=None, verbose=0,
                                    mp_pool_size=1):
    """
    Calculate the covariance matrix from the fisher matrix for an
    arbitrary photometric + astrometric BAGLE microlens model.
    The order of the parameters in the fisher matrix will be that
    of params.keys().

    This function allows efficient exploration of different cadences
    without recalculating the model every time. The original time stamps
    sent in can be sub-sampled with different sets of indices via the
    list_of_cadence_indices parameter. The models will not be recalculated,
    saving significant time.

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
        Dictionary of model parameters name: value. Uncertainties will be
        estimated around these values.
    params_fixed : dict
        Dictionary of fixed model parameters. Usually this includes
        raL and decL for the R.A. and Dec of the lens.
    params_in_matrix : dict
        Dictionary containing the same list of parameters as in params; but with
        a boolean indicating whether this parameter should be included in the
        covariance matrix or not. Run time is significantly reduced if you can
        drop parameters.
    list_of_cadence_indices : list of lists
        List of lists containing the time indices to include in each of the
        different cadence tests.

    Optional Parameters
    -------------------
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

    # Determine which parameters will be included in the Fisher matrix.
    in_matrix = [params_in_matrix[i] for i in params.keys()]
    n_in_matrix = np.sum(in_matrix)

    derivs_phot = {}
    derivs_astr = {}

    p_in_matrix = []

    if mp_pool_size > 1:
        # Multiprocessing... collect inputs.
        inputs = []

        for i in params.keys():
            # Skip the parameter if it shouldn't be in the matrix
            if not params_in_matrix[i]:
                continue

            # This parameter will be in the matrix. Keep track of the names.
            p_in_matrix = np.append(p_in_matrix, i)

            inputs += [(model_class.__name__, params, params_fixed, param_delta, num_deriv_frac, i, t, verbose)]

        pool = Pool(processes=mp_pool_size)

        results = pool.starmap(calc_derivatives_phot_astr, inputs)

        for pp in range(len(p_in_matrix)):
            derivs_phot[p_in_matrix[pp]] = results[pp][0]
            derivs_astr[p_in_matrix[pp]] = results[pp][1]

    else:
        for i in params.keys():
            # Skip the parameter if it shouldn't be in the matrix
            if not params_in_matrix[i]:
                continue

            # This parameter will be in the matrix. Keep track of the names.
            p_in_matrix = np.append(p_in_matrix, i)

            derivs_phot[i], derivs_astr[i] = calc_derivatives_phot_astr(model_class.__name__, params,
                                                                        params_fixed, param_delta,
                                                                        num_deriv_frac, i, t, verbose)

    # Make the Fisher matrix.
    n_cadences = len(list_of_cadence_indices)
    fish_mat = np.zeros((n_cadences, n_in_matrix, n_in_matrix), dtype=float)

    #param_names = list(params.keys())
    param_names = p_in_matrix

    for i in range(n_in_matrix):
        for j in range(n_in_matrix):
            ikey = param_names[i]
            jkey = param_names[j]

            fish_mat_phot_tmp = derivs_phot[ikey] * derivs_phot[jkey] / mag_err ** 2
            fish_mat_astr_tmp = derivs_astr[ikey] * derivs_astr[jkey] / ast_err ** 2

            for c in range(n_cadences):
                # Sample the times at the specified cadence.
                cdx = list_of_cadence_indices[c]
                fish_mat[c, i, j] = np.sum(fish_mat_phot_tmp[cdx]) + np.sum(fish_mat_astr_tmp[cdx])

            if verbose > 2:
                # if fish_mat[i, j] == 0 and i == j:
                with np.printoptions(precision=2):
                    print(f'{i}, {j}, phot {fish_mat[i, j]:.2e} '
                          f'{ikey} = {derivs_phot[ikey].max():.2e}, '
                          f'{jkey} = {derivs_phot[jkey][0].max():.2e}')
                    print(f'{i}, {j}, astr {fish_mat[i, j]:.2e} '
                          f'{ikey} = {derivs_astr[ikey].max():.2e}, '
                          f'{jkey} = {derivs_astr[jkey][0].max():.2e}')

    # Loop through the cadences and calculate the covariance matrix.
    cov_mat = np.zeros((n_cadences, n_in_matrix, n_in_matrix), dtype=float)

    for t in range(n_cadences):
        try:
            cov_mat[t, :, :] = np.linalg.inv(fish_mat[t, :, :])
        except np.linalg.LinAlgError:
            cov_mat[t, :, :] = np.nan

    return param_names, cov_mat


def calc_derivatives_phot_astr(model_name, params, params_fixed, param_delta, num_deriv_frac, param, t,
                               verbose=False):
    model_class = getattr(model, model_name)

    params_lo = copy.deepcopy(params)
    params_hi = copy.deepcopy(params)

    if param_delta is not None:
        dp = param_delta[param]
    else:
        dp = params[param] * num_deriv_frac  # Step size for differentiation will be 1%

    params_lo[param] = params[param] - dp
    params_hi[param] = params[param] + dp

    mod_lo = model.get_model(model_class, params_lo, params_fixed)
    mod_hi = model.get_model(model_class, params_hi, params_fixed)
    m_lo = mod_lo.get_photometry(t)
    m_hi = mod_hi.get_photometry(t)
    p_lo = mod_lo.get_astrometry(t)
    p_hi = mod_hi.get_astrometry(t)

    derivs_phot = (m_hi - m_lo) / (2.0 * dp)
    derivs_astr = (p_hi - p_lo) / (2.0 * dp)

    if verbose > 1:
        with np.printoptions(precision=5):
            # print(f'{i} {params_lo[i]:.3f} {params_hi[i]:.3f} {dp:.3f} {m_hi[mid]:.5f} {m_lo[mid]:.5f}')
            # print(f'{i} {params_lo[i]:.3f} {params_hi[i]:.3f} {dp:.3f} {p_hi[mid]} {p_lo[mid]}')
            print(f'{param} low = {params_lo[param]:.3f} hi = {params_hi[param]:.3f} '
                  f'dp = {dp:.3e} m_hi={m_hi.max():.2e} m_lo={m_lo.min():.2e}')
            print(f'{param} low = {params_lo[param]:.3f} hi = {params_hi[param]:.3f} '
                  f'dp = {dp:.3e} p_hi={p_hi.max():.2e} p_lo={p_lo.min():.2e}')
            print(f'{param} derivs_phot min = {derivs_phot.min():.3e} max = {derivs_phot.max():.3e}')
            print(f'{param} derivs_astr min = {derivs_astr.min():.3e} max = {derivs_astr.max():.3e}')

    return derivs_phot, derivs_astr
