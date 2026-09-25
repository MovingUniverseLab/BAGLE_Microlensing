"""Upper-bound check for ``b_sff`` on combined photometry+astrometry fits.

The astrometric centroid treats a negative lens flux (``b_sff > 1``) as a
dark lens. A prior that can still draw those values makes the centroid
prior-dependent in a way photometry does not, so combined datasets must
truncate ``b_sff`` at 1 or below before sampling starts.
"""

import numpy as np


def _as_float(value):
    """Convert a scalar-like bound to ``float``.

    Parameters
    ----------
    value : array_like
        Bound stored on a prior (Python float, NumPy scalar, or a
        length-1 array).

    Returns
    -------
    bound : float or None
        The first element as a float, or ``None`` when ``value`` is
        missing.
    """
    if value is None:
        return None

    # JAX / NumPy scalars and length-1 arrays all collapse the same way.
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None

    return float(arr[0])


def _scipy_support_upper(prior):
    """Upper edge from a scipy frozen ``support()``, if that call works.

    Parameters
    ----------
    prior : object
        Candidate scipy frozen distribution. Other objects are ignored.

    Returns
    -------
    upper : float or None
        Inclusive upper bound, or ``None`` when ``prior`` is not a
        scipy-style frozen rv.
    """
    support = getattr(prior, "support", None)
    if not callable(support):
        return None

    # scipy's frozen rv.support() takes no arguments and returns (low, high).
    # NumPyro constraints are also callable, but they require a point.
    try:
        bounds = support()
    except TypeError:
        return None

    if not isinstance(bounds, tuple) or len(bounds) != 2:
        return None

    return _as_float(bounds[1])


def _numpyro_high(prior):
    """Upper edge from a NumPyro ``.high`` attribute.

    Parameters
    ----------
    prior : object
        NumPyro distribution, or anything else.

    Returns
    -------
    upper : float or None
        ``prior.high`` when it is a numeric scalar, else ``None``.
    """
    high = getattr(prior, "high", None)
    if high is None or callable(high):
        return None

    return _as_float(high)


def _pymc_upper(prior):
    """Upper edge of a PyMC random variable, when the op encodes one.

    Parameters
    ----------
    prior : object
        PyMC random variable (a tensor with ``owner``), or anything else.

    Returns
    -------
    upper : float or None
        Inclusive upper bound for ``uniform`` and ``truncated_*`` ops.
        ``None`` when the op is not one of those (the caller then treats
        the prior as unbounded).

    Notes
    -----
    PyMC v6 stores distribution parameters on ``owner.inputs`` after the
    RNG and size arguments. A normal has no upper input; do not treat the
    last input of every op as a bound.
    """
    owner = getattr(prior, "owner", None)
    if owner is None:
        return None

    op = getattr(owner, "op", None)
    if op is None:
        return None

    op_name = str(getattr(op, "name", "") or type(op).__name__).lower()
    consts = []
    for inp in list(getattr(owner, "inputs", []))[2:]:
        data = getattr(inp, "data", None)
        value = _as_float(data)
        if value is None:
            continue
        consts.append(value)

    # Uniform inputs are (lower, upper).
    if op_name in ("uniform", "uniformrv") and len(consts) >= 2:
        return consts[1]

    # TruncatedNormal inputs end in (lower, upper).
    if "truncat" in op_name and len(consts) >= 2:
        return consts[-1]

    return None


def prior_upper_bound(prior):
    """Inclusive upper bound of a prior, or ``+inf`` if it is unbounded.

    Parameters
    ----------
    prior : object
        A scipy frozen distribution, a NumPyro distribution, a PyMC
        random variable, or a custom prior.

    Returns
    -------
    upper : float
        Inclusive upper bound of the support. ``np.inf`` when the prior
        is unbounded or its support cannot be read.

    Notes
    -----
    Unknown custom priors fail closed. A combined photometry and
    astrometry fit has to show that ``b_sff`` cannot exceed 1.
    """
    # NumPyro Uniform / TruncatedNormal publish ``.high``.
    high = _numpyro_high(prior)
    if high is not None:
        return high

    # scipy frozen distributions: support() -> (low, high).
    scipy_high = _scipy_support_upper(prior)
    if scipy_high is not None:
        return scipy_high

    # PyMC RVs. A recognized op with no finite upper stays unbounded.
    if getattr(prior, "owner", None) is not None:
        pymc_high = _pymc_upper(prior)
        if pymc_high is None:
            return np.inf
        return pymc_high

    # Custom / unrecognized priors are treated as unbounded.
    return np.inf


def _dataset_label(phot_idx, ast_idx):
    """Human-readable dataset numbers (1-based, matching ``b_sffN``).

    Parameters
    ----------
    phot_idx : int
        Zero-based photometry dataset index.
    ast_idx : int
        Zero-based astrometry dataset index.

    Returns
    -------
    label : str
        Phrase naming both datasets.
    """
    label = (
        f"photometry dataset {phot_idx + 1} "
        f"(paired with astrometry dataset {ast_idx + 1})"
    )
    return label


def check_b_sff_astrom_priors(fitter):
    """Raise if a combined phot+astrom dataset allows ``b_sff > 1``.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver whose ``priors``, ``n_phot_sets``, ``n_ast_sets``, and
        ``map_phot_idx_to_ast_idx`` are already built. Priors may be
        scipy, PyMC, or NumPyro objects, or a custom distribution.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        The ``b_sff`` prior for a dataset that has both photometry and
        astrometry has an upper bound above 1, or no finite upper bound.
        The message names the parameter, the dataset, and the bound.
        Unbounded priors (a plain Gaussian, or a custom prior whose
        support cannot be read) must be truncated at 1.

    Notes
    -----
    Photometry-only datasets are not checked. Astrometry-only fits have
    no ``b_sff`` photometry parameter and are skipped. Dataset numbers
    follow the fitter's 1-based names (``b_sff1`` matches photometry
    index 0). When ``phot_data`` / ``ast_data`` were stored as strings,
    ``map_phot_idx_to_ast_idx`` can be longer than ``n_ast_sets``; only
    the first ``n_ast_sets`` entries are used.
    """
    n_phot = int(getattr(fitter, "n_phot_sets", 0) or 0)
    n_ast = int(getattr(fitter, "n_ast_sets", 0) or 0)

    # Photometry-only and astrometry-only fits are outside this rule.
    if n_phot == 0 or n_ast == 0:
        return None

    mapping = list(getattr(fitter, "map_phot_idx_to_ast_idx", []) or [])
    priors = getattr(fitter, "priors", None) or {}

    # One photometry partner per astrometric dataset.
    n_check = min(n_ast, len(mapping))
    for ast_i in range(n_check):
        phot_idx = int(mapping[ast_i])
        if phot_idx < 0 or phot_idx >= n_phot:
            continue

        # b_sff1 is photometry dataset 0. Bare ``b_sff`` is the same set.
        param = f"b_sff{phot_idx + 1}"
        if param not in priors and phot_idx == 0 and "b_sff" in priors:
            param = "b_sff"
        if param not in priors:
            continue

        upper = prior_upper_bound(priors[param])
        dataset = _dataset_label(phot_idx, ast_i)

        # A finite bound above 1 can still draw a negative lens flux.
        if np.isfinite(upper) and upper > 1.0:
            raise ValueError(
                f"{param} prior for {dataset} has upper bound {upper}; "
                "combined photometry+astrometry datasets require an "
                "upper bound of 1 or less."
            )

        # Plain Gaussians and unknown custom priors have no usable cap.
        if not np.isfinite(upper):
            raise ValueError(
                f"{param} prior for {dataset} is unbounded "
                f"(upper bound {upper}). The prior must be truncated "
                "at 1 or below."
            )

    return None
