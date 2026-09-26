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


def phot_dataset_has_astrometry(fitter, phot_idx):
    """Return whether a photometry dataset is paired with astrometry.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver with ``n_ast_sets`` and ``map_phot_idx_to_ast_idx``.
    phot_idx : int
        Zero-based photometry dataset index (``b_sff1`` is 0).

    Returns
    -------
    paired : bool
        True when one of the first ``n_ast_sets`` map entries points
        at ``phot_idx``.

    Notes
    -----
    String ``phot_data`` / ``ast_data`` can leave the map longer than
    ``n_ast_sets``. Entries past ``n_ast_sets`` are ignored, matching
    ``check_b_sff_astrom_priors``.
    """
    n_ast = int(getattr(fitter, "n_ast_sets", 0) or 0)
    if n_ast == 0:
        return False

    mapping = list(getattr(fitter, "map_phot_idx_to_ast_idx", []) or [])
    n_check = min(n_ast, len(mapping))
    target = int(phot_idx)
    for ast_i in range(n_check):
        if int(mapping[ast_i]) == target:
            return True

    return False


def default_b_sff_upper(fitter, param_name, filt_index, template_upper):
    """Upper edge used when a default ``b_sff`` prior is generated.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver with the photometry-to-astrometry index map.
    param_name : str
        Parameter name such as ``b_sff1``. Used when ``filt_index``
        is ``None``.
    filt_index : int or None
        1-based photometry index from ``split_param_filter_index1``.
    template_upper : float
        Upper edge from ``default_priors`` (1.5 for photometry-only).

    Returns
    -------
    upper : float
        ``1.0`` when this photometry dataset is paired with astrometry
        and ``template_upper`` is larger than 1. Otherwise
        ``template_upper``.

    Notes
    -----
    User-assigned priors are not passed through this helper. They stay
    as written and are still rejected by ``check_b_sff_astrom_priors``
    when the upper bound exceeds 1.
    """
    if filt_index is None:
        digits = "".join(c for c in str(param_name) if c.isdigit())
        phot_idx = int(digits) - 1 if digits else 0
    else:
        phot_idx = int(filt_index) - 1

    template = float(template_upper)
    if phot_dataset_has_astrometry(fitter, phot_idx) and template > 1.0:
        return 1.0

    return template


def _fingerprint_number(value):
    """First finite-looking float stored on a prior, or ``None``.

    Parameters
    ----------
    value : object
        Constructor argument, keyword, or ``high`` attribute.

    Returns
    -------
    number : float or None
        The first element as a float. ``None`` for missing values,
        strings, and callables (NumPyro constraints are callable).
    """
    if value is None or callable(value):
        return None
    if isinstance(value, (str, bytes)):
        return None

    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None

    if arr.size == 0:
        return None

    return float(arr[0])


def _prior_fingerprint(prior):
    """Identity and stored edges of one prior object.

    Parameters
    ----------
    prior : object
        A distribution, or ``None`` when the name is absent.

    Returns
    -------
    fingerprint : tuple
        ``id(prior)`` plus numeric ``args``, ``kwds``, and ``high``
        already stored on the object.

    Notes
    -----
    ``support()`` is not called. Replacing ``fitter.priors[name]``
    changes ``id``. Editing stored constructor numbers on the same
    object changes the numeric tail, so either edit misses the cache.
    """
    if prior is None:
        return (None,)

    bits = [id(prior)]

    args = getattr(prior, "args", None)
    if isinstance(args, tuple):
        for arg in args:
            number = _fingerprint_number(arg)
            if number is not None:
                bits.append(number)

    kwds = getattr(prior, "kwds", None)
    if isinstance(kwds, dict):
        for key in sorted(kwds):
            number = _fingerprint_number(kwds[key])
            if number is not None:
                bits.append((str(key), number))

    high = getattr(prior, "high", None)
    number = _fingerprint_number(high)
    if number is not None:
        bits.append(("high", number))

    return tuple(bits)


def _relevant_b_sff_names(fitter):
    """Parameter names the combined-dataset check would inspect.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver with dataset counts, the index map, and ``priors``.

    Returns
    -------
    names : tuple of str
        ``b_sffN`` (or bare ``b_sff``) for each astrometry partner
        that has a prior. Empty when the fit is not phot+astrom.
    """
    n_phot = int(getattr(fitter, "n_phot_sets", 0) or 0)
    n_ast = int(getattr(fitter, "n_ast_sets", 0) or 0)
    if n_phot == 0 or n_ast == 0:
        return tuple()

    mapping = list(getattr(fitter, "map_phot_idx_to_ast_idx", []) or [])
    priors = getattr(fitter, "priors", None) or {}
    n_check = min(n_ast, len(mapping))
    names = []
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

        names.append(param)

    return tuple(names)


def _b_sff_prior_state_key(fitter):
    """Cache token for the combined-dataset ``b_sff`` prior check.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver whose priors and dataset map define the check.

    Returns
    -------
    key : tuple
        Comparable token. A later call with the same token skips the
        support inspection.

    Notes
    -----
    The token stores ``id`` of each relevant prior plus numeric
    constructor arguments already sitting on the object. It does not
    call ``support()``. Replacing ``fitter.priors['b_sff1']`` changes
    the id. Dataset counts and ``map_phot_idx_to_ast_idx`` are included
    so a dataset-map edit is re-validated too.
    """
    n_phot = int(getattr(fitter, "n_phot_sets", 0) or 0)
    n_ast = int(getattr(fitter, "n_ast_sets", 0) or 0)
    mapping = list(getattr(fitter, "map_phot_idx_to_ast_idx", []) or [])
    # Only the entries the checker uses. A longer string-style map
    # must not change the token by itself.
    used = tuple(int(idx) for idx in mapping[:n_ast])
    priors = getattr(fitter, "priors", None) or {}
    names = _relevant_b_sff_names(fitter)
    parts = tuple(
        (name, _prior_fingerprint(priors.get(name))) for name in names
    )
    key = (n_phot, n_ast, used, parts)
    return key


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

    A passing check is cached on ``fitter._b_sff_prior_cache_key``.
    ``Prior``, ``Prior_copy``, and ``Prior_from_post`` call this on
    every sample, but a repeated call with the same prior objects and
    dataset map returns without reading supports. Replacing
    ``fitter.priors['b_sffN']`` changes the key and forces a new check.
    A failing check is not cached, so the next call still raises.
    ``solve`` still calls this explicitly.
    """
    key = _b_sff_prior_state_key(fitter)
    cached = getattr(fitter, "_b_sff_prior_cache_key", None)
    if cached == key:
        return None

    _validate_b_sff_astrom_priors(fitter)

    # Only a passing check is remembered. ValueError leaves the old key.
    fitter._b_sff_prior_cache_key = key
    return None


def _validate_b_sff_astrom_priors(fitter):
    """Read supports and raise if a paired ``b_sff`` exceeds 1.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver whose ``priors``, ``n_phot_sets``, ``n_ast_sets``, and
        ``map_phot_idx_to_ast_idx`` are already built.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        A paired ``b_sff`` prior has an upper bound above 1, or no
        finite upper bound. See ``check_b_sff_astrom_priors``.

    Notes
    -----
    This is the uncached body. ``check_b_sff_astrom_priors`` skips it
    when ``_b_sff_prior_cache_key`` still matches the priors and the
    photometry/astrometry map.
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
