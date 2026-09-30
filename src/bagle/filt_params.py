"""Filter-indexed parameter lists, cube expansion, and sky-origin helpers.

``filt_param_names`` and ``filt_param_usage`` are the declaration.
``phot_param_names`` and ``astrom_param_names`` are derived views.
The sampled cube walks ``fitter_param_names`` and expands each contiguous
run of filter-indexed names index-major, then drops fixed slots without
renumbering the suffixes.
"""

import ctypes
import json
import warnings

import numpy as np


# Usage of every name that has one value per filter. Shared geometric
# names are absent from this map and stay unsuffixed in the cube.
FILT_USAGE = {
    'b_sff': 'both',
    'mag_src': 'phot',
    'mag_base': 'phot',
    'mag_src_pri': 'phot',
    'mag_src_sec': 'both',
    'dmag_Lp_Ls': 'both',
    'fratio_bin': 'both',
    'xS0_E': 'astrom',
    'xS0_N': 'astrom',
    'pi_ref_frame': 'astrom',
}

_ALLOWED_USAGE = ('phot', 'astrom', 'both')

# Fixed value written into the model when a slot is not sampled.
FIXED_VALUE = 0.0


class _UsageView:
    """Read-only view of ``filt_param_names`` filtered by usage.

    Parameters
    ----------
    usages : tuple of str
        Usage strings to keep. ``'both'`` is included by the caller
        when the view should overlap the other one.
    """

    def __init__(self, usages):
        self.usages = tuple(usages)
        return None

    def __get__(self, obj, owner):
        names = list(getattr(owner, 'filt_param_names', []))
        usage = list(getattr(owner, 'filt_param_usage', []))
        kept = [
            name for name, use in zip(names, usage) if use in self.usages
        ]
        return kept

    def __set__(self, obj, value):
        raise AttributeError(
            'phot_param_names and astrom_param_names are derived from '
            'filt_param_names and filt_param_usage'
        )


# Class-body descriptors. Subclasses must not assign these names.
phot_param_view = _UsageView(('phot', 'both'))
astrom_param_view = _UsageView(('astrom', 'both'))


def split_param_filter_index1(name):
    """Split a suffixed parameter name into base name and 1-based index.

    Parameters
    ----------
    name : str
        Parameter name, for example ``'xS0_E1'`` or ``'t0'``.

    Returns
    -------
    base : str
        Name with a trailing filter suffix removed.
    filt_index : int or None
        1-based filter index, or None when ``name`` has no suffix.

    Notes
    -----
    A trailing ``0`` is not treated as a suffix, matching the historical
    splitter. ``'xS0_E'`` therefore stays unsuffixed.
    """
    base = name.rstrip('123456789')
    if len(base) == len(name):
        return base, None

    filt_index = int(name[len(base):])
    return base, filt_index


def lookup_param(params, name):
    """Return one sampled value, accepting an unsuffixed legacy name.

    Parameters
    ----------
    params : mapping
        Name to value. Keys may be ``xS0_E`` or ``xS0_E1``.
    name : str
        Sampled name, possibly with a 1-based filter suffix.

    Returns
    -------
    value
        ``params[name]`` when that key exists. Otherwise the value
        stored under the unsuffixed name.

    Notes
    -----
    One historical ``xS0_E`` is reused for every suffixed slot that
    does not have its own entry. A name present in neither form
    raises ``KeyError``.
    """
    if name in params:
        return params[name]

    base, filt_index = split_param_filter_index1(name)
    if filt_index is not None and base in params:
        return params[base]
    raise KeyError(name)


def validate_param_declaration(cls):
    """Check one parameter class's parallel filter lists.

    Parameters
    ----------
    cls : type
        Class that may define ``filt_param_names``.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        The class assigns ``phot_param_names`` or ``astrom_param_names``.
    ValueError
        The parallel lists disagree with ``fitter_param_names``.
    """
    # Classes that never declare filter parameters are not part of this
    # schema. Concrete leaves inherit the lists from a parameter mixin.
    has_filt = any('filt_param_names' in base.__dict__ for base in cls.__mro__)
    if not has_filt:
        return None

    for forbidden in ('phot_param_names', 'astrom_param_names'):
        if forbidden not in cls.__dict__:
            continue
        value = cls.__dict__[forbidden]
        if isinstance(value, _UsageView):
            continue
        raise TypeError(
            f'{cls.__name__} assigns {forbidden}. Declare filt_param_names '
            'and filt_param_usage instead.'
        )

    names = list(cls.filt_param_names)
    usage = list(cls.filt_param_usage)
    if len(names) != len(usage):
        raise ValueError(
            f'{cls.__name__}: filt_param_names and filt_param_usage '
            f'have lengths {len(names)} and {len(usage)}'
        )

    if len(names) != len(set(names)):
        raise ValueError(f'{cls.__name__}: filt_param_names are not unique')

    for use in usage:
        if use not in _ALLOWED_USAGE:
            raise ValueError(
                f'{cls.__name__}: usage {use!r} is not phot, astrom, or both'
            )

    fitter = list(cls.fitter_param_names)
    positions = []
    for name in names:
        if fitter.count(name) != 1:
            raise ValueError(
                f'{cls.__name__}: {name} must appear once in fitter_param_names'
            )
        positions.append(fitter.index(name))

    if positions != sorted(positions):
        raise ValueError(
            f'{cls.__name__}: filt_param_names are not in fitter_param_names order'
        )

    return None


def lists_for_fitter(fitter_param_names):
    """Build the parallel lists from a fitter-name sequence.

    Parameters
    ----------
    fitter_param_names : sequence of str
        Full base order, including filter-indexed names.

    Returns
    -------
    filt_param_names : list of str
        Filter-indexed names in the order they appear above.
    filt_param_usage : list of str
        Usage of each of those names.

    Raises
    ------
    ValueError
        A filter-indexed name is missing from ``FILT_USAGE``.
    """
    filt_param_names = []
    filt_param_usage = []
    for name in fitter_param_names:
        if name not in FILT_USAGE:
            continue
        filt_param_names.append(name)
        filt_param_usage.append(FILT_USAGE[name])

    return filt_param_names, filt_param_usage


def dataset_names(value, n_sets, kind):
    """Return catalog names, or synthesize ``phot1`` / ``ast1`` labels.

    Parameters
    ----------
    value : sequence of str, str, or None
        ``phot_data`` or ``ast_data``. A plain string is the historical
        label, such as ``'sim'``, and is not a list of catalogs.
    n_sets : int
        Number of ``t_phot`` or ``t_ast`` arrays.
    kind : str
        ``'phot'`` or ``'ast'``. Used as the synthesized-name stem and
        in the length-mismatch message.

    Returns
    -------
    names : list of str
        One name per data array. Shape is a Python list of length
        ``n_sets``.

    Notes
    -----
    A list whose length is not ``n_sets`` is an error. ``None`` and a
    string both mean the arrays are unnamed, so the names are
    ``{kind}1``, ``{kind}2``, and so on.
    """
    n_sets = int(n_sets)
    if isinstance(value, (str, bytes)) or value is None:
        return ['%s%d' % (kind, i + 1) for i in range(n_sets)]

    names = [str(item) for item in value]
    if len(names) != n_sets:
        raise ValueError(
            f'{kind}_data length does not match the number of '
            f't_{kind} arrays'
        )
    return names


def expand_fitter_names(fitter_param_names, filt_param_names, n_filters):
    """Expand contiguous filter-indexed runs, index-major.

    Parameters
    ----------
    fitter_param_names : sequence of str
        Unsuffixed base order.
    filt_param_names : sequence of str
        Names that take one value per filter.
    n_filters : int
        Length of the unified filter list. Zero yields only shared names.

    Returns
    -------
    expanded : list of str
        Shared names once, and each filter run as ``name1`` .. ``nameN``
        with the names of the run cycling inside a filter.

    Notes
    -----
    Two runs separated by a shared name stay two runs. ``xS0_E, xS0_N``
    therefore expand in place, and ``b_sff, mag_src`` expand later.
    """
    filt_set = set(filt_param_names)
    expanded = []
    index = 0
    names = list(fitter_param_names)
    n_filters = int(n_filters)

    while index < len(names):
        if names[index] not in filt_set:
            expanded.append(names[index])
            index += 1
            continue

        # One contiguous run. Do not merge across a shared name.
        run = []
        while index < len(names) and names[index] in filt_set:
            run.append(names[index])
            index += 1

        for filt in range(1, n_filters + 1):
            for name in run:
                expanded.append(f'{name}{filt}')

    return expanded


def fixed_slots(expanded, filt_param_names, filt_param_usage, has_phot, has_ast):
    """Drop suffixed names the filter's data do not use.

    Parameters
    ----------
    expanded : sequence of str
        Output of ``expand_fitter_names``.
    filt_param_names : sequence of str
        Base filter-indexed names.
    filt_param_usage : sequence of str
        Usage parallel to ``filt_param_names``.
    has_phot : sequence of bool, shape (n_filters,)
        True when that filter has a light curve.
    has_ast : sequence of bool, shape (n_filters,)
        True when that filter has a track.

    Returns
    -------
    sampled : list of str
        Names that remain in the cube. Suffixes are not renumbered.
    fixed : dict
        Suffixed name to the fixed value (0).

    Notes
    -----
    ``'astrom'`` is fixed when the filter has no astrometry.
    ``'phot'`` is fixed when the filter has no photometry.
    ``'both'`` is always sampled.
    """
    usage = dict(zip(filt_param_names, filt_param_usage))
    sampled = []
    fixed = {}

    for name in expanded:
        base, filt_index = split_param_filter_index1(name)
        if filt_index is None or base not in usage:
            sampled.append(name)
            continue

        k = filt_index - 1
        use = usage[base]
        drop = False
        if use == 'astrom' and not bool(has_ast[k]):
            drop = True
        elif use == 'phot' and not bool(has_phot[k]):
            drop = True

        if drop:
            fixed[name] = FIXED_VALUE
        else:
            sampled.append(name)

    return sampled, fixed


def build_filt_index(phot_data, ast_data):
    """Join photometric and astrometric names into one filter list.

    Parameters
    ----------
    phot_data : sequence of str
        Names in ``t_phot`` order. May be empty.
    ast_data : sequence of str
        Names in ``t_ast`` order. May be empty.

    Returns
    -------
    filt_names : list of str
        Photometric names, then astrometry-only names.
    has_phot : ndarray of bool, shape (n_filters,)
        True where the filter has photometry.
    has_ast : ndarray of bool, shape (n_filters,)
        True where the filter has astrometry.
    phot_series : list of int or None
        Index into ``phot_data`` for each filter.
    ast_series : list of int or None
        Index into ``ast_data`` for each filter.

    Raises
    ------
    ValueError
        A name is repeated inside one list.
    """
    phot_data = list(phot_data)
    ast_data = list(ast_data)

    if len(phot_data) != len(set(phot_data)):
        raise ValueError('phot_data has a repeated name')
    if len(ast_data) != len(set(ast_data)):
        raise ValueError('ast_data has a repeated name')

    filt_names = list(phot_data)
    for name in ast_data:
        if name not in filt_names:
            filt_names.append(name)

    n_filters = len(filt_names)
    has_phot = np.zeros(n_filters, dtype=bool)
    has_ast = np.zeros(n_filters, dtype=bool)
    phot_series = [None] * n_filters
    ast_series = [None] * n_filters

    for i, name in enumerate(phot_data):
        k = filt_names.index(name)
        has_phot[k] = True
        phot_series[k] = i
    for j, name in enumerate(ast_data):
        k = filt_names.index(name)
        has_ast[k] = True
        ast_series[k] = j

    return filt_names, has_phot, has_ast, phot_series, ast_series


def resolve_obs_locations(obs_location, filt_names):
    """Turn a string, list, or dict into one body name per filter.

    Parameters
    ----------
    obs_location : str, sequence of str, dict, or None
        ``None`` and a missing key both mean Earth for every filter.
        A string is repeated. A dict is keyed by filter name. A list
        is in unified filter order.
    filt_names : sequence of str
        Unified filter names.

    Returns
    -------
    locations : list of str
        Body name for each filter. Length is ``len(filt_names)``, or
        ``['earth']`` when there are no filters.

    Raises
    ------
    ValueError
        The list length is wrong, or a dict is missing a filter.
    """
    n_filters = len(filt_names)
    if n_filters == 0:
        return ['earth']

    if obs_location is None:
        return ['earth'] * n_filters

    if isinstance(obs_location, str):
        return [obs_location] * n_filters

    if isinstance(obs_location, dict):
        missing = [name for name in filt_names if name not in obs_location]
        if missing:
            raise ValueError(
                'obsLocation is missing filters: ' + ', '.join(missing)
            )
        return [str(obs_location[name]) for name in filt_names]

    locations = list(obs_location)
    if len(locations) != n_filters:
        raise ValueError(
            f'obsLocation has length {len(locations)}; expected {n_filters}'
        )
    return [str(item) for item in locations]


def stack_en(east, north):
    """Stack East/North components into a sky position.

    Parameters
    ----------
    east : float or array_like
        East coordinate. A scalar is one filter. A 1-d array is one
        value per filter.
    north : float or array_like
        North coordinate, same broadcasting rules as ``east``.

    Returns
    -------
    origin : ndarray
        Shape ``(2,)`` when both inputs are scalar (one filter, the
        historical layout). Shape ``(n_filters, 2)`` otherwise.

    Notes
    -----
    ``np.array([east, north])`` is shape ``(2, n)`` when each component
    is length ``n``. This helper always puts the sky axis last.
    """
    east_arr = np.atleast_1d(np.asarray(east, dtype=float)).ravel()
    north_arr = np.atleast_1d(np.asarray(north, dtype=float)).ravel()
    n_east = int(east_arr.size)
    n_north = int(north_arr.size)
    n_filters = max(n_east, n_north)

    if n_east == 1 and n_filters > 1:
        east_arr = np.repeat(east_arr, n_filters)
    if n_north == 1 and n_filters > 1:
        north_arr = np.repeat(north_arr, n_filters)

    if east_arr.size != north_arr.size:
        raise ValueError(
            f'East/North lengths {east_arr.size} and {north_arr.size} differ'
        )

    # One filter keeps the historical shape so existing arithmetic that
    # adds a (2,) vector still works.
    if n_filters == 1 and n_east == 1 and n_north == 1:
        return np.array([east_arr[0], north_arr[0]], dtype=float)

    return np.stack([east_arr, north_arr], axis=1)


def en_components(origin):
    """Split a sky position into East and North components.

    Parameters
    ----------
    origin : array_like
        Shape ``(2,)`` or ``(n_filters, 2)``.

    Returns
    -------
    east : float or ndarray
        Scalar when ``origin`` is one filter, otherwise shape
        ``(n_filters,)``.
    north : float or ndarray
        Same layout as ``east``.
    """
    arr = np.asarray(origin, dtype=float)
    if arr.shape == (2,):
        return arr[0], arr[1]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f'sky position has shape {arr.shape}, expected (2,) or (n, 2)'
        )
    if arr.shape[0] == 1:
        return arr[0, 0], arr[0, 1]
    return arr[:, 0].copy(), arr[:, 1].copy()


def sky_origin(origin, filt_idx=0):
    """Return one filter's East/North position as shape ``(2,)``.

    Parameters
    ----------
    origin : array_like
        Shape ``(2,)`` for one filter, or ``(n_filters, 2)``.
    filt_idx : int
        0-based filter index.

    Returns
    -------
    position : ndarray, shape (2,)
        East and North for that filter.

    Notes
    -----
    A shape ``(2,)`` array is the single catalog position. It is
    returned for filter 0. A later filter index on that array is an
    error, because there is no second catalog frame stored.
    """
    arr = np.asarray(origin, dtype=float)
    index = int(filt_idx)
    if arr.shape == (2,):
        if index != 0:
            raise IndexError(
                f'filt_idx={index} but the sky position has only one filter'
            )
        return arr
    return np.asarray(arr[index], dtype=float)


def filt_scalar(value, filt_idx=0):
    """Return one filter's scalar, or the value itself if it is scalar.

    Parameters
    ----------
    value : float or array_like
        A scalar, or a 1-d array of length ``n_filters``.
    filt_idx : int
        0-based filter index.

    Returns
    -------
    scalar : float
        The value for that filter.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 1:
            return float(arr[0])
        return float(arr[int(filt_idx)])
    return float(value)


def pack_constructor_params(sampled_names, sampled_values, class_fitter_names,
                            filt_param_names, n_filters, fixed):
    """Merge a sampled cube and fixed slots into constructor arguments.

    Parameters
    ----------
    sampled_names : sequence of str
        Cube column names, including suffixes and holes.
    sampled_values : sequence
        One value per sampled name, in the same order.
    class_fitter_names : sequence of str
        Unsuffixed ``fitter_param_names`` on the parameter class.
        This is the constructor order.
    filt_param_names : sequence of str
        Names that are sequences of length ``n_filters``.
    n_filters : int
        Unified filter count.
    fixed : dict
        Suffixed name to fixed float.

    Returns
    -------
    arguments : list
        Positional constructor values. Filter-indexed entries are
        lists of length ``n_filters``.

    Notes
    -----
    A missing suffix is taken from ``fixed``. A suffix that is in
    neither place raises ``KeyError``.
    """
    by_name = {name: sampled_values[i] for i, name in enumerate(sampled_names)}
    filt_set = set(filt_param_names)
    arguments = []

    for name in class_fitter_names:
        if name not in filt_set:
            if name not in by_name:
                raise KeyError(name)
            arguments.append(by_name[name])
            continue

        sequence = []
        for filt in range(1, int(n_filters) + 1):
            suffixed = f'{name}{filt}'
            if suffixed in by_name:
                sequence.append(by_name[suffixed])
            elif suffixed in fixed:
                sequence.append(fixed[suffixed])
            else:
                raise KeyError(suffixed)
        arguments.append(sequence)

    return arguments


def pack_optional_param_dicts(sampled_names, sampled_values, optional_names):
    """Group suffixed optional parameters into constructor dictionaries.

    Parameters
    ----------
    sampled_names : sequence of str
        Cube column names, including suffixes.
    sampled_values : sequence
        One value per sampled name, in the same order.
    optional_names : sequence of str
        Unsuffixed optional names, in constructor order. GP classes
        declare these on ``phot_optional_param_names``.

    Returns
    -------
    dicts : list of dict
        One dictionary per optional name, keyed by the 0-based filter
        index. Empty when none of those names were sampled. ``add_err``
        and ``mult_err`` are not in this list; they stay on the cube
        and are not constructor arguments.

    Notes
    -----
    A missing optional name becomes an empty dictionary so the
    remaining names stay aligned with the constructor. Values are
    Python floats. A ctypes pointer's entries are floats already;
    converting them keeps a NumPy scalar from leaking into the model.
    """
    optional = list(optional_names)
    if not optional:
        return []

    grouped = {name: {} for name in optional}
    present = False
    allowed = set(optional)
    for name, value in zip(sampled_names, sampled_values):
        base, filt_index = split_param_filter_index1(name)
        if filt_index is None or base not in allowed:
            continue
        present = True
        grouped[base][int(filt_index) - 1] = float(value)

    if not present:
        return []
    return [grouped[name] for name in optional]


def cube_has_derived_room(params, n_params):
    """Return whether derived parameters can be written into ``params``.

    Parameters
    ----------
    params : array_like
        Positional cube. Mappings are not passed here.
    n_params : int
        Sampled names plus derived names. That is the MultiNest
        allocation.

    Returns
    -------
    has_room : bool
        True for a ctypes pointer. PyMultiNest passes the cube that
        way: item access works, ``len`` does not, and the buffer is
        ``n_params`` long. True for a sequence at least that long.
        False for a shorter sequence, which is left unchanged.

    Notes
    -----
    Any other object that supports indexing but not ``len`` is treated
    as that same MultiNest buffer.
    """
    if isinstance(params, ctypes._Pointer):
        return True
    try:
        size = len(params)
    except TypeError:
        return True
    return size >= int(n_params)


def warn_unmatched_suffixed_priors(priors, fitter_param_names):
    """Warn when a suffixed prior is not a sampled cube name.

    Parameters
    ----------
    priors : mapping or None
        Name to prior. Keys the user assigned after the defaults.
    fitter_param_names : sequence of str
        Names MultiNest, PyMC, and NumPyro actually sample.

    Returns
    -------
    None

    Notes
    -----
    ``priors['xS0_E1'] = ...`` does not replace the default on
    ``xS0_E3``. The extra key is ignored and the fit keeps the wide
    default. An unsuffixed key is left alone: several generators are
    still stored under the base name.
    """
    if not priors:
        return None

    sampled = [str(name) for name in fitter_param_names]
    sampled_set = set(sampled)
    for name in priors:
        base, filt_index = split_param_filter_index1(str(name))
        if filt_index is None or name in sampled_set:
            continue

        # Same base, different suffix: that is the name the user meant.
        matches = [
            other for other in sampled
            if split_param_filter_index1(other)[0] == base
        ]
        if matches:
            have = ', '.join(matches)
        else:
            have = 'none'
        warnings.warn(
            f'Prior {name!r} is not in the sampled cube, so it is '
            f'ignored and the default prior stays in effect. '
            f'Sampled {base} names: {have}.',
            UserWarning,
            stacklevel=4,
        )
    return None


def scalar_bound(value):
    """Return a Python float when a bound is one number.

    Parameters
    ----------
    value : float or array_like
        Lower or upper edge passed to a prior generator. A length-1
        array is the scalar produced by ``np.array([x])``.

    Returns
    -------
    bound : float or array_like
        ``float`` when ``value`` has one element. A longer sequence
        is returned unchanged, so a vectorized distribution stays
        vectorized.

    Notes
    -----
    One cube slot cannot store a vector draw. ``prior_draw_scalar``
    raises when ``ppf`` returns more than one value.
    """
    arr = np.asarray(value, dtype=float)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return value


def prior_draw_scalar(value, name):
    """Turn one prior draw into a Python float for a cube slot.

    Parameters
    ----------
    value : float or array_like
        Result of ``prior.ppf`` or one posterior sample.
    name : str
        Cube parameter this draw belongs to.

    Returns
    -------
    draw : float
        The single value. A length-1 array is unwrapped. A NumPy
        scalar is converted with ``float`` so a ctypes ``c_double``
        slot can store it.

    Raises
    ------
    ValueError
        If ``value`` does not contain exactly one element. A
        multi-element draw is not reduced to its first entry.
    """
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError(
            f"Prior for {name!r} returned {int(arr.size)} values; "
            "each cube parameter needs one scalar."
        )
    return float(arr[0])


def cube_as_floats(cube, n):
    """Copy the first ``n`` cube entries, including a ctypes pointer.

    Parameters
    ----------
    cube : array_like
        Sequence or ctypes pointer. Entry ``i`` is ``cube[i]``.
    n : int
        Number of values to read.

    Returns
    -------
    values : ndarray, shape (n,)
        Float64 copy.

    Notes
    -----
    ``numpy.asarray`` is not used. A MultiNest pointer has no length
    and does not wrap as an array.
    """
    values = [float(cube[i]) for i in range(int(n))]
    return np.asarray(values, dtype=np.float64)


def last_filt_run(fitter_param_names, filt_param_names):
    """Return the last contiguous filter-indexed run.

    Parameters
    ----------
    fitter_param_names : sequence of str
        Unsuffixed base order.
    filt_param_names : sequence of str
        Names that expand per filter.

    Returns
    -------
    run : list of str
        The last run, or an empty list when there is none.

    Notes
    -----
    Photometric names are appended at the end of ``fitter_param_names``,
    so this run is the block that optional photometric parameters used
    to be interleaved with.
    """
    filt_set = set(filt_param_names)
    last = []
    index = 0
    names = list(fitter_param_names)
    while index < len(names):
        if names[index] not in filt_set:
            index += 1
            continue
        run = []
        while index < len(names) and names[index] in filt_set:
            run.append(names[index])
            index += 1
        last = run
    return last


def interleave_optional(sampled, fitter_param_names, filt_param_names,
                        n_filters, optional_by_filter, ast_optional):
    """Insert per-filter optional names after the last filter run.

    Parameters
    ----------
    sampled : sequence of str
        Expanded names after fixed slots are removed.
    fitter_param_names : sequence of str
        Unsuffixed class order.
    filt_param_names : sequence of str
        Filter-indexed base names.
    n_filters : int
        Unified filter count.
    optional_by_filter : sequence of sequence of str
        Already-suffixed optional names for each filter. Empty for a
        filter that has no photometry.
    ast_optional : sequence of str
        Already-suffixed astrometric optional names, appended at the end.

    Returns
    -------
    names : list of str
        Sampled cube order.
    """
    last = set(last_filt_run(fitter_param_names, filt_param_names))
    prefix = []
    groups = {}
    for name in sampled:
        base, filt_index = split_param_filter_index1(name)
        if base in last and filt_index is not None:
            groups.setdefault(filt_index, []).append(name)
        else:
            prefix.append(name)

    names = list(prefix)
    for filt in range(1, int(n_filters) + 1):
        names.extend(groups.get(filt, []))
        if filt - 1 < len(optional_by_filter):
            names.extend(list(optional_by_filter[filt - 1]))
    names.extend(list(ast_optional))
    return names


def longest_ast_series(data, ast_series):
    """Pick the astrometric series with the longest time baseline.

    Parameters
    ----------
    data : dict
        Data dictionary with ``t_ast{j+1}`` arrays.
    ast_series : sequence of int or None
        Astrometric-list index for each unified filter.

    Returns
    -------
    series : int or None
        Index into ``ast_data``. None when no astrometry is present.
    """
    best = None
    best_span = -1.0
    for series in ast_series:
        if series is None:
            continue
        times = np.asarray(data['t_ast' + str(int(series) + 1)], dtype=float)
        if times.size == 0:
            span = 0.0
        else:
            span = float(np.nanmax(times) - np.nanmin(times))
        if span > best_span:
            best_span = span
            best = int(series)
    return best


# Legacy columns already announced, so a second file does not repeat it.
_LEGACY_COLUMN_WARNED = set()

# Results files written by this layout.
RESULTS_SCHEMA = 2


def adapt_legacy_filter_columns(table, n_filters):
    """Copy unsuffixed legacy columns onto every filter slot.

    Parameters
    ----------
    table : astropy.table.Table
        Chain table. May contain ``xS0_E``, ``xS0_N``, or
        ``pi_ref_frame`` without a filter suffix.
    n_filters : int
        Number of filters in the model being loaded.

    Returns
    -------
    table : astropy.table.Table
        The same table. Each legacy column is duplicated as
        ``name1`` .. ``name{n_filters}``.

    Notes
    -----
    A table that already has both the unsuffixed name and any
    suffixed copy is rejected. The unsuffixed column is left in
    place. One filter is a rename onto the ``1`` suffix, not a
    fill with zero. ``pi_ref_frame`` is copied only when the old
    column is present.
    """
    n_filters = int(n_filters)
    if n_filters < 1:
        return table

    for base in ('xS0_E', 'xS0_N', 'pi_ref_frame'):
        suffixed = [f'{base}{k}' for k in range(1, n_filters + 1)]
        has_plain = base in table.colnames
        present = [name for name in suffixed if name in table.colnames]
        if has_plain and present:
            raise ValueError(
                f'{base} is both unsuffixed and suffixed ({present[0]})'
            )
        if not has_plain or present:
            continue

        # Warn once per legacy name for this process.
        if base not in _LEGACY_COLUMN_WARNED:
            warnings.warn(
                f'Copying legacy column {base} onto every filter slot',
                stacklevel=2,
            )
            _LEGACY_COLUMN_WARNED.add(base)
        for name in suffixed:
            table[name] = np.array(table[base], copy=True)
    return table


def write_results_schema2(outroot, sampled_names, fixed_dataset_params,
                          filt_names, has_phot, has_ast, obs_locations):
    """Write schema-2 JSON and FITS extensions next to a results file.

    Parameters
    ----------
    outroot : str
        Results path without an extension. The JSON sidecar is
        ``outroot + '.json'`` and the FITS file is ``outroot + '.fits'``.
    sampled_names : sequence of str
        Names stored as columns of the chain.
    fixed_dataset_params : dict
        Suffixed name to fixed value. These are not chain columns.
    filt_names : sequence of str
        Unified filter names.
    has_phot, has_ast : sequence of bool
        Whether each filter has photometry or astrometry.
    obs_locations : sequence of str
        Observer for each filter.

    Returns
    -------
    None

    Notes
    -----
    The primary chain table is left as written by the sampler loader.
    ``FIXED`` and ``FILTERS`` are extra extensions. A missing FITS file
    still gets the JSON sidecar.
    """
    from astropy.io import fits
    from astropy.table import Table

    payload = {
        'schema': RESULTS_SCHEMA,
        'sampled_names': list(sampled_names),
        'fixed': {
            str(key): float(val)
            for key, val in dict(fixed_dataset_params or {}).items()
        },
        'filt_names': list(filt_names),
        'has_phot': [bool(flag) for flag in has_phot],
        'has_ast': [bool(flag) for flag in has_ast],
        'obs_locations': [str(loc) for loc in obs_locations],
    }
    with open(outroot + '.json', 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')

    fits_path = outroot + '.fits'
    try:
        hdul = fits.open(fits_path, mode='update')
    except FileNotFoundError:
        return None

    # Drop a previous copy of these extensions before appending.
    drop = [
        idx for idx, hdu in enumerate(hdul)
        if hdu.name in ('FIXED', 'FILTERS')
    ]
    for idx in reversed(drop):
        del hdul[idx]

    hdul[0].header['SCHEMA'] = RESULTS_SCHEMA
    fixed_tab = Table(
        names=('name', 'value'),
        dtype=('U64', float),
    )
    for key, val in payload['fixed'].items():
        fixed_tab.add_row((key, val))
    filt_tab = Table()
    filt_tab['name'] = payload['filt_names']
    filt_tab['has_phot'] = payload['has_phot']
    filt_tab['has_ast'] = payload['has_ast']
    filt_tab['obs_location'] = payload['obs_locations']
    hdul.append(fits.BinTableHDU(fixed_tab, name='FIXED'))
    hdul.append(fits.BinTableHDU(filt_tab, name='FILTERS'))
    hdul.close()
    return None
