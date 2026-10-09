"""Per-filter origin, usage lists, and filt_idx behavior."""

import inspect

import numpy as np
import pytest

from bagle import model
from bagle import model_jax
from bagle.model import validate_param_declaration


def _param_classes(module):
    """Yield classes that declare a fitter parameter list."""
    found = []
    for name in dir(module):
        cls = getattr(module, name)
        if not isinstance(cls, type):
            continue
        if 'fitter_param_names' not in cls.__dict__:
            continue
        if name.startswith('_'):
            continue
        found.append(cls)
    return found


def test_every_param_class_validates_filt_usage():
    """Every parameter class has a legal filt_param declaration."""
    classes = _param_classes(model)
    assert len(classes) > 20
    for cls in classes:
        validate_param_declaration(cls)
        # Derived views follow usage and are not a second copy of the cube.
        phot = list(cls.phot_param_names)
        ast = list(cls.astrom_param_names)
        for name, use in zip(cls.filt_param_names, cls.filt_param_usage):
            if use in ('phot', 'both'):
                assert name in phot
            if use in ('astrom', 'both'):
                assert name in ast
    return None


def test_jax_param_classes_match_numpy_filt_lists():
    """Shared JAX parameter classes use the same filter lists."""
    for cls in _param_classes(model_jax):
        validate_param_declaration(cls)
        numpy_cls = getattr(model, cls.__name__, None)
        if numpy_cls is None:
            continue
        assert list(cls.filt_param_names) == list(numpy_cls.filt_param_names)
        assert list(cls.filt_param_usage) == list(numpy_cls.filt_param_usage)
    return None


def test_gp_class_inherits_filt_lists():
    """A GP mean class keeps the parent's filter declaration."""
    parent = model.PSPL_PhotAstromParam1
    child = model.PSPL_GP_PhotAstromParam1
    assert list(child.filt_param_names) == list(parent.filt_param_names)
    assert list(child.filt_param_usage) == list(parent.filt_param_usage)
    return None


def _two_filter_kwargs(cls, xS0_E, xS0_N):
    """Keyword arguments for a two-filter no-parallax model."""
    params = {}
    for name in cls.fitter_param_names:
        if name.startswith('mag'):
            params[name] = 18.0
        elif name in ('b_sff', 'fratio_bin'):
            params[name] = 1.0
        elif name == 't0':
            params[name] = 57000.0
        elif name in ('tE', 'thetaE', 'mL', 'dL', 'sep', 'mLp', 'mLs'):
            params[name] = 10.0
        elif name == 'dL_dS':
            params[name] = 0.5
        elif name == 'dS':
            params[name] = 4000.0
        elif name in ('muL_E', 'muL_N'):
            params[name] = 0.0
        elif name == 'muS_E':
            params[name] = 4.0
        elif name == 'muS_N':
            params[name] = -3.0
        elif name == 'beta':
            params[name] = 0.4
        elif name == 'radiusS':
            params[name] = 1.0e-6
        elif name in ('log10_thetaE',):
            params[name] = 0.0
        else:
            params[name] = 0.1
    for name in cls.filt_param_names:
        params[name] = np.full(2, params[name])
    params['xS0_E'] = np.asarray(xS0_E, dtype=float)
    params['xS0_N'] = np.asarray(xS0_N, dtype=float)
    params['raL'] = 270.0
    params['decL'] = -30.0
    return params


def _assert_origin_shift(instance):
    """Astrometry of two filters differs by their xS0 at one epoch."""
    t = np.array([57000.0])
    pos0 = np.asarray(instance.get_astrometry(t, filt_idx=0), dtype=float)
    pos1 = np.asarray(instance.get_astrometry(t, filt_idx=1), dtype=float)
    delta = np.asarray(instance.xS0[1] - instance.xS0[0], dtype=float)
    np.testing.assert_allclose(pos1[0] - pos0[0], delta, atol=1e-10)
    return None


def test_pspl_two_zero_points():
    """A constant xS0 offset shifts astrometry and cancels in u."""
    cls = model.PSPL_PhotAstrom_noPar_Param1
    kw = _two_filter_kwargs(cls, [0.0, 0.2], [0.0, -0.1])
    instance = cls(**kw)
    assert instance.xS0.shape == (2, 2)
    _assert_origin_shift(instance)
    t = np.array([56990.0, 57000.0, 57010.0])
    u0 = instance.get_u(t, filt_idx=0)
    u1 = instance.get_u(t, filt_idx=1)
    np.testing.assert_allclose(u0, u1, atol=1e-10)
    return None


def test_fspl_filt_idx_reaches_astrometry():
    """FSPL image positions use the requested filter origin."""
    cls = model.FSPL_PhotAstrom_noPar_Param1
    kw = _two_filter_kwargs(cls, [0.0, 0.05], [0.0, -0.02])
    instance = cls(**kw)
    _assert_origin_shift(instance)
    return None


def test_bspl_filt_idx_reaches_astrometry():
    """BSPL resolved astrometry uses the requested filter origin."""
    cls = model.BSPL_PhotAstrom_noPar_Param1
    kw = _two_filter_kwargs(cls, [0.0, 0.04], [0.0, 0.01])
    instance = cls(**kw)
    _assert_origin_shift(instance)
    amp0 = instance.get_amplification(np.array([57000.0]), filt_idx=0)
    amp1 = instance.get_amplification(np.array([57000.0]), filt_idx=1)
    np.testing.assert_allclose(amp0, amp1, atol=1e-8)
    return None


def test_bsbl_filt_idx_reaches_astrometry():
    """BSBL astrometry uses the requested filter origin."""
    cls = model.BSBL_PhotAstrom_noPar_Param1
    # Skip cleanly if this leaf needs arguments the helper cannot invent.
    try:
        kw = _two_filter_kwargs(cls, [0.0, 0.03], [0.0, -0.02])
        instance = cls(**kw)
    except TypeError as exc:
        pytest.skip(f'BSBL constructor needs extra arguments: {exc}')
    _assert_origin_shift(instance)
    return None


def test_one_filter_xs0_is_shape_1_2():
    """One filter stores xS0 with shape (1, 2)."""
    instance = model.PSPL_PhotAstrom_noPar_Param1(
        5.0, 57000.0, 0.2, 2000.0, 0.5,
        0.01, -0.02,
        1.0, -1.0, 2.0, 3.0,
        [1.0], [18.0],
    )
    assert instance.xS0.shape == (1, 2)
    assert instance.xS0_E.shape == (1,)
    np.testing.assert_allclose(instance.xS0[0], [0.01, -0.02])
    pos = instance.get_astrometry(np.array([57000.0, 57010.0]))
    assert pos.shape == (2, 2)
    return None


def test_refpar_pi_ref_frame_is_per_filter():
    """Each astrometric filter can carry its own pi_ref_frame."""
    # A constant parallax table makes the offset analytic.
    calls = []

    def _fake(ra, dec, mjd, obsLocation='earth'):
        calls.append(str(obsLocation))
        times = np.atleast_1d(np.asarray(mjd, dtype=float))
        return np.column_stack([
            np.full(times.size, 2.0),
            np.full(times.size, -1.0),
        ])

    import bagle.parallax as parallax
    original = parallax.parallax_in_direction
    parallax.parallax_in_direction = _fake
    try:
        cls = model.PSPL_PhotAstrom_RefPar_Param3
        sig = inspect.signature(model.PSPL_PhotAstromParam3_RefPar.__init__)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if name.startswith('mag'):
                kwargs[name] = np.array([18.0, 18.0])
            elif name == 'b_sff':
                kwargs[name] = np.array([1.0, 1.0])
            elif name == 'xS0_E':
                kwargs[name] = np.array([0.0, 0.0])
            elif name == 'xS0_N':
                kwargs[name] = np.array([0.0, 0.0])
            elif name == 'pi_ref_frame':
                kwargs[name] = np.array([0.0, 0.4])
            elif name == 't0':
                kwargs[name] = 57000.0
            elif name in ('tE', 'thetaE', 'piS'):
                kwargs[name] = 1.0
            elif name == 'log10_thetaE':
                kwargs[name] = 0.0
            elif param.default is inspect.Parameter.empty:
                kwargs[name] = 0.1
        kwargs['raL'] = 270.0
        kwargs['decL'] = -30.0
        instance = cls(**kwargs)
        t = np.array([57000.0])
        pos0 = instance.get_astrometry(t, filt_idx=0)[0]
        pos1 = instance.get_astrometry(t, filt_idx=1)[0]
        # pi_ref_frame is added to the source and the lens together,
        # so the centroid still moves with the reference-frame offset.
        expected = np.array([0.4 * 2.0, 0.4 * -1.0]) * 1e-3
        np.testing.assert_allclose(pos1 - pos0, expected, atol=1e-8)
    finally:
        parallax.parallax_in_direction = original
    return None


def _origin_inputs(n_filters, kind, east, north):
    """Scalar, list, or array origin components."""
    if kind == 'scalar':
        # A scalar is repeated onto every filter.
        return east[0], north[0]
    if n_filters == 1:
        values_e = [east[0]]
        values_n = [north[0]]
    else:
        values_e = list(east[:n_filters])
        values_n = list(north[:n_filters])
    if kind == 'list':
        return values_e, values_n
    return (
        np.asarray(values_e, dtype=float),
        np.asarray(values_n, dtype=float),
    )


def _expected_origin(n_filters, kind, east, north):
    """Stored (n_filters, 2) origin for one input kind."""
    if kind == 'scalar':
        row = [east[0], north[0]]
        return np.array([row] * n_filters, dtype=float)
    if n_filters == 1:
        return np.array([[east[0], north[0]]], dtype=float)
    return np.column_stack([
        np.asarray(east[:n_filters], dtype=float),
        np.asarray(north[:n_filters], dtype=float),
    ])


def _assert_stored_origin(inst, expected):
    """xS0, xL0, and unlensed source position share filt_idx."""
    n_filters = expected.shape[0]
    assert inst.xS0.shape == (n_filters, 2)
    assert inst.xS0_E.shape == (n_filters,)
    assert inst.xS0_N.shape == (n_filters,)
    np.testing.assert_allclose(inst.xS0, expected)
    assert inst.xL0.shape == (n_filters, 2)
    if hasattr(inst, 'xL0_E'):
        assert inst.xL0_E.shape == (n_filters,)
        assert inst.xL0_N.shape == (n_filters,)
    t0 = np.array([inst.t0])
    for filt_idx in range(n_filters):
        # A source binary's unlensed centroid is not the primary origin.
        if hasattr(inst, 'get_resolved_source_astrometry_unlensed'):
            both = inst.get_resolved_source_astrometry_unlensed(
                t0, filt_idx=filt_idx)
            np.testing.assert_allclose(
                both[0, 0], inst.xS0[filt_idx], atol=1e-12)
        else:
            pos = inst.get_source_astrometry_unlensed(
                t0, filt_idx=filt_idx)
            np.testing.assert_allclose(
                pos[0], inst.xS0[filt_idx], atol=1e-12)
    if hasattr(inst, 'xS0_pri'):
        assert inst.xS0_pri.shape == (n_filters, 2)
        np.testing.assert_allclose(inst.xS0_pri, inst.xS0)
    if hasattr(inst, 'xS0_sec'):
        assert inst.xS0_sec.shape == (n_filters, 2)
    if hasattr(inst, 'xS0_com'):
        assert inst.xS0_com.shape == (n_filters, 2)
    return None


def test_xs0_stored_shape_is_n_filters_by_2():
    """xS0 is (n_filters, 2) for scalar, list, and array input.

    Phot-only models have no astrometric origin. Phot+astrom,
    a source binary, and the JAX PSPL model all store arrays.
    """
    phot = model.PSPL_Phot_noPar_Param1(
        57000.0, 0.1, 30.0, 0.01, 0.02, [1.0], [18.0])
    assert not hasattr(phot, 'xS0')
    assert not hasattr(phot, 'xS0_E')

    east = [0.01, -0.02]
    north = [0.03, 0.04]
    for n_filters in (1, 2):
        b_sff = np.ones(n_filters)
        mag = np.full(n_filters, 18.0)
        dmag = np.full(n_filters, 20.0)
        for kind in ('scalar', 'list', 'array'):
            xE, xN = _origin_inputs(n_filters, kind, east, north)
            expected = _expected_origin(n_filters, kind, east, north)
            pspl = model.PSPL_PhotAstrom_noPar_Param1(
                5.0, 57000.0, 0.2, 2000.0, 0.5,
                xE, xN, 1.0, -1.0, 2.0, 3.0, b_sff, mag)
            psbl = model.PSBL_PhotAstrom_noPar_Param1(
                10.0, 5.0, 57000.0, xE, xN, 0.4,
                0.0, 0.0, 4.0, -3.0, 2000.0, 4000.0,
                10.0, 90.0, b_sff, mag, dmag)
            bspl = model.BSPL_PhotAstrom_noPar_Param1(
                5.0, 57000.0, 0.2, 2000.0, 0.5,
                xE, xN, 1.0, -1.0, 2.0, 3.0,
                5.0, 40.0, mag, mag + 1.0, b_sff)
            jax_mod = model_jax.PSPL_PhotAstrom_noPar_Param1(
                5.0, 57000.0, 0.2, 2000.0, 0.5,
                xE, xN, 1.0, -1.0, 2.0, 3.0, b_sff, mag)
            _assert_stored_origin(pspl, expected)
            _assert_stored_origin(psbl, expected)
            _assert_stored_origin(bspl, expected)
            _assert_stored_origin(jax_mod, expected)
    return None
