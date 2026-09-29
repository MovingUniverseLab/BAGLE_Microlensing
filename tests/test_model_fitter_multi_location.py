"""Cube expansion, mixed filters, observers, and the legacy loader."""

import os
import warnings

import numpy as np
import pytest
from astropy.table import Table

from bagle import model
from bagle import model_fitter
from bagle.filt_params import (
    adapt_legacy_filter_columns,
    expand_fitter_names,
)


def _peaked_mag(t, t0=57000.0):
    """A bright peak so the t0 prior window is not empty."""
    return 19.0 - np.exp(-0.5 * ((t - t0) / 4.0) ** 2)


def make_data(phot_names, ast_names, obs=None):
    """Minimal photometry and astrometry for a fitter.

    Parameters
    ----------
    phot_names, ast_names : sequence of str
        Dataset names. Astrometry-only names are not repeated in phot.
    obs : dict or None
        Observer keyed by filter name.

    Returns
    -------
    data : dict
        Fitter data dictionary.
    """
    data = {
        'raL': 270.0,
        'decL': -29.0,
        'target': 'unit',
        'phot_data': list(phot_names),
        'ast_data': list(ast_names),
    }
    for i in range(len(phot_names)):
        t = 57000.0 + np.linspace(-30.0, 30.0, 15)
        data[f't_phot{i + 1}'] = t
        data[f'mag{i + 1}'] = _peaked_mag(t)
        data[f'mag_err{i + 1}'] = np.full(t.size, 0.02)
    for i in range(len(ast_names)):
        t = 56800.0 + np.linspace(0.0, 400.0, 8)
        data[f't_ast{i + 1}'] = t
        data[f'xpos{i + 1}'] = np.linspace(0.0, 0.01, t.size)
        data[f'ypos{i + 1}'] = np.linspace(0.0, -0.01, t.size)
        data[f'xpos_err{i + 1}'] = np.full(t.size, 1.0e-3)
        data[f'ypos_err{i + 1}'] = np.full(t.size, 1.0e-3)
    if obs is not None:
        data['obsLocation'] = obs
    return data


def _solver(data, cls):
    """Build a NumPy fitter without sampling."""
    out = '/tmp/bagle_multi_loc_unit_'
    os.makedirs('/tmp', exist_ok=True)
    return model_fitter.MicrolensSolver(
        data,
        cls,
        outputfiles_basename=out,
        verbose=False,
    )


def test_n1_cube_matches_old_order_after_xs0_rename():
    """One joint filter is the historical cube with xS0 renamed."""
    data = make_data(['ogle'], ['ogle'])
    fitter = _solver(data, model.PSPL_PhotAstrom_noPar_Param1)
    base = list(model.PSPL_PhotAstromParam1.fitter_param_names)
    # xS0 and pi_ref_frame were not filter-indexed in old cubes.
    old_filt = [
        name for name in model.PSPL_PhotAstromParam1.filt_param_names
        if name not in ('xS0_E', 'xS0_N', 'pi_ref_frame')
    ]
    old = expand_fitter_names(base, old_filt, 1)
    renamed = []
    for name in fitter.fitter_param_names:
        if name in ('xS0_E1', 'xS0_N1', 'pi_ref_frame1'):
            renamed.append(name[:-1])
        else:
            renamed.append(name)
    assert renamed == old
    assert 'xS0_E' not in fitter.fitter_param_names
    assert 'xS0_E1' in fitter.fitter_param_names
    return None


def test_ast_on_second_phot_filter_keeps_suffix_2():
    """A hole does not renumber the astrometric filter."""
    data = make_data(['ogle', 'spitzer'], ['spitzer'])
    fitter = _solver(data, model.PSPL_PhotAstrom_noPar_Param1)
    names = fitter.fitter_param_names
    assert 'xS0_E2' in names
    assert 'xS0_N2' in names
    assert 'xS0_E1' not in names
    assert 'mag_src1' in names
    assert 'mag_src2' in names
    assert fitter.fixed_dataset_params['xS0_E1'] == 0.0
    assert fitter.fixed_dataset_params['xS0_N1'] == 0.0
    return None


def test_mixed_four_filter_cube():
    """Phot-only, joint, and ast-only slots keep their indices."""
    data = make_data(
        ['ogle', 'spitzer', 'keck'],
        ['keck', 'gaia'],
        obs={
            'ogle': 'earth',
            'spitzer': 'mars',
            'keck': 'earth',
            'gaia': 'jupiter',
        },
    )
    fitter = _solver(data, model.PSPL_PhotAstrom_Par_Param1)
    names = list(fitter.fitter_param_names)
    assert names.count('xS0_E3') == 1
    assert 'xS0_E4' in names and 'xS0_N4' in names
    assert 'xS0_E1' not in names and 'xS0_E2' not in names
    assert 'mag_src4' not in names
    assert 'b_sff4' in names
    assert 'mag_src1' in names and 'mag_src3' in names
    assert fitter.fixed_dataset_params['mag_src4'] == 0.0
    assert fitter.map_phot_idx_to_ast_idx == [2, 3]
    assert list(fitter.obs_locations) == [
        'earth', 'mars', 'earth', 'jupiter',
    ]
    # Suffix 3 stays 3. Nothing is renumbered into the hole.
    assert 'b_sff3' in names
    return None


def test_numpy_and_jax_likelihood_see_second_observer(monkeypatch):
    """Both fitters send each filter's observer into the parallax table."""
    seen = []

    def _fake(ra, dec, mjd, obsLocation='earth'):
        seen.append(str(obsLocation))
        times = np.atleast_1d(np.asarray(mjd, dtype=float))
        if str(obsLocation) == 'spitzer':
            east, north = 0.4, -0.2
        else:
            east, north = 0.05, 0.01
        return np.column_stack([
            np.full(times.size, east),
            np.full(times.size, north),
        ])

    monkeypatch.setattr(
        'bagle.parallax.parallax_in_direction', _fake
    )
    data = make_data(
        ['ogle', 'spitzer'],
        ['ogle'],
        obs={'ogle': 'earth', 'spitzer': 'spitzer'},
    )
    numpy_fit = _solver(data, model.PSPL_PhotAstrom_Par_Param1)
    cube = []
    for name in numpy_fit.fitter_param_names:
        raw = numpy_fit.priors[name].ppf(0.5)
        value = float(np.asarray(raw).ravel()[0])
        # The t0 window can be empty for a flat light curve. Use the
        # epoch the fake photometry was built around.
        if not np.isfinite(value):
            value = 57000.0 if name == 't0' else 0.1
        cube.append(value)
    cube = np.asarray(cube, dtype=float)
    seen.clear()
    ln_numpy = float(numpy_fit.log_likely(cube))
    assert 'spitzer' in seen
    assert 'earth' in seen

    from bagle import model_fitter_jax
    jax_fit = model_fitter_jax.MicrolensSolver(
        data,
        model_fitter_jax.mmodel.PSPL_PhotAstrom_Par_Param1,
        outputfiles_basename='/tmp/bagle_multi_loc_jax_',
        verbose=False,
    )
    assert list(jax_fit.fitter_param_names) == list(
        numpy_fit.fitter_param_names
    )
    seen.clear()
    ln_jax = float(jax_fit.evaluate_loglik_jax(cube))
    assert 'spitzer' in seen
    assert np.isfinite(ln_numpy) and np.isfinite(ln_jax)
    np.testing.assert_allclose(ln_jax, ln_numpy, rtol=1e-5, atol=1e-4)
    return None


def test_legacy_fits_copies_unsuffixed_origin():
    """An old chain's single xS0 and pi_ref_frame fill every filter."""
    table = Table()
    table['xS0_E'] = np.array([0.01, 0.02])
    table['xS0_N'] = np.array([-0.03, -0.04])
    table['pi_ref_frame'] = np.array([0.2, 0.3])
    table['t0'] = np.array([57000.0, 57001.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = adapt_legacy_filter_columns(table, 3)
    assert any('xS0_E' in str(item.message) for item in caught)
    for suffix in (1, 2, 3):
        np.testing.assert_array_equal(out[f'xS0_E{suffix}'], table['xS0_E'])
        np.testing.assert_array_equal(out[f'xS0_N{suffix}'], table['xS0_N'])
        np.testing.assert_array_equal(
            out[f'pi_ref_frame{suffix}'], table['pi_ref_frame']
        )
    # The legacy column is kept, not overwritten with zero.
    np.testing.assert_array_equal(out['xS0_E'], np.array([0.01, 0.02]))
    return None


def test_legacy_loader_on_fitter(tmp_path):
    """load_mnest_results copies unsuffixed columns for this model."""
    data = make_data(['ogle', 'keck'], ['ogle', 'keck'])
    outroot = str(tmp_path / 'chain')
    fitter = model_fitter.MicrolensSolver(
        data,
        model.PSPL_PhotAstrom_noPar_Param1,
        outputfiles_basename=outroot,
        verbose=False,
    )
    # A pre-written FITS file, so the loader does not rebuild from txt.
    table = Table()
    table['weights'] = np.array([1.0])
    table['logLike'] = np.array([0.0])
    table['xS0_E'] = np.array([0.11])
    table['xS0_N'] = np.array([-0.22])
    table.write(outroot + '.fits', overwrite=True)
    loaded = fitter.load_mnest_results(remake_fits=False)
    assert loaded['xS0_E1'][0] == pytest.approx(0.11)
    assert loaded['xS0_E2'][0] == pytest.approx(0.11)
    assert loaded['xS0_N1'][0] == pytest.approx(-0.22)
    assert loaded['xS0_N2'][0] == pytest.approx(-0.22)
    return None


def test_both_plain_and_suffixed_origin_raises():
    """A holed cube cannot also carry the unsuffixed legacy name."""
    table = Table()
    table['xS0_E'] = np.array([0.1])
    table['xS0_E1'] = np.array([0.2])
    with pytest.raises(ValueError):
        adapt_legacy_filter_columns(table, 2)
    return None


def test_get_model_accepts_unsuffixed_origin():
    """A fake-data dict keyed by ``xS0_E`` fills every filter slot."""
    data = make_data(['ogle', 'keck'], ['ogle', 'keck'])
    fitter = _solver(data, model.PSPL_PhotAstrom_noPar_Param1)
    params = {}
    for name in fitter.fitter_param_names:
        if name.startswith('xS0_'):
            continue
        median = fitter.priors[name].ppf(0.5)
        params[name] = 0.0 if not np.isfinite(median) else float(median)

    # One historical origin is copied into both joint filters.
    params['xS0_E'] = 0.01
    params['xS0_N'] = -0.02
    mod = fitter.get_model(params)
    np.testing.assert_allclose(mod.xS0[:, 0], 0.01)
    np.testing.assert_allclose(mod.xS0[:, 1], -0.02)
    return None


def test_sim_string_is_not_a_catalog_list():
    """The historical ``phot_data='sim'`` label still builds a fitter."""
    data = make_data(['ogle'], ['ogle'])
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'
    fitter = _solver(data, model.PSPL_PhotAstrom_noPar_Param1)
    assert fitter.filt_names == ['phot1']
    assert 'xS0_E1' in fitter.fitter_param_names
    return None


def test_refpar_jax_likelihood_uses_per_filter_pi_ref(monkeypatch):
    """NumPy and JAX log-likelihoods both apply each filter's offset."""
    def _fake(ra, dec, mjd, obsLocation='earth'):
        times = np.atleast_1d(np.asarray(mjd, dtype=float))
        if str(obsLocation) == 'spitzer':
            east, north = 0.4, -0.2
        else:
            east, north = 0.05, 0.01
        return np.column_stack([
            np.full(times.size, east),
            np.full(times.size, north),
        ])

    monkeypatch.setattr(
        'bagle.parallax.parallax_in_direction', _fake
    )
    data = make_data(
        ['ogle', 'spitzer'],
        ['ogle', 'spitzer'],
        obs={'ogle': 'earth', 'spitzer': 'spitzer'},
    )
    numpy_fit = _solver(data, model.PSPL_PhotAstrom_RefPar_Param3)
    names = list(numpy_fit.fitter_param_names)
    assert 'pi_ref_frame1' in names
    assert 'pi_ref_frame2' in names
    # Prior medians put piE at 0, which makes thetaE_hat undefined.
    # Use a physical point, with a different offset on each filter.
    defaults = {
        't0': 57000.0,
        'u0_amp': 0.05,
        'tE': 30.0,
        'log10_thetaE': float(np.log10(0.8)),
        'piS': 0.15,
        'piE_E': 0.01,
        'piE_N': 0.02,
        'xS0_E': 0.0,
        'xS0_N': 0.0,
        'muS_E': 0.0,
        'muS_N': 0.0,
        'b_sff': 0.8,
        'mag_base': 19.0,
    }
    cube = []
    for name in names:
        if name == 'pi_ref_frame1':
            cube.append(0.4)
        elif name == 'pi_ref_frame2':
            cube.append(-0.7)
        else:
            cube.append(defaults[name.rstrip('123456789')])
    cube = np.asarray(cube, dtype=float)
    ln_numpy = float(numpy_fit.log_likely(cube))

    from bagle import model_fitter_jax
    jax_fit = model_fitter_jax.MicrolensSolver(
        data,
        model_fitter_jax.mmodel.PSPL_PhotAstrom_RefPar_Param3,
        outputfiles_basename='/tmp/bagle_refpar_jax_',
        verbose=False,
    )
    assert list(jax_fit.fitter_param_names) == names
    assert list(jax_fit.obs_locations) == ['earth', 'spitzer']
    ln_jax = float(jax_fit.evaluate_loglik_jax(cube))
    assert np.isfinite(ln_numpy) and np.isfinite(ln_jax)
    np.testing.assert_allclose(ln_jax, ln_numpy, rtol=1e-5, atol=1e-4)

    # Filter 2's offset is in the likelihood, not ignored.
    shifted = cube.copy()
    shifted[names.index('pi_ref_frame2')] = 1.5
    ln_shift = float(numpy_fit.log_likely(shifted))
    ln_shift_jax = float(jax_fit.evaluate_loglik_jax(shifted))
    assert abs(ln_shift - ln_numpy) > 1.0
    np.testing.assert_allclose(
        ln_shift_jax, ln_shift, rtol=1e-5, atol=1e-4
    )
    return None


def _physical_cube(fitter):
    """Sampled values that keep piE and the GP dictionaries finite.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver whose ``fitter_param_names`` set the cube order.

    Returns
    -------
    values : list of float
        One value per sampled name.
    """
    defaults = {
        't0': 57000.0,
        'u0_amp': 0.1,
        'tE': 30.0,
        'thetaE': 0.8,
        'log10_thetaE': float(np.log10(0.8)),
        'piS': 0.15,
        'piE_E': 0.05,
        'piE_N': 0.05,
        'xS0_E': 0.0,
        'xS0_N': 0.0,
        'muS_E': 0.0,
        'muS_N': 0.0,
        'b_sff': 0.9,
        'mag_src': 19.0,
        'mag_base': 19.0,
        'gp_log_sigma': -1.0,
        'gp_log_rho': 0.5,
        'gp_log_S0': -2.0,
        'gp_log_omega0': 0.0,
        'gp_rho': 1.5,
        'gp_log_omega04_S0': -6.0,
        'gp_log_omega0_S0': -2.0,
        'gp_log_jit_sigma': -4.0,
        'pi_ref_frame': 0.0,
        'mL': 1.0,
        'beta': 0.4,
        'dL': 4000.0,
        'dL_dS': 0.5,
        'muL_E': 1.0,
        'muL_N': -1.0,
    }
    values = []
    for name in fitter.fitter_param_names:
        base = name.rstrip('123456789')
        if base in defaults:
            values.append(defaults[base])
            continue
        raw = fitter.priors[name].ppf(0.5)
        value = float(np.asarray(raw).ravel()[0])
        if not np.isfinite(value):
            value = 1.0
        values.append(value)
    return values


def _ctypes_cube(values, n_params):
    """A PyMultiNest-style pointer plus the backing buffer.

    Parameters
    ----------
    values : sequence of float
        Sampled parameters. Derived slots stay zero.
    n_params : int
        Full MultiNest length, including derived parameters.

    Returns
    -------
    pointer : ctypes.POINTER(ctypes.c_double)
        What ``pymultinest`` passes into the likelihood.
    buffer : ctypes.Array
        The allocation ``pointer`` refers to. Keep this alive.
    """
    import ctypes

    buf = (ctypes.c_double * int(n_params))()
    for i, value in enumerate(values):
        buf[i] = float(value)
    pointer = ctypes.cast(buf, ctypes.POINTER(ctypes.c_double))
    return pointer, buf


def test_get_model_accepts_ctypes_multinest_cube():
    """A ctypes cube builds the model and receives derived parameters."""
    import ctypes

    data = make_data(['ogle'], ['ogle'])
    fitter = _solver(data, model.PSPL_PhotAstrom_noPar_Param1)
    sampled = _physical_cube(fitter)
    pointer, buf = _ctypes_cube(sampled, fitter.n_params)
    # This is the TypeError on 3ecd079: LP_c_double has no len().
    with pytest.raises(TypeError):
        len(pointer)
    assert isinstance(pointer, ctypes._Pointer)

    mod = fitter.get_model(pointer)
    assert np.isfinite(mod.t0)
    written = [pointer[fitter.n_dims + i]
               for i in range(len(fitter.additional_param_names))]
    assert any(np.isfinite(value) and value != 0.0 for value in written)

    # A sampled-only list is not extended.
    short = list(sampled)
    fitter.get_model(short)
    assert len(short) == fitter.n_dims

    lnL = float(fitter.log_likely(pointer))
    assert np.isfinite(lnL)

    from bagle import model_fitter_jax
    jax_fit = model_fitter_jax.MicrolensSolver(
        data,
        model_fitter_jax.mmodel.PSPL_PhotAstrom_noPar_Param1,
        outputfiles_basename='/tmp/bagle_ctypes_jax_',
        verbose=False,
    )
    jax_pointer, _jax_buf = _ctypes_cube(
        _physical_cube(jax_fit), jax_fit.n_params
    )
    jax_mod = jax_fit.get_model(jax_pointer)
    assert np.isfinite(jax_mod.t0)
    ln_jax = float(jax_fit.log_likely(jax_pointer))
    assert np.isfinite(ln_jax)
    np.testing.assert_allclose(ln_jax, lnL, rtol=1e-5, atol=1e-4)
    return None


def test_gp_get_model_passes_optional_dicts():
    """GP hyperparameters reach the constructor as per-filter dicts."""
    cases = [
        (
            ['ogle'],
            [],
            model.PSPL_Phot_Par_GP_Param2,
            'PSPL_Phot_Par_GP_Param2',
        ),
        (
            ['ogle'],
            ['ogle'],
            model.PSPL_PhotAstrom_Par_GP_Param2,
            'PSPL_PhotAstrom_Par_GP_Param2',
        ),
    ]
    from bagle import model_fitter_jax

    for phot, ast, numpy_cls, jax_name in cases:
        data = make_data(phot, ast)
        fitter = _solver(data, numpy_cls)
        assert any(name.startswith('gp_') for name in fitter.fitter_param_names)
        pointer, _buf = _ctypes_cube(
            _physical_cube(fitter), fitter.n_params
        )
        mod = fitter.get_model(pointer)
        assert isinstance(mod.gp_log_sigma, dict)
        assert 0 in mod.gp_log_sigma
        assert list(mod.use_gp_phot) == [True]
        lnL = float(fitter.log_likely(pointer))
        assert np.isfinite(lnL)

        jax_cls = getattr(model_fitter_jax.mmodel, jax_name)
        jax_fit = model_fitter_jax.MicrolensSolver(
            data,
            jax_cls,
            outputfiles_basename='/tmp/bagle_gp_jax_',
            verbose=False,
        )
        jax_pointer, _jax_buf = _ctypes_cube(
            _physical_cube(jax_fit), jax_fit.n_params
        )
        jax_mod = jax_fit.get_model(jax_pointer)
        assert isinstance(jax_mod.gp_log_sigma, dict)
        assert 0 in jax_mod.gp_log_sigma
        ln_jax = float(jax_fit.log_likely(jax_pointer))
        assert np.isfinite(ln_jax)
    return None


def test_auto_observer_names():
    """Catalog auto-mapping sends Spitzer off Earth and leaves OGLE."""
    from bagle.data import _auto_observer

    assert _auto_observer('Ch1_Spitzer') == 'spitzer'
    assert _auto_observer('I_OGLE') == 'earth'
    assert _auto_observer('Kp_Keck') == 'earth'
    assert _auto_observer('HST_F814W') == 'earth'
    assert _auto_observer('MOA') == 'earth'
    assert _auto_observer('KMT_DIA') == 'earth'
    return None
