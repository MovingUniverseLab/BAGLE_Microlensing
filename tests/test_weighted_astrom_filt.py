"""Weighted astrometric likelihood must use the mapped filter index."""

import numpy as np
import pytest

import bagle.model as model_np
import bagle.model_fitter as fitter_np
import bagle.model_fitter_jax as fitter_jax
import bagle.model_jax as model_jax


def _phot_astrom_model(model_mod):
    """PSPL phot+astrom model with two filters.

    Filter 0 has source flux fraction 1. Filter 1 does not, so the
    astrometric centroid depends on which filter index is used.

    Parameters
    ----------
    model_mod : module
        ``bagle.model`` or ``bagle.model_jax``.

    Returns
    -------
    mod : PSPL_PhotAstrom_noPar_Param1
        Model with ``b_sff = [1.0, 0.4]``.
    """
    # Source flux fraction: mapped astrometry must not fall back to 1.
    b_sff = np.array([1.0, 0.4])
    mag_src = np.array([19.0, 18.5])
    mod = model_mod.PSPL_PhotAstrom_noPar_Param1(
        10.0, 57000.0, 0.4, 4000.0, 0.5,
        0.0, 0.0,
        0.0, -7.0,
        1.5, -0.5,
        b_sff, mag_src,
    )
    return mod


def _two_filter_data(mod):
    """Photometry in two filters; astrometry paired with the second.

    Parameters
    ----------
    mod : PSPL_PhotAstrom_noPar_Param1
        Model used to place the astrometric observations.

    Returns
    -------
    data : dict
        Fitter data. ``ast_data`` names the second photometric filter,
        so ``map_phot_idx_to_ast_idx`` is ``[1]``.
    """
    t_phot = np.linspace(56800.0, 57200.0, 40)
    t_ast = np.linspace(56900.0, 57100.0, 12)
    # Observations sit on the filter-1 centroid, not filter 0.
    pos = np.asarray(mod.get_astrometry(t_ast, filt_idx=1))
    mag = 18.0 + np.linspace(-0.4, 0.4, t_phot.size)
    mag_err = np.full(t_phot.shape, 0.02)
    ast_err = np.full(t_ast.shape, 1.0e-4)

    data = {
        "target": "filtmap",
        "phot_data": ["I", "Kp"],
        "ast_data": ["Kp"],
        "phot_files": ["I.dat", "Kp.dat"],
        "ast_files": ["Kp.ast"],
        "t_phot1": t_phot,
        "mag1": mag,
        "mag_err1": mag_err,
        "t_phot2": t_phot,
        "mag2": mag + 0.3,
        "mag_err2": mag_err,
        "t_ast1": t_ast,
        "xpos1": pos[:, 0],
        "ypos1": pos[:, 1],
        "xpos_err1": ast_err,
        "ypos_err1": ast_err,
    }
    return data


@pytest.mark.parametrize(
    "fitter_mod, model_mod",
    [
        pytest.param(fitter_np, model_np, id="numpy"),
        pytest.param(fitter_jax, model_jax, id="jax"),
    ],
)
def test_weighted_astrometry_uses_mapped_filter(
    tmp_path, fitter_mod, model_mod,
):
    """Unit-weight astrometric lnL matches the base solver.

    Astrometry is paired with photometric filter 1, whose source flux
    fraction is not 1. The weighted solver must pass that filter index
    rather than the default of 0.
    """
    mod = _phot_astrom_model(model_mod)
    data = _two_filter_data(mod)
    assert mod.b_sff[1] != 1.0

    # Filter 0 and the mapped filter must actually move the centroid.
    t_ast = data["t_ast1"]
    lnL_filt0 = mod.log_likely_astrometry(
        t_ast, data["xpos1"], data["ypos1"],
        data["xpos_err1"], data["ypos_err1"], filt_idx=0,
    )
    lnL_filt1 = mod.log_likely_astrometry(
        t_ast, data["xpos1"], data["ypos1"],
        data["xpos_err1"], data["ypos_err1"], filt_idx=1,
    )
    assert not np.isclose(lnL_filt0, lnL_filt1)

    model_cls = model_mod.PSPL_PhotAstrom_noPar_Param1
    common = dict(
        n_live_points=20,
        max_iter=1,
        dump_callback=None,
        verbose=False,
    )
    base = fitter_mod.MicrolensSolver(
        data, model_cls,
        outputfiles_basename=str(tmp_path / "base_"),
        **common,
    )
    # weights=None is an array of ones: each dataset has unit weight.
    weighted = fitter_mod.MicrolensSolverWeighted(
        data, model_cls,
        outputfiles_basename=str(tmp_path / "weighted_"),
        weights=None,
        **common,
    )

    # Astrometry is the second photometric filter, not index 0.
    assert base.n_phot_sets == 2
    assert base.n_ast_sets == 1
    assert list(base.map_phot_idx_to_ast_idx) == [1]
    assert list(weighted.map_phot_idx_to_ast_idx) == [1]
    np.testing.assert_allclose(weighted.weights, 1.0)

    lnL_base = base.log_likely_astrometry(mod)
    lnL_wgt = weighted.log_likely_astrometry(mod)
    assert lnL_wgt == pytest.approx(lnL_base)
    assert lnL_base == pytest.approx(float(np.sum(lnL_filt1)))

    # Verbose printout still reports the per-dataset weight.
    weighted.verbose = True
    lnL_verbose = weighted.log_likely_astrometry(mod)
    assert lnL_verbose == pytest.approx(lnL_base)
    return None
