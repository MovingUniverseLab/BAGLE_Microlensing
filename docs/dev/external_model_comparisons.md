# External model comparisons: VBM, pyLIMA, and MulensModel

**Audit date:** 2026-11-19.
**Tree:** `jax_pymc` at `fb10d81`.

This is a pytest coverage audit. A cell is **Yes** only when an active test
builds that variant's model class and compares it to the external code.
`bagle.model` is the NumPy variant. `bagle.model_jax` is the JAX variant.
A skipped test is not coverage. A comparison of a kernel (`bagle.jax.fspl`,
`bagle.jax.fsbl_contour`, `bagle.fsbl_contour`) is not a model-class
comparison and is called out in the notes.

FSBL and BFSBL have no classes in `model.py`. GP classes have no comparison
to any of the three codes.

## Coverage

| Family | VBM, NumPy | VBM, JAX | pyLIMA, NumPy | pyLIMA, JAX | MulensModel, NumPy | MulensModel, JAX |
|---|---|---|---|---|---|---|
| PSPL Phot | No | Yes, phot only | Yes, no parallax. Parallax test is skipped | Same as NumPy | Yes, phot and geocentric projection. The geoproj test is skipped | Geoproj test exists and is skipped |
| PSPL PhotAstrom | Yes, centroid on the Lee et al. (2010) track | No | No | No | No | No |
| PSBL Phot | Yes, static and Keplerian | Yes, static phot only | Yes | Yes | Yes | No |
| PSBL PhotAstrom | Yes, static input frames and Keplerian orbits | Yes, static input frames only | No | No | No | No |
| BSPL Phot | No | Yes | No | No | No | No |
| BSPL PhotAstrom | Yes, elliptical orbits, flux-weighted point-source | Same test on both | No | No | No | No |
| BSBL Phot | No | Yes | No | No | No | No |
| BSBL PhotAstrom | No | No | No | No | No | No |
| FSPL Phot | Yes, with and without parallax | Yes, phot only. The limb-darkened check is the JAX kernel, not the model class | No | No | No | No |
| FSPL PhotAstrom | Yes, including parallax, resolved images, and limb darkening | No class-level test. The limb-darkened JAX kernel is checked against a quadrature, and only the NumPy model is asserted against VBM | No | No | No | No |
| BFSPL Phot | No NumPy class | Yes | No | No | No | No |
| BFSPL PhotAstrom | No NumPy class | Yes, amplification for the registered classes | No | No | No | No |
| FSBL Phot | No model class. The NumPy contour module is checked | Yes | No | No | No | No |
| FSBL PhotAstrom | No model class. The NumPy contour covers uniform and limb-darkened images | Yes: static frames, phot-only, and the class sweep in `test_fsbl_vbm_parity`. The JAX contour kernel is also checked | No | No | No | No |
| BFSBL Phot | No NumPy class | Yes | No | No | No | No |
| BFSBL PhotAstrom | No NumPy class | Yes, amplification only, in the same class sweep | No | No | No | No |
| PSTL Phot | Yes | Yes | No | No | No | No |
| PSTL PhotAstrom | Yes | Yes, plus image count and positions | No | No | No | No |
| GP (any family) | No | No | No | No | No | No |

Both variants have an active comparison for:

- PSBL Phot against VBM and against pyLIMA.
- PSBL static PhotAstrom against VBM. Keplerian PhotAstrom is NumPy only.
- BSPL elliptical-orbit PhotAstrom Param1 and Param2 against VBM. VBM has
  no binary-source Keplerian orbit, so the check is a flux-weighted
  `PSPLMag` on the BAGLE source-lens separation.
- FSPL Phot, no parallax, against VBM.
- PSTL Phot and PhotAstrom against VBM.

PSPL photometry against pyLIMA is on both variants, but the parallax case
is skipped on both. PSPL and PSBL against MulensModel are NumPy only. The
JAX copy of the geocentric-projection test is skipped.

Everything else is one-sided, kernel-only, or absent. `BSBL_Phot_noPar_Param1`
exists in `model.py` and has no NumPy VBM test.

## Where the tests live

- `tests/test_phot_only_vs_vbm.py` builds JAX classes:
  `PSPL_Phot_noPar_Param1`, `PSBL_Phot_noPar_Param1`,
  `PSTL_Phot_noPar_Param1`, `BSPL_Phot_noPar_Param1`,
  `BSBL_Phot_noPar_Param1`, `FSPL_Phot_noPar_Param1`,
  `FSBL_Phot_noPar_Param1`, `BFSPL_Phot_noPar_Param1`, and
  `PSTL_PhotAstrom_noPar_Param1`.
- `tests/test_model_vs_vbm.py` builds NumPy FSPL photometry (no parallax
  and with parallax), FSPL PhotAstrom, resolved FSPL images, and the PSPL
  PhotAstrom Lee et al. (2010) centroid
  (`test_pspl_lee10_fig1_njx_vs_vbm`).
  `test_fspl_kernel_phot_vs_vbm` and
  `test_fspl_kernel_astrom_uhat_vs_vbm` take the source-lens separation
  from a NumPy model and compare the JAX `fspl_magnification` kernel to
  VBM. `test_fspl_limb_njx_jax_vs_quad_and_vbm` builds the NumPy
  `FSPL_Limb_PhotAstrom_noPar_Param2` class and compares that class and
  the JAX `fspl_magnification_centroid` kernel to a quadrature. The
  assertion against VBM (3e-4) is on the NumPy amplification and centroid
  shift. It does not construct `model_jax.FSPL_Limb_PhotAstrom_*`.
  `test_fspl_limb_jax_likelihood_vs_njx` compares the traced JAX
  likelihood to the NumPy forward model, not to VBM.
- `tests/test_fsbl_vs_vbm.py` compares the NumPy contour module
  `bagle.fsbl_contour` to VBM `BinaryMag2` (point source, finite source,
  off-caustic, limb darkening). There is no `model.FSBL_*` class.
  `test_bfspl_matches_two_fspl_sources` is an internal BAGLE check, not
  a VBM comparison.
- `tests/test_fsbl_jax_contour.py` `test_uniform_vs_vbm` compares the JAX
  kernel `fsbl_magnification_centroid` to `BinaryMag2`.
- `tests/test_binary_frames_vs_vbm.py` parametrizes both `bagle.model` and
  `bagle.model_jax`. PSBL PhotAstrom cases are noPar Param1, Param2, and
  Param4, and Par Param5 and Param7. FSBL PhotAstrom cases are noPar
  Param4 and Par Param5 and Param7. A missing class is skipped, so the
  three FSBL cases run on JAX only. Tolerance is 1e-3 for point source
  and 5e-3 for FSBL.
- `tests/test_binary_orbit_frames.py` uses VBM for Keplerian orbits.
  PSBL Phot EllOrbs Param1 (prograde and retrograde), CircOrbs Param1,
  PhotAstrom EllOrbs Param1, and PhotAstrom CircOrbs Param1 are built
  from `bagle.model` only. BSPL EllOrbs Param1 and Param2 loop both
  modules. CircOrbs is not in that BSPL VBM check.
- `tests/test_get_vbm_model.py` uses `bagle.model` for a PSPL parameter
  list (not a light curve), PSTL photometry, PSTL PhotAstrom, PSBL
  photometry, PSBL EllOrbs photometry, and PSBL EllOrbs PhotAstrom. The
  FSBL tests import `bagle.model_jax`: `FSBL_Phot_noPar_Param1` is a
  light-curve comparison, and `FSBL_Phot_noPar_EllOrbs_Param1` checks
  parameter wiring. BAGLE's own FSBL Keplerian magnification is not
  asserted against the point-source VBM curve.
- `tests/test_pstl_image_count_vbm.py` builds JAX PSTL photometry and
  PhotAstrom and compares image count and positions to VBM
  `MultiImageContours`.
- `tests/test_swap_parity.py` `test_fsbl_vbm_parity` builds `model_jax`
  only, for every FSBL and BFSBL name in the likelihood registry. It
  skips parallax binaries and linear or accelerated orbits. BFSBL
  compares amplification only. FSBL also compares the centroid at 1e-4.
  The slow set is the four limb PhotAstrom Param1 names.
- `tests/test_model.py` and `tests/test_model_jax.py` each contain
  `test_PSBL_phot_vs_pyLIMA`, `test_PSPL_phot_vs_pyLIMA_noparallax`,
  `test_PSPL_phot_vs_pyLIMA_parallax` (skipped), and `test_geoproj_mm`
  (skipped).
- `tests/test_frame_convert.py` imports `bagle.model` only.
  `test_bagle_mulens_psbl_phot_set` compares PSBL photometry to both
  MulensModel and pyLIMA. `test_bagle_mulens_set` compares PSPL
  photometry to MulensModel. There is no `model_jax` comparison in this
  file.

`src/bagle/frame_convert.py` `get_pylima_model` can build pyLIMA PSPL,
FSPL, and PSBL models. That is library code, not a test. The FSPL and
PSBL photometry-plus-astrometry conversion raises.

## Not counted

- `test_PSPL_phot_vs_pyLIMA_parallax` is skipped on both variants
  ("broken: pyLIMA parallax differs slightly (erfa vs. jpl?)").
- `test_geoproj_mm` is skipped on both variants ("package dependency").
  It checks geocentric projection of photometry parameters, not
  PhotAstrom.
- `docs/dev/paper1_figures/fig_parallax_comparison.py` plots PSBL and
  BSPL with parallax against MulensModel and VBM. It is a figure script,
  not a pytest.
