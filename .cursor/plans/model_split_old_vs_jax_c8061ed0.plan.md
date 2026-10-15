# Model split: old (NumPy) vs JAX

## Phase 3 — PSPL Phot astrometry parity

PSPL_Phot classes expose `get_astrometry` in **Einstein-radii** coordinates (not arcsec PhotAstrom). The JAX dispatch path must match the NumPy reference, not the PhotAstrom reduced/physical path.

### Test pattern (PSPL_Phot)

- **`tests/test_model.py`**: `test_pspl_phot_get_astrometry_matches_evaluate_jax` — parametrized over `PSPL_Phot_noPar_Param1`, `PSPL_Phot_Par_Param1`; builds instances via `model_old_vs_jax_fixtures.build_paired_instances` / canonical init; compares `bagle.model.get_astrometry` to `bagle.jax.evaluate.try_get_astrometry` (`evaluate_astrometry_jax`).
- **`tests/test_model_jax.py`**: `test_pspl_phot_model_jax_get_astrometry_matches_numpy` — same classes; compares `bagle.model` reference to `model_jax.get_astrometry` on paired instances.
- **`evaluate_astrometry_jax`** must implement `pspl_phot_static` / `pspl_phot_log` via `pspl_phot_astrometry` in `jax_physics.py` (amp-weighted +/- images + blend flux), independent of `likelihood_mode == "phot"`.

Run:

```bash
zsh -lic 'cd /Users/jlu/code/python/bagle && PYTHONPATH=src python -m pytest tests/test_model.py -k "phot and astrom" tests/test_model_jax.py -k "phot and astrom" --tb=short -q'
```
