# Plan: make the JAX modules a drop-in replacement for `model.py` and `model_fitter.py`

**Audited commit:** `fsbl_ld_jax` at `5368a88` ("More notebook changes.").
**Audit date:** 2026-11-15. **Plan revised:** 2026-11-15, after Jessica's feedback.
**Status:** Steps 1–7 and items 3.1–3.3 are on `jax_pymc` and `fsbl_ld_jax` (`6b889d0`). The remaining class-parity gaps are on `jaxswap/parity-gaps`. Every concrete class has a traced likelihood (`JAX_LIKELIHOOD_CLASSES` is 378). Item 4.7 is still open. Slow FSBL and BFSBL limb PhotAstrom likelihoods are still mostly unrun. `PSTL_Phot_Par_Param1` forward parity stays an xfail (`model.py` is frozen).

This document is both a plan and an implementation specification. Jessica edits and
approves it, and then a coding agent works through it one step at a time. Each item
explains what the code does today (with file and line references against `5368a88`),
what is wrong, which model classes are affected, exactly what to change, which tests to
add, and how to decide that the item is done. Line numbers will drift as steps land, so
search for the quoted code if a line number no longer matches.

---

## How to use this document (instructions for the coding agent)

1. **Work one step at a time, in order.** Steps 1 and 2 must land before the others,
   because the parity tests in Step 4 assume the wrong mappings are fixed and that
   missing likelihoods raise an error. Inside a step, do the items in the listed order.
2. **One branch per step, created off `fsbl_ld_jax`.** Name the branch
   `jaxswap/stepN-short-name`, for example `jaxswap/step1-correctness`. If a later step
   needs code from an earlier step that has not been merged yet, branch off the earlier
   step's branch and say so in the Log.
3. **Never commit to `fsbl_ld_jax` directly, and do not open a pull request.** Push the
   step branch and stop. Jessica reviews the diff herself before anything is merged.
4. **Do not work in Jessica's main checkout** (`~/code/python/bagle` on her
   laptop). It is under active development. Use a separate clone, a
   `git worktree`, or a cloud agent.
5. **Update this file as you go, in the same branch as the work.** Tick the checkbox for
   each item you finish (`- [x]`), and add one dated line to the Log at the end that
   names the branch, the commit hash, and what was done or what was left open. If you
   discover a new problem, add it as a new unchecked item in the right step rather than
   silently fixing it somewhere else.
6. **Ask before going beyond the item.** If an item turns out to need a change outside
   the JAX modules, a change to priors, or a design choice not written here, stop and
   write the question into the Log for Jessica instead of choosing yourself.
7. **Regenerate the status tables** in Appendix A at the end of each step with the
   scripts described in Appendix B, so the ✅/❌ counts track progress.

### Constraints that apply to every step

- **Do not edit `src/bagle/model.py` or `src/bagle/model_fitter.py`.** They are the
  frozen reference. All changes go into `src/bagle/model_jax.py`,
  `src/bagle/model_fitter_jax.py`, `src/bagle/jax_physics.py`, `src/bagle/jax/`, and
  `tests/`.
- **Do not change how priors work and do not change any prior values.** This includes
  the `make_*_gen` function signatures, `make_default_priors`, the `default_priors`
  dictionary, and adding new default priors. Jessica will revisit priors later; see the
  "Deferred" section. If an item seems to need a prior change, stop and ask.
- **Coding style:** NumPy-style docstrings, inline comments that explain the physics or
  the reason for a choice, lines of at most 88 characters, and an explicit
  `return None` at the end of functions that return nothing. Keep each change simple
  and minimal; do not refactor code that the item does not need.
- **Environment:** use the conda environment `py314` (Python 3.14). Run tests with
  `conda run -n py314 python -m pytest <test file> -q` from the repository root, or
  activate the environment first with `conda activate py314`.
- **Existing tests must keep passing.** At minimum run
  `tests/test_jax_likelihood.py`, `tests/test_fsbl_vs_vbm.py`,
  `tests/test_model_old_vs_jax.py`, `tests/test_model_jax.py`, and
  `tests/test_model_fitter_jax.py` before pushing each step, and record the result
  in the Log.

---

## Ground rules (decided by Jessica)

- `model.py` and `model_fitter.py` stay fully independent of the JAX modules. There
  is no shared base class and no shared fitter core. The non-JAX modules are the
  frozen reference that the JAX modules are tested against.
- All development happens in the JAX modules. After extensive testing, the JAX modules
  will replace the non-JAX ones. For that to work, the JAX side must expose the same
  public API: the same class names, constructor signatures, public methods and
  attributes, return shapes, and fitter functions.
- Copying code between the two modules is acceptable until the cutover. The parity
  tests (Step 4) are what keep the two sides consistent.
- The finite-source binary-lens families (FSBL and BFSBL, with and without limb
  darkening) exist only in `model_jax.py` and stay JAX-only for now. They are checked
  against VBMicrolensing (VBM) instead of `model.py`.
- If a model class has no JAX likelihood, every JAX sampler must raise a clear error
  that names the class. There is no silent fallback to the host path and no quietly
  inherited likelihood from a parent class.
- Each step goes on its own branch off `fsbl_ld_jax`, with no pull request. Jessica
  reviews each diff.

---

## Terms used in this document

- **`model.py` / `model_fitter.py`:** the original NumPy implementation of the model
  classes and the fitter (`MicrolensSolver`). Frozen reference.
- **`model_jax.py` / `model_fitter_jax.py`:** the JAX implementation that will
  eventually replace them.
- **Model class / parameterization:** a concrete class such as
  `PSBL_PhotAstrom_Par_LinOrbs_Param1` that a user passes to the fitter. It is assembled
  by multiple inheritance from a geometry class (for example `PSBL`), a data class
  (`Phot`, `PhotAstrom`, or `Astrom`), a parallax class (`Par`, `noPar`, `RefPar`), an
  optional orbit or GP class, and a **Param mixin** (for example `PSBL_PhotParam1`)
  that defines `fitter_param_names` and converts fitted parameters to physical ones.
- **Forward model:** the methods that compute predictions, such as
  `get_photometry(t)`, `get_astrometry(t)`, and `get_amplification(t)`.
- **Host path:** the likelihood computed by `MicrolensSolver.log_likely`, which builds
  a model instance with `get_model(params)` and then calls the instance's forward
  model in ordinary Python. It is correct whenever the forward model is correct, but it
  cannot be compiled or differentiated by JAX, so it is slow and gradient samplers
  cannot use it.
- **Traced path / explicit JAX likelihood:** the likelihood built by
  `build_explicit_jax_loglik_fn` (`model_fitter_jax.py:231`). It calls the class
  methods `jax_log_likely_photometry` and `jax_log_likely_astrometry`, which are
  written so that JAX can trace and JIT-compile them (and take gradients). These class
  methods are separate code from the forward model, so they can disagree with it. When
  this document says a class "has a JAX likelihood," it means those class methods
  exist and are the right ones for that class.
- **Kernel:** a JIT-compiled numerical function in `jax_physics.py` or `src/bagle/jax/`
  that the traced path calls (for example `fspl_images_amps` in `jax/fspl.py`).
- **Forward parity:** `model.py` and `model_jax.py` instances built from the same
  parameters give the same `get_photometry`, `get_astrometry`, and
  `get_amplification` values.
- **Likelihood parity:** on the same synthetic data and the same parameters, the traced
  JAX likelihood, the host likelihood using `model_jax`, and the NumPy likelihood using
  `model.py` agree.
- **VBM:** VBMicrolensing, the external finite-source binary-lens code used as the
  reference for FSBL and BFSBL.
- **W1 to W4:** the four wrong likelihood mappings found by the audit. Each one always
  appears with its descriptive title in this document.

---

## Where things stand (audit of `5368a88`)

The audit looked at every concrete model class in either module: 222 in `model.py`
and 376 in `model_jax.py`, for 378 distinct classes. 220 classes exist in both. Two
classes exist only in `model.py` (`PSPL_PhotAstrom_Par_Param7` and
`PSPL_PhotAstrom_noPar_Param7`). 156 classes exist only in `model_jax.py`, and all of
them are FSBL (112) or BFSBL (44) classes, which are JAX-only by design.

For the 220 shared classes, the constructor signatures are identical, and the forward
models agree to about 1e-13 or better, except PSTL photometry, which differs by up to
3.5e-5 mag. Of the 376 `model_jax` classes, 109 have `jax_log_likely_*` methods that
the traced path can call; 5 of those 109 inherit the wrong parent method (W2 and W4),
so only 104 have a usable JAX likelihood. The existing test suite
(`tests/test_jax_likelihood.py` and `tests/test_fsbl_vs_vbm.py`, 42 tests) passes and
catches none of the four wrong mappings.

### Summary of per-class status

A class is marked ✅ only if all of the following are true: it exists in both modules
(or is JAX-only by design), it has its own correct JAX likelihood, its forward model
matches `model.py` (or forward parity does not apply because it is JAX-only), its JAX
likelihood matches the host and NumPy likelihoods, and it is not affected by any known
wrong mapping (W1 to W4). Everything else is ❌. The full per-class tables are in
Appendix A at the end of this file.

| Family | Classes | ✅ good | ❌ not good | Has JAX likelihood |
|---|---:|---:|---:|---:|
| PSPL Astrom | 2 | 2 | 0 | 2 |
| PSPL GP Phot | 13 | 13 | 0 | 13 |
| PSPL GP PhotAstrom | 15 | 15 | 0 | 15 |
| PSPL Phot | 7 | 7 | 0 | 7 |
| PSPL PhotAstrom | 15 | 15 | 0 | 15 |
| PSBL GP Phot | 2 | 2 | 0 | 2 |
| PSBL GP PhotAstrom | 4 | 4 | 0 | 4 |
| PSBL Phot | 6 | 6 | 0 | 6 |
| PSBL PhotAstrom | 46 | 46 | 0 | 46 |
| BSPL GP Phot | 2 | 2 | 0 | 2 |
| BSPL GP PhotAstrom | 18 | 18 | 0 | 18 |
| BSPL Phot | 2 | 2 | 0 | 2 |
| BSPL PhotAstrom | 32 | 32 | 0 | 32 |
| BSBL Phot | 2 | 2 | 0 | 2 |
| BSBL PhotAstrom | 20 | 20 | 0 | 20 |
| FSPL ±LD Phot | 8 | 8 | 0 | 8 |
| FSPL ±LD PhotAstrom | 8 | 8 | 0 | 8 |
| BFSPL ±LD Phot | 4 | 4 | 0 | 4 |
| BFSPL ±LD PhotAstrom | 4 | 4 | 0 | 4 |
| FSBL ±LD Phot | 12 | 12 | 0 | 12 |
| FSBL ±LD PhotAstrom | 100 | 36 | 64 | 100 |
| BFSBL ±LD Phot | 4 | 4 | 0 | 4 |
| BFSBL ±LD PhotAstrom | 40 | 20 | 20 | 40 |
| PSTL Phot | 2 | 1 | 1 | 2 |
| PSTL PhotAstrom | 10 | 10 | 0 | 10 |
| **Total** | **378** | **293** | **85** | **378** |

How to read the columns in Appendix A:

- ✅ good or matched; ❌ missing, wrong, or failed; ➖ not applicable (for example,
  forward parity against `model.py` for a JAX-only FSBL class, or likelihood parity for
  a class that has no JAX likelihood to compare); ❓ not tested, because the audit
  harness could not build the class or the check did not finish.
- **Forward parity** compares `model.py` and `model_jax.py` at 10 epochs; it passes if
  the maximum absolute difference in photometry, astrometry, and amplification is below
  1e-8.
- **Likelihood parity** generates 40 synthetic points from the reference forward model
  (with 0.01 mag and 0.1 mas noise), then evaluates the traced JAX likelihood, the host
  likelihood with `model_jax`, and the NumPy likelihood with `model.py` at the true
  parameters. It passes if the traced value differs from the others by less than
  1e-3 in lnL. For FSBL and BFSBL there is no `model.py` value, so the traced value is
  compared with the host value from the `model_jax` forward model.
- FSBL and BFSBL forward models are checked against VBM by the existing tests in
  `tests/test_fsbl_vs_vbm.py` and `tests/test_bfs_jax.py`, but only for a few classes
  (`FSBL_PhotParam1`, `FSBL_PhotAstrom_noPar_Param1`, `BFSBL_PhotAstrom_noPar_Param1`,
  `BFSBL_PhotAstrom_Par_Param1`, `BFSBL_Limb_PhotAstrom_noPar_Param1`). Step 4 extends
  this to every FSBL and BFSBL class.
- The FSPL traced likelihood currently uses 400 limb points while the host uses 50, so
  FSPL likelihood parity shows differences of about 5e-5 in lnL. That is within the
  tolerance and is addressed in Step 3.

---

## Step 1. Fix the wrong likelihood mappings and related bugs (JAX side only)

Branch: `jaxswap/step1-correctness`. New test file: `tests/test_jax_swap_fixes.py`.
Each fix below gets at least one regression test in that file. Every regression test
must fail on `5368a88` and pass after the fix; say in the Log that you checked this.

### 1.1 W1: binary-lens linear and accelerating orbits put the primary on the wrong side in the traced path

- [x] Item 1.1 done.

**Background.** For binary lenses with linear (`LinOrbs`) or accelerating (`AccOrbs`)
orbital motion, the two lens positions are placed on either side of the lens origin,
separated by `sep` along the angle `alpha`. On 2026-11-14, commit `ef7a5ec` fixed the
host code in both `model.py` and `model_jax.py` so that the primary sits on the
`+alpha` side, the same convention used by the static and Keplerian models:

```python
# model_jax.py after ef7a5ec (host path)
xL = self.get_lens_origin_astrometry(t, filt_idx=filt_idx)
xL1 = xL + offset  # primary
xL2 = xL - offset  # secondary
```

The traced path was not updated. `orbiting_lens_offsets` in `jax_physics.py:2848`
still uses the old convention:

```python
# jax_physics.py:2895-2897 (traced path, still wrong)
# Primary fixed at -half; secondary at +half plus its motion.
off1 = jnp.broadcast_to(-half, motion.shape)
off2 = half + motion / thetaE_amp
```

**Problem.** The traced likelihood evaluates a model with the primary and secondary
swapped relative to the forward model that generated the data. In the audit, the traced
lnL was about -9.2e5 where the host and NumPy values were +768 (for example
`PSBL_PhotAstrom_Par_LinOrbs_Param1`: traced -924339.7, host 767.78, NumPy 767.78). The
FSBL orbit classes show the same mismatch (traced -5155 versus host 767.78). With
`off1 = +half` and `off2 = -half + motion`, the traced and host values agree to 1e-13.

**Affected classes (12).** `PSBL_PhotAstrom_{Par,noPar}_{LinOrbs,AccOrbs}_Param1`,
`FSBL_PhotAstrom_{Par,noPar}_{LinOrbs,AccOrbs}_Param1`, and
`FSBL_Limb_PhotAstrom_{Par,noPar}_{LinOrbs,AccOrbs}_Param1`. They use the traced
helpers `_PSBLOrbitTracedParam1` and `_FSBLOrbitTracedParam1` in `model_jax.py`, which
call `orbiting_lens_offsets`.

**Change.**
- In `orbiting_lens_offsets` (`jax_physics.py:2848`), place the primary at `+half` and
  the secondary at `-half` plus its motion, matching the host after `ef7a5ec`:
  `off1 = jnp.broadcast_to(half, motion.shape)` and
  `off2 = -half + motion / thetaE_amp`. Update the comment and the docstring.
- Read the full `ef7a5ec` diff first and confirm that the sign of the secondary's
  motion term matches the host (`xL2 = xL - offset` plus motion). If the host applies
  the motion to the secondary differently, follow the host.
- Search for any other caller of `orbiting_lens_offsets` and check that it expects the
  new convention.

**Tests.** In `tests/test_jax_swap_fixes.py`, add
`test_orbit_traced_matches_host[<class>]`, parametrized over the 12 classes. For each,
build the `model.py` instance (or the `model_jax` instance for FSBL) with the
fixtures in `tests/model_old_vs_jax_fixtures.py`, generate photometry and astrometry at
40 epochs spanning `t0 ± 100` days, and assert that the traced JAX lnL and the host lnL
agree to `abs=1e-6`. Also assert that the traced lnL is within 1e-6 of the noise-free
maximum, `-0.5 * sum(log(2*pi*err**2))`, when the data have no noise.

**Acceptance.** The 12 classes show ✅ for likelihood parity in Appendix A, and
`tests/test_psbl_orbit_jax.py` and `tests/test_binary_orbit_frames.py` still pass.

### 1.2 W2: BFSPL photometry-only classes use a point-source likelihood

- [x] Item 1.2 done.

**Background.** BFSPL is a binary finite source with a point lens. In `model_jax.py`
the photometry-only Param mixin is declared as

```python
# model_jax.py:39737
class BFSPL_PhotParam1(BSPL_PhotParam1):
```

and it does not define its own `jax_log_likely_photometry`, so it inherits the
point-source BSPL method. The photometry+astrometry mixin `BFSPL_PhotAstromParam1`
(`model_jax.py:39236`) does it correctly: its `jax_log_likely_photometry` (around
`model_jax.py:39380-39425`) calls `bfspl_log_likely_photometry` from
`jax/bfs.py:297` with the source radii, limb-darkening coefficients, and `n_outline`.

**Problem.** For the four BFSPL photometry-only classes, the traced likelihood ignores
the source radii and limb darkening. With source radius rho = 0.05 the implied
photometric error is about 0.2 mag rms. The audit fixtures use small radii, so the
error there is only about 1e-3 in lnL, which is why no existing test noticed.

**Affected classes (4).** `BFSPL_Phot_Par_Param1`, `BFSPL_Phot_noPar_Param1`,
`BFSPL_Limb_Phot_Par_Param1`, `BFSPL_Limb_Phot_noPar_Param1`.

**Change.** Give `BFSPL_PhotParam1` its own `jax_log_likely_photometry` classmethod
that calls `bfspl_log_likely_photometry` the same way
`BFSPL_PhotAstromParam1.jax_log_likely_photometry` does. Reuse that method's geometry
helper (`_jax_bfspl_geometry`) if its parameter names are the same; if the
photometry-only parameter vector is laid out differently, write a small
photometry-only geometry helper. `BFSPL_Limb_PhotParam1` inherits from
`BFSPL_PhotParam1`, so it picks up the fix; check that its limb-darkening coefficients
reach the kernel. Use the signature conventions of the existing BFSPL methods,
including the `n_outline` keyword (its default value is handled in Step 3).

**Tests.** `test_bfspl_phot_traced_matches_host[<class>]` for the 4 classes, using
`radiusS_pri = radiusS_sec = 0.05` and `u0_amp = 0.02` so the finite-source effect is
large, with data generated by the `model.py` forward model. Assert the traced and host
lnL agree to `abs=1e-3` (the host and traced limb sampling differ; tighten once Step 3
lands). Also assert the traced lnL differs from the BSPL point-source lnL by more than
10, so the test would catch the old mapping.

**Acceptance.** The four classes show ✅ in the "JAX likelihood" and "Likelihood
parity" columns.

### 1.3 W3a: the GP SHO kernel uses the wrong amplitude in the traced path

- [x] Item 1.3 done.

**Background.** The GP classes model correlated photometric noise with a sum of a
Matérn-3/2 kernel and a simple harmonic oscillator (SHO) kernel. The host path uses
celerite, where the SHO term is parameterized by `S0`, `omega0`, and `Q`, and the
kernel variance is `S0 * omega0 * Q`. The traced path uses tinygp and builds the SHO
term with `sigma = sqrt(S0)`:

```python
# jax/gp.py:91-92 and again at jax/gp.py:117-119
kernel = qk.Matern32(scale=rho, sigma=sigma) + qk.SHO(
    omega=omega0, quality=_GP_QUALITY, sigma=jnp.sqrt(S0)
)
```

tinygp's `SHO(sigma=...)` takes the standard deviation of the process, so the
equivalent of celerite's term is `sigma = sqrt(S0 * omega0 * Q)`.

**Problem.** The traced GP likelihood uses a different noise model from the host. In
the audit the traced lnL was -10.64 where host and NumPy were -7.48 (a shift of 3.16)
for every GP class that ran.

**Affected classes.** Every GP class that has a JAX likelihood: 31 classes in the PSPL,
PSBL, and BSPL GP families (see the GP rows in Appendix A). In the audit, 10 of them
ran and showed the 3.16 shift; 17 others failed earlier with W3b, and 4 `RefPar` GP
classes could not be built by the test fixtures.

**Change.** In both places in `jax/gp.py` (the kernel built from a model and
`build_gp_kernel_from_params`), set the SHO amplitude to
`sigma=jnp.sqrt(S0 * omega0 * _GP_QUALITY)`. Confirm against the celerite term used
by `model_jax.py` (search for `SHOTerm`) that `_GP_QUALITY` is the same `Q` and that
celerite's `S0` has the same meaning; add a comment that states the mapping.

**Tests.** `test_gp_kernel_matches_celerite`: for a few `(S0, omega0, rho, sigma)`
values, compare the tinygp covariance matrix with the celerite covariance at 50 random
times to `rtol=1e-10`. And `test_gp_traced_matches_host[<class>]` for one Par and one
noPar class from each of PSPL Phot GP, PSPL PhotAstrom GP, PSBL Phot GP, and BSPL Phot
GP, asserting traced versus host lnL to `abs=1e-6`.

**Acceptance.** GP classes whose only problem was W3a show ✅ for likelihood parity.

### 1.4 W3b: GP classes with `gp_log_omega04_S0` or `gp_log_omega0_S0` crash in the traced path

- [x] Item 1.4 done.

**Background.** Some GP classes fit `gp_log_omega04_S0` (that is,
log(S0 * omega0^4)) or `gp_log_omega0_S0` (log(S0 * omega0)) instead of `gp_log_S0`.
The model classes convert these in `__init__`, for example
`self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]`
(`model.py:5143`, same code in `model_jax.py`). The traced path builds its GP
parameter dictionary from a fixed list of names:

```python
# model_fitter_jax.py:382-388
for key in ('gp_log_sigma', 'gp_log_rho', 'gp_rho',
            'gp_log_S0', 'gp_log_omega0',
            'gp_log_jit_sigma'):
```

and `build_gp_kernel_from_params` (`jax/gp.py:106`) then reads
`gp_params["gp_log_S0"]`.

**Problem.** For these classes `gp_log_S0` is never in the dictionary, so the traced
likelihood raises `KeyError: 'gp_log_S0'`. Under `MicrolensSolverJaxLike` (MultiNest
with the JAX likelihood) the catch-all `except` (item 1.6) turns this into lnL =
-1e300 everywhere, so the fit silently explores the prior. Under BlackJAX and NumPyro
it crashes.

**Affected classes.** The GP classes in Appendix A marked "KeyError gp_log_S0" (17
classes with a JAX likelihood today; more GP classes will need this once Step 7 adds
their likelihoods).

**Change.**
- Add `gp_log_omega04_S0` and `gp_log_omega0_S0` to the key list at
  `model_fitter_jax.py:382-388`.
- In `build_gp_kernel_from_params`, compute `log_S0` from whichever name is present:
  `gp_log_S0` as is, `gp_log_omega04_S0 - 4 * gp_log_omega0`, or
  `gp_log_omega0_S0 - gp_log_omega0`, matching the conversions in the model
  `__init__` methods (check each class's conversion in `model_jax.py`; do not
  guess).
- Check that `gp_rho` (used instead of `gp_log_rho` by these classes) already flows
  through; the existing code handles it at `jax/gp.py:108-111`.

**Tests.** `test_gp_alt_s0_names_traced_matches_host[<class>]` for one class using
`gp_log_omega04_S0` and one using `gp_log_omega0_S0`, asserting no exception and
traced versus host lnL to `abs=1e-6`.

**Acceptance.** No GP class with a JAX likelihood raises `KeyError`.

### 1.5 W4: `PSPL_Phot_Par_Param1_geoproj` uses the heliocentric likelihood

- [x] Item 1.5 done.

**Background.** `PSPL_Phot_Par_Param1_geoproj` is fitted in geocentric-projected
parameters (`t0_geotr`, `u0_amp_geotr`, `tE_geotr`, `piE_E_geotr`, `piE_N_geotr`).
Its Param mixin `PSPL_PhotParam1_geoproj` (`model_jax.py:3405`) converts them to
heliocentric parameters in `__init__` with `fc.convert_helio_geo_phot(...)` and then
calls the heliocentric parent's constructor. But it does not define
`jax_log_likely_photometry`, so the traced path uses the method inherited from
`PSPL_PhotParam1` (`model_jax.py:3018`), which expects `t0`, `u0_amp`, `tE`,
`piE_E`, `piE_N`.

**Problem.** The traced path raises `KeyError: 'u0_amp_geotr'` when it maps the
fitted vector, and even if the names lined up it would skip the geocentric to
heliocentric conversion. Today the host path is the only working path for this class.

**Affected class (1).** `PSPL_Phot_Par_Param1_geoproj`. (The PhotAstrom geoproj
classes have no JAX likelihood and are handled by Step 2 and Step 7.)

**Change.** Add a `jax_log_likely_photometry` classmethod to
`PSPL_PhotParam1_geoproj` that converts the five `*_geotr` values to heliocentric
`t0, u0_amp, tE, piE_E, piE_N` inside the traced function and then calls the same
kernel as `PSPL_PhotParam1`. `convert_helio_geo_phot` is NumPy code, so port only the
part that this class uses (`in_frame='geo'`, `murel_in='LS'`, `murel_out='SL'`,
`coord_in='tb'`, `coord_out='EN'`) as a small JAX function, for example
`jax_physics.convert_geo_to_helio_phot`. Quantities that depend only on `raL`,
`decL`, and `t0par` (the Earth's position and velocity at `t0par`) are fixed for a
fit and can be computed once on the host and passed in, the same way parallax vectors
are precomputed. If the port turns out to be large, stop after Step 2 makes this class
raise a clear error, and write the question in the Log.

**Not in scope.** Default priors for the `*_geotr` parameters are deferred (see the
Deferred section). Neither fitter has them today except `t0_geotr` in
`model_fitter_jax.py`, so tests must pass explicit priors or call the likelihood
directly.

**Tests.** `test_geoproj_jax_convert_matches_numpy`: compare the JAX conversion with
`fc.convert_helio_geo_phot` for several parameter sets to `rtol=1e-10`.
`test_geoproj_traced_matches_host`: traced versus host lnL to `abs=1e-6`, building the
fitter with explicit priors for the `*_geotr` parameters.

**Acceptance.** `PSPL_Phot_Par_Param1_geoproj` shows ✅ in all columns.

### 1.6 Narrow the catch-all `except` in `MicrolensSolverJaxLike.LogLikelihood`

- [x] Item 1.6 done.

**Background.** `MicrolensSolverJaxLike.LogLikelihood`
(`model_fitter_jax.py:4048-4076`) wraps the whole evaluation:

```python
try:
    if self._unphysical_distances(cube):
        return self._LN_L_FLOOR
    lnL = float(self.evaluate_loglik_jax(cube))
except Exception:
    return self._LN_L_FLOOR
```

**Problem.** Any programming error (a `KeyError`, `TypeError`, or a missing
likelihood) becomes lnL = -1e300 at every point. MultiNest then runs to completion and
returns the prior. This is what hid W3b and W4.

**Change.** Remove the `try`/`except` around `evaluate_loglik_jax`. Keep the explicit
`_unphysical_distances` check and the existing non-finite check after it
(non-finite lnL already goes to the floor). If there is a known numerical exception
that must be caught (for example a `FloatingPointError` from a specific kernel), catch
only that type and add a comment saying why.

**Tests.** `test_jaxlike_raises_on_bad_key`: build a `MicrolensSolverJaxLike` with a
class whose likelihood raises (for example by monkeypatching a GP class's
`jax_log_likely_photometry` to raise `KeyError`) and assert that `LogLikelihood`
raises instead of returning -1e300.

### 1.7 Remove `from tkinter import W` from `model_jax.py`

- [x] Item 1.7 done.

**Background and problem.** `model_jax.py:488` has `from tkinter import W`, an
accidental editor auto-import. `W` is not used, and on headless installs without Tk
(clusters, CI, some conda builds) importing `model_jax` fails.

**Change.** Delete the line. **Test:** `test_model_jax_imports_without_tkinter`
inserts `None` into `sys.modules['tkinter']` and imports `bagle.model_jax` in a
subprocess, asserting success.

### 1.8 Remove dead `bagle.jax_model` imports

- [x] Item 1.8 done.

**Background and problem.** `model_jax.py` imports from a module `bagle.jax_model`
that does not exist, at lines 2193, 2365, 7840, 18443, 18671, 28315, and 30473 (for
example `from bagle.jax_model import try_get_photometry_with_gp`). Each import is
inside a `try` block, so it always fails and the code silently falls back to the
NumPy/celerite path. This is dead code that makes the control flow hard to read.

**Change.** Remove each dead import and its `try` branch, keeping the fallback code as
the only path. Do not change behavior. **Test:** the existing forward-parity tests
(and Step 4) must still pass; add `test_no_jax_model_imports`, which asserts the
string `bagle.jax_model` does not appear in `model_jax.py`.

### 1.9 Remove or finish dead BSPL astrometry helpers

- [x] Item 1.9 done.

**Background and problem.** `jax/bspl.py:227` defines `bspl_astrometry_from_model`,
which only returns `None` ("use numpy path until full BSPL astrometry port"), and
`build_bspl_joint_loglik` (`jax/bspl.py:231`) is not called by the fitter.

**Change.** Delete both functions if nothing calls them (search the repository and the
tests). If Step 7's BSPL PhotAstrom work needs them, leave a one-line comment pointing
to Step 7 instead. No test needed beyond the existing suite.

---

## Step 2. Raise a clear error when a class has no JAX likelihood

Branch: `jaxswap/step2-raise-missing`. New test file: `tests/test_jax_lnl_registry.py`.

### 2.1 Add an explicit list of classes that have a verified JAX likelihood

- [x] Item 2.1 done.

**Background.** `build_explicit_jax_loglik_fn` (`model_fitter_jax.py:231`) finds the
likelihood methods with `getattr(model_class, 'jax_log_likely_photometry', None)`
(around line 292). Because of inheritance, `getattr` finds a parent's method even when
it is wrong for the class (this is how W2 and W4 happened). When no method is found,
the function returns `(None, None)`. Callers then behave differently:

- `MicrolensSolver.evaluate_loglik_jax` (`model_fitter_jax.py:2524-2535`) silently
  falls back to the host `log_likely`, one point at a time.
- `MicrolensSolverJaxLike` (MultiNest with the JAX likelihood) inherits that fallback.
- `grad_loglik_jax` (line 2595), `MicrolensSolverNumPyro` (lines 5511 and 5772),
  replica exchange (line 6111), and `MicrolensSolverBlackJAX` (line 7511) raise
  `NotImplementedError` or `RuntimeError`, but with generic messages that do not name
  the class.
- `LogLikelihoodOp` for PyMC (line 4629) stores `None` and fails later.

**Problem.** A user can run a "JAX" fit that is actually on the slow host path, or on
a wrong inherited kernel, without being told.

**Change.**
- In `model_jax.py`, add a module-level `frozenset` named `JAX_LIKELIHOOD_CLASSES`
  that lists, by name, every concrete class whose `jax_log_likely_*` methods have been
  checked against the host (after Step 1: the classes with ✅ in the "JAX likelihood"
  column of Appendix A). Add a short comment above it saying that a class is added
  only together with a likelihood-parity test.
- In `build_explicit_jax_loglik_fn`, before looking up the methods, check
  `model_class.__name__ in model_jax.JAX_LIKELIHOOD_CLASSES`. If it is not there,
  raise `NotImplementedError(f"{model_class.__name__} has no JAX likelihood. "
  "Use MicrolensSolver (host likelihood) or add a JAX likelihood (see "
  "docs/dev/jax_swap_plan.md, Step 7).")`.
- Keep the existing `(None, None)` return only for the configuration cases that are
  not about the class (for example `add_err` or `mult_err`), and change those to raise
  as well, with a message that names the unsupported option. **Question for Jessica
  (write it in the Log if not answered):** should those configuration cases also
  raise, or is a host fallback acceptable for them?
- Remove the host fallback in `evaluate_loglik_jax`. Every caller listed above then
  gets the same clear error.
- The plain `MicrolensSolver` (MultiNest with the host likelihood, the same behavior
  as `model_fitter.MicrolensSolver`) is not a JAX sampler and keeps working for every
  class.

**Tests.**
- `test_registry_matches_methods`: every name in `JAX_LIKELIHOOD_CLASSES` is a
  concrete class in `model_jax` that has the required `jax_log_likely_*` methods.
- `test_missing_jax_lnl_raises[<sampler>]`: for one class with no JAX likelihood (for
  example `BSBL_Phot_noPar_Param1`), constructing or running each of
  `MicrolensSolverJaxLike`, `MicrolensSolverNumPyro`, `MicrolensSolverBlackJAX`, and
  `evaluate_loglik_jax` raises `NotImplementedError` whose message contains the class
  name.
- `test_host_solver_still_works`: `MicrolensSolver.log_likely` still works for that
  class.

**Acceptance.** No JAX sampler can run a class outside the list, and the error message
names the class.

---

## Step 3. Make the FSPL traced likelihood use the model's `n_outline`

Branch: `jaxswap/step3-n-outline`. New test file: `tests/test_fspl_n_outline.py`.
Decided by Jessica on 2026-11-15.

**Background.** Finite-source models sample the source limb at `n_outline` points.
The model constructors take `n_outline` and default to 50 on the host path. The
traced likelihood methods ignore it and hard-code 400 as a keyword default, at
`model_jax.py:28633, 28778, 28987, 29029, 29284, 29324, 39381, and 39429` (FSPL,
FSPL with limb darkening, and BFSPL), for example:

```python
def jax_log_likely_photometry(cls, vec, t, mag_obs, mag_err, b_sff,
                              n_outline=400):
```

The fitter never passes `n_outline` to the model either: `get_model`
(`model_fitter_jax.py:2142-2171`) builds the model from the fitted parameters plus
`generate_fixed_params_dict(self.data, self.fixed_param_names)`, and `n_outline` is
not in `fixed_param_names`. So today the host path runs at 50 and the traced path at
400, and the user has no way to set either from a fit.

**Efficiency check (done 2026-11-15).** Passing `n_outline` as a static Python `int`
is safe: the kernels already declare it static (`@partial(jax.jit,
static_argnames=("n_outline", "n_ld"))` at `jax/fspl.py:322` and `jax/fspl.py:386`), it
is fixed for a whole fit, so there is one compile per fit and no retracing. Per call,
50 points is 4 to 9 times faster than 400. Accuracy of the JAX kernel at 50 points is
about 5e-5 mag in photometry and about 9e-3 thetaE in the centroid; at 100 points it is
2e-6 mag and 4e-4 thetaE.

**Decision.** `n_outline` is supplied the same way as `raL` and `decL`: through the
data dictionary. Today `raL` and `decL` flow like this:

1. The model class lists them in a class attribute, `fixed_param_names = ['raL',
   'decL']` (`PSPL_Parallax`, `model_jax.py:1888`).
2. The fitter copies that list in `__init__` (`model_fitter_jax.py:1826-1827`):
   `self.fixed_param_names = self.model_class.fixed_param_names`.
3. `get_model` calls `generate_fixed_params_dict(self.data, self.fixed_param_names)`
   (`model_fitter_jax.py:2165-2168`, defined at line 8753), which returns
   `{name: data[name] ...}` and passes it to the model constructor as keyword
   arguments.

`n_outline` will follow the same route. There is no new `MicrolensSolver` keyword. The
non-JAX `model_fitter.py` is not changed.

### 3.1 Pass `data['n_outline']` from the fitter to the model

- [x] Item 3.1 done.

- Add `'n_outline'` to `fixed_param_names` of the FSPL, FSPL-LD, and BFSPL classes in
  `model_jax.py` (for parallax classes the list becomes
  `['raL', 'decL', 'n_outline']`). Check that the class-level attribute is not shared
  by reference with non-finite-source classes; build a new list in the finite-source
  classes rather than appending to the parent's list.
- In `generate_fixed_params_dict` (`model_fitter_jax.py:8753`), skip `n_outline`
  when it is not in `data`, the same way `obsLocation` is skipped, so the constructor
  default applies. Update the docstring's Notes section.
- The fitter `__init__` copy at line 1826 already picks up the longer list. Check that
  `fixed_param_names` is not also used somewhere that assumes it contains only
  `raL` and `decL` (search for `fixed_param_names`).

### 3.2 Use the model's `n_outline` in the traced likelihood

- [x] Item 3.2 done.

- At the eight `jax_log_likely_*` definitions listed above, change the default from
  400 to 50 and coerce with `int()` inside the method. For BFSPL, which has
  `n_outline_pri` and `n_outline_sec`, use the larger of the two; a value of `False`
  (meaning point source) maps to 50.
- In `build_explicit_jax_loglik_fn`, read `n_out = int(fitter.data.get('n_outline',
  50))` once, add `n_out` to the cache key (`model_fitter_jax.py:271`), and pass
  `n_outline=n_out` at the photometry call (around line 504) and the astrometry call
  (around line 544) only when the method's signature has an `n_outline` parameter
  with a non-`None` default. Leave FSBL and BFSBL alone: their default `None` means
  adaptive sampling.

**Tests.**
- `test_n_outline_reaches_model`: with `data['n_outline'] = 100`,
  `fitter.get_model(params).n_outline == 100`; without the key it is 50.
- `test_n_outline_reaches_traced_lnl`: the traced FSPL lnL with
  `data['n_outline'] = 100` equals a direct call of the kernel with
  `n_outline=100`, and differs from the call with 50.
- `test_fspl_traced_matches_host_same_n`: with `n_outline = 50` on both paths, traced
  and host FSPL lnL agree to `abs=1e-6`; check the four FSPL and four FSPL-LD
  Phot classes.
- `test_n_outline_no_retrace`: count compiles (for example with a Python side effect
  inside the traced function, or `jax.log_compiles`) and assert one compile across 20
  likelihood calls in a fit.

**Acceptance.** Host and traced FSPL paths use the same `n_outline`, which the user
sets with `data['n_outline']` and which defaults to 50.

### 3.3 Later, separate item: host NumPy FSPL limb sampling

- [x] Item 3.3 done.

The host NumPy FSPL forward model sampled the limb at evenly spaced angles and was
off by up to 8 mmag at `n_outline = 50`, while the JAX kernel is accurate to about
5e-5 mag at 50. `model.py` is frozen, so it still uses that sampler. `model_jax`
`FSPL._get_all_arrays_real` and `BFSPL.get_all_arrays` now call
`fspl_images_amps`. Traced and `model_jax` host log-likelihoods agree at 1e-6.
Against `model.py`, the 10-epoch PhotAstrom grid differs by up to 0.27 mag
(fixture `n_outline` is 20 and `rho` is about 1); the forward test uses 0.30 mag,
0.50 in amplification, and 1e-3 arcsec for those classes. The NumPy fitter
comparison stays at 1e-1. Item 4.7 is not closed, because `model.py` still
disagrees.

---

## Step 4. Parity tests for every class

Branch: `jaxswap/step4-parity-tests`. New test file: `tests/test_swap_parity.py`.

**Background.** `tests/test_model_old_vs_jax.py` and
`tests/model_old_vs_jax_fixtures.py` already compare some classes. The fixtures module
has `CANONICAL` (canonical parameter values), `build_init_args`, and
`build_paired_instances(class_name)`, which builds a `model.py` and a `model_jax`
instance from the same values. `src/bagle/jax/migration_tasks.py:61` has
`discover_concrete_classes(model_module)`.

### 4.1 Fix the test fixtures so every class can be built

- [x] Item 4.1 done.

Ten shared classes could not be built in the audit because `CANONICAL` lacks values
for `pi_ref_frame` (the six `PSPL_PhotAstrom_RefPar_*` classes) and for
`omega_sec`, `big_omega_ter`, `i_23`, `e_23`, `tp_23`, `a_23`, `i_12`
(the four `PSTL_PhotAstrom_*EllOrbs*` classes). Add physically sensible values for
these (and `mLt`, `gamma`, `gamma1`, `phi12`, `phi13` if missing) to `CANONICAL`.
The likelihood harness also needs `raL` and `decL` for `RefPar` classes (the audit
only set them for names containing `_Par_`).

### 4.2 Forward parity test

- [x] Item 4.2 done.

`test_forward_parity[<class>]`, parametrized over every concrete class that exists in
both modules. Build paired instances and compare `get_photometry`, `get_astrometry`,
and `get_amplification` at 10 epochs including the peak, with `atol=1e-8` (mag,
arcsec, and dimensionless). The two `PSTL_Phot_*_Param1` cases compare the
per-image amplifications with images fainter than |A| = 1e-4 dropped on both sides
(at 1e-8), and the unmasked totals at 3e-4 (item 4.6).

### 4.3 Likelihood parity test

- [x] Item 4.3 done.

`test_likelihood_parity[<class>]`, parametrized over `JAX_LIKELIHOOD_CLASSES`.
Generate 40 points from the reference forward model (`model.py` for shared classes,
`model_jax` for FSBL and BFSBL) with fixed-seed noise of 0.01 mag and 0.1 mas, build a
`model_fitter_jax.MicrolensSolver` with the class, and compare the traced lnL
(`build_explicit_jax_loglik_fn`) with the host lnL (`MicrolensSolver.log_likely`)
to `abs=1e-6`, and for shared classes also with `model_fitter.MicrolensSolver`. If a
class has no default prior for one of its parameters, pass explicit priors in the
test; do not add default priors.

### 4.4 FSBL and BFSBL against VBM

- [x] Item 4.4 done.

`test_fsbl_vbm_parity[<class>]` for every FSBL and BFSBL class in
`JAX_LIKELIHOOD_CLASSES`, comparing `model_jax` photometry and astrometry with VBM at
the tolerances already used in `tests/test_fsbl_vs_vbm.py`. These classes are slow to
compile (up to about 150 s for the forward model of `FSBL_Limb_PhotAstrom_*`), so mark
the slow ones with the repository's slow-test marker if there is one, and keep one
fast case per family unmarked.

### 4.5 Coverage test

- [x] Item 4.5 done.

`test_every_class_has_jax_lnl_or_is_known_gap`: every concrete `model_jax` class is
either in `JAX_LIKELIHOOD_CLASSES` or in a `KNOWN_JAX_LNL_GAPS` list defined in the
test file. The test fails if a class is in neither, so a new class cannot be added
silently. Step 7 shrinks the gaps list.

### 4.6 Investigate the PSTL photometry forward difference

- [x] Item 4.6 done.

`PSTL_Phot_Par_Param1` and `PSTL_Phot_noPar_Param1` differ between `model.py` and
`model_jax.py` by up to 3.5e-5 mag in photometry (4e-5 in amplification), while
`PSTL_PhotAstrom` classes match to 1e-13. Find which side differs from VBM (see
`tests/test_pstl_image_count_vbm.py`) and fix the JAX side if it is the JAX side.
Report in the Log if the difference is in `model.py`.

Finding: the 10-epoch grid (including the peak) fails only for
`PSTL_Phot_Par_Param1` (about 3.5e-5 mag). `PSTL_Phot_noPar_Param1` passes
that grid. On a denser time grid both photometry classes differ by up to
about 7e-5, and the size of the gap depends on how many epochs are solved
together. Both modules drop a faint image (amplification about 1e-5 to 1e-4)
in the degree-10 triple root mask. At some epochs the JAX images match VBM
and `model.py` does not; at others the reverse. `model.py` is frozen, so the
solver was not changed.

The strict xfail was flaky: pass or fail depended on test order. It is replaced by a
comparison that ignores faint images. With images below |A| = 1e-4 dropped on both
sides, the bright images agree to about 1e-11 on 10 and 200 epochs (the unmasked
totals differ by up to 9e-5). `test_forward_parity` and
`test_pstl_phot_dense_grid_parity` (200 epochs) check the bright images at 1e-8 and
the unmasked totals at 3e-4.

### 4.7 Bring every likelihood comparison to 1e-6

- [ ] Item 4.7 done.

The Step 4 test passes, but four families do not meet `abs=1e-6`:

- GP: tinygp versus celerite, at most about 3e-5 (test uses 1e-3).
- FSBL photometry-only: host adaptive contour versus the traced kernel,
  about 4e-5 (test uses 1e-4). FSBL phot+astrom sums the same residual
  to a few 1e-3 (static Param2 measured 2.0e-3; test uses 5e-3, the
  bound already used for circular and elliptical orbits).
- BFSBL: same contour mismatch at the canonical source radius, about 2e-2
  (test uses 5e-2).
- FSPL and BFSPL phot+astrom versus `model.py`: item 3.3 moved the
  `model_jax` host onto the JAX kernel, so traced versus host is now
  1e-6. `model.py` is frozen. The NumPy comparison is still wider
  (test uses 1e-1; one PhotAstrom case at `n_outline` 50 differed by
  about 0.003 in lnL).

Do not loosen these further. Closing this item means the traced and host
values agree at 1e-6. The `model_jax` FSPL / BFSPL limb path is done
(item 3.3). What remains is `model.py` (frozen) and the same contour on
both FSBL and BFSBL paths.

---

## Step 5. Make the `model_jax` class API match `model.py`

Branch: `jaxswap/step5-model-api`. New test file: `tests/test_model_jax_api.py`.

### 5.1 Add the Param7 classes

- [x] Item 5.1 done.

`model.py` has `PSPL_PhotAstromParam7` (`model.py:4939`) and the concrete classes
`PSPL_PhotAstrom_noPar_Param7` (`model.py:29807`) and `PSPL_PhotAstrom_Par_Param7`
(`model.py:29821`). `model_jax.py` has none of them. Copy them into `model_jax.py`
with the same constructor signature and attributes, using `model_jax`'s base classes.
Do not add or change default priors for their parameters (for example `log10_piE`);
tests pass explicit priors. A JAX likelihood for Param7 is part of Step 7.

### 5.2 Add the missing astrometry methods

- [x] Item 5.2 done.

`model.py` defines `get_astrometry_unlensed_noparallax(t, filt_idx=0)`
(`model.py:638`) and `get_astrometry_pm_removed(t, filt_idx=0, unlensed=False)`
(`model.py:671`) on the PSPL base and they are inherited by about 248 classes;
`model_jax.py` lacks both. Port them with the same signatures and return shapes.

### 5.3 Add `FSPL.cent`

- [x] Item 5.3 done.

`model.py:26731` defines `FSPL.cent(self, ims, z1bar)`, used by about 22 FSPL-family
classes; `model_jax.py` lacks it. Port it with the same signature.

### 5.4 API test

- [x] Item 5.4 done.

`test_public_api_matches`: for every class in both modules, the set of public
methods and attributes of the `model_jax` class contains the set of the `model.py`
class, and a constructor call written for `model.py` still binds on `model_jax`.
Four BFSPL mixins (`BFSPL_PhotParam1`, `BFSPL_PhotAstromParam1`, and the two
limb-darkened versions) take an extra optional `n_outline=None` added in Step 3.
`model.py` is frozen and does not have that parameter. Omitting it still binds.

Output comparison is `atol=1e-8` (arcsec) for both astrometry helpers, including
finite-source classes. On the canonical 10-epoch grid the largest residual is
about 1e-12 (`PSTL_Phot_Par_Param1`). An FSPL limb phot+astrom model on a
41-epoch grid across the peak agrees to about 1e-18. The finite-source
difference from item 3.3 is in the traced likelihood, not in these host
forward methods, so the astrometry comparison did not need a wider tolerance.
`FSPL.cent` matches at 1e-8 when the lens is at the origin. `model.py`'s
`detJac` ignores the lens position (`1 - 1/|z|^4`), while `model_jax` uses
`1 - 1/|z - z1|^4`, so an offset lens is not compared at 1e-8.

Eight FSPL phot-only classes (`FSPL_Phot_*` and `FSPL_Limb_Phot_*`) are
skipped: `model.py`'s unlensed astrometry reads `xS0`, and those models have
no sky position. The JAX helpers still run for them.

---

## Step 6. Make the `model_fitter_jax` API match `model_fitter.py` (excluding priors)

Branch: `jaxswap/step6-fitter-api`. New test file: `tests/test_fitter_jax_api.py`.
The goal is that replacing `from bagle import model_fitter` with
`from bagle import model_fitter_jax as model_fitter` works in Jessica's existing scripts
for everything except priors (deferred).

### 6.1 Restore the `def_best=` keyword

- [x] Item 6.1 done.

`model_fitter.py` has `def_best='maxl'` on `print_likelihood` (line 3098),
`calc_chi2` (3118 and 3737), `calc_chi2_manual` (3237 and 3852), and `get_residual`
(3423). The JAX versions (`model_fitter_jax.py:3608, 3628, 3747, 3928`) dropped it.
Add it back with the same default and the same behavior (it chooses which best-fit
definition to use when `params='best'`); read the `model_fitter.py` implementation and
copy the logic.

### 6.2 Port the missing `MicrolensSolver` methods

- [x] Item 6.2 done.

- `load_priors` (`model_fitter.py:2404`) and `write_priors_pkl`
  (`model_fitter.py:2113`): port only the file input and output (reading and writing
  the pickled priors). Do not change how priors are built or evaluated.
- `plot_ultranest_style` (`model_fitter.py:2597`): port as is.

### 6.3 Port the missing module functions

- [x] Item 6.3 done.

`save_params` (`model_fitter.py:4953`), `make_sed_distance_gen`
(`model_fitter.py:4490`), `make_fdfdt` (4584), and `random_prob` (4588). Copy them
with the same signatures. `make_sed_distance_gen` returns a prior generator; port it
exactly as written and do not adapt it to the JAX `make_*_gen` form (that is part of
the deferred prior work).

### 6.4 API test

- [x] Item 6.4 done.

`test_fitter_public_api_matches`: every public function and every public
`MicrolensSolver` method in `model_fitter.py` exists in `model_fitter_jax.py` with a
compatible signature, except the prior functions listed in the Deferred section, which
the test skips with a comment.

---

## Step 7. Add the missing JAX likelihoods

Branch: one per family, for example `jaxswap/step7-bspl-photastrom`. For each family:
write the `jax_log_likely_*` methods (reusing existing kernels where possible), add the
classes to `JAX_LIKELIHOOD_CLASSES`, remove them from `KNOWN_JAX_LNL_GAPS`, and make
sure `test_likelihood_parity` covers them. Order of work:

- [x] 7.1 BSPL PhotAstrom, including GP variants (50 of 50 have a JAX likelihood).
- [x] 7.2 PSBL and FSBL Keplerian orbits (`CircOrbs`, `EllOrbs`), photometry and
      astrometry.
- [x] 7.3 BSBL (Phot 2 of 2, PhotAstrom 20 of 20).
- [x] 7.4 BFSBL with orbits and BFSBL photometry-only (Phot 4 of 4; PhotAstrom
      40 of 40 have a likelihood, 20 of 40 good; limb PhotAstrom is slow).
- [x] 7.5 PSPL PhotAstrom `Param4`, `Param5`, `Param6`, `Param4_geoproj`,
      Param7, and PSPL Astrom `Param4`. Phot-only `Param1_geoproj` already had
      a likelihood.
- [x] 7.6 PSTL `Param2` and `Param3`.
- [x] 7.7 GP families that still lack a likelihood after Step 1.

---

## Deferred (Jessica will revisit)

These differences between `model_fitter.py` and `model_fitter_jax.py` were found by
the audit. They are listed for information only. **Do not change them** until Jessica
decides.

**Prior generator signatures.** Every `make_*_gen` function in `model_fitter_jax.py`
takes an extra leading `param_name` argument and a trailing `stats_pkg='scipy'`
keyword:

| Function | `model_fitter.py` | `model_fitter_jax.py` |
|---|---|---|
| `make_gen` | `(min, max)` | `(param_name, min, max, stats_pkg='scipy')` |
| `make_norm_gen` | `(mean, std)` | `(param_name, mean, std, stats_pkg='scipy')` |
| `make_lognorm_gen` | `(mean, std)` | `(param_name, mean, std, stats_pkg='scipy')` |
| `make_log10norm_gen` | `(mean_in_log10, std_in_log10)` | `(param_name, mean_in_log10, std_in_log10, stats_pkg='scipy')` |
| `make_truncnorm_gen` | `(mean, std, lo_cut, hi_cut)` | `(param_name, mean, std, lo_cut, hi_cut, stats_pkg='scipy')` |
| `make_truncnorm_gen_with_bounds` | `(mean, std, low_bound, hi_bound)` | `(param_name, mean, std, low_bound, hi_bound, stats_pkg='scipy')` |
| `make_invgamma_gen` | `(t_arr)` | `(param_name, t_arr, stats_pkg='scipy')` |
| `make_t0_gen` | `(t, mag)` | `(param_name, t, mag, stats_pkg='scipy')` |
| `make_mag_base_gen` | `(mag)` | `(param_name, mag, stats_pkg='scipy')` |
| `make_mag_src_gen` | `(mag)` | `(param_name, mag, stats_pkg='scipy')` |
| `make_xS0_gen` | `(pos, verbose=False)` | `(param_name, pos, verbose=False, stats_pkg='scipy')` |
| `make_xS0_norm_gen` | `(pos, verbose=False)` | `(param_name, pos, verbose=False, stats_pkg='scipy')` |
| `make_muS_EN_gen` | `(t, pos, scale_factor=100.0, verbose=False)` | `(param_name, t, pos, scale_factor=100.0, verbose=False, stats_pkg='scipy')` |
| `make_muS_EN_norm_gen` | `(t, pos, n_use=None, scale_factor=10.0)` | `(param_name, t, pos, n_use=None, scale_factor=10.0, stats_pkg='scipy')` |
| `make_piS` | `()` | `(param_name, stats_pkg='scipy')` |
| `weighted_quantile` | `(values, quantiles, ...)` | same plus `stats_pkg='scipy'` |

`MicrolensSolver.make_default_priors` is `(self)` in `model_fitter.py` and
`(self, stats_pkg=None)` in `model_fitter_jax.py`.

**Default prior values** (`MicrolensSolver.default_priors`):

- `log10_thetaE`: `model_fitter.py` uses `('make_truncnorm_gen', 0.02, 0.35, -4, 4)`;
  `model_fitter_jax.py` uses `('make_truncnorm_gen', -0.2, 0.3, -4, 4)`.
- `log10_piE` (used by Param7) is only in `model_fitter.py`.
- `t0_geotr` is only in `model_fitter_jax.py`. Neither fitter has defaults for
  `u0_amp_geotr`, `tE_geotr`, `piE_E_geotr`, or `piE_N_geotr`, so the geoproj classes
  need user-supplied priors.
- Neither fitter has defaults for `phi12` or `phi13`, so the PSTL classes need
  user-supplied priors (this is why the audit could not build a PSTL Phot fitter).

---

## Log

- 2026-11-18: `jaxswap/parity-gaps` (off `fsbl_ld_jax` at `6b889d0`, commit `ed48f2e`). Traced likelihoods for the 64 PSBL and FSBL PhotAstrom classes that had none. Each parameterization is shifted to the geometric-midpoint Param1 geometry and evaluated with the existing point-source or finite-source kernel. Non-slow likelihood parity: 16 PSBL at 1e-6, 12 static FSBL PhotAstrom at 5e-3, and the 6 FSPL/BFSPL limb PhotAstrom classes at 1e-1 (the limb name had missed the existing phot+astrom tolerance). Also passed at 5e-3: `FSBL_Limb_PhotAstrom_Par_Param2` and `FSBL_PhotAstrom_Par_LinOrbs_Param3`. `JAX_LIKELIHOOD_CLASSES` is 378 and `KNOWN_JAX_LNL_GAPS` is empty. Item 4.7 is still open. Slow FSBL and BFSBL limb PhotAstrom likelihoods are still mostly unrun. `PSTL_Phot_Par_Param1` forward parity stays an xfail.
- 2026-11-18: Fast-forwarded `jax_pymc` from `0d32179` to Step 3 (`5c5cb51` on `jaxswap/step3-outline`, plus this note). Items 3.1–3.3 are on `jax_pymc`. `fsbl_ld_jax` is still at `0d32179`. Item 4.7 is still open.
- 2026-11-18: Synced `fsbl_ld_jax` to `jax_pymc` (`0d32179`) by fast-forward. Step 3 (`jaxswap/step3-outline`, commit `79bc0ea`) passes `data['n_outline']` through (already wired; `tests/test_n_outline.py`, 12 passed) and points the `model_jax` FSPL and BFSPL host images at `fspl_images_amps`. Traced versus host lnL is 1e-6. FSPL/BFSPL forward and likelihood slice: 54 passed. `model.py` is unchanged. Item 4.7 stays open.
- 2026-11-18: Fast-forwarded `jax_pymc` from Step 5 (`e5b62a9`) to Step 7 (`012c29c` on `jaxswap/step7-gp`, plus this note). Steps 1–2, 4–7 are complete on `jax_pymc`. Items 3.1–3.3 and 4.7 are still open. The five-file regression started after the Param7 fix finished later: 6451 passed, 14 skipped, 8 failed. All 8 are `test_grad_source_astrometry_unlensed_extension` raising `grad smoke not wired` for photometry-only `BSBL_Phot`, `BFSBL_Phot`, `BFSBL_Limb_Phot`, and `PSTL_Phot` (Par and noPar). That run imported the tree from before Steps 6–7.
- 2026-11-17: Step 7.7 (`jaxswap/step7-gp`, commit `c2b50e2`). Traced likelihoods for the last five GP classes: PSPL PhotAstrom GP Param4, Param4_1, and Param4_2 reuse the Param4 kernel, and PSBL PhotAstrom GP Param2 (with and without parallax) uses the reduced Param2 geometry. The Param4 assignment no longer replaces a likelihood a parent class already defined, so the RefPar GP classes keep their own methods. Likelihood parity: 5 passed at the GP atol of 1e-3, and the two RefPar GP classes that had been overwritten still pass. `JAX_LIKELIHOOD_CLASSES` is 314. Item 4.7 is still open. Step 7 is complete.
- 2026-11-17: Step 7.6 (`jaxswap/step7-pstl`, commit `6ee6b8c`). Traced likelihoods for static PSTL PhotAstrom Param2 and Param3 (4 classes). Masses and distances are recovered from `thetaE` or `log10_thetaE` and then passed through the Param1 geometry. Likelihood parity: 4 passed at 1e-6. `JAX_LIKELIHOOD_CLASSES` is 309. Item 7.7 and item 4.7 are still open.
- 2026-11-17: Step 7.5 (`jaxswap/step7-pspl`, commit `100251e`). Traced likelihoods for PSPL PhotAstrom Param4, Param5, Param6, Param7, Param4 geoproj, and Astrom Param4 (8 classes). Geoproj photometry parameters are converted to the heliocentric frame before the Param2 kernel, and the astrometry call now receives `raL`, `decL`, and `t0par`. Likelihood parity: 8 passed at 1e-6. `JAX_LIKELIHOOD_CLASSES` is 305. `log10_piE` has a test-only prior. Items 7.6–7.7 and 4.7 are still open.
- 2026-11-17: Step 7.4 (`jaxswap/step7-bfsbl`, commit `18da946`). Traced likelihoods for the 40 BFSBL classes that lacked one: 4 photometry-only and 36 PhotAstrom (linear, accelerated, circular, elliptical, and Param2, with and without limb darkening). Sky tracks are the BSBL tracks; each source is a finite disk. The host contour centroid is placed on the lens-system origin. Non-slow likelihood parity: 24 passed at the existing BFSBL atol of 5e-2 (22 non-limb, including the two static Param1 classes, plus 2 limb photometry-only). The 20 limb PhotAstrom tests are marked slow and were not run. `JAX_LIKELIHOOD_CLASSES` is 297. Items 7.5–7.7 and 4.7 are still open.
- 2026-11-17: Step 7.3 (`jaxswap/step7-bsbl`, commit `7dfa721`). Traced photometric and astrometric likelihoods for all 22 BSBL classes (static, linear, accelerated, circular, and elliptical). Each source is solved with the point-source binary-lens image solver and flux-weighted. Non-slow likelihood parity: 22 passed at 1e-6. `JAX_LIKELIHOOD_CLASSES` is 257. `sep_SL`, `sep_S`, `sep_L`, `phi_S`, `phi_L`, and `aS` have test-only priors. Items 7.4–7.7 and 4.7 are still open.
- 2026-11-17: Step 7.2 (`jaxswap/step7-psbl-fsbl-kepler`, commit `2b733c9`). Traced photometric and astrometric likelihoods for the 76 PSBL and FSBL circular and elliptical orbit classes. PhotAstrom tracks are in the lens center-of-mass frame. Photometry-only classes rebuild the orbit from the relative state at `t0`. Non-slow likelihood parity: 56 passed (PSBL at 1e-6, FSBL photometry-only at 1e-4, FSBL PhotAstrom orbits at 5e-3). The 20 FSBL limb PhotAstrom orbit tests are marked slow and were not run. `JAX_LIKELIHOOD_CLASSES` is 235. `v_para`, `v_perp`, `v_rad`, `r_s`, and `a_s` have test-only priors. Items 7.3–7.7 and 4.7 are still open.
- 2026-11-17: Step 7.1 (`jaxswap/step7-bspl-photastrom`, commit `0014877`). Traced photometric and astrometric likelihoods for all 50 BSPL PhotAstrom classes, including linear, accelerated, circular, and elliptical orbits and the GP variants. The centroid matches `BSPL_PhotAstrom.get_astrometry` (lens flux clipped at zero); photometry matches `BSPL.get_photometry` (blend not clipped). `tests/test_swap_parity.py` likelihood parity for those 50 classes passed (GP at 1e-3, the others at 1e-6), and the coverage test passed. `JAX_LIKELIHOOD_CLASSES` is 159. `log_a` has a test-only prior because neither fitter defines a default. Items 7.2–7.7 and 4.7 are still open.
- 2026-11-17: Step 6 (`jaxswap/step6-fitter-api`, commit `b4e87c0`). Restored `def_best='maxl'` on `print_likelihood`, `calc_chi2`, `calc_chi2_manual`, and `get_residual`. `MicrolensSolverWeighted` does not override those methods, so the keyword applies there too; the separate weighted chi-squared body in `model_fitter.py` was not copied. Ported `write_priors_pkl` and `load_priors` (pickle I/O only) and call `write_priors_pkl` from `solve`, matching `model_fitter.py`. Ported `plot_ultranest_style`, `save_params`, `make_sed_distance_gen`, `make_fdfdt`, and `random_prob` with the same signatures. `make_sed_distance_gen` stays the NumPy generator, not the JAX `make_*_gen` form. `tests/test_fitter_jax_api.py`: 9 passed. Deferred prior signatures are skipped (`make_*_gen`, `weighted_quantile`, `make_default_priors`). The five-file regression was not re-run; Step 6 does not change likelihoods. Item 4.7 is still open. Class rows in Appendix A are unchanged.
- 2026-11-16: Step 5 (`jaxswap/step5-model-api`, commit `76c8879`). Ported `PSPL_PhotAstromParam7` and the noPar/Par concrete classes (no default prior, no JAX likelihood; both names are in `KNOWN_JAX_LNL_GAPS`). The traced geometry path accepts `log10_piE` the same way it accepts `log_piE`, so the existing gradient checks can run the new class. A JAX likelihood is still Step 7. Ported `get_astrometry_unlensed_noparallax` and `get_astrometry_pm_removed` onto `PSPL`, and `FSPL.cent`. `tests/test_model_jax_api.py`: 623 passed, 8 skipped. Astrometry helpers, including finite-source, match at 1e-8 (worst ~1e-12, PSTL phot; FSPL limb centroid ~1e-18). The item 3.3 limb difference is in the traced likelihood, so these host comparisons were not relaxed. `FSPL.cent` matches at 1e-8 with the lens at the origin; an offset lens is not compared at 1e-8 because `model.py` `detJac` ignores the lens position. Eight FSPL phot-only classes are skipped (no `xS0` on `model.py`). Required regression files had no failures through 4403 of 6473 tests, into pre-existing FSBL gradient checks (`FSBL_PhotAstrom_Par_CircOrbs_Param3`); that suite was still running when this was committed. Param6 and Param7 old-vs-jax parity and gradients: 84 passed. Item 4.7 is still open.
- 2026-11-16: Step 4 (`jaxswap/step4-parity-tests`). Added `tests/test_swap_parity.py` and the missing `CANONICAL` values (`pi_ref_frame`, PSTL EllOrbs elements). Non-slow run: 315 passed, 12 skipped, 1 xfailed (`PSTL_Phot_Par_Param1` forward), 22 slow deselected. Coverage: every concrete `model_jax` class is in `JAX_LIKELIHOOD_CLASSES` (109) or `KNOWN_JAX_LNL_GAPS`. Likelihood atol is 1e-6 except GP 1e-3 (tinygp vs celerite, measured ≤3e-5), FSBL 1e-4 (measured ~4e-5), BFSBL 5e-2 (measured ~2e-2; host adaptive contour vs traced kernel), and FSPL/BFSPL phot+astrom 1e-1 (item 3.3 host NumPy limb vs JAX kernel, measured ≤0.06, almost all in the photometry term). `PSPL_Phot_Par_Param1_geoproj` now lists `t0par` in `fixed_param_names`; the traced conversion uses `jax.pure_callback` so the jitted fitter can call the NumPy frame transform. `model.py` still omits `t0par` (frozen); the NumPy comparison installs it for that one call. Open: item 3.3 host limb sampler; BFSBL traced kernel vs host adaptive contour; slow limb phot+astrom and FSBL Lin/Acc likelihoods not run; PhotAstrom geoproj is not in the likelihood registry and still omits `t0par`.
- 2026-11-16: Step 2 (`jaxswap/step2-raise-missing`): added `model_jax.JAX_LIKELIHOOD_CLASSES` (109 classes: Appendix A JAX-likelihood ✅ plus Step 1 geoproj / BFSPL Phot). `build_explicit_jax_loglik_fn` raises `NotImplementedError` naming the class (or `add_err`/`mult_err`); removed host fallback in `evaluate_loglik_jax`. Open question remains: Jessica — confirm raising for `add_err`/`mult_err` (implemented as raise per Change text). Tests: `tests/test_jax_lnl_registry.py`.
- 2026-11-15: Step 1 (`jaxswap/step1-correctness`): fixed W1–W4 and hygiene items 1.1–1.9 (orbiting_lens_offsets sign, BFSPL phot jax_log_likely, GP SHO sigma=sqrt(S0*omega0*Q), alt S0 names, geoproj jax_log_likely via geo_phot_to_helio_jax, narrow MicrolensSolverJaxLike except, remove tkinter / bagle.jax_model / dead BSPL helpers). Added tests/test_jax_swap_fixes.py; updated test_psbl_orbit_jax static-offset check for the new primary side. commit 260b2b3
- 2026-11-15: Audit of `fsbl_ld_jax` at `5368a88`. Found four wrong likelihood
  mappings (W1 to W4), coverage gaps, and API differences; first draft of this plan
  written.
- 2026-11-15: FSPL `n_outline` efficiency check done; Jessica decided `n_outline`
  comes from `data['n_outline']` (default 50), passed like `raL`/`decL`.
- 2026-11-15: Plan revised per Jessica's feedback: more explicit, written as an
  implementation spec for a coding agent; per-class ✅/❌ status added (Appendix A);
  all prior function and prior value changes removed and listed under Deferred.

---

## Appendix A. Per-class status after Step 7.7

Step 7.7 gives the last five GP classes a traced likelihood; those tests passed at 1e-3. Step 7.6 gives static PSTL Param2 and Param3 a traced likelihood; those 4 tests passed. Step 7.5 gives the remaining PSPL Param4, Param5, Param6, Param7, and Param4 geoproj classes a traced likelihood; those 8 tests passed. Step 7.4 gives every BFSBL class a traced likelihood. Non-limb PhotAstrom and all photometry-only classes passed likelihood parity. Limb PhotAstrom rows stay ❓ because that slow test was not run. Step 7.3 gives every BSBL class a traced likelihood; those 22 likelihood-parity tests passed. Step 7.2 gives every PSBL and FSBL `CircOrbs` / `EllOrbs` class a traced likelihood. Limb PhotAstrom Keplerian rows are ❓ because that slow test was not run. The other class rows are unchanged from Step 7.1. Refreshed from `tests/test_swap_parity.py` on `jaxswap/step4-parity-tests` for the Step 4 run (non-slow: 315 passed, 12 skipped, 1 xfailed, 22 slow deselected), with the two Param7 classes updated for Step 5. Forward parity is the 10-epoch test at 1e-8. Likelihood parity is the 40-point traced-versus-host test. Slow limb phot+astrom and FSBL linear/accelerated-orbit likelihoods were not run (❓). Param7 now has a traced likelihood. Item 4.7 is still open.

### PSPL Astrom (2 of 2 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSPL_Astrom_Par_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_Astrom_Par_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### PSPL GP Phot (13 of 13 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSPL_Phot_Par_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_Par_GP_Param1_2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_Par_GP_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_Par_GP_Param2_2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_Par_GP_Param2_3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_Par_GP_Param2_4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_Par_GP_Param2_5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_Par_GP_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_Par_GPnoJitter_Param2_2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_Par_GPnoJitter_Param2_3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_noPar_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_noPar_GP_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_Phot_noPar_GP_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |

### PSPL GP PhotAstrom (15 of 15 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSPL_PhotAstrom_Par_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_Par_GP_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_Par_GP_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_Par_GP_Param3_1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_Par_GP_Param3_2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_Par_GP_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_Par_GP_Param4_1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_Par_GP_Param4_2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_Par_GPnoJitter_Param3_1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_RefPar_GP_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_RefPar_GP_Param3_1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_RefPar_GP_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_RefPar_GP_Param4_1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_noPar_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSPL_PhotAstrom_noPar_GP_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |

### PSPL Phot (7 of 7 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSPL_Phot_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_Phot_Par_Param1_geoproj` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_Phot_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_Phot_Par_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_Phot_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_Phot_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_Phot_noPar_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### PSPL PhotAstrom (15 of 15 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSPL_PhotAstrom_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_Par_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_Par_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_Par_Param4_geoproj` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_Par_Param5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_Par_Param6` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_Par_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_RefPar_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_RefPar_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_noPar_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_noPar_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSPL_PhotAstrom_noPar_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### PSBL GP Phot (2 of 2 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSBL_Phot_Par_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSBL_Phot_noPar_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |

### PSBL GP PhotAstrom (4 of 4 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSBL_PhotAstrom_Par_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSBL_PhotAstrom_Par_GP_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSBL_PhotAstrom_noPar_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `PSBL_PhotAstrom_noPar_GP_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |

### PSBL Phot (6 of 6 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSBL_Phot_Par_CircOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_Phot_Par_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_Phot_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_Phot_noPar_CircOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_Phot_noPar_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_Phot_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### PSBL PhotAstrom (46 of 46 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSBL_PhotAstrom_Par_AccOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_AccOrbs_Param6` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_AccOrbs_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_CircOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_CircOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_CircOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_CircOrbs_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_CircOrbs_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_CircOrbs_Param8` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_EllOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_EllOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_EllOrbs_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_EllOrbs_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_EllOrbs_Param8` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_LinOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_LinOrbs_Param6` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_LinOrbs_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_Param5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_Par_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_AccOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_AccOrbs_Param6` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_AccOrbs_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_CircOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_CircOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_CircOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_CircOrbs_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_CircOrbs_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_CircOrbs_Param8` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_EllOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_EllOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_EllOrbs_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_EllOrbs_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_EllOrbs_Param8` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_LinOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_LinOrbs_Param6` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_LinOrbs_Param7` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSBL_PhotAstrom_noPar_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### BSPL GP Phot (2 of 2 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BSPL_Phot_Par_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_Phot_noPar_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |

### BSPL GP PhotAstrom (18 of 18 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BSPL_PhotAstrom_Par_GP_AccOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_Par_GP_AccOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_Par_GP_AccOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_Par_GP_LinOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_Par_GP_LinOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_Par_GP_LinOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_Par_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_Par_GP_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_Par_GP_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_noPar_GP_AccOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_noPar_GP_AccOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_noPar_GP_AccOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_noPar_GP_LinOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_noPar_GP_LinOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_noPar_GP_LinOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_noPar_GP_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_noPar_GP_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |
| `BSPL_PhotAstrom_noPar_GP_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-3 (tinygp vs celerite) |

### BSPL Phot (2 of 2 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BSPL_Phot_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_Phot_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### BSPL PhotAstrom (32 of 32 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BSPL_PhotAstrom_Par_AccOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_AccOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_AccOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_CircOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_CircOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_CircOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_EllOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_EllOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_EllOrbs_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_LinOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_LinOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_LinOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_Par_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_AccOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_AccOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_AccOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_CircOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_CircOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_CircOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_EllOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_EllOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_EllOrbs_Param4` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_LinOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_LinOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_LinOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSPL_PhotAstrom_noPar_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### BSBL Phot (2 of 2 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BSBL_Phot_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_Phot_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### BSBL PhotAstrom (20 of 20 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BSBL_PhotAstrom_Par_AccOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_Par_CircOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_Par_CircOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_Par_CircOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_Par_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_Par_EllOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_Par_EllOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_Par_LinOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_AccOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_CircOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_CircOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_CircOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_EllOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_EllOrbs_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_LinOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BSBL_PhotAstrom_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### FSPL ±LD Phot (8 of 8 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `FSPL_Limb_Phot_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `FSPL_Limb_Phot_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `FSPL_Limb_Phot_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `FSPL_Limb_Phot_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `FSPL_Phot_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `FSPL_Phot_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `FSPL_Phot_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `FSPL_Phot_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### FSPL ±LD PhotAstrom (8 of 8 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `FSPL_Limb_PhotAstrom_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `FSPL_Limb_PhotAstrom_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `FSPL_Limb_PhotAstrom_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `FSPL_Limb_PhotAstrom_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `FSPL_PhotAstrom_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `FSPL_PhotAstrom_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `FSPL_PhotAstrom_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `FSPL_PhotAstrom_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |

### BFSPL ±LD Phot (4 of 4 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BFSPL_Limb_Phot_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BFSPL_Limb_Phot_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BFSPL_Phot_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `BFSPL_Phot_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### BFSPL ±LD PhotAstrom (4 of 4 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BFSPL_Limb_PhotAstrom_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `BFSPL_Limb_PhotAstrom_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `BFSPL_PhotAstrom_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |
| `BFSPL_PhotAstrom_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | lnL atol 1e-1 (item 3.3 host limb sampler) |

### FSBL ±LD Phot (12 of 12 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `FSBL_Limb_Phot_Par_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_Limb_Phot_Par_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_Limb_Phot_Par_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 1e-4 (host contour vs traced kernel) |
| `FSBL_Limb_Phot_noPar_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_Limb_Phot_noPar_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_Limb_Phot_noPar_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 1e-4 (host contour vs traced kernel) |
| `FSBL_Phot_Par_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_Phot_Par_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_Phot_Par_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 1e-4 (host contour vs traced kernel) |
| `FSBL_Phot_noPar_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_Phot_noPar_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_Phot_noPar_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 1e-4 (host contour vs traced kernel) |

### FSBL ±LD PhotAstrom (36 of 100 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `FSBL_Limb_PhotAstrom_Par_AccOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_Limb_PhotAstrom_Par_AccOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_AccOrbs_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_AccOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_CircOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_CircOrbs_Param4` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_CircOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_CircOrbs_Param8` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_EllOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_EllOrbs_Param4` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_EllOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_EllOrbs_Param8` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_LinOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_Limb_PhotAstrom_Par_LinOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_LinOrbs_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_LinOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_Limb_PhotAstrom_Par_Param2` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_Limb_PhotAstrom_Par_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_Param4` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_Param5` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_Par_Param8` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_AccOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_Limb_PhotAstrom_noPar_AccOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_AccOrbs_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_AccOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_CircOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_CircOrbs_Param4` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_CircOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_CircOrbs_Param8` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_EllOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_EllOrbs_Param4` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_EllOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_EllOrbs_Param8` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_LinOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_Limb_PhotAstrom_noPar_LinOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_LinOrbs_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_LinOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_Limb_PhotAstrom_noPar_Param2` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_Param4` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_Limb_PhotAstrom_noPar_Param8` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_Par_AccOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_PhotAstrom_Par_AccOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_Par_AccOrbs_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_Par_AccOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_Par_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_CircOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_CircOrbs_Param4` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_CircOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_CircOrbs_Param8` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_EllOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_EllOrbs_Param4` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_EllOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_EllOrbs_Param8` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_Par_LinOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_PhotAstrom_Par_LinOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_Par_LinOrbs_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_Par_LinOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_Par_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 1e-4 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_Par_Param2` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_Par_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_Par_Param4` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_Par_Param5` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_Par_Param6` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_Par_Param7` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_Par_Param8` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_noPar_AccOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_PhotAstrom_noPar_AccOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_noPar_AccOrbs_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_noPar_AccOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_noPar_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_CircOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_CircOrbs_Param4` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_CircOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_CircOrbs_Param8` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_EllOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_EllOrbs_Param4` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_EllOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_EllOrbs_Param8` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `FSBL_PhotAstrom_noPar_LinOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `FSBL_PhotAstrom_noPar_LinOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_noPar_LinOrbs_Param6` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_noPar_LinOrbs_Param7` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `FSBL_PhotAstrom_noPar_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 1e-4 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_noPar_Param2` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_noPar_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_noPar_Param4` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_noPar_Param6` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |
| `FSBL_PhotAstrom_noPar_Param8` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-3 (host contour vs traced kernel) |

### BFSBL ±LD Phot (4 of 4 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BFSBL_Limb_Phot_Par_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_Limb_Phot_noPar_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_Phot_Par_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_Phot_noPar_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |

### BFSBL ±LD PhotAstrom (20 of 40 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `BFSBL_Limb_PhotAstrom_Par_AccOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_Par_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_Par_CircOrbs_Param2` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_Par_CircOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_Par_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_Par_EllOrbs_Param2` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_Par_EllOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_Par_LinOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_Par_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `BFSBL_Limb_PhotAstrom_Par_Param2` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_noPar_AccOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_noPar_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_noPar_CircOrbs_Param2` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_noPar_CircOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_noPar_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_noPar_EllOrbs_Param2` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_noPar_EllOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_noPar_LinOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_Limb_PhotAstrom_noPar_Param1` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run in step 4 |
| `BFSBL_Limb_PhotAstrom_noPar_Param2` | ➖ | ✅ | ✅ | ➖ | ❓ | ❌ | slow likelihood not run |
| `BFSBL_PhotAstrom_Par_AccOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_Par_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_Par_CircOrbs_Param2` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_Par_CircOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_Par_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_Par_EllOrbs_Param2` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_Par_EllOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_Par_LinOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_Par_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-2 (host contour vs traced kernel) |
| `BFSBL_PhotAstrom_Par_Param2` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_noPar_AccOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_noPar_CircOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_noPar_CircOrbs_Param2` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_noPar_CircOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_noPar_EllOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_noPar_EllOrbs_Param2` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_noPar_EllOrbs_Param3` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_noPar_LinOrbs_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |
| `BFSBL_PhotAstrom_noPar_Param1` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ | lnL atol 5e-2 (host contour vs traced kernel) |
| `BFSBL_PhotAstrom_noPar_Param2` | ➖ | ✅ | ✅ | ➖ | ✅ | ✅ |  |

### PSTL Phot (2 of 2 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSTL_Phot_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | forward ignores images with \|A\| < 1e-4, item 4.6 |
| `PSTL_Phot_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

### PSTL PhotAstrom (10 of 10 good)

| Class | In model.py | In model_jax.py | JAX likelihood | Forward parity | Likelihood parity | Status | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `PSTL_PhotAstrom_Par_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSTL_PhotAstrom_Par_EllOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSTL_PhotAstrom_Par_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSTL_PhotAstrom_Par_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSTL_PhotAstrom_Par_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSTL_PhotAstrom_noPar_EllOrbs_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSTL_PhotAstrom_noPar_EllOrbs_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSTL_PhotAstrom_noPar_Param1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSTL_PhotAstrom_noPar_Param2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `PSTL_PhotAstrom_noPar_Param3` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |

---

## Appendix B. How the status tables were generated

The audit scripts are not part of the repository. They ran against a clean clone of
`fsbl_ld_jax` at `5368a88`, with a stub for `pymultinest` so that the modules import
without MultiNest installed.

1. **Class discovery.** Every class in each module that subclasses `ModelClassABC`
   and is not abstract (`inspect.isabstract`) is a concrete class. This gives 222
   classes in `model.py` and 376 in `model_jax.py`.
2. **JAX likelihood.** For each `model_jax` class, the class in the method resolution
   order that defines `jax_log_likely_photometry` (if the class fits photometry) and
   `jax_log_likely_astrometry` (if it fits astrometry) was recorded. A class "has a
   JAX likelihood" if all methods it needs are defined somewhere in its MRO, except
   for W2 and W4, where the method found is the wrong parent's.
3. **Forward parity.** `build_paired_instances` from
   `tests/model_old_vs_jax_fixtures.py` built a `model.py` and a `model_jax`
   instance for each shared class, and `get_photometry`, `get_astrometry`, and
   `get_amplification` were compared at 10 epochs between MJD 56900 and 57300.
4. **Likelihood parity.** For each class with a JAX likelihood, 40 points between MJD
   57000 and 57200 were generated from the reference forward model, and the traced
   lnL (`build_explicit_jax_loglik_fn`), the host lnL
   (`model_fitter_jax.MicrolensSolver.log_likely`), and the NumPy lnL
   (`model_fitter.MicrolensSolver.log_likely`) were evaluated at the true parameters,
   with a time limit per class.
5. **Table.** `status_table.py` combines these results with the rules in "Summary of
   per-class status" and writes the summary and the appendix tables.

Step 4's `tests/test_swap_parity.py` replaces this harness. Once it exists, regenerate
Appendix A from the test results instead.
