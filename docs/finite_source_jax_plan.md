# Finite-source models on `jax_pymc`

Planning note for Jessica. No implementation is in this branch.

Surveyed tree: `origin/jax_pymc` at `64cd87e2b15acfc32b69561a1a56897f7385cbf2`
("Merge per-filter geoproj origin and joint-likelihood slots into jax_pymc").
Line numbers below are that commit. I did not run the finite-source tests, and
I did not execute `jax.grad`, `jax.jit`, or a timing comparison. Where a claim
is from reading the source only, it is marked as such.

Names in the repository:

| Asked name | Actual name | Meaning |
| --- | --- | --- |
| FSPL | `FSPL` | Finite source, point lens |
| FSBL | `FSBL` | Finite source, binary lens |
| FBSPL | `BFSPL` | Two finite sources, point lens |
| FBSBL | none | Two finite sources, binary lens. No class, on this branch or on the branches below |

`PSTL` (point-source triple lens) exists only on `origin/jax_dex`. It is not a
finite-source model and is out of scope here.

## 1. Current state on `jax_pymc`

### 1.1 What "JAX support" means today

`src/bagle/jax/fspl.py` (68 lines) is a host-NumPy wrapper. Its docstrings say
the contour is "not wired" and that amplification uses the adaptive-mesh path
until a JAX contour exists. Every function calls `model.get_photometry`,
`get_amplification`, `get_astrometry`, or `get_all_arrays` and returns
`numpy` arrays. Nothing in that file is differentiable.

`jax_log_likely_photometry` / `jax_log_likely_astrometry` in `model_jax.py`
stop at line 14965, on a static BSPL parameterization. No class whose name
starts with `FSPL`, `FSBL`, or `BFSPL` defines those methods. The JAX fitter
looks them up with `getattr` (`model_fitter_jax.py` around line 264) and
otherwise cannot trace a finite-source likelihood. `jax_physics.py` has no
finite-source kernel. `src/bagle/jax/` has PSPL, PSBL, BSPL, BSBL, GP, and
orbit helpers, plus the host wrapper above.

Both fitters still say "FSPL implementation in-progress"
(`model_fitter.py` line 140, `model_fitter_jax.py` line 574). The only
finite-source default prior is

```text
'radiusS': ('make_gen', 1E-4, 1E-2)
```

(`model_fitter.py` line 229, `model_fitter_jax.py` line 664). There is no
`rho`, `radiusS_pri`, `radiusS_sec`, or `utilde` prior. `n_outline` is a
constructor integer, not a sampled parameter.

The old-vs-JAX "grad" tests for these classes are finite differences of the
host methods. Example: `test_grad_fspl_photastrom_param1_phot` is documented
as "FSPL phot grad smoke via host AMG finite-difference"
(`tests/test_model_old_vs_jax.py` line 823). A finite-difference pass does
not mean `jax.grad` of a jitted likelihood is finite.

### 1.2 FSPL

Present in both `model.py` (line 22303) and `model_jax.py` (line 23136).
Neither copy is marked do-not-use. The two copies are not the same algorithm.

**NumPy (`model.py`)**

- `FSPL.get_all_arrays_CI` (line 22306) is a uniform limb contour. Image
  positions of the outline come from the point-lens lens equation. Area and
  centroid use Green's theorem plus the second-order parabolic correction in
  Bozza et al. (2021), cited in the docstring (Eqs. 9, 10, 19, 21, 22).
  Implementation is vectorized NumPy (`np.diff`, `np.sum` over the outline).
  Amplification divides the image area by `pi * radiusS**2` (line 22419).
- `FSPL.get_all_arrays_amg` (line 22435) is an adaptive mesh on the source
  limb. More samples are added when the limb enters the Einstein ring. The
  point-lens images are the closed form `im_pos1` / `im_pos_all` (line 22476).
  East is real, North is imaginary. The limb is centered on
  `self.xS0[filt_idx]` (confirmed in the AMG body; the contour path uses the
  same `filt_idx` through `get_resolved_astrometry_outline`). The body is a
  Python loop over epochs, which is the slow path the Dex speedup replaces.
- `FSPL.get_all_arrays` (line 22666) dispatches on `astrometryFlag`:
  astrometry always takes AMG; photometry always takes the uniform contour.
  It does not switch on `|u|` versus `rho`.
- Photometry (`get_photometry`, line 22729) sums the two image
  amplifications, converts `mag_src[filt_idx]` to flux, and blends with the
  per-filter source flux fraction `b_sff[filt_idx]`.
- Astrometry lives on `FSPL_PhotAstrom` (line 22780). The unresolved position
  is the magnification-weighted centroid of the two images. The centroid
  integrals are the same Bozza contour, so the finite-source astrometric
  shift is the difference between that centroid and the unlensed source.
  `src/bagle/jax/fspl.py` `fspl_centroid_shift_from_model` reports that shift
  in mas, still via the host method.
- `get_u` (line 22677) is the usual rectilinear track plus
  `parallax.parallax_in_direction(..., obsLocation=self.obsLocation[filt_idx])`.
  Per-filter observers are already threaded here.

**`model_jax.py` FSPL is still host NumPy, with a few `jnp` calls mixed in**

- `im_pos_all` (line 23144) uses `jnp.conjugate` on values produced by NumPy.
- `get_all_arrays_amg` (line 23164) returns six arrays (areas and centroid
  components), not `(images, amps)`. `get_all_arrays_amg_only` (line 23347)
  returns the `(images, amps)` pair. The dispatcher (line 23726) sends
  astrometry to `get_all_arrays_amg_only` and photometry to
  `get_all_arrays_CI`.
- `get_all_arrays_CI` (line 23569) is the Bozza contour in NumPy, then, if
  the object has `xS0`, `muS`, and `thetaE_amp`, it does `jnp.where` on
  `|u| <= 2 * radiusS` (line 23685) and unpacks `get_all_arrays_amg` into
  the six area/centroid arrays. `jnp.where` with one argument returns a
  tuple of index arrays. I did not execute this branch. It is not the
  photometry path of a phot-only model (those usually lack `xS0`), and it is
  not the astrometry path (that uses `amg_only`). It is a landmine if CI is
  called on a phot+astrom object. The local name
  `radiusS = self.source_radius_thetaE_units()` (line 23608) is the radius
  in Einstein-radius units; the amplification denominator at line 23707
  still divides by `self.radiusS**2`. Whether those two uses are consistent
  depends on the units of the outline, which I did not trace through a run.

**Parameter classes (both files, unless noted)**

| Class | Role | `radiusS` unit |
| --- | --- | --- |
| `FSPL_PhotParam1` (`model.py` 23505) | `t0`, `u0_amp`, `tE`, `piE_E`, `piE_N`, `radiusS`, `b_sff`, `mag_src` | Einstein radii (`model.py` 23531; phot-only `rho = self.radiusS` at 23434) |
| `FSPL_PhotParam2` | `piEN_piEE` style, same radius | Einstein radii (`source_radius_thetaE_units` docstring, `model_jax.py` 23157) |
| `FSPL_PhotAstromParam1` (`model.py` 23748) | physical: `mL`, `t0`, `beta`, distances, per-filter `xS0_E`/`xS0_N`, proper motions, `radiusS`, `b_sff`, `mag_src` | arcsec (`model.py` 23790) |
| `FSPL_PhotAstromParam2` | inherits `PSPL_PhotAstromParam2` plus the source radius | arcsec on the physical side; confirm before fitting |
| `FSPL_noParallax`, `FSPL_Parallax` | flags only | |

Concrete leaves, both files: `FSPL_Phot_{noPar,Par}_Param{1,2}` and
`FSPL_PhotAstrom_{noPar,Par}_Param{1,2}` (`model.py` 27660–27751,
`model_jax.py` 35085–35192). Eight leaves. No geoproj leaf, no RefPar leaf,
no GP leaf. `filt_param_names` on the phot+astrom Param1 are
`xS0_E`, `xS0_N`, `b_sff`, `mag_src`, with usage `astrom`, `astrom`, `both`,
`phot`. That matches the multi-location convention. `xS0` storage as
`(n_filters, 2)` is inherited from `PSPL_Param` and is already on this
commit.

**Tests**

- Host tests in `tests/test_model.py` and `tests/test_model_jax.py`:
  source astrometry, centroid shift, boundary, lens astrometry, lens origin,
  phot+astrom methods, phot methods, and (JAX file only)
  `test_refpar_and_fspl_param_lists_match_numpy`,
  `test_fspl_photparam1_numpy_matches_jax`.
- `tests/test_model_multi_location.py` `test_fspl_filt_idx_reaches_astrometry`
  (line 130).
- `tests/test_model_old_vs_jax.py`: parity and host finite-difference grad
  for Phot Param2, PhotAstrom Param1 and Param2, extended methods, outline,
  and resolved astrometry. I did not re-run them for this plan. A previous
  session saw long old-vs-JAX runs abort inside a JAX compile (signal 6);
  those aborts were not treated as finite-source bugs.

**Missing or broken**

- No traced likelihood, so MultiNest can call the host model but NumPyro,
  BlackJAX NUTS/MCLMC, and replica exchange cannot differentiate it.
- `model.py` and `model_jax.py` dispatch and AMG return shapes differ.
- One default prior covers both an Einstein-radius `radiusS` (phot) and an
  arcsec `radiusS` (phot+astrom). `1e-2` arcsec is about 10 mas.
- No FSPL GP parameterization and no concrete GP leaf.
- Limb darkening is a separate, unfinished hierarchy (next subsection).

### 1.3 FSPL limb darkening

`FSPL_Limb` (`model.py` 24051, and twice in `model_jax.py` at 25090 and
30771) integrates a linear limb-darkening law in the radial coordinate.
`F(r)` is the standard linear-law cumulative weight with coefficient
`utilde` in `[0, 1]` (`model.py` 24052). `get_amplification` (line 24114)
loops over `nr` annuli in Python and over times in Python. `0` is uniform;
`1` is fully darkened at the limb. There is no Claret four-parameter law,
no quadratic law, and no per-filter coefficient.

`FSPL_Limb_PhotAstromParam1` is marked "DO NOT USE -- in progress"
(`model.py` 24219, `model_jax.py` 25256 and 30937). It does not go through
`PSPL_Param`'s per-filter `xS0` stacking in the usual way: it sets

```text
self.xL0 = self.xS0[0] - self.thetas0 * 1e-3
```

at `model.py` 24266, `model_jax.py` 25303, and `model_jax.py` 30984. The
comment says "Filter 0 East/North origin." For `n_filters > 1` this pins the
lens to filter 0's origin and ignores the other filters. The second copy in
`model_jax.py` is a duplicate class body (the second definition wins at
import time). There is no concrete `ModelClassABC` leaf for the limb
classes, and no fitter prior for `utilde` or `nr`.

The commented block at `model.py` 28465 (`# class FSBL(ABC):` and the
parallax/orbit stubs through about 29431) is dead code. It mentions
`utilde` and an older ray-shooting / cluster integrator. It is not the live
FSBL.

### 1.4 FSBL

Live FSBL exists only in `model_jax.py`, class `FSBL(PSBL)` at line 25314.
`model.py` has no live `class FSBL`. The docstring says the magnification is
"inspired by `caustics` by F. Bartolic" and links
`https://github.com/fbartolic/caustics`. I did not read that repository's
license, and this tree does not vendor it. The polynomial and the contour
are written out in `model_jax.py`.

Algorithm, from reading the methods (not from a run):

- Binary-lens images: Witt (1995) quintic, companion-matrix eigenvalues
  (`quintic_roots`, line 25341; `binary_lens_poly_coeffs`, line 25352).
  `get_image_pos_arr` (line 25377) `vmap`s one root solve per limb point and
  rejects roots that fail the lens equation (`root_tol`, floor `1e-8`).
- Finite source: `images_on_source_limb` (line 25888) starts from a uniform
  limb, then for `fsbl_niter = 10` iterations inserts points where the image
  tracks jump. The insert uses `jnp.insert` inside a Python `for` loop.
  After the loop it perturbs duplicate roots with
  `jax.random.uniform` from `PRNGKey(0)` (line 25930). The limb length
  therefore depends on `n_outline` and `fsbl_niter`, which are Python ints,
  and the sample locations depend on the data.
- `magnification_one` (line 25938) permutes image tracks, splits closed and
  open segments, and integrates with `vmap` and `lax.cond`. Magnification is
  signed area over `pi * rho**2`.
- `get_all_arrays_CI` (line 26002) `vmap`s `magnification_one` over times.
  The source position `u` and the two lens positions are computed with the
  NumPy `get_u` and `get_resolved_lens_astrometry` first, then cast to JAX.
  Photometry uses `rho = self.radiusS` (Einstein radii, the PSBL convention).
  Astrometry converts `radiusS` arcsec through `source_radius_thetaE_units`
  (line 25335).
- `get_all_arrays` (line 25474) returns four arrays: images, parity,
  amplification, per-image area. That is a different signature from FSPL's
  `(images, amps)`. `fspl_resolved_amplification_from_model` assumes a
  two-tuple and will mis-unpack FSBL (`jax/fspl.py` line 52).

Differentiability, from the structure, not from `jax.grad`:

- The inner root solve and the area sum are JAX ops, so a fixed limb could
  be differentiated.
- The adaptive insert changes which limb samples exist as a function of the
  parameters. Under `jit`, the Python loop is unrolled only if `niter` and
  the number of inserted points are static. The indices are data-dependent.
  I do not know whether this traces. I would not put it under NUTS until
  someone runs `jax.jit` and `jax.grad` on `magnification_one` alone.
- `jax.random` inside the limb makes the magnification a noisy function of
  a fixed seed. A gradient through that perturbation is not a gradient of
  the physical magnification.
- The outer method is not a pure JAX function because parallax and orbits
  are NumPy.
- At a caustic, image tracks appear and disappear and the Jacobian
  determinant crosses zero. The magnification of a finite source stays
  finite; its derivative with respect to source position is large and, for
  an unsmoothed contour, can jump when a segment is classified as closed or
  open. That is a bad target for HMC. Nested sampling and SMC do not need
  that derivative.

**Parameter classes** (`model_jax.py` only), all subclassing the matching
PSBL parameterization and adding `radiusS`:

- Phot: `FSBL_PhotParam1` (27047). Orbit phot: `FSBL_Phot_EllOrbs_Param1`,
  `FSBL_Phot_CircOrbs_Param1`.
- Phot+astrom Param1 through Param8, plus Lin/Acc/Ell/Circ orbit variants
  where the PSBL parent exists (27140–30201).
- GP parameter containers only: `FSBL_GP_PhotParam1`,
  `FSBL_GP_PhotAstromParam1`, `FSBL_GP_PhotAstromParam2` (30522–30679).
  Nothing inherits them. There is no instantiable FSBL GP model.

Concrete leaves are the long `ModelClassABC` list at 33491–34096 (phot,
phot+astrom, noPar/Par, and the orbit flavors). No RefPar leaf and no
geoproj leaf.

**Tests:** `tests/test_model_old_vs_jax.py` has parity and host
finite-difference tests for Phot Param1, orbit phot, PhotAstrom Param1–8,
and resolved methods. `tests/test_psbl_roots.py`
`test_fsbl_images_on_source_limb_uses_fast_solver` (line 132) checks the
root solver, not a full light curve against VBBL. Because `model.py` has no
live FSBL, "parity" here is the JAX-file class against itself or against a
fixture path in `tests/model_old_vs_jax_fixtures.py`. I did not open every
fixture. Treat a green parity row as "the host method runs," not as "the
likelihood is jitted."

### 1.5 BFSPL

`BFSPL` is "Binary Finite-Source, Point-Lens": two sources, one lens
(`model.py` 24278, `model_jax.py` 30998). The magnification is not the
Bozza contour. `get_all_arrays` (model.py 24281) integrates the Lee et al.
(2009) finite-source point-lens expressions (the comment cites ApJ 695, 200)
with a fixed Simpson-style sum, `n = 20` (doubled when `u` is inside the
source). Each source has its own `rho` (`radiusS_pri`, `radiusS_sec` in
arcsec, converted at line 24557) and its own `u` (`u_pri`, `u_sec` from
`get_u`, line 24547). Image positions are placed along each source's `u`
hat, scaled by the centroid integrals. The two sources are then combined
with their fluxes (`mag_src_pri`, `mag_src_sec`). I did not read the flux
combination line by line past 24599; the fitter names are `mag_src_pri` and
`mag_src_sec`, and `b_sff` is one source-flux-fraction array per filter
(both sources together over total flux), matching the BSPL convention in
the BSPL docstring at `model_jax.py` 14931:
`b_sff = (f_S1 + f_S2) / (f_S1 + f_S2 + f_L + f_N)`.

The sum is a Python loop over times and over the 20 abscissae. It is NumPy.
The `model_jax.py` copy is the same structure (class at 30998,
`get_all_arrays` at 31001). `jax/fspl.py` special-cases BFSPL with a
different `swapaxes` than FSPL (line 54).

**Classes.** `BFSPL_PhotAstrom`, `BFSPL_noParallax`, `BFSPL_Parallax`,
`BFSPL_PhotAstromParam1` (`model.py` 25042). The Param1 `fitter_param_names`
list `radiusS` once (line 25120) while `__init__` takes `radiusS_pri`,
`radiusS_sec`, and `sep` (line 25139). That mismatch will confuse the cube
builder; I did not instantiate the class. The only concrete leaf is
`BFSPL_PhotAstrom_noPar_Param1` (model.py 27643, model_jax.py 35067). No
parallax leaf, no phot-only leaf, no GP, no RefPar, no geoproj.

**Tests:** `test_parity_bfspl_photastrom_param1` and
`test_grad_bfspl_photastrom_param1` in `tests/test_model_old_vs_jax.py`
(lines 1099 and 1618). Host finite differences again.

### 1.6 Multi-location conventions these classes must keep

Already true for the live FSPL and BFSPL phot+astrom parameterizations, and
for FSBL because it subclasses PSBL:

- `xS0` stored as `(n_filters, 2)` for every filter count, including one.
- `xS0_E` and `xS0_N` are length `n_filters`.
- `obsLocation[filt_idx]` in `get_u`.
- `b_sff` and `mag_src` (or `mag_src_pri` / `mag_src_sec`) are per-filter
  arrays. The name is source flux fraction.
- Cube suffixes are 1-based. `tie_fit_param` must be read inside the traced
  likelihood the same way `_class_vector_slots` does for PSPL, or the
  tied-origin bug returns.
- `filt_idx` is 0-based and is an argument of `get_u`, `get_photometry`,
  and `get_astrometry`.

Not true:

- The three `FSPL_Limb_PhotAstromParam1` assignments of `xL0` from
  `xS0[0]` only.
- FSBL GP containers have no concrete model, so they never see a cube.
- A new BFSBL class does not exist to violate or follow the convention.

Geoproj and RefPar finite-source classes do not exist. `get_geoproj_ast_params`
on the PSPL base already takes `filt_idx` and converts `xS0[filt_idx]` with
Earth's parallax offset. A finite-source geoproj leaf can call that later.
Do not invent one in the first milestone.

## 2. Other branches

### 2.1 `origin/jax_dex`

Tip `e1bc6ec` (2026-09-16), Dex Bhadra, "FSPL speedup from the fspl_speedup
branch". Merge-base with `jax_pymc`:

```text
b1b2579142f07ff704634f59711184ce6b7c6c5e
2026-05-11  Faster PSBL runtimes
```

`git rev-list --left-right --count origin/jax_pymc...origin/jax_dex` is
**136 left, 4 right**. The four commits on the Dex side are:

| Commit | Author | Subject |
| --- | --- | --- |
| `e1bc6ec` | Dex Bhadra | FSPL speedup from the fspl_speedup branch (`model.py` only, +554/−618) |
| `ea87c40` | 10ay | Adding PSTL model_fitters and FSBL tests |
| `90f9209` | 10ay | FSBL Changes |
| `2df0010` | 10ay | New PSTL |

`git diff --stat` from the merge-base touches only `src/bagle/fake_data.py`,
`src/bagle/model.py` (about +5087), `src/bagle/model_fitter.py` (+17), and
`tests/test_model.py` (+378). There is no `model_jax.py` and no `src/bagle/jax/`.

`git merge-tree origin/jax_pymc origin/jax_dex` reports content conflicts in
`src/bagle/model.py` and `tests/test_model.py` only. That understates the
scientific drift: `jax_pymc` has since split the JAX models into
`model_jax.py`, changed `xS0` to `(n_filters, 2)`, and added the fitter cube.
A clean merge would still leave the speedup in the NumPy file and the live
JAX classes untouched.

What the speedup actually is (read from `origin/jax_dex:src/bagle/model.py`
around 20773–20870, not timed):

- NumPy, not `jax.grad`. `import jax` / `jnp` in that file is the older
  binary-lens polynomial code, mixed with `np.asarray` of JAX results
  (for example the root check near line 5519 of that file). I did not find
  a `jax.jit` or `jax.grad` call on the FSPL contour.
- `FSPL.get_all_arrays` vectorizes over time. If every epoch has
  `|u| >= 2 rho`, it uses the uniform contour. If every epoch is close, it
  uses AMG. If the light curve is mixed, it splits the times and fills a
  preallocated `(n_times, 2, 2)` image array. The comment in that method
  says the far path "is fully vectorized over times."
- `get_all_arrays_amg` preallocates up to `maxsamps = 1000` (5000 if
  `n_outline` is large) and pads unused slots by repeating the first point.
  The comment says adaptive sampling is sequential in angle and independent
  across times, so every epoch is updated together. Helpers are named
  `amg_numpy` and `amg_contour_integrals`.
- No accuracy number and no speed number are in the commit message. I did
  not benchmark it. The parent commit on `fspl_speedup` says the numerical
  artifact in the contour was fixed; I did not compare light curves.

`10ay`'s three commits add PSTL and FSBL tests on top of the single
`model.py`. The FSBL class on that branch starts at line 22388 of
`origin/jax_dex`'s `model.py`. `jax_pymc` already has a larger FSBL in
`model_jax.py`. Porting `10ay`'s FSBL would fight that code. PSTL is a
different lens (three point masses) and should stay on `jax_dex` unless
Jessica asks for it.

### 2.2 `origin/fspl_speedup`

Tip `7b9f5bc` (2026-10-01), nsabrams, "add log10_piE to default priors".
Merge-base `e77c6a8` (2025-12-05). **142 commits on `jax_pymc`, 72 on
`fspl_speedup`.**

The finite-source commits worth knowing:

| Commit | Author | What |
| --- | --- | --- |
| `c913b38` | nsabrams | "FSPL speedup from vectorizing over shared number of bins in number of samples, fixed source parallax, and fixed contour numerical artifact with cursor." `model.py` only, +150/−135. This is the original vectorization. |
| `2e42e1a` | Dex Bhadra | Same message as `e1bc6ec`. Dex's `jax_dex` commit is this work brought onto `jax_dex`. |
| `fb5f091` | nsabrams | "comment out broken fspl classes" |
| `e6b4547` | | `fspl_photastromparam3` |

`git merge-tree` against `jax_pymc` conflicts in `pyproject.toml`,
`src/bagle/data.py`, `src/bagle/model.py`, `src/bagle/model_fitter.py`,
`tests/conftest.py` (add/add), `tests/test_frame_convert.py`, and
`tests/test_model_fitter.py`. The other ~70 commits are docker, RefPar,
plot notebooks, and fitter bookkeeping that `jax_pymc` has either re-done
or does not want. Merging this branch is not realistic.

Same conclusion as `jax_dex`: the useful piece is a NumPy vectorization of
the FSPL limb, plus a claimed contour-artifact fix. No published error or
timing in the message. Attribute the algorithm to Natasha Abrams
(`c913b38`) and the `jax_dex` port to Dex Bhadra (`e1bc6ec` / `2e42e1a`).
The commit message also says the vectorization was done with Cursor; keep
that in the attribution comment if the code is ported.

### 2.3 `origin/new_fspl` and `origin/bagle-jax`

`origin/new_fspl` tip `d88e856` (2025-07-17, Jessica Lu, "Merge branch 'dev'
into new_fspl") **is** the merge-base. `jax_pymc` is 235 commits ahead and
`new_fspl` is 0 ahead. There is nothing to port.

`origin/bagle-jax` tip `ba4001f` (2024-05-27, Hunter Harling). Merge-base
`256daa42` (2023-01-27). **530 commits on `jax_pymc`, 15 on `bagle-jax`.**
`merge-tree` conflicts include an add/add on `src/bagle/model_jax.py` and
`tests/test_model_jax.py`. That `model_jax.py` is an early 10k-line file.
Do not merge it.

No remote matches `jax_dex_*` other than `origin/jax_dex`. I also looked at
`origin/jax_pymc_hi` only as a name in the branch list; I did not review it
for finite-source code.

### 2.4 What to port, and what to rewrite

Port by hand, with an attribution comment, from `origin/jax_dex`
`FSPL` (or equivalently `c913b38` / `2e42e1a` on `fspl_speedup`):

- The mixed-time dispatch: uniform contour when `|u| >= 2 rho`, adaptive
  mesh only near the Einstein ring, one preallocated output.
- `amg_numpy` / `amg_contour_integrals`: vectorize across epochs, pad unused
  limb slots to a fixed `maxsamps`.

Rewrite while porting, because the old functions sit on pre-multi-location
`model.py`:

- `filt_idx` on every astrometry and parallax call.
- `xS0[filt_idx]` as the East/North pair, never `xS0[0]` as East.
- `obsLocation[filt_idx]`.
- `radiusS` unit: arcsec when `thetaE_amp` exists, Einstein radii otherwise
  (`source_radius_thetaE_units` already states this in `model_jax.py` 23152).
- Apply the same dispatch in **both** `model.py` and `model_jax.py`. The
  speedup on `jax_dex` does not touch `model_jax.py`, and the two files have
  already drifted (AMG return shape, `amg_only`, the `jnp.where` branch).

Do not port:

- `10ay`'s PSTL classes and fitters.
- `10ay`'s FSBL body. `model_jax.py` FSBL is the one the tests call.
- Anything from `bagle-jax`.
- A wholesale `model.py` from `fspl_speedup` (includes `fb5f091`, which
  comments classes out).

The JAX kernel in milestone (b) is new code. The Dex/Abrams work makes the
host reference faster; it does not provide `jax.grad`.

## 3. Staged implementation

Work on a branch `jax_finite_source` cut from current `jax_pymc`. Each
milestone is its own commit series (or its own short-lived branch) and is
shown as a diff before it lands on `jax_finite_source`. Sizes are lines of
diff, not calendar time. They are estimates from the size of the current
methods.

Dependency order: (a) and the NumPy half of (b) can land first and in
parallel. The JAX kernel in (b) blocks (c). (c) blocks (g) and the FSPL half
of (i). (d) blocks BFSBL in (e) and the FSBL half of (i). BFSPL in (e) can
start once (b)'s point-lens image map exists; it does not need the binary
contour. (f) is a constraint on every milestone, not a later project. (h)
is written with the milestone it tests. (j) is last.

### (a) Minimal fixes in the classes that already exist

**Do this before any new physics.**

- Set the limb lens origin from each filter's source origin, or stop
  storing one `xL0` and compute it inside `get_lens_astrometry(t, filt_idx)`
  as `xS0[filt_idx]` minus the source-lens offset. The three sites are
  `model.py` 24266 and `model_jax.py` 25303 and 30984. Delete the duplicate
  `FSPL_Limb` class body in `model_jax.py` (the copy that starts at 30771)
  so one definition remains.
- Make `model.py` `FSPL.get_all_arrays` and `model_jax.py`
  `FSPL.get_all_arrays` choose the same path for the same `astrometryFlag`
  and the same `|u|/rho`. Remove the `jnp.where` call in the NumPy contour
  (`model_jax.py` 23685) or replace it with a NumPy mask. Do not leave a
  function that returns six arrays under the name `get_all_arrays_amg` in
  one file and two arrays in the other.
- Align `BFSPL_PhotAstromParam1.fitter_param_names` (`radiusS`, line 25120)
  with `__init__` (`radiusS_pri`, `radiusS_sec`, `sep`). Until that is
  decided, the fitter cube and the constructor disagree.
- Leave the commented `FSBL` block in `model.py` (line 28465) alone, or
  delete it in a separate commit if Jessica wants the dead ray-shooting
  code gone. It is not required for the JAX work.

**Files:** `src/bagle/model.py`, `src/bagle/model_jax.py`. Tests in
`tests/test_model.py`, `tests/test_model_jax.py`,
`tests/test_model_multi_location.py`.

**Size:** about 200–400 lines, most of it deletion and a shared dispatch.

**Risk:** low for the limb `xL0` change if the tests construct one filter.
Medium if any fixture depended on the six-array AMG return. The limb class
is marked do-not-use, so there may be no test at all; add one.

**Tests:** one-filter limb `xL0` unchanged; two-filter limb uses filter 1's
`xS0[1]` when `filt_idx=1`. FSPL photometry from `model` and `model_jax`
agree to the existing old-vs-JAX tolerance on Param1. BFSPL Param1 can be
constructed with the names the fitter will actually sample.

**Depends on:** nothing.

### (b) Finite-source magnification kernel

Two deliverables, on purpose.

**B1. Host speedup (NumPy).** Port the vectorized AMG and the `|u|` split
from `jax_dex` `FSPL.get_all_arrays` / `amg_numpy`, rewritten for
`filt_idx` and `(n_filters, 2)`. Put the same functions where both model
files can call them, or duplicate them with a comment that they must match.
This is the reference the JAX kernel is checked against. It is not jittable.

**B2. JAX kernel, new, in `src/bagle/jax/fspl.py`.** Recommend a uniform
source limb and the point-lens closed-form images, then the Bozza contour
as `jnp` reductions over a static outline. Reasons:

- The point-lens image map is analytic, so `jax.grad` through it is the
  derivative of a smooth function except exactly at `u = 0`, which a finite
  source does not hit as a single point.
- A static `n_outline` (pass it as a static `jit` argument; a good default
  is the constructor value, often 100–300) keeps shapes static, so `jit`
  and `vmap` over times work. `vmap` over filters is a later step; the
  likelihood already loops filters outside the kernel.
- Adaptive mesh (`jnp.insert`, data-dependent point counts) does not `jit`
  cleanly. Keep AMG as the NumPy reference for epochs with `|u| < 2 rho`,
  and use a finer static outline in JAX for those epochs instead of porting
  the adaptive insert.
- Do not depend on VBBL, pyLIMA, or MulensModel. None of them is in
  `pyproject.toml`. They are optional oracles in tests, skipped if absent,
  the way the current pyLIMA tests use `pytest.importorskip`.

Suggested accuracy targets, to be measured, not claimed yet:

- Relative error of the total magnification against a high-order reference
  (Gauss–Legendre on the Lee 2009 integrands, or `n_outline` several times
  larger) below `1e-3` for `rho` in `[1e-3, 0.1]` and `u` from `0` to `5`,
  including a point inside the source (`u < rho`).
- Point-source limit: for `rho <= 1e-3` and `u > 10 rho`, absolute error
  against `(u^2 + 2) / (u sqrt(u^2 + 4))` below `1e-4`.
- `jax.grad` of magnification with respect to `u` and `rho` finite for
  `rho >= 1e-3` and `u` away from the origin. At `u = 0` the two images are
  symmetric; the gradient of the total amplification can be zero by
  symmetry and should still be finite.
- One `jit` of a `(n_times=256, n_outline=128)` limb on CPU. Record the
  compile time in the test output. I have no measurement now. If compile
  exceeds a few seconds, the outline length is the knob, not a custom
  sparse solver.

`jax.vmap` over the time axis inside the jitted function. Do not `vmap` over
a Python list of filters inside the kernel; the fitter already handles
filters outside.

**Files:** `src/bagle/jax/fspl.py` (replace the host wrappers with a kernel
plus a thin host fallback), `src/bagle/jax/__init__.py` if it exports
names, new `tests/test_fspl_kernel.py`.

**Size:** B1 about 300–500 lines. B2 about 400–800 lines plus about 300
lines of tests.

**Risk:** the Bozza parabolic correction uses second differences of the
image track. At the point where the minor image crosses the lens, the track
is sensitive to rounding. The `fspl_speedup` message says a contour
artifact was fixed; whoever ports B1 should read that diff (`c913b38`) and
keep the fix, and should say in a comment what the artifact was. I did not
extract the numerical change from the diff. Uniform-source FSPL does not
need limb-darkening coefficients. Those stay in milestone (f) or a tail of
(c).

**Tests:** the accuracy targets above, a `jit`+`vmap` smoke test, and a
finite-difference check of `jax.grad` against the NumPy kernel to about
`1e-4` relative on `u` and `rho`.

**Depends on:** (a) for a stable host reference. B2 can be written against
the current host if (a) slips, but then the reference itself is the slow
AMG.

### (c) FSPL model, parameters, and the fitter

Add `jax_log_likely_photometry` and, on the phot+astrom Param classes,
`jax_log_likely_astrometry` on `FSPL_PhotParam1`, `FSPL_PhotParam2`,
`FSPL_PhotAstromParam1`, and `FSPL_PhotAstromParam2` in `model_jax.py`.
Follow the PSPL pattern: the class method reads one parameter vector in
class order, calls the kernel, applies per-filter `b_sff` and `mag_src`.
Geometry (`t0`, `u0`, `tE`, `piE`, or the physical `mL`/`beta`/`dL` set)
should reuse `bagle.jax.geometry` where the PSPL derivation already exists,
then append `radiusS`.

`rho` is not a new public name. Phot parameterizations already store
`radiusS` in Einstein-radius units. Phot+astrom stores arcsec and converts
with `source_radius_thetaE_units`. The kernel should take `rho` in Einstein
radii so both paths share it. Document that in the Param docstrings.

Priors, in both fitters:

- Phot: keep a dimensionless prior. The current `make_gen(1e-4, 1e-2)` is
  a plausible `rho` if and only if it is applied to phot classes. Say so in
  the prior table.
- Phot+astrom: a separate prior in arcsec, or sample `log10_rho` and
  convert. Do not reuse `1e-2` arcsec. This needs Jessica's choice
  (open question 4).
- `n_outline` stays fixed. Putting it in the cube makes the JAX shape
  data-dependent.

The eight concrete leaves already exist. No new leaf is required for the
first FSPL fit. Parallax is already a flag class (`FSPL_Parallax`); the
kernel must accept a parallax vector the same way PSPL does, evaluated in
NumPy outside the `jit` or via the existing JAX parallax helper if one is
pure. I did not check whether `parallax.parallax_in_direction` is JAX.
Assume it stays a precomputed `parallax_vectors` argument, which is what
the current `jax_log_likely_*` signatures take.

**Files:** `model_jax.py` (four Param classes), `model_fitter.py` and
`model_fitter_jax.py` (prior table only), `src/bagle/jax/fspl.py`.

**Size:** about 400–700 lines. The likelihood methods are short if the
kernel is done; the risk is the physical-parameter geometry, which is
larger than the contour.

**Risk:** phot+astrom Param1's derived `thetaE_amp` has to be computed
inside the traced function, not read off a host instance, or the gradient
with respect to `mL` and `dL` is zero. Copy the PSPL derived-geometry
pattern. `tie_fit_param` on `xS0_E` / `xS0_N` must use the source-slot
index, or the tied-origin bug comes back.

**Tests:** NumPy `get_photometry` versus the traced log-likelihood's
predicted magnitudes, relative `1e-5` away from the limb and `1e-3` on a
limb-crossing epoch. `jax.grad` of the log-likelihood finite and nonzero
in `radiusS` and `t0`. One filter and two filters, second filter's `xS0`
tied to the first.

**Depends on:** (b) B2. (a) for the host comparison.

### (d) FSBL

Do not `jit` the current `images_on_source_limb`. The plan for a sampler
kernel:

1. Fixed limb, `n_outline` static, no `jnp.insert`, no `jax.random`.
2. Reuse the existing quintic (`get_image_pos_arr`) and a static version of
   `polygonal_area_with_parity` / `integrate_unif`. The commented
   `polygonal_area` at line 25442 is a sketch of the Green's theorem form;
   the live integrator is `polygonal_area_with_parity` at 25742.
3. Image-track assignment (which root is which image) is the fragile part.
   A permutation that flips when two roots cross will make the centroid
   jump even if the total magnification is continuous. For photometry, sum
   of absolute inverse Jacobians over roots that land on the source is
   safer than labeled tracks. For astrometry, labeled tracks are required,
   and the gradient will be wrong across a caustic unless the assignment is
   smoothed. State that limit in the docstring.
4. Cost: a quintic eigendecomposition per limb point per time. For
   `n_times = 256` and `n_outline = 64` that is about `1.6e4` complex 5×5
   eigenproblems per likelihood call. That is acceptable for nested
   sampling of a single event and expensive for long NUTS. Record one
   compile and one call in the test. I have no number now.
5. Caustic policy for gradients: ship the kernel, test `jax.grad` on a
   trajectory that does not cross a caustic, and test that a caustic
   crossing returns a finite magnification. Do not require a finite
   gradient on the caustic in the first version. Document that NUTS and
   MCLMC are the wrong default for a caustic-crossing FSBL, and that
   MultiNest, numpyro NS, BlackJAX NS, and SMC are the ones to use.
   Replica exchange would differentiate every rung, so it inherits the
   same warning.

Ray shooting and hexadecapole are alternatives if the contour's image
assignment cannot be made stable. Ray shooting is easy to write and hard to
differentiate (a pixel is inside or outside). Hexadecapole (Gould 2008) is
a point-lens expansion and does not apply to FSBL. I would not start with
either. Contour integration is already what this file is trying to do.

**Files:** new `src/bagle/jax/fsbl.py`; thin calls from `FSBL` in
`model_jax.py`; `jax_log_likely_*` on `FSBL_PhotParam1` and
`FSBL_PhotAstromParam1` first. Orbit and Param2–8 wait until Param1 matches
the host to the test tolerance. Do not try to cover every orbit leaf in
the first FSBL diff.

**Size:** about 800–1500 lines for a static-limb phot kernel and Param1
likelihood. Each additional parameterization is closer to 100–200 lines if
it only changes geometry.

**Risk:** high. Root polish, parity, and caustic topology. The host method
itself has not been compared to VBBL in this survey. The first test should
be host-versus-new-kernel on a wide binary that does not cross a caustic,
then a close binary that does.

**Depends on:** (b) for shared complex-arithmetic helpers and the test
style. Independent of FSPL photometry.

### (e) Binary-source variants

**BFSPL.** Port the Lee integrands (`function_photometry_one`,
`function_photometry_two`, the Simpson sums in `model.py` 24286–24570) to
`jnp` with a fixed `n = 20` or, better, a static Gauss–Legendre grid so the
accuracy target is explicit. Two calls per epoch (primary and secondary),
each with its own `rho` and `u`. Combine fluxes with `mag_src_pri[filt_idx]`
and `mag_src_sec[filt_idx]`. `b_sff[filt_idx]` is the combined source flux
fraction, consistent with BSPL. Fix the Param1 name mismatch from (a)
before sampling. Add the missing `BFSPL_PhotAstrom_Par_Param1` leaf only
when the no-parallax leaf matches; parallax is a one-line flag if `get_u`
already honors it. I believe `get_u` does, because BFSPL subclasses PSPL.
Confirm when implementing.

There is no phot-only BFSPL. Add one only if Jessica wants it. The
magnification does not need astrometry, so a phot leaf is a thin subclass.

**BFSBL.** New. Nothing to port. A first version is FSBL's static contour
evaluated at two source centers, with `rho_pri`, `rho_sec`, `sep`, `phi` or
the physical binary-source offsets, and the same flux combination as BFSPL.
This is the most expensive model in the set (two contours, each a binary
lens). Do it after (d) is trusted. Do not start it by copying the dead
commented FSBL in `model.py`.

**Files:** `model.py`, `model_jax.py`, `src/bagle/jax/fspl.py` (BFSPL) or a
small `bfspl.py`, later `src/bagle/jax/bfsbl.py`.

**Size:** BFSPL kernel and one likelihood about 400–700 lines. BFSBL first
phot+astrom leaf about 600–1000 lines on top of (d), mostly bookkeeping.

**Risk:** BFSPL medium (the integrals already exist; the bug surface is the
parameter names and the flux sum). BFSBL high, and it should not block the
FSPL release.

**Depends on:** BFSPL on (b). BFSBL on (d) and on the BFSPL flux convention.

### (f) Multi-location, astrometry, GP

Most of this is already the calling convention. The work is to not break it.

- Every new `get_*` and every `jax_log_likely_*` takes `filt_idx` and reads
  `xS0[filt_idx]`, `b_sff[filt_idx]`, `mag_src[filt_idx]`,
  `obsLocation[filt_idx]`.
- Finite-source astrometry is the magnification-weighted image centroid
  minus the unlensed source, in the filter's coordinate origin. FSPL
  already computes that centroid in the Bozza integrals. The JAX kernel
  must return centroids, not only the scalar magnification, or the
  astrometric likelihood cannot be traced.
- Limb darkening: keep linear `utilde` as an optional fixed coefficient
  (not sampled) on FSPL, implemented as a radial quadrature of the uniform
  kernel (the structure of `FSPL_Limb.get_amplification`, but with `vmap`
  over a static set of annuli). One `utilde` per event is enough until
  someone has per-filter coefficients. Do not add Claret coefficients in
  this round. The limb class stays "do not use" until the quadrature
  matches the uniform kernel at `utilde = 0` and the `xL0` fix from (a) is
  in.
- GP: no FSPL GP class exists. After (c), a GP leaf is the PSPL GP
  parameterization plus `radiusS`, and the GP itself stays in `bagle.jax.gp`
  (tinygp) outside the magnification kernel. FSBL GP parameter classes
  exist without leaves; add a leaf only after FSBL Param1's likelihood
  exists. Do not build GP into the contour.
- Geoproj / RefPar: not in the first six milestones. A later leaf can reuse
  `get_geoproj_ast_params(t0par, filt_idx)`.

**Files:** the same model files, `src/bagle/jax/gp.py` only if a leaf is
added, tests in `tests/test_model_multi_location.py`.

**Size:** astrometric centroid in the FSPL kernel is part of (b)/(c), about
150 extra lines. Linear limb darkening about 200 lines. One GP leaf about
150 lines.

**Risk:** low if it is only convention. Medium for the centroid, because
image labeling errors move the centroid by an Einstein radius even when the
magnification is right.

**Depends on:** (a), (b), (c). GP leaf depends on (c). FSBL centroid depends
on (d).

### (g) Fitters

Once `jax_log_likely_*` exists, `evaluate_loglik_jax` and the solvers pick
it up through the existing `getattr`. No new solver class.

Check, rather than redesign:

- The traced function reads tied suffixes from the source slot
  (`_class_vector_slots`). Finite-source tests must include a tied `xS0`.
- `evaluate_loglik_jax` on a batch `vmap`s that function. The kernel's
  `n_outline` must be static across the batch. Non-finite log-likelihoods
  already map to `-1e300` on the batch path; a NaN magnification should hit
  that path and not a NaN gradient. Prefer the kernel to return a large
  finite penalty rather than NaN.
- MultiNest does not need the gradient. It can keep calling the host model.
  NumPyro NUTS, BlackJAX NUTS, and BlackJAX MCLMC need (c) or (d).
  BlackJAX SMC and nested sampling, nautilus, and pocoMC need a finite
  scalar, not a gradient. Replica exchange needs the gradient on every rung.
- Priors from (c) and (e) go in both `model_fitter.py` and
  `model_fitter_jax.py`.

**Files:** `model_fitter_jax.py` only if the lookup or the prior table
changes; `model_fitter.py` for the prior table and the "in-progress" sentence.

**Size:** about 50–150 lines if (c) did the likelihood. More if the class
vector order for `radiusS_pri` / `radiusS_sec` needs a new slot helper.

**Risk:** low. The failure mode is a silent fallback to a missing method.

**Depends on:** (c) for FSPL, (d) for FSBL, (e) for the binary-source names.

### (h) Tests

Put these next to the milestone that should go red without them.

| Check | Where | Tolerance to start from |
| --- | --- | --- |
| `model` vs `model_jax` host photometry and astrometry | existing old-vs-JAX tests | existing tolerances; do not loosen them |
| JAX kernel vs host | `tests/test_fspl_kernel.py`, later `test_fsbl_kernel.py` | relative `1e-3` on magnification, `1e-3` arcsec or the host tolerance on the centroid |
| `jax.grad` vs central differences | same files | relative `1e-4` on `u`, `rho`, `tE` |
| Point-source limit `rho -> 0` | FSPL kernel test | absolute `1e-4` on the analytic PSPL magnification for `u > 10 rho` |
| Inside the source, `u = 0`, uniform disk | FSPL kernel test | the known finite-source value at zero impact; compute it from the reference integrator, do not hard-code a remembered constant without a citation |
| Lee (2009) or a long Gauss–Legendre integral | BFSPL | relative `1e-3` |
| VBBL, MulensModel, pyLIMA | optional, `importorskip` | only if the package imports; pyLIMA is already skipped in this environment |
| Caustic crossing stays finite | FSBL | magnification finite; gradient not required |
| Two filters, tied origin | fitter test, like the existing tied-`xS0` test | log-likelihood agrees with NumPy at the usual `rtol=1e-5`, `atol=1e-4`, and the gradient w.r.t. the source slot is finite and nonzero |
| `utilde = 0` matches uniform FSPL | limb test | relative `1e-3` |

I did not run VBBL, MulensModel, or pyLIMA here. They are not installed in
this environment.

### (i) Timing

There is no file named `BAGLE_time_tests`. The timing harness on this branch
is `tests/psbl_sampler_compare/run_comparison.py`. It has `PSPL_SCENARIOS`
and `SCENARIOS` (PSBL). Commit `eff7332` ("Finished all time tests comparing
the inference engines") wrote results under `tests/psbl_sampler_compare/runs/`.

Add two scenarios in that style, after the likelihood exists:

- `fspl_rho0p01`: one filter, photometry, `rho` about `0.01`, no caustic
  (point lens). Compare host NumPy and the JAX likelihood, then one short
  nested-sampling run if MultiNest or numpyro NS imports.
- `fsbl_wide`: binary lens, source radius small enough that the trajectory
  misses the caustics, so a gradient sampler is allowed to run. A second
  scenario `fsbl_caustic` is timed only with a gradient-free sampler.

Do not check in the run products (png, fits, html). `.gitignore` already
ignores time-test outputs from `eff7332`'s follow-up `467774c`.

**Size:** about 150–250 lines in `run_comparison.py` and a short note in
`tests/PSBL_SAMPLER_COMPARE.md`.

**Depends on:** (c) and (g) for FSPL; (d) and (g) for FSBL.

### (j) Docs and a notebook

After the FSPL kernel and one fitter path work:

- `docs/new_models.rst`: replace the finite-source placeholder. State
  `radiusS` units per parameterization, `b_sff` as source flux fraction,
  and that NUTS is not the default for caustic-crossing FSBL.
- One notebook `docs/notebooks/example_FSPL.ipynb`, executed, with a single
  filter and a short light curve. Do not call `solve()` unless MultiNest is
  actually installed. A plot of model photometry and the centroid shift is
  enough.
- Attribution comment at the top of the ported NumPy helper: Natasha Abrams
  (`c913b38`), Dex Bhadra (`e1bc6ec`), Bozza et al. (2021) for the contour,
  Lee et al. (2009) for BFSPL, Witt (1995) for the quintic, and the Bartolic
  `caustics` library as the inspiration for the existing FSBL contour (not
  as vendored code).

**Size:** a few hundred lines of rst and one notebook.

**Depends on:** (c), and (d) only for the FSBL paragraph.

## 4. Risks and open questions

1. **Accuracy versus speed.** The host AMG is adaptive and slow. The JAX
   plan uses a static outline. A static outline that is fine for `rho = 0.01`
   can be wrong for `rho = 0.1` near a cusp, or wasteful for `rho = 1e-3`.
   `n_outline` has to be chosen per fit and reported. I have no timing numbers
   from Dex's branch or from this tree.

2. **Compile time.** Unknown. A static outline of a few hundred points, vmapped
   over a few hundred times, is one XLA program of dense linear algebra (the
   FSBL quintic) or of elementary functions (FSPL). Expect the first call to
   dominate a unit test. Do not `jit` a new shape for every `n_outline` the
   user might type; cache a small set.

3. **Caustics.** Finite-source magnification is finite on a caustic.
   The derivative is not a friendly HMC target: image creation, parity flips,
   and segment classification are piecewise. Recommendation: gradient samplers
   for FSPL and for FSBL trajectories that stay off the caustic; nested
   sampling or SMC for caustic crossings. Please confirm before anyone
   wires FSBL into NUTS by default.

4. **`radiusS` units and the default prior.** Phot classes: Einstein radii.
   Phot+astrom classes: arcsec (`model.py` 23531 versus 23790). One prior
   `make_gen(1e-4, 1e-2)` cannot mean both. Which prior should a phot+astrom
   fit use?

5. **Limb darkening.** The only law in the tree is linear in `utilde`.
   Is that the law to support, and is `utilde` fitted or fixed? I recommend
   fixed, one value per event, after the uniform kernel works.

6. **External libraries.** I recommend not depending on VBBL, MulensModel,
   or pyLIMA. Use them as optional tests. Bartolic's `caustics` is the
   stated inspiration for FSBL and is not imported. Copying that repository
   in would add a license question I have not checked. Keep the equations
   that are already written in `model_jax.py`.

7. **Attribution.** Module authors listed at the top of `model.py` include
   Jessica Lu, Michael Medford, Casey Lam, Dex Bhadra, and Edward Broadberry.
   The FSPL vectorization to port is Natasha Abrams, `c913b38`, with Dex
   Bhadra's port `e1bc6ec` / `2e42e1a`, and the commit message credits Cursor
   as well. `10ay` wrote the PSTL and FSBL commits on `jax_dex`; I am not
   recommending those be ported. Hunter Harling wrote the 2024 `bagle-jax`
   prototype; do not merge it.

8. **Not verified.** I did not run the finite-source tests, `jax.grad`, or
   `jax.jit`. I did not measure the Dex speedup. I did not diff `c913b38`
   line by line to name the contour artifact. I did not confirm that every
   old-vs-JAX FSBL "parity" row has a real NumPy twin (there is no live
   `FSBL` in `model.py`). I did not finish reading BFSPL's flux sum past
   the image assembly at `model.py` 24590. `BFSPL_PhotAstromParam1`'s
   `fitter_param_names` versus `__init__` mismatch is from reading the
   signature, not from constructing the class.

9. **Duplicate limb class and the `jnp.where` branch** are real defects in
   the current tree, described in section 1. They are small enough to fix
   in (a), and they should not be mixed into the kernel diff.

## 5. Branch strategy

Cut `jax_finite_source` from `jax_pymc` at `64cd87e` or later, when this
plan is accepted. Do not merge `jax_dex`, `fspl_speedup`, `new_fspl`, or
`bagle-jax`.

Bring the Dex/Abrams FSPL speedup in by porting the functions named in
section 2.4, with the attribution comment. A cherry-pick of `e1bc6ec` is a
1172-line rewrite of an old `model.py` and will conflict on the
multi-location `xS0` code. A merge is worse: 136 commits the other way, and
the speedup never enters `model_jax.py`.

Keep `jax_finite_source` mergeable by landing one milestone at a time, each
as a diff Jessica can read before it hits `jax_pymc`:

| Branch (suggested) | Lands on `jax_finite_source` | Review focuses on |
| --- | --- | --- |
| `jax_finite_source_a` | (a) limb `xL0`, duplicate class, dispatch alignment, BFSPL names | behavior of existing tests |
| `jax_finite_source_b` | (b) NumPy vectorization, then the JAX FSPL kernel | accuracy table and one `jit` timing |
| `jax_finite_source_c` | (c) FSPL `jax_log_likely_*` and priors | gradient and two-filter tie |
| `jax_finite_source_d` | (d) static-limb FSBL, Param1 only | non-caustic agreement, finite magnification on a caustic |
| `jax_finite_source_e` | (e) BFSPL kernel; BFSBL only after (d) is accepted | two radii, flux fraction |
| `jax_finite_source_fg` | (f) centroid, linear limb, optional GP leaf; (g) fitter wiring | solvers see the new methods |
| `jax_finite_source_hij` | (h) leftover tests, (i) scenarios, (j) docs and notebook | the notebook runs |

Those names are proposals. Short-lived `cursor/` branches are fine if that
is how review is done; the long-lived line is `jax_finite_source`. Each
merge into `jax_pymc` should be one of these diffs, not all of them at once.

## Recommendation

Build a new static-shape JAX FSPL kernel and wire the eight FSPL leaves that
already exist. Use the Abrams/Bhadra NumPy vectorization as the fast host
reference, ported onto the multi-location calling convention, and do not
merge `jax_dex` or `fspl_speedup`. Treat the current FSBL code as a host
algorithm that uses JAX ops internally and is not a sampler likelihood until
the adaptive limb and the random perturbation are removed. Add BFSPL's Lee
integrals in JAX after FSPL. Add BFSBL only after the FSBL kernel is trusted.
Leave NUTS off caustic-crossing fits until the gradient across a caustic has
been measured.
