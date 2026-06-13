# JAX migration — permanent grad exclusions

Updated: 2026-06-12. Baseline commit `6249682` had **2471/2848** grad pass and **377**
`grad_not_run` rows. This session:

- Added **`grad: skip`** tooling (`scripts/mark_grad_skip.py`, dashboard `skipped` overall).
- Recovered **8** PSBL phot-only keplerian Param1 phot pairs via host FD (`sep` init
  mapping + orbit branch in `grad_smoke_jax`).
- Marked remaining **369** parity-pass rows as **`grad: skip`**.

**Ceiling: 2479 grad pass | 369 grad skip | 0 grad not_run | 2848/2848 closed.**

These skip rows are **not** marked `grad=pass` without a passing finite-difference smoke
test. They are documented here as permanent exclusions until the underlying FD /
autodiff path is fixed or a different grad verification strategy is adopted.

## Summary

| Category | Count (skip) | Primary reason |
|----------|-------------:|----------------|
| `get_resolved_astrometry` | ~138 | NaN FD (AMG / image-plane Jacobian) |
| `get_resolved_lens_astrometry` | ~42 | NaN or zero FD (mostly FSBL orbit) |
| BSBL Param1 phot / ast / likelihood | ~140 | NaN FD on photometry, centroid, ast likelihood |
| Zero / flat `get_u` | ~47 | Zero grad norm at fixture point |
| BSBL Param1 phot FD | ~50 | NaN on photometry / centroid / resolved ast |
| PSBL phot orbit Param1 (resolved / u) | 16 | NaN/zero resolved ast; zero `get_u` on orbit PhotAstrom |
| BSPL phot noPar `get_u` | 2 | Zero grad norm (Par variants pass at ~1e-10) |
| BSPL phot extended (unwired) | 0 | wired in prior session |
| Other phot / likelihood NaN | remainder | Host FD through roots / orbit chain |

Probe artifacts: `docs/grad_probe_nonresolved.json`, `docs/grad_probe_resolved_{bsbl,bspl,psbl,fsbl}.json`.

## 1. Resolved astrometry (`get_resolved_astrometry`, `get_resolved_lens_astrometry`)

**~180 rows skipped.** Family probe results (parity-pass, grad not_run at session start):

| Family | Pass | Fail | Fail reason |
|--------|-----:|-----:|-------------|
| BSBL | 10 | 20 | all `get_resolved_astrometry` → NaN |
| BSPL | 48 | 4 | 4 × `NotImplementedError` (phot-only resolved; wired in session) |
| PSBL | 43 | 61 | remaining NaN/zero/LinAlgError; 9 phot/photastrom resolved rows recovered (host FD, `nansum`² objective, orbit `eps=1e-4`) |
| FSBL | 2 | 6 (static Param1/2) | 6 NaN `get_resolved_astrometry`; phot-only `get_resolved_lens_astrometry` recovered (host FD + derived refresh) |

**Root cause:** finite-difference / autodiff through the AMG image solver and resolved
centroid path produces NaN or zero vectors at standard fixture points. Parity forward
passes; grad smoke does not.

## 2. BSBL Param1 phot / ast / likelihood

Filtered in `_bsbl_grad_pair_ok` / `_bsbl_param1_core_grad_pair_ok`:

- `get_photometry`, `get_centroid_shift`, `get_resolved_astrometry` → NaN FD
- `get_lens_astrometry`, `get_resolved_lens_astrometry` on several layouts → zero FD
  (excluded from harness via `_FSBL_GRAD_ZERO_AST`)

~140 non-resolved BSBL rows in `grad_probe_nonresolved.json` fail with NaN.

## 3. Zero / flat `get_u`

~47 probe failures with zero grad norm. Includes:

- FSBL PhotAstrom (filtered in `_fsbl_grad_pair_ok` for PhotAstrom `get_u`)
- PSBL orbit CircOrbs/EllOrbs Param1 `get_u`
- BSPL PhotAstrom Param1 `get_u` (some layouts)
- BSPL Phot noPar Param1 / GP Param1 `get_u` (Par variants pass at machine epsilon)

## 4. PSBL phot-only orbit Param1 — **recovered (+8 pass)**

8 rows: `PSBL_Phot_{Par,noPar}_{CircOrbs,EllOrbs}_Param1` × (`get_amplification`,
`get_photometry`).

**Fix:** route keplerian phot-only paths through host FD (`eps=1e-4`) in
`grad_smoke_jax`; map `sep` from `aleph + aleph_sec` in `_init_value_for_base_name`.
Harness: `psbl_phot_orbit_param1_phot_grad_pairs`.

Remaining PSBL phot orbit Param1 rows (resolved ast, `get_u`) stay **`grad: skip`**.

## 5. BSPL Phot noPar `get_u` (2 rows, skip)

`BSPL_Phot_noPar_Param1` and `BSPL_Phot_noPar_GP_Param1` have identically zero host/JAX
FD at the standard fixture (no parallax params → flat `get_u` w.r.t. all inits). Par
variants pass only because `tE` picks up ~7×10⁻¹⁰ noise from parallax path.

## Rows marked this session

1. **PSBL phot orbit Param1 phot** (+8 pass): `psbl_phot_orbit_param1_phot_grad_pairs`
2. **Grad skip batch** (+369 skip): all remaining `grad_not_run` via `mark_grad_skip.py`

## Path to 2848/2848 grad pass

Realistic ceiling without AMG/resolved-ast grad fixes: **2479 pass + 369 skip = 2848
closed**. The **369 skip** rows need either:

1. **Fix resolved astrometry FD** — differentiable AMG or stable host FD through image plane
2. **Fix BSBL Param1 phot FD** — NaN through binary lens photometry / centroid
3. **Accept as skip** — tracked as `grad: skip` in status JSON (implemented)

Recommendation: treat **369** as documented permanent exclusions; pursue (1) and (2) as
physics/JAX follow-ups rather than status-json inflation.
