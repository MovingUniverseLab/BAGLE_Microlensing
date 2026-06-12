# JAX migration — permanent grad exclusions

Updated: 2026-06-12. Baseline commit `1a46710` had **2283/2848** full_done and **565**
`grad_not_run` rows (parity pass, grad not verified). This session marked **188** additional
passes (**2471/2848** full_done); **377** rows remain `grad_not_run`.

These rows are **not** marked `grad=pass` without a passing finite-difference smoke test.
They are documented here as permanent exclusions until the underlying FD / autodiff path is
fixed or a different grad verification strategy is adopted.

## Summary

| Category | Count | Primary reason |
|----------|------:|----------------|
| `get_resolved_astrometry` | 138 | NaN FD (AMG / image-plane Jacobian) |
| `get_resolved_lens_astrometry` | 42 | NaN or zero FD (mostly FSBL orbit) |
| BSBL Param1 phot / ast / likelihood | ~140 | NaN FD on photometry, centroid, ast likelihood |
| Zero / flat `get_u` | ~47 | Zero grad norm at fixture point |
| BSBL Param1 phot FD | ~50 | NaN on `get_photometry`, `get_centroid_shift`, `get_resolved_astrometry` |
| PSBL phot orbit Param1 | 8 | `ValueError`: cannot map base fitter `sep` |
| BSPL phot noPar `get_u` | 2 | Zero grad norm (Par variants pass) |
| BSPL phot extended (unwired) | 2 | `NotImplementedError` before host-FD wiring |
| Other phot / likelihood NaN | remainder | Host FD through roots / orbit chain |

Probe artifacts: `docs/grad_probe_nonresolved.json`, `docs/grad_probe_resolved_{bsbl,bspl,psbl}.json`.

## 1. Resolved astrometry (`get_resolved_astrometry`, `get_resolved_lens_astrometry`)

**~180 rows remaining.** Family probe results (parity-pass, grad not_run at session start):

| Family | Pass | Fail | Fail reason |
|--------|-----:|-----:|-------------|
| BSBL | 10 | 20 | all `get_resolved_astrometry` → NaN |
| BSPL | 48 | 4 | 4 × `NotImplementedError` (phot-only resolved; wired in session) |
| PSBL | 34 | 70 | 58 NaN, 8 zero, 4 LinAlgError |
| FSBL | — | ~86 | probe incomplete (orbit resolved very slow); Param1/2 static `get_resolved_astrometry` → NaN |

**Root cause:** finite-difference / autodiff through the AMG image solver and resolved
centroid path produces NaN or zero vectors at standard fixture points. Parity forward
passes; grad smoke does not.

**Pairs that pass** (marked in session): BSBL Param1 `get_resolved_lens_astrometry` (10),
BSPL resolved batch (48), PSBL resolved subset (34).

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
- BSPL Phot noPar Param1 / GP Param1 `get_u` (Par variants pass)

## 4. PSBL phot-only orbit Param1

8 rows: `PSBL_Phot_{Par,noPar}_{CircOrbs,EllOrbs}_Param1` × (`get_amplification`,
`get_photometry`).

**Reason:** `ValueError: cannot map base fitter 'sep' from init parameters` — grad_smoke
layout registry gap for keplerian phot-only classes.

## 5. FSBL PhotAstrom `get_u` (non-zero subset)

Session marked **36** FSBL PhotAstrom `get_u` pairs that pass probe. Remaining FSBL
`get_u` / lens-ast failures are zero-FD or resolved-astrometry related.

## Rows marked this session (+188)

1. **Non-resolved probe batch** (+86): `grad_probe_nonresolved_pass_pairs`
2. **Resolved probe batches** (+92): `grad_probe_resolved_pass_pairs` (BSBL/BSPL/PSBL)
3. **BSPL phot extended host-FD** (+10): `bspl_phot_extended_probe_grad_pairs`
   (wired `bspl_phot` → `_fd_grad_host` for `get_u`, chi2, log-likelihood)

## Path to 2848/2848

Realistic ceiling without AMG/resolved-ast grad fixes: **~2471 + 0 = 2471** (current).
The remaining **377** rows need either:

1. **Fix resolved astrometry FD** — differentiable AMG or stable host FD through image plane
2. **Fix BSBL Param1 phot FD** — NaN through binary lens photometry / centroid
3. **Wire PSBL phot orbit Param1** — add `sep` mapping in grad_smoke layout registry
4. **Accept exclusions** — track as `grad: skip` in status JSON (not implemented in
   `mark_old_vs_jax_pairs.py` yet) or exclude from applicable task count

Recommendation: treat **377** as documented permanent exclusions for the migration dashboard;
pursue (1) and (2) as physics/JAX follow-ups rather than status-json inflation.
