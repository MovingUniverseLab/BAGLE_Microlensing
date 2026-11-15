# Changelog

## Unreleased

### Phot-only binary-source along-track (Option A)

- Phot-only `BSPL` / `BSBL` / `BFSPL` `get_u` now keeps the full static
  `(sep, phi)` offset: `u_sec(t) = u_pri(t) + sep_vec` with a shared
  primary epoch. The secondary no longer always peaks at `t0_pri` when
  the binary axis has an along-track component. Attribute `u0_sec` is the
  snapshot at `t0_pri`; `u0_amp_sec` / `get_t0_sec()` remain derived
  closest-approach diagnostics only (`model.py`, `model_jax.py`,
  `jax/geometry.py`, `jax/bspl.py`).

### LinOrbs / AccOrbs lens placement

- Binary-lens `LinOrbs` / `AccOrbs` `get_resolved_lens_astrometry` now
  places the primary at `xL0 + (sep/2)(sin α, cos α)`, matching the static
  and Keplerian classes (it previously used the opposite sign). Primary-
  input Param6 / Param7 conversions already assumed the static side, so
  their input frames are honored after this fix (`model.py` and
  `model_jax.py`).

### BSPL Keplerian `xS0` at `t0`

- `BSPL_PhotAstrom_EllOrbs` / `CircOrbs` Param1–4 now set
  `xS0_com = xS0_pri - d_pri(t0)` from `oal2xy` at the input `t0`, so the
  primary track recovers the input `xS0`. `alpha` / `sep_t0` are the
  projected primary→secondary axis at `t0`; `sep = aleph + aleph_sec`
  remains the relative semi-major axis.

### Binary reference frames (PSBL / FSBL / BSBL / BFSBL)

Fit results for the classes below change: their constructors now honor the
input frame exactly and fill every frame consistently. See
`docs/audits/binary_frames/` for the audit and the plan.

- **Suffix rule.** `_com` = source CoM to lens CoM, `_p` = primary source to
  primary lens, `_mid` = primary source to lens geometric midpoint. The
  suffix applies to `t0_*`, `u0_amp_*` / `beta_*` and the source position;
  the input `xS0` is always the input source point at the input `t0`.
- **Keplerian internal frame = CoM.** `EllOrbs` / `CircOrbs` PhotAstrom
  classes store the CoM pair in `t0` / `u0_amp` / `u0` / `beta` / `xL0` (and
  `xS0` at `t0_com`). The midpoint pair (`t0_mid`, `u0_amp_mid`, `beta_mid`,
  `xL0_mid`) and the primary pair (`t0_p`, `u0_amp_p`, `beta_p`, `xS0_p`)
  are root-solved on first access and cached (`bagle.binary_frames`), so
  construction and likelihood evaluation never root-solve. PSBL Param2 /
  Param3 (midpoint inputs) keep the midpoint pair as their internal frame.
  Static, linear and accelerated models are unchanged; their `t0_mid` etc.
  alias `t0` etc. Phot-only Keplerian models keep the midpoint frame.
- **BSBL / BFSBL Keplerian Param1 / 2 / 3 rewritten** on the new
  `frame_convert.convert_u0_amp_t0_bsbl_orbit` (both orbits), replacing the
  static-binary `q_eff` shifts. Param2 now takes CoM-CoM inputs (its
  `t0_com` / `u0_amp_com` / `xS0` used to refer to the primary source).
  `alphaL` / `alphaS` and new `sepL_t0` / `sepS_t0` are reported at `t0`
  instead of `tp`; `alphaS` points from the primary to the secondary
  source, as in the static classes.
- **BSBL / BFSBL static Param2**: `xS0` is now the primary source at the
  input `t0_p`; `u0_amp` keeps its sign; new CoM attributes.
- **Input renames** (`fitter_param_names` change):
  - PSBL / FSBL Keplerian Param1, 4, 8: `xS0_E/N` → `xS0_com_E/N`.
  - PSBL / FSBL Keplerian Param7: `xS0_E/N` → `xS0_p_E/N`.
  - BSBL / BFSBL Keplerian Param1: `beta` → `beta_com`,
    `xS0_E/N` → `xS0_com_E/N`; Param2: `xS0_E/N` → `xS0_com_E/N`;
    Param3: `xS0_E/N` → `xS0_p_E/N`.
  - BSBL / BFSBL static Param2: `xS0_E/N` → `xS0_p_E/N`, and a new input
    `qS` (source mass ratio `mSs / mSp`) after `alphaS`.
- **Attribute renames:** `mass_sources`, `mass_source_primary`,
  `mass_source_secondary` → `mS`, `mSp`, `mSs` (floats, Msun), plus
  `qS = mSs / mSp`; `t0_geo` → `t0_mid`. The `frame_convert` coords string
  `'geom_mid'` is unchanged.
- **Fitter:** priors for `xS0_com_E/N`, `xS0_p_E/N` (data-driven, like
  `xS0_E/N`) and `qS` (`make_gen`, 0-10); the unused `xS0_system_E/N`
  priors are retired.
- **Docs:** `Orbit.oal2xy` and `bagle.jax.orbits.oal2xy` document the angle
  conventions (Omega from North through East, shared ascending node, omega
  along the motion, secondary at omega + 180).
- Removed the orphaned `bagle/jax/bsbl.py` (imported a module that no
  longer exists).
