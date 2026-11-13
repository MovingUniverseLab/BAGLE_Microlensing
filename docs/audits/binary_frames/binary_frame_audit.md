# Binary-lens reference-frame audit

**Convention under test.** Internally, `self.t0`, `self.u0` / `u0_amp` and
`self.xL0` always describe the source's closest approach to the
**geometric midpoint** of the lens pair, and every class also stores the
lens **CoM** closest approach (`t0_com`, `u0_amp_com`, `xL0_com`) where
masses exist. Inputs may be given about the midpoint, the CoM, or the
**primary**, and must be honored in that frame. For orbiting lenses the
midpoint and primary move, so their `(t0, u0)` are defined at an instant:
at `t0` the unlensed source is `u0` from the point's position *at that
time*, perpendicular to the relative proper motion. The CoM moves on a
straight line, so its closest approach is exact. Lens orbits are always
evaluated about the CoM.

**Result.** All PSBL / FSBL parameterizations, static and Keplerian
(including phot-only), now honor their input frame and keep the internal
geometric frame. The remaining failures are the binary-source BSBL / BFSBL
classes and the primary-input linear / accelerated orbit classes (see
Remaining issues).

## Method

`make_binary_frame_audit.py` (this directory) builds one model per binary-
lens Param class in `bagle.model_jax` (PSBL, FSBL, BSBL, BFSBL; FSBL/BFSBL
exist only there), with parallax switched off so tracks are straight. For
each reference point it measures where the (primary) source passes:

* the **input reference point** (midpoint, CoM = mass-weighted, or primary)
  — the time should equal the input `t0*` and the separation the input
  `u0*` / `beta* / thetaE`;
* the **actual lens midpoint** — the time should equal the internal
  `self.t0`, the separation `|self.u0_amp|`, and `self.xL0` should be the
  midpoint at `self.t0`;
* the input **`xS0`** at the time named in the docstring.

Static points and the CoM use the true closest approach on a fine time grid.
Moving points (midpoint or primary of an orbiting binary) use the
instantaneous definition at the reference time: the along-track offset is
reported as a time shift and the cross-track distance as the separation.

Figures are in Einstein radii, in the frame co-moving with the lens CoM
(so the source track is the relative-motion line, static lenses stay fixed,
and orbiting lenses trace their orbits), centered on the lens midpoint at
the internal `t0`, with East to the left. Blue marks the input frame (open circle: input
reference point at the input time; filled circle: source then; ▼: actual
closest approach to that point). Red marks the internal frame (◇: internal
`xL0`; ■: source at internal `t0`; ▲: actual closest approach to the lens
midpoint). Stars are the lenses (filled at internal `t0`, open at the input
time when the lenses moved); × is the midpoint and + the CoM. Green ✚ is
the input `xS0`. A ✅ needs both markers of a color to coincide. For
orbiting lenses the ▼ / ▲ markers are the linear closest approach, so they
need not coincide with the instantaneous definition.

## Representative cases

| # | Case | Class | Input frame | Input honored | Internal geometric |
|---|---|---|---|---|---|
| 1 | geometric, static | `PSBL_PhotAstrom_noPar_Param1` | geometric (`t0`, `beta`) | ✅ | ✅ |
| 2 | geometric, static (phot-only) | `PSBL_Phot_noPar_Param1` | geometric (`t0`, `u0_amp`) | ✅ | ✅ |
| 3 | geometric, static (binary source) | `BSBL_PhotAstrom_noPar_Param1` | geometric (`t0`, `beta`) | ✅ | ✅ |
| 4 | CoM, static | `PSBL_PhotAstrom_noPar_Param4` | CoM (`t0_com`, `u0_amp_com`) | ✅ | ✅ |
| 5 | primary (t0_prim), static | `PSBL_PhotAstrom_Par_Param5` | primary (`t0_prim`, `u0_amp_prim`) | ✅ | ✅ |
| 6 | primary (t0_p), static | `PSBL_PhotAstrom_Par_Param7` | primary (`t0_p`, `beta_p`) | ✅ | ✅ |
| 7 | geometric, Keplerian orbit | `PSBL_PhotAstrom_noPar_EllOrbs_Param2` | geometric (`t0`, `u0_amp`) | ✅ | ✅ |
| 8 | CoM (beta_com), Keplerian orbit | `PSBL_PhotAstrom_noPar_EllOrbs_Param1` | CoM (`t0_com`, `beta_com`) | ✅ | ✅ |
| 9 | CoM (u0_amp_com), Keplerian orbit | `PSBL_PhotAstrom_noPar_EllOrbs_Param4` | CoM (`t0_com`, `u0_amp_com`) | ✅ | ✅ |
| 10 | primary, Keplerian orbit | `PSBL_PhotAstrom_noPar_EllOrbs_Param7` | primary (`t0_p`, `beta_p`) | ✅ | ✅ |
| 11 | phot-only Keplerian orbit | `PSBL_Phot_noPar_EllOrbs_Param1` | geometric (`t0`, `u0_amp`) | ✅ | ✅ |
| 12 | binary source, Keplerian orbit | `BSBL_PhotAstrom_noPar_EllOrbs_Param1` | CoM (`t0_com`, `beta`) | ❌ | ❌ |

### 1. geometric, static: `PSBL_PhotAstrom_noPar_Param1`

Geometric inputs (`t0`, `beta`), static lenses. Input and internal frames are the same and exact.

![PSBL_PhotAstrom_noPar_Param1](case01_PSBL_PhotAstrom_noPar_Param1.png)

### 2. geometric, static (phot-only): `PSBL_Phot_noPar_Param1`

Phot-only (Einstein radii), geometric inputs. Exact.

![PSBL_Phot_noPar_Param1](case02_PSBL_Phot_noPar_Param1.png)

### 3. geometric, static (binary source): `BSBL_PhotAstrom_noPar_Param1`

Binary source, geometric lens inputs, measured with the primary source. Exact.

![BSBL_PhotAstrom_noPar_Param1](case03_BSBL_PhotAstrom_noPar_Param1.png)

### 4. CoM, static: `PSBL_PhotAstrom_noPar_Param4`

CoM inputs (`t0_com`, `u0_amp_com`), converted with `convert_u0_amp_t0_psbl`. Both frames exact (the old sin φ sign error is fixed).

![PSBL_PhotAstrom_noPar_Param4](case04_PSBL_PhotAstrom_noPar_Param4.png)

### 5. primary (t0_prim), static: `PSBL_PhotAstrom_Par_Param5`

Primary inputs (`t0_prim`, `u0_amp_prim`). Both frames exact.

![PSBL_PhotAstrom_Par_Param5](case05_PSBL_PhotAstrom_Par_Param5.png)

### 6. primary (t0_p), static: `PSBL_PhotAstrom_Par_Param7`

Primary inputs (`t0_p`, `beta_p`) via the rewritten `fc.convert_u0_t0_psbl`. Both frames exact.

![PSBL_PhotAstrom_Par_Param7](case06_PSBL_PhotAstrom_Par_Param7.png)

### 7. geometric, Keplerian orbit: `PSBL_PhotAstrom_noPar_EllOrbs_Param2`

Geometric inputs with a Keplerian orbit. The CoM closest approach comes from the orbit at `t0` (`convert_u0_amp_t0_psbl_orbit`), so `xL0` is the midpoint at `t0` and the inputs hold at that instant.

![PSBL_PhotAstrom_noPar_EllOrbs_Param2](case07_PSBL_PhotAstrom_noPar_EllOrbs_Param2.png)

### 8. CoM (beta_com), Keplerian orbit: `PSBL_PhotAstrom_noPar_EllOrbs_Param1`

CoM inputs (`t0_com`, `beta_com`) with a Keplerian orbit. CoM inputs exact; the geometric `t0` solves for the instant when the source - midpoint vector is perpendicular to the motion.

![PSBL_PhotAstrom_noPar_EllOrbs_Param1](case08_PSBL_PhotAstrom_noPar_EllOrbs_Param1.png)

### 9. CoM (u0_amp_com), Keplerian orbit: `PSBL_PhotAstrom_noPar_EllOrbs_Param4`

CoM inputs (`t0_com`, `u0_amp_com`) with a Keplerian orbit. Both frames exact (geometric `t0` at the orbit configuration at `t0`).

![PSBL_PhotAstrom_noPar_EllOrbs_Param4](case09_PSBL_PhotAstrom_noPar_EllOrbs_Param4.png)

### 10. primary, Keplerian orbit: `PSBL_PhotAstrom_noPar_EllOrbs_Param7`

Primary inputs with a Keplerian orbit: primary at `t0_p` → CoM → midpoint at `t0`. Both frames exact.

![PSBL_PhotAstrom_noPar_EllOrbs_Param7](case10_PSBL_PhotAstrom_noPar_EllOrbs_Param7.png)

### 11. phot-only Keplerian orbit: `PSBL_Phot_noPar_EllOrbs_Param1`

Phot-only Keplerian orbit. `t0` / `u0_amp` are geometric at `t0`; the source track (`get_complex_pos`, `get_u`) runs from the CoM closest approach `t0_com` / `u0_com`, since the lenses orbit the CoM.

![PSBL_Phot_noPar_EllOrbs_Param1](case11_PSBL_Phot_noPar_EllOrbs_Param1.png)

### 12. binary source, Keplerian orbit: `BSBL_PhotAstrom_noPar_EllOrbs_Param1`

Binary source and binary lens, both orbiting. Neither the input nor the internal frame is consistent; the BSBL/BFSBL orbit classes need their own design pass (CircOrbs Param3 is badly broken).

![BSBL_PhotAstrom_noPar_EllOrbs_Param1](case12_BSBL_PhotAstrom_noPar_EllOrbs_Param1.png)

## Every binary-lens parameterization

Times in days, separations in Einstein radii, `xS0` offset in mas. `Δt_in` / `|u|_in`: closest approach to the input reference point relative to the input time, and the separation there (compare with `input |u|`). `Δt_geo` / `|u|_geo`: closest approach to the actual lens midpoint relative to the internal `t0`, and the separation there (compare with `u0_amp`). `xL0`: distance from the internal `xL0` to the actual midpoint at `t0`. `xS0`: source distance from the input `xS0` at its documented time.

| Param class | Input | Δt_in | \|u\|_in | input \|u\| | Δt_geo | \|u\|_geo | u0_amp | xL0 | xS0 (time) | In | Int |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `BFSBL_PhotAstromParam1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `BFSBL_PhotAstromParam2` | primary `t0_p` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.2352 | 0.2352 | 0.0000 | 1.200 (t0_p) | ❌ | ✅ |
| `BFSBL_PhotAstrom_AccOrbs_Param1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `BFSBL_PhotAstrom_CircOrbs_Param1` | CoM `t0_com` | -1.86 | 0.0396 | 0.0724 | -27.29 | 0.1105 | 0.0724 | 0.0798 | 0.193 (t0_com) | ❌ | ❌ |
| `BFSBL_PhotAstrom_CircOrbs_Param2` | CoM `t0_com` | +0.23 | 0.1375 | 0.1000 | -0.65 | 0.1326 | 0.1262 | 0.0149 | 0.190 (t0_com) | ❌ | ❌ |
| `BFSBL_PhotAstrom_CircOrbs_Param3` | primary `t0_p` | -21.88 | 0.2085 | 0.0724 | -38.15 | 0.4447 | 0.3418 | 0.1159 | 0.313 (t0_p) | ❌ | ❌ |
| `BFSBL_PhotAstrom_EllOrbs_Param1` | CoM `t0_com` | -0.24 | 0.0355 | 0.0724 | -26.22 | 0.0874 | 0.0724 | 0.0623 | 0.206 (t0_com) | ❌ | ❌ |
| `BFSBL_PhotAstrom_EllOrbs_Param2` | CoM `t0_com` | +0.79 | 0.1331 | 0.1000 | +1.88 | 0.1370 | 0.1262 | 0.0271 | 0.166 (t0_com) | ❌ | ❌ |
| `BFSBL_PhotAstrom_EllOrbs_Param3` | primary `t0_p` | -20.07 | 0.2428 | 0.0724 | -37.49 | 0.4218 | 0.3418 | 0.1023 | 0.298 (t0_p) | ❌ | ❌ |
| `BFSBL_PhotAstrom_LinOrbs_Param1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `BFSBL_PhotParam1` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | – | – | ✅ | ✅ |
| `BSBL_PhotAstromParam1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `BSBL_PhotAstromParam2` | primary `t0_p` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.2352 | 0.2352 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `BSBL_PhotAstrom_AccOrbs_Param1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `BSBL_PhotAstrom_CircOrbs_Param1` | CoM `t0_com` | -1.86 | 0.0396 | 0.0724 | -27.29 | 0.1105 | 0.0724 | 0.0798 | 0.283 (t0) | ❌ | ❌ |
| `BSBL_PhotAstrom_CircOrbs_Param2` | CoM `t0_com` | +0.23 | 0.1375 | 0.1000 | -0.65 | 0.1326 | 0.1262 | 0.0149 | 0.190 (t0_com) | ❌ | ❌ |
| `BSBL_PhotAstrom_CircOrbs_Param3` | primary `t0_p` | -21.88 | 0.2085 | 0.0724 | -38.15 | 0.4447 | 0.3418 | 0.1159 | 0.140 (t0) | ❌ | ❌ |
| `BSBL_PhotAstrom_EllOrbs_Param1` | CoM `t0_com` | -0.24 | 0.0355 | 0.0724 | -26.22 | 0.0874 | 0.0724 | 0.0623 | 0.276 (t0) | ❌ | ❌ |
| `BSBL_PhotAstrom_EllOrbs_Param2` | CoM `t0_com` | +0.79 | 0.1331 | 0.1000 | +1.88 | 0.1370 | 0.1262 | 0.0271 | 0.166 (t0_com) | ❌ | ❌ |
| `BSBL_PhotAstrom_EllOrbs_Param3` | primary `t0_p` | -20.07 | 0.2428 | 0.0724 | -37.49 | 0.4218 | 0.3418 | 0.1023 | 0.120 (t0) | ❌ | ❌ |
| `BSBL_PhotAstrom_LinOrbs_Param1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `BSBL_PhotParam1` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | – | – | ✅ | ✅ |
| `FSBL_PhotAstromParam1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstromParam2` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstromParam3` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstromParam4` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1514 | 0.1514 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstromParam5` | primary `t0_prim` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.2543 | 0.2543 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstromParam6` | primary `t0_prim` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.2543 | 0.2543 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstromParam7` | primary `t0_p` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.2352 | 0.2352 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstromParam8` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1514 | 0.1514 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstrom_AccOrbs_Param1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstrom_AccOrbs_Param3` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstrom_AccOrbs_Param6` | primary `t0_prim` | -51.45 | 0.4087 | 0.1000 | +0.00 | 0.2543 | 0.2543 | 0.0000 | 0.000 (t0) | ❌ | ✅ |
| `FSBL_PhotAstrom_AccOrbs_Param7` | primary `t0_p` | -175.32 | 0.3981 | 0.0724 | +0.00 | 0.2352 | 0.2352 | 0.0000 | 0.000 (t0) | ❌ | ✅ |
| `FSBL_PhotAstrom_CircOrbs_Param1` | CoM `t0_com` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.1512 | 0.1512 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `FSBL_PhotAstrom_CircOrbs_Param3` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstrom_CircOrbs_Param4` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.0901 | 0.0901 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `FSBL_PhotAstrom_CircOrbs_Param7` | primary `t0_p` | +0.00 | 0.0724 | 0.0724 | -0.00 | 0.3085 | 0.3085 | 0.0000 | 0.000 (t0_p) | ✅ | ✅ |
| `FSBL_PhotAstrom_CircOrbs_Param8` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.0901 | 0.0901 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `FSBL_PhotAstrom_EllOrbs_Param1` | CoM `t0_com` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.1325 | 0.1325 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `FSBL_PhotAstrom_EllOrbs_Param3` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstrom_EllOrbs_Param4` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.0993 | 0.0993 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `FSBL_PhotAstrom_EllOrbs_Param7` | primary `t0_p` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.2519 | 0.2519 | 0.0000 | 0.000 (t0_p) | ✅ | ✅ |
| `FSBL_PhotAstrom_EllOrbs_Param8` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.0993 | 0.0993 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `FSBL_PhotAstrom_LinOrbs_Param1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstrom_LinOrbs_Param3` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `FSBL_PhotAstrom_LinOrbs_Param6` | primary `t0_prim` | -51.45 | 0.4087 | 0.1000 | +0.00 | 0.2543 | 0.2543 | 0.0000 | 0.000 (t0) | ❌ | ✅ |
| `FSBL_PhotAstrom_LinOrbs_Param7` | primary `t0_p` | -175.32 | 0.3981 | 0.0724 | +0.00 | 0.2352 | 0.2352 | 0.0000 | 0.000 (t0) | ❌ | ✅ |
| `FSBL_PhotParam1` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | – | – | ✅ | ✅ |
| `FSBL_Phot_CircOrbs_Param1` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | – | – | ✅ | ✅ |
| `FSBL_Phot_EllOrbs_Param1` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | – | – | ✅ | ✅ |
| `PSBL_PhotAstromParam1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstromParam2` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstromParam3` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstromParam4` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1514 | 0.1514 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstromParam5` | primary `t0_prim` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.2543 | 0.2543 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstromParam7` | primary `t0_p` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.2352 | 0.2352 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstrom_AccOrbs_Param1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstrom_AccOrbs_Param6` | primary `t0_prim` | -51.45 | 0.4087 | 0.1000 | +0.00 | 0.2543 | 0.2543 | 0.0000 | 0.000 (t0) | ❌ | ✅ |
| `PSBL_PhotAstrom_AccOrbs_Param7` | primary `t0_p` | -175.32 | 0.3981 | 0.0724 | +0.00 | 0.2352 | 0.2352 | 0.0000 | 0.000 (t0) | ❌ | ✅ |
| `PSBL_PhotAstrom_CircOrbs_Param1` | CoM `t0_com` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.1512 | 0.1512 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `PSBL_PhotAstrom_CircOrbs_Param2` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstrom_CircOrbs_Param3` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstrom_CircOrbs_Param4` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.0901 | 0.0901 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `PSBL_PhotAstrom_CircOrbs_Param7` | primary `t0_p` | +0.00 | 0.0724 | 0.0724 | -0.00 | 0.3085 | 0.3085 | 0.0000 | 0.000 (t0_p) | ✅ | ✅ |
| `PSBL_PhotAstrom_CircOrbs_Param8` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.0901 | 0.0901 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `PSBL_PhotAstrom_EllOrbs_Param1` | CoM `t0_com` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.1325 | 0.1325 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `PSBL_PhotAstrom_EllOrbs_Param2` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstrom_EllOrbs_Param3` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstrom_EllOrbs_Param4` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.0993 | 0.0993 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `PSBL_PhotAstrom_EllOrbs_Param7` | primary `t0_p` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.2519 | 0.2519 | 0.0000 | 0.000 (t0_p) | ✅ | ✅ |
| `PSBL_PhotAstrom_EllOrbs_Param8` | CoM `t0_com` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.0993 | 0.0993 | 0.0000 | 0.000 (t0_com) | ✅ | ✅ |
| `PSBL_PhotAstrom_LinOrbs_Param1` | geometric `t0` | +0.00 | 0.0724 | 0.0724 | +0.00 | 0.0724 | 0.0724 | 0.0000 | 0.000 (t0) | ✅ | ✅ |
| `PSBL_PhotAstrom_LinOrbs_Param6` | primary `t0_prim` | -51.45 | 0.4087 | 0.1000 | +0.00 | 0.2543 | 0.2543 | 0.0000 | 0.000 (t0) | ❌ | ✅ |
| `PSBL_PhotAstrom_LinOrbs_Param7` | primary `t0_p` | -175.32 | 0.3981 | 0.0724 | +0.00 | 0.2352 | 0.2352 | 0.0000 | 0.000 (t0) | ❌ | ✅ |
| `PSBL_PhotParam1` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | – | – | ✅ | ✅ |
| `PSBL_Phot_CircOrbs_Param1` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | – | – | ✅ | ✅ |
| `PSBL_Phot_EllOrbs_Param1` | geometric `t0` | +0.00 | 0.1000 | 0.1000 | +0.00 | 0.1000 | 0.1000 | – | – | ✅ | ✅ |

## Fixes applied

1. **Static CoM / primary ↔ geometric shift.** The old scalar formula
   `u0_amp = u0_amp_com ∓ ½·qeff·sep·sin φ/θE` used one sign, but the sign
   of the shift depends on which side of the motion the positive-`u0_amp`
   direction lies, so it was right for southward motion only. The shift is
   now done on the closest-approach vector
   (`bagle.model.convert_u0_amp_t0_psbl`; JAX:
   `derive_psbl_param4_heliocentric`). Used by Param4/5/6/8.
2. **`fc.convert_u0_t0_psbl` rewritten** from the binary-axis geometry
   (it shifted u0 by component signs, and most `t0` branches had the wrong
   sign). Param7 also keeps the sign of `u0_amp`.
3. **Keplerian orbits** convert frames with the orbit at the reference
   time (`convert_u0_amp_t0_psbl_orbit`) instead of a snapshot at `tp` with
   separation `a`; `alpha` / `phi` / `sep_t0` are reported at `t0`. The
   traced PSTL EllOrbs Param2 path uses the same definition.
4. **Phot-only orbits** store `t0_com` / `u0_com` and run the source track
   about the CoM in `get_complex_pos` and `get_u`; `phi_rho1_rad` is now in
   radians.
5. **EllOrbs / CircOrbs Param1** list `beta_com` (not `beta`) in
   `fitter_param_names`.
6. **`xS0` is always at the input reference time** (`t0_com`, `t0` or
   `t0_p`); EllOrbs / CircOrbs Param3 and Param7 now follow Param1/2/4/8.
7. **Phot-only orbit constructors** take their arguments in
   `fitter_param_names` order (`t0, u0_amp, tE, piE_E, piE_N, q, sep,
   v_para, v_perp, v_rad, r_s, a_s, b_sff, mag_src`; FSBL adds `radiusS`),
   and derive `aleph` / `aleph_sec` from `sep`, `r_s`, `a_s` and `q`.
8. **Phot-only orbit elements** come from `bagle.orbits.relative_state_to_elements`,
   written in the `Orbit.oal2xy` (Thiele-Innes) frame, so the orbit
   reproduces `sep`, the axis angle, `r_s` and the velocities at `t0`
   (circular orbits included). A new `phi` (after `sep`, as in
   `PSBL_PhotParam1`) sets the binary axis; light curves match
   VBMicrolensing's `BinaryLightCurveKepler` (`w1, w2, w3, sz/s, a/r` =
   `v_para, v_perp, v_rad, r_s, a_s` per day).
9. **`fitter_param_names` match the constructors** (checked for every Param
   class by `tests/test_fitter_param_names.py`): Param4 / CircOrbs Param8
   use `t0_com`, `u0_amp_com`; the Param6 family uses `t0_prim`,
   `u0_amp_prim`; swapped or shifted arguments were fixed in
   `PSPL_PhotAstromParam5`, BSBL / BFSBL `LinOrbs` / `AccOrbs` Param1,
   BSBL / BFSBL `CircOrbs_Param3`, the BSPL GP orbit classes and the FSBL
   limb GP classes; `EllOrbs_Param4` passes a true `mag_base` to its parent.

Regression tests: `tests/test_binary_frames_vs_vbm.py` (static classes vs
VBMicrolensing in three motion directions), `tests/test_binary_orbit_frames.py`
(orbit definitions read back from the lens / source positions), and
`tests/test_frame_convert.py::test_convert_u0_t0_psbl_geometry`.

## Remaining issues

* **BSBL / BFSBL** static Param2 and all orbit classes are inconsistent in
  both frames.
* **Linear / accelerated primary-input classes** (`LinOrbs` / `AccOrbs`
  Param6 / Param7): internal frame correct, input frame not honored. Their
  `get_resolved_lens_astrometry` branch places the primary at
  `xL0 - (sep/2)(sin α, cos α)`, the opposite side from the static and
  Keplerian classes, so the static conversion lands on the wrong lens.

## Reproduce

```bash
PYTHONPATH=src python docs/audits/binary_frames/make_binary_frame_audit.py
```

