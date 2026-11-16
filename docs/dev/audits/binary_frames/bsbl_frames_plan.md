# Plan: consistent frames for the BSBL / BFSBL parameterizations

Status: implemented 2026-11-14 (uncommitted). Deviations from the text
below:

* **PSBL / FSBL Keplerian Param2 / Param3** (midpoint inputs) keep the
  midpoint pair as their internal `t0` / `u0_amp` / `xL0`, with `t0_mid`
  etc. set to the inputs; no root solve. Making `t0` the CoM there would
  have broken the rule that the attribute named like a fitter parameter
  holds its input value (`model_fitter` builds parameter labels from it).
* **PSTL** (`PSTL_PhotAstrom_EllOrbs_Param1/2`, model_jax only) keep their
  `xS0_E/N` input names; they pass them to the PSBL Keplerian classes,
  which also set the internal `xS0_E/N`, so either naming works.
* The fitter's legacy per-filter column loop (`bases = ['xS0_E', 'xS0_N']`)
  only renames old unsuffixed columns, which never existed for the new
  names, so it is unchanged.

## 1. Goal and scope

Make the static `BSBL_PhotAstromParam2` and the six Keplerian
`BSBL_PhotAstrom_{Ell,Circ}Orbs_Param{1,2,3}` classes honor their input
frame and fill the internal frame consistently, as the PSBL / FSBL classes
already do (see `binary_frame_audit.md`).

Covered automatically by inheritance: every `BFSBL_*`, `BFSBL_Limb_*`,
`*_Par_*` and `*_noPar_*` variant of these classes, in both `model.py`
and `model_jax.py`.

Also in scope (D2): the existing PSBL / FSBL Keplerian PhotAstrom classes
move to the same option-C internal frame (section 5.4).

Not changing: the position-evaluation code. Both Keplerian branches,
`BSBL_PhotAstrom.get_resolved_lens_astrometry` (model.py 22989-23026) and
`BSBL_PhotAstrom.get_resolved_source_astrometry_unlensed` (model.py
23717-23740), already move each binary's CoM on a straight line,
`x_com(t) = x0_com + mu (t - t0_com)`, and add the `oal2xy` offsets. They
need correct `xL0_com`, `xS0_com` and `t0_com` and nothing else. None of
these classes has a JAX likelihood kernel, so only constructors change.

## 2. Naming conventions

### 2.1 "mid", not "geo", for the lens geometric midpoint

"geo" is reserved for the geocentric (geo-tr) frame (`t0_geotr`,
`u0_amp_geotr`, unchanged). This applies to attribute and variable names.
The released `frame_convert` coords string `'geom_mid'` is kept (D4).

| Current | New | Where |
|---|---|---|
| `'geom_mid'` (coords string) | **unchanged** | Released in v1.0.4 and v1.0.5 as a `frame_convert` argument value; kept as is (D4). New converters accept the same strings. |
| `self.t0_geo` | `self.t0_mid` (lazy property; see section 6) | `BSBL_PhotAstrom_EllOrbs_Param1` |
| `t0_geom_temp` | removed | same class (the static q_eff shift goes away) |

### 2.2 Source masses: `mS`, `mSp`, `mSs`

Replace `mass_sources`, `mass_source_primary` and `mass_source_secondary`
with `mS` (total source mass), `mSp` (primary) and `mSs` (secondary), in
the same Msun units, matching `mL`, `mLp`, `mLs`. Occurrences (model.py;
the same in model_jax.py):

* `BSPL_PhotAstrom_EllOrbs_Param1` (20729), `_Param2` (20970),
  `_Param3` (21260), `_Param4` (21482, 21608)
* `BSBL_PhotAstrom_EllOrbs_Param1` (25289), `_Param3` (26202)

Store floats (`.value`), not astropy Quantities, so `mS` behaves like
`mL`.

### 2.3 Reference-point suffixes

One rule for every binary-source / binary-lens model:

| Suffix | Source point | Lens point | Kind |
|---|---|---|---|
| `_com` | source CoM | lens CoM | true closest approach (both CoMs move linearly) |
| `_p` | primary source | primary lens | instantaneous (see 3.2) |
| none | primary source | lens midpoint | internal frame; instantaneous |

The suffix applies to the whole input set: `t0_*`, `u0_amp_*` / `beta_*`
**and** the source position (the source point at that `t0`). For the
position the suffix comes before the component: `xS0_com_E`,
`xS0_com_N`, `xS0_p_E`, `xS0_p_N`. That keeps the `_E` / `_N` ending the
fitter relies on, and the per-filter fitter names become `xS0_com_E1`,
`xS0_p_E1`, and so on. The vector attributes are `xS0_com` and `xS0_p`.
For single-source models
(PSBL, FSBL) the source is its own CoM, so `_com` keeps its current
meaning there. The midpoint pair also gets explicit `_mid` names
(`t0_mid`, `u0_amp_mid`, `xL0_mid`); for Keplerian classes they are lazy
properties (section 6).

Input renames (changes `fitter_param_names`; `xS0_*` is a per-filter
parameter):

| Class | Current inputs | New inputs | Frame |
|---|---|---|---|
| `BSBL_PhotAstromParam2` (static) | `t0_p`, `beta_p`, `xS0_E/N` | `t0_p`, `beta_p`, `xS0_p_E`, `xS0_p_N`, new `qS` (D3) | `_p` |
| `BSBL_PhotAstrom_EllOrbs/CircOrbs_Param1` | `t0_com`, `beta`, `xS0_E/N` | `t0_com`, `beta_com`, `xS0_com_E`, `xS0_com_N` | `_com` |
| `BSBL_PhotAstrom_EllOrbs/CircOrbs_Param2` | `t0_com`, `u0_amp_com`, `xS0_E/N` (primary source ↔ lens CoM) | `t0_com`, `u0_amp_com`, `xS0_com_E`, `xS0_com_N` | `_com` (meaning changes; decision D1) |
| `BSBL_PhotAstrom_EllOrbs/CircOrbs_Param3` | `t0_p`, `beta_p`, `xS0_E/N` | `t0_p`, `beta_p`, `xS0_p_E`, `xS0_p_N` | `_p` |

Knock-on changes for the `xS0` renames:

* `model_fitter.py` / `model_fitter_jax.py`:
  * `multi_filt_params` (line 66) and the `xS0_E` / `xS0_N` priors
    (189-190) need `xS0_com_{E,N}` / `xS0_p_{E,N}` entries.
  * The data-driven `make_xS0_gen` handling must accept them. The filter
    renaming loop (`bases = ['xS0_E', 'xS0_N']`, line 985) needs the new
    bases. The prior lookup compares `priors_name == 'xS0_E'` /
    `'xS0_N'` (1591-1593); extend it to any `xS0*_E` → xpos and
    `xS0*_N` → ypos. The JAX fitter has the same code at
    model_fitter_jax.py 74, 664-665, 1463 and 2087.
  * Retire the `xS0_system_E/N` priors (model_fitter.py 262-263,
    model_fitter_jax.py 736-737; no model uses them) and add
    `xS0_com_E` / `xS0_com_N` (D5).
  * Add a `qS` prior for the static Param2 input (D3). Suggested
    `('make_gen', 0, 10)`, since the photometric primary need not be the
    more massive star.
  * The user's uncommitted edits in `model_fitter.py` must be preserved.
* `tests/model_old_vs_jax_fixtures.py`, `tests/test_fitter_param_names.py`
  (`RENAMES` allowlist), the audit script `VALS`, and the docs.

## 3. Target internal convention

### 3.1 Attributes every BSBL / BFSBL instance carries

| Attributes | Meaning |
|---|---|
| Attributes | Static / linear / accelerated classes | Keplerian classes (option C) |
|---|---|---|
| `t0`, `u0_amp`, `u0`, `beta`, `xL0` | primary source ↔ lens midpoint (internal) | same as the `_com` values (internal frame is CoM–CoM) |
| `t0_mid`, `u0_amp_mid`, `xL0_mid` | aliases of the internal values | lazy cached properties (root solve on first access) |
| `t0_com`, `u0_amp_com`, `u0_com`, `beta_com` | source CoM ↔ lens CoM | same, computed explicitly |
| `xL0_com`, `xS0_com` | lens and source CoM positions at `t0_com` | same (used by the evaluation code) |
| `t0_p`, `u0_amp_p`, `beta_p` | primary source ↔ primary lens (explicit) | input values for Param3; lazy properties otherwise |
| `xS0_pri`, `xS0_sec`, `xS0_p` | source positions at `t0`, `t0`, `t0_p` | same; `xS0_p` lazy unless it is the input |
| `sepL_t0`, `alphaL`, `sepS_t0`, `alphaS` | binary axes (fixed or at `t0`) | binary axes **at `t0`** (not at `tp`); `sepL = aL` stays the semi-major axis |
| `mL`, `mLp`, `mLs`, `q` | lens masses (Msun), `q = mLs / mLp` | same |
| `mS`, `mSp`, `mSs`, `qS` | source masses (Msun) when known; `qS = mSs / mSp` always | `qS = alephS / aleph_secS` (Param1/2) or `mSs / mSp` (Param3) |

`q` keeps its current meaning (lens mass ratio). `qS` is the new source
mass ratio, needed for the source CoM (D3).

* **Sign convention:** every `u0_amp_*` carries the Gould sign,
  `u0_amp = u0 · û₊` with `û₊ = u0_hat_from_thetaE_hat(θ̂, +1)`. Never use
  `sqrt(u0_E² + u0_N²)`.
* **`xS0` epoch:** the input `xS0_*` is the input source point at the
  input `t0_*`.

### 3.2 Instantaneous definition (moving points)

The lens midpoint, the primary lens and the primary source orbit, so their
`(t0, u0)` are defined at an instant. At `t0` the source point is
`u0 · thetaE` from the lens point's position at that time, and `u0` is
perpendicular to the relative proper motion. `t0` is a reference epoch,
not a closest-approach time. Only the `_com` pair is a true closest
approach, which is why it is the internal frame for Keplerian classes.

## 4. Equations

### 4.1 Notation

* θ̂ is the direction of μ_rel; û₊ is the positive-u0 direction. Offsets
  are in Einstein units: `oal2xy` output (arcsec) / (thetaE · 1e-3).
* Lens orbit: **d**_Lp(t), **d**_Ls(t) are the lens primary and secondary
  offsets from the lens CoM; the midpoint offset is
  **d**_Lm = (**d**_Lp + **d**_Ls) / 2; **d**_L,com = 0.
* Source orbit: **d**_Sp(t) is the primary-source offset from the source
  CoM; **d**_S,com = 0. Static source: **d** = 0 (or the fixed offset).
* Source CoM minus lens CoM, a straight line:

$$\mathbf u_{\rm com}(t)=\mathbf u_{0,\rm com}+\frac{t-t_{0,\rm com}}{t_E}\hat\theta,\qquad \mathbf u_{0,\rm com}\perp\hat\theta .$$

* For a source point $a$ and a lens point $b$:

$$\mathbf u_{ab}(t)=\mathbf u_{\rm com}(t)+\mathbf d_{S,a}(t)-\mathbf d_{L,b}(t).$$

### 4.2 Definition of $(t_{0,ab},u_{0,ab})$

$$\mathbf u_{ab}(t_{0,ab})\cdot\hat\theta=0,\qquad u_{0,ab}=\mathbf u_{ab}(t_{0,ab})\cdot\hat u_+ .$$

### 4.3 Input pair → CoM line (always explicit, no root)

$$\mathbf u_{\rm com}(t_{\rm in})=u_{0,\rm in}\hat u_+ +\mathbf d_{L,b}(t_{\rm in})-\mathbf d_{S,a}(t_{\rm in})$$

$$t_{0,\rm com}=t_{\rm in}-t_E\,\mathbf u_{\rm com}(t_{\rm in})\cdot\hat\theta,\qquad \mathbf u_{0,\rm com}=\mathbf u_{\rm com}(t_{\rm in})-\big(\mathbf u_{\rm com}(t_{\rm in})\cdot\hat\theta\big)\hat\theta .$$

### 4.4 CoM line → output pair (root solve unless the output is `_com`)

$$f(t)=\frac{t-t_{0,\rm com}}{t_E}+\big[\mathbf d_{S,a}(t)-\mathbf d_{L,b}(t)\big]\cdot\hat\theta=0 .$$

* **Bracket:** the root lies within
  $\pm\,t_E\,(d_{L,\max}+d_{S,\max})$ of $t_{0,\rm com}$, where
  $d_{\max}=(\aleph+\aleph_2)(1+e)/\theta_E$ for each orbit.
* **Uniqueness:** the root is unique when
  $|\dot{\mathbf d}_{S,a}-\dot{\mathbf d}_{L,b}|\cdot\hat\theta < 1/t_E$,
  i.e. the orbital speeds along θ̂ stay below μ_rel.
* **Otherwise:** sample $f$ on a grid and take the sign change nearest
  $t_{\rm in}$.
* **Then:**
  $u_{0,ab}=\big[\mathbf u_{\rm com}(t_{0,ab})+\mathbf d_{S,a}(t_{0,ab})-\mathbf d_{L,b}(t_{0,ab})\big]\cdot\hat u_+$.

### 4.5 Positions (mas), from the input source point $\mathbf x_{S0}$ at $t_{\rm in}$

$$\mathbf x_{S,\rm com}(t_{\rm in})=\mathbf x_{S0}-\theta_E\,\mathbf d_{S,a}(t_{\rm in}),\qquad \mathbf x_{S0,\rm com}=\mathbf x_{S,\rm com}(t_{\rm in})+\boldsymbol\mu_S\,(t_{0,\rm com}-t_{\rm in})$$

$$\mathbf x_{L0,\rm com}=\mathbf x_{S0,\rm com}-\theta_E\,\mathbf u_{0,\rm com}$$

$$\mathbf x_{L0,\rm mid}=\mathbf x_{L0,\rm com}+\boldsymbol\mu_L\,(t_{0,\rm mid}-t_{0,\rm com})+\theta_E\,\mathbf d_{Lm}(t_{0,\rm mid})$$

$$\mathbf x_{S0,\rm pri}=\mathbf x_{S0,\rm com}+\boldsymbol\mu_S\,(t_0-t_{0,\rm com})+\theta_E\,\mathbf d_{Sp}(t_0)$$

Times are in years inside the μ terms; multiply by 1e-3 to store in
arcsec. For Keplerian classes $t_0=t_{0,\rm com}$ and
$\mathbf x_{L0}=\mathbf x_{L0,\rm com}$ (option C). $\mathbf x_{L0,\rm mid}$
is then computed lazily together with $t_{0,\rm mid}$.

### 4.6 Static `BSBL_PhotAstromParam2`

Both binaries are rigid, so every conversion is exact
(`frame_convert.convert_u0_t0_psbl`, already used at model.py 24923). The
changes:

$$\mathbf x_{S,p}(t_0)=\mathbf x_{S0,p}+\boldsymbol\mu_S\,(t_0-t_{0,p}),\qquad \mathbf x_{L0}=\mathbf x_{S,p}(t_0)-\theta_E\,\mathbf u_0,\qquad u_{0,\rm amp}=\mathbf u_0\cdot\hat u_+ .$$

**Source CoM from a mass ratio (D3).** Only the mass ratio
$q_S = m_{Ss}/m_{Sp}$ is needed, not the masses themselves. With
$\hat{\mathbf s}_S=(\sin\alpha_S,\cos\alpha_S)$ the unit vector from the
primary source to the secondary (the BSBL source-α convention), the
primary sits on the far side of the CoM:

$$\mathbf x_{S,\rm com}=\mathbf x_{S,p}+\frac{q_S}{1+q_S}\,s_S\,\hat{\mathbf s}_S,\qquad \mathbf d_{Sp}=-\frac{q_S}{1+q_S}\,\frac{s_S}{\theta_E}\,\hat{\mathbf s}_S .$$

$q_S=0$ puts the CoM on the primary. Because both binaries are rigid,
all three pairs follow explicitly from 4.3 with constant offsets: $f(t)$
in 4.4 is linear, so no root is needed. The pairs are `_p` (input),
`_com`, and the internal midpoint pair. Then set `xL0_com` and `xS0_com`
from 4.5, and rebuild `xS0_sec` from the moved `xS0_pri`.

## 5. Code changes

### 5.1 `frame_convert.py`

* **New:** `convert_u0_amp_t0_bsbl_orbit(t0_in, u0_amp_in, tE, thetaE_hat,
  orbL, orbS, thetaE_orb, lens_in, lens_out, src_in, src_out)`.
  * It implements 4.3 and 4.4, reusing `keplerian_point_offset` for both
    orbits.
  * `orbS=None` means a static source.
  * Lens points are `'geom_mid'`, `'prim_center'` and `'COM'` (the
    existing strings); source points are `'prim_center'` and `'COM'`.
* `convert_u0_amp_t0_psbl_orbit` becomes a thin wrapper (`orbS=None`), so
  the PSBL results are unchanged.
* `'geom_mid'` stays as the coords string (D4).

### 5.2 Constructors (model.py line / model_jax.py line)

| Class | Replace | Keep |
|---|---|---|
| `BSBL_PhotAstromParam2` | 24929 (unsigned `u0_amp`); 24940-24965 (`xL0` and secondary from `xS0` at `t0_p`, add the CoM attributes) / 25886, 25897-25922 | conversion at 24923-24928 |
| `EllOrbs_Param1` | 25131-25135 (`sepS`, `mL`); 25163-25165 and 25187-25189 (α at `tp`); 25281-25313 (q_eff `t0`, `mass_source*`, `xL0_com = xL0`, `xS0_com = xS0`) / +957 lines | orbit setup 25150-25186 |
| `EllOrbs_Param2` | 25690-25692, 25714-25716 (α at `tp`); 25736-25740 (q_eff `t0`, `u0_amp`); 25784-25804 (`xL0 = xL0_com`, `xS0_com = xS0 + com_vec`) / +957 | `mLp`/`mLs` from `q` (25664-25665) |
| `EllOrbs_Param3` | 26194-26196, 26235-26237 (α at `tp`); 26269-26366 (static conversions, unsigned `u0_amp` 26310, `beta` overwrite 26357, scalar `thetaS0_com` 26363, `xS0_com = xS0` 26364) / +971 | source masses `mSp`/`mSs` (26218-26219) |
| `CircOrbs_Param1/2/3`, all BFSBL subclasses | docstrings only | inherit |

* **Shared tail:** all three Keplerian constructors end with one shared
  helper.
  1. Build `orbL` and `orbS`.
  2. Call `convert_u0_amp_t0_bsbl_orbit` from the class's input pair to
     `_com` (explicit). Set the internal `t0`, `u0_amp` and `xL0` to the
     `_com` values.
  3. Apply the position formulas of 4.5.
  4. Register the lazy `_mid` (and, for Param1/2, `_p`) properties; they
     call the same converter on first access (section 6).

  Each constructor only declares its input points.
* **Removed:** the static-binary formulas
  `t0 = t0_com − 0.5 q_eff tE s cos φ` and their `mass_source*` / `q_source`
  helpers.
* **Docstrings:** state the frame and epoch of every `t0_*`, `u0_amp_*` /
  `beta_*` and `xS0_*`, with "reference epoch, not closest approach" notes
  like the PSBL classes.

### 5.4 PSBL / FSBL Keplerian PhotAstrom classes (option C, D2)

Classes: `PSBL_PhotAstrom_{Ell,Circ}Orbs_Param{1,2,3,4,7,8}` (model.py and
model_jax.py) and the inheriting
`FSBL_PhotAstrom_*` / `FSBL_Limb_PhotAstrom_*` `{Ell,Circ}Orbs_Param{1,3,4,7,8}`
(model_jax.py), plus all Par / noPar variants.

The rule is the same as BSBL: the internal `t0`, `u0_amp`, `u0`, `beta`
and `xL0` become the lens-CoM values, which for a single source is the
`_com` pair. The midpoint pair becomes the lazy `t0_mid`, `u0_amp_mid`,
`xL0_mid`.

* **Source track:** these classes evaluate it as
  `xS0 + muS (t - t0)` (`PSPL.get_source_astrometry_unlensed`, via
  `PSBL_PhotAstrom.get_complex_pos`). So the stored `self.xS0` must be the
  source position at the new internal `t0 = t0_com`:
  $\mathbf x_{S0}\leftarrow\mathbf x_{S0,\rm in}+\boldsymbol\mu_S\,(t_{0,\rm com}-t_{\rm in})$.
  The input value is kept under its input name.
* **Lens track:** unchanged; it already uses `xL0_com` and `t0_com`.

| Class | Input pair (current names) | Conversion today (model.py / model_jax.py) | After |
|---|---|---|---|
| `Param1` | `t0_com`, `beta_com`, `xS0_E/N` | COM → mid root, 8672 / 9292 | internal = input; `_mid` lazy; inputs renamed `xS0_com_E`, `xS0_com_N` |
| `Param2` | `t0`, `u0_amp`, `xS0_E/N` (mid) | mid → COM explicit, 9542 / 10176 | internal = `_com` (explicit); `t0_mid` = input (no root) |
| `Param3` | `t0`, `u0_amp`, `xS0_E/N` (mid) | mid → COM explicit, 10368 / 11002 | same as Param2 |
| `Param4` | `t0_com`, `u0_amp_com`, `xS0_E/N` | COM → mid root, 10969 / 11603 | as Param1; inputs `xS0_com_E`, `xS0_com_N` |
| `Param7` | `t0_p`, `beta_p`, `xS0_E/N` (primary lens) | primary → COM explicit 12525, COM → mid root 12528 / 13162, 13165 | internal = `_com` (explicit); `_mid` lazy; inputs `xS0_p_E`, `xS0_p_N` |
| `Param8` | `t0_com`, `u0_amp_com`, `xS0_E/N` | COM → mid root, 13128 / 13765 | as Param1; inputs `xS0_com_E`, `xS0_com_N` |

* **Phot-only exception:** `PSBL_Phot_{Ell,Circ}Orbs_Param1` and the
  inheriting `FSBL_Phot` / `FSBL_Limb_Phot` classes stay as they are.
  * Their orbit is defined by the binary state at `t0`
    (`relative_state_to_elements`), with no root solve (13672 is
    mid → COM, explicit).
  * Their `get_u` works in the midpoint frame at `t0`.
  * Document them as the exception: internal frame = midpoint at the
    input `t0`.
* **Renames:** the `xS0_com_E` / `xS0_p_E` renames follow the 2.3 rule
  and change the PSBL `fitter_param_names` too. The single source is its
  own CoM, so `_com` here names the reference epoch and pair.
* **Existing tests:** `tests/test_binary_orbit_frames.py` checks the
  internal midpoint for PSBL orbits. Update it to read `t0_mid` /
  `u0_amp_mid` / `xL0_mid`, and add a check that the PhotAstrom
  photometry and astrometry are unchanged to 1e-10 before and after
  (only the internal labels move).

### 5.5 Documentation and changelog (D6)

* **`CHANGELOG.md` (new, repo root):** the repo has no changelog yet.
  Start an "Unreleased" section listing:
  * the frame fixes (which classes, and that fit results change);
  * the input renames (`xS0_com_E`, `xS0_p_E`, `beta_com`, `qS`) in the
    BSBL / BFSBL classes and in PSBL / FSBL Keplerian Param1, 4, 7 and 8,
    and the BSBL Param2 frame change;
  * `t0_geo` → `t0_mid`, and the `mass_source*` → `mS` / `mSp` / `mSs`
    renames;
  * the retired `xS0_system_E/N` prior;
  * `alphaL` / `alphaS` moving from `tp` to `t0`;
  * the option-C internal frame for Keplerian classes.

  Mirror it in `docs/changelog.rst` and add that page to the
  `docs/index.rst` toctree.
* **Internal parameterization docs:** add a "Reference points and the
  internal frame" subsection to the "Parameterization Class Family"
  section of `docs/new_models.rst` (line 96). It should cover:
  * the suffix rule of 2.3 and the attribute table of 3.1;
  * the instantaneous definition of 3.2;
  * the option-C rule (Keplerian internal frame = `_com`; `_mid` / `_p`
    lazy);
  * the sign convention, and the `xS0` epoch rule.
* **Model-type pages:** add a short "Reference points" section to
  `docs/PSBL.rst`, `docs/BSPL.rst` and `docs/BSBL.rst`. Each states which
  inputs each parameterization takes (midpoint, `_com` or `_p`), what
  `t0` / `u0_amp` / `xL0` mean internally for static versus Keplerian
  classes, and links to the `new_models.rst` subsection. Add a one-line
  pointer at the top of `BSBL_Param.rst`, `PSBL_Param.rst` and
  `BSPL_Param.rst`.
* **Class docstrings:** the same clarifying language, kept short, in each
  changed Param class.

### 5.6 Parallel changes in the JAX implementation

Every change above is made in both `bagle.model` (NumPy reference) and
`bagle.model_jax`. The JAX-side items:

* **`model_jax.py` constructors** (the same edits as 5.2 and 5.4):
  * `BSBL_PhotAstromParam2`: 25886, 25897-25922.
  * `BSBL_PhotAstrom_EllOrbs_Param1`: 26088-26092, 26120-26122,
    26144-26146, 26238-26270.
  * `_Param2`: 26647-26649, 26671-26673, 26693-26697, 26741-26761.
  * `_Param3`: 27165-27167, 27206-27208, 27240-27337.
  * PSBL Keplerian `Param1` 9292, `Param2` 10176, `Param3` 11002,
    `Param4` 11603, `Param7` 13162 / 13165, `Param8` 13765.
* **JAX-only classes:** these inherit the fixed constructors and need
  docstring updates only.
  * The `FSBL_PhotAstrom_*` and `FSBL_Limb_PhotAstrom_*` Keplerian
    classes (Param1, 3, 4, 7, 8).
  * The `BFSBL_*` and `BFSBL_Limb_*` BSBL subclasses (37713 onward).
  * `FSBL_Phot_*` / `FSBL_Limb_Phot_*` orbit classes: no change
    (phot-only exception, 5.4).
* **Renames in `model_jax.py`:**
  * `mass_source*` → `mS` / `mSp` / `mSs` in `BSPL_PhotAstrom_EllOrbs_Param1`
    (21677), `_Param2` (21918), `_Param3` (22208), `_Param4` (22430,
    22556), `BSBL_PhotAstrom_EllOrbs_Param1` (26246) and `_Param3` (27173).
  * `self.t0_geo` (26243) → `t0_mid`.
* **Lazy `_mid` / `_p` properties:** defined on the shared Param mixins,
  so the `model` and `model_jax` classes behave identically. They are
  NumPy / SciPy code run outside any traced function.
* **JAX likelihood kernels:** none of the affected classes has one. The
  only orbit-class kernels are on the PSBL / FSBL LinOrbs / AccOrbs
  classes, which this plan does not touch (they belong to the
  `lens-linacc-flip` task). Option C means that any future Keplerian
  kernel can work in the CoM frame with no root solve inside the traced
  function.
* **`src/bagle/jax/`:**
  * `jax/orbits.py`: give `oal2xy` the same angle-convention docstring
    as `orbits.Orbit.oal2xy` (Omega from North through East, ω from the
    shared ascending node along the motion, the secondary at ω + 180).
  * `jax/geometry.py`: no change. Its PSBL derivations are for static
    classes, which keep the midpoint frame; its `geom` variables mean
    "geometry tuple", not midpoint.
  * `jax/bspl.py`: no change here. It reads `t0_pri` / `t0_sec` /
    `u0_sec`, which belong to the `bsource-alongtrack` task.
  * `jax/bsbl.py`: an orphaned stub. It imports the removed
    `bagle.jax.layout_registry`, fails on import, and nothing imports
    it. Delete it (or fix the import) as cleanup.
* **`model_fitter_jax.py`:** the same edits as the fitter section of 2.3.
  * `multi_filt_params` (74).
  * The `xS0_E` / `xS0_N` priors (664-665) gain `xS0_com_*` / `xS0_p_*`.
  * Retire `xS0_system_E/N` (736-737) and add a `qS` prior.
  * The per-filter rename loop (1463) and the `make_xS0_gen` lookup
    (2087) accept `xS0*_E` / `xS0*_N`.
* **Tests:**
  * `tests/test_model_jax.py` and `tests/test_model.py` reference the
    PSBL / BSBL Keplerian classes; update their parameter names.
  * `tests/model_old_vs_jax_fixtures.py` (old-vs-JAX parity) gets the new
    keys for both implementations.
  * The new tests in section 7 run on `model` and `model_jax`
    (parametrize over both modules).
  * Rerun `tests/test_jax_physics.py` and `tests/test_jax_likelihood.py`
    as a no-regression check.

## 6. Root solving: where it is needed, and do we need it?

**Where.** The root solve of 4.4 is needed whenever an *output* pair
involves a moving point. Converting any input to the CoM line is always
explicit.

| Parameterization | Input pair | Root solves |
|---|---|---|
| `PSBL_PhotAstrom_{Ell,Circ}Orbs_Param1`, `_Param4`, `_Param8` (+ FSBL / Par variants, model_jax.py) | lens CoM | internal mid `t0`, `u0_amp` (existing: model.py 8672, 10969, 13128) |
| `PSBL_PhotAstrom_{Ell,Circ}Orbs_Param7` | lens primary | internal mid (12528; the primary → CoM step at 12525 is explicit) |
| `PSBL_PhotAstrom_{Ell,Circ}Orbs_Param2`, `_Param3`, `PSBL_Phot_EllOrbs_Param1` | lens mid | none (mid → CoM is explicit) |
| Phot-only orbit classes (`relative_state_to_elements`) | state at `t0` | none |
| `BSBL/BFSBL_PhotAstrom_{Ell,Circ}Orbs_Param1`, `_Param2` (new) | `_com` | internal mid; `_p` if stored |
| `BSBL/BFSBL_PhotAstrom_{Ell,Circ}Orbs_Param3` (new) | `_p` | internal mid (`_p` → `_com` is explicit) |
| `BSBL/BFSBL_PhotAstromParam2` (static) | `_p` | none (rigid binaries) |

So every Keplerian class whose input is not the internal pair needs one
root solve, and only to fill reporting attributes.

**What uses the root-solved values.**

* Not model evaluation. The Keplerian positions use only `t0_com`,
  `xL0_com` and `xS0_com`. (For single-source PSBL, the source track uses
  `xS0` at `t0`, which is correct for any choice of `t0` as long as `xS0`
  is moved there.)
* No outward converter (VBM, MulensModel, pyLIMA) uses them for orbit
  classes. VBM's Keplerian model wants the CoM closest approach, which is
  explicit.
* They serve the audit's internal convention and users who compare
  orbiting and static fits by midpoint `t0` / `u0`.

**Cost.** Measured on `PSBL_PhotAstrom_noPar_EllOrbs_Param1`:

| Step | Time |
|---|---|
| CoM → mid conversion (root solve) | 0.34 ms |
| mid → CoM conversion (explicit) | 0.03 ms |
| Whole constructor | 0.72 ms |
| `get_photometry` at 1000 epochs | 13.4 ms |

The root adds about 2.5% per likelihood call. The real risks are
non-unique roots for fast orbits and lack of differentiability in JAX fits
(not an issue for BSBL, which has no JAX kernel).

**Options.**

* **A. Keep the eager root solve** (current PSBL behavior). Simple and
  consistent with the audit, but construction can fail or pick a branch
  for fast orbits.
* **B. Drop the instantaneous midpoint pair for orbiting classes.** Make
  `_com` the internal frame (`t0 = t0_com`, `u0_amp = u0_amp_com`,
  `xL0 = xL0_com`) and report the midpoint position at `t0_com` only.
  * No roots anywhere.
  * But "internal = midpoint" would hold only for static and linear /
    accelerated classes, and the audit convention changes.
  * PSBL `get_source_astrometry_unlensed` keeps working because `xS0` is
    moved to the chosen `t0`.
* **C. Compute the midpoint and `_p` pairs lazily** (cached properties
  such as `t0_mid`, `u0_amp_mid`, `t0_p`). Construction and fits never
  root-solve; the audit and users get the values on demand, with the
  grid-plus-brentq fallback of 4.4.
  * `self.t0` / `self.u0_amp` would then follow option B's `_com`
    definition for Keplerian classes.

**Decision: C.** Keplerian fits stay root-free and robust, and the
midpoint values remain available on request.

Implementation:

* **Internal frame:** for Keplerian classes the internal `t0` /
  `u0_amp` / `u0` / `beta` / `xL0` are the `_com` values.
* **Lazy properties:** `t0_mid`, `u0_amp_mid` and `xL0_mid` (and `t0_p`,
  `u0_amp_p`, `xS0_p` when they are not inputs) are
  `functools.cached_property` values.
  * They call `convert_u0_amp_t0_bsbl_orbit` with the grid-plus-brentq
    root of 4.4.
  * If no root exists, the property raises an informative `ValueError`;
    construction is never affected.
* **Static, linear and accelerated classes:** unchanged. There `t0` is
  the midpoint pair and `t0_mid` is an alias.
* **Source track:** anything that evaluates it from `xS0` at `t0` keeps
  working, because `xS0` is stored at the chosen `t0`.

The existing PSBL / FSBL Keplerian PhotAstrom classes move to option C in
the same change (D2, section 5.4). That removes their four eager root
solves (Param1, Param4, Param7, Param8).

## 7. Tests and audit

* **`tests/test_binary_orbit_frames.py`:** a parametrized test over all 14
  BSBL / BFSBL classes (static Param2 and Ell/Circ Param1-3), checking:
  1. the input pair at the input epoch (4.2);
  2. the internal pair (`_com` for Keplerian classes, midpoint
     otherwise), and the lazy `_mid` / `_p` properties against 4.2;
  3. `xS0_*` equals the input source point at the input `t0_*`;
  4. `t0_com` / `u0_amp_com` is the true CoM-CoM closest approach;
  5. signed `u0_amp_*` values.
* **Round trips:** each converter direction recovers its inputs to 1e-9,
  including static sources (`orbS=None`) against
  `convert_u0_amp_t0_psbl_orbit`.
* **Audit:** update `make_binary_frame_audit.py` (new names, case 12, the
  "Remaining issues" text) and regenerate. Expect every BSBL / BFSBL row
  to pass.
  * Its "internal geometric" column must read `t0_mid` for Keplerian
    classes, since internal `t0` is now `_com` there.
  * Update the convention paragraph at the top of
    `binary_frame_audit.md` to match.
* **Docs build:** the new and edited `.rst` pages build without new Sphinx
  warnings.
* **Regression runs:** `tests/test_fitter_param_names.py` (with the
  `RENAMES` allowlist updated), the BSBL parity subset of
  `tests/test_model_jax.py`, `tests/test_model.py`,
  `tests/test_frame_convert.py`, and `tests/test_model_fitter.py` priors.

## 8. Decisions

* **D1. Param2's input frame — decided.** Switch Param2 to `_com`
  (source CoM ↔ lens CoM, inputs `t0_com`, `u0_amp_com`, `xS0_com_E`,
  `xS0_com_N`), so Param1 and Param2 share a frame.
* **D2. Root solving — decided: option C** (section 6) for the BSBL /
  BFSBL classes and, in the same change, the PSBL / FSBL Keplerian
  PhotAstrom classes (section 5.4). Phot-only orbit classes are the
  documented exception.
* **D3. Static `BSBL_PhotAstromParam2` `_com` values — decided: a mass
  ratio only.** Add the input `qS = mSs / mSp` (4.6). The source masses
  themselves are not needed. Every other BSBL class derives `qS` from
  its existing inputs.
* **D4. `'geom_mid'` — decided: unchanged.** It has been in
  `frame_convert.convert_u0_t0_psbl` since 2023-06-09 (commit `0e9855f`)
  and shipped in v1.0.4 and v1.0.5. It stays as the coords string, and
  the new converter accepts it. The "mid, not geo" rule (2.1) applies to
  attribute names only (`t0_geo` → `t0_mid`).
* **D5. `xS0_system_E/N` — decided.** Retire it; the new names are
  `xS0_com_E` / `xS0_com_N`. Only the fitter prior tables reference it.
* **D6. Changelog and docs — decided.** Document the changes in the new
  changelog, and add clarifying language to the internal parameterization
  docs (`new_models.rst`), the model-type pages (`PSBL.rst`, `BSPL.rst`,
  `BSBL.rst`, and the `*_Param.rst` pages) and the class docstrings
  (section 5.5).
