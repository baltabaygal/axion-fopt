# `xi_model`

Compact inference package for the accepted paper model of the lattice energy
density ratio `xi`.

## Purpose

Given

- `hstar` = \(H_*/M_\phi\)
- `vw` = \(v_w\)
- `theta0` = \(\theta_0\)
- `beta_over_h` = \(\beta/H_*\)

the package returns

- `xi`
- a local fit-parameter uncertainty estimate
- interpolated internal quantities such as `kappa`, `f_bm`, `g_bm`, and `xi_dm`
- a selectable homogeneous baseline for `xi_dm`
- warnings when the request falls outside the validated domain

## Accepted model

\[
\xi_{\rm pred}
=
\xi_{\rm DM}\left[(1-f_{\rm BM}(\kappa))^{\lambda(\theta_0)}
+ A_0(\theta_0)\,v_w^p\,f_{\rm BM}(\kappa)\,G_{\rm BM}(\kappa)\right]
\]

with

\[
\kappa = s(v_w)\,\kappa_{\rm pilot}(\theta_0,v_w,H_*),
\qquad
q=2,
\]

\[
\lambda(\theta_0)=A_\lambda\,x^{\gamma_\lambda},
\qquad
x=\ln\!\left(\frac{e}{\cos^2(\theta_0/2)}\right).
\]

The package uses the current accepted broken-powerlaw-baseline fit stored in:

- [data/summary.json](./data/summary.json)

The currently packaged accepted version is the broken-powerlaw-baseline refit,
so the frozen grid bundled with the package is the accepted
`xi_dm_broken_powerlaw` grid from that model.

## `xi_dm` baseline modes

The package supports two homogeneous-baseline modes:

- `broken_powerlaw_ftilde`
  - default
  - reconstructs
    \[
    \xi_{\rm DM}
    =
    \frac{\tilde f_{\rm anh}}{f_{\rm anh}^{\rm noPT}}
    \left(\frac{2t_p}{3}\right)^{3/2}
    \]
    from the compact ODE paper ansatz
- `frozen_grid`
  - uses the accepted frozen interpolated `xi_dm_broken_powerlaw` grid bundled with the fit

The broken-power-law mode uses
\[
\tilde f_{\rm anh}
=
\left[
\big(C(v_w) f_\infty(\theta_0)\big)^{r(v_w)}
+
\frac{f_{\rm anh}^{\rm noPT}(\theta_0)^{r(v_w)}}{\left(\frac{2}{3}t_p\right)^{3r(v_w)/2}}
\right]^{1/r(v_w)},
\qquad
f_\infty(\theta_0)=A_\infty x^{\gamma_\infty},
\qquad
C(v_w)=C_0 v_w^{p_C},
\qquad
r(v_w)=r_0+r_1 v_w.
\]

The fitted parameters for that mode are stored in:

- [data/effective_ftilde_by_vw_summary.json](./data/effective_ftilde_by_vw_summary.json)

In both modes, the percolation time `t_p` is computed directly from the RD
kernel in:

- `ode/hom_ODE/percolation.py:t_perc_RD(H_*/M_\phi,\beta/H_*,v_w)`

rather than interpolated from the sampled fit table.

## Optional thermodynamic `g_*` correction

For physical abundance ratios, the package can optionally include the factor

\[
\frac{g_{*s}(T_{\rm osc})}{g_{*s}(T_{\rm PT})}
\left(\frac{g_*(T_{\rm PT})}{g_*(T_{\rm osc})}\right)^{3/4}
\]

so that the returned quantity corresponds to

\[
\Omega_\phi^{\rm (PT)}
=
\left[
\frac{g_{*s}(T_{\rm osc})}{g_{*s}(T_{\rm PT})}
\left(\frac{g_*(T_{\rm PT})}{g_*(T_{\rm osc})}\right)^{3/4}
\right]
\xi\,
\Omega_\phi^{\rm (no\,PT)}.
\]

This correction requires the **physical** axion mass `mphi_ev`, because the
temperatures `T_PT` and `T_osc` are not fixed by `H_*/M_\phi` alone.

Use:

```python
result = model.predict(
    hstar=0.2,
    vw=0.5,
    theta0=1.2,
    beta_over_h=8.0,
    clip=True,
    xi_dm_mode="broken_powerlaw_ftilde",
    mphi_ev=1e-16,
    include_gstar=True,
)
```

If `include_gstar=False` (default), the package returns the original compact
dimensionless `xi` without the thermodynamic correction.

## Domain

The validated interpolation domain is the fitted lattice grid:

- `theta0` in `[0.261799, 2.879793]`
- `vw` in `[0.3, 0.9]`
- `hstar` in `[0.05, 2.0]`
- `beta_over_h` in `[4.0, 40.0]`

Version 1 is an interpolation tool. If a point lies outside this domain, the
API can either:

- raise an error, or
- clip angle / wall-speed / `\beta/H_*` inputs to the nearest supported value
- for `H_*/M_\phi` below the validated baseline domain, keep the requested value,
  compute `t_p` from the RD kernel, extrapolate frozen-grid `xi_DM` if that
  baseline mode is selected, freeze `\kappa_{\rm pilot}` to the
  low-`H_*/M_\phi` plateau inferred from `pilot_kappa.csv`, and evaluate the
  geometry bank at its low-`H_*/M_\phi` boundary
- for `H_*/M_\phi` above the available geometry-bank support, keep the requested
  value for the baseline / pilot sector and evaluate the geometry bank at its
  high-`H_*/M_\phi` boundary with a warning

This low-`H_*/M_\phi` policy avoids generating new geometry files once the
pilot-`kappa` sector has already plateaued.

## Data included

- accepted compact-model summaries and fit grid
- local parameter-error summary
- copied geometry bank JSON files
- copied pilot-kappa table
- copied lattice `energy_ratio_by_theta_data_v*.txt` tables

This keeps the folder self-contained for paper work.

## Python API

```python
from xi_model import load_default_model

model = load_default_model()
result = model.predict(
    hstar=0.2,
    vw=0.5,
    theta0=1.2,
    beta_over_h=6.0,
    clip=True,
    xi_dm_mode="broken_powerlaw_ftilde",
)

print(result.xi, result.xi_lo, result.xi_hi)
print(result.warnings)
print(result.kappa_plateau_applied, result.kappa_plateau_hstar)
```

## CLI

```bash
python -m xi_model.cli --hstar 0.2 --vw 0.5 --theta0 1.2 --betaH 6
```

To use the frozen sampled baseline instead:

```bash
python -m xi_model.cli --hstar 0.2 --vw 0.5 --theta0 1.2 --betaH 6 --xi-dm-mode frozen_grid
```

To clip out-of-domain inputs instead of failing:

```bash
python -m xi_model.cli --hstar 0.003 --vw 0.2 --theta0 1.2 --betaH 6 --clip
```

## Uncertainty semantics

The returned uncertainty is a local model-parameter envelope, not a full
Bayesian or bootstrap interval.

It is built from the stored local `1σ` parameter errors in
[data/local_summary.json](./data/local_summary.json) by varying:

- `p`
- `A_lambda`
- `gamma_lambda`
- the effective interpolated `s(vw)`
- the effective interpolated `A0(theta0)`

and taking the pointwise min/max over the resulting predictions.

So:

- `xi_lo`, `xi_hi` are the envelope bounds
- `xi_err` is `max(xi - xi_lo, xi_hi - xi)`

This captures local fit-parameter sensitivity only. It does **not** include
model-form uncertainty or uncontrolled extrapolation error.

## Low-`H_*/M_\phi` metadata

Predictions now expose the geometry-side low-`H_*/M_\phi` handling directly:

- `kappa_plateau_applied`
  - whether `\kappa_{\rm pilot}` was frozen to the low-`H_*/M_\phi` plateau
- `kappa_plateau_hstar`
  - interpolated plateau edge in `H_*/M_\phi`
- `geometry_hstar`
  - actual `H_*/M_\phi` used for geometry-bank evaluation

So a low-`H_*/M_\phi` request can still retain the requested baseline input
while making it explicit that the geometry sector was frozen to the available
plateau / boundary behavior.

## Prediction status

Each prediction now includes:

- `prediction_status`
  - `validated`
  - `continued`
  - `validated_with_generated_geometry`
- `status_flags`
  - explicit tags such as
    - `continued_low_h_kappa_plateau`
    - `continued_low_h_geometry_boundary`
    - `continued_high_h_geometry_boundary`
    - `continued_vw_geometry_boundary`
    - `continued_kappa_clipped_to_geometry`
    - `continued_outside_fit_domain`
    - `generated_geometry`

This is the fast way to distinguish:

- interpolation on the calibrated fit domain
- controlled continuation outside the calibrated domain
- on-demand geometry-bank extension

## Files

- `xi_model/` package source
- `data/` frozen model assets
- `examples/` simple usage example
- `tests/` smoke tests
