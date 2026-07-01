# `xi_model`

Reference implementation of the semi-analytic enhancement factor `xi` for the
axion dark-matter abundance produced during a first-order phase transition, as
defined in the paper.

Given

- `hstar` = $H_*/M_\phi$
- `vw` = $v_w$ (bubble wall velocity)
- `theta0` = $\theta_0$ (initial misalignment angle)
- `beta_over_h` = $\beta/H_*$

the package returns `xi = rho_PT / rho_noPT` (the late-time comoving energy-density
ratio with / without the transition), together with the internal quantities
`xi_dm`, `f_bm`, `g_bm`, `rc_mean`, `kappa`, and `tp`.

## Model

$$
\xi = \xi_{\rm dm}\left[(1-\varepsilon_{\rm BM})^{\lambda}
 + A_0\,(\gamma_w v_w)^{p}\,\left(\frac{a_p}{a_*}\right)^{-2} G_{\rm BM}\right]
$$

with

$$
\xi_{\rm dm}=\left[1+\left(C_0\,\frac{\mathcal{F}_\infty}{\mathcal{F}_{\rm anh}}\,
 \tau^{3/2}\right)^{r}\right]^{1/r},\quad r=r_0+r_1 v_w,\quad \tau=\tfrac{2}{3}t_p,
$$

$$
\varepsilon_{\rm BM}=f_{\rm BM}(\kappa),\quad
G_{\rm BM}=\varepsilon_{\rm BM}\left\langle\frac{1}{M_\phi R_c}\right\rangle,\quad
\kappa = s\,v_w\,\kappa_{\rm pilot}(\theta_0,v_w,H_*)^{q_k},
$$

$$
\left(\frac{a_p}{a_*}\right)^{-2}=\left[1+\frac{\sqrt3}{2}\,\frac{H_* R_c}{v_w}\right]^{-2},
\qquad p = p_{\rm mink}\simeq 0.60 .
$$

Here $\lambda$ and $A_0$ are **constants** (from the global fit), the anharmonic
factors $\mathcal{F}_{\rm anh},\mathcal{F}_\infty$ use the manuscript log-form,
and $t_p$ is the percolation time in radiation domination. Fitted parameters are
stored in [data/summary.json](./data/summary.json) (global fit to the lattice
simulations, log-RMSE ≈ 10%).

### Geometry

The bubble-collision geometry $f_{\rm BM}$, $G_{\rm BM}$, and $\langle R_c\rangle$
is computed **on the fly** by integrating a single expanding bubble
(`_geom_compact.compute_geometry_point`) with an **adaptive comoving support**
$R_{\max}\propto 1/H_*$. This is essential at low $H_*/M_\phi$: a fixed support
truncates the collision-radius distribution and biases the geometry. In this
regime the model reproduces the bubble-misalignment scaling $\xi\propto
(H_*/M_\phi)^{-1/2}$ (hence $f_\phi\propto(H_*/M_\phi)^{1/4}$).

## Python API

```python
from xi_model import load_default_model

model = load_default_model()
res = model.predict(hstar=0.2, vw=0.5, theta0=1.2, beta_over_h=6.0)
print(res.xi, res.xi_dm, res.f_bm, res.g_bm, res.rc_mean)
print(res.to_dict())
```

## CLI

```bash
python -m xi_model.cli --hstar 0.2 --vw 0.5 --theta0 1.2 --beta-over-h 6
```

## Calibrated domain

The lattice fit covers `theta0` in `[0, pi)`, `vw` in `[0.3, 0.9]`, `hstar` in
`[0.05, 2.0]`, `beta_over_h` in `[4, 40]`. Outside this range the model
continues smoothly (the geometry is recomputed on the fly), and the low-$H_*$
behavior is the physically expected bubble-misalignment asymptote — but it is an
extrapolation of the fit and should be treated as such.

## Files

- `xi_model/` — package source (`api.py`, `_geom_compact.py`, `_percolation.py`, `cli.py`)
- `data/` — fit summary, geometry bank, and pilot-kappa / lattice tables
- `examples/` — minimal usage example
- `tests/` — smoke tests
