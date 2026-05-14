# sim_core

Lattice simulation of axion field dynamics during a first-order cosmological
phase transition, used to produce the results in [*paper citation*].

## Physics

Solves the classical axion equation of motion on a cubic periodic lattice
in a radiation-dominated FRW background:

$$\theta'' + 2\mathcal{H}\theta' - \nabla^2\theta + a^2 m^2(\tau,\mathbf{x})\sin\theta = 0$$

Bubble nucleation follows an exponential rate
$\Gamma(t) = \Gamma_0\,e^{\beta(t-t_\mathrm{PT})}$ with $\Gamma_0 = H_\mathrm{PT}^4$.
Bubbles expand at constant comoving wall speed $v_w$.  The spatially varying
mass field $m^2(\tau,\mathbf{x})$ is updated at each timestep and smoothly
interpolated between bubble snapshots.

Time integration uses a leapfrog (Störmer–Verlet) scheme with an adaptive
CFL timestep.

## Files

| File | Purpose |
|------|---------|
| `axion_sim.py` | Core simulation: field evolution, bubble nucleation, energy diagnostics |
| `run_sweep.py` | Parameter sweep driver: runs all $(H_*, \beta/H, \theta_0)$ combinations serially |

## Requirements

```
numpy
numba
pyfftw
scipy
matplotlib
tqdm
```

Install with:
```bash
pip install numpy numba pyfftw scipy matplotlib tqdm
```

> **Note:** `pyfftw` requires FFTW3 libraries.  On macOS: `brew install fftw`.
> On Linux: `apt install libfftw3-dev` or equivalent.

## Quick start

```python
from axion_sim import create_simulation_pt_at_zero

sim, tau_final = create_simulation_pt_at_zero(
    H_PT=0.7,
    beta=5.6,            # beta/H * H_PT
    start_time=0.0,
    end_time=10.0 / 0.7,
    Ngrid=64,
    num_tracers=100_000,
    enable_nucleation=True,
    theta0_initial=1.0,
    v_bubble=0.6,
)

results = sim.run_simulation(tau_final)
energy_history = results['energy_history']  # list of dicts, one per saved step
```

To run the full parameter sweep used in the paper:
```bash
python run_sweep.py
```
Edit the parameter lists at the bottom of `run_sweep.py` before running.
Results are written to `sweep_results/`.

## Threading

BLAS and MKL are forced to single-threaded mode; Numba uses all available
cores for the parallelised kernels.  Set `numba.set_num_threads(N)` near the
top of `axion_sim.py` to match your hardware.

## Key classes and functions

### `create_simulation_pt_at_zero(H_PT, beta, start_time, end_time, **kwargs)`
Factory function.  Returns `(sim, tau_final)`.

### `SimulationConfig`
Dataclass holding all simulation parameters.  Pass keyword arguments to
`create_simulation_pt_at_zero` to override defaults.

### `AxionSimulation.run_simulation(tau_final)`
Run the simulation.  Returns a dict with:
- `times_cosmic` — saved cosmic times
- `energy_history` — list of energy dicts (kinetic, gradient, potential, total)
- `false_vacuum_fractions_main` — $f_\mathrm{fv}(\tau)$
- `bubble_counts` — number of nucleated bubbles at each saved step
- `config`, `cosmology` — simulation metadata
